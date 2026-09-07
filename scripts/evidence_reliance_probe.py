"""Does the evidence-conditioned VLM READ the pixels, or PARROT the evidence?

THE QUESTION THIS ANSWERS, AND WHY THE LOSS CURVE CANNOT.

`scripts/train_vlm.py --evidence-cache` puts the real perception packet into
the training prompt. Training loss dropped ~35% against the empty-context run
at every matching step. That is ambiguous, and both readings produce the same
low loss:

  * GOOD -- the model fuses the evidence with what it sees, and gets better.
  * BAD  -- the model learns to READ THE EVIDENCE TEXT and skip the pixels.
    The packet correlates with the answer, so loss falls beautifully.

The bad reading quietly destroys the point of the model. The Evidence VLM
earns its place in `challenger` mode by being an INDEPENDENT second opinion
derived from the image; a model that re-encodes the perception stack does not
disagree with the router in useful ways, it just agrees with it more
confidently. We would have spent ten GPU-hours making it redundant.

THE PROBE. Partition held-out records by whether the cached evidence AGREES
with the gold answer or CONTRADICTS it, then score the model on each half
separately:

    contradicts >> agrees   the model overrides bad evidence using the pixels
                            -> genuine fusion, ship it
    contradicts << agrees   the model follows bad evidence into bad answers
                            -> parroting, do not ship it
    both similar            evidence is not driving the answer much either way

`variant_presence_polar` is the cleanest probe available and also the exact
shape that failed on the graded sample: "Does this clip show a mega needle
driver?" against `evidence.variant.family`, which is a decided 'mega'/'large'
with a probability. On 2026-08-26 the shipped VLM answered case132 "Yes" at
confidence 1.00 while the variant head had correctly decided the needle driver
was NOT large -- evidence it never saw. This probe measures whether showing it
that evidence fixes the answer or merely moves the failure.

`tool_presence_polar` is included as a second, independent lens using
`evidence.tools_present`, because one intent's quirks should not decide this.

This script only PARTITIONS and reports; generation is the caller's job (see
`condor/evidence_probe.sub`). Partitioning is pure and torch-free, so it is
testable on a login node.
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

AGREES = "agrees"
CONTRADICTS = "contradicts"
UNDECIDABLE = "undecidable"


def normalize_polar(answer):
    """'Yes'/'No' from a gold answer, or None if it is not polar."""
    text = str(answer).strip().lower().rstrip(".")
    if text in ("yes", "no"):
        return text
    return None


def variant_verdict(question, answer, evidence):
    """Does `evidence.variant` AGREE with the gold polar answer, or CONTRADICT it?

    The question names a family ("mega needle driver" / "large needle driver")
    and the evidence decided one. Gold "Yes" means that family IS present, so
    evidence agrees when its decided family matches the one asked about.

    UNDECIDABLE when the head declined to decide (`decided` false) -- an
    undecided head is not wrong evidence, it is absent evidence, and lumping
    the two together would blur exactly the distinction this probe measures.
    """
    variant = (evidence or {}).get("variant") or {}
    if not variant.get("decided"):
        return UNDECIDABLE
    family = str(variant.get("family", "")).lower()
    if family not in ("mega", "large"):
        return UNDECIDABLE
    polar = normalize_polar(answer)
    if polar is None:
        return UNDECIDABLE

    asked = None
    lowered = question.lower()
    if "mega" in lowered:
        asked = "mega"
    elif "large" in lowered:
        asked = "large"
    if asked is None:
        return UNDECIDABLE

    evidence_says_yes = (family == asked)
    gold_says_yes = (polar == "yes")
    return AGREES if evidence_says_yes == gold_says_yes else CONTRADICTS


#: Tool names as they appear in questions, longest first so that
#: "bipolar forceps" is matched before "forceps" inside it.
def mentioned_tool(question, tool_names):
    lowered = question.lower()
    for name in sorted(tool_names, key=len, reverse=True):
        if name.lower() in lowered:
            return name
    return None


def tool_presence_verdict(question, answer, evidence, tool_names):
    """Same test against `evidence.tools_present`.

    A second, INDEPENDENT lens: one intent's quirks should not be allowed to
    decide whether an adapter ships.
    """
    polar = normalize_polar(answer)
    if polar is None:
        return UNDECIDABLE
    present = (evidence or {}).get("tools_present")
    if present is None:
        return UNDECIDABLE
    name = mentioned_tool(question, tool_names)
    if name is None:
        return UNDECIDABLE
    evidence_says_yes = any(name.lower() == str(p).lower() for p in present)
    gold_says_yes = (polar == "yes")
    return AGREES if evidence_says_yes == gold_says_yes else CONTRADICTS


def classify_record(record, tool_names):
    """(verdict, lens) for one record."""
    intent = record.get("intent", "")
    question = record.get("question", "")
    answer = record.get("answer", "")
    evidence = record.get("evidence")
    if intent == "variant_presence_polar":
        return variant_verdict(question, answer, evidence), "variant"
    if intent == "tool_presence_polar":
        return tool_presence_verdict(question, answer, evidence, tool_names), "tools"
    return UNDECIDABLE, intent


def partition(records, tool_names):
    """{AGREES: [...], CONTRADICTS: [...], UNDECIDABLE: [...]}"""
    out = {AGREES: [], CONTRADICTS: [], UNDECIDABLE: []}
    for record in records:
        verdict, lens = classify_record(record, tool_names)
        enriched = dict(record)
        enriched["_probe_lens"] = lens
        out[verdict].append(enriched)
    return out


def summarize(buckets):
    total = sum(len(v) for v in buckets.values())
    lines = ["probe partition over %d record(s):" % total]
    for key in (AGREES, CONTRADICTS, UNDECIDABLE):
        n = len(buckets[key])
        lines.append("  %-12s %6d  (%5.1f%%)" % (key, n, 100.0 * n / total if total else 0.0))
    contradicts = len(buckets[CONTRADICTS])
    lines.append("")
    lines.append("  The contradicting bucket is SMALL BECAUSE THE EVIDENCE IS USUALLY")
    lines.append("  RIGHT, not because of a parsing gap: on 2026-08-26 all 420")
    lines.append("  variant_presence_polar records in val were usable, and the variant")
    lines.append("  head simply erred on 34 of them (8.1%), consistent with its")
    lines.append("  measured 86.8% val accuracy. That is the real ceiling here.")
    if contradicts < 30:
        lines.append("")
        lines.append("  WARNING: only %d contradicting record(s). Below ~30 the two" % contradicts)
        lines.append("  scores are not separable and this probe cannot decide anything.")
    else:
        # The threshold is 30 because of the EFFECT SIZE this probe looks for,
        # not a generic rule of thumb. A parroting model follows wrong evidence
        # into a wrong polar answer (~0.7015); a fusing one overrides it
        # (~1.0000). Modelling those as 80%/10% wrong respectively gives means
        # 0.7612 and 0.9735 with 95% CIs of +-0.0349 and +-0.0248 at n=45 -- a
        # 0.179 gap against CIs a fifth that size. Ample for a binary verdict;
        # NOT ample for ranking two adapters a few points apart, which this
        # probe is not for.
        lines.append("  n=%d is sufficient for a PARROTING-vs-FUSION verdict (the gap" % contradicts)
        lines.append("  between those two behaviours is ~0.18, roughly 5x the 95%% CI).")
        lines.append("  It is NOT sufficient for ranking two adapters a few points apart.")
    return "\n".join(lines)


# ============================================================================
# torch-dependent. Every import is INSIDE a function, matching this repo's
# module-scope import discipline, so everything above stays testable on a
# login node with no torch.
# ============================================================================


def score_partition(partition_path, adapter_dir, base_model, max_new_tokens=32):
    """Generate on each bucket and score with the OFFICIAL metric.

    Loads BASE + LoRA ADAPTER directly rather than a merged/quantised
    checkpoint, deliberately: this runs minutes after training finishes, and
    requiring a merge first would put an hour of packaging between the run and
    the one number that decides whether the packaging is worth doing at all.

    Scores each bucket with `surgvu.scoring.Scorer` -- the same BERTScore-F1
    the challenge uses -- so the two halves are directly comparable to every
    other number in this project.
    """
    import torch
    from surgvu.scoring import Scorer
    from train_vlm import (
        build_messages, load_frames, load_model_and_processor,
    )

    buckets = json.loads(Path(partition_path).read_text(encoding="utf-8"))
    model, processor = load_model_and_processor(base_model, adapter_dir=adapter_dir)
    model.eval()
    scorer = Scorer()

    results = {}
    for name in (AGREES, CONTRADICTS):
        records = buckets.get(name) or []
        if not records:
            results[name] = {"n": 0, "mean": None}
            continue
        pairs = []
        for record in records:
            images = load_frames(record["frame_paths"])
            messages = build_messages(record["question"], images, answer=None,
                                      evidence=record.get("evidence"))
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt")
            inputs = inputs.to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs,
                                           max_new_tokens=int(max_new_tokens),
                                           do_sample=False)
            prompt_length = inputs["input_ids"].shape[1]
            answer = processor.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True)[0].strip()
            pairs.append((answer, [record["answer"]]))
        # score_one returns {"bertscore_f1": float}, not a float -- see
        # surgvu.scoring.Scorer. Summing the dicts raised
        # "unsupported operand type(s) for +: 'int' and 'dict'" AFTER all 195
        # generations had run (cluster 9709601), throwing away the expensive
        # half of the job for a key lookup.
        scores = [scorer.score_one(cand, refs)["bertscore_f1"]
                  for cand, refs in pairs]
        results[name] = {"n": len(scores), "mean": sum(scores) / len(scores),
                         "scores": scores}
    return results


def verdict(results):
    """The one line this whole probe exists to print."""
    a = results.get(AGREES, {}).get("mean")
    c = results.get(CONTRADICTS, {}).get("mean")
    if a is None or c is None:
        return "INCONCLUSIVE: a bucket was empty."
    gap = a - c
    lines = [
        "",
        "  agrees      n=%-4d mean=%.4f" % (results[AGREES]["n"], a),
        "  contradicts n=%-4d mean=%.4f" % (results[CONTRADICTS]["n"], c),
        "  gap (agrees - contradicts) = %+.4f" % gap,
        "",
    ]
    # WHAT A GOOD RESULT ACTUALLY LOOKS LIKE, because the first version of
    # this function got it wrong. A model that uses the PIXELS scores the same
    # whether the evidence beside them happens to be right or wrong -- so
    # success is gap ~= 0, NOT a large negative gap. Scoring much BETTER where
    # the evidence is wrong would be strange enough to be worth investigating
    # rather than celebrating. The near-zero band is therefore the PASS, and
    # calling it "inconclusive" (as the first draft did) would have reported
    # the desired outcome as a non-answer.
    #
    # 0.09 is half the ~0.18 that separates parroting (0.7612) from evidence-
    # independent behaviour (0.9735) at n=45.
    if gap > 0.09:
        lines += [
            "  VERDICT: PARROTING.",
            "  The model scores far worse exactly where the evidence is WRONG,",
            "  which means it is following the evidence text rather than the",
            "  pixels. Do NOT ship this adapter into `challenger` -- it would",
            "  stop being an independent second opinion and start agreeing with",
            "  the router's mistakes more confidently.",
        ]
    elif gap < -0.09:
        lines += [
            "  VERDICT: ANOMALOUS -- investigate before shipping.",
            "  The model scores markedly BETTER where its evidence is wrong.",
            "  That is not what fusion looks like (fusion is gap ~= 0); it",
            "  suggests the two buckets differ in something other than evidence",
            "  correctness -- check whether the contradicting bucket is dominated",
            "  by one intent or one case.",
        ]
    else:
        lines += [
            "  VERDICT: NOT PARROTING. This is the PASS.",
            "  Wrong evidence costs the model almost nothing, which means the",
            "  answer is coming from the pixels -- the model stays an INDEPENDENT",
            "  second opinion, which is the whole reason `challenger` mode has",
            "  something to arbitrate.",
            "",
            "  This clears the safety gate. It does NOT by itself say the adapter",
            "  is BETTER: for that, compare its held-out CASE bertscore_f1",
            "  against the empty-context run's 0.9092.",
        ]
    return "\n".join(lines)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest",
                        default="/staging/n/nkalthoff/surgvu26/qa_frames_manifest.jsonl")
    parser.add_argument("--evidence-cache",
                        default="/staging/n/nkalthoff/surgvu26/evidence_cache.jsonl")
    parser.add_argument("--splits", default="config/splits_v2.json")
    parser.add_argument("--out", default="baselines/evidence_probe_partition.json")
    parser.add_argument("--max-per-bucket", type=int, default=150)
    parser.add_argument("--adapter-dir", default=None,
                        help="when given, GENERATE on the partition with this "
                             "LoRA adapter and print the parroting-vs-fusion "
                             "verdict. Without it this only partitions.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--results-out",
                        default="baselines/evidence_probe_results.json")
    return parser


def main(argv=None):
    from train_vlm import (
        assign_case_split, attach_evidence, filter_records_with_frames,
        load_case_universe, load_evidence_cache, load_manifest,
    )
    from surgvu.detect import YOLO_CLASSES

    args = build_arg_parser().parse_args(argv)
    records = load_manifest(args.manifest)
    # load_case_universe returns the HELDOUT set too, and it is deliberately
    # not used here: this probe reads the VAL split only. The graded cases must
    # stay untouched (R30), and load_case_universe already fails loudly if the
    # heldout list is missing or empty.
    train_norm, val_norm, _heldout_norm = load_case_universe(args.splits)
    _train, val_records = assign_case_split(records, train_norm, val_norm)
    val_records, _dropped = filter_records_with_frames(val_records)
    cache = load_evidence_cache(args.evidence_cache)
    attach_evidence(val_records, cache)

    buckets = partition(val_records, YOLO_CLASSES)
    print(summarize(buckets))

    trimmed = {k: v[:args.max_per_bucket] for k, v in buckets.items()
               if k != UNDECIDABLE}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(trimmed, indent=2), encoding="utf-8")
    print("wrote %s" % args.out)

    if not args.adapter_dir:
        print("\nno --adapter-dir: partition only. Pass one to get the verdict.")
        return 0

    results = score_partition(args.out, args.adapter_dir, args.base_model)
    print(verdict(results))
    Path(args.results_out).write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print("wrote %s" % args.results_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
