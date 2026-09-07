#!/usr/bin/env python3
"""Per-intent comparison of the router against the Evidence VLM, and the
`vlm_intents` list that `config/arbiter.json`'s `per_intent` mode consumes.

WHAT THIS IS FOR
-----------------
`arbiter.MODE_PER_INTENT` ships inert: `vlm_intents` defaults to empty, which
makes it byte-identical to `fallback`. This script is the ONLY sanctioned way
to populate that list. The mode's whole value is that a regression is
attributable to a single named intent, and that property survives only if
every name in the list got there from a measurement rather than from someone
reading a table and forming an impression.

WHY THE JOIN IS BY INDEX, AND WHY THAT IS SAFE
------------------------------------------------
`scripts/train_vlm.py --eval-only` writes `eval_report.json` whose
`results` list is built by `Scorer.score_many` from `pairs`, which is built by
iterating `eval_records` in order. So `results[i]` is the VLM's score for
`eval_records[i]`. This script rebuilds `eval_records` by calling
train_vlm's OWN `sample_eval_records` with the same manifest, the same splits
and the same seed, so the two orderings are the same object computed twice.

That is an argument, not a guarantee, so it is CHECKED: `case_id` in the
report is `"case|part|t_start"`, this script recomputes that string at each
index, and a single mismatch aborts. The alternative -- joining ON that key --
is what is actually unsafe: several questions share one (case, part, t_start)
window, so the key is not unique and a key-join would silently pair a
question's VLM score with a different question's router answer.

WHY THE ROUTER IS SCORED HERE RATHER THAN READ OFF AN EXISTING NUMBER
-----------------------------------------------------------------------
The router's per-intent score has to be measured on the SAME 2400 records the
VLM was measured on, or the difference between them is confounded by which
records each saw. Both sides are therefore scored in this one process, against
one `Scorer`, from one sample.

THE PERCEPTION THE ROUTER IS GIVEN IS THE CACHED ONE
------------------------------------------------------
`record["evidence"]` (from `evidence_cache.jsonl`) is not a rendered string --
it is the perception dict itself: `tools`, `tools_present`, `task`,
`task_top`, `motion_v2`, `yolo`, `variant`. Those are exactly the keys
`router.answer_question` reads, so the router here is answering from the same
model outputs it would get at serving time. `--evidence-cache` is therefore
REQUIRED by this script even when the VLM being compared was trained bare:
the cache is the router's input, independently of whether it was also in the
VLM's prompt.

WHICH ADAPTER'S REPORT YOU FEED THIS MATTERS
----------------------------------------------
There are two trained adapters. `models/vlm_lora` is the one merged into the
shipped container and was trained with NO evidence in its prompt;
`models/vlm_lora_evidence` was trained with it and would need
`vlm_evidence_context: true` plus a re-merge before anything measured on it
could ship. A `vlm_intents` list derived from the wrong one is armable only in
principle. `--label` is recorded in the output so a table cannot be mistaken
for the other adapter's later.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from surgvu import router  # noqa: E402


#: How many standard errors of the DIFFERENCE the VLM must clear before an
#: intent is armed. 2.0 is a ~95% two-sided bar.
#:
#: WHY A BAR AT ALL, RATHER THAN "VLM > ROUTER". With eleven intents, ranking
#: two noisy estimates and keeping every intent where one happens to exceed
#: the other arms roughly half of them by chance alone -- and every one so
#: armed is a coin-flip that has already been paid for with a submission. The
#: project has four submissions left and no way to A/B two intents at once.
DEFAULT_SIGMA = 2.0


def paired_stats(router_scores, vlm_scores):
    """Mean difference (VLM - router) and the standard error OF THAT MEAN.

    PAIRED, NOT TWO-SAMPLE. Both models answered the SAME records, so the
    per-record difference removes the record-to-record variance -- which
    dominates here, because some questions are easy for everything and some
    are hard for everything. Treating the two score lists as independent
    samples would inflate the standard error by roughly the between-record
    spread and hide real differences.
    """
    n = len(router_scores)
    if n != len(vlm_scores):
        raise ValueError("score lists differ in length: %d vs %d"
                         % (n, len(vlm_scores)))
    if n == 0:
        return 0.0, float("inf"), 0
    diffs = [v - r for v, r in zip(vlm_scores, router_scores)]
    mean = sum(diffs) / n
    if n < 2:
        return mean, float("inf"), n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    return mean, math.sqrt(var / n), n


def case_key(record):
    """The `case_id` string `train_vlm.run_eval` writes into the report.

    Kept byte-identical to that f-string, deliberately duplicated rather than
    imported, because this function's JOB is to detect drift between the two
    -- importing the same expression would make the integrity check below
    tautological.
    """
    return "%s|%s|%.3f" % (record["case"], record["part"], record["t_start"])


def build_eval_records(args):
    """train_vlm's own pipeline, re-run, so the sample cannot drift."""
    import train_vlm as tv

    records = tv.load_manifest(args.manifest)
    if not records:
        raise SystemExit("no records read from %s" % args.manifest)
    train_norm, val_norm, heldout_norm = tv.load_case_universe(args.splits)
    tv.verify_manifest_clean(records, train_norm, val_norm, heldout_norm)
    _, val_records = tv.assign_case_split(records, train_norm, val_norm)
    val_records, _ = tv.filter_records_with_frames(val_records)
    eval_records = tv.sample_eval_records(
        val_records, args.max_eval_examples, args.seed)
    cache = tv.load_evidence_cache(args.evidence_cache)
    tv.attach_evidence(eval_records, cache)
    return eval_records


def verify_alignment(eval_records, results):
    """Abort unless the rebuilt sample matches the report position for
    position. See the module docstring on why this is a check and not a join.
    """
    if len(eval_records) != len(results):
        raise SystemExit(
            "rebuilt %d eval record(s) but the report has %d result(s). The "
            "sample drifted -- check --manifest/--splits/--seed/"
            "--max-eval-examples match the run that produced the report."
            % (len(eval_records), len(results)))
    for i, (record, row) in enumerate(zip(eval_records, results)):
        expected, got = case_key(record), row.get("case_id")
        if expected != got:
            raise SystemExit(
                "index %d: rebuilt record is %s but the report says %s. The "
                "orderings do not correspond; an index join here would pair "
                "each question with another question's score." % (i, expected, got))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-report", required=True,
                        help="eval_report.json from train_vlm.py --eval-only")
    parser.add_argument("--evidence-cache", required=True,
                        help="evidence_cache.jsonl -- the ROUTER's perception "
                             "input, required even for a bare-trained adapter")
    parser.add_argument("--manifest", default="/staging/n/nkalthoff/surgvu26/qa_frames_manifest.jsonl")
    # DEFAULT_SPLITS, not the literal "config/splits.json". train_vlm.py's own
    # default is config/splits_v2.json (the one with a heldout list holding the
    # 11 graded cases); config/splits.json is the SUPERSEDED file and has no
    # heldout list at all. Rebuilding the sample against the wrong one would
    # draw a different val set, which the alignment check would catch -- but as
    # a confusing abort rather than as the plain bug it is. Imported from
    # train_vlm so the two cannot drift.
    parser.add_argument("--splits", default=None,
                        help="defaults to train_vlm.DEFAULT_SPLITS")
    parser.add_argument("--max-eval-examples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--label", required=True,
                        help="which adapter the report came from, recorded in "
                             "the output so two tables cannot be confused")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    if args.splits is None:
        import train_vlm as tv
        args.splits = tv.DEFAULT_SPLITS

    report = json.loads(Path(args.eval_report).read_text(encoding="utf-8"))
    results = report["results"]
    eval_records = build_eval_records(args)
    verify_alignment(eval_records, results)
    print("alignment verified: %d record(s) match the report position for position"
          % len(results))

    from surgvu.scoring import Scorer
    scorer = Scorer()

    by_intent = {}
    for record, row in zip(eval_records, results):
        question = record["question"]
        intent = router.classify_question(question)
        router_answer = router.answer_question(question, record["evidence"])
        router_f1 = scorer.score_one(router_answer, [record["answer"]])["bertscore_f1"]
        bucket = by_intent.setdefault(intent, {"router": [], "vlm": []})
        bucket["router"].append(router_f1)
        bucket["vlm"].append(row["bertscore_f1"])

    rows, armed = [], []
    for intent in sorted(by_intent):
        b = by_intent[intent]
        r_mean = sum(b["router"]) / len(b["router"])
        v_mean = sum(b["vlm"]) / len(b["vlm"])
        delta, se, n = paired_stats(b["router"], b["vlm"])
        arm = bool(se > 0 and math.isfinite(se) and delta > args.sigma * se)
        rows.append({"intent": intent, "n": n, "router": r_mean, "vlm": v_mean,
                     "delta": delta, "se": se, "arm": arm})
        if arm:
            armed.append(intent)

    total = sum(r["n"] for r in rows)
    router_overall = sum(r["router"] * r["n"] for r in rows) / total
    vlm_overall = sum(r["vlm"] * r["n"] for r in rows) / total
    best_of = sum(max(r["router"], r["vlm"]) * r["n"] for r in rows) / total
    armed_overall = sum((r["vlm"] if r["arm"] else r["router"]) * r["n"]
                        for r in rows) / total

    print("\nadapter: %s   n=%d   sigma=%.1f" % (args.label, total, args.sigma))
    print("%-26s %6s %8s %8s %9s %9s  %s"
          % ("intent", "n", "router", "vlm", "delta", "se", "arm"))
    for r in rows:
        print("%-26s %6d %8.4f %8.4f %+9.4f %9.4f  %s"
              % (r["intent"], r["n"], r["router"], r["vlm"], r["delta"],
                 r["se"], "ARM" if r["arm"] else ""))
    print("\nrouter alone      %.4f" % router_overall)
    print("vlm alone         %.4f" % vlm_overall)
    print("per_intent ARMED  %.4f   (%+.4f vs router alone)"
          % (armed_overall, armed_overall - router_overall))
    print("oracle best-of    %.4f   (%+.4f) -- NOT achievable; upper bound only"
          % (best_of, best_of - router_overall))
    print("\nvlm_intents = %s" % json.dumps(armed))

    out = {"label": args.label, "n": total, "sigma": args.sigma,
           "rows": rows, "router_overall": router_overall,
           "vlm_overall": vlm_overall, "armed_overall": armed_overall,
           "oracle_best_of": best_of, "vlm_intents": armed}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("wrote %s" % args.output)
    return out


if __name__ == "__main__":
    main()
