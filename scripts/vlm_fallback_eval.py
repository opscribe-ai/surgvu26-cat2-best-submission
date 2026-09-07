"""Does a VLM beat the router's generic sentence on questions it cannot route?

    python scripts/vlm_fallback_eval.py <sample_root> --out-dir .

THE MEASUREMENT PROBLEM, STATED BEFORE THE MEASUREMENT
------------------------------------------------------
We have organizer-written gold references for exactly 11 questions. Inventing
new questions means inventing their references, which makes the measurement
circular and worthless -- so nothing here invents a reference.

None of the 11 is unroutable today. Two of them are answered from CONSTANTS
rather than from the CNN taxonomy, and those two are the only real questions
whose content is "big picture" in the sense that matters here:

    case129  "What procedure is this summary describing?"   -> PROCEDURE_ANSWER
    case130  "What is the purpose of using forceps ...?"    -> PURPOSES[forceps]

Everything else is a taxonomy question -- 12 instrument classes and 8 task
classes -- which is exactly what the CNNs are trained on and exactly what the
VLM was already measured to lose at. So the honest comparison is: on those two
questions, and on paraphrases of them, what would the generic fallback score,
what does the VLM score, and what does the shipped constant score?

n = 2. That is thin, and the report says so rather than dressing it up.

THREE GROUPS, KEPT SEPARATE
---------------------------
  real      the 2 real organizer questions. The only rows with no phrasing
            invented by us at all. n=2.
  variant   paraphrases of those 2 from tests/fixtures/question_variants.json,
            scored against the SOURCE case's real gold. The gold is the
            organizers'; only the phrasing is ours. This measures robustness
            to phrasing, NOT more content: 12 rows over 2 distinct answers.
            The six case130 paraphrases that name a DIFFERENT instrument
            ("what is the role of the needle driver?") are excluded, because
            case130's gold is the forceps purpose and would be the wrong
            reference for them.
  routable  case127 (organ) and its paraphrases. The router answers these from
            the task CNN and they would NEVER reach the VLM. Included, and
            labelled, because they are the only picture we have of what the
            VLM does to an open question of a shape it might one day be handed
            by accident.

WHAT IT RUNS
------------
The real serving code. Perception comes from `scripts/inference.py`'s own
`infer`, the answers come from `surgvu.router.answer_question`, and the VLM
candidate comes from `surgvu.vlm.QwenVlmFallback.answer` -- the same object
the container constructs. A number measured through a private harness would
not be a number about the thing we ship.

Two prompt arms are run and BOTH are reported. Picking the better one on n=2
would be picking noise.
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import inference                                                    # noqa: E402
from score_sample import load_sample_cases                          # noqa: E402
from surgvu.router import FALLBACK_OPEN, answer_question            # noqa: E402
from surgvu.vlm import DEFAULT_MODEL_DIR, QwenVlmFallback           # noqa: E402

VARIANTS = REPO / "tests" / "fixtures" / "question_variants.json"

# The two real questions whose gold is world knowledge rather than taxonomy.
OPEN_CONSTANT_CASES = ("case129", "case130")
# Answered from the task CNN. Reported separately; never routed to the VLM.
ROUTABLE_OPEN_CASES = ("case127",)

# case130's gold is the purpose of FORCEPS. A paraphrase naming another
# instrument is a different question with a different (unavailable) gold, so
# it is dropped rather than scored against the wrong reference.
_CASE130_KEEP = "forceps"


def variant_questions(path=VARIANTS):
    """[(case_id, question)] -- paraphrases whose source gold still applies.

    Two exclusions, both because the source's references would be the WRONG
    references for the paraphrase, which is the one thing this measurement is
    not allowed to do:

      * a paraphrase that names a different instrument. case130's gold is the
        purpose of FORCEPS; "what is the role of the needle driver?" has a
        different answer and we do not have it.
      * a paraphrase that turned the question polar. "Is this a laparoscopic
        procedure?" wants "Yes"; case129's references are all noun phrases
        naming the procedure, and scoring "Yes" against them would measure
        nothing.
    """
    from surgvu.router import is_polar_question

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    keep = []
    for variant in data.get("variants", ()):
        source = variant.get("source")
        question = variant.get("question")
        if source not in OPEN_CONSTANT_CASES + ROUTABLE_OPEN_CASES:
            continue
        if source == "case130" and _CASE130_KEEP not in question.lower():
            continue
        if is_polar_question(question):
            continue
        keep.append((source, question))
    return keep


def build_items(cases):
    """Every (id, case_id, question, group) row the VLM will be asked."""
    items = []
    for case_id in OPEN_CONSTANT_CASES:
        items.append(("%s/real" % case_id, case_id,
                      cases[case_id].question, "real"))
    for case_id in ROUTABLE_OPEN_CASES:
        items.append(("%s/real" % case_id, case_id,
                      cases[case_id].question, "routable"))
    seen = {}
    for case_id, question in variant_questions():
        seen[case_id] = seen.get(case_id, 0) + 1
        group = "routable" if case_id in ROUTABLE_OPEN_CASES else "variant"
        items.append(("%s/v%d" % (case_id, seen[case_id]), case_id,
                      question, group))
    return items


def perceive_cases(cases, videos, args):
    """{case_id: perception record}, through the serving path, once per case."""
    config = inference.load_config(args.config)
    frames_wanted = config["decode"]["frames"]
    size = config["decode"]["size"]
    devices = inference.resolve_devices(args.device)
    records, frames = {}, {}
    for case_id in sorted(cases):
        timings = []
        decoded = inference.decode_clip(videos[case_id],
                                        n_frames=frames_wanted, size=size)
        records[case_id] = inference.infer_with_retry(
            decoded, config, devices, timings, args.models_dir)
        frames[case_id] = decoded
        print("%s tools_present=%s task_top=%s"
              % (case_id, records[case_id]["tools_present"],
                 records[case_id]["task_top"]), flush=True)
    return records, frames


def run_arm(fallback, items, records, frames, use_context):
    """{item_id: (answer, seconds)} for one prompt arm."""
    fallback.use_context = use_context
    out = {}
    for item_id, case_id, question, _group in items:
        started = time.time()
        answer = fallback.answer(question, records[case_id], frames[case_id])
        elapsed = time.time() - started
        out[item_id] = (answer, elapsed)
        print("  %-14s %-6.1fs %-60r %s"
              % (item_id, elapsed, question[:58], answer), flush=True)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sample_root")
    parser.add_argument("--config", default=str(REPO / "config" / "perception.json"))
    parser.add_argument("--models-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vlm-model")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args(argv)

    sample_root = Path(args.sample_root)
    cases = load_sample_cases(sample_root)
    videos = {case_id: next(sample_root.glob("%s/%s.mp4" % (case_id, case_id)))
              for case_id in cases}
    records, frames = perceive_cases(cases, videos, args)

    shipped = {case_id: answer_question(case.question, records[case_id])
               for case_id, case in cases.items()}
    print("\nshipped router answers:")
    for case_id in sorted(shipped):
        print("  %-9s %r" % (case_id, shipped[case_id]))

    items = build_items(cases)
    fallback = QwenVlmFallback(
        model_dir=args.vlm_model or DEFAULT_MODEL_DIR,
        log=lambda message: print("    [vlm] %s" % (message,), flush=True))

    arms = {}
    for arm, use_context in (("context", True), ("nocontext", False)):
        print("\n== arm %s ==" % (arm,), flush=True)
        arms[arm] = run_arm(fallback, items, records, frames, use_context)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Three candidate files over all 11 cases, for scripts/score_sample.py.
    # Only the two constant-answered cases differ between them, so the per-case
    # rows of the table are the comparison and the means are context.
    def substituted(case_id, replacement):
        """The shipped answer everywhere except the two ablated cases.

        `replacement` returning None -- the VLM declining -- lands on
        FALLBACK_OPEN, which is exactly what the gate does at serving time.
        """
        if case_id not in OPEN_CONSTANT_CASES:
            return shipped[case_id]
        return replacement(case_id) or FALLBACK_OPEN

    written = {}
    for label, replacement in (
            ("shipped", lambda case_id: shipped[case_id]),
            ("fallbackopen", lambda case_id: FALLBACK_OPEN),
            ("vlmcontext",
             lambda case_id: arms["context"]["%s/real" % case_id][0]),
            ("vlmnocontext",
             lambda case_id: arms["nocontext"]["%s/real" % case_id][0])):
        candidates = {case_id: substituted(case_id, replacement)
                      for case_id in cases}
        path = out_dir / ("%s_candidates.json" % label)
        path.write_text(json.dumps(candidates, indent=2, sort_keys=True),
                        encoding="utf-8")
        written[label] = candidates
        print("wrote %s" % (path,))

    # Every row, for the paraphrase extension: scored directly against the
    # SOURCE case's real references.
    pairs = []
    for item_id, case_id, question, group in items:
        references = list(cases[case_id].references)
        pairs.append({"id": item_id, "case_id": case_id, "group": group,
                      "question": question, "references": references,
                      "candidates": {
                          "fallback": FALLBACK_OPEN,
                          "shipped": shipped[case_id],
                          "vlmcontext": arms["context"][item_id][0],
                          "vlmnocontext": arms["nocontext"][item_id][0]},
                      "seconds": {
                          "vlmcontext": arms["context"][item_id][1],
                          "vlmnocontext": arms["nocontext"][item_id][1]}})
    (out_dir / "vlm_fallback_pairs.json").write_text(
        json.dumps(pairs, indent=2), encoding="utf-8")
    print("wrote %s (%d rows)"
          % (out_dir / "vlm_fallback_pairs.json", len(pairs)))

    declined = sum(1 for row in pairs if not row["candidates"]["vlmcontext"])
    print("\nVLM declined (returned None) on %d of %d context-arm rows"
          % (declined, len(pairs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
