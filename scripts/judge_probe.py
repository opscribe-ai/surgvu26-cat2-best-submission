"""Does the decision VLM actually pick the better answer?

THE ONLY QUESTION THAT DECIDES WHETHER `judge` MODE SHIPS, and it is cheap:
the judge is consulted only when the router and the VLM disagree, and on the
graded sample that is FOUR cases. Four generations against one model load,
rather than a full eleven-case validation.

The four, measured on the real image (cluster 9705192, GPU draw, VLM live):

  case122  router "No"              VLM "Yes"           gold "No"      router
  case124  router "Bipolar Forceps" VLM "Clip applier"  gold "Cadiere Forceps"  NEITHER
  case130  router "...surgery."     VLM "...surgery"    gold "...surgery."      router
  case132  router "No"              VLM "Yes"           gold "No"      router

THE BAR IS HIGH AND WORTH STATING BEFORE THE RESULT. The router already wins
three of these four outright, and the fourth is unwinnable by choosing. So a
judge that picks perfectly scores exactly what `fallback` scores, and every
mistake it makes is a loss against that. `judge` mode is therefore only worth
shipping if it is nearly perfect HERE and expected to help on questions the
graded sample does not contain -- the uncovered intents where the router emits
a generic answer and the VLM might be right.

That asymmetry is the point of running this: it is a test the judge can only
fail or draw, on this sample. A draw is a pass.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

#: (case, question, router, vlm, gold, who_is_right)
DISAGREEMENTS = [
    ("case122", "Are there forceps being used here?",
     "No", "Yes", "No", "router"),
    ("case124", "What type of forceps is mentioned?",
     "Bipolar Forceps", "Clip applier", "Cadiere Forceps", "neither"),
    ("case130", "What is the purpose of using forceps in this procedure?",
     "To grasp and hold tissues or objects during the surgery.",
     "To grasp and hold tissues or objects during the surgery",
     "To grasp and hold tissues or objects during the surgery.", "router"),
    ("case132", "Was a large needle driver used during this clip?",
     "No", "Yes", "No", "router"),
]


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--judge-dir",
                   default="/staging/n/nkalthoff/surgvu26/models/qwen3vl-4b-judge-nf4")
    p.add_argument("--sample-dir",
                   default="/staging/groups/bhaskar_opscribe/surgvu/cat2_sample")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--out", default="baselines/judge_probe.json")
    return p


def main(argv=None):
    from surgvu import evidence_vlm, judge as judge_mod

    args = build_arg_parser().parse_args(argv)
    results, correct, wrong, undecided = [], 0, 0, 0

    for case, question, router, vlm, gold, who in DISAGREEMENTS:
        video = Path(args.sample_dir) / case / ("%s.mp4" % case)
        if not video.is_file():
            print("  %s: no video at %s -- skipped" % (case, video))
            continue
        frames = evidence_vlm.sample_frames({"path": str(video)}, n=args.frames)
        prompt = judge_mod.build_judge_prompt(question, "", (router, vlm))
        reply = evidence_vlm.call_vlm(frames, prompt, {}, temperature=0.0,
                                      model_dir=args.judge_dir)
        answer, source = judge_mod.parse_judgement(reply, (router, vlm))

        if source != "choice":
            verdict = "NO CHOICE"
            undecided += 1
        elif answer == router and who == "router":
            verdict = "correct (router)"
            correct += 1
        elif who == "neither":
            verdict = "unwinnable -- both candidates wrong"
        else:
            verdict = "WRONG (picked the worse answer)"
            wrong += 1

        print("  %-8s reply=%-28r -> %s" % (case, reply.strip()[:28], verdict))
        results.append({"case": case, "reply": reply, "parsed": answer,
                        "source": source, "verdict": verdict})

    print()
    print("  correct %d | wrong %d | no-choice %d of %d"
          % (correct, wrong, undecided, len(results)))
    print()
    if wrong == 0 and undecided == 0:
        print("  PASS: the judge never picked the worse answer, and always")
        print("  produced a parseable choice. It matches fallback here and may")
        print("  help on intents the graded sample does not contain.")
    elif undecided:
        print("  FORMAT PROBLEM: the judge did not reply with a candidate label.")
        print("  A base model that answers the question instead of judging is")
        print("  suppressed to None (judge.ALLOW_FREETEXT_ANSWER is False), so")
        print("  this degrades to fallback rather than hurting -- but the judge")
        print("  is then doing nothing and should not ship.")
    else:
        print("  FAIL: the judge picked the worse answer %d time(s)." % wrong)
        print("  Shipping `judge` mode would score BELOW fallback. Do not ship it.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
