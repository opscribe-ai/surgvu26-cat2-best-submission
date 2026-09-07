"""What would the motion rule have DONE to the eleven public sample answers?

THE QUESTION. `scripts/calibrate_motion.py` measured, on the training split,
that gating "is tissue being cut?" on motion would flip about one cutting
answer in ten from Yes to No. That is a rate over windows nobody grades. The
eleven public sample cases are the only place we hold both a question and its
gold references, so they are the only place the rule can be PRICED rather than
estimated -- and exactly one of them is a cutting question:

    case131  "Is tissue being cut during this clip?"
    gold     "Yes" / "Yes, tissue is being cut." / ... -- unanimous across 5

We currently answer "Yes", which is byte-identical to reference 0 and scores
1.0000. So if the rule would have fired on case131, it would have turned the
one perfect answer in the set into a wrong one, at the measured bare-form
penalty of 0.2985 -- which is 0.0271 of the eleven-case mean, on a leaderboard
number that has never moved.

This measures each sample clip's activity through the REAL serving decoder, so
the answer is about the clips that are graded rather than about the training
distribution.

WHAT IT DOES NOT DO. It does not open the gate or change any answer. It
reports what would have happened, next to the gold, so that the decision to
keep `STATIC_ACTIVITY_THRESHOLD = None` is priced instead of merely cautious.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.motion import motion_record_from_bursts                # noqa: E402
from surgvu.perceive import decode_clip_bursts                     # noqa: E402

SAMPLE = "/staging/groups/bhaskar_opscribe/surgvu/cat2_sample"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample-root", default=SAMPLE)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frames-per-burst", type=int, default=3)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    root = Path(args.sample_root)
    cases = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not cases:
        raise SystemExit("no case directories under %s -- nothing to measure, "
                         "which is a failure and not an empty result" % root)

    rows = []
    for case in cases:
        video = root / case / ("%s.mp4" % case)
        qfile = root / case / ("%s_question.json" % case)
        gfile = root / case / ("%s.json" % case)
        if not video.exists():
            print("SKIP %s: no video at %s" % (case, video))
            continue
        centres, bursts = decode_clip_bursts(
            video, n_frames=args.frames, per_burst=args.frames_per_burst,
            size=args.size)
        motion = motion_record_from_bursts(bursts, centres)
        question = json.loads(qfile.read_text()) if qfile.exists() else None
        gold = json.loads(gfile.read_text()) if gfile.exists() else []
        rows.append({
            "case": case,
            "question": question,
            "gold_first": gold[0] if gold else None,
            "micro_mean": motion["micro"]["mean"],
            "macro_mean": motion["macro"]["mean"],
            "bursts_measured": motion["bursts_measured"],
            "bursts": motion["bursts"],
        })
        print("%-9s micro %7s  macro %7s  (%d/%d bursts)  %s"
              % (case,
                 "n/a" if motion["micro"]["mean"] is None
                 else "%.3f" % motion["micro"]["mean"],
                 "n/a" if motion["macro"]["mean"] is None
                 else "%.3f" % motion["macro"]["mean"],
                 motion["bursts_measured"], motion["bursts"],
                 (question or "")[:44]), flush=True)

    measured = [r for r in rows if r["micro_mean"] is not None]
    if not measured:
        raise SystemExit("no clip yielded a motion measurement")
    values = sorted(r["micro_mean"] for r in measured)
    report = {
        "cases": len(rows), "measured": len(measured),
        "micro_min": values[0], "micro_max": values[-1],
        "micro_median": values[len(values) // 2],
        "rows": rows,
        # The training-split decile boundary from calibrate_motion.py. Quoted
        # rather than recomputed: the rule would have been calibrated there and
        # applied here, so that is the number that would actually have fired.
        "train_p10_reference": 2.512,
    }
    below = [r["case"] for r in measured if r["micro_mean"] <= 2.512]
    report["would_fire_on"] = below
    print("\nsample micro activity: min %.3f  median %.3f  max %.3f"
          % (values[0], report["micro_median"], values[-1]))
    print("training-split bottom-decile boundary was 2.512")
    print("clips at or below it: %s" % (below or "none"))
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
