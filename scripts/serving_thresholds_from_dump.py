"""Serving thresholds for a model we already dumped per-frame probabilities for.

`scripts/tune_serving_thresholds.py` does its own GPU pass over the validation
split. That pass has already happened for every model in the v2 sweep --
`scripts/dump_frame_probs.py` wrote the per-frame probabilities to an .npz --
so re-running it would spend an hour of GPU to recompute numbers sitting on
disk. This produces the SAME report format from the dump instead, on CPU, in
seconds.

WHY THE SHIPPED CUTS ARE SELF-TUNED AND THE REPORTED SCORE IS NOT. Two
different jobs, and conflating them is the easiest mistake here:

  * the number you REPORT must be honest, so it comes from tuning on one case
    fold and scoring the other. That is what says how the model will do on
    cases it has never seen.
  * the cuts you SHIP should use every window available, because a threshold
    fitted on half the data is simply a worse threshold. Holding data out of
    the final fit buys nothing at serving time -- there is no test left to
    protect.

So `serving_thresholds` below are self-tuned on all validation windows, the
way v1's shipped cuts were, and `measurements` carries the honest two-fold
number beside the optimistic one so the gap is never invisible.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.aggregate import AGGREGATORS            # noqa: E402
from surgvu.frames import sample_frame_indices      # noqa: E402
from surgvu.holdout import case_folds, honest_macro_f1   # noqa: E402
from surgvu.metrics import macro_f1, tune_thresholds     # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probs", required=True, nargs="+",
                        help="one or more dump_frame_probs.py .npz. Several "
                             "are averaged per frame, which is the ensemble "
                             "serving path.")
    parser.add_argument("--frames", type=int, default=16,
                        help="must match config decode.frames -- the cuts are "
                             "only valid for the clip probability they were "
                             "tuned against")
    parser.add_argument("--aggregation", default="mean",
                        choices=sorted(AGGREGATORS))
    parser.add_argument("--checkpoint-thresholds",
                        help="JSON list of the checkpoint's own per-frame "
                             "cuts, for the report's drift guard. Read from "
                             "the checkpoint when omitted requires torch, so "
                             "pass it explicitly on a torch-free node.")
    parser.add_argument("--checkpoint-sha256",
                        help="sha256 of the PRIMARY checkpoint. "
                             "build_perception_config.py refuses a serving "
                             "vector whose provenance does not name the "
                             "weights the config binds, which is what stops a "
                             "retrain from silently inheriting stale cuts.")
    parser.add_argument("--members", nargs="*", default=[],
                        help="checkpoint basenames, recorded in provenance")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    sources = [np.load(path, allow_pickle=False) for path in args.probs]
    data = sources[0]
    tools_target = data["tools_target"]
    classes = [str(c) for c in data["tool_classes"]]
    depth = int(data["depth"][0])

    # Same guard as the sweep: averaging two dumps that describe different
    # windows in different orders is silent and merely scores badly.
    for path, other in zip(args.probs[1:], sources[1:]):
        if not np.array_equal(other["tools_target"], tools_target):
            raise ValueError("%s does not describe the same windows as %s"
                             % (path, args.probs[0]))

    if args.frames > depth:
        raise ValueError("asked for %d frames from a %d-frame dump"
                         % (args.frames, depth))

    picks = sample_frame_indices(depth, args.frames)
    stack = np.mean([src["tools_id"] for src in sources], axis=0)
    clip = AGGREGATORS[args.aggregation](stack[:, picks, :])

    fold_a, fold_b = case_folds(data["cases"], tools_target)
    honest = honest_macro_f1(tools_target, clip, fold_a, fold_b)

    cuts = tune_thresholds(tools_target, clip)
    self_pred = (clip >= cuts).astype(np.float32)
    self_tuned = macro_f1(tools_target, self_pred)

    checkpoint_cuts = (json.loads(args.checkpoint_thresholds)
                       if args.checkpoint_thresholds
                       else [0.5] * len(classes))

    report = {
        "classes": classes,
        "serving_thresholds": [float(c) for c in cuts],
        "checkpoint_thresholds": [float(c) for c in checkpoint_cuts],
        "measurements": {
            "honest_two_fold": honest["honest"],
            "honest_measurable": honest["honest_measurable"],
            "self_tuned": float(self_tuned),
            "per_class_f1_honest": dict(zip(classes, honest["per_class"])),
            "note": "serving_thresholds are the SELF-TUNED cuts over all "
                    "validation windows -- the best cuts available for "
                    "serving. honest_two_fold is what to expect on unseen "
                    "cases and is the number to quote.",
        },
        "provenance": {
            "checkpoint_sha256": args.checkpoint_sha256,
            "probs": list(args.probs),
            "members": list(args.members),
            "aggregation": args.aggregation,
            "frames": args.frames,
            "windows": int(len(tools_target)),
            "split": "splits_v2 val",
            "tuned_by": "scripts/serving_thresholds_from_dump.py",
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("windows=%d frames=%d aggregation=%s"
          % (len(tools_target), args.frames, args.aggregation))
    print("honest two-fold  %.4f   (measurable %.4f)"
          % (honest["honest"], honest["honest_measurable"]))
    print("self-tuned       %.4f   <- the cuts being shipped" % (self_tuned,))
    for name, cut, f1 in zip(classes, cuts, honest["per_class"]):
        print("  %-32s cut %.2f   honest F1 %.4f" % (name, cut, f1))
    print("wrote %s" % (args.out,))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
