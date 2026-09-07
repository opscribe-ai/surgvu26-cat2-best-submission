"""Experiments 1-3, swept offline over the cached per-frame probabilities.

    aggregation  x  frame count  x  test-time augmentation

`scripts/dump_frame_probs.py` paid for the forward passes. Everything here is
numpy over that tensor, so the whole three-dimensional sweep costs seconds and
every cell is measured on identical windows in identical order.

WHY THE HEADLINE NUMBER IS TWO-FOLD AND NOT THE WHOLE SPLIT
------------------------------------------------------------
Per-class thresholds are TUNED on validation. Tuning cuts on the same windows
you then score reports how well the tuner fits, not how well the model
generalises -- with 12 classes and a 0.05-0.95 grid there is a lot of room to
fit noise, and the rare tail (tip-up at 157 positives in the whole training
corpus) is exactly where that bites.

So every cell is scored twice: tune on fold A, score fold B; tune on B, score
A; report the mean. The folds are split **by case**, not by window, because
windows from one case are 30-second neighbours of each other and a window-wise
split would put near-duplicates on both sides.

`self_tuned` is reported alongside as the optimistic number, precisely so the
gap between the two is visible. The shipped +0.0196 serving-threshold gain was
measured self-tuned; if that gap is large here, it was too.

TASK IS SCORED DIFFERENTLY ON PURPOSE. The tool head is multi-label and needs
thresholds; the task head is a softmax over 8 classes and takes an argmax, so
it has no thresholds to overfit and is scored by plain accuracy on all
windows.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.holdout import (case_folds, honest_macro_f1,            # noqa: E402
                            unmeasurable_classes)
from surgvu.aggregate import AGGREGATORS                           # noqa: E402
from surgvu.frames import sample_frame_indices                     # noqa: E402


# Arm combinations. A combination averages the per-frame probabilities of its
# members before aggregating, which is what test-time augmentation means.
COMBOS = {
    "id": ("id",),
    "hflip": ("hflip",),
    "scale448": ("scale448",),
    "id+hflip": ("id", "hflip"),
    "id+scale448": ("id", "scale448"),
    "id+hflip+scale448": ("id", "hflip", "scale448"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probs", required=True, nargs="+",
                        help="one or more .npz from dump_frame_probs.py. "
                             "Give several and their per-frame probabilities "
                             "are averaged before aggregation, which is "
                             "experiment 4's ensemble -- the second spotter. "
                             "Averaging happens at the PROBABILITY level, "
                             "not at the prediction level, because two models "
                             "that each miss a class under their own "
                             "threshold can still average above a re-tuned "
                             "one; voting on hard predictions throws away "
                             "exactly the evidence the ensemble is for.")
    parser.add_argument("--out", help="write the full result table as JSON")
    parser.add_argument("--frames", default="8,16,24,30")
    parser.add_argument("--top", type=int, default=20, help="rows to print")
    args = parser.parse_args(argv)

    sources = [np.load(path, allow_pickle=False) for path in args.probs]
    data = sources[0]
    arms = [str(a) for a in data["arms"]]
    depth = int(data["depth"][0])
    tools_target = data["tools_target"]
    task_target = data["task_target"]
    cases = data["cases"]
    # Targets passed on purpose: the balanced split matters here. An
    # alternating one left vessel sealer at 50/186 and tip-up at 0/68.
    fold_a, fold_b = case_folds(cases, tools_target)

    # Every source must describe the SAME windows in the SAME order, or the
    # average silently pairs one model's window 7 with another's window 7 from
    # a different shard ordering. Checked on the labels rather than trusted,
    # because the failure is invisible in the output -- it just scores badly.
    for path, other in zip(args.probs[1:], sources[1:]):
        if not np.array_equal(other["tools_target"], tools_target):
            raise ValueError(
                "%s does not describe the same windows as %s: targets differ. "
                "Ensembling these would align window i of one model with a "
                "different window i of the other." % (path, args.probs[0]))
        if [str(a) for a in other["arms"]] != arms:
            raise ValueError("%s has arms %r, expected %r"
                             % (path, [str(a) for a in other["arms"]], arms))
    if len(sources) > 1:
        print("ENSEMBLE of %d models: %s" % (len(sources), args.probs))

    print("windows=%d depth=%d arms=%s cases=%d  folds %d/%d windows"
          % (len(tools_target), depth, arms, len(set(cases.tolist())),
             len(fold_a), len(fold_b)))
    print("tool positives per class:",
          dict(zip([str(c) for c in data["tool_classes"]],
                   tools_target.sum(axis=0).astype(int).tolist())))
    blind = unmeasurable_classes(tools_target, fold_a, fold_b,
                                 [str(c) for c in data["tool_classes"]])
    if blind:
        print("UNMEASURABLE under case-level folds (all their validation "
              "windows come from a single case, so one fold has no positives "
              "and their F1 is a structural zero in that direction): %s"
              % (", ".join(blind),))
        print("  -> `toolsF1` averages those zeros in and is comparable to "
              "the shipped macro-F1, which does the same.")
        print("  -> `meas` excludes them and is the number that MOVES when a "
              "model improves. Read both.")
    print()

    frame_counts = [int(n) for n in args.frames.split(",") if int(n) <= depth]
    combos = {name: members for name, members in COMBOS.items()
              if all(m in arms for m in members)}
    if not combos:
        raise ValueError("none of the arm combinations are present in %s" % (arms,))

    rows = []
    for combo_name, members in combos.items():
        # Average over arms AND over sources in one step: an ensemble of two
        # models each with three TTA arms is six equally-weighted views of the
        # same window, and there is no reason to privilege the grouping.
        tools_stack = np.mean([src["tools_%s" % m]
                               for src in sources for m in members], axis=0)
        task_stack = np.mean([src["task_%s" % m]
                              for src in sources for m in members], axis=0)
        for n_frames in frame_counts:
            picks = sample_frame_indices(depth, n_frames)
            tools_sub = tools_stack[:, picks, :]
            task_sub = task_stack[:, picks, :]
            for agg_name, aggregate in AGGREGATORS.items():
                result = honest_macro_f1(
                    tools_target, aggregate(tools_sub), fold_a, fold_b)
                task_pred = aggregate(task_sub).argmax(axis=1)
                valid = task_target >= 0
                task_acc = float((task_pred[valid] == task_target[valid]).mean())
                rows.append({
                    "arms": combo_name, "frames": n_frames, "aggregation": agg_name,
                    "tools_macro_f1": result["honest"],
                    "tools_macro_f1_self_tuned": result["self_tuned"],
                    "tools_macro_f1_measurable": result["honest_measurable"],
                    "per_class_f1": result["per_class"],
                    "task_accuracy": task_acc,
                })

    baseline = next(r for r in rows if r["arms"] == "id"
                    and r["frames"] == 16 and r["aggregation"] == "mean")
    for row in rows:
        row["delta_vs_shipped"] = row["tools_macro_f1"] - baseline["tools_macro_f1"]

    rows.sort(key=lambda r: -r["tools_macro_f1"])
    header = ("%-20s %6s %-9s %9s %9s %8s %8s %8s"
              % ("arms", "frames", "agg", "toolsF1", "delta", "meas",
                 "selftune", "taskAcc"))
    print(header)
    print("-" * len(header))
    print("%-20s %6d %-9s %9.4f %9s %8.4f %8.4f %8.4f   <- SHIPPED"
          % (baseline["arms"], baseline["frames"], baseline["aggregation"],
             baseline["tools_macro_f1"], "--",
             baseline["tools_macro_f1_measurable"],
             baseline["tools_macro_f1_self_tuned"], baseline["task_accuracy"]))
    print("-" * len(header))
    for row in rows[:args.top]:
        print("%-20s %6d %-9s %9.4f %+9.4f %8.4f %8.4f %8.4f"
              % (row["arms"], row["frames"], row["aggregation"],
                 row["tools_macro_f1"], row["delta_vs_shipped"],
                 row["tools_macro_f1_measurable"],
                 row["tools_macro_f1_self_tuned"], row["task_accuracy"]))

    best = rows[0]
    print("\nBEST  arms=%s frames=%d aggregation=%s"
          % (best["arms"], best["frames"], best["aggregation"]))
    print("  tools macro-F1 %.4f (shipped %.4f, delta %+.4f)"
          % (best["tools_macro_f1"], baseline["tools_macro_f1"],
             best["delta_vs_shipped"]))
    print("  self-tuned      %.4f  <- the optimistic number; the gap is the "
          "threshold overfit" % (best["tools_macro_f1_self_tuned"],))
    print("  task accuracy   %.4f (shipped %.4f)"
          % (best["task_accuracy"], baseline["task_accuracy"]))

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"probs": args.probs, "baseline": baseline, "rows": rows},
            indent=2), encoding="utf-8")
        print("\nwrote %s (%d cells)" % (args.out, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
