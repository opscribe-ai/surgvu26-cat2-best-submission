"""Does the 3D task head know anything the 2D task head does not?

THE QUESTION THIS ANSWERS, AND WHY IT IS NOT "which head is better". Head to
head the 3D model is handicapped four ways at once -- 18 layers against a
ResNet-50, Kinetics against ImageNet, 112px against 384, a 2 s burst against
30 s of window -- so if it loses, the loss is unattributable. Fusion is the
test that survives those confounds: a weaker model still moves a blend when it
is seeing something the stronger one is not. If no blend weight beats the 2D
head alone, the 3D head is redundant given the 2D head, which is a statement
about information rather than about capacity.

This is the same experiment `fuse_2d_3d.py` ran on tools, where the answer was
a flat null: no weight beat 2D alone, and the folds disagreed about which
weight to prefer. Tools was the wrong place to look -- instrument identity is
an appearance question. Task labels name what the instruments are DOING, so
this is the place the tools null does not settle.

WHAT IS SCORED. Description accuracy first, plain class accuracy second.
Three task classes share a modal description, so a confusion inside that group
never reaches an answer; a fusion that improved class accuracy while leaving
description accuracy flat would have improved nothing that ships.

TWO FOLDS, EVEN THOUGH NOTHING NEEDS TUNING. The heads themselves take an
argmax and have no thresholds. The BLEND WEIGHT is a tuned parameter, though,
and a weight chosen on the windows it is then scored on reports the choosing.
So w is picked on one case fold and scored on the other, both directions, and
if the two folds pick different weights the gain is noise -- the same rule
that has now killed six apparent gains in this project.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.descriptions import description_accuracy, load_corpus  # noqa: E402
from surgvu.frames import sample_frame_indices                  # noqa: E402
from surgvu.holdout import case_folds                           # noqa: E402
from surgvu.metrics import macro_f1                             # noqa: E402
from surgvu.taxonomy import TASK_CLASSES                        # noqa: E402

TWO_D = "/staging/n/nkalthoff/surgvu26/v2/frame_probs_resnetlong_val.npz"
THREE_D = "/staging/n/nkalthoff/surgvu26/dumps/clip_task_r2plus1d.npz"
REPO = Path(__file__).resolve().parents[1]


def load_pair(two_path, three_path, frames, three_arm):
    two = np.load(two_path, allow_pickle=False)
    three = np.load(three_path, allow_pickle=False)

    if not np.array_equal(two["cases"], three["cases"]):
        raise SystemExit(
            "the two dumps do not describe the same windows in the same order "
            "(%d vs %d rows)." % (len(two["cases"]), len(three["cases"])))
    if not np.array_equal(two["task_target"], three["task_target"]):
        raise SystemExit(
            "the two dumps disagree on the task labels themselves. The 2D dump "
            "writes -1 for a task name it does not recognise while the 3D dump "
            "refuses to; a mismatch here most likely means unlabelled windows.")

    picks = sample_frame_indices(two["task_id"].shape[1], frames)
    p2 = two["task_id"][:, picks, :].mean(axis=1)
    p3 = three["task_%s" % three_arm]
    if p2.shape != p3.shape:
        raise SystemExit("shape mismatch: 2D %r vs 3D %r" % (p2.shape, p3.shape))
    return (p2.astype(np.float32), p3.astype(np.float32),
            two["task_target"].astype(np.int64), two["cases"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--two-d", default=TWO_D)
    parser.add_argument("--three-d", default=THREE_D)
    parser.add_argument("--descriptions",
                        default=str(REPO / "config" / "descriptions.yaml"))
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--three-arm", default="mean3")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    p2, p3, truth, cases = load_pair(args.two_d, args.three_d, args.frames,
                                     args.three_arm)
    corpus = load_corpus(args.descriptions)
    eye = np.eye(len(TASK_CLASSES), dtype=np.float32)
    fold_a, fold_b = case_folds(cases, eye[truth])
    grid = [round(w, 2) for w in np.arange(0.0, 1.0 + args.step / 2, args.step)]

    def described(probs, rows):
        pred = probs[rows].argmax(axis=1)
        return float(description_accuracy(truth[rows], pred, corpus))

    def accurate(probs, rows):
        return float((probs[rows].argmax(axis=1) == truth[rows]).mean())

    print("windows %d | classes %d | folds %d / %d"
          % (len(truth), len(TASK_CLASSES), len(fold_a), len(fold_b)))
    print("2D: %s (mean over %d frames)" % (Path(args.two_d).name, args.frames))
    print("3D: %s (arm %s)" % (Path(args.three_d).name, args.three_arm))

    everything = np.arange(len(truth))
    print("\npooled, for provenance:")
    print("  2D alone   acc %.4f  macroF1 %.4f  desc %.4f"
          % (accurate(p2, everything),
             macro_f1(eye[truth], eye[p2.argmax(axis=1)]),
             described(p2, everything)))
    print("  3D alone   acc %.4f  macroF1 %.4f  desc %.4f"
          % (accurate(p3, everything),
             macro_f1(eye[truth], eye[p3.argmax(axis=1)]),
             described(p3, everything)))
    print("  the 2D row should sit near the shipped task head's 0.9456; if it "
          "does not, this dump is not the checkpoint that number came from.")

    # Column names say WHERE a score was measured, not "tuned on A scored on
    # B": at a fixed w nothing is being tuned, so the honest label is the fold.
    # The tuning shows up once, below, in which w each fold picks.
    print("\n%-6s %10s %10s %10s %10s"
          % ("w", "desc(A)", "desc(B)", "acc(A)", "acc(B)"))
    table = {}
    for w in grid:
        blend = (1.0 - w) * p2 + w * p3
        table[w] = {
            "desc_a": described(blend, fold_a),
            "desc_b": described(blend, fold_b),
            "acc_a": accurate(blend, fold_a),
            "acc_b": accurate(blend, fold_b),
        }
        print("%-6.2f %10.4f %10.4f %10.4f %10.4f"
              % (w, table[w]["desc_a"], table[w]["desc_b"],
                 table[w]["acc_a"], table[w]["acc_b"]))

    def pick(rows):
        """The weight this fold would choose, judged on itself."""
        best, chosen = -1.0, None
        for w in grid:
            score = described((1.0 - w) * p2 + w * p3, rows)
            if score > best:
                best, chosen = score, w
        return chosen

    w_a, w_b = pick(fold_a), pick(fold_b)
    mean_desc = {w: 0.5 * (table[w]["desc_a"] + table[w]["desc_b"])
                 for w in grid}
    baseline = mean_desc[0.0]
    # Cross, not self: the weight fold A picked is scored on B and vice versa.
    fused = 0.5 * (table[w_a]["desc_b"] + table[w_b]["desc_a"])
    agree = w_a == w_b

    print("\nfold A chooses w=%.2f, fold B chooses w=%.2f" % (w_a, w_b))
    print("2D alone (w=0)              %.4f" % baseline)
    print("3D alone (w=1)              %.4f" % mean_desc[1.0])
    print("fusion at the chosen w      %.4f   delta %+.4f"
          % (fused, fused - baseline))
    print("smallest nudge (w=%.2f)     %.4f   delta %+.4f"
          % (grid[1], mean_desc[grid[1]], mean_desc[grid[1]] - baseline))

    best_w = max(grid, key=lambda w: mean_desc[w])
    print()
    if not agree:
        print("FOLD AGREEMENT: DISAGREE (%.2f vs %.2f)." % (w_a, w_b))
    else:
        print("FOLD AGREEMENT: both folds chose %.2f." % w_a)

    if best_w == 0.0:
        print("VERDICT: no weight beats the 2D head alone. The 3D task head is "
              "redundant given the 2D one -- its 2 s of motion carries nothing "
              "the 16 sampled frames do not already carry, on the axis where "
              "motion had the best chance of mattering.")
    elif not agree:
        print("VERDICT: a weight wins on the grid but the folds disagree about "
              "which. Treat as a null.")
    else:
        print("VERDICT: fusion wins at w=%.2f with both folds agreeing, "
              "+%.4f description accuracy. Motion carries something appearance "
              "does not, and the perception config should carry both heads."
              % (best_w, mean_desc[best_w] - baseline))

    if args.out:
        Path(args.out).write_text(json.dumps({
            "two_d": args.two_d, "three_d": args.three_d,
            "frames": args.frames, "three_arm": args.three_arm,
            "windows": int(len(truth)), "classes": list(TASK_CLASSES),
            "grid": {str(w): dict(table[w], mean_desc=mean_desc[w])
                     for w in grid},
            "fold_a_weight": w_a, "fold_b_weight": w_b,
            "folds_agree": bool(agree),
            "baseline_2d": baseline, "fused": fused,
            "delta": fused - baseline,
        }, indent=2), encoding="utf-8")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
