"""Does the 3D model know anything the 2D model does not?

THE QUESTION THIS ANSWERS, AND WHY IT IS THE RIGHT ONE. Comparing the two
models head to head asks whether the 3D path is BETTER. That is not what we
need to know. A 3D model can be worse overall and still be worth having, if
what it gets right is different from what the 2D model gets right -- motion
evidence layered on top of appearance evidence. The head-to-head comparison
cannot see that; fusion can.

It is also a FAIRER test of the temporal hypothesis than the head-to-head,
because it does not require the 3D model to overcome its handicaps. It has
three simultaneously -- 18 layers against 50, Kinetics-400 against ImageNet,
112px against 384 -- plus a fourth that is easy to miss: the 2D path sees 16
frames spread across the whole 30-second labelled window, while the 3D path
sees a 2-second burst from its centre. The 3D model is answering a question
about 30 seconds having watched 2 of them. Fusion asks only whether that 2
seconds contains anything ADDITIONAL, which is a much lower bar.

READ THE SMALL WEIGHTS FIRST. At w=0.05 the 3D model contributes a 5% nudge to
an otherwise unchanged 2D prediction. A genuinely orthogonal signal shows up
there as a small gain even when the member is weak. If even w=0.05 loses in
both fold directions, the member is not adding an independent view -- it is
adding noise -- and no weighting rescues that. This is the same shape as the
v2 finding that ensembling only pays when members are comparable in quality,
tested at the weight where quality matters least.

PROTOCOL. Honest two folds by case: choose the weight on the tuning fold,
score the other, average the two directions. And the FOLD AGREEMENT CHECK --
if the two folds choose different weights, the gain is noise regardless of its
size. That check has killed five apparent gains on this project so far.

ALIGNMENT IS ASSERTED, NOT ASSUMED. The two dumps come from different shard
pools (sparse and dense) and are only comparable because those pools hold the
same windows in the same order. If the case arrays or the targets ever
disagree, every number below would be a fluent average of mismatched rows, so
this refuses to run rather than reporting it.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.frames import sample_frame_indices                  # noqa: E402
from surgvu.holdout import case_folds                           # noqa: E402
from surgvu.metrics import macro_f1, tune_thresholds            # noqa: E402
from surgvu.taxonomy import TOOL_CLASSES                        # noqa: E402

TWO_D = "/staging/n/nkalthoff/surgvu26/v2/frame_probs_resnetlong_val.npz"
THREE_D = "/staging/n/nkalthoff/surgvu26/dumps/clip_r2plus1d.npz"


def load_pair(two_path, three_path, frames, three_arm):
    """Clip-level probabilities from both models, over identical rows."""
    two = np.load(two_path, allow_pickle=False)
    three = np.load(three_path, allow_pickle=False)

    if not np.array_equal(two["cases"], three["cases"]):
        raise SystemExit(
            "the two dumps do not describe the same windows in the same order "
            "(%d vs %d rows). Fusing them would average mismatched rows and "
            "report a number that looks fine."
            % (len(two["cases"]), len(three["cases"])))
    if not np.array_equal(two["tools_target"], three["tools_target"]):
        raise SystemExit("the two dumps disagree on the labels themselves.")

    # 16 frames, not all 30: that is what serving samples, and it is also the
    # 2D path's best honest configuration (0.7802 against 0.7721 at 30).
    picks = sample_frame_indices(two["tools_id"].shape[1], frames)
    p2 = two["tools_id"][:, picks, :].mean(axis=1)
    p3 = three["tools_%s" % three_arm]
    return p2.astype(np.float32), p3.astype(np.float32), \
        two["tools_target"].astype(np.float32), two["cases"]


def scored(target, probs, tune, score):
    cuts = tune_thresholds(target[tune], probs[tune])
    return macro_f1(target[score], (probs[score] >= cuts).astype(np.float32))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--two-d", default=TWO_D)
    parser.add_argument("--three-d", default=THREE_D)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--three-arm", default="mean3",
                        help="which clip-offset arm of the 3D dump to fuse")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    p2, p3, target, cases = load_pair(args.two_d, args.three_d, args.frames,
                                      args.three_arm)
    fold_a, fold_b = case_folds(cases, target)
    grid = [round(w, 2) for w in np.arange(0.0, 1.0 + args.step / 2, args.step)]

    print("windows %d | classes %d | folds %d / %d"
          % (len(target), target.shape[1], len(fold_a), len(fold_b)))
    print("2D: %s (mean over %d frames)" % (Path(args.two_d).name, args.frames))
    print("3D: %s (arm %s)\n" % (Path(args.three_d).name, args.three_arm))

    print("%-6s %10s %10s" % ("w", "A->B", "B->A"))
    table = {}
    for w in grid:
        blend = (1.0 - w) * p2 + w * p3
        table[w] = (scored(target, blend, fold_a, fold_b),
                    scored(target, blend, fold_b, fold_a))
        print("%-6.2f %10.4f %10.4f" % (w, table[w][0], table[w][1]))

    def pick(tune):
        """The weight the tuning fold would choose, judged on itself."""
        best, chosen = -1.0, None
        for w in grid:
            blend = (1.0 - w) * p2 + w * p3
            score = scored(target, blend, tune, tune)
            if score > best:
                best, chosen = score, w
        return chosen

    w_a, w_b = pick(fold_a), pick(fold_b)
    baseline = float(np.mean(table[0.0]))
    fused = float(np.mean([table[w_a][0], table[w_b][1]]))
    agree = w_a == w_b

    print("\nfold A chooses w=%.2f, fold B chooses w=%.2f" % (w_a, w_b))
    print("2D alone (w=0)              %.4f" % baseline)
    print("3D alone (w=1)              %.4f" % float(np.mean(table[1.0])))
    print("fusion at the chosen w      %.4f   delta %+.4f"
          % (fused, fused - baseline))
    print("smallest nudge (w=%.2f)     %.4f   delta %+.4f"
          % (grid[1], float(np.mean(table[grid[1]])),
             float(np.mean(table[grid[1]])) - baseline))

    print()
    if not agree:
        print("FOLD AGREEMENT: DISAGREE (%.2f vs %.2f). Any gain here is noise "
              "by the standing rule, whatever its size." % (w_a, w_b))
    else:
        print("FOLD AGREEMENT: both folds chose %.2f." % w_a)

    best_w = max(grid, key=lambda w: np.mean(table[w]))
    if best_w == 0.0:
        print("VERDICT: no weight beats the 2D model alone -- not even the "
              "smallest nudge. The 3D model is not contributing an independent "
              "view; within the accuracy this protocol can resolve, its 2 s of "
              "motion holds nothing the 16 sampled frames do not already carry.")
    elif not agree:
        print("VERDICT: a weight wins on the grid but the folds disagree about "
              "which. Treat as a null.")
    else:
        print("VERDICT: fusion wins at w=%.2f and both folds agree. Motion "
              "carries something appearance does not." % best_w)

    if args.out:
        Path(args.out).write_text(json.dumps({
            "two_d": args.two_d, "three_d": args.three_d,
            "frames": args.frames, "three_arm": args.three_arm,
            "windows": int(len(target)),
            "classes": list(TOOL_CLASSES),
            "grid": {str(w): {"a_to_b": table[w][0], "b_to_a": table[w][1],
                              "mean": float(np.mean(table[w]))} for w in grid},
            "fold_a_weight": w_a, "fold_b_weight": w_b,
            "folds_agree": bool(agree),
            "baseline_2d": baseline, "fused": fused,
            "delta": fused - baseline,
        }, indent=2), encoding="utf-8")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
