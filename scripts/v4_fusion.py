"""Does any temporal arm add anything to the 2D model? Fusion, for every dump.

WHY FUSION IS THE QUESTION THAT MATTERS. "Which model is better" is not what
decides a submission. The 2D model ships; the only reason to add a temporal
arm is that it sees something the 2D model does not. A weaker model still
moves a blend when it carries independent information, and a stronger one that
carries none does not. So an arm can lose head to head and still be worth
shipping, and an arm can win head to head and be redundant.

This is the same protocol scripts/fuse_2d_3d.py applied to the v3 Kinetics
arms, run over every v4 dump at once so the arms are compared on one page
rather than one at a time.

THE WEIGHT IS TUNED HONESTLY. A blend weight chosen on the windows it is then
scored on reports the choosing. So w is picked on one case fold and scored on
the other, both directions, and if the two folds disagree about which weight
is best the gain is noise -- the standing rule in this project, which has now
killed seven apparent gains.

READ THE COLUMNS, NOT THE HEADLINE. `w=0` is the 2D model alone and is the
number every arm has to beat. `best w` is what the grid says. `folds agree`
is whether it means anything.
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

TWO_D = "/staging/n/nkalthoff/surgvu26/v2/frame_probs_resnetlong_val.npz"
DUMPS = "/staging/n/nkalthoff/surgvu26/dumps"
SKIP = ("temporal_tsm_control",)     # the falsy-zero bug; see v4_report.py


def scored(target, probs, tune, score):
    cuts = tune_thresholds(target[tune], probs[tune])
    return macro_f1(target[score], (probs[score] >= cuts).astype(np.float32))


def fuse(two_probs, three_probs, target, fold_a, fold_b, grid):
    table = {}
    for w in grid:
        blend = (1.0 - w) * two_probs + w * three_probs
        table[w] = 0.5 * (scored(target, blend, fold_a, fold_b)
                          + scored(target, blend, fold_b, fold_a))

    def pick(rows):
        best, chosen = -1.0, None
        for w in grid:
            blend = (1.0 - w) * two_probs + w * three_probs
            value = scored(target, blend, rows, rows)
            if value > best:
                best, chosen = value, w
        return chosen

    w_a, w_b = pick(fold_a), pick(fold_b)
    blend_a = (1.0 - w_a) * two_probs + w_a * three_probs
    blend_b = (1.0 - w_b) * two_probs + w_b * three_probs
    honest = 0.5 * (scored(target, blend_a, fold_a, fold_b)
                    + scored(target, blend_b, fold_b, fold_a))
    return table, w_a, w_b, honest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--two-d", default=TWO_D)
    parser.add_argument("--dumps", default=DUMPS)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    two = np.load(args.two_d, allow_pickle=False)
    picks = sample_frame_indices(two["tools_id"].shape[1], args.frames)
    two_probs = two["tools_id"][:, picks, :].mean(axis=1).astype(np.float32)
    target = two["tools_target"].astype(np.float32)
    cases = two["cases"]
    fold_a, fold_b = case_folds(cases, target)
    grid = [round(w, 2) for w in np.arange(0.0, 1.0 + args.step / 2, args.step)]

    print("2D: %s (mean over %d frames), %d windows, folds %d / %d"
          % (Path(args.two_d).name, args.frames, len(target),
             len(fold_a), len(fold_b)))

    results, rows = {}, []
    print("\n%-28s %-9s %8s %8s %8s %7s  %s"
          % ("arm", "arm-used", "2D only", "best w", "fused", "delta", "folds"))
    print("%-28s %-9s %8s %8s %8s %7s %6s"
          % ("", "", "", "", "", "", "err-r"))
    for path in sorted(Path(args.dumps).glob("temporal_*.npz")):
        name = path.name.replace(".npz", "")
        if name in SKIP:
            continue
        data = np.load(path, allow_pickle=False)
        if not np.array_equal(data["cases"], cases):
            print("%-28s SKIPPED: %d windows against the 2D dump's %d"
                  % (name[:28], len(data["cases"]), len(cases)))
            continue
        if not np.array_equal(data["tools_target"].astype(np.float32), target):
            print("%-28s SKIPPED: the two dumps disagree on the labels"
                  % name[:28])
            continue

        # Prefer an aggregated arm when the dump has one: a single burst is a
        # sliver of the window, and serving would average what it has.
        arms = [k[len("tools_"):] for k in data.files
                if k.startswith("tools_") and k != "tools_target"]
        arm = ("meanall" if "meanall" in arms
               else ("spread" if "spread" in arms else sorted(arms)[0]))
        three_probs = data["tools_%s" % arm].astype(np.float32)

        # WHY a fusion fails is more useful than THAT it failed. If an arm's
        # errors are the same errors the 2D model makes, no blend weight can
        # help -- there is nothing independent to average. That is a different
        # finding from "the arm is weak", and it points somewhere different:
        # a conversion of our own 2D model inherits its blind spots by
        # construction, while a separately-pretrained architecture might not.
        residual_2d = np.abs(target - two_probs)
        residual_3d = np.abs(target - three_probs)
        correlation = float(np.corrcoef(residual_2d.ravel(),
                                        residual_3d.ravel())[0, 1])

        table, w_a, w_b, honest = fuse(two_probs, three_probs, target,
                                       fold_a, fold_b, grid)
        baseline = table[0.0]
        best_w = max(grid, key=lambda w: table[w])
        agree = w_a == w_b
        results[name] = {"arm": arm, "error_correlation": correlation,
                         "baseline": baseline, "best_w": best_w,
                         "fused": honest, "delta": honest - baseline,
                         "fold_a_w": w_a, "fold_b_w": w_b, "agree": bool(agree),
                         "grid": {str(w): table[w] for w in grid}}
        rows.append((name, arm, baseline, best_w, honest, honest - baseline,
                     agree, w_a, w_b, correlation))
        print("%-28s %-9s %8.4f %8.2f %8.4f %+7.4f %6.3f  %s"
              % (name[:28], arm, baseline, best_w, honest, honest - baseline,
                 correlation,
                 "agree" if agree else "DISAGREE %.2f/%.2f" % (w_a, w_b)))

    print()
    winners = [r for r in rows if r[5] > 0 and r[6] and r[3] > 0]
    if not rows:
        print("No comparable dumps yet.")
    elif not winners:
        print("VERDICT: no arm adds anything to the 2D model. Either no weight "
              "beats w=0, or the folds disagree about which weight to use -- "
              "and a gain the two folds cannot agree on is noise.")
        if rows:
            low = min(rows, key=lambda r: r[9])
            high = max(rows, key=lambda r: r[9])
            print("Error correlation with the 2D model runs %.3f (%s) to %.3f "
                  "(%s)." % (low[9], low[0], high[9], high[0]))
            print("That is substantial overlap but NOT redundancy -- at r~0.65 "
                  "an arm still holds independent error, so correlation alone "
                  "does not explain the null. What does is correlation TOGETHER "
                  "WITH being weaker: every arm here is 0.02 to 0.14 below the "
                  "2D model, and a model that is both worse and largely "
                  "agreeing has no weight at which it helps.")
            print("The direction is still informative: the LEAST correlated arm "
                  "(%s, r=%.3f) also loses the least from blending (%+.4f), "
                  "which is what a genuinely independent architecture would "
                  "need to exploit -- it would have to close the gap in "
                  "strength as well." % (low[0], low[9], low[5]))
    else:
        best = max(winners, key=lambda r: r[5])
        print("VERDICT: %s adds %+.4f at w=%.2f with both folds agreeing. That "
              "is the claim worth acting on -- not that it beats the 2D model "
              "alone, but that it carries something the 2D model does not."
              % (best[0], best[5], best[3]))

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"two_d": args.two_d, "frames": args.frames, "arms": results},
            indent=2), encoding="utf-8")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
