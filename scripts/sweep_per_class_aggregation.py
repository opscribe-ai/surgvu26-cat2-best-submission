"""Experiment 1b: let every class choose its own clip aggregator.

The global sweep says top3 beats mean by +0.0142 -- but that average is 6
classes improving and 5 regressing. The split is not random:

    force bipolar          +0.0756      clip applier   -0.0192
    permanent cautery      +0.0491      cadiere        -0.0090
    prograsp               +0.0310      needle driver  -0.0084
    vessel sealer          +0.0302      stapler        -0.0081
    monopolar              +0.0178

The winners are instruments that appear INTERMITTENTLY inside a window; the
losers are the ones that are on screen almost the whole time. That is exactly
what the two aggregators are for. A top-k recovers a tool visible in six
frames of thirty, which a mean averages into the floor. On a tool that is
present in all thirty, a top-k throws away the evidence that would have
separated it from a confident false positive, so the mean is strictly better
information.

So a single global choice is the wrong shape for this decision, and there is
no reason to make one: `perceive.tools_present` already applies a PER-CLASS
threshold vector. Carrying a per-class aggregator alongside it costs one more
list in the config and no extra compute at serving time.

HOW THE CHOICE STAYS HONEST
---------------------------
The aggregator for a class is chosen using ONLY the tuning fold -- tune that
class's threshold on fold A, score it on fold A, take the argmax -- and then
the chosen pair is evaluated on fold B, which was never consulted. Then the
folds swap and the two results are averaged, the same protocol the global
sweep uses. Selecting on the fold you report would manufacture most of the
gain out of noise, and with 9 aggregators x 12 classes there is a lot of
noise available to manufacture from.

The in-sample step is why `selection_agreement` is printed: if the two folds
independently pick the same aggregator for a class, that class's choice is a
property of the instrument. If they disagree, it is a property of the fold,
and the honest reading is that the class has no preference worth shipping.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.aggregate import AGGREGATORS                          # noqa: E402
from surgvu.frames import sample_frame_indices                    # noqa: E402
from surgvu.holdout import case_folds, unmeasurable_classes       # noqa: E402
from surgvu.metrics import per_class_f1, tune_thresholds          # noqa: E402
from surgvu.taxonomy import TOOL_CLASSES                          # noqa: E402


def one_class_f1(target_col, prob_col, threshold):
    pred = (prob_col >= threshold).astype(np.float32)
    return float(per_class_f1(target_col[:, None], pred[:, None])[0])


def choose_and_score(clip_by_agg, target, tune_idx, score_idx, names):
    """Per class: pick an aggregator on `tune_idx`, score it on `score_idx`."""
    chosen, scores = [], []
    for c in range(target.shape[1]):
        best_name, best_in_sample, best_cut = None, -1.0, 0.5
        for name in names:
            probs = clip_by_agg[name][:, c]
            cuts = tune_thresholds(target[tune_idx][:, c:c + 1],
                                   probs[tune_idx][:, None])
            f1 = one_class_f1(target[tune_idx][:, c], probs[tune_idx],
                              float(cuts[0]))
            if f1 > best_in_sample:
                best_name, best_in_sample, best_cut = name, f1, float(cuts[0])
        chosen.append(best_name)
        scores.append(one_class_f1(target[score_idx][:, c],
                                   clip_by_agg[best_name][score_idx][:, c],
                                   best_cut))
    return chosen, np.array(scores)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probs", required=True, nargs="+")
    parser.add_argument("--frames", type=int, default=8,
                        help="8 is where the global sweep peaked")
    parser.add_argument("--aggregators", default="mean,top3,top5,q90,trim20,q75")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    sources = [np.load(p, allow_pickle=False) for p in args.probs]
    data = sources[0]
    target = data["tools_target"]
    depth = int(data["depth"][0])
    picks = sample_frame_indices(depth, args.frames)
    stack = np.mean([s["tools_id"] for s in sources], axis=0)[:, picks, :]
    fold_a, fold_b = case_folds(data["cases"], target)

    names = [n.strip() for n in args.aggregators.split(",") if n.strip()]
    clip_by_agg = {n: AGGREGATORS[n](stack) for n in names}
    blind = set(unmeasurable_classes(target, fold_a, fold_b))

    chosen_a, score_b = choose_and_score(clip_by_agg, target, fold_a, fold_b, names)
    chosen_b, score_a = choose_and_score(clip_by_agg, target, fold_b, fold_a, names)
    per_class = (score_a + score_b) / 2.0

    # Global baselines under the identical protocol, for the comparison.
    def global_score(name):
        out = []
        for tune_idx, score_idx in ((fold_a, fold_b), (fold_b, fold_a)):
            cuts = tune_thresholds(target[tune_idx], clip_by_agg[name][tune_idx])
            pred = (clip_by_agg[name][score_idx] >= cuts).astype(np.float32)
            out.append(per_class_f1(target[score_idx], pred))
        return np.mean(out, axis=0)

    baselines = {n: global_score(n) for n in names}

    print("frames=%d  aggregators=%s  sources=%d\n" % (args.frames, names, len(sources)))
    print("%-32s %8s %8s %9s   %-8s %-8s %s"
          % ("class", "mean", "top3", "per-class", "foldA", "foldB", "agree"))
    agree = 0
    for c, name in enumerate(TOOL_CLASSES):
        same = chosen_a[c] == chosen_b[c]
        agree += int(same)
        print("%-32s %8.4f %8.4f %9.4f   %-8s %-8s %s%s"
              % (name, baselines["mean"][c], baselines.get("top3", baselines["mean"])[c],
                 per_class[c], chosen_a[c], chosen_b[c], "yes" if same else "NO",
                 "   <-- unmeasurable" if c in blind else ""))

    keep = [c for c in range(target.shape[1]) if c not in blind]
    print("\n%-32s %8.4f %8.4f %9.4f"
          % ("MACRO (all 12)", baselines["mean"].mean(),
             baselines["top3"].mean(), per_class.mean()))
    print("%-32s %8.4f %8.4f %9.4f"
          % ("MACRO (measurable)", baselines["mean"][keep].mean(),
             baselines["top3"][keep].mean(), per_class[keep].mean()))
    print("\nselection_agreement: %d/%d classes chose the same aggregator on "
          "both folds." % (agree, len(TOOL_CLASSES)))
    print("A class whose folds disagree has no preference worth shipping -- "
          "its apparent gain is fold noise.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "frames": args.frames, "aggregators": names, "sources": args.probs,
            "chosen_fold_a": chosen_a, "chosen_fold_b": chosen_b,
            "per_class_f1": per_class.tolist(),
            "macro_all": float(per_class.mean()),
            "macro_measurable": float(per_class[keep].mean()),
            "global": {n: float(v.mean()) for n, v in baselines.items()},
        }, indent=2), encoding="utf-8")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
