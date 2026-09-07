"""Tune the tool cuts against ANSWER accuracy instead of against macro-F1.

THE MISMATCH THIS EXISTS TO MEASURE. `serving_thresholds` maximise per-class
macro-F1 over validation windows. The graded output is not macro-F1 -- it is a
sentence, and for the largest intent (`tool_presence_polar`) that sentence is
"Yes" or "No". Those two objectives do not share an optimum:

  * F1 ignores TRUE NEGATIVES entirely. Answer accuracy counts them, and for a
    rare tool the overwhelming majority of questions have the answer "No".
  * F1 on a rare class is dominated by recall, so its optimal cut sits LOW --
    a false positive costs little F1 but flips a correct "No" into a wrong
    "Yes".

So an F1-tuned cut should systematically over-answer "Yes" on rare tools, and
the rarer the tool the worse it should get. That is a prediction, and this
script tests it on the 4,635 validation windows we already have probabilities
for. No GPU, no new data.

WHAT THE SCORE MEANS. Polarity errors are not free but they are cheap: a
flipped Yes/No scores 0.7015 against the reference, where an exact match
scores 1.0000. Both numbers are measured, not assumed. So the expected
contribution of one polar question is

    accuracy * 1.0000 + (1 - accuracy) * 0.7015

and maximising accuracy maximises the graded score. The absolute value is
reported alongside so the size of the prize is visible rather than implied.

HONESTY. Thresholds are tuned on one case fold and scored on the other, the
same protocol as every other number in this project, because a cut tuned and
scored on the same windows is an optimistic fiction -- it has been worth up to
0.016 elsewhere here.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.aggregate import AGGREGATORS            # noqa: E402
from surgvu.frames import sample_frame_indices      # noqa: E402
from surgvu.holdout import case_folds               # noqa: E402
from surgvu.metrics import macro_f1, tune_thresholds  # noqa: E402

EXACT = 1.0000
WRONG_POLARITY = 0.7015

GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def accuracy_cuts(target, clip):
    """Per-class cut maximising ANSWER accuracy on polar questions.

    One independent sweep per class: a polar question names one tool, so the
    classes do not interact the way they would under a joint objective.
    """
    cuts = np.zeros(target.shape[1], dtype=np.float32)
    for c in range(target.shape[1]):
        truth = target[:, c] > 0.5
        best, best_cut = -1.0, 0.5
        for cut in GRID:
            acc = float(((clip[:, c] >= cut) == truth).mean())
            if acc > best:
                best, best_cut = acc, float(cut)
        cuts[c] = best_cut
    return cuts


def report(target, clip, cuts):
    """Per-class answer accuracy, plus the Yes-rate against the truth rate."""
    rows = []
    for c in range(target.shape[1]):
        truth = target[:, c] > 0.5
        said_yes = clip[:, c] >= cuts[c]
        rows.append({
            "accuracy": float((said_yes == truth).mean()),
            "said_yes_rate": float(said_yes.mean()),
            "true_rate": float(truth.mean()),
            # The two failure directions, kept apart on purpose: over-answering
            # "Yes" and under-answering it are both accuracy losses but they
            # come from opposite ends of the cut and want opposite fixes.
            "false_yes": float((said_yes & ~truth).mean()),
            "false_no": float((~said_yes & truth).mean()),
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probs", required=True, nargs="+")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--aggregation", default="mean", choices=sorted(AGGREGATORS))
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    sources = [np.load(p, allow_pickle=False) for p in args.probs]
    data = sources[0]
    target = data["tools_target"]
    classes = [str(c) for c in data["tool_classes"]]
    depth = int(data["depth"][0])

    picks = sample_frame_indices(depth, args.frames)
    stack = np.mean([s["tools_id"] for s in sources], axis=0)
    clip = AGGREGATORS[args.aggregation](stack[:, picks, :])

    fold_a, fold_b = case_folds(data["cases"], target)

    honest = {"f1": [], "acc": []}
    per_class = {"f1": [], "acc": []}
    for tune, score in ((fold_a, fold_b), (fold_b, fold_a)):
        f1_cuts = tune_thresholds(target[tune], clip[tune])
        acc_cuts = accuracy_cuts(target[tune], clip[tune])
        for name, cuts in (("f1", f1_cuts), ("acc", acc_cuts)):
            rows = report(target[score], clip[score], cuts)
            honest[name].append(np.mean([r["accuracy"] for r in rows]))
            per_class[name].append([r["accuracy"] for r in rows])

    acc_f1 = float(np.mean(honest["f1"]))
    acc_acc = float(np.mean(honest["acc"]))
    pc_f1 = np.mean(per_class["f1"], axis=0)
    pc_acc = np.mean(per_class["acc"], axis=0)

    # Reported on the full split so the shipped-cut diagnosis is visible; the
    # HEADLINE comparison above is the honest two-fold one.
    shipped = tune_thresholds(target, clip)
    diag = report(target, clip, shipped)

    print("polar-question answer accuracy, honest two-fold")
    print("  cuts tuned for macro-F1   %.4f" % acc_f1)
    print("  cuts tuned for accuracy   %.4f" % acc_acc)
    print("  delta                     %+.4f" % (acc_acc - acc_f1))
    print()
    print("expected BERTScore contribution per polar question")
    print("  (exact %.4f, wrong polarity %.4f)" % (EXACT, WRONG_POLARITY))
    for name, acc in (("macro-F1 cuts", acc_f1), ("accuracy cuts", acc_acc)):
        print("  %-14s %.4f" % (name, acc * EXACT + (1 - acc) * WRONG_POLARITY))
    print()
    print("%-32s %6s %6s %8s | %8s %8s %8s"
          % ("class", "F1cut", "ACCcut", "delta", "trueRate", "yesRate", "falseYes"))
    for i, name in enumerate(classes):
        d = diag[i]
        print("%-32s %6.4f %6.4f %+8.4f | %8.3f %8.3f %8.3f"
              % (name, pc_f1[i], pc_acc[i], pc_acc[i] - pc_f1[i],
                 d["true_rate"], d["said_yes_rate"], d["false_yes"]))

    print()
    print("macro-F1 under each cut set, for the cost side of the trade:")
    for name, cuts in (("f1", tune_thresholds(target, clip)),
                       ("acc", accuracy_cuts(target, clip))):
        print("  %-4s macro-F1 %.4f" % (name, macro_f1(target, (clip >= cuts).astype(np.float32))))

    if args.out:
        Path(args.out).write_text(json.dumps({
            "accuracy_f1_cuts": acc_f1, "accuracy_acc_cuts": acc_acc,
            "classes": classes,
            "per_class_f1_cuts": pc_f1.tolist(),
            "per_class_acc_cuts": pc_acc.tolist(),
            "diagnostic_full_split": diag,
            "accuracy_cuts": accuracy_cuts(target, clip).tolist(),
        }, indent=2), encoding="utf-8")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
