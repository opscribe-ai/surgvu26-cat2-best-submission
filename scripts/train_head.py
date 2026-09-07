"""Experiment 6, part 2: heads on frozen EndoViT features.

`endovit_features.py` cached the expensive part, so a head trains in seconds
and a whole grid of them trains in minutes. That changes what is worth asking:
instead of one head and a verdict, this sweeps head architecture, learning
rate, epochs and -- the interesting one -- the `pos_weight` ceiling.

WHY THE pos_weight CEILING IS SWEPT HERE
-----------------------------------------
That ceiling is experiment 5's whole hypothesis. `scripts/train_tools.py`
clips per-class weights at 50, while tip-up fenestrated grasper's true
negatives/positives ratio is ~175 and stapler's ~200. Both classes are the
ones dragging macro-F1 down, and tip-up scores exactly 0.0000 in the shipped
checkpoint -- it is never predicted at all.

Testing that on the real CNN costs a training run per setting. Testing it on
frozen features costs seconds. If the ceiling turns out not to matter here it
is weak evidence about the CNN, but if it matters a lot it says the full
retrain is worth its GPU hours -- which is the decision this is informing.

SCORING IS `surgvu.holdout`, THE SAME FUNCTION THE AGGREGATION SWEEP CALLS,
so an EndoViT number and an EfficientNet number are directly comparable. Both
are clip-level macro-F1 on splits_v2 val with thresholds tuned on one case
fold and scored on the other. Neither is the per-frame 0.6605 recorded in the
shipped checkpoint; do not compare against that number.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.aggregate import AGGREGATORS                  # noqa: E402
from surgvu.holdout import (case_folds, honest_macro_f1,  # noqa: E402
                            unmeasurable_classes)
from surgvu.taxonomy import TASK_CLASSES, TOOL_CLASSES   # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def pos_weights(targets, ceiling):
    """negatives/positives per class, clipped at `ceiling`, floored at 1.

    Counted from the actual training targets rather than read from
    config/tool_frequency.json. The table's denominator is a documented
    stand-in (the most common class's count) that understates mid-frequency
    weights; here the exact counts are already in hand, so there is no reason
    to inherit the approximation.
    """
    positives = targets.sum(axis=0)
    negatives = len(targets) - positives
    raw = np.divide(negatives, np.maximum(positives, 1.0))
    return np.clip(raw, 1.0, ceiling).astype(np.float32), raw


def build_head(kind, in_dim, out_dim, dropout=0.2):
    import torch.nn as nn

    if kind == "linear":
        return nn.Linear(in_dim, out_dim)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(in_dim, 512), nn.GELU(),
                             nn.Dropout(dropout), nn.Linear(512, out_dim))
    raise ValueError("unknown head %r" % (kind,))


def fit(features, targets, kind, out_dim, weights, epochs, lr, device,
        multilabel=True, batch=4096, seed=7):
    """Train one head on flattened per-frame features. Returns the module."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    head = build_head(kind, features.shape[1], out_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    if multilabel:
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(weights, device=device))
    else:
        loss_fn = nn.CrossEntropyLoss()

    x = torch.from_numpy(features.astype(np.float32))
    y = torch.from_numpy(targets)
    n = len(x)
    for _ in range(epochs):
        head.train()
        order = torch.randperm(n)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            xb = x[idx].to(device)
            yb = y[idx].to(device)
            optimizer.zero_grad()
            loss_fn(head(xb), yb).backward()
            optimizer.step()
    return head


def predict(head, features, device, multilabel=True, batch=8192):
    """(N, D) features -> (N, C) probabilities."""
    import torch

    head.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(features), batch):
            xb = torch.from_numpy(
                features[start:start + batch].astype(np.float32)).to(device)
            logits = head(xb)
            probs = (torch.sigmoid(logits) if multilabel
                     else torch.softmax(logits, dim=1))
            out.append(probs.float().cpu().numpy())
    return np.concatenate(out)


def flatten(features, targets):
    """(W, F, D) + (W, ...) -> (W*F, D) + labels repeated per frame."""
    windows, frames, dim = features.shape
    return (features.reshape(windows * frames, dim),
            np.repeat(targets, frames, axis=0))


def standardiser(features):
    """Per-dimension mean and std, fitted on TRAIN frames only.

    Not cosmetic. EndoViT's raw CLS vectors sit in a very narrow cone -- mean
    pairwise cosine 0.995 across the validation split -- because a large
    shared component dominates every embedding. That is normal for MAE/ViT
    features, and it means the informative variation lives in directions whose
    scale is tiny next to the common one. A head fed raw vectors spends its
    early epochs learning to subtract that constant instead of learning the
    task, which is visible as a model that is still improving at the last
    epoch you gave it.

    Fitted on train and applied to val, never refitted on val: the statistics
    are part of the model, and refitting them per split would leak.
    """
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", required=True, help="train features .npz")
    parser.add_argument("--val", required=True, help="val features .npz")
    parser.add_argument("--out", help="write the result table as JSON")
    parser.add_argument("--heads", default="linear,mlp")
    parser.add_argument("--lrs", default="1e-3,3e-4")
    parser.add_argument("--epochs", default="8,20")
    parser.add_argument("--pos-weight-ceilings", default="50,200")
    parser.add_argument("--no-standardise", action="store_true",
                        help="feed the head raw features. Kept as a flag so "
                             "the effect of standardising is measurable "
                             "rather than asserted.")
    parser.add_argument("--aggregation", default="mean",
                        help="how per-frame probabilities are reduced to a\n"
                             "clip. Must match whatever the EfficientNet\n"
                             "number it is compared against used -- the\n"
                             "sweep measured top3 at +0.0142 over mean, so a\n"
                             "mean-vs-top3 comparison would attribute that\n"
                             "gap to the backbone.")
    parser.add_argument("--export-probs",
                        help="write the BEST config's per-frame validation "
                             "probabilities as a dump_frame_probs-compatible "
                             ".npz, so EndoViT can be ensembled with the CNNs "
                             "by scripts/sweep_aggregation.py")
    parser.add_argument("--reference-dump",
                        help="a dump_frame_probs .npz to copy cases, targets "
                             "and TASK probabilities from. Required with "
                             "--export-probs: the ensemble sweep averages the "
                             "task head across sources too, and this "
                             "experiment is about the TOOLS head, so task is "
                             "held constant by copying rather than left to a "
                             "head that was never trained.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    import torch

    device = args.device
    if device in (None, "", "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train = np.load(args.train, allow_pickle=False)
    val = np.load(args.val, allow_pickle=False)
    xtr, ytr = flatten(train["features"], train["tools_target"])
    xva_windows = val["features"]
    yva = val["tools_target"]
    fold_a, fold_b = case_folds(val["cases"], yva)

    print("train %s -> %s frames | val %s windows, %d cases"
          % (train["features"].shape, xtr.shape, xva_windows.shape,
             len(set(val["cases"].tolist()))))
    _, raw = pos_weights(ytr, 1e9)
    print("true neg/pos ratios:",
          {c: round(float(r), 1) for c, r in zip(TOOL_CLASSES, raw)})
    print()

    xva_flat, _ = flatten(xva_windows, yva)
    windows, frames = xva_windows.shape[0], xva_windows.shape[1]

    if args.no_standardise:
        print("features: RAW (--no-standardise)")
    else:
        mean, std = standardiser(xtr.astype(np.float32))
        xtr = ((xtr.astype(np.float32) - mean) / std)
        xva_flat = ((xva_flat.astype(np.float32) - mean) / std)
        print("features: standardised on train (mean/std per dim)")

    grid = list(itertools.product(
        [h.strip() for h in args.heads.split(",")],
        [float(v) for v in args.lrs.split(",")],
        [int(v) for v in args.epochs.split(",")],
        [float(v) for v in args.pos_weight_ceilings.split(",")]))
    print("%d configurations, aggregation=%s\n"
          % (len(grid), args.aggregation))

    rows = []
    header = "%-8s %8s %7s %9s %10s %10s" % (
        "head", "lr", "epochs", "posw_cap", "toolsF1", "selftune")
    print(header)
    print("-" * len(header))
    for kind, lr, epochs, ceiling in grid:
        weights, _ = pos_weights(ytr, ceiling)
        head = fit(xtr, ytr, kind, len(TOOL_CLASSES), weights, epochs, lr, device)
        frame_probs = predict(head, xva_flat, device).reshape(
            windows, frames, len(TOOL_CLASSES))
        result = honest_macro_f1(
            yva, AGGREGATORS[args.aggregation](frame_probs), fold_a, fold_b)
        honest, self_tuned = result["honest"], result["self_tuned"]
        rows.append({"head": kind, "lr": lr, "epochs": epochs,
                     "pos_weight_ceiling": ceiling,
                     "tools_macro_f1": honest,
                     "tools_macro_f1_self_tuned": self_tuned,
                     "tools_macro_f1_measurable": result["honest_measurable"],
                     "per_class_f1": result["per_class"]})
        print("%-8s %8.0e %7d %9.0f %10.4f %10.4f"
              % (kind, lr, epochs, ceiling, honest, self_tuned), flush=True)

    if args.export_probs:
        if not args.reference_dump:
            raise SystemExit("--export-probs needs --reference-dump; see its help")
        best_cfg = max(rows, key=lambda r: r["tools_macro_f1"])
        weights, _ = pos_weights(ytr, best_cfg["pos_weight_ceiling"])
        head = fit(xtr, ytr, best_cfg["head"], len(TOOL_CLASSES), weights,
                   best_cfg["epochs"], best_cfg["lr"], device)
        frame_probs = predict(head, xva_flat, device).reshape(
            windows, frames, len(TOOL_CLASSES))
        ref = np.load(args.reference_dump, allow_pickle=False)
        if not np.array_equal(ref["tools_target"], yva):
            raise SystemExit(
                "%s describes different windows than %s. Exporting anyway "
                "would align EndoViT's window i with a CNN's different window "
                "i." % (args.reference_dump, args.val))
        payload = {k: ref[k] for k in ("arms", "tool_classes", "task_classes",
                                       "tools_target", "task_target", "cases",
                                       "depth")}
        for arm in [str(a) for a in ref["arms"]]:
            # The same EndoViT probabilities under every arm name: this model
            # has no TTA arms, and the sweep averages arms within a source.
            # Giving it one arm would make the arm axis ragged across sources.
            payload["tools_%s" % arm] = frame_probs.astype(np.float32)
            payload["task_%s" % arm] = ref["task_%s" % arm]
        np.savez_compressed(args.export_probs, **payload)
        print("\nexported %s from head=%s lr=%.0e epochs=%d (macro-F1 %.4f)"
              % (args.export_probs, best_cfg["head"], best_cfg["lr"],
                 best_cfg["epochs"], best_cfg["tools_macro_f1"]))
        print("  task probabilities COPIED from %s -- this exports a TOOLS "
              "model only." % args.reference_dump)

    rows.sort(key=lambda r: -r["tools_macro_f1"])
    best = rows[0]
    print("\nBEST  head=%s lr=%.0e epochs=%d pos_weight_ceiling=%.0f"
          % (best["head"], best["lr"], best["epochs"],
             best["pos_weight_ceiling"]))
    print("  clip-level macro-F1 %.4f (self-tuned %.4f)"
          % (best["tools_macro_f1"], best["tools_macro_f1_self_tuned"]))
    print("  Compare against the EfficientNet baseline from "
          "sweep_aggregation.py, NOT against the checkpoint's per-frame "
          "0.6605 -- different protocol, different number.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"train": args.train, "val": args.val, "rows": rows}, indent=2),
            encoding="utf-8")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
