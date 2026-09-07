"""Train the tool install-state recogniser: 12 independent sigmoids.

NOT top-3. "Three instruments installed" counts arm slots, and two arms carry
the same class in nearly half of sampled moments -- the window-weighted
distribution over distinct classes peaks at 3 with only 44.6% share, so a hard
three-class constraint would be wrong close to half the time.

Thresholds are tuned on validation after training and frozen into the
checkpoint. The corpus is 90x imbalanced (cadiere forceps 15,600 windows,
stapler 137), so pos_weight is mandatory: without it the rare classes are
never predicted and macro-F1 collapses while accuracy looks fine.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.dataset import ShardFrames, shard_paths_for_split   # noqa: E402
from surgvu.metrics import macro_f1, per_class_f1, tune_thresholds  # noqa: E402
from surgvu.models import build_model                            # noqa: E402
from surgvu.taxonomy import TOOL_CLASSES                         # noqa: E402
from surgvu.train import (                                       # noqa: E402
    run_epoch, save_checkpoint, seed_everything,
)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", default="/staging/groups/bhaskar_opscribe/surgvu/shards")
    parser.add_argument("--splits", default="config/splits.json")
    parser.add_argument("--frequency", default="config/tool_frequency.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--backbone", default="efficientnet_v2_s")
    parser.add_argument("--pos-weight-ceiling", type=float, default=50.0,
                        help="cap on per-class negatives/positives weight. "
                             "50 is the shipped value and clips tip-up (~175) "
                             "and stapler (~200) to the same number; v2 "
                             "experiment 5 raises it.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--frames-per-window", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-shards", type=int, default=0,
                        help="smoke-test knob: use only the first N shards of "
                             "each split (0 = all). A run capped this way is "
                             "for wiring only -- its macro-F1 means nothing.")
    parser.add_argument("--deterministic", action="store_true",
                        help="pin cuDNN's algorithm choice (see "
                             "surgvu.train.seed_everything). Two runs at this "
                             "config and seed differed by ~0.012 without it. "
                             "Costs throughput; OFF by default because the "
                             "shipped checkpoints were trained without it.")
    return parser


def main():
    args = build_parser().parse_args()

    seed_everything(args.seed, deterministic=args.deterministic)
    print("cudnn.deterministic:", args.deterministic,
          "(the shipped checkpoints were trained with False)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    train_shards = shard_paths_for_split(args.shards, args.splits, "train")
    val_shards = shard_paths_for_split(args.shards, args.splits, "val")
    if args.max_shards:
        train_shards = train_shards[:args.max_shards]
        val_shards = val_shards[:args.max_shards]
        print("SMOKE TEST: capped at %d shards per split. The macro-F1 this "
              "run reports is NOT a result." % args.max_shards)
    print("shards: %d train / %d val" % (len(train_shards), len(val_shards)))

    train_set = ShardFrames(train_shards, args.frames_per_window, args.seed, True)
    val_set = ShardFrames(val_shards, args.frames_per_window, args.seed, False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4)

    # pos_weight ~ negatives/positives per class. The exact denominator is the
    # number of windows the frequency table was counted over -- 27,556
    # ENUMERATED train windows, which is what scripts/build_tool_frequency.py
    # sees before stratified selection thins them. The table does not carry
    # that total, so the most common class's count (cadiere forceps, 15,600)
    # stands in as a lower bound.
    #
    # 24,578 is NOT that number, and is not the train split either: it is the
    # whole extracted corpus, all 155 cases. The extracted train split is
    # 19,498 windows under config/splits.json and 18,412 under
    # config/splits_v2.json, counted from the shard headers.
    #
    # The stand-in understates the weight for mid-frequency classes, and by
    # more than "lower bound" suggests -- shipped vs recomputed against the
    # real splits_v2 train counts: grasping retractor 2.9 vs 5.0, prograsp
    # 4.0 vs 6.1, clip applier 18.0 vs 22.2. It matters less than that looks
    # because the clip below is doing most of the work at the end that
    # actually hurts: stapler (153.7) and tip-up (205.9) both clip to 50 under
    # any denominator, and validation-tuned thresholds absorb the residual
    # calibration error on the rest.
    frequency = json.loads(Path(args.frequency).read_text(encoding="utf-8"))
    total = max(frequency.values())
    weights = np.array([max(total - frequency[c], 1) / max(frequency[c], 1)
                        for c in TOOL_CLASSES], dtype=np.float32)
    # The ceiling is a TUNABLE, not a constant, because it is the one thing
    # standing between the rare tail and a usable weight. Under the shipped
    # 50, tip-up fenestrated grasper (true ratio ~175) and stapler (~200) both
    # clip to the same number despite differing in rarity, and tip-up scores
    # exactly 0.0000 in the shipped checkpoint -- it is never predicted at all.
    # Raising it is v2 experiment 5. Default unchanged so every existing
    # command reproduces the shipped behaviour.
    weights = np.clip(weights, 1.0, args.pos_weight_ceiling)
    print("pos_weight:", dict(zip(TOOL_CLASSES, weights.round(1).tolist())))

    model = build_model(len(TOOL_CLASSES), args.backbone, pretrained=True).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = -1.0
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        take_tools = lambda tools, task: tools
        train_stats = run_epoch(model, train_loader, loss_fn, take_tools,
                                optimizer, device, args.image_size)
        val_stats = run_epoch(model, val_loader, loss_fn, take_tools, None,
                              device, args.image_size)

        probs = 1.0 / (1.0 + np.exp(-val_stats["probs"]))
        thresholds = tune_thresholds(val_stats["targets"], probs)
        pred = (probs >= thresholds).astype(np.float32)
        score = macro_f1(val_stats["targets"], pred)
        print("epoch %d  train_loss %.4f  val_loss %.4f  val_macroF1 %.4f"
              % (epoch, train_stats["loss"], val_stats["loss"], score), flush=True)

        if score > best:
            best = score
            per_class = per_class_f1(val_stats["targets"], pred)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(args.out, model, {
                "macro_f1": score,
                "per_class_f1": dict(zip(TOOL_CLASSES, per_class.tolist())),
                "thresholds": thresholds.tolist(),
                "classes": list(TOOL_CLASSES),
                "backbone": args.backbone,
                "epochs": epoch + 1,
                "frames_per_window": args.frames_per_window,
                "image_size": args.image_size,
            })
            print("  saved (best so far)", flush=True)

    if best <= 0.0:
        raise SystemExit(
            "macro-F1 never exceeded 0.0 across %d epochs. Something is wrong "
            "with the labels or the loader; do not proceed." % args.epochs)
    print("BEST val macro-F1: %.4f" % best)


if __name__ == "__main__":
    main()
