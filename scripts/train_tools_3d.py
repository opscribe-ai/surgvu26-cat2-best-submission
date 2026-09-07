"""Train the tool expert as a 3D CNN over dense clips.

THE QUESTION. Every model this project has trained treats the frames of a clip
as unordered images and averages their predictions -- shuffle them and the
answer is bit-identical. That discards motion entirely, and several of the
question types we are graded on are ABOUT motion: a scissors resting on tissue
and a scissors closing on it are near-identical in any single still.

This is the first model that can see the difference. It reads a contiguous run
of frames in order from the DENSE shards (2 s at 15 fps, 67 ms apart) and
convolves across time as well as space.

READ THE CONFOUNDS BEFORE READING THE RESULT. A loss here does NOT
straightforwardly mean "motion does not help":

  * Kinetics-400 pretraining is human action video -- far smaller than
    ImageNet and much further from surgery -- and these are 18-layer networks
    against the ResNet-50 that produced 0.7802.
  * They expect 112x112 where the 2D path runs 384, and instrument tips are
    small objects.

So a fair reading needs the sequence-model control over existing 2D features,
which separates "temporal structure carries signal" from "this backbone is any
good". Interpreting this run alone would be the same mistake as reading
EndoViT's 0.6627 as "domain pretraining does not work".

The comparison target is the honest CLIP-level two-fold number, not the
per-frame max-over-epochs printed here -- that statistic is biased upward by
the epoch count and is not comparable across runs of different length.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.dataset import ShardClips, shard_paths_for_split   # noqa: E402
from surgvu.metrics import macro_f1, per_class_f1, tune_thresholds  # noqa: E402
from surgvu.models import (VIDEO_MEAN, VIDEO_STD,               # noqa: E402
                           build_video_model)
from surgvu.taxonomy import TOOL_CLASSES                        # noqa: E402
from surgvu.train import run_clip_epoch, save_checkpoint, seed_everything  # noqa: E402

DENSE = "/staging/n/nkalthoff/surgvu26/shards_dense"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards", default=DENSE)
    parser.add_argument("--splits", default="config/splits_v2.json")
    parser.add_argument("--frequency", default="config/tool_frequency.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--backbone", default="r2plus1d_18",
                        choices=("r2plus1d_18", "r3d_18"))
    parser.add_argument("--pos-weight-ceiling", type=float, default=50.0)
    parser.add_argument("--epochs", type=int, default=20)
    # A clip is 16x the pixels of a frame, so the batch that fits is far
    # smaller than the 2D path's 48.
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--clip-length", type=int, default=16,
                        help="contiguous frames per clip; 16 is what the "
                             "Kinetics weights were trained on")
    parser.add_argument("--image-size", type=int, default=112,
                        help="112 is these models' pretraining resolution. "
                             "Raising it keeps more instrument detail and "
                             "costs memory quadratically; it is a knob "
                             "because the right value is not obvious.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="1e-4, not the 2D default 3e-4: 3e-4 collapsed "
                             "swin and convnext to a degenerate constant "
                             "prediction (exactly 0.2785 for both). That "
                             "default is an EfficientNet/ResNet number.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-shards", type=int, default=0,
                        help="smoke-test knob: its macro-F1 means nothing")
    parser.add_argument("--no-normalise", action="store_true",
                        help="feed [0,1] like the 2D path instead of the "
                             "Kinetics statistics these weights expect")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device %s | backbone %s | clip %d frames @ %dpx | lr %g"
          % (device, args.backbone, args.clip_length, args.image_size, args.lr))

    train_shards = shard_paths_for_split(args.shards, args.splits, "train")
    val_shards = shard_paths_for_split(args.shards, args.splits, "val")
    if args.max_shards:
        train_shards = train_shards[:args.max_shards]
        val_shards = val_shards[:args.max_shards]
        print("SMOKE TEST: capped at %d shards per split. The macro-F1 this "
              "run reports is NOT a result." % args.max_shards)
    print("shards: %d train, %d val" % (len(train_shards), len(val_shards)))

    # State the frame spacing rather than trusting the path. Pointed at the
    # sparse pool this trains perfectly happily on frames a SECOND apart,
    # which is the null hypothesis wearing the experiment's clothes.
    from surgvu.extract import read_shard
    _, probe_meta = read_shard(train_shards[0])
    probe_fps = probe_meta[0].get("fps")
    print("shard fps: %s  => frames are %.0f ms apart"
          % (probe_fps, 1000.0 / float(probe_fps or 1)))
    if probe_fps and float(probe_fps) <= 1.0:
        print("WARNING: this is the SPARSE pool. A 3D model over 1 fps frames "
              "sees scene changes, not motion. Point --shards at %s." % DENSE)

    train_set = ShardClips(train_shards, args.clip_length, args.seed, True)
    val_set = ShardClips(val_shards, args.clip_length, args.seed, False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4)

    # Identical pos_weight derivation to scripts/train_tools.py, including its
    # documented stand-in denominator, so a 3D-vs-2D comparison is not also a
    # comparison of class weighting.
    frequency = json.loads(Path(args.frequency).read_text(encoding="utf-8"))
    total = max(frequency.values())
    weights = np.array([max(total - frequency[c], 1) / max(frequency[c], 1)
                        for c in TOOL_CLASSES], dtype=np.float32)
    weights = np.clip(weights, 1.0, args.pos_weight_ceiling)

    model = build_video_model(len(TOOL_CLASSES), args.backbone,
                              pretrained=True).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    mean = None if args.no_normalise else VIDEO_MEAN
    std = None if args.no_normalise else VIDEO_STD
    take_tools = lambda tools, task: tools

    best = -1.0
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        train_stats = run_clip_epoch(model, train_loader, loss_fn, take_tools,
                                     optimizer, device, args.image_size,
                                     mean, std)
        val_stats = run_clip_epoch(model, val_loader, loss_fn, take_tools,
                                   None, device, args.image_size, mean, std)

        probs = 1.0 / (1.0 + np.exp(-val_stats["probs"]))
        thresholds = tune_thresholds(val_stats["targets"], probs)
        pred = (probs >= thresholds).astype(np.float32)
        score = macro_f1(val_stats["targets"], pred)
        print("epoch %d  train_loss %.4f  val_loss %.4f  val_macroF1 %.4f"
              % (epoch, train_stats["loss"], val_stats["loss"], score),
              flush=True)

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
                # `frames_per_window` in a 2D checkpoint means "frames sampled
                # independently". Here it is the CLIP LENGTH, and the two are
                # not interchangeable -- recorded under its own name so no
                # serving path can mistake one for the other.
                "clip_length": args.clip_length,
                "frames_per_window": args.clip_length,
                "image_size": args.image_size,
                "temporal": True,
                "shard_fps": probe_fps,
                "normalisation": "kinetics" if mean else "unit",
            })
            print("  saved (best so far)", flush=True)

    if best <= 0.0:
        raise SystemExit(
            "macro-F1 never exceeded 0.0 across %d epochs -- the run produced "
            "nothing usable." % args.epochs)
    print("BEST val macro-F1: %.4f" % best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
