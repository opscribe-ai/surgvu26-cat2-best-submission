"""Train the 8-way task classifier and measure DESCRIPTION accuracy too.

Task accuracy is not the number that matters. The task class exists to
retrieve a `matched_description`, which is verbatim the text the challenge's
ground-truth answers were generated from -- and three of the eight classes
('other', 'retraction and collision avoidance', 'suturing') share the same
modal description, so a confusion between them costs nothing downstream.
Reporting only task accuracy understates the model exactly where it is
already good enough. Both numbers are printed every epoch and both go into
the checkpoint; `description_accuracy` is the one to optimise against.

Selection is still on macro-F1 rather than either accuracy: the task
distribution is long-tailed, and accuracy would happily pick the epoch that
learned the head classes and abandoned the tail.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.dataset import ShardFrames, shard_paths_for_split   # noqa: E402
from surgvu.descriptions import description_accuracy, load_corpus  # noqa: E402
from surgvu.metrics import (                                    # noqa: E402
    macro_f1, multiclass_accuracy, per_class_f1,
)
from surgvu.models import build_model                           # noqa: E402
from surgvu.taxonomy import TASK_CLASSES                        # noqa: E402
from surgvu.train import (                                      # noqa: E402
    run_epoch, save_checkpoint, seed_everything,
)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", default="/staging/groups/bhaskar_opscribe/surgvu/shards")
    parser.add_argument("--splits", default="config/splits.json")
    parser.add_argument("--descriptions", default="config/descriptions.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--backbone", default="efficientnet_v2_s")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--frames-per-window", type=int, default=8)
    # Must match scripts/train_tools.py, and must be whatever Task 8's
    # predict_window is served at. EfficientNetV2-S was pretrained at 384 and
    # the shards hold 512x512, so the resize is happening either way; what
    # matters is that it happens identically in both experts and at serving
    # time. predict_window takes ONE image_size for the whole ensemble, so a
    # tool recogniser trained at 384 and a task classifier trained at 512
    # guarantees one of the two is served at the wrong resolution -- and that
    # loss is silent, showing up as a mediocre model rather than an error.
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

    corpus = load_corpus(args.descriptions)

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
    # No persistent_workers: ShardFrames.set_epoch is silently inert under
    # persistent workers, freezing the shard order and frame subsample at
    # epoch 0 for the whole run. See the ShardFrames docstring.
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4)

    model = build_model(len(TASK_CLASSES), args.backbone, pretrained=True).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    eye = np.eye(len(TASK_CLASSES), dtype=np.float32)
    best = -1.0
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        take_task = lambda tools, task: task    # noqa: E731
        train_stats = run_epoch(model, train_loader, loss_fn, take_task,
                                optimizer, device, args.image_size)
        val_stats = run_epoch(model, val_loader, loss_fn, take_task, None,
                              device, args.image_size)

        logits, truth = val_stats["probs"], val_stats["targets"]
        pred = logits.argmax(axis=1)
        accuracy = multiclass_accuracy(truth, logits)
        # per_class_f1/macro_f1 are column-wise over a multi-hot matrix, so
        # the single-label case goes through them as one-hot rows.
        score = macro_f1(eye[truth], eye[pred])
        described = description_accuracy(truth, pred, corpus)
        print("epoch %d  train_loss %.4f  val_loss %.4f  val_acc %.4f  "
              "macroF1 %.4f  desc_acc %.4f"
              % (epoch, train_stats["loss"], val_stats["loss"], accuracy,
                 score, described), flush=True)

        # Guaranteed by construction: an exact class hit is a description hit.
        # If this ever fires, description_accuracy is broken, and every
        # desc_acc printed above it is a number to disbelieve.
        if described < accuracy - 1e-9:
            raise SystemExit(
                "description accuracy (%.4f) fell below task accuracy (%.4f), "
                "which is impossible unless description_accuracy is wrong. "
                "Fix it before trusting any desc_acc in this log."
                % (described, accuracy))

        if score > best:
            best = score
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(args.out, model, {
                "accuracy": accuracy,
                "macro_f1": score,
                "per_class_f1": dict(zip(
                    TASK_CLASSES, per_class_f1(eye[truth], eye[pred]).tolist())),
                "description_accuracy": described,
                "classes": list(TASK_CLASSES),
                "backbone": args.backbone,
                "epochs": epoch + 1,
                "frames_per_window": args.frames_per_window,
                "image_size": args.image_size,
            })
            print("  saved (best so far)", flush=True)

    if best <= 0.0:
        raise SystemExit(
            "task macro-F1 never exceeded 0.0 across %d epochs, so no "
            "checkpoint was ever written. Every class was missed on every "
            "epoch -- suspect the task labels or the loader, not the "
            "learning rate; do not proceed to description retrieval."
            % args.epochs)
    print("BEST val macro-F1: %.4f" % best)


if __name__ == "__main__":
    main()
