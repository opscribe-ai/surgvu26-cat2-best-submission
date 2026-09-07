"""Train the TASK classifier as a 3D CNN over dense clips.

WHY THIS RUNS EVEN THOUGH THE 3D TOOL HEAD WAS A NULL. The tool experiment
came back negative three independent ways -- order worth +0.0001 at the 30 s
scale, 3D 0.031 behind head to head across two architectures, and no fusion
weight beating the 2D model alone. That is a strong result about TOOLS, and
tools are the wrong place to have looked for motion.

"Is a needle driver installed" is a question about APPEARANCE. An instrument
either is or is not in the frame, and a single still answers it; the temporal
axis was never going to add much, which is roughly what three nulls said.

Task labels are different in kind. 'Suturing', 'retraction', 'uterine horn
mobilization' and 'range of motion' name what the instruments are DOING, and
several pairs are close to indistinguishable in any single frame -- a needle
driver holding tissue still and a needle driver passing a needle differ by
motion and by very little else. If temporal modelling pays anywhere in this
project, it pays here. Testing it on tools and stopping would have been
testing the hypothesis in the one place it was least likely to hold.

THE PRIOR IS STILL AGAINST IT, and the confounds carry over unchanged: 18
layers against a ResNet-50, Kinetics-400 against ImageNet, 112px against 384,
and a 2 s burst standing in for a 30 s labelled window. The 2D task head is
already at 0.9456 macro-F1, so there is not much headroom to win -- which
means a null here is cheap and a win would be surprising and worth a lot.

WHAT TO COMPARE AGAINST. Not the number this prints. "BEST val macro-F1" is a
max over epochs with thresholds tuned on the windows it scores; the comparable
figure is the honest clip-level one from scripts/dump_clip_probs.py. That
distinction cost a wrong conclusion once already on the tool head.

DESCRIPTION ACCURACY IS THE REAL TARGET, as in scripts/train_task.py: three of
the eight classes share a modal description, so a confusion between them costs
nothing downstream. Both numbers are printed and both go in the checkpoint.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.dataset import ShardClips, shard_paths_for_split    # noqa: E402
from surgvu.descriptions import description_accuracy, load_corpus  # noqa: E402
from surgvu.metrics import (macro_f1, multiclass_accuracy,      # noqa: E402
                            per_class_f1)
from surgvu.models import (VIDEO_MEAN, VIDEO_STD,               # noqa: E402
                           build_video_model)
from surgvu.taxonomy import TASK_CLASSES                        # noqa: E402
from surgvu.train import (run_clip_epoch, save_checkpoint,      # noqa: E402
                          seed_everything)

DENSE = "/staging/n/nkalthoff/surgvu26/shards_dense"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards", default=DENSE)
    parser.add_argument("--splits", default="config/splits_v2.json")
    parser.add_argument("--descriptions", default="config/descriptions.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--backbone", default="r2plus1d_18",
                        choices=("r2plus1d_18", "r3d_18"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=112)
    # 1e-4, matching train_tools_3d.py. The 2D task head's 3e-4 is an
    # EfficientNet/ResNet number and collapsed swin and convnext to a
    # degenerate constant; there is no reason to expect a 3D ResNet to be
    # more forgiving than a 2D one, and the tool run at 1e-4 trained cleanly.
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-shards", type=int, default=0,
                        help="smoke-test knob: its macro-F1 means nothing")
    parser.add_argument("--no-normalise", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device %s | backbone %s | clip %d frames @ %dpx | lr %g"
          % (device, args.backbone, args.clip_length, args.image_size, args.lr))

    corpus = load_corpus(args.descriptions)

    train_shards = shard_paths_for_split(args.shards, args.splits, "train")
    val_shards = shard_paths_for_split(args.shards, args.splits, "val")
    if args.max_shards:
        train_shards = train_shards[:args.max_shards]
        val_shards = val_shards[:args.max_shards]
        print("SMOKE TEST: capped at %d shards per split. NOT a result."
              % args.max_shards)
    print("shards: %d train, %d val" % (len(train_shards), len(val_shards)))

    # State the frame spacing rather than trusting the path -- pointed at the
    # sparse pool this trains happily on frames a SECOND apart, which is the
    # null hypothesis wearing the experiment's clothes.
    from surgvu.extract import read_shard
    _, probe_meta = read_shard(train_shards[0])
    probe_fps = probe_meta[0].get("fps")
    print("shard fps: %s => frames are %.0f ms apart"
          % (probe_fps, 1000.0 / float(probe_fps or 1)))
    if probe_fps and float(probe_fps) <= 1.0:
        print("WARNING: this is the SPARSE pool. A 3D model over 1 fps frames "
              "sees scene changes, not motion. Point --shards at %s." % DENSE)

    train_set = ShardClips(train_shards, args.clip_length, args.seed, True)
    val_set = ShardClips(val_shards, args.clip_length, args.seed, False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4)

    model = build_video_model(len(TASK_CLASSES), args.backbone,
                              pretrained=True).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    mean = None if args.no_normalise else VIDEO_MEAN
    std = None if args.no_normalise else VIDEO_STD
    take_task = lambda tools, task: task            # noqa: E731

    eye = np.eye(len(TASK_CLASSES), dtype=np.float32)
    best = -1.0
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        train_stats = run_clip_epoch(model, train_loader, loss_fn, take_task,
                                     optimizer, device, args.image_size,
                                     mean, std)
        val_stats = run_clip_epoch(model, val_loader, loss_fn, take_task,
                                   None, device, args.image_size, mean, std)

        logits, truth = val_stats["probs"], val_stats["targets"].astype(np.int64)
        pred = logits.argmax(axis=1)
        accuracy = multiclass_accuracy(truth, logits)
        score = macro_f1(eye[truth], eye[pred])
        described = description_accuracy(truth, pred, corpus)
        print("epoch %d  train_loss %.4f  val_loss %.4f  val_acc %.4f  "
              "macroF1 %.4f  desc_acc %.4f"
              % (epoch, train_stats["loss"], val_stats["loss"], accuracy,
                 score, described), flush=True)

        # Guaranteed by construction: an exact class hit is a description hit.
        # If this fires, description_accuracy is broken and every desc_acc
        # above it is a number to disbelieve.
        if described < accuracy - 1e-9:
            raise SystemExit(
                "description accuracy (%.4f) fell below task accuracy (%.4f), "
                "which is impossible unless description_accuracy is wrong."
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
                # `frames_per_window` in a 2D checkpoint means "frames sampled
                # independently". Here it is the CLIP LENGTH and the two are
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
    print("NOT comparable to the 2D task head's 0.9456: that is a max over "
          "epochs with self-tuned selection. Use the honest clip-level "
          "protocol before concluding anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
