"""Save a converted model BEFORE any training, so the conversion can be scored alone.

THE QUESTION THIS ANSWERS. The TSM arm inherits weights that are already
fine-tuned on this corpus and scores 0.6277 after one epoch, against the 2D
model's 0.7802. Two completely different stories fit that:

    the shift disrupts the features the classifier was fitted on, and the
    network needs to re-adapt around it

    the shift is nearly harmless, and one epoch of Adam at 1e-4 on an
    already-converged model simply destroyed it

The second is not speculation -- TSM's training loss reached 0.0053 by epoch 1
while its validation loss TRIPLED from 0.2315 to 0.6209, which is what
overfitting a converged model looks like. But the two stories imply opposite
fixes, and nothing measured so far separates them.

Scoring the conversion at initialisation separates them completely. If an
untrained TSM lands near 0.7802, the shift costs almost nothing and the
learning rate is the whole problem. If it lands near 0.50, the shift really
does disrupt the representation and the arm needs adaptation -- gently.

WHY THIS IS NOT JUST `--epochs 0`. train_temporal.py refuses to save a
checkpoint whose macro-F1 never exceeded zero, on purpose, so a zero-epoch run
produces nothing. And it should not be relaxed: that guard is what stops a
crashed run from leaving a checkpoint that looks trained.

The checkpoint written here carries `epochs: 0` and `untrained: True` in its
metadata so nothing downstream can mistake it for a trained arm.

FOR THE RESIDUAL ARM THIS IS NOT AN ABLATION, IT IS THE BASELINE. At alpha=0
the residual model is the 2D model by construction, so its untrained dump is
what the trained arm has to beat -- and it is also the check that the new pool,
the new layout and the evaluator agree with the shipped number. They are not
the same frames the 0.7802 reference used. Both put sample i at (i + 0.5) / 16
of the window -- the sampling DESIGN is identical, not merely similar -- but
the sparse pool quantises to its 1 fps frame grid while a burst centre lands
at continuous time. Computed, not assumed: the displacement is at most
0.438 s against a 1.0 s grid (the table is in docs/V4_REPORT.md).

So the untrained residual dump should land NEAR 0.7802 -- the prediction
recorded before running it is WITHIN ABOUT 0.01, the cuDNN run-to-run noise
floor, because a sub-half-second shift of the sampled moment should barely
move a label that is constant across the whole 30 s window. Materially below
that is a bug to find, not a pool to accept. Whatever it lands on is the arm's
own base, and the gain is measured against that, not against 0.7802.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.taxonomy import TASK_CLASSES, TOOL_CLASSES          # noqa: E402
from surgvu.temporal import (TWO_D_IMAGE_SIZE, build_i3d_model,  # noqa: E402
                             build_residual_model, build_tsm_model)
from surgvu.train import save_checkpoint                        # noqa: E402

# The SHIPPED checkpoint -- see the note in scripts/train_temporal.py. The
# x40 variant used earlier tonight is not what config/perception.json serves.
TOOLS_2D = "/staging/n/nkalthoff/surgvu26/models/tools_resnet50_long.pt"
TASK_2D = "/staging/n/nkalthoff/surgvu26/models/task_resnet50_long.pt"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mechanism", choices=("tsm", "i3d", "residual"),
                        required=True)
    # The task reference (0.9456 accuracy / 0.9581 description) has never been
    # reproduced through this pipeline. The tools reference had not been either
    # until 04:10, and reproducing it is what exposed both the aggregation
    # mismatch and the wrong base checkpoint. The task claim is tonight's
    # headline, so it gets the same treatment rather than the benefit of the
    # doubt.
    parser.add_argument("--head", choices=("tools", "task"), default="tools")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--layout", default=None)
    parser.add_argument("--bursts", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--fold-div", type=int, default=8,
                        help="0 writes the CONTROL: no shift at all, which "
                             "must score exactly what the 2D model scores")
    parser.add_argument("--temporal", type=int, default=3)
    parser.add_argument("--tap", choices=("layer3", "layer4"), default="layer4")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--frames-per-burst", type=int, default=3)
    parser.add_argument("--sequence", action="store_true",
                        help="include the across-burst branch")
    args = parser.parse_args(argv)

    classes = TOOL_CLASSES if args.head == "tools" else TASK_CLASSES
    if args.checkpoint is None:
        args.checkpoint = TOOLS_2D if args.head == "tools" else TASK_2D
    layout = args.layout or {"tsm": "spread", "residual": "bursts"}.get(
        args.mechanism, "contiguous")
    image_size = args.image_size or (
        TWO_D_IMAGE_SIZE if args.mechanism in ("tsm", "residual") else 224)

    if args.mechanism == "residual":
        if args.frames % args.frames_per_burst:
            raise SystemExit("--frames %d is not a whole number of %d-frame "
                             "bursts" % (args.frames, args.frames_per_burst))
        model, meta = build_residual_model(
            len(classes), checkpoint=args.checkpoint, backbone=args.backbone,
            bursts=args.frames // args.frames_per_burst,
            frames_per_burst=args.frames_per_burst, tap=args.tap,
            hidden=args.hidden, expect_classes=classes,
            sequence=args.sequence)
        print("motion branch on %s, alpha=%.1f: at this alpha the model IS "
              "the 2D model, and this dump is the base the trained arm must "
              "beat -- not an ablation of it."
              % (args.tap, float(model.alpha.item())))
    elif args.mechanism == "tsm":
        model, meta = build_tsm_model(len(classes),
                                      checkpoint=args.checkpoint,
                                      backbone=args.backbone,
                                      segments=args.frames,
                                      fold_div=args.fold_div,
                                      expect_classes=classes)
        print("shift in %d residual blocks, fold_div=%d%s"
              % (meta.get("tsm_blocks", 0), args.fold_div,
                 "  <- CONTROL, no shift" if not args.fold_div else ""))
    else:
        model, meta = build_i3d_model(len(classes),
                                      checkpoint=args.checkpoint,
                                      backbone=args.backbone,
                                      temporal=args.temporal,
                                      expect_classes=classes)
        print("inflated with temporal extent %d" % args.temporal)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.out, model, {
        # macro_f1 is 0.0 and says so. Everything downstream prints the
        # checkpoint's self-reported score next to the honest one, and a
        # fabricated number here would read as a real result.
        "macro_f1": 0.0,
        "classes": list(classes),
        "head": args.head,
        "backbone": args.backbone,
        "mechanism": args.mechanism,
        "initialised_from": args.checkpoint,
        "untrained": True,
        "epochs": 0,
        "clip_length": args.frames,
        "frames_per_window": args.frames,
        "layout": layout,
        # Without this the evaluator falls back to the legacy `i * step`
        # sampler, which on a 30-frame window asked for 16 frames and returned
        # frames 0-15. These checkpoints are built to be scored on exactly the
        # 2D model's input, so getting the sampler wrong would measure the
        # opposite of what they exist to measure.
        "spread_sampler": "bin_centres",
        "bursts": args.bursts,
        "clips_per_window": 0,
        "image_size": image_size,
        "temporal": True,
        "fold_div": args.fold_div if args.mechanism == "tsm" else None,
        "i3d_temporal": args.temporal if args.mechanism == "i3d" else None,
        "residual_tap": args.tap if args.mechanism == "residual" else None,
        "residual_hidden": (args.hidden if args.mechanism == "residual"
                            else None),
        "residual_frames_per_burst": (args.frames_per_burst
                                      if args.mechanism == "residual"
                                      else None),
        "residual_sequence": (bool(args.sequence)
                              if args.mechanism == "residual" else None),
        "alpha": (float(model.alpha.detach().cpu().item())
                  if args.mechanism == "residual" else None),
        "normalisation": "unit",
    })
    print("wrote %s (UNTRAINED: the conversion alone, no gradient steps)"
          % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
