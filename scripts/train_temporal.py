"""Train OUR 2D ResNet-50 with time added -- TSM or I3D inflation.

WHAT THIS ARM IS FOR. The Kinetics 3D nets lost to the 2D model by 0.031 on
tools, and that comparison confounded four things: 18 layers against 50,
YouTube pretraining against ImageNet-plus-surgical-fine-tuning, 112px against
384, and 1.07 s of a 30 s window against all of it. This arm removes the first
three by construction. It IS the 2D model -- same weights, same depth, same
preprocessing -- with a temporal mechanism added, so what it measures is the
temporal mechanism.

TWO MECHANISMS, both in surgvu/temporal.py:

    tsm   a channel shift inside every residual block. No new parameters.
    i3d   2D kernels inflated to 3D, initialised so that on a static clip the
          inflated network computes exactly what the 2D network computes.

THE LOADER FIX SHIPS WITH IT. `ShardClips` yielded ONE clip per window per
epoch while `ShardFrames` yielded thirty frames, so at equal epochs the 3D
models saw ~30x fewer gradient samples per window than the 2D model did.
`ShardTemporal` takes `--clips-per-window`, and on the multi-burst pool those
clips come from different parts of the window rather than from the same
sliver.

PREPROCESSING IS INHERITED AND NOT OVERRIDABLE BY ACCIDENT. These weights were
fitted on [0, 1] RGB with no mean/std normalisation. The Kinetics constants
that `train_tools_3d.py` correctly applies to ITS weights would degrade these
for a reason that reads as a bad architecture, so this trainer refuses to
apply them and says so in the checkpoint.

WHAT TO COMPARE AGAINST. Not the number this prints -- that is a max over
epochs with thresholds tuned on the windows it scores. The comparable figure
is the honest clip-level one, against the 2D model's 0.7802 on tools.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.dataset import ShardTemporal, shard_paths_for_split  # noqa: E402
from surgvu.extract import read_shard                            # noqa: E402
from surgvu.holdout import case_folds, honest_macro_f1           # noqa: E402
from surgvu.metrics import macro_f1, per_class_f1, tune_thresholds  # noqa: E402
from surgvu.descriptions import (description_accuracy,           # noqa: E402
                                 load_corpus)
from surgvu.metrics import multiclass_accuracy                   # noqa: E402
from surgvu.taxonomy import TASK_CLASSES, TOOL_CLASSES           # noqa: E402
from surgvu.temporal import (TWO_D_IMAGE_SIZE, build_i3d_model,  # noqa: E402
                             build_residual_model, build_tsm_model)
from surgvu.train import run_clip_epoch, save_checkpoint, seed_everything  # noqa: E402

MULTI = "/staging/n/nkalthoff/surgvu26/shards_multi"
# THE SHIPPED CHECKPOINTS, which config/perception.json points at and which
# produced the 0.7802 / 0.9456 references every arm is compared against.
#
# Every conversion tonight was built from *_resnet50_x40.pt instead -- a
# variant from an earlier session that is documented nowhere and is NOT what
# ships. The reference dump is named frame_probs_resnetLONG_val.npz for the
# checkpoint it came from, and the name is the only place that was written
# down. A conversion of the wrong base model is still a valid experiment about
# conversions; it is not a valid comparison against a number the other model
# produced.
TOOLS_2D = "/staging/n/nkalthoff/surgvu26/models/tools_resnet50_long.pt"
TASK_2D = "/staging/n/nkalthoff/surgvu26/models/task_resnet50_long.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # THE TASK HEAD IS WHERE MOTION SHOULD PAY, and it has never been tried
    # with tonight's corrected recipe. v3 measured the 3D tool head 0.031
    # behind the 2D one while the 3D TASK head reached parity (0.9353 against
    # 0.9456) -- which is what the hypothesis predicts, since "is a needle
    # driver installed" is an appearance question and "suturing versus
    # retraction" is a motion one. Everything tonight fixed -- frozen
    # BatchNorm, the wider input, the lower rate -- applies unchanged.
    parser.add_argument("--head", choices=("tools", "task"), default="tools")
    # WHICH METRIC PICKS THE CHECKPOINT. macro-F1 for tools, because that is
    # what the tool head is scored on everywhere. For the task head the right
    # answer is DESCRIPTION accuracy: three classes share a modal description,
    # so a confusion inside that group never reaches an answer, and macro-F1
    # counts it as an error anyway. Measured on task_tsm_multi, the two
    # disagree about which epoch is best --
    #     epoch 1  macroF1 0.8234  desc 0.9648   <- macro-F1 picks this
    #     epoch 2  macroF1 0.8012  desc 0.9670   <- description picks this
    # -- by 0.0022, which is small but in the direction that matters. The
    # default follows the head rather than being a global choice.
    parser.add_argument("--select", choices=("macro_f1", "description"),
                        default=None,
                        help="default: macro_f1 for tools, description for task")
    parser.add_argument("--descriptions",
                        default="config/descriptions.yaml")
    parser.add_argument("--mechanism",
                        choices=("tsm", "i3d", "video", "residual"),
                        required=True,
                        help="video = a Kinetics 3D backbone, run through THIS "
                             "loader so it gets burst-aware sampling; "
                             "residual = the 2D model plus a zero-initialised "
                             "motion branch, which starts AT the 2D number "
                             "instead of below it")
    parser.add_argument("--tap", choices=("layer3", "layer4"), default="layer4",
                        help="residual only: which stage the motion branch "
                             "differences")
    parser.add_argument("--hidden", type=int, default=256,
                        help="residual only: motion branch width")
    parser.add_argument("--frames-per-burst", type=int, default=3,
                        help="residual only: frames in one burst of the pool. "
                             "shards_multi16 holds 3.")
    # THE SECOND TIMESCALE. `--tap`/`--hidden` configure the LOCAL branch,
    # which reads motion inside one 0.2 s burst and whose per-burst
    # corrections are then averaged -- an order-invariant operation, so a
    # window and its reverse give the same answer. --sequence adds a branch
    # across the sixteen burst centres, which span the whole 30 s.
    #
    # ITS OWN GATE. beta is a separate zero-initialised scalar, so the two
    # timescales are separately attributable: the trained alpha and beta ARE
    # the ablation, readable off the checkpoint, without running four arms.
    parser.add_argument("--sequence", action="store_true",
                        help="residual only: add the across-burst branch, "
                             "gated by its own beta")
    parser.add_argument("--chunk", type=int, default=48,
                        help="residual only: frames per trunk call. Bounds "
                             "peak GPU memory; 0 runs the whole batch at once.")
    parser.add_argument("--scratch", action="store_true",
                        help="video only: skip Kinetics init. The ablation.")
    parser.add_argument("--shards", default=MULTI)
    parser.add_argument("--splits", default="config/splits_v2.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default=TOOLS_2D,
                        help="the 2D weights to convert; '' trains from "
                             "ImageNet init instead, which is a different "
                             "experiment and is labelled as one")
    parser.add_argument("--backbone", default=None,
                        help="default: resnet50 for tsm/i3d, r2plus1d_18 for "
                             "video")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--layout", choices=("contiguous", "spread"),
                        default=None,
                        help="default: spread for tsm, contiguous for i3d")
    parser.add_argument("--bursts", type=int, default=4,
                        help="bursts per window in the pool; 1 for the "
                             "centre-only dense pool")
    parser.add_argument("--clips-per-window", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=0,
                        help="0 = the 2D model's own 384 for tsm, 224 for i3d")
    # DEFAULT DEPENDS ON WHERE THE WEIGHTS CAME FROM, and this is measured.
    # A conversion starts from a model already fine-tuned 20 epochs on this
    # corpus, so it is not being trained -- it is being nudged around a shift
    # or an inflation. At 1e-4 Adam does not nudge it, it destroys it:
    #
    #     untrained conversion, self-tuned   0.7609
    #     after 1 epoch at 1e-4              0.6277
    #     after 2 epochs at 1e-4             0.4981
    #     after 3 epochs at 1e-4             0.4280   <- val_loss 0.23 -> 1.15
    #
    # while the same arm at 1e-5 opened at 0.6288 with val_loss 0.1987, the
    # lowest of any epoch at either rate. A Kinetics backbone is the opposite
    # case -- it has never seen surgery and genuinely needs to train -- so it
    # keeps 1e-4.
    #
    # The I3D evidence is weaker than the TSM evidence and is labelled as such:
    # it oscillated at 1e-4 (0.5139, 0.4693, 0.5523, 0.6403, 0.4387) rather
    # than collapsing monotonically. It gets the lower rate for consistency,
    # since it starts from the same converged weights, and the dense-pool arm
    # at 1e-4 is retained as the comparison.
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fold-div", type=int, default=8,
                        help="TSM shift fraction; 0 disables the shift and "
                             "builds the 2D control")
    parser.add_argument("--temporal", type=int, default=3,
                        help="I3D temporal kernel extent")
    # ON BY DEFAULT WHEN CONVERTING. The full trajectories, not just the
    # first epoch -- the first epoch alone told a stronger story than the data
    # supports, and the correction is the interesting part:
    #
    #     untrained conversion, self-tuned   0.7609
    #     BN frozen,  lr 1e-5   epoch 0      0.7530
    #     BN free,    lr 1e-5   epochs 0-3   0.6288  0.5199  0.4950  0.7100
    #     BN free,    lr 1e-4   epochs 0-2   0.6277  0.4981  0.4280   (killed)
    #
    # So unfrozen BatchNorm at 1e-5 does NOT destroy the model. It forces a
    # three-epoch DETOUR: the running statistics move away from what the
    # classifier was fitted against, the classifier chases them, and by epoch
    # 3 it has caught up -- reaching 0.7100 with the lowest validation loss of
    # that run. Freezing skips the detour and arrives higher in ONE epoch.
    #
    # At 1e-4 the two effects compound and it really is destructive: three
    # epochs of monotonic collapse with validation loss going 0.23 -> 1.15.
    #
    # Freezing is therefore the right default on a budget of 8-10 epochs --
    # three of them is a third of the run -- rather than because the
    # alternative is fatal. A Kinetics backbone is the opposite case: its
    # statistics describe YouTube and MUST move, so the default follows the
    # initialisation rather than being a global switch.
    parser.add_argument("--freeze-bn", dest="freeze_bn", action="store_true",
                        default=None,
                        help="keep BatchNorm in eval mode while training")
    parser.add_argument("--no-freeze-bn", dest="freeze_bn",
                        action="store_false",
                        help="let BatchNorm update -- the measured wrong "
                             "choice for a conversion, kept for the control")
    parser.add_argument("--max-shards", type=int, default=0)
    args = parser.parse_args()

    backbone = args.backbone or ("r2plus1d_18" if args.mechanism == "video"
                                 else "resnet50")
    converting = args.mechanism in ("tsm", "i3d") and bool(args.checkpoint)
    if args.mechanism == "residual" and args.clips_per_window != 1:
        # The bursts layout emits the WHOLE window as one clip, so there is no
        # second clip to draw -- and leaving the default at 2 would divide the
        # learning rate by a clip count the loader never produced.
        print("residual: --clips-per-window forced to 1, the bursts layout "
              "already emits the whole window as one clip")
        args.clips_per_window = 1
    lr = args.lr if args.lr is not None else (1e-5 if converting else 1e-4)
    if args.lr is None and not converting and args.clips_per_window > 1:
        # SCALE THE RATE BY THE STEPS PER EPOCH. The multi-clip loader was
        # added to fix a 30x gradient-sample deficit, and it does -- but it
        # also multiplies the steps in an epoch by clips_per_window, so a
        # schedule tuned at one clip is being run several times faster than it
        # was measured at. Both Kinetics arms diverged at epoch 4 with four
        # clips per window at 1e-4:
        #
        #     r2plus1d_multi    0.5954  0.4483  0.1971   val_loss 0.23 -> 1.24
        #     r2plus1d_scratch  0.4986  0.1474            val_loss 0.24 -> 0.77
        #
        # while v3's one-clip run at the same rate trained to 0.7739 over 20
        # epochs. The setup changed underneath the learning rate; dividing by
        # the clip count restores the per-epoch step budget it was chosen for.
        lr = lr / float(args.clips_per_window)
    freeze_bn = args.freeze_bn if args.freeze_bn is not None else converting
    if args.mechanism == "residual" and freeze_bn:
        # Not wrong, just redundant and confusing in the log: ResidualTemporal
        # keeps its whole trunk in eval mode from inside train(), so there are
        # no running statistics left for this to pin.
        print("residual: --freeze-bn is redundant, the trunk never leaves "
              "eval mode")
    layout = args.layout or {"tsm": "spread", "residual": "bursts"}.get(
        args.mechanism, "contiguous")
    image_size = args.image_size or {"tsm": TWO_D_IMAGE_SIZE, "i3d": 224,
                                     "video": 112,
                                     # The 2D model's own resolution, and NOT
                                     # negotiable here the way it is for a
                                     # conversion: the residual arm's whole
                                     # claim is that its 2D path reproduces the
                                     # shipped model, and the shipped model was
                                     # fitted at 384.
                                     "residual": TWO_D_IMAGE_SIZE,
                                     }[args.mechanism]
    if args.mechanism == "residual":
        if layout != "bursts":
            raise SystemExit(
                "the residual arm splits its clip at burst boundaries, so it "
                "needs the 'bursts' layout; %r would hand it frames whose "
                "boundaries are somewhere else." % layout)
        if args.frames % args.frames_per_burst:
            raise SystemExit(
                "--frames %d is not a multiple of --frames-per-burst %d"
                % (args.frames, args.frames_per_burst))

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device %s | %s over %s | %d frames @ %dpx | %s layout | %d "
          "clips/window | lr %g"
          % (device, args.mechanism, backbone, args.frames, image_size,
             layout, args.clips_per_window, lr))
    if args.lr is None:
        print("lr %g chosen by default: %s" % (
            lr,
            "converting weights already fine-tuned on this corpus, which "
            "1e-4 was measured to destroy" if converting
            else "training a randomly-initialised branch beside a frozen "
                 "model -- the collapse that forced 1e-5 on the conversions "
                 "cannot happen here, because nothing in the 2D path moves"
            if args.mechanism == "residual"
            else "training a backbone that has never seen surgery"))

    classes = TOOL_CLASSES if args.head == "tools" else TASK_CLASSES
    if args.head == "task" and args.checkpoint == TOOLS_2D:
        # The tools checkpoint has 12 outputs and the task head needs 8, so
        # loading it here would fail on the classifier shape -- loudly, which
        # is fine, but the fix is to point at the TASK checkpoint rather than
        # to relax the load. Named explicitly so the failure never becomes a
        # silent strict=False.
        args.checkpoint = TASK_2D
    checkpoint = args.checkpoint or None
    mean = std = None
    if args.mechanism == "video":
        # A Kinetics backbone, but run through ShardTemporal rather than
        # ShardClips. That is the whole point of this arm: on the multi-burst
        # pool a contiguous run drawn from anywhere in the 32 frames can
        # STRADDLE two bursts, splicing a 7.5-second jump cut into the middle
        # of what the model is told is continuous motion. ShardTemporal draws
        # inside one burst.
        from surgvu.models import VIDEO_MEAN, VIDEO_STD, build_video_model
        model = build_video_model(len(classes), backbone,
                                  pretrained=not args.scratch)
        meta = {"pretrained": "kinetics400" if not args.scratch else "none"}
        mean, std = VIDEO_MEAN, VIDEO_STD
        checkpoint = None
    elif args.mechanism == "residual":
        model, meta = build_residual_model(
            len(classes), checkpoint=checkpoint, backbone=backbone,
            bursts=args.frames // args.frames_per_burst,
            frames_per_burst=args.frames_per_burst, tap=args.tap,
            hidden=args.hidden, expect_classes=classes, chunk=args.chunk,
            sequence=args.sequence)
    elif args.mechanism == "tsm":
        model, meta = build_tsm_model(len(classes), checkpoint=checkpoint,
                                      backbone=backbone,
                                      segments=args.frames,
                                      fold_div=args.fold_div,
                                      expect_classes=classes)
    else:
        model, meta = build_i3d_model(len(classes), checkpoint=checkpoint,
                                      backbone=backbone,
                                      temporal=args.temporal,
                                      expect_classes=classes)
    model = model.to(device)
    if args.mechanism == "video":
        print("initialised from %s | Kinetics normalisation"
              % meta["pretrained"])
    else:
        print("initialised from %s | unit normalisation, inherited from the "
              "2D weights" % (checkpoint or "ImageNet (NOT our surgical "
                              "weights)"))
    if args.mechanism == "residual":
        print("motion branch on %s: %d trainable parameters against %d frozen. "
              "alpha%s starts at 0, so this model IS the 2D model until "
              "training moves it."
              % (args.tap, meta["residual_trainable_params"],
                 sum(p.numel() for p in model.parameters()) -
                 meta["residual_trainable_params"],
                 " and beta" if args.sequence else ""), flush=True)
        print("timescales: local 0.2s within a burst%s"
              % (" + sequence 30s across %d burst centres"
                 % (args.frames // args.frames_per_burst)
                 if args.sequence else " ONLY (order-invariant after the mean)"),
              flush=True)
    if args.mechanism == "tsm":
        print("shift installed in %d residual blocks, fold_div=%d%s"
              % (meta.get("tsm_blocks", 0), args.fold_div,
                 "  <- CONTROL: no shift, this is the 2D model"
                 if not args.fold_div else ""))

    train_shards = shard_paths_for_split(args.shards, args.splits, "train")
    val_shards = shard_paths_for_split(args.shards, args.splits, "val")
    if args.max_shards:
        train_shards = train_shards[:args.max_shards]
        val_shards = val_shards[:args.max_shards]
        print("SMOKE TEST: %d shards per split. NOT a result." % args.max_shards)
    print("shards: %d train, %d val" % (len(train_shards), len(val_shards)))

    _, probe = read_shard(train_shards[0])
    depth = int(probe[0].get("frames") or 0) or None
    print("pool: fps=%s window length=%.1fs bursts assumed=%d"
          % (probe[0].get("fps"), probe[0].get("length", 0.0), args.bursts))
    if args.bursts > 1 and float(probe[0].get("length", 0)) < 5.0:
        print("WARNING: rows are %.1fs long, which is the CENTRE-ONLY dense "
              "pool. --bursts %d would slice one burst into %d and call the "
              "pieces spread. Pass --bursts 1."
              % (float(probe[0].get("length", 0)), args.bursts, args.bursts))

    train_set = ShardTemporal(train_shards, frames=args.frames, layout=layout,
                              bursts=args.bursts,
                              clips_per_window=args.clips_per_window,
                              seed=args.seed, shuffle=True)
    val_set = ShardTemporal(val_shards, frames=args.frames, layout=layout,
                            bursts=args.bursts, clips_per_window=1,
                            seed=args.seed, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4)

    def freeze_batchnorm(module):
        """Stop BatchNorm from updating its running statistics.

        WHY THIS EXISTS. Dropping the learning rate from 1e-4 to 1e-5 changed
        almost nothing about the first epoch -- 0.6277 against 0.6288, from an
        untrained 0.7609 -- and a tenfold change in step size that does not
        move the result is evidence that the step size is not the mechanism.

        BatchNorm is the obvious candidate because it updates in train mode
        REGARDLESS of learning rate. Every forward pass overwrites the running
        mean and variance the 2D classifier was fitted against, using batches
        of 16 images of freshly-shifted features. The classifier then receives
        activations normalised by statistics it has never seen, which is a
        corruption no optimiser setting can slow down.

        Freezing keeps the affine weights trainable -- they are parameters and
        the gradient should reach them -- and only pins the buffers.
        """
        frozen = 0
        for child in module.modules():
            if isinstance(child, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                  torch.nn.BatchNorm3d)):
                child.eval()
                frozen += 1
        return frozen

    if freeze_bn:
        # OVERRIDE train(), rather than freezing once before the loop.
        # run_clip_epoch calls model.train(training) on every epoch, which puts
        # BatchNorm straight back into training mode -- so a freeze applied
        # from out here would be silently undone on the first line of the first
        # epoch, and the run would look exactly like an unfrozen one. Binding
        # the invariant to train() itself makes it impossible to revert from
        # the outside, without touching the shared epoch function.
        base_train = model.train

        def train_with_frozen_bn(mode=True):
            base_train(mode)
            if mode:
                freeze_batchnorm(model)
            return model

        model.train = train_with_frozen_bn
        print("froze %d BatchNorm layers: running statistics stay as the 2D "
              "model fitted them" % freeze_batchnorm(model), flush=True)

    select = args.select or ("macro_f1" if args.head == "tools"
                             else "description")
    if args.head == "tools":
        loss_fn = torch.nn.BCEWithLogitsLoss()
        take_target = lambda tools, task: tools                  # noqa: E731
        corpus = None
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        take_target = lambda tools, task: task                   # noqa: E731
        corpus = load_corpus(args.descriptions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best = -1.0
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        # mean/std are None on purpose: see the module docstring. These weights
        # were fitted on [0, 1] and Kinetics statistics would corrupt them.
        train_stats = run_clip_epoch(model, train_loader, loss_fn, take_target,
                                     optimizer, device, image_size, mean, std)
        val_stats = run_clip_epoch(model, val_loader, loss_fn, take_target,
                                   None, device, image_size, mean, std)

        probs, truth = val_stats["probs"], val_stats["targets"]
        extra = {}
        if args.head == "tools":
            cuts = tune_thresholds(truth, probs)
            score = macro_f1(truth, (probs >= cuts).astype(np.float32))
            print("epoch %d  train_loss %.4f  val_loss %.4f  val_macroF1 %.4f"
                  % (epoch, train_stats["loss"], val_stats["loss"], score),
                  flush=True)
        else:
            truth = truth.astype(np.int64)
            pred = probs.argmax(axis=1)
            eye = np.eye(len(classes), dtype=np.float32)
            accuracy = multiclass_accuracy(truth, probs)
            described = description_accuracy(truth, pred, corpus)
            # macro-F1 is the selection metric for BOTH heads so the "best so
            # far" rule means the same thing across arms, but accuracy is what
            # the 2D task head's 0.9456 is, and description accuracy is what
            # actually reaches an answer -- so all three are printed and all
            # three are recorded.
            score = macro_f1(eye[truth], eye[pred])
            cuts = None
            extra = {"accuracy": accuracy, "description_accuracy": described}
            print("epoch %d  train_loss %.4f  val_loss %.4f  val_acc %.4f  "
                  "macroF1 %.4f  desc_acc %.4f"
                  % (epoch, train_stats["loss"], val_stats["loss"], accuracy,
                     score, described), flush=True)
            if described < accuracy - 1e-9:
                raise SystemExit(
                    "description accuracy (%.4f) below task accuracy (%.4f), "
                    "which is impossible unless description_accuracy is broken."
                    % (described, accuracy))

        # SELECTION VALUE AND REPORTED SCORE ARE SEPARATE. Overwriting `score`
        # with the description accuracy -- which is what the first version of
        # this did -- also overwrote what the epoch line PRINTS as macroF1 and
        # what the checkpoint stores under "macro_f1". The relaunched task arm
        # duly reported "macroF1 0.9657 desc_acc 0.9657" for an epoch whose
        # real macro-F1 was 0.8092. Selecting on one metric must not relabel
        # another.
        selection = (extra.get("description_accuracy", score)
                     if select == "description" else score)
        if selection > best:
            best = selection
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(args.out, model, {
                "macro_f1": score,
                "selection_value": selection,
                "thresholds": cuts.tolist() if cuts is not None else None,
                "classes": list(classes),
                "head": args.head,
                "selected_by": select,
                "backbone": backbone,
                "mechanism": args.mechanism,
                "initialised_from": (meta.get("pretrained")
                                     if args.mechanism == "video"
                                     else (checkpoint or "imagenet")),
                "epochs": epoch + 1,
                "clip_length": args.frames,
                "frames_per_window": args.frames,
                "layout": layout,
                # Which spread sampler this arm trained with, so its evaluator
                # can reproduce the frames it actually saw.
                "spread_sampler": "bin_centres",
                "bursts": args.bursts,
                "clips_per_window": args.clips_per_window,
                "image_size": image_size,
                "temporal": True,
                "fold_div": args.fold_div if args.mechanism == "tsm" else None,
                # Everything dump_temporal_probs needs to rebuild this arm.
                # Written unconditionally as None for other mechanisms so a
                # dump can tell "not a residual arm" from "an older residual
                # arm that predates the field".
                "residual_tap": args.tap if args.mechanism == "residual" else None,
                "residual_hidden": (args.hidden if args.mechanism == "residual"
                                    else None),
                "residual_frames_per_burst": (args.frames_per_burst
                                              if args.mechanism == "residual"
                                              else None),
                "alpha": (float(model.alpha.detach().cpu().item())
                          if args.mechanism == "residual" else None),
                "residual_sequence": (bool(args.sequence)
                                      if args.mechanism == "residual" else None),
                # The attribution, recorded per saved epoch. A beta that never
                # leaves zero says the 30 s branch earned nothing, which is a
                # result rather than a missing one.
                "beta": (float(model.beta.detach().cpu().item())
                         if args.mechanism == "residual" and args.sequence
                         else None),
                "i3d_temporal": args.temporal if args.mechanism == "i3d" else None,
                "lr": lr,
                "frozen_bn": bool(freeze_bn),
                "normalisation": ("kinetics" if args.mechanism == "video"
                                  else "unit"),
                "shard_fps": probe[0].get("fps"),
                "shard_length": probe[0].get("length"),
                **extra,
            })
            print("  saved (best so far, by %s)" % select, flush=True)

    if best <= 0.0:
        raise SystemExit("macro-F1 never exceeded 0.0 -- nothing usable.")
    print("BEST val macro-F1: %.4f" % best)
    print("NOT comparable to the 2D model's 0.7802: that is a max over epochs "
          "with self-tuned cuts. Run scripts/dump_clip_probs.py for the "
          "honest number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
