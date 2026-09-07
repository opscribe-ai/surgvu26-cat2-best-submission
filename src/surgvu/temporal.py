"""Turning OUR 2D ResNet-50 into a temporal model, three ways.

WHY THIS EXISTS. Every 3D result so far used torchvision's Kinetics-400
video backbones -- 18 layers, pretrained on YouTube human actions, run at
112px. Our 2D ResNet-50 is 50 layers, ImageNet-pretrained AND fine-tuned for
20 epochs on this surgical corpus at 384px, and it scores 0.7802 on tools
where the 3D nets score 0.7494. Comparing them tests four things at once:
depth, pretraining domain, resolution, and temporal modelling. Only the last
one is the hypothesis.

Both constructions here keep the surgical weights and add time, so the
comparison against the 2D model isolates the temporal axis:

  TSM      shifts a fraction of channels along the time axis inside each
           residual block. ZERO new parameters -- it is a memory copy -- so
           the network is byte-identical to the 2D one apart from where the
           features come from. Cheap enough to run at the 2D model's own
           resolution.

  I3D      inflates 2D kernels into 3D by repeating them along time and
           dividing by the temporal extent, so the inflated network starts out
           computing exactly what the 2D one computes on a static clip and
           learns motion from there. Genuinely 3D convolution, at genuinely 3D
           memory cost, which is why it runs at lower resolution.

  RESIDUAL leaves the 2D network alone entirely and adds a small motion branch
           beside it, gated by a scalar initialised to zero. The other two
           CONVERT the network and therefore spend appearance capacity to buy
           temporal capacity -- measured, both start 0.011-0.018 BELOW the 2D
           model before any training. This one starts exactly AT it.

THE INVARIANT THAT MAKES THIS CHECKABLE. With the shift disabled, TSM must
reproduce the 2D model exactly; with a clip of identical frames, an inflated
convolution must reproduce the 2D convolution at interior time positions. Both
are asserted in tests/test_temporal.py. A silent wiring error here would look
exactly like "temporal modelling does not help", which is the conclusion we are
trying to test rather than manufacture.

PREPROCESSING IS INHERITED, NOT CHOSEN. These carry surgical weights that were
fitted on [0, 1] RGB at 384px with NO mean/std normalisation (see
train.prepare_batch). Feeding them Kinetics statistics -- which the r2plus1d
path correctly uses for ITS weights -- would degrade them for a reason that
looks like a bad architecture. `build_temporal_model` records the expected
preprocessing alongside the model so a trainer cannot guess.
"""
import torch
from torch import nn

#: What the 2D checkpoints were trained with. Not a choice made here: it is a
#: property of those weights, read off scripts/train_tools.py.
TWO_D_IMAGE_SIZE = 384
TWO_D_NORMALISATION = "unit"


def shift_channels(x, segments, fold_div=8):
    """Move a fraction of channels one step along time. TSM's whole mechanism.

    `x` is (B*T, C, H, W) -- frames folded into the batch, which is what makes
    this free: the 2D convolutions never learn that time exists, they just
    receive features that have already been mixed across it.

    A 1/fold_div slice shifts backward (each frame sees the NEXT frame's
    channels), another shifts forward (sees the PREVIOUS frame's), and the
    remaining 1 - 2/fold_div stays put. Edge frames shift in zeros, which is
    the standard treatment: there is no earlier frame to borrow from.

    fold_div=0 disables the shift entirely and is the control -- with it, the
    network is exactly the 2D model applied per frame.
    """
    if not fold_div:
        return x
    nt, c, h, w = x.shape
    if nt % segments:
        raise ValueError(
            "%d rows is not divisible by %d segments. The batch and the time "
            "axis are folded together here, so a mismatch would silently mix "
            "channels ACROSS CLIPS -- frames of one window borrowing from "
            "another window's." % (nt, segments))
    fold = c // fold_div
    if fold == 0:
        raise ValueError("fold_div=%d leaves no channels to shift out of %d"
                         % (fold_div, c))
    view = x.view(nt // segments, segments, c, h, w)
    out = torch.zeros_like(view)
    out[:, :-1, :fold] = view[:, 1:, :fold]                  # borrow from next
    out[:, 1:, fold:2 * fold] = view[:, :-1, fold:2 * fold]  # borrow from prev
    out[:, :, 2 * fold:] = view[:, :, 2 * fold:]             # unchanged
    return out.view(nt, c, h, w)


class TemporalShiftHook:
    """Applies `shift_channels` to a conv's input, as a forward pre-hook.

    A HOOK RATHER THAN A WRAPPER MODULE, deliberately. Wrapping `block.conv1`
    in a container renames the parameter to `layer1.0.conv1.net.weight`, and
    then loading our 2D checkpoint either fails loudly or -- with strict=False,
    which is how this usually goes wrong -- succeeds while silently leaving the
    convolutions at their random initialisation. A hook changes no names, so
    the state dict loads as it would into the 2D model.
    """

    def __init__(self, segments, fold_div):
        self.segments = segments
        self.fold_div = fold_div

    def __call__(self, module, inputs):
        return (shift_channels(inputs[0], self.segments, self.fold_div),)


class TemporalWrapper(nn.Module):
    """(B, C, T, H, W) in, per-clip logits out, with a 2D network inside.

    The input layout matches `train.prepare_clip_batch` so the 3D and the
    converted-2D paths take exactly the same batches. Frames are folded into
    the batch, the 2D network runs once over all of them, and the per-frame
    logits are averaged -- TSM's "consensus", and the same aggregation the 2D
    serving path already applies over its sampled frames.
    """

    def __init__(self, backbone, segments, per_frame=False):
        super().__init__()
        self.backbone = backbone
        self.segments = segments
        # PER-FRAME LOGITS, for evaluation only. Training averages logits and
        # feeds BCEWithLogitsLoss, which is a legitimate choice -- but the 2D
        # reference path applies sigmoid PER FRAME and averages PROBABILITIES,
        # and sigmoid(mean(logits)) is not mean(sigmoid(logits)). Scoring a
        # converted model the first way against a 2D number computed the second
        # way compares two aggregations, not two models.
        self.per_frame = per_frame

    def forward(self, x):
        batch, channels, time, height, width = x.shape
        if time != self.segments:
            raise ValueError(
                "clip has %d frames but the shift was built for %d segments. "
                "The shift folds time into the batch, so a mismatch mixes "
                "frames across clips." % (time, self.segments))
        folded = x.permute(0, 2, 1, 3, 4).reshape(
            batch * time, channels, height, width)
        logits = self.backbone(folded).view(batch, time, -1)
        return logits if self.per_frame else logits.mean(dim=1)


def _inflate_conv(conv2d, temporal=3):
    """Conv2d -> Conv3d, kernels repeated along time and divided by its extent.

    Dividing by `temporal` is what makes the inflation an IDENTITY on static
    input: summing t copies of W/t over t identical frames returns exactly what
    W returned on one frame. Without the division the inflated network's
    activations are t times too large and every batch-norm statistic it
    inherited is wrong.
    """
    kernel = (temporal,) + tuple(conv2d.kernel_size)
    stride = (1,) + tuple(conv2d.stride)
    padding = (temporal // 2,) + tuple(conv2d.padding)
    conv3d = nn.Conv3d(conv2d.in_channels, conv2d.out_channels, kernel,
                       stride=stride, padding=padding,
                       bias=conv2d.bias is not None)
    with torch.no_grad():
        weight = conv2d.weight.data.unsqueeze(2).repeat(1, 1, temporal, 1, 1)
        conv3d.weight.copy_(weight / float(temporal))
        if conv2d.bias is not None:
            conv3d.bias.copy_(conv2d.bias.data)
    return conv3d


def _inflate_bn(bn2d):
    bn3d = nn.BatchNorm3d(bn2d.num_features, eps=bn2d.eps,
                          momentum=bn2d.momentum, affine=bn2d.affine,
                          track_running_stats=bn2d.track_running_stats)
    with torch.no_grad():
        if bn2d.affine:
            bn3d.weight.copy_(bn2d.weight.data)
            bn3d.bias.copy_(bn2d.bias.data)
        if bn2d.track_running_stats:
            bn3d.running_mean.copy_(bn2d.running_mean.data)
            bn3d.running_var.copy_(bn2d.running_var.data)
            bn3d.num_batches_tracked.copy_(bn2d.num_batches_tracked.data)
    return bn3d


def inflate(module, temporal_convs=("conv1",), temporal=3, path=""):
    """Recursively replace 2D layers with 3D ones, in place, keeping names.

    `temporal_convs` names which convolutions get a temporal extent > 1. Every
    other convolution is inflated with extent 1, which is a 2D convolution
    applied independently per frame -- so this is not "3D everywhere", it is
    the standard recipe of putting temporal kernels where they buy the most
    per unit of memory.

    NAMES ARE PRESERVED because the replacement is a setattr on the parent
    under the same attribute. That is what lets the inflated network's state
    dict be built from the 2D checkpoint's keys directly.
    """
    for name, child in list(module.named_children()):
        full = "%s.%s" % (path, name) if path else name
        if isinstance(child, nn.Conv2d):
            extent = temporal if name in temporal_convs else 1
            setattr(module, name, _inflate_conv(child, extent))
        elif isinstance(child, nn.BatchNorm2d):
            setattr(module, name, _inflate_bn(child))
        elif isinstance(child, nn.MaxPool2d):
            setattr(module, name, nn.MaxPool3d(
                kernel_size=(1, child.kernel_size, child.kernel_size),
                stride=(1, child.stride, child.stride),
                padding=(0, child.padding, child.padding)))
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(module, name, nn.AdaptiveAvgPool3d((1, 1, 1)))
        else:
            inflate(child, temporal_convs, temporal, full)
    return module


class InflatedWrapper(nn.Module):
    """(B, C, T, H, W) in, per-clip logits out, with an inflated net inside.

    torchvision's ResNet flattens after its pool, and a 3D pool leaves an extra
    axis, so the flatten has to be told about it. Subclassing rather than
    monkey-patching `forward` keeps the parameter names -- and therefore the
    checkpoint compatibility -- intact.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        net = self.backbone
        out = net.conv1(x)
        out = net.bn1(out)
        out = net.relu(out)
        out = net.maxpool(out)
        for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
            out = layer(out)
        out = net.avgpool(out)
        return net.fc(torch.flatten(out, 1))


def _load_surgical_weights(model, checkpoint, expect_classes=None):
    """Our fine-tuned 2D weights, loaded STRICTLY.

    strict=True is the point. Both constructions here preserve parameter names
    precisely so this can be strict; the failure mode it prevents -- a rename
    that leaves half the network at its random initialisation while training
    proceeds to a plausible loss -- is exactly the kind of bug that would be
    reported as a negative result about temporal modelling.
    """
    payload = torch.load(str(checkpoint), map_location="cpu",
                         weights_only=False)
    meta = payload.get("meta", {})
    if expect_classes is not None and meta.get("classes"):
        if list(meta["classes"]) != list(expect_classes):
            raise ValueError("%s was trained on %r, not %r"
                             % (checkpoint, meta.get("classes"),
                                list(expect_classes)))
    if meta.get("temporal"):
        raise ValueError(
            "%s is a TEMPORAL checkpoint. These builders inflate a 2D network; "
            "handed 3D weights the names would not match." % checkpoint)
    model.load_state_dict(payload["state_dict"])
    return meta


def build_tsm_model(num_outputs, checkpoint=None, backbone="resnet50",
                    segments=8, fold_div=8, expect_classes=None,
                    per_frame=False):
    """Our 2D network with a temporal shift in every residual block.

    Returns (model, meta). `fold_div=0` builds the control: no shift at all,
    which must score exactly what the 2D model scores.
    """
    from surgvu.models import build_model

    model = build_model(num_outputs, backbone, pretrained=checkpoint is None)
    meta = {}
    if checkpoint:
        meta = _load_surgical_weights(model, checkpoint, expect_classes)

    # The residual blocks only. NOT the stem: shifting there mixes raw pixels
    # across frames before any feature exists, which is not what TSM does.
    hooked = 0
    for module in model.modules():
        conv1 = getattr(module, "conv1", None)
        if isinstance(conv1, nn.Conv2d) and hasattr(module, "bn1") \
                and hasattr(module, "conv2"):
            conv1.register_forward_pre_hook(
                TemporalShiftHook(segments, fold_div))
            hooked += 1
    if checkpoint and hooked == 0:
        raise ValueError(
            "no residual blocks found in %r, so no shift was installed and "
            "this is the 2D model wearing a temporal name." % backbone)
    return (TemporalWrapper(model, segments, per_frame=per_frame),
            dict(meta, tsm_blocks=hooked))


class MotionBranch(nn.Module):
    """Per-burst motion descriptor from a frozen trunk's feature maps.

    Input is (N, P, C, H, W): N bursts, P frames each, already through the 2D
    trunk. What it reads is the DIFFERENCE between consecutive frames' feature
    maps -- at 67 ms spacing the appearance is nearly identical, so the
    difference is almost entirely motion, and handing the branch the raw
    features instead would let it re-learn appearance the frozen trunk already
    encodes better.

    GroupNorm, not BatchNorm. Every other normalisation in this file is
    inherited from the 2D checkpoint and must be pinned; this branch is new, so
    it gets a normalisation whose statistics do not depend on the batch at all.
    That removes the failure mode that cost three epochs on every conversion
    arm -- there are no running statistics here to corrupt.
    """

    def __init__(self, in_channels, frames, num_outputs, hidden=256):
        super().__init__()
        if frames < 2:
            raise ValueError(
                "a burst of %d frame(s) has no motion in it: the branch reads "
                "differences between consecutive frames and there are none. "
                "Extract at least 2 frames per burst." % frames)
        if hidden % 32:
            raise ValueError("hidden must be a multiple of 32 for GroupNorm's "
                             "32 groups, got %d" % hidden)
        self.frames = frames
        self.reduce = nn.Conv2d(in_channels * (frames - 1), hidden, 1)
        self.norm1 = nn.GroupNorm(32, hidden)
        self.spatial = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, hidden)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(hidden, num_outputs)

    def forward(self, feats):
        bursts, frames, channels, height, width = feats.shape
        if frames != self.frames:
            raise ValueError(
                "burst holds %d frames but the branch was built for %d"
                % (frames, self.frames))
        diffs = (feats[:, 1:] - feats[:, :-1]).reshape(
            bursts, (frames - 1) * channels, height, width)
        out = self.act(self.norm1(self.reduce(diffs)))
        out = self.act(self.norm2(self.spatial(out)))
        return self.head(torch.flatten(self.pool(out), 1))


class SequenceBranch(nn.Module):
    """Motion across the WHOLE window, not inside a single burst.

    WHY THIS EXISTS. `MotionBranch` reads differences between frames 67 ms
    apart and produces one correction per burst; those corrections are then
    averaged. Averaging is ORDER-INVARIANT, so a model built only from it
    cannot distinguish a sequence from its reverse -- it sees sixteen
    disconnected 0.2 s twitches and takes their mean. "Grasper enters,
    retracts tissue, then cuts" and the same three events in any other order
    are the same number to it.

    That is the wrong resolution for the question being asked. The label
    describes 30 seconds, and what happens over 30 seconds -- an instrument
    entering and leaving, tissue deforming, one phase becoming another -- is
    exactly the structure a per-burst mean destroys.

    THE DATA ALREADY CONTAINS IT. The 16 burst centres are spread across the
    full window at 1.875 s spacing, so the long timescale is in the pool
    already; only the architecture was discarding it. This needs no
    re-extraction, and it costs no extra trunk forwards, because it consumes
    features the trunk has already computed.

    Dilated temporal convolutions over the burst axis, 1/2/4, which gives a
    receptive field of 15 bursts -- essentially the whole 16-burst window --
    in three cheap layers. GroupNorm for the same reason MotionBranch uses it:
    nothing here should depend on the batch.
    """

    def __init__(self, in_channels, bursts, num_outputs, hidden=256,
                 dilations=(1, 2, 4)):
        super().__init__()
        if hidden % 32:
            raise ValueError("hidden must be a multiple of 32 for GroupNorm's "
                             "32 groups, got %d" % hidden)
        if bursts < 2:
            raise ValueError(
                "a window of %d burst(s) has no sequence in it: this branch "
                "models change ACROSS bursts and there is nothing to cross."
                % bursts)
        self.bursts = bursts
        self.project = nn.Conv1d(in_channels, hidden, 1)
        blocks = []
        for dilation in dilations:
            blocks.append(nn.Sequential(
                nn.Conv1d(hidden, hidden, 3, padding=dilation,
                          dilation=dilation),
                nn.GroupNorm(32, hidden),
                nn.ReLU(inplace=True)))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv1d(hidden, num_outputs, 1)
        # Receptive field of stacked kernel-3 convs: 1 + 2*sum(dilations).
        self.receptive_field = 1 + 2 * sum(dilations)

    def forward(self, feats):
        """(B, bursts, C) -> (B, bursts, num_outputs), order-aware."""
        bursts = feats.shape[1]
        if bursts != self.bursts:
            raise ValueError(
                "window holds %d bursts but this branch was built for %d"
                % (bursts, self.bursts))
        # (B, bursts, C) -> (B, C, bursts): time is the CONVOLVED axis here,
        # which is the whole point. Getting this transpose wrong would
        # convolve across feature channels and still train to a plausible loss.
        out = self.project(feats.transpose(1, 2))
        for block in self.blocks:
            out = out + block(out)          # residual, so depth cannot hurt
        return self.head(out).transpose(1, 2)


class ResidualTemporal(nn.Module):
    """The 2D model, plus a temporal correction that starts at exactly zero.

    WHY THIS RATHER THAN A CONVERSION. TSM and I3D do not ADD temporal
    capacity to the 2D model, they TRADE appearance capacity for it: fold_div=8
    replaces a quarter of every block's channels with time-shifted copies, and
    an inflated kernel is divided by its extent. That is why the UNTRAINED
    conversions already cost 0.011-0.018 before a single gradient step -- the
    mechanism has to earn back what installing it spent. Measured over a night
    of arms, the best one reached parity (0.7793 against 0.7802) and none beat
    it.

    This construction cannot start behind, because at initialisation it IS the
    2D model:

        logits_i = twod_logits(centre of burst i) + alpha * motion(burst i)

    with `alpha` a learned scalar initialised to ZERO. At alpha=0 the second
    term vanishes identically and the aggregated output equals the 2D model's
    own aggregation over the same 16 moments -- asserted in
    scripts/verify_temporal.py, not merely intended. The temporal branch
    switches on only insofar as training earns it, and the worst case is that
    alpha stays near zero and the arm reproduces the 2D number.

    ALPHA IS ZERO WHILE THE BRANCH IS RANDOM, which is the ReZero
    initialisation and not an oversight. The gradient of the loss with respect
    to alpha is the branch's output dotted with the upstream gradient, which is
    non-zero, so alpha moves off zero on the first step and the branch trains
    from there. The alternative -- alpha=1 with a zero-initialised head -- is
    also live at init but lets a randomly-scaled correction reach the logits as
    soon as the head moves.

    THE TRUNK IS FROZEN AND STAYS IN EVAL MODE. That is what makes "cannot
    hurt" structural rather than a property of the first step: the 2D path
    cannot drift, its BatchNorm statistics cannot be overwritten, and the only
    trainable parameters are alpha and the branch. It also makes the arm cheap
    -- the trunk runs under no_grad, so 48 frames per window cost forward
    passes and no stored activations.
    """

    #: Layers a torchvision ResNet exposes, in the order this walks them.
    _TRUNK = ("conv1", "bn1", "relu", "maxpool",
              "layer1", "layer2", "layer3", "layer4")

    def __init__(self, backbone, bursts, frames_per_burst, num_outputs,
                 tap="layer4", hidden=256, per_frame=False,
                 freeze_backbone=True, chunk=48, sequence=False):
        super().__init__()
        for name in self._TRUNK:
            if not hasattr(backbone, name):
                raise ValueError(
                    "backbone exposes no %r, so the trunk cannot be walked and "
                    "the branch has nothing to tap. This construction is "
                    "written against torchvision's ResNet layout." % name)
        if tap not in ("layer3", "layer4"):
            raise ValueError("tap must be layer3 or layer4, got %r" % (tap,))
        self.backbone = backbone
        self.bursts = bursts
        self.frames_per_burst = frames_per_burst
        self.tap = tap
        self.per_frame = per_frame
        self.freeze_backbone = freeze_backbone
        # Frames per trunk call. 0 runs them all at once, which is only safe
        # when the caller knows the batch is small.
        self.chunk = int(chunk)

        # READ OFF THE STAGE, not hardcoded per depth. A Bottleneck ends in
        # bn3 and a BasicBlock in bn2, so resnet50 gives 2048/1024 and
        # resnet18 gives 512/256 without this needing to know which is which.
        # A wrong constant here is a shape error at the first batch, which is
        # loud -- but only after a GPU slot has been waited for.
        last = getattr(backbone, tap)[-1]
        norm = getattr(last, "bn3", None) or getattr(last, "bn2")
        self.branch = MotionBranch(norm.num_features, frames_per_burst,
                                   num_outputs, hidden=hidden)
        # A SCALAR, not a per-class vector. A per-class alpha is twelve
        # separately-escaping gates and any one of them can move the shipped
        # prediction on its own; one scalar makes "did temporal help" a single
        # number that can be read off the checkpoint and reported.
        self.alpha = nn.Parameter(torch.zeros(1))

        # THE SECOND TIMESCALE, off by default. `branch` reads motion INSIDE a
        # burst -- 0.2 s -- and its per-burst corrections are then averaged,
        # which is order-invariant: a window and its reverse produce the same
        # number. `sequence` reads across the sixteen burst centres, which span
        # the whole 30 s at 1.875 s spacing, so it sees the window's shape
        # rather than sixteen disconnected twitches.
        #
        # A SEPARATE GATE, also zero. Two branches behind one gate would be
        # one experiment with two explanations; behind two gates the trained
        # values ARE the attribution, readable off the checkpoint.
        self.sequence = None
        self.beta = None
        if sequence:
            self.sequence = SequenceBranch(norm.num_features, bursts,
                                           num_outputs, hidden=hidden)
            self.beta = nn.Parameter(torch.zeros(1))

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    def train(self, mode=True):
        """Trainable parts follow `mode`; a frozen trunk never leaves eval.

        Bound to train() itself rather than applied once before the loop, for
        the same reason train_temporal.py binds its BatchNorm freeze there: the
        shared epoch function calls model.train(True) at the top of every
        epoch, and a freeze applied from outside would be silently undone on
        the first line of the first one.
        """
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _trunk_once(self, folded):
        """One chunk of frames through the 2D network: tap features, logits."""
        net = self.backbone
        out = folded
        tapped = None
        for name in self._TRUNK:
            out = getattr(net, name)(out)
            if name == self.tap:
                tapped = out
        logits = net.fc(torch.flatten(net.avgpool(out), 1))
        return tapped, logits

    def _trunk(self, folded):
        """All frames through the 2D network, in chunks.

        CHUNKED BECAUSE THE FRAME COUNT IS THE POINT. A window here is 16
        bursts of 3 at the 2D model's own 384px, so a batch of 4 windows is 192
        images -- and conv1 alone holds 192x64x192x192 floats, 1.8 GB, before
        anything else is allocated. The frames are independent (the trunk has
        no temporal mechanism in it; that is the branch's whole job), so
        splitting them changes nothing about the result and bounds the peak.
        """
        if not self.chunk or folded.shape[0] <= self.chunk:
            return self._trunk_once(folded)
        tapped, logits = [], []
        for start in range(0, folded.shape[0], self.chunk):
            piece = self._trunk_once(folded[start:start + self.chunk])
            tapped.append(piece[0])
            logits.append(piece[1])
        return torch.cat(tapped), torch.cat(logits)

    def forward(self, x):
        batch, channels, time, height, width = x.shape
        expect = self.bursts * self.frames_per_burst
        if time != expect:
            raise ValueError(
                "clip holds %d frames but this model reads %d bursts of %d = "
                "%d. The bursts are contiguous runs inside the clip, so a "
                "mismatch would split them at the wrong frames and call two "
                "halves of different moments a motion."
                % (time, self.bursts, self.frames_per_burst, expect))
        folded = x.permute(0, 2, 1, 3, 4).reshape(
            batch * time, channels, height, width)

        if self.freeze_backbone:
            with torch.no_grad():
                tapped, logits = self._trunk(folded)
            tapped = tapped.detach()
            logits = logits.detach()
        else:
            tapped, logits = self._trunk(folded)

        per_burst = self.frames_per_burst
        # (B, bursts, P, classes) -> the CENTRE frame of each burst. That frame
        # is the moment the 2D path would have sampled; the two flanking it
        # exist only so the branch has motion to read.
        logits = logits.view(batch, self.bursts, per_burst, -1)
        centre = logits[:, :, per_burst // 2]

        feats = tapped.view(batch * self.bursts, per_burst,
                            *tapped.shape[-3:])
        correction = self.branch(feats).view(batch, self.bursts, -1)

        out = centre + self.alpha * correction

        if self.sequence is not None:
            # One descriptor per burst, from its CENTRE frame -- the same frame
            # the 2D path reads, so this branch and that one are looking at the
            # same sixteen moments and differ only in whether they see them as
            # a sequence or independently. Pooled over space: what travels
            # across 1.875 s is which structures are present and how much,
            # not where a pixel went.
            pooled = tapped.view(batch, self.bursts, per_burst,
                                 *tapped.shape[-3:])[:, :, per_burst // 2]
            pooled = pooled.mean(dim=(-2, -1))          # (B, bursts, C)
            out = out + self.beta * self.sequence(pooled)

        return out if self.per_frame else out.mean(dim=1)


def build_residual_model(num_outputs, checkpoint=None, backbone="resnet50",
                         bursts=16, frames_per_burst=3, tap="layer4",
                         hidden=256, expect_classes=None, per_frame=False,
                         freeze_backbone=True, chunk=48, sequence=False):
    """The 2D model with a zero-initialised temporal branch beside it.

    Returns (model, meta). Unlike the conversions, this REQUIRES a checkpoint:
    the whole construction is "start from the shipped model and add", and
    starting from ImageNet instead would be an ordinary 2D training run with an
    inert branch bolted on.
    """
    from surgvu.models import build_model

    if not checkpoint:
        raise ValueError(
            "build_residual_model needs the 2D checkpoint it is meant to "
            "correct. Without it alpha=0 means 'this is ImageNet', not 'this "
            "is the shipped model', and the arm cannot be compared to 0.7802.")
    model = build_model(num_outputs, backbone, pretrained=False)
    meta = _load_surgical_weights(model, checkpoint, expect_classes)
    wrapped = ResidualTemporal(model, bursts, frames_per_burst, num_outputs,
                               tap=tap, hidden=hidden, per_frame=per_frame,
                               freeze_backbone=freeze_backbone, chunk=chunk,
                               sequence=sequence)
    trainable = sum(p.numel() for p in wrapped.parameters()
                    if p.requires_grad)
    return wrapped, dict(meta, residual_bursts=bursts,
                         residual_frames_per_burst=frames_per_burst,
                         residual_tap=tap, residual_hidden=hidden,
                         residual_trainable_params=trainable,
                         residual_sequence=bool(sequence),
                         frozen_backbone=bool(freeze_backbone))


def build_i3d_model(num_outputs, checkpoint=None, backbone="resnet50",
                    temporal=3, expect_classes=None):
    """Our 2D network inflated to 3D convolutions. Returns (model, meta)."""
    from surgvu.models import build_model

    model = build_model(num_outputs, backbone, pretrained=checkpoint is None)
    meta = {}
    if checkpoint:
        meta = _load_surgical_weights(model, checkpoint, expect_classes)
    inflate(model, temporal_convs=("conv1",), temporal=temporal)
    return InflatedWrapper(model), dict(meta, i3d_temporal=temporal)
