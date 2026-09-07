"""The residual temporal model, and the one property that makes it worth running.

Every temporal arm before this one was a CONVERSION: it rebuilt the 2D network
with a shift or an inflation inside it, and paid for the mechanism up front --
the untrained conversions score 0.011-0.018 below the 2D model before a single
gradient step. `ResidualTemporal` is constructed so that it cannot: at alpha=0
it computes exactly what the 2D model computes, and the assertion that it does
is the first test here.

scripts/verify_temporal.py runs the same invariant against the REAL shipped
checkpoint on a GPU node. This file runs it on random weights, in the test
suite, where a regression is caught before a slot is waited for.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from surgvu.models import build_model
from surgvu.taxonomy import TOOL_CLASSES

from surgvu.temporal import (  # noqa: E402
    MotionBranch, ResidualTemporal, SequenceBranch, build_residual_model,
)

CLASSES = len(TOOL_CLASSES)


def _checkpoint(tmp_path, num_outputs=CLASSES, backbone="resnet50"):
    """A 2D checkpoint on disk, in the layout _load_surgical_weights expects."""
    torch.manual_seed(0)
    model = build_model(num_outputs, backbone, pretrained=False)
    path = tmp_path / "twod.pt"
    torch.save({"state_dict": model.state_dict(),
                "meta": {"classes": list(TOOL_CLASSES)[:num_outputs]}}, path)
    return path, model


def _clip(bursts=2, per_burst=3, size=64):
    torch.manual_seed(1)
    return torch.rand(1, 3, bursts * per_burst, size, size)


def test_at_alpha_zero_it_is_the_2d_model_frame_for_frame(tmp_path):
    """THE invariant, compared LIKE FOR LIKE so it can be exact.

    The residual model pushes all six frames through the trunk in one batch.
    Running the 2D model on just the two centre frames computes the same
    function on a different batch, and float32 reduction order depends on
    batch composition -- measured, that is ~5e-6 on these logits, which is
    noise but is not zero. Comparing against the SAME six-frame batch removes
    the last degree of freedom and makes the assertion exact, so no tolerance
    has to be chosen and nothing can hide under one.
    """
    path, twod = _checkpoint(tmp_path)
    model, _ = build_residual_model(CLASSES, checkpoint=str(path), bursts=2,
                                    frames_per_burst=3, per_frame=True)
    model.eval()
    twod.eval()

    clip = _clip()
    with torch.no_grad():
        out = model(clip)
        every = twod(clip[0].permute(1, 0, 2, 3))       # all six frames
        expected = every[[1, 4]]                        # the burst centres

    assert out.shape == (1, 2, CLASSES)
    assert torch.equal(out[0], expected), (
        "alpha is zero and the branch is therefore inert, so on the same "
        "batch these must be the SAME numbers: max diff %g"
        % (out[0] - expected).abs().max().item())


def test_the_2d_path_picks_the_centre_frame_and_not_a_flanking_one(tmp_path):
    """The exactness above is only meaningful if the index is also right.

    torch.equal against `every[[1, 4]]` would pass just as well for a model
    that read frames 0 and 3 if those happened to be compared too -- so this
    asserts the flanking frames are DIFFERENT numbers, which is what makes
    "it matched the centre" a real claim rather than an arithmetic identity.
    """
    path, twod = _checkpoint(tmp_path)
    model, _ = build_residual_model(CLASSES, checkpoint=str(path), bursts=2,
                                    frames_per_burst=3, per_frame=True)
    model.eval()
    twod.eval()
    clip = _clip()
    with torch.no_grad():
        out = model(clip)[0]
        every = twod(clip[0].permute(1, 0, 2, 3))
    for wrong in (0, 2):
        assert not torch.allclose(out[0], every[wrong], atol=1e-4), (
            "burst 0's output matches frame %d as well as frame 1, so this "
            "clip cannot tell the centre from its flank and the test above "
            "proves nothing." % wrong)


def test_at_alpha_zero_the_aggregated_output_is_the_2d_aggregation(tmp_path):
    """The same claim at the level the score is actually computed on.

    Scoring averages PROBABILITIES over the sampled moments -- see
    dump_temporal_probs --aggregate probs -- so the invariant that matters is
    that this model's aggregate equals the 2D model's aggregate, not just that
    the logits line up before the mean.
    """
    path, twod = _checkpoint(tmp_path)
    model, _ = build_residual_model(CLASSES, checkpoint=str(path), bursts=2,
                                    frames_per_burst=3, per_frame=True)
    model.eval()
    twod.eval()

    clip = _clip()
    with torch.no_grad():
        mine = torch.sigmoid(model(clip)).mean(dim=1)
        every = twod(clip[0].permute(1, 0, 2, 3))
        theirs = torch.sigmoid(every[[1, 4]]).mean(dim=0, keepdim=True)

    assert torch.equal(mine, theirs), (
        "max diff %g" % (mine - theirs).abs().max().item())


def test_the_branch_reaches_the_output_once_alpha_moves(tmp_path):
    """The other half: inert at zero is only useful if it can switch on."""
    path, _ = _checkpoint(tmp_path)
    model, _ = build_residual_model(CLASSES, checkpoint=str(path), bursts=2,
                                    frames_per_burst=3, per_frame=True)
    model.eval()
    clip = _clip()
    with torch.no_grad():
        before = model(clip)
        model.alpha.fill_(1.0)
        after = model(clip)
    assert not torch.allclose(before, after), (
        "alpha=1 changed nothing, so the branch is wired to nothing and this "
        "arm would report the 2D number whatever it learned.")


def test_alpha_receives_gradient_from_the_first_step(tmp_path):
    """ReZero only works if alpha can escape zero. Assert that it can.

    A zero-initialised gate multiplying a randomly-initialised branch is the
    standard construction, but it is easy to write a version where the branch
    output is also zero at init -- and then the gradient with respect to alpha
    is zero too, alpha never moves, and the arm trains to exactly the 2D
    number while looking like it ran.
    """
    path, _ = _checkpoint(tmp_path)
    model, _ = build_residual_model(CLASSES, checkpoint=str(path), bursts=2,
                                    frames_per_burst=3)
    model.train()
    out = model(_clip())
    out.sum().backward()
    assert model.alpha.grad is not None
    assert model.alpha.grad.abs().item() > 0.0


def test_only_alpha_and_the_branch_are_trainable(tmp_path):
    """A frozen trunk is what makes 'cannot hurt' structural, not momentary."""
    path, _ = _checkpoint(tmp_path)
    model, meta = build_residual_model(CLASSES, checkpoint=str(path), bursts=2,
                                       frames_per_burst=3)
    trainable = {name for name, p in model.named_parameters()
                 if p.requires_grad}
    assert trainable
    assert all(name == "alpha" or name.startswith("branch.")
               for name in trainable), sorted(trainable)
    assert meta["residual_trainable_params"] == sum(
        p.numel() for p in model.parameters() if p.requires_grad)


def test_training_the_model_leaves_the_trunk_in_eval(tmp_path):
    """The BatchNorm lesson, bound to train() so it cannot be undone outside.

    run_clip_epoch calls model.train(True) at the top of every epoch. A freeze
    applied once before the loop is silently reverted on the first line of the
    first one, which is how three conversion arms spent three epochs
    recovering from statistics they should never have moved.
    """
    path, _ = _checkpoint(tmp_path)
    model, _ = build_residual_model(CLASSES, checkpoint=str(path), bursts=2,
                                    frames_per_burst=3)
    model.train()
    assert not model.backbone.training
    assert model.branch.training


def test_a_frozen_trunk_does_not_move_when_the_branch_trains(tmp_path):
    """Not just requires_grad=False: the weights are the same after a step."""
    path, _ = _checkpoint(tmp_path)
    model, _ = build_residual_model(CLASSES, checkpoint=str(path), bursts=2,
                                    frames_per_burst=3)
    before = model.backbone.fc.weight.detach().clone()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model.train()
    for _ in range(2):
        optimizer.zero_grad()
        model(_clip()).sum().backward()
        optimizer.step()
    assert torch.equal(before, model.backbone.fc.weight)
    assert model.alpha.detach().abs().item() > 0.0, (
        "two steps at lr 1e-2 and alpha is still zero -- it is not in the "
        "optimiser's parameter list.")


def test_a_clip_that_does_not_split_into_bursts_is_refused(tmp_path):
    path, _ = _checkpoint(tmp_path)
    model, _ = build_residual_model(CLASSES, checkpoint=str(path), bursts=2,
                                    frames_per_burst=3)
    with pytest.raises(ValueError, match="2 bursts of 3"):
        model(_clip(bursts=2, per_burst=4))


def test_it_refuses_a_backbone_whose_trunk_it_cannot_walk():
    """Silence here would mean tapping the wrong layer, or none."""
    model = build_model(CLASSES, "efficientnet_v2_s", pretrained=False)
    with pytest.raises(ValueError, match="trunk cannot be walked"):
        ResidualTemporal(model, 2, 3, CLASSES)


def test_a_single_frame_burst_is_refused():
    with pytest.raises(ValueError, match="no motion in it"):
        MotionBranch(64, 1, CLASSES)


def test_the_branch_needs_a_group_norm_friendly_width():
    with pytest.raises(ValueError, match="multiple of 32"):
        MotionBranch(64, 3, CLASSES, hidden=100)


def test_building_without_the_2d_checkpoint_is_refused():
    """alpha=0 has to mean 'the shipped model', or the arm means nothing."""
    with pytest.raises(ValueError, match="needs the 2D checkpoint"):
        build_residual_model(CLASSES, checkpoint=None)


def test_the_tap_must_name_a_real_stage(tmp_path):
    path, _ = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="tap must be"):
        build_residual_model(CLASSES, checkpoint=str(path), tap="layer2")


def test_the_branch_width_follows_the_tapped_stage(tmp_path):
    """layer3 is 1024 channels on a Bottleneck ResNet and layer4 is 2048.

    Read off the stage rather than hardcoded, so this is the test that the
    reading is right.
    """
    path, _ = _checkpoint(tmp_path)
    deep, _ = build_residual_model(CLASSES, checkpoint=str(path), bursts=2,
                                   frames_per_burst=3, tap="layer4")
    mid, _ = build_residual_model(CLASSES, checkpoint=str(path), bursts=2,
                                  frames_per_burst=3, tap="layer3")
    # Two frames of difference per burst of three, so in_channels x 2.
    assert deep.branch.reduce.in_channels == 2048 * 2
    assert mid.branch.reduce.in_channels == 1024 * 2


def test_a_static_burst_makes_the_correction_a_constant(tmp_path):
    """No motion, no signal: identical frames must give identical corrections.

    The branch reads differences, so a burst of repeated frames carries no
    information about WHICH frame it was -- two different static bursts must
    therefore receive the same correction. If they do not, the branch is
    reading appearance through a path that is supposed to difference it away.
    """
    path, _ = _checkpoint(tmp_path)
    model, _ = build_residual_model(CLASSES, checkpoint=str(path), bursts=2,
                                    frames_per_burst=3, per_frame=True)
    model.eval()
    torch.manual_seed(2)
    one = torch.rand(1, 3, 1, 64, 64).repeat(1, 1, 3, 1, 1)
    two = torch.rand(1, 3, 1, 64, 64).repeat(1, 1, 3, 1, 1)
    clip = torch.cat([one, two], dim=2)
    with torch.no_grad():
        folded = clip.permute(0, 2, 1, 3, 4).reshape(6, 3, 64, 64)
        feats = model._trunk(folded)[0]
        corrections = model.branch(feats.view(2, 3, *feats.shape[-3:]))
    assert torch.allclose(corrections[0], corrections[1], atol=1e-5), (
        "two different static bursts got different corrections, so the "
        "difference is not the only thing the branch sees.")


def test_the_recorded_geometry_survives_into_the_checkpoint_meta(tmp_path):
    """A dump cannot rebuild the model without these, and must not guess."""
    path, _ = _checkpoint(tmp_path)
    _, meta = build_residual_model(CLASSES, checkpoint=str(path), bursts=16,
                                   frames_per_burst=3, tap="layer4")
    assert meta["residual_bursts"] == 16
    assert meta["residual_frames_per_burst"] == 3
    assert meta["residual_tap"] == "layer4"
    assert meta["frozen_backbone"] is True


def test_it_refuses_weights_from_a_different_class_list(tmp_path):
    path, _ = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="was trained on"):
        build_residual_model(CLASSES, checkpoint=str(path),
                             expect_classes=["a", "b"])


def test_the_correction_is_per_burst_not_per_window(tmp_path):
    """Each burst gets its own motion reading, or the mean hides all of them."""
    path, _ = _checkpoint(tmp_path)
    model, _ = build_residual_model(CLASSES, checkpoint=str(path), bursts=3,
                                    frames_per_burst=3, per_frame=True)
    model.eval()
    model.alpha.data.fill_(1.0)
    torch.manual_seed(3)
    clip = torch.rand(1, 3, 9, 64, 64)
    with torch.no_grad():
        out = model(clip)
    assert out.shape == (1, 3, CLASSES)
    spread = np.array([out[0, i].numpy() for i in range(3)])
    assert spread.std(axis=0).max() > 0.0


# --- the motion control ------------------------------------------------------
#
# MotionBranch reads DIFFERENCES, so on a static burst its output collapses to
# a constant driven by its biases, and `alpha * constant` is a per-class logit
# offset. On the tools head that mostly washes out, because per-class
# thresholds are re-tuned on a held-out fold and absorb a constant. On the TASK
# head it does not: that is an argmax over a softmax, a constant offset changes
# predictions outright, and the task head is precisely where the temporal
# hypothesis predicts a gain. So "the arm improved" has to be separable into
# motion and recalibration by measurement rather than by argument.

def _freeze_bursts():
    """The dump script's helper, imported without running the script."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from dump_temporal_probs import freeze_bursts
    return freeze_bursts


def test_freezing_replaces_each_burst_with_its_own_centre_frame():
    freeze_bursts = _freeze_bursts()
    clips = np.arange(12, dtype=np.uint8).reshape(1, 12, 1, 1, 1)
    frozen = freeze_bursts(clips, 3)
    assert frozen[0, :, 0, 0, 0].tolist() == [1, 1, 1, 4, 4, 4,
                                              7, 7, 7, 10, 10, 10]


def test_freezing_keeps_exactly_the_frames_the_2d_path_samples():
    """What makes the decomposition clean rather than merely suggestive.

    The control must differ from the alpha=0 base in the motion branch ONLY.
    If it also moved which frame the 2D path sees, `static - base` would mix a
    bias effect with a resampling effect and mean nothing.
    """
    freeze_bursts = _freeze_bursts()
    clips = np.arange(48, dtype=np.uint8).reshape(1, 48, 1, 1, 1)
    frozen = freeze_bursts(clips, 3)[0, :, 0, 0, 0].tolist()
    centres = list(range(1, 48, 3))                  # 1, 4, ..., 46
    assert frozen[::3] == centres
    assert sorted(set(frozen)) == centres


def test_freezing_leaves_no_motion_at_all():
    freeze_bursts = _freeze_bursts()
    rng = np.random.default_rng(0)
    clips = rng.integers(0, 255, size=(2, 12, 4, 4, 3), dtype=np.uint8)
    frozen = freeze_bursts(clips, 3).reshape(2, 4, 3, -1).astype(np.int16)
    assert (np.diff(frozen, axis=2) == 0).all(), (
        "a within-burst difference survived, so the branch would still see "
        "motion and the control would not be a control")


def test_freezing_does_not_mutate_its_input():
    freeze_bursts = _freeze_bursts()
    clips = np.arange(12, dtype=np.uint8).reshape(1, 12, 1, 1, 1)
    before = clips.copy()
    freeze_bursts(clips, 3)
    assert np.array_equal(clips, before)


def test_freezing_refuses_a_burst_with_no_motion_in_it():
    freeze_bursts = _freeze_bursts()
    clips = np.arange(12, dtype=np.uint8).reshape(1, 12, 1, 1, 1)
    with pytest.raises(ValueError, match="no motion to freeze"):
        freeze_bursts(clips, 1)


def test_freezing_refuses_a_clip_that_does_not_split_into_bursts():
    freeze_bursts = _freeze_bursts()
    clips = np.arange(12, dtype=np.uint8).reshape(1, 12, 1, 1, 1)
    with pytest.raises(ValueError, match="whole number of 5-frame bursts"):
        freeze_bursts(clips, 5)


def test_a_static_clip_makes_the_model_a_pure_offset_on_the_2d_logits(tmp_path):
    """End to end: the control's output really is base + a CONSTANT.

    Not just "the branch sees no motion" -- the claim being relied on is that
    what reaches the logits is the SAME vector for every burst, which is what
    makes it a recalibration rather than a temporal signal.
    """
    freeze_bursts = _freeze_bursts()
    path, twod = _checkpoint(tmp_path)
    model, _ = build_residual_model(CLASSES, checkpoint=str(path), bursts=4,
                                    frames_per_burst=3, per_frame=True)
    model.eval()
    twod.eval()
    model.alpha.data.fill_(1.0)

    torch.manual_seed(5)
    clip = torch.rand(1, 3, 12, 64, 64)
    # freeze_bursts works on the loader's (clips, frames, H, W, C); the model
    # takes (batch, C, frames, H, W). Round-trip the axes through it rather
    # than hand-rolling the freeze here, so this exercises the REAL helper --
    # a reimplementation that agreed with itself would prove nothing.
    as_loader = clip.permute(0, 2, 1, 3, 4).numpy()
    frozen = torch.from_numpy(
        np.ascontiguousarray(freeze_bursts(as_loader, 3))
    ).permute(0, 2, 1, 3, 4).contiguous()

    with torch.no_grad():
        out = model(frozen)[0]
        every = twod(frozen[0].permute(1, 0, 2, 3))
        base = every[[1, 4, 7, 10]]
        offset = out - base
    spread = (offset - offset[0]).abs().max().item()
    assert spread < 1e-4, (
        "the correction differs by %g across bursts of identical static "
        "content, so it is not the constant the control assumes" % spread)


# --- the loader and the evaluator must pick the SAME frames ------------------
#
# ShardTemporal._starts and dump_temporal_probs.clip_indices each implement the
# bursts layout independently -- one for training, one for scoring. If they
# drift apart, an arm is scored on frames it never trained on, and the result
# reads as a property of the architecture.
#
# This is not hypothetical. The `spread` layout used `step = depth // frames`,
# which on a 30-frame window asked for 16 frames and returned frames 0-15 --
# the first sixteen seconds of a thirty-second window, labelled "spread". It
# survived because it was HARMLESS on the 32-frame pool and only wrong on the
# other one, so no single run contradicted itself. The fix had to be versioned
# into the checkpoint because arms trained before it had learned on the old
# picks. The cheapest way not to repeat that is to assert the two agree.

def _clip_indices():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from dump_temporal_probs import clip_indices
    return clip_indices


BURST_GEOMETRIES = [
    (48, 48, 16),      # the real shards_multi16 configuration
    (48, 24, 16),      # half the bursts: the memory lever
    (48, 6, 16),       # two bursts
    (48, 3, 16),       # one burst
    (48, 48, 8),
    (32, 32, 4),       # the older 4-burst pool
    (32, 16, 4),
    (24, 12, 8),
    (96, 24, 16),
]


@pytest.mark.parametrize("depth,frames,bursts", BURST_GEOMETRIES)
def test_the_evaluator_scores_the_frames_the_loader_trained_on(depth, frames,
                                                               bursts):
    from surgvu.dataset import ShardTemporal

    clip_indices = _clip_indices()
    loader = ShardTemporal(["unread.npz"], frames=frames, layout="bursts",
                           bursts=bursts, shuffle=False)
    trained_on = loader._starts(depth, random.Random(0))
    scored_on = clip_indices(depth, frames, "bursts", bursts)

    assert len(trained_on) == 1, "the bursts layout emits exactly one clip"
    assert len(scored_on) == 1
    assert trained_on[0] == scored_on[0][1], (
        "depth=%d frames=%d bursts=%d: the loader trains on %s and the "
        "evaluator scores %s"
        % (depth, frames, bursts, trained_on[0], scored_on[0][1]))


@pytest.mark.parametrize("frames", (7, 50))
def test_both_sides_refuse_a_ragged_frame_count_the_same_way(frames):
    """Agreeing on the answer is not enough; they must agree on the error too.

    One side raising while the other quietly returns something is how a
    mismatch becomes a number instead of a crash.
    """
    from surgvu.dataset import ShardTemporal

    clip_indices = _clip_indices()
    with pytest.raises(ValueError):
        ShardTemporal(["unread.npz"], frames=frames, layout="bursts",
                      bursts=16, shuffle=False)._starts(48, random.Random(0))
    with pytest.raises(ValueError):
        clip_indices(48, frames, "bursts", 16)


def test_the_real_geometry_covers_every_frame_of_every_burst_in_order():
    """48 frames, 16 bursts of 3: the arm must see the whole window, in time."""
    from surgvu.dataset import ShardTemporal

    picks = ShardTemporal(["unread.npz"], frames=48, layout="bursts",
                          bursts=16, shuffle=False)._starts(48,
                                                            random.Random(0))[0]
    assert picks == list(range(48))


# --- the second timescale ----------------------------------------------------
#
# MotionBranch reads inside a burst (0.2 s) and its per-burst corrections are
# then AVERAGED, which is order-invariant: a window and its reverse produce an
# identical number. SequenceBranch reads across the sixteen burst centres,
# which span the whole 30 s window at 1.875 s spacing.
#
# Off by default, and behind its OWN zero-initialised gate, so the two
# timescales are separately attributable -- two branches behind one gate would
# be one experiment with two explanations.

def test_the_sequence_branch_is_off_unless_asked_for(tmp_path):
    path, _ = _checkpoint(tmp_path)
    model, meta = build_residual_model(CLASSES, checkpoint=str(path), bursts=4,
                                       frames_per_burst=3)
    assert model.sequence is None
    assert model.beta is None
    assert meta["residual_sequence"] is False


def test_both_gates_start_at_zero_so_the_model_is_still_the_2d_model(tmp_path):
    """The invariant has to survive the second branch, or it was never load-bearing."""
    path, twod = _checkpoint(tmp_path)
    model, meta = build_residual_model(CLASSES, checkpoint=str(path), bursts=4,
                                       frames_per_burst=3, per_frame=True,
                                       sequence=True)
    model.eval()
    twod.eval()
    assert meta["residual_sequence"] is True
    assert float(model.alpha.item()) == 0.0
    assert float(model.beta.item()) == 0.0

    clip = _clip(bursts=4, per_burst=3)
    with torch.no_grad():
        out = model(clip)
        every = twod(clip[0].permute(1, 0, 2, 3))
        expected = every[[1, 4, 7, 10]]
    assert torch.equal(out[0], expected), (
        "with both gates at zero this must still BE the 2D model: max diff %g"
        % (out[0] - expected).abs().max().item())


def test_the_sequence_branch_reads_ORDER_which_the_mean_cannot(tmp_path):
    """The whole reason this branch exists.

    A per-burst branch followed by a mean gives the same answer for a window
    and its reverse. This asserts the sequence branch does not -- reversing the
    bursts must change its output, or it is a more expensive way to compute
    something order-invariant and buys nothing over the mean.
    """
    path, _ = _checkpoint(tmp_path)
    model, _ = build_residual_model(CLASSES, checkpoint=str(path), bursts=4,
                                    frames_per_burst=3, per_frame=True,
                                    sequence=True)
    model.eval()
    torch.manual_seed(9)
    feats = torch.rand(1, 4, model.sequence.project.in_channels)
    with torch.no_grad():
        forward = model.sequence(feats)
        backward = model.sequence(feats.flip(dims=[1]))
    assert not torch.allclose(forward, backward.flip(dims=[1]), atol=1e-5), (
        "reversing the burst order produced the mirrored output, so this "
        "branch is order-blind and adds nothing the mean did not already do.")


def test_the_local_branch_by_contrast_IS_order_blind_after_the_mean(tmp_path):
    """States the problem the sequence branch solves, as a measurement.

    Not a defect of MotionBranch -- it is what a per-burst correction plus a
    mean necessarily is. Asserting it here means the motivation cannot quietly
    stop being true.
    """
    path, _ = _checkpoint(tmp_path)
    model, _ = build_residual_model(CLASSES, checkpoint=str(path), bursts=4,
                                    frames_per_burst=3)
    model.eval()
    model.alpha.data.fill_(1.0)
    torch.manual_seed(10)
    clip = _clip(bursts=4, per_burst=3)
    # Reverse the ORDER of whole bursts, keeping each burst's frames intact.
    order = torch.cat([torch.arange(s, s + 3) for s in (9, 6, 3, 0)])
    with torch.no_grad():
        forward = model(clip)
        reversed_ = model(clip[:, :, order])
    assert torch.allclose(forward, reversed_, atol=1e-5), (
        "the local arm distinguished a window from its reverse, which a "
        "per-burst correction followed by a mean cannot do -- so one of the "
        "two is not doing what this file claims.")


def test_the_sequence_branch_sees_the_whole_window(tmp_path):
    """Receptive field must span the bursts, or 'long range' is a name only."""
    path, _ = _checkpoint(tmp_path)
    model, _ = build_residual_model(CLASSES, checkpoint=str(path), bursts=16,
                                    frames_per_burst=3, sequence=True)
    assert model.sequence.receptive_field >= 15, (
        "receptive field is %d bursts of 16, so the branch cannot relate the "
        "start of the window to its end" % model.sequence.receptive_field)


def test_the_sequence_branch_needs_more_than_one_burst():
    with pytest.raises(ValueError, match="no sequence in it"):
        SequenceBranch(64, 1, CLASSES)


def test_both_gates_are_trainable_and_nothing_else_new_is(tmp_path):
    path, _ = _checkpoint(tmp_path)
    model, _ = build_residual_model(CLASSES, checkpoint=str(path), bursts=4,
                                    frames_per_burst=3, sequence=True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert "alpha" in trainable and "beta" in trainable
    assert all(n in ("alpha", "beta") or n.startswith(("branch.", "sequence."))
               for n in trainable), sorted(trainable)
