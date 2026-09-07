"""Prove the 2D-to-temporal conversions are wired right, before training on them.

WHY THIS RUNS FIRST. A silent error in either construction -- a renamed
parameter that leaves half the network at its initialisation, an inflation
that scales activations by 3, a shift that mixes frames across clips --
produces a model that trains to a plausible loss and scores badly. That is
indistinguishable from "temporal modelling does not help", which is the
hypothesis under test. So the wiring gets checked against invariants that must
hold by construction, and the training jobs are gated on this passing.

THE INVARIANTS

  1. shift with fold_div=0 is the identity. The control arm must be exactly
     the 2D model, or the control is not a control.

  2. the shift moves the slices it claims to. Checked against an explicitly
     constructed tensor rather than against itself.

  3. an inflated convolution reproduces its 2D original at INTERIOR time
     positions, given identical frames. This is what dividing the repeated
     kernel by its temporal extent buys, and it is the single most likely
     thing to get wrong.

  4. a network inflated with temporal extent 1 reproduces the 2D network
     EXACTLY, frame by frame. Extent 1 is a 2D convolution wearing a 3D shape,
     so any difference here is a structural bug -- pooling, flattening, or a
     batch-norm that did not carry its running statistics.

  5. temporal extent 3 changes the output. Otherwise the temporal path is
     dead and every result would be the 2D model's, relabelled.

  6. both constructions load OUR surgical checkpoint strictly, and the TSM
     control reproduces the 2D model's logits on real weights.

  7. THE RESIDUAL ARM AT alpha=0 IS THE SHIPPED MODEL, on the shipped weights,
     to floating-point equality -- both frame by frame and after the
     probability averaging the score is actually computed on. This is the one
     invariant that makes that arm worth running at all: it is what lets it
     start AT 0.7802 rather than 0.011-0.018 below, the way every conversion
     does. If it fails, the arm is just another conversion with extra
     parameters.

  8. and alpha can still escape zero. An inert branch that cannot switch on
     would reproduce the 2D number forever and look like a completed
     experiment.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch                                                  # noqa: E402
from torch import nn                                          # noqa: E402

from surgvu.models import build_model                          # noqa: E402
from surgvu.temporal import (build_i3d_model, build_residual_model,  # noqa: E402
                             build_tsm_model, inflate, shift_channels)

# THE SHIPPED CHECKPOINT, not the x40 variant this defaulted to for the whole
# of the v4 night. For a wiring check the base weights barely matter -- but
# every conversion arm inherited x40 from a default exactly like this one while
# being compared against a number _long produced, and that mismatch was worth
# 0.048. The default is the shipped model everywhere now.
TOOLS_2D = "/staging/n/nkalthoff/surgvu26/models/tools_resnet50_long.pt"
CHECKS = []


def check(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


@check("shift with fold_div=0 is the identity")
def _identity(_):
    x = torch.randn(8, 16, 4, 4)
    assert torch.equal(shift_channels(x, segments=4, fold_div=0), x)
    return {}


@check("the shift moves exactly the slices it claims")
def _slices(_):
    # One clip, 4 frames, 8 channels: frame f is filled with the value f, so
    # a shifted channel reports which frame it came from.
    x = torch.zeros(4, 8, 1, 1)
    for frame in range(4):
        x[frame] = float(frame)
    out = shift_channels(x, segments=4, fold_div=4)     # fold = 2 channels
    # channels 0-1 borrow from the NEXT frame, and the last frame gets zeros
    assert out[0, 0].item() == 1.0 and out[2, 1].item() == 3.0
    assert out[3, 0].item() == 0.0
    # channels 2-3 borrow from the PREVIOUS frame, first frame gets zeros
    assert out[1, 2].item() == 0.0 and out[3, 3].item() == 2.0
    assert out[0, 2].item() == 0.0
    # channels 4-7 are untouched
    assert out[2, 5].item() == 2.0
    return {}


@check("an inflated conv reproduces the 2D conv at interior positions")
def _inflated_conv(_):
    from surgvu.temporal import _inflate_conv

    torch.manual_seed(0)
    conv2d = nn.Conv2d(3, 8, kernel_size=3, padding=1).eval()
    conv3d = _inflate_conv(conv2d, temporal=3).eval()
    frame = torch.randn(1, 3, 16, 16)
    clip = frame.unsqueeze(2).repeat(1, 1, 5, 1, 1)     # 5 identical frames
    with torch.no_grad():
        flat = conv2d(frame)
        cube = conv3d(clip)
    # Positions 1..3 see three real frames; 0 and 4 see zero padding, so they
    # are EXPECTED to differ and are not asserted on.
    for t in (1, 2, 3):
        assert torch.allclose(cube[:, :, t], flat, atol=1e-5), \
            "interior position %d differs by %.2e" % (
                t, (cube[:, :, t] - flat).abs().max())
    edge = (cube[:, :, 0] - flat).abs().max().item()
    return {"interior_max_abs_diff": float((cube[:, :, 2] - flat).abs().max()),
            "edge_diff_expected_nonzero": edge}


@check("a network inflated at extent 1 IS the 2D network")
def _inflated_network(_):
    torch.manual_seed(0)
    two_d = build_model(12, "resnet50", pretrained=False).eval()
    three_d = build_model(12, "resnet50", pretrained=False).eval()
    three_d.load_state_dict(two_d.state_dict())
    from surgvu.temporal import InflatedWrapper
    wrapped = InflatedWrapper(inflate(three_d, temporal=1)).eval()

    frame = torch.randn(2, 3, 64, 64)
    clip = frame.unsqueeze(2).repeat(1, 1, 4, 1, 1)
    with torch.no_grad():
        flat = two_d(frame)
        cube = wrapped(clip)
    diff = (cube - flat).abs().max().item()
    assert diff < 1e-4, "inflated network drifts from the 2D one by %.2e" % diff
    return {"max_abs_diff": diff}


@check("temporal extent 3 actually changes the answer")
def _extent_matters(_):
    torch.manual_seed(0)
    base = build_model(12, "resnet50", pretrained=False).eval()
    from surgvu.temporal import InflatedWrapper

    flat_net = build_model(12, "resnet50", pretrained=False)
    flat_net.load_state_dict(base.state_dict())
    one = InflatedWrapper(inflate(flat_net, temporal=1)).eval()

    deep_net = build_model(12, "resnet50", pretrained=False)
    deep_net.load_state_dict(base.state_dict())
    three = InflatedWrapper(inflate(deep_net, temporal=3)).eval()

    # A clip with real motion: each frame is a different noise field.
    clip = torch.randn(1, 3, 8, 64, 64)
    with torch.no_grad():
        a, b = one(clip), three(clip)
    delta = (a - b).abs().max().item()
    assert delta > 1e-5, "extent 3 gives the same answer as extent 1"
    return {"max_abs_diff": delta}


@check("TSM with the shift off reproduces the 2D model on real weights")
def _tsm_control(args):
    if not Path(args.checkpoint).exists():
        return {"skipped": "checkpoint %s not present" % args.checkpoint}
    payload = torch.load(args.checkpoint, map_location="cpu",
                         weights_only=False)
    classes = payload["meta"]["classes"]
    two_d = build_model(len(classes), payload["meta"].get("backbone", "resnet50"),
                        pretrained=False)
    two_d.load_state_dict(payload["state_dict"])
    two_d.eval()

    model, meta = build_tsm_model(len(classes), checkpoint=args.checkpoint,
                                  backbone=payload["meta"].get("backbone",
                                                               "resnet50"),
                                  segments=4, fold_div=0)
    model.eval()
    frame = torch.randn(1, 3, 64, 64)
    clip = frame.unsqueeze(2).repeat(1, 1, 4, 1, 1)
    with torch.no_grad():
        diff = (model(clip) - two_d(frame)).abs().max().item()
    assert diff < 1e-4, "TSM control drifts from the 2D model by %.2e" % diff

    shifted, _ = build_tsm_model(len(classes), checkpoint=args.checkpoint,
                                 backbone=payload["meta"].get("backbone",
                                                              "resnet50"),
                                 segments=4, fold_div=8)
    shifted.eval()
    moving = torch.randn(1, 3, 4, 64, 64)
    with torch.no_grad():
        live = (shifted(moving) - model(moving)).abs().max().item()
    assert live > 1e-5, "the shift changes nothing; the temporal path is dead"
    return {"control_max_abs_diff": diff, "shift_effect": live,
            "blocks_hooked": meta.get("tsm_blocks")}


@check("I3D loads the surgical checkpoint strictly")
def _i3d_loads(args):
    if not Path(args.checkpoint).exists():
        return {"skipped": "checkpoint %s not present" % args.checkpoint}
    payload = torch.load(args.checkpoint, map_location="cpu",
                         weights_only=False)
    classes = payload["meta"]["classes"]
    model, meta = build_i3d_model(len(classes), checkpoint=args.checkpoint,
                                  backbone=payload["meta"].get("backbone",
                                                               "resnet50"))
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 8, 112, 112))
    assert out.shape == (1, len(classes)), "got %r" % (tuple(out.shape),)
    assert torch.isfinite(out).all()
    return {"outputs": int(out.shape[1]), "temporal": meta.get("i3d_temporal")}


@check("the residual arm at alpha=0 IS the 2D model, on real weights")
def _residual_is_the_2d_model(args):
    """The invariant the whole construction exists to provide.

    Checked two ways because the arm is judged two ways: frame by frame, and
    after the probability averaging that produces the 0.7802-comparable
    number. The second is not implied by the first being *close* -- it is only
    implied by the first being exact, which is why the tolerance here is
    floating-point noise and not a threshold.
    """
    payload = torch.load(args.checkpoint, map_location="cpu",
                         weights_only=False)
    classes = payload["meta"].get("classes") or range(12)
    backbone = payload["meta"].get("backbone", "resnet50")

    model, meta = build_residual_model(len(classes), checkpoint=args.checkpoint,
                                       backbone=backbone, bursts=2,
                                       frames_per_burst=3, per_frame=True)
    twod = build_model(len(classes), backbone, pretrained=False)
    twod.load_state_dict(payload["state_dict"])
    model.eval()
    twod.eval()

    torch.manual_seed(0)
    clip = torch.rand(1, 3, 6, 128, 128)
    with torch.no_grad():
        mine = model(clip)
        # SAME BATCH, so the comparison is exact. The residual model pushes all
        # six frames through the trunk at once; running the 2D model on just
        # the two centres computes the same function on a different batch, and
        # float32 reduction order depends on batch composition -- about 5e-6
        # on these logits. Both are reported: the exact one is the invariant,
        # the batched one is the size of the noise it would otherwise hide in.
        every = twod(clip[0].permute(1, 0, 2, 3))
        exact_diff = (mine[0] - every[[1, 4]]).abs().max().item()
        centres = clip[0, :, [1, 4]].permute(1, 0, 2, 3)
        theirs = twod(centres)
        frame_diff = (mine[0] - theirs).abs().max().item()
        agg_diff = (torch.sigmoid(mine).mean(dim=1)
                    - torch.sigmoid(theirs).mean(dim=0)).abs().max().item()
        # What a REAL difference looks like on this input, so the tolerances
        # above can be read as "noise" rather than merely "small".
        model.alpha.data.fill_(1.0)
        live = (model(clip)[0] - mine[0]).abs().max().item()
        model.alpha.data.zero_()

    assert float(model.alpha.item()) == 0.0, "alpha did not start at zero"
    assert exact_diff == 0.0, ("on the same batch the residual model must BE "
                               "the 2D model, and it differs by %g" % exact_diff)
    assert frame_diff < 1e-5, "per-frame logits differ by %g" % frame_diff
    assert agg_diff < 1e-6, "aggregated probabilities differ by %g" % agg_diff
    assert live > 1000 * max(frame_diff, 1e-12), (
        "the branch's effect (%g) is not clearly larger than the numerical "
        "noise (%g), so neither number means anything" % (live, frame_diff))
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"same_batch_max_abs_diff": exact_diff,
            "per_frame_max_abs_diff": frame_diff,
            "aggregated_max_abs_diff": agg_diff,
            "effect_of_alpha_1_for_scale": live,
            "trainable": meta["residual_trainable_params"],
            "frozen": frozen}


@check("the residual branch can still switch on")
def _residual_branch_is_live(args):
    """Inert at init is only useful if it is not inert forever."""
    payload = torch.load(args.checkpoint, map_location="cpu",
                         weights_only=False)
    classes = payload["meta"].get("classes") or range(12)
    model, _ = build_residual_model(
        len(classes), checkpoint=args.checkpoint,
        backbone=payload["meta"].get("backbone", "resnet50"),
        bursts=2, frames_per_burst=3)
    model.train()
    torch.manual_seed(0)
    model(torch.rand(1, 3, 6, 128, 128)).sum().backward()
    grad = float(model.alpha.grad.abs().item())
    assert grad > 0.0, "alpha receives no gradient, so it can never leave zero"

    model.eval()
    with torch.no_grad():
        clip = torch.rand(1, 3, 6, 128, 128)
        before = model(clip)
        model.alpha.data.fill_(1.0)
        effect = (model(clip) - before).abs().max().item()
    assert effect > 0.0, "alpha=1 changed nothing; the branch reaches nothing"
    return {"alpha_grad": grad, "alpha_1_effect": effect}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out")
    parser.add_argument("--checkpoint", default=TOOLS_2D)
    args = parser.parse_args(argv)

    results, failed = {}, []
    for name, fn in CHECKS:
        try:
            results[name] = fn(args) or {"ok": True}
            print("PASS  %s  %s" % (name, results[name]), flush=True)
        except Exception as error:                    # noqa: BLE001
            failed.append(name)
            results[name] = {"failed": str(error)}
            print("FAIL  %s\n      %s" % (name, error), flush=True)

    print("\n%d/%d checks passed" % (len(CHECKS) - len(failed), len(CHECKS)))
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"checks": results, "failed": failed}, indent=2), encoding="utf-8")
    if failed:
        print("DO NOT TRAIN on these constructions until this is green: a "
              "wiring bug here is indistinguishable from a negative result.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
