"""Shared training substrate for all three perception experts.

One loop, one checkpoint format, one seeding routine, so the three models
differ only in their head width, loss, and labels -- and so a metric measured
on one is comparable to a metric measured on another.
"""
import random

import numpy as np
import torch


def seed_everything(seed, deterministic=False):
    """Seed every RNG this project draws from, and optionally pin cuDNN.

    Seeding python, numpy and torch is NOT sufficient on CUDA. cuDNN selects
    convolution algorithms at runtime and several of them accumulate in
    nondeterministic order, so two runs at the same config and the same seed
    diverge. Measured on the task model, `scripts/train_task.py`, identical
    arguments:

        run A   val_acc 0.6175     epoch-3 macro-F1 0.8695
        run B   val_acc 0.6293     epoch-3 macro-F1 0.8657

    That is a floor of roughly 0.012 on any difference we might claim between
    two configurations, and it is a property a challenge submission has to
    state rather than discover.

    `deterministic=True` sets `torch.backends.cudnn.deterministic` and clears
    `torch.backends.cudnn.benchmark`. BOTH are required: `benchmark=True`
    re-runs algorithm selection per input shape, which reintroduces the choice
    that `deterministic` was meant to remove.

    THE TRADE-OFF, and why this is opt-in. Deterministic cuDNN restricts the
    algorithm pool -- it forbids the fastest kernels for some convolution
    shapes -- so it costs training throughput. The size of that cost on this
    workload (EfficientNetV2-S, 384x384, batch 48, T4/L40-class GPUs) has NOT
    been measured here, and the default is therefore left where it was rather
    than changed on an assumption.

    WHICH SETTING PRODUCED THE SHIPPED NUMBERS. `deterministic=False`, i.e.
    PyTorch's own defaults (`cudnn.deterministic False`, `cudnn.benchmark
    False`) -- the flags did not exist when those runs happened. Everything
    bound by `config/perception.json` was trained that way: `tools_v2.pt` at
    val macro-F1 0.6605 and `task_v2.pt` at accuracy 0.8695 / macro-F1 0.6803.
    Those figures carry the ~0.012 band above. Reproducing them exactly is not
    possible retroactively; runs launched with `--deterministic` from here on
    are reproducible among themselves, and are not directly comparable to a
    non-deterministic run at the same seed.

    Both branches WRITE both flags rather than leaving them alone. The flags
    are process-global, so a function that only ever turns determinism on
    cannot turn it back off, and a second run in the same process would
    silently inherit the first one's setting.

    Not enabled here: `torch.use_deterministic_algorithms(True)`, which is
    stricter -- it covers non-cuDNN kernels too and RAISES on ops that have no
    deterministic implementation. Several of those are reachable from this
    backbone's backward pass, so switching it on would convert an unstated
    variance into a hard failure. It has not been measured on this workload
    and is deliberately out of scope.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = False


def prepare_batch(frames_uint8, device, image_size=None):
    """(B, H, W, 3) uint8 BGR -> (B, 3, H, W) float32 RGB in [0, 1].

    The resize lives here, not in the caller, so training and inference
    cannot disagree about it. Shards hold 512x512; EfficientNetV2-S was
    pretrained at 384. Training at one resolution and serving at another is
    a silent accuracy loss that looks like a bad architecture choice.
    """
    array = np.asarray(frames_uint8)
    rgb = array[..., ::-1].copy()             # OpenCV BGR -> RGB
    tensor = torch.from_numpy(rgb).to(device)
    batch = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    if image_size:
        batch = torch.nn.functional.interpolate(
            batch, size=(image_size, image_size),
            mode="bilinear", align_corners=False)
    return batch


def prepare_clip_batch(clips_uint8, device, image_size=112,
                       mean=None, std=None):
    """(B, T, H, W, 3) uint8 BGR -> (B, 3, T, H, W) float32 RGB, normalised.

    The channel order a 3D convolution wants is (B, C, T, H, W), not the
    (B, T, C, H, W) that falls out of stacking frames -- and getting it wrong
    does not raise. Conv3d would happily convolve across the CHANNEL axis as
    if it were time and train to a plausible-looking loss on nothing.

    `mean`/`std` are applied when given. The 2D path deliberately passes
    neither and feeds [0, 1] directly; the Kinetics video weights were trained
    with their own statistics, so they get them. Each model sees the
    preprocessing its own pretraining used, which is the fair comparison --
    forcing a shared convention would just mean one of them is mis-fed.
    """
    array = np.asarray(clips_uint8)
    rgb = array[..., ::-1].copy()                  # OpenCV BGR -> RGB
    tensor = torch.from_numpy(rgb).to(device)
    # (B, T, H, W, C) -> (B, C, T, H, W)
    batch = tensor.permute(0, 4, 1, 2, 3).float().div_(255.0)

    if image_size:
        b, c, t, h, w = batch.shape
        # interpolate() takes one spatial grid at a time, so fold time into
        # the batch axis, resize, and unfold. Passing a 5-D tensor to a 2-D
        # mode would resize across time instead, blending frames together.
        flat = batch.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        flat = torch.nn.functional.interpolate(
            flat, size=(image_size, image_size),
            mode="bilinear", align_corners=False)
        batch = flat.reshape(b, t, c, image_size, image_size).permute(
            0, 2, 1, 3, 4)

    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
        std_t = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
        batch = (batch - mean_t) / std_t
    return batch


def run_clip_epoch(model, loader, loss_fn, target_fn, optimizer=None,
                   device="cpu", image_size=112, mean=None, std=None):
    """One pass over CLIPS. Mirrors run_epoch; the batch shape is the only
    difference, and it is isolated in prepare_clip_batch."""
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    probs, targets = [], []

    with torch.set_grad_enabled(training):
        for clips, tools, task in loader:
            batch = prepare_clip_batch(clips.numpy(), device, image_size,
                                       mean, std)
            target = target_fn(tools, task).to(device)
            logits = model(batch)
            loss = loss_fn(logits, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += float(loss) * len(batch)
            count += len(batch)
            probs.append(logits.detach().float().cpu().numpy())
            targets.append(target.detach().float().cpu().numpy())

    if not count:
        raise ValueError(
            "run_clip_epoch saw zero batches. A silent empty loader reports "
            "loss 0.0 and looks like a converged run.")
    # Same dict shape as run_epoch, so callers and the metric code downstream
    # do not have to care which one produced the stats.
    return {"loss": total / count,
            "probs": np.concatenate(probs),
            "targets": np.concatenate(targets)}


def run_epoch(model, loader, loss_fn, target_fn, optimizer=None, device="cpu",
              image_size=None):
    """One pass. Training when `optimizer` is given, evaluation otherwise.

    `target_fn(tools, task)` chooses the label. It is a required argument, not
    inferred from the loss: all three models share this loader but consume
    different labels, and guessing from the loss class would hand the action
    model the 12-way tool vector while still training and reporting happily.
    """
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    probs, targets = [], []

    with torch.set_grad_enabled(training):
        for frames, tools, task in loader:
            batch = prepare_batch(frames.numpy(), device, image_size)
            target = target_fn(tools, task).to(device)
            logits = model(batch)
            loss = loss_fn(logits, target)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total += float(loss) * batch.shape[0]
            count += batch.shape[0]
            probs.append(logits.detach().float().cpu().numpy())
            targets.append(target.detach().cpu().numpy())

    if count == 0:
        raise ValueError(
            "the loader produced zero batches. Training would report a "
            "meaningless loss of 0.0 and exit successfully.")
    return {"loss": total / count,
            "probs": np.concatenate(probs),
            "targets": np.concatenate(targets)}


def save_checkpoint(path, model, meta):
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path, model):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return payload["meta"]
