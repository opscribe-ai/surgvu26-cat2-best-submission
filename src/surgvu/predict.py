"""Frame predictions -> clip predictions.

The graded unit is a 30-second clip. Installation state is constant across a
window by construction, so the 30 frames are 30 independent views of one
label and averaging them is a free variance reduction -- and specifically a
defence against the known blind spot, a tool briefly occluded in some frames.
"""
import numpy as np
import torch

from .train import prepare_batch


def aggregate_window(frame_probs):
    """(N_frames, N_classes) per-frame probabilities -> (N_classes,) mean."""
    frame_probs = np.asarray(frame_probs, dtype=np.float32)
    if frame_probs.ndim != 2:
        # A (C,) input is this function's own output shape, so feeding a
        # result back in is the easy mistake -- and numpy would not complain:
        # it would average across the CLASS axis and hand back one scalar
        # standing in for 12 tool probabilities. Shape, not just emptiness.
        raise ValueError(
            "frame_probs must be 2-D (n_frames, n_classes); got shape %r. "
            "A 1-D vector is already aggregated." % (frame_probs.shape,))
    if frame_probs.shape[0] == 0:
        raise ValueError("no frames to aggregate; a window must have frames")
    return frame_probs.mean(axis=0)


def predict_window_frames(model, frames, device, image_size, activation="sigmoid"):
    """(N_frames, N_classes) per-frame probabilities. No aggregation.

    Split out of `predict_window` so a caller that needs the frames -- an
    ensemble averaging two models BEFORE reducing, or a non-mean aggregator --
    can have them without reimplementing the forward pass. `predict_window`
    is now this plus `aggregate_window`, so the serving path that existed
    before this split is byte-identical to what it was.

    Averaging two models at the FRAME level rather than after aggregation is
    not equivalent for any aggregator except the mean, and the measured best
    aggregator is not the mean. Reducing each model separately and then
    averaging would take the top-5 frames of each model independently, which
    can be five different moments; averaging first asks both models about the
    same moments and then picks the best five, which is the question the
    ensemble is supposed to be answering.
    """
    model.eval()
    with torch.no_grad():
        batch = prepare_batch(frames, device, image_size)
        logits = model(batch)
        if activation == "sigmoid":
            probs = torch.sigmoid(logits)
        elif activation == "softmax":
            probs = torch.softmax(logits, dim=1)
        else:
            raise ValueError("unknown activation %r" % (activation,))
        return probs.float().cpu().numpy()


def predict_window(model, frames, device, image_size, activation="sigmoid"):
    """Per-class probabilities for one window's frame stack.

    `frames` is a numpy uint8 array of shape (N_frames, H, W, 3) in OpenCV
    BGR order -- exactly what a shard yields -- and is handed to
    `prepare_batch` unchanged. Note that `train.run_epoch` calls
    `prepare_batch(frames.numpy(), ...)` instead; that is not an
    inconsistency to copy, it is because its frames arrive from a DataLoader
    as torch tensors while a serving caller's arrive as arrays. A CPU torch
    uint8 tensor also happens to work here (`np.asarray` accepts it via the
    array protocol), but a CUDA tensor does not -- `np.asarray` raises on
    one. Move frames to the device by passing `device`, never by handing this
    function a tensor that is already there.

    `image_size` is REQUIRED, with no default, and must be whatever the
    checkpoint was trained at -- both training scripts record it in
    `meta["image_size"]`. It is forwarded to `prepare_batch`, which is where
    the resize happens. It has no default because the wrong value here is
    silent: shards hold 512x512 and both experts train at 384, so a forgotten
    argument would serve a model at a resolution it never saw and show up as
    a mediocre model rather than an error. Pass `None` to mean "no resize",
    but pass it on purpose.

    `activation` is explicit: the tool head is multi-label (sigmoid) and the
    task head is multi-class (softmax). Defaulting one of them silently would
    produce well-formed numbers that do not sum the way the caller assumes.
    """
    return aggregate_window(
        predict_window_frames(model, frames, device, image_size, activation))
