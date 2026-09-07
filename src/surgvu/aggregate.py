"""Clip aggregators: (windows, frames, classes) -> (windows, classes).

Shared by the experiment 1-3 sweep and the EndoViT head sweep so the two
cannot drift. A conclusion of the form "EndoViT scores X and EfficientNet
scores Y" is only meaningful if both were reduced from per-frame
probabilities by the SAME function, and the shipped container's `mean` is
only one of several defensible choices.

The measured ranking on the shipped EfficientNet (splits_v2 val, clip-level
macro-F1, thresholds tuned on one case fold and scored on the other):

    top3      0.6889     trim20   0.6784
    top5      0.6847     q75      0.6762
    q90       0.6839     mean     0.6757   <- what the container ships
                         noisy_or 0.6272

`mean` is near the BOTTOM of the reasonable options. The order statistics win
because a tool present in only part of a window is averaged away by the mean
and survives a top-k -- and the training label is window-level installation
state, so a tool that is genuinely installed but visible in six frames of
thirty is a positive the mean cannot see.

`noisy_or` is included because it is the textbook answer for "present in any
frame" and it is the worst of the lot: 1 - prod(1 - p) saturates to 1.0 for
almost every class once there are thirty frames, which destroys the ordering
the threshold needs.
"""
import numpy as np


def _mean(probs):
    return probs.mean(axis=1)


def _max(probs):
    return probs.max(axis=1)


def _quantile(q):
    def aggregate(probs):
        return np.quantile(probs, q, axis=1)
    return aggregate


def _trimmed(fraction):
    """Mean after dropping the `fraction` lowest and highest frames.

    The defence `mean` is documented to provide -- robustness to a briefly
    occluded tool -- is actually what a trimmed mean provides; a plain mean
    is only robust to the LOW tail in proportion to how few frames it spans.
    """
    def aggregate(probs):
        frames = probs.shape[1]
        drop = int(np.floor(frames * fraction))
        if drop == 0:
            return probs.mean(axis=1)
        ordered = np.sort(probs, axis=1)
        return ordered[:, drop:frames - drop, :].mean(axis=1)
    return aggregate


def _topk_mean(k):
    """Mean of the k most confident frames. `max` is this with k=1."""
    def aggregate(probs):
        k_eff = min(k, probs.shape[1])
        ordered = np.sort(probs, axis=1)
        return ordered[:, -k_eff:, :].mean(axis=1)
    return aggregate


def _noisy_or(probs):
    """1 - prod(1 - p). Saturates hard at 30 frames; thresholds absorb it."""
    return 1.0 - np.prod(1.0 - probs, axis=1)


AGGREGATORS = {
    "mean": _mean,                       # the shipped behaviour
    "max": _max,
    "q75": _quantile(0.75),
    "q90": _quantile(0.90),
    "trim10": _trimmed(0.10),
    "trim20": _trimmed(0.20),
    "top3": _topk_mean(3),
    "top5": _topk_mean(5),
    "noisy_or": _noisy_or,
}



AGGREGATOR_NAMES = tuple(AGGREGATORS)
