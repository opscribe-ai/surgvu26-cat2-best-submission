"""Which frames of a clip to look at. Deliberately free of heavy imports.

This lives here rather than in `perceive` because `perceive` imports torch,
OpenCV and the model builders at module scope, and the frame-index rule is
needed by callers that have none of those:

  * `surgvu.vlm` carries a copy commented "mirrors perceive.sample_frame_
    indices, which cannot be imported here" -- a duplicate that can drift;
  * the v2 offline sweeps are pure numpy and run on a login node where torch
    is not installed at all.

One definition, importable from anywhere. `perceive` re-exports it so every
existing `from surgvu.perceive import sample_frame_indices` keeps working and
the serving path is unchanged.
"""


def sample_frame_indices(total, n_frames):
    """`n_frames` frame indices evenly spaced across a clip of `total` frames.

    Bin centres, not endpoints: frame 0 and frame `total - 1` are where fades,
    black leader frames and truncated final packets live, and at 16 samples
    those two would be an eighth of the evidence for the whole clip.

    Asking for more frames than the clip holds returns each frame once. The
    alternative -- padding by repeating frames -- would be double-counted by
    the mean in `aggregate_window` and silently weight one moment twice.
    """
    total = int(total)
    n_frames = int(n_frames)
    if total <= 0:
        raise ValueError("no frames to sample from; total was %r" % (total,))
    if n_frames <= 0:
        raise ValueError("must sample at least one frame; got %r" % (n_frames,))
    if total <= n_frames:
        return list(range(total))
    step = total / float(n_frames)
    return [int((i + 0.5) * step) for i in range(n_frames)]
