"""Dense optical flow, reduced to four scalars per frame pair.

WHY THIS EXISTS, stated against what it replaces. `surgvu/motion.py` measures
mean absolute frame difference. That statistic answers "did pixels change"
and cannot answer "did the CAMERA move or did an INSTRUMENT move" -- its own
docstring says so. A scope push changes every pixel and reads as high
activity; so does a dissection. The router cannot tell those apart from a
difference scalar, and the difference matters for exactly the questions the
motion gate was opened for.

Flow separates them without a model. THE PRIMARY DISCRIMINATOR is coverage: a
camera pan moves EVERY pixel with coherent vectors (moving_fraction ≈ 1.0,
coherence ≈ 1.0); an instrument moves FEW pixels that may well agree among
themselves (moving_fraction ≈ 0.08, coherence ≈ 0.75). Coherence alone is
not a sufficient discriminator (1.0 vs 0.75 is only 1.33x separation), but
moving_fraction gives 12x separation. Direction agreement remains informative
as a secondary signal.

MEASURED: global roll coherence=1.0 moving_fraction=1.0; local patch
coherence=0.75 moving_fraction=0.08.

CPU BY CONSTRUCTION. Grand Challenge may allocate no GPU at all, and evidence
that vanishes on half the draws is not evidence. Farneback at 128x128 is
milliseconds and needs nothing but OpenCV.
"""
import cv2
import numpy as np

#: Flow is computed at this resolution. Large enough that an instrument tip
#: spans several pixels between frames, small enough to be free. Larger sizes
#: mostly buy resolution on JPEG noise.
FLOW_SIZE = 128

#: A vector counts as agreeing with the field's dominant direction if it lies
#: within this many degrees of it. 30 degrees is wide enough to tolerate the
#: aperture-problem wobble a real pan produces and narrow enough that
#: independent motion falls outside it.
COHERENCE_DEGREES = 30.0

#: Vectors shorter than this carry no reliable direction -- the angle of a
#: near-zero vector is noise -- so they are excluded from the coherence
#: fraction rather than counted as disagreeing. A still frame therefore has
#: no opinion about coherence instead of a random one.
MIN_MAGNITUDE = 0.25

FLOW_VERSION = 1


def _to_gray(frame, size=FLOW_SIZE):
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[-1] == 3:
        gray = (0.299 * array[..., 2] + 0.587 * array[..., 1]
                + 0.114 * array[..., 0])
    elif array.ndim == 2:
        gray = array
    else:
        raise ValueError("expected (H, W) or (H, W, 3), got %r"
                         % (array.shape,))
    return cv2.resize(gray.astype(np.float32), (size, size),
                      interpolation=cv2.INTER_AREA)


def flow_features(frame_a, frame_b, work_size=FLOW_SIZE):
    """Four scalars describing the motion field between two frames.

    Returns {"version", "mag_mean", "mag_p90", "coherence", "moving_fraction"}.
    All scalar values are plain Python floats: this lands in a record written
    with json.dumps, which refuses float32.

    `version` is FLOW_VERSION and is always included for downstream consumers
    to disambiguate semantics if tuning parameters (MIN_MAGNITUDE,
    COHERENCE_DEGREES) change.

    `moving_fraction` is the primary discriminator between camera motion
    (high, ~1.0) and instrument motion (low, ~0.1). It is the fraction of
    flow vectors with magnitude >= MIN_MAGNITUDE.

    `coherence` is the fraction of MOVING vectors within COHERENCE_DEGREES of
    the field's dominant direction, where the dominant direction is the angle
    of the summed vector rather than a mean of angles -- averaging angles
    across the +/-pi wrap gives a direction no vector points in. It provides
    secondary discrimination (camera ~1.0, instrument ~0.75) and may be used
    to detect anomalies, but is insufficient alone.
    """
    a = np.asarray(frame_a)
    b = np.asarray(frame_b)
    if a.shape != b.shape:
        raise ValueError(
            "flow needs two frames of the same shape, got %r and %r. "
            "Resizing one to match would invent motion at the seams."
            % (a.shape, b.shape))
    gray_a, gray_b = _to_gray(a, work_size), _to_gray(b, work_size)
    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0)
    dx, dy = flow[..., 0], flow[..., 1]
    magnitude = np.sqrt(dx * dx + dy * dy)

    moving = magnitude >= MIN_MAGNITUDE
    if not moving.any():
        coherence = 0.0
    else:
        sum_x, sum_y = float(dx[moving].sum()), float(dy[moving].sum())
        if sum_x == 0.0 and sum_y == 0.0:
            coherence = 0.0
        else:
            norm = np.hypot(sum_x, sum_y)
            unit_x, unit_y = sum_x / norm, sum_y / norm
            # cos of the angle between each moving vector and the dominant
            # direction, via the normalised dot product.
            dot = (dx[moving] * unit_x + dy[moving] * unit_y) / magnitude[moving]
            coherence = float(
                (dot >= np.cos(np.deg2rad(COHERENCE_DEGREES))).mean())

    return {
        "version": FLOW_VERSION,
        "mag_mean": float(magnitude.mean()),
        "mag_p90": float(np.percentile(magnitude, 90)),
        "coherence": coherence,
        "moving_fraction": float(moving.mean()),
    }
