"""Direct tests for surgvu/flow.py.

Moving_fraction is the primary discriminator (camera moves every pixel;
instrument moves few). Coherence is secondary. Both are tested by
CONSTRUCTION: a pure translation of the whole frame is what a camera pan
looks like (high moving_fraction, high coherence); independent motion of a
small patch against a static background is what an instrument looks like
(low moving_fraction, medium coherence).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.flow import flow_features  # noqa: E402


def _texture(size=192, seed=0):
    """A textured frame. Flow needs gradient; a flat field has no solution."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    return np.repeat(base[:, :, None], 3, axis=2)


def test_still_frames_have_near_zero_magnitude():
    frame = _texture()
    out = flow_features(frame, frame.copy())
    assert out["mag_mean"] < 0.1


def test_global_translation_is_coherent():
    """A camera pan: every pixel moves the same way."""
    frame = _texture()
    shifted = np.roll(frame, 6, axis=1)
    out = flow_features(frame, shifted)
    assert out["mag_mean"] > 1.0
    assert out["coherence"] > 0.8


def test_global_translation_has_high_moving_fraction():
    """A camera pan moves every pixel."""
    frame = _texture()
    shifted = np.roll(frame, 6, axis=1)
    out = flow_features(frame, shifted)
    assert out["moving_fraction"] > 0.9


def test_local_motion_has_low_moving_fraction():
    """An instrument moves few pixels; coverage is the primary discriminator."""
    frame = _texture()
    moved = frame.copy()
    patch = frame[40:90, 40:90].copy()
    moved[40:90, 60:110] = patch
    out = flow_features(frame, moved)
    assert out["moving_fraction"] < 0.3


def test_coherence_is_bounded():
    out = flow_features(_texture(), np.roll(_texture(), 4, axis=0))
    assert 0.0 <= out["coherence"] <= 1.0


def test_output_is_json_safe():
    import json
    out = flow_features(_texture(), np.roll(_texture(), 3, axis=1))
    json.loads(json.dumps(out, allow_nan=False))


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        flow_features(_texture(size=192), _texture(size=128))


def test_output_includes_version():
    """Version key is always present for downstream consumer disambiguation."""
    from surgvu.flow import FLOW_VERSION
    out = flow_features(_texture(), _texture())
    assert "version" in out
    assert out["version"] == FLOW_VERSION


def test_work_size_parameter_threaded_to_resize():
    """work_size must be threaded through to BOTH _to_gray call sites.

    Regression test for R5 fix: a parameter that is accepted but ignored is a
    trap for later callers. Verify that non-default work_size reaches
    cv2.resize at both frame-A and frame-B processing sites.
    """
    from unittest import mock
    import cv2

    frame = _texture()

    # Instrument cv2.resize to track what sizes it's called with
    resize_calls = []
    original_resize = cv2.resize

    def instrumented_resize(src, dsize, **kwargs):
        resize_calls.append(dsize)
        return original_resize(src, dsize, **kwargs)

    with mock.patch("cv2.resize", side_effect=instrumented_resize):
        # Call with non-default work_size
        non_default_size = 64
        out = flow_features(frame, frame, work_size=non_default_size)

    # Verify cv2.resize was called exactly twice (once for each _to_gray)
    assert len(resize_calls) == 2, f"Expected 2 resize calls, got {len(resize_calls)}"

    # Verify both calls used the requested size, not the default FLOW_SIZE (128)
    expected_size = (non_default_size, non_default_size)
    assert all(call == expected_size for call in resize_calls), \
        f"Expected all resize calls to use {expected_size}, got {resize_calls}"
