"""Tests for the v2 motion record.

The property that matters most is the one v1 already established and v2 must
not lose: an unavailable measurement is None, not zero. A missing flank is
not evidence that nothing moved, and a record that says 0.0 where it means
"unknown" will be averaged into a threshold as though it were a measurement.

R8 adds a ninth slot to the vector: flow_moving_fraction. Task 1's flow
module found that moving_fraction, not coherence, is the discriminator that
actually separates camera motion from instrument motion (12x separation vs
coherence's 1.33x) -- so it rides along with the other three flow slots,
None when unmeasured, a real float when a probe pair is available.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.flow import flow_features  # noqa: E402
from surgvu.motion import (MOTION_V2_VERSION, motion_record_v2,  # noqa: E402
                           motion_vector)


def _frame(value, size=96, seed=None):
    if seed is None:
        return np.full((size, size, 3), value, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)


def test_still_scene_reads_near_zero_micro():
    still = _frame(100)
    probe = {133: (still.copy(), still.copy()),
             400: (still.copy(), still.copy())}
    vector = motion_vector(still.copy(), still, still.copy(), probe)
    assert vector["micro_short"] == pytest.approx(0.0, abs=1e-5)


def test_moving_scene_reads_above_still():
    base = _frame(0, seed=1)
    moved = np.roll(base, 12, axis=1)
    probe = {133: (base.copy(), moved.copy())}
    active = motion_vector(base.copy(), base, moved, probe)

    still = _frame(100)
    quiet = motion_vector(still.copy(), still, still.copy(),
                          {133: (still.copy(), still.copy())})
    assert active["micro_short"] > quiet["micro_short"]


def test_unavailable_offset_is_none_not_zero():
    still = _frame(100)
    vector = motion_vector(still.copy(), still, still.copy(), {133: None})
    assert vector["micro_short"] is None


def test_missing_neighbour_leaves_macro_none():
    still = _frame(100)
    vector = motion_vector(None, still, None, {133: (still, still)})
    assert vector["macro_prev"] is None
    assert vector["macro_next"] is None


def test_record_carries_its_version():
    frames = np.stack([_frame(0, seed=i) for i in range(4)])
    probes = [{133: (frames[0], frames[1])} for _ in range(4)]
    record = motion_record_v2(frames, probes)
    assert record["version"] == MOTION_V2_VERSION
    assert record["version"] != 1        # must not collide with the v1 record


def test_record_is_valid_strict_json():
    frames = np.stack([_frame(0, seed=i) for i in range(4)])
    probes = [{133: None} for _ in range(4)]
    record = motion_record_v2(frames, probes)
    json.loads(json.dumps(record, allow_nan=False))


def test_summary_ignores_none_rather_than_counting_it_as_zero():
    frames = np.stack([_frame(0, seed=i) for i in range(4)])
    probes = [{133: None}, {133: None}, {133: None}, {133: None}]
    record = motion_record_v2(frames, probes)
    assert record["summary"]["micro_short"]["mean"] is None
    assert record["summary"]["micro_short"]["measured"] == 0


# --------------------------------------------------------------------------
# R8: moving_fraction is the primary camera-vs-tool discriminator, so it
# rides along in the vector like the other three flow slots.
# --------------------------------------------------------------------------

def test_flow_moving_fraction_present_and_populated_for_real_pair():
    still = _frame(100)
    base = _frame(0, seed=7)
    moved = np.roll(base, 10, axis=1)
    probe = {133: (base.copy(), moved.copy())}
    vector = motion_vector(still.copy(), still, still.copy(), probe)
    assert "flow_moving_fraction" in vector
    assert vector["flow_moving_fraction"] is not None
    assert isinstance(vector["flow_moving_fraction"], float)


def test_flow_moving_fraction_is_none_without_a_probe_pair():
    still = _frame(100)
    vector = motion_vector(still.copy(), still, still.copy(), {133: None})
    assert vector["flow_moving_fraction"] is None


# --------------------------------------------------------------------------
# Flow locks onto the SHORTEST AVAILABLE probe pair, not the first key in
# VECTOR_SLOTS order incidentally, and not the first key of the dict. Flow's
# small-displacement assumption breaks down by 1.2 s -- a tool can cross the
# whole frame -- so picking the wrong pair here is a silent correctness bug,
# not a cosmetic one. These tests pin the requirement itself rather than the
# incidental fact that VECTOR_SLOTS happens to be written in ascending order.
# --------------------------------------------------------------------------

def test_flow_uses_shortest_available_pair_when_several_are_present():
    """All three offsets have a real pair. The 133ms pair is two identical
    frames (near-zero flow); the 1200ms pair is a big shift (unmistakably
    high flow). If the flow-pair loop picked anything but the shortest, the
    vector's flow slots would read high instead of near-zero."""
    still = _frame(150)
    texture = _frame(0, seed=3)
    shifted = np.roll(texture, 20, axis=1)
    probe = {133: (still.copy(), still.copy()),
             400: (still.copy(), still.copy()),
             1200: (texture.copy(), shifted.copy())}

    vector = motion_vector(still.copy(), still, still.copy(), probe)
    expected_short = flow_features(still, still)
    expected_long = flow_features(texture, shifted)

    # Sanity: the two candidate pairs really are distinguishable.
    assert expected_long["mag_mean"] > 1.0
    assert expected_short["mag_mean"] < 0.1

    assert vector["flow_mag_mean"] == pytest.approx(
        expected_short["mag_mean"], abs=1e-6)
    assert vector["flow_moving_fraction"] == pytest.approx(
        expected_short["moving_fraction"], abs=1e-6)
    # The failure mode this guards against: picking the 1200ms pair instead.
    assert vector["flow_mag_mean"] != pytest.approx(
        expected_long["mag_mean"], abs=0.5)


def test_flow_falls_through_to_the_middle_offset_when_the_shortest_is_absent():
    """The shortest offset (133) is entirely absent from the probe entry --
    not even a None placeholder, just a missing key -- and the dict is built
    with the LONGER offset inserted first, so an implementation that iterated
    `probe_entry.items()`/`.keys()` instead of walking VECTOR_SLOTS in order
    would encounter the 1200ms pair before the 400ms one. The correct
    behaviour is to use the shortest AVAILABLE pair: 400ms, not 1200ms and
    not "whichever key happens to come first in the dict"."""
    still = _frame(150)
    texture = _frame(0, seed=3)
    shifted = np.roll(texture, 20, axis=1)
    # Insertion order deliberately puts the longer offset first.
    probe = {1200: (texture.copy(), shifted.copy()),
             400: (still.copy(), still.copy())}

    vector = motion_vector(still.copy(), still, still.copy(), probe)
    expected_mid = flow_features(still, still)
    expected_long = flow_features(texture, shifted)

    assert expected_long["mag_mean"] > 1.0
    assert expected_mid["mag_mean"] < 0.1

    assert vector["flow_mag_mean"] == pytest.approx(
        expected_mid["mag_mean"], abs=1e-6)
    assert vector["flow_mag_mean"] != pytest.approx(
        expected_long["mag_mean"], abs=0.5)
