"""Direct tests for surgvu/motion.py.

WHY THIS FILE EXISTS. On 2026-08-16 the motion gate was opened, which made
`motion_record_from_bursts` the input to a rule that decides a SHIPPED answer.
At that point the module had no direct tests at all -- `tests/test_router.py`
exercised the router's accessors against hand-written dicts, so every property
of the computation itself was unasserted. The number was trusted because it
looked reasonable, which is the failure mode this project has paid for most.

These test the statistic, not the router: that it responds to motion, that it
is comparable across cases, that a missing measurement is excluded rather than
counted as stillness, and that the record is valid JSON under a strict parser.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.motion import (MOTION_VERSION, WORK_SIZE, macro_activity,   # noqa: E402
                           micro_activity, motion_record_from_bursts)


def _still(n, value=100, size=128):
    """`n` identical frames -- nothing moves."""
    return np.full((n, size, size, 3), value, dtype=np.uint8)


BASE = 30


def _moving(n, step=20, size=128):
    """`n` frames whose brightness marches, so every pair differs by `step`.

    REFUSES to wrap past 255. An earlier version took `% 256`, which made the
    final burst of a long stack read 226 instead of `step` and quietly broke
    the premise of any test comparing bursts to each other. A fixture that
    lies is worse than no fixture.
    """
    top = BASE + (n - 1) * step
    if top > 255:
        raise ValueError(
            "%d frames at step %d reaches %d and would wrap through 0, making "
            "the last pair read 255-ish instead of %d"% (n, step, top, step))
    frames = np.empty((n, size, size, 3), dtype=np.uint8)
    for index in range(n):
        frames[index] = np.uint8(BASE + index * step)
    return frames


def _strict(obj):
    """Round-trip through a parser that REJECTS NaN/Infinity, as JSON does."""
    def reject(constant):
        raise ValueError("not valid JSON: %s" % constant)
    return json.loads(json.dumps(obj), parse_constant=reject)


# --------------------------------------------------------------------------
# The statistic responds to motion, and its units mean something.
# --------------------------------------------------------------------------

def test_a_still_window_reads_zero():
    assert micro_activity(_still(12), 3) == pytest.approx(0.0)


def test_a_moving_window_reads_the_brightness_step():
    """Values are mean |difference| in 0-255 units, which is what makes a
    threshold fitted on one split meaningful on another."""
    activity = micro_activity(_moving(12, step=20), 3)
    assert activity == pytest.approx(20.0, abs=0.5)


def test_activity_rises_with_the_size_of_the_change():
    small = micro_activity(_moving(6, step=5), 3).mean()
    large = micro_activity(_moving(6, step=40), 3).mean()
    assert small < large


def test_micro_reports_one_value_per_burst_and_macro_one_per_gap():
    frames = _moving(12)
    assert micro_activity(frames, 3).shape == (4,)
    assert macro_activity(frames, 3).shape == (3,)


def test_micro_and_macro_are_not_the_same_measurement():
    """A window that is still WITHIN each burst but jumps BETWEEN them: the
    case the two-timescale design exists for. Micro must not see it."""
    frames = np.concatenate([_still(3, 10), _still(3, 200),
                             _still(3, 10), _still(3, 200)])
    assert micro_activity(frames, 3).mean() == pytest.approx(0.0)
    assert macro_activity(frames, 3).mean() > 100.0


def test_downsampling_does_not_change_the_scale():
    """A 512 frame and a 64 frame of the same content must read the same, or a
    threshold would silently depend on the serving resolution."""
    big = micro_activity(_moving(6, step=20, size=512), 3).mean()
    small = micro_activity(_moving(6, step=20, size=WORK_SIZE), 3).mean()
    assert big == pytest.approx(small, abs=0.5)


# --------------------------------------------------------------------------
# Ragged input is refused rather than silently mismeasured.
# --------------------------------------------------------------------------

def test_a_partial_burst_is_refused():
    """13 frames is not a whole number of 3-frame bursts. Differencing anyway
    would compare across a 1.9 s gap and call it a 67 ms motion."""
    with pytest.raises(ValueError, match="whole number"):
        micro_activity(_moving(13, step=10), 3)


def test_a_batch_axis_is_refused():
    with pytest.raises(ValueError, match="N, H, W, 3"):
        micro_activity(np.zeros((2, 6, 64, 64, 3), dtype=np.uint8), 3)


def test_no_frames_is_refused():
    with pytest.raises(ValueError, match="no frames"):
        micro_activity(np.zeros((0, 64, 64, 3), dtype=np.uint8), 3)


def test_a_single_frame_burst_has_no_motion_to_measure():
    with pytest.raises(ValueError, match="no within-burst motion"):
        micro_activity(_moving(6), 1)


def test_one_burst_has_no_across_burst_change():
    with pytest.raises(ValueError, match="no across-burst change"):
        macro_activity(_moving(3), 3)


# --------------------------------------------------------------------------
# The serving record: a missing measurement is not a zero.
# --------------------------------------------------------------------------

def test_the_record_matches_the_array_functions_on_the_same_frames():
    """The serving path computes per burst; the training path computes over
    the whole stack. They must agree, or a threshold fitted on one is wrong on
    the other -- which is precisely the mistake that produced 2.512."""
    frames = _moving(12, step=17)
    bursts = [frames[i * 3:(i + 1) * 3] for i in range(4)]
    centres = frames[1::3]
    record = motion_record_from_bursts(bursts, centres)

    assert record["micro"]["per_burst"] == pytest.approx(
        [float(v) for v in micro_activity(frames, 3)], abs=1e-4)
    assert record["macro"]["per_gap"] == pytest.approx(
        [float(v) for v in macro_activity(frames, 3)], abs=1e-4)


def test_a_missing_burst_is_excluded_not_counted_as_stillness():
    """THE CENTRAL CLAIM of the None contract. A failed read must not drag the
    mean toward zero, because a clip whose flanks failed would then look
    static and the gate would answer No on absent evidence."""
    moving = [b for b in _moving(9, step=25).reshape(3, 3, 128, 128, 3)]
    full = motion_record_from_bursts(moving)
    holed = motion_record_from_bursts([moving[0], None, moving[2]])

    assert holed["bursts"] == 3
    assert holed["bursts_measured"] == 2
    assert holed["micro"]["mean"] == pytest.approx(full["micro"]["mean"],
                                                   abs=1e-4)
    assert len(holed["micro"]["per_burst"]) == 2
    # What the contract is protecting against: had the missing burst been
    # counted as a still 0.0, the window would read a third less active, and
    # near the threshold that is the difference between Yes and No.
    counted_as_still = (full["micro"]["mean"] * 2) / 3.0
    assert counted_as_still < holed["micro"]["mean"] - 1.0


def test_an_entirely_unmeasurable_window_reports_none_not_zero():
    """None means "cannot say", and router.scene_is_static must receive that
    rather than a 0.0 that would read as maximally static."""
    record = motion_record_from_bursts([None, None, None])
    assert record["bursts"] == 3
    assert record["bursts_measured"] == 0
    assert record["micro"]["mean"] is None
    assert record["micro"]["per_burst"] == []


def test_a_one_frame_burst_is_unmeasurable_rather_than_nan():
    """Regression, 2026-08-16. A burst of one frame differences to an empty
    array whose mean is NaN, and NaN is NOT valid JSON: a strict parser
    rejects the entire response, so the failure would arrive as an unreadable
    answer rather than a wrong one. Reachable via per_burst=1."""
    single = np.zeros((1, 64, 64, 3), dtype=np.uint8)
    record = motion_record_from_bursts([single, single])

    assert record["bursts_measured"] == 0
    assert record["micro"]["mean"] is None
    _strict(record)


def test_the_record_is_json_native_under_a_strict_parser():
    """numpy floats survive json.dumps in some versions and not others, and
    the response file is parsed by the grader, not by us."""
    frames = _moving(12)
    bursts = [frames[i * 3:(i + 1) * 3] for i in range(4)]
    record = motion_record_from_bursts(bursts, frames[1::3])

    assert _strict(record) == record
    assert record["version"] == MOTION_VERSION
    for block in ("micro", "macro"):
        for key in ("mean", "max", "std"):
            assert isinstance(record[block][key], float)


def test_macro_is_skipped_when_there_is_only_one_centre():
    record = motion_record_from_bursts([_moving(3)], _moving(1))
    assert record["macro"]["mean"] is None
    assert record["micro"]["mean"] is not None
