"""Tests for the motion-v2 calibration objective.

The point of this file is that the objective is REAL -- a statistic that
cannot separate annotated task intervals from the gaps between them is not
measuring surgical activity, whatever its docstring claims. AUC is used
rather than accuracy because the classes are unbalanced and a threshold has
not been chosen yet.

Controller ruling R14: the fixture below writes the REAL tasks.csv header
(index, start_part, start_time, stop_part, stop_time, duration, taskname,
groundtruth_taskname, matched_description, case) -- verified directly against
/staging/groups/bhaskar_opscribe/surgvu/labels_cat2/SURGVU25_train_labels/
case_*/tasks.csv. An earlier version of this fixture wrote a simplified
start/stop header that does not exist in the real corpus, which is exactly
how the column-name mismatch this file now guards against was missed the
first time.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from calibrate_motion_v2 import (activity_labels, dump_offsets_ms,  # noqa: E402
                                separability)

_HEADER = ("index,start_part,start_time,stop_part,stop_time,duration,"
          "taskname,groundtruth_taskname,matched_description,case")


def _tasks_csv(tmp_path, rows=None):
    """A tasks.csv fixture using the REAL column names, not a simplified
    start/stop schema. Each row in `rows` is
    (start_part, start_time, stop_part, stop_time, taskname).
    """
    if rows is None:
        rows = [
            ("1.0", 10.0, "1.0", 20.0, "suturing"),
            ("1.0", 40.0, "1.0", 50.0, "dissection"),
        ]
    path = tmp_path / "tasks.csv"
    lines = [_HEADER]
    for index, (start_part, start_time, stop_part, stop_time, taskname) in \
            enumerate(rows):
        duration = stop_time - start_time
        lines.append(
            "%d,%s,%s,%s,%s,%s,%s,%s,case_test"
            % (index, start_part, start_time, stop_part, stop_time, duration,
               taskname, taskname))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_timestamp_inside_an_interval_is_active(tmp_path):
    labels = activity_labels(_tasks_csv(tmp_path), [15.0])
    assert labels == [True]


def test_timestamp_between_intervals_is_idle(tmp_path):
    labels = activity_labels(_tasks_csv(tmp_path), [30.0])
    assert labels == [False]


def test_interval_boundaries_are_inclusive(tmp_path):
    assert activity_labels(_tasks_csv(tmp_path), [10.0, 20.0]) == [True, True]


def test_part_none_considers_every_part(tmp_path):
    """Most cases have a single video part; part=None (the default) must
    keep working exactly as it did before the `part` argument existed."""
    assert activity_labels(_tasks_csv(tmp_path), [15.0]) == [True]


def test_part_argument_filters_by_part(tmp_path):
    rows = [
        ("1.0", 10.0, "1.0", 20.0, "suturing"),   # part 1: active 10-20
        ("2.0", 30.0, "2.0", 40.0, "dissection"),  # part 2: active 30-40
    ]
    path = _tasks_csv(tmp_path, rows)
    assert activity_labels(path, [15.0], part="1.0") == [True]
    assert activity_labels(path, [15.0], part="2.0") == [False]
    assert activity_labels(path, [35.0], part="2.0") == [True]
    assert activity_labels(path, [35.0], part="1.0") == [False]
    # part=None sees both regardless of which part a row belongs to.
    assert activity_labels(path, [15.0, 35.0]) == [True, True]


def test_part_argument_accepts_any_normalize_part_spelling(tmp_path):
    """'1', 1, and '1.0' must all mean the same part."""
    path = _tasks_csv(tmp_path, [("1.0", 10.0, "1.0", 20.0, "suturing")])
    assert activity_labels(path, [15.0], part="1") == [True]
    assert activity_labels(path, [15.0], part=1) == [True]
    assert activity_labels(path, [15.0], part="1.0") == [True]


def test_row_spanning_a_part_boundary_is_dropped(tmp_path):
    """start_part != stop_part means timestamps reset mid-row; it is not a
    usable interval regardless of which part is requested."""
    path = _tasks_csv(tmp_path, [("1.0", 500.0, "2.0", 50.0, "spanning")])
    assert activity_labels(path, [500.0], part="1.0") == [False]
    assert activity_labels(path, [500.0]) == [False]


def test_perfect_separation_scores_one():
    assert separability([0.1, 0.2, 5.0, 6.0],
                        [False, False, True, True]) == pytest.approx(1.0)


def test_reversed_separation_scores_zero():
    assert separability([5.0, 6.0, 0.1, 0.2],
                        [False, False, True, True]) == pytest.approx(0.0)


def test_no_signal_scores_one_half():
    assert separability([1.0, 1.0, 1.0, 1.0],
                        [False, True, False, True]) == pytest.approx(0.5)


def test_none_values_are_dropped_not_counted_as_zero():
    """An unavailable measurement must not be scored as a quiet one."""
    assert separability([None, 0.1, 5.0], [True, False, True]) == pytest.approx(1.0)


def test_refuses_a_single_class():
    with pytest.raises(ValueError):
        separability([1.0, 2.0], [True, True])


# ------------------------------------------------- R16: offsets read, not asserted
#
# The written config used to carry a hand-typed [133, 400, 1200] regardless
# of what the dump was actually produced with. dump_offsets_ms() reads it
# from the dump records themselves instead, so config/motion_v2.json can
# never assert offsets the calibrator was not actually handed.

def test_dump_offsets_ms_reads_the_common_value():
    records = [{"offsets_ms": [133, 400, 1200]},
              {"offsets_ms": [133, 400, 1200]}]
    assert dump_offsets_ms(records) == [133, 400, 1200]


def test_dump_offsets_ms_rejects_disagreeing_records():
    """A dump mixing offsets means the vectors pooled into one threshold are
    not comparable -- that must be a loud error, not a silently chosen one
    of the two, and never a fallback to a hardcoded literal."""
    records = [{"offsets_ms": [133, 400, 1200]},
              {"offsets_ms": [67, 400, 1200]}]
    with pytest.raises(ValueError, match="disagree"):
        dump_offsets_ms(records)


def test_dump_offsets_ms_rejects_records_missing_the_field():
    """A dump produced before dump_motion_v2.py recorded its own offsets
    carries no provenance at all; guessing it matched the current default
    would be exactly the bug R16 fixes, so this must raise rather than
    substitute a literal."""
    with pytest.raises(ValueError, match="offsets_ms"):
        dump_offsets_ms([{"case": "case_x", "part": "1.0", "t": 1.0,
                          "vector": {}}])


def test_dump_offsets_ms_rejects_an_empty_dump():
    with pytest.raises(ValueError):
        dump_offsets_ms([])
