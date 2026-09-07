"""Tests for the motion-v2 source-video producer's torch-free logic.

`scripts/dump_motion_v2.py` needs `surgvu.perceive.decode_clip_multiscale` to
actually decode a video, and that import pulls in torch, which is not
installed on the machine this suite runs on. The torch-touching import is
therefore pushed inside `_decode_span`'s function body (never at module
scope), so everything ELSE in the module -- argument handling, stratified
span planning, and the record shape recovered from a decoded clip -- is
importable and testable here without touching perceive, torch, or any real
video file. What those functions do with real 5-hour source videos is
exercised only by the Condor job; this file is what verifies the sampling
logic is not the reason that job would produce a useless (single-class,
or misaligned-timestamp) dump.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dump_motion_v2 import (                                  # noqa: E402
    OFFSETS_MS, _anchor_records, _budget_requests, _complement,
    _merge_intervals, find_case_videos, plan_span_requests, resolve_cases,
)


# ---------------------------------------------------------------- resolve_cases

def test_resolve_cases_defaults_to_every_available_case():
    available = ["case_000", "case_001", "case_002"]
    assert resolve_cases(None, available) == available


def test_resolve_cases_int_limit_takes_the_first_n():
    available = ["case_000", "case_001", "case_002"]
    assert resolve_cases("2", available) == ["case_000", "case_001"]


def test_resolve_cases_explicit_list_preserves_the_requested_order():
    available = ["case_000", "case_001", "case_002"]
    assert resolve_cases("case_002,case_000", available) == ["case_002", "case_000"]


def test_resolve_cases_rejects_a_case_not_on_disk():
    with pytest.raises(ValueError):
        resolve_cases("case_999", ["case_000"])


def test_resolve_cases_rejects_a_non_positive_limit():
    with pytest.raises(ValueError):
        resolve_cases("0", ["case_000"])


# --------------------------------------------------- interval merge/complement

def test_merge_intervals_joins_overlapping_spans():
    assert _merge_intervals([(10, 20), (15, 25), (40, 50)]) == [(10, 25), (40, 50)]


def test_merge_intervals_drops_empty_or_inverted_spans():
    assert _merge_intervals([(10, 10), (20, 15)]) == []


def test_complement_covers_before_between_and_after():
    merged = [(10, 25), (40, 50)]
    assert _complement(merged, 100.0) == [(0.0, 10), (25, 40), (50, 100.0)]


def test_complement_of_no_spans_is_the_whole_range():
    assert _complement([], 100.0) == [(0.0, 100.0)]


def test_complement_of_full_coverage_is_empty():
    assert _complement([(0.0, 100.0)], 100.0) == []


# ------------------------------------------------------------- span planning

def test_plan_span_requests_covers_the_full_budget():
    import random
    rng = random.Random(0)
    requests = plan_span_requests(1000.0, [(100.0, 200.0), (500.0, 900.0)],
                                  20, rng)
    assert sum(n for _, _, n in requests) == 20


def test_plan_span_requests_stays_inside_the_source_spans():
    import random
    rng = random.Random(1)
    intervals = [(100.0, 140.0), (600.0, 620.0)]
    duration = 1000.0
    requests = plan_span_requests(duration, intervals, 12, rng,
                                  window_seconds=8.0, batch_frames=16)
    active = _merge_intervals(intervals)
    idle = _complement(active, duration)
    allowed = active + idle
    for start, length, _ in requests:
        stop = start + length
        assert any(a_start - 1e-9 <= start and stop <= a_stop + 1e-9
                  for a_start, a_stop in allowed), (start, stop, allowed)


def test_plan_span_requests_falls_back_when_no_active_span_exists():
    """A case with zero annotated task time still gets its full budget --
    all of it idle -- rather than silently losing half the windows."""
    import random
    rng = random.Random(0)
    requests = plan_span_requests(500.0, [], 10, rng)
    assert sum(n for _, _, n in requests) == 10


def test_plan_span_requests_falls_back_when_no_idle_span_exists():
    """A case entirely covered by one task segment still gets its full
    budget, all of it active."""
    import random
    rng = random.Random(0)
    requests = plan_span_requests(500.0, [(0.0, 500.0)], 10, rng)
    assert sum(n for _, _, n in requests) == 10


def test_plan_span_requests_rejects_non_positive_duration():
    import random
    with pytest.raises(ValueError):
        plan_span_requests(0.0, [(0.0, 1.0)], 10, random.Random(0))


def test_plan_span_requests_rejects_non_positive_window_budget():
    import random
    with pytest.raises(ValueError):
        plan_span_requests(100.0, [(0.0, 1.0)], 0, random.Random(0))


def test_budget_requests_weights_by_span_length_not_count():
    """A single long span should usually win over many short ones -- checked
    as a distributional property over many draws, not a single sample."""
    import random
    rng = random.Random(0)
    long_span = (0.0, 1000.0)
    short_spans = [(2000.0 + i, 2000.0 + i + 0.1) for i in range(20)]
    picks = []
    for seed in range(50):
        requests = _budget_requests([long_span] + short_spans, 1,
                                    random.Random(seed), window_seconds=8.0,
                                    batch_frames=16)
        picks.append(requests[0][0] < 1000.0)
    assert sum(picks) > 40


# ---------------------------------------------------------------- find_case_videos

def test_find_case_videos_parses_the_real_naming_convention(tmp_path):
    case_dir = tmp_path / "case_056"
    case_dir.mkdir()
    (case_dir / "case_056_video_part_001.mp4").write_bytes(b"")
    (case_dir / "case_056_video_part_002.mp4").write_bytes(b"")
    videos = find_case_videos(tmp_path, "case_056")
    assert [part for part, _ in videos] == [1, 2]
    assert videos[0][1].name == "case_056_video_part_001.mp4"


def test_find_case_videos_returns_empty_for_a_missing_case_dir(tmp_path):
    assert find_case_videos(tmp_path, "case_999") == []


def test_find_case_videos_skips_a_file_with_no_readable_part(tmp_path):
    case_dir = tmp_path / "case_056"
    case_dir.mkdir()
    (case_dir / "case_056_video_part_001.mp4").write_bytes(b"")
    (case_dir / "case_056_video_part_zz.mp4").write_bytes(b"")
    videos = find_case_videos(tmp_path, "case_056")
    assert len(videos) == 1


# ------------------------------------------------------------- record shape
#
# _anchor_records(case, part, first, sample_span, fps, per_anchor): `first`
# is the frame index decode_clip_multiscale's `index_range` started at (an
# absolute index in the SOURCE file, ruling R15 -- not a temporary clip's own
# frame numbering, which an earlier draft of this module used), and
# `sample_span` is `last - first + 1` from that same range.

def test_anchor_records_default_offsets_come_from_surgvu_motion():
    """R16: OFFSETS_MS is imported from surgvu.motion.PROBE_OFFSETS_MS, not
    re-typed here, and every record carries the offsets it was ACTUALLY
    produced with -- so config/motion_v2.json can be fitted against the
    real value rather than a literal that could silently diverge from it."""
    from surgvu.motion import PROBE_OFFSETS_MS

    assert tuple(OFFSETS_MS) == PROBE_OFFSETS_MS

    vector = {key: 1.0 for key in
              ("micro_short", "micro_mid", "micro_long", "macro_prev",
               "macro_next", "flow_mag_mean", "flow_mag_p90",
               "flow_coherence", "flow_moving_fraction")}
    records = _anchor_records("case_x", "1.0", 0, 60, 60.0, [vector])
    assert tuple(records[0]["offsets_ms"]) == PROBE_OFFSETS_MS


def test_anchor_records_offsets_ms_reflects_what_was_actually_passed():
    """A span decoded with a non-default offsets_ms must record THAT value,
    not the module default -- otherwise a dump mixing calibration runs at
    different offsets would look uniform when it is not."""
    vector = {key: 1.0 for key in
              ("micro_short", "micro_mid", "micro_long", "macro_prev",
               "macro_next", "flow_mag_mean", "flow_mag_p90",
               "flow_coherence", "flow_moving_fraction")}
    records = _anchor_records("case_x", "1.0", 0, 60, 60.0, [vector],
                              offsets_ms=(50, 200, 900))
    assert records[0]["offsets_ms"] == [50, 200, 900]


def test_anchor_records_recovers_absolute_time_from_index_range_start():
    """4 anchors, 60fps, index_range starting at frame 600 (=10.0s) spanning
    240 frames: sample_frame_indices(240, 4) picks bin centres 30/90/150/210,
    so the absolute frames are 630/690/750/810 -> 10.5/11.5/12.5/13.5s."""
    vector = {key: 1.0 for key in
              ("micro_short", "micro_mid", "micro_long", "macro_prev",
               "macro_next", "flow_mag_mean", "flow_mag_p90",
               "flow_coherence", "flow_moving_fraction")}
    per_anchor = [dict(vector) for _ in range(4)]
    records = _anchor_records("case_x", "1.0", 600, 240, 60.0, per_anchor)
    assert [round(r["t"], 3) for r in records] == [10.5, 11.5, 12.5, 13.5]
    assert all(r["case"] == "case_x" for r in records)
    assert all(r["part"] == "1.0" for r in records)


def test_anchor_records_keeps_none_as_none_not_zero():
    vector = {key: None for key in
              ("micro_short", "micro_mid", "micro_long", "macro_prev",
               "macro_next", "flow_mag_mean", "flow_mag_p90",
               "flow_coherence", "flow_moving_fraction")}
    records = _anchor_records("case_x", "1.0", 0, 60, 60.0, [vector])
    assert records[0]["vector"]["micro_short"] is None


def test_anchor_records_are_strict_json_safe():
    """A numpy float32 leaking through must not survive as a numpy scalar --
    json.dumps would refuse it, and that failure belongs here, at the unit
    that builds the record, not three hours into a Condor job."""
    import json
    import numpy as np

    vector = {key: np.float32(1.5) for key in
              ("micro_short", "micro_mid", "micro_long", "macro_prev",
               "macro_next", "flow_mag_mean", "flow_mag_p90",
               "flow_coherence", "flow_moving_fraction")}
    records = _anchor_records("case_x", "1.0", 0, 60, 60.0, [vector])
    dumped = json.dumps(records)  # raises TypeError if a numpy scalar leaked
    reloaded = json.loads(dumped)
    assert reloaded[0]["vector"]["micro_short"] == pytest.approx(1.5)
    assert isinstance(records[0]["t"], float)
    assert type(records[0]["vector"]["micro_short"]) is float
