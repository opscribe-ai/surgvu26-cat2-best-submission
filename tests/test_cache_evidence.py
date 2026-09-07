"""Tests for the torch-free parts of scripts/cache_evidence.py.

Deliberately torch-free, mirroring tests/test_train_vlm.py: this login node
has no torch installed at all (confirmed: `import torch` fails), so any
module-scope import of it would fail every test below at collection, not
just the ones that exercise it. See scripts/cache_evidence.py's own module
docstring, "WHAT COULD NOT BE RUN OR TESTED HERE", for what is NOT exercised
here: loading any checkpoint, decoding real video frames, and every model
forward pass (the tools/task CNN heads, the YOLO detector, the variant
head, motion_v2). Those all live behind imports inside function bodies in
cache_evidence.py's "TORCH-TOUCHING" section, exercised only by
condor/cache_evidence.sub on a real GPU node.

`resolve_window_video`/`frame_index_range` (imported from build_qa_pairs,
itself torch-free at module scope) ARE exercised here for real, including
against a genuine tiny .mp4 written with cv2 -- cv2 IS installed on this
login node (confirmed: cv2.__version__ == "4.13.0"), unlike torch.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_qa_pairs import frame_index_range, resolve_window_video  # noqa: E402

from cache_evidence import (                                        # noqa: E402
    build_record, enumerate_distinct_windows, load_cached_keys,
    verify_heldout_excluded, window_key,
)
from train_vlm import load_case_universe, verify_manifest_clean     # noqa: E402


# ============================================================================
# enumerate_distinct_windows: the dedup this whole cache is built around
# ============================================================================


def test_enumerate_distinct_windows_collapses_shared_records():
    """Many QA records share one 30s window -- tool presence, task, organ,
    count, ... are all asked about the SAME clip. Breaks if: the grouping
    key stops including all four of case/part/t_start/t_stop (e.g. drops
    `part`), which would silently merge two DIFFERENT windows that happen
    to share a case and timestamp across parts."""
    records = [
        {"case": "case_000", "part": "1.0", "t_start": 10.0, "t_stop": 40.0,
         "question": "q1"},
        {"case": "case_000", "part": "1.0", "t_start": 10.0, "t_stop": 40.0,
         "question": "q2"},
        {"case": "case_000", "part": "1.0", "t_start": 40.0, "t_stop": 70.0,
         "question": "q3"},
        {"case": "case_001", "part": "1.0", "t_start": 10.0, "t_stop": 40.0,
         "question": "q4"},
    ]
    windows = enumerate_distinct_windows(records)
    keys = {(w["case"], w["part"], w["t_start"], w["t_stop"]) for w in windows}
    assert len(windows) == 3
    assert ("case_000", "1.0", 10.0, 40.0) in keys
    assert ("case_000", "1.0", 40.0, 70.0) in keys
    assert ("case_001", "1.0", 10.0, 40.0) in keys


def test_enumerate_distinct_windows_is_deterministically_ordered():
    """Breaks if: the sort key is dropped (dict/set iteration order is not
    guaranteed stable across runs for this), which would make a resumed
    job's log output -- and the order windows are attempted in -- vary run
    to run for no reason."""
    records = [
        {"case": "case_002", "part": "1.0", "t_start": 0.0, "t_stop": 30.0},
        {"case": "case_001", "part": "1.0", "t_start": 30.0, "t_stop": 60.0},
        {"case": "case_001", "part": "1.0", "t_start": 0.0, "t_stop": 30.0},
    ]
    first = enumerate_distinct_windows(records)
    second = enumerate_distinct_windows(list(reversed(records)))
    assert first == second
    assert [w["case"] for w in first] == ["case_001", "case_001", "case_002"]


def test_enumerate_distinct_windows_on_the_real_manifest_shape():
    """The exact numbers this project's own manifest produces (23,355
    records / 15,087 distinct windows / 144 cases), reproduced here at
    small scale so a future change to the grouping key is caught without
    needing the real 23,355-record file on this login node."""
    records = [
        {"case": "case_000", "part": "1.0", "t_start": 1.0, "t_stop": 31.0},
        {"case": "case_000", "part": "1.0", "t_start": 1.0, "t_stop": 31.0},
        {"case": "case_000", "part": "1.0", "t_start": 1.0, "t_stop": 31.0},
    ]
    windows = enumerate_distinct_windows(records)
    assert len(windows) == 1


# ============================================================================
# R30: verify_heldout_excluded -- the complementary half of ruling R30's
# guard, on top of train_vlm's reused verify_manifest_clean
# ============================================================================


def test_verify_heldout_excluded_passes_on_a_clean_manifest():
    """The normal, healthy case: 11 configured heldout ids, none of them
    present among this manifest's cases -- `present_norm` here stands in
    for train_vlm.verify_manifest_clean's return value."""
    present_norm = {"case_001", "case_002"}
    heldout_norm = {"case_122", "case_123"}
    excluded = verify_heldout_excluded(present_norm, heldout_norm)
    assert excluded == heldout_norm


def test_verify_heldout_excluded_raises_when_exclusion_removed_zero():
    """Ruling R30's own failure shape: if EVERY configured heldout case
    also turns up 'present' (the exact bug that produced the variant
    head's contaminated 0.9011 run, one step upstream of this script),
    `heldout_norm - present_norm` is empty and this must refuse to call
    that a clean manifest.

    Breaks if: the `-` set-difference is replaced by `&` (intersection),
    which would raise on the HEALTHY case above instead of this one."""
    present_norm = {"case_001", "case_122", "case_123"}
    heldout_norm = {"case_122", "case_123"}
    with pytest.raises(RuntimeError, match="zero-exclusion"):
        verify_heldout_excluded(present_norm, heldout_norm)


def test_verify_heldout_excluded_raises_when_partially_overlapping():
    """A case122/case_122 spelling mismatch would leave one heldout id
    'confirmed absent' while the identically-graded case sits in
    `present_norm` under a different-looking key that was never even
    compared -- but `verify_manifest_clean` (called first, in
    process_manifest) already raises RuntimeError on that case before this
    function ever runs, so this only has to prove that a MIXED overlap
    still returns the correctly-narrowed set rather than raising."""
    present_norm = {"case_001", "case_122"}
    heldout_norm = {"case_122", "case_123"}
    excluded = verify_heldout_excluded(present_norm, heldout_norm)
    assert excluded == {"case_123"}


def test_verify_heldout_excluded_composes_with_train_vlm_reused_guard():
    """End-to-end: the same two-guard sequence process_manifest actually
    runs, using train_vlm's own load_case_universe/verify_manifest_clean
    unmodified plus this script's own verify_heldout_excluded, against a
    manifest that is genuinely clean."""
    train_norm = {"case_001", "case_002"}
    val_norm = {"case_003"}
    heldout_norm = {"case_122", "case_123"}
    records = [{"case": "case_001"}, {"case": "case2"}, {"case": "case_003"}]
    present_norm = verify_manifest_clean(records, train_norm, val_norm, heldout_norm)
    excluded = verify_heldout_excluded(present_norm, heldout_norm)
    assert excluded == heldout_norm


def test_verify_manifest_clean_still_raises_before_reaching_this_guard():
    """A leaked heldout case must never reach verify_heldout_excluded at
    all -- verify_manifest_clean (reused, unmodified) raises first. This
    pins the ORDER process_manifest calls the two guards in."""
    train_norm = {"case_001"}
    val_norm = set()
    heldout_norm = {"case_122"}
    records = [{"case": "case_001"}, {"case": "case122"}]
    with pytest.raises(RuntimeError, match="HELDOUT LEAKAGE"):
        verify_manifest_clean(records, train_norm, val_norm, heldout_norm)


def test_load_case_universe_reads_the_real_splits_file_and_finds_11_heldout():
    """Sanity check against the real config/splits_v2.json this project
    ships -- 115 train / 29 val / 11 heldout, matching the manifest's own
    144 = 115 + 29 distinct cases (verified separately against the real
    manifest file, not reproduced here)."""
    splits_path = Path(__file__).resolve().parents[1] / "config" / "splits_v2.json"
    train_norm, val_norm, heldout_norm = load_case_universe(splits_path)
    assert len(train_norm) == 115
    assert len(val_norm) == 29
    assert len(heldout_norm) == 11
    assert not (train_norm & val_norm & heldout_norm)


# ============================================================================
# window_key / load_cached_keys: resume bookkeeping
# ============================================================================


def test_window_key_is_stable_across_a_json_round_trip():
    """Breaks if: t_start/t_stop are compared as strings instead of floats
    (a manifest's `2163.3008689999997` and a re-serialised
    `2163.3008689999997` must compare equal after going through
    json.dumps/json.loads, which is exactly what load_cached_keys relies
    on for resume to work at all)."""
    original = window_key("case_000", "1.0", 2163.3008689999997, 2193.3008689999997)
    round_tripped_case = json.loads(json.dumps("case_000"))
    round_tripped_t_start = json.loads(json.dumps(2163.3008689999997))
    round_tripped_t_stop = json.loads(json.dumps(2193.3008689999997))
    again = window_key(round_tripped_case, "1.0", round_tripped_t_start,
                       round_tripped_t_stop)
    assert original == again


def test_load_cached_keys_returns_empty_set_for_a_missing_file(tmp_path):
    assert load_cached_keys(tmp_path / "does_not_exist.jsonl") == set()


def test_load_cached_keys_reads_back_exactly_what_build_record_wrote(tmp_path):
    """The resume contract end to end: a record this script would itself
    write, read back by load_cached_keys, must be recognised as the SAME
    window."""
    out_path = tmp_path / "evidence_cache.jsonl"
    record = build_record("case_000", "1.0", 2163.3008689999997,
                          2193.3008689999997, evidence={"tools_present": []})
    out_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    keys = load_cached_keys(out_path)
    assert window_key("case_000", "1.0", 2163.3008689999997,
                      2193.3008689999997) in keys


def test_load_cached_keys_skips_a_malformed_trailing_line(tmp_path):
    """Breaks if: a partial write from a job killed mid-line (no trailing
    newline, half a JSON object) raises instead of being skipped -- a
    resumed job must still recover the windows it DID finish."""
    out_path = tmp_path / "evidence_cache.jsonl"
    good = build_record("case_000", "1.0", 0.0, 30.0, evidence={})
    out_path.write_text(json.dumps(good) + "\n" + '{"case": "case_001", "par',
                        encoding="utf-8")
    keys = load_cached_keys(out_path)
    assert keys == {window_key("case_000", "1.0", 0.0, 30.0)}


def test_load_cached_keys_skips_blank_lines(tmp_path):
    out_path = tmp_path / "evidence_cache.jsonl"
    record = build_record("case_000", "1.0", 0.0, 30.0, evidence={})
    out_path.write_text("\n" + json.dumps(record) + "\n\n", encoding="utf-8")
    assert load_cached_keys(out_path) == {window_key("case_000", "1.0", 0.0, 30.0)}


# ============================================================================
# build_record: the exact shape written per window
# ============================================================================


def test_build_record_has_exactly_the_five_documented_keys():
    """Breaks if: an extra key (e.g. a stray `question` carried over from
    the manifest record) leaks into the cache -- the whole point of this
    cache's shape is that it carries ONLY {case, part, t_start, t_stop,
    evidence}, nothing question-specific, since one window's evidence is
    shared across every QA record that names it."""
    record = build_record("case_000", "1.0", 10.0, 40.0, evidence={"task_top": "x"})
    assert set(record.keys()) == {"case", "part", "t_start", "t_stop", "evidence"}
    assert record["case"] == "case_000"
    assert record["part"] == "1.0"
    assert record["t_start"] == 10.0
    assert record["t_stop"] == 40.0


def test_build_record_passes_evidence_through_unchanged():
    """The whole point: evidence is never reshaped, relabelled or
    filtered here -- whatever shape `cache_one_window` (or, in a test,
    a stand-in) produces is exactly what is written, so
    evidence_vlm.build_sampling_prompt sees the identical dict shape
    `surgvu.perceive.clip_record` produces at serving time."""
    evidence = {
        "tools": {"needle driver": 0.91}, "tools_present": ["needle driver"],
        "task": {"suturing": 0.7}, "task_top": "suturing", "n_frames": 16,
        "motion_v2": {"version": 2}, "yolo": {"version": 1}, "variant": None,
    }
    record = build_record("case_000", "1.0", 10.0, 40.0, evidence)
    assert record["evidence"] is evidence


def test_build_record_serialises_to_json_and_back(tmp_path):
    """The record must actually be JSONL-writable -- a numpy scalar or
    other non-plain-python value slipping into `evidence` would raise
    inside json.dumps deep in a multi-hour run, exactly the failure mode
    surgvu.perceive.clip_record's own docstring warns about ('json.dumps
    refuses float32')."""
    record = build_record("case_000", "1.0", 10.0, 40.0,
                          evidence={"tools_present": ["needle driver"],
                                    "task_top": "suturing", "n_frames": 16})
    path = tmp_path / "one.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    reloaded = json.loads(path.read_text(encoding="utf-8").strip())
    assert reloaded == record


# ============================================================================
# R28: video resolution by (case, part), reused unmodified from
# build_qa_pairs -- exercised here against a REAL tiny .mp4 (cv2 is
# installed on this login node; torch is not).
# ============================================================================


def _write_tiny_video(path, n_frames=90, fps=30.0, size=64):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (size, size))
    for i in range(n_frames):
        writer.write(np.full((size, size, 3), i % 255, dtype=np.uint8))
    writer.release()


def test_resolve_window_video_resolves_by_case_and_part_from_a_real_file(tmp_path):
    """R28: the filename is built directly from (case, part) -- never by
    scanning a case's directory and guessing which file covers a
    timestamp. A genuinely decodable tiny .mp4, written with cv2, so
    `info.total`/`info.fps` come from real file metadata rather than a
    hand-built stand-in."""
    video_root = tmp_path / "videos"
    video_path = video_root / "case_000" / "case_000_video_part_001.mp4"
    _write_tiny_video(video_path, n_frames=90, fps=30.0)

    info, reason = resolve_window_video(video_root, "case_000", "1.0", {})
    assert reason is None
    assert info is not None
    assert info.path == video_path
    assert info.total == 90
    assert info.fps == pytest.approx(30.0, rel=0.05)


def test_resolve_window_video_never_falls_back_to_a_different_part(tmp_path):
    """R28's whole point: only part 1's file exists; a window that names
    part 2 must resolve to 'video_missing', never silently pair its
    timestamp with part 1's pixels."""
    video_root = tmp_path / "videos"
    _write_tiny_video(video_root / "case_000" / "case_000_video_part_001.mp4")

    info, reason = resolve_window_video(video_root, "case_000", "2.0", {})
    assert info is None
    assert reason == "video_missing"


def test_resolve_window_video_reports_unreadable_not_missing(tmp_path):
    """A file that EXISTS but is not a real video (e.g. a truncated
    download, or a placeholder) must be told apart from one that is
    absent -- both are drop reasons, but they point at different bugs."""
    video_root = tmp_path / "videos"
    bad_path = video_root / "case_000" / "case_000_video_part_001.mp4"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_bytes(b"not a real video file")

    info, reason = resolve_window_video(video_root, "case_000", "1.0", {})
    assert info is None
    assert reason == "video_unreadable"


def test_frame_index_range_matches_a_real_video_decoded_length(tmp_path):
    """The exact (first, last) span a 30s-at-30fps window resolves to
    against a real 90-frame file, and that an out-of-bounds window (a
    manifest timestamp beyond what this particular file actually
    contains) reports None rather than a clamped, silently-shorter span."""
    video_root = tmp_path / "videos"
    video_path = video_root / "case_000" / "case_000_video_part_001.mp4"
    _write_tiny_video(video_path, n_frames=90, fps=30.0)
    info, reason = resolve_window_video(video_root, "case_000", "1.0", {})
    assert reason is None

    span = frame_index_range(0.0, 3.0, info.fps, info.total)
    assert span == (0, 89)

    out_of_range = frame_index_range(100.0, 130.0, info.fps, info.total)
    assert out_of_range is None
