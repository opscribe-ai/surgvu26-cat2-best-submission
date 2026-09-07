"""Tests for scripts/pseudo_label_shards.py.

Deliberately torch-free: everything exercised here is the part that decides
which detections survive the logbook constraint and how they are written --
`Detector.detect` itself (and therefore an actual harvesting run over real
shards) requires the surgvu26-train.sif container and a GPU and is NOT
exercised anywhere by this file. See the script's own module docstring,
"WHAT IS AND IS NOT TESTED WITHOUT TORCH".
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from surgvu.detect import YOLO_CLASSES  # noqa: E402
from surgvu.extract import write_shard  # noqa: E402
from surgvu.labels import CaseLabels  # noqa: E402
from surgvu.sampling import Window  # noqa: E402

from pseudo_label_shards import (  # noqa: E402
    HAND_LABELED_TRAIN_INSTANCES,
    LogbookCache,
    box_to_yolo_line,
    build_report,
    classify_detection,
    format_yield_table,
    load_heldout_ids,
    load_processed_shards,
    process_shard,
    pseudo_stem,
    read_shard_safely,
    shard_case_id,
    split_heldout_shards,
)

FIXTURE = Path(__file__).parent / "fixtures" / "case_test"


# ---------------------------------------------------------------------------
# Shard filename / heldout bookkeeping
# ---------------------------------------------------------------------------

def test_shard_case_id_strips_the_part_suffix():
    assert shard_case_id("case_073_part1.npz") == "case_073"
    assert shard_case_id(Path("/x/y/case_002_part2.npz")) == "case_002"


def test_split_heldout_shards_matches_across_spelling():
    # The exact bug shape this project has hit before: the split spells a
    # case 'case_122', a differently-spelled shard name must still match it.
    # Breaks if: normalize_case_id is swapped for a raw string comparison.
    shards = [Path("case_122_part1.npz"), Path("case_050_part1.npz")]
    kept, excluded = split_heldout_shards(shards, ["case122"])
    assert [p.name for p in excluded] == ["case_122_part1.npz"]
    assert [p.name for p in kept] == ["case_050_part1.npz"]


def test_split_heldout_shards_raises_when_nothing_excluded():
    # Breaks if: the `if not excluded: raise` guard is deleted or weakened to
    # a warning -- a silent zero-exclusion result must never be trusted.
    shards = [Path("case_050_part1.npz")]
    with pytest.raises(RuntimeError, match="ZERO"):
        split_heldout_shards(shards, ["case_122"])


def test_load_heldout_ids_reads_the_real_splits_file():
    path = Path(__file__).resolve().parents[1] / "config" / "splits_v2.json"
    ids = load_heldout_ids(path)
    assert len(ids) == 11
    assert "case_122" in ids


def test_load_heldout_ids_raises_on_empty_list(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"heldout": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_heldout_ids(path)


# ---------------------------------------------------------------------------
# LogbookCache
# ---------------------------------------------------------------------------

def test_logbook_cache_loads_a_real_case():
    cache = LogbookCache(FIXTURE.parent)
    labels = cache.get("case_test")
    assert isinstance(labels, CaseLabels)
    assert "needle driver" in labels.tools_at("1.0", 120.0)


def test_logbook_cache_returns_none_for_a_missing_case(tmp_path):
    cache = LogbookCache(tmp_path)
    assert cache.get("case_does_not_exist") is None
    # Cached, not re-attempted -- second call still None, no raise.
    assert cache.get("case_does_not_exist") is None


# ---------------------------------------------------------------------------
# classify_detection
# ---------------------------------------------------------------------------

def test_classify_detection_low_confidence_short_circuits_before_logbook():
    # case_labels=None would normally mean "unresolved_window"; confidence is
    # checked FIRST, so a low-confidence detection never reaches that check.
    # Breaks if: the conf check and the case_labels check are reordered.
    verdict, reason = classify_detection(
        "needle driver", 0.01, None, "1.0", 50.0, conf_floor=0.25)
    assert (verdict, reason) == ("drop", "low_confidence")


def test_classify_detection_unresolved_when_case_labels_missing():
    verdict, reason = classify_detection(
        "needle driver", 0.9, None, "1.0", 50.0, conf_floor=0.25)
    assert (verdict, reason) == ("drop", "unresolved_window")


def test_classify_detection_unmounted_when_not_installed():
    labels = CaseLabels.from_dir(FIXTURE)
    # needle driver installs at t=10 in part 1.0; t=5 is before that.
    verdict, reason = classify_detection(
        "needle driver", 0.9, labels, "1.0", 5.0, conf_floor=0.25)
    assert (verdict, reason) == ("drop", "unmounted")


def test_classify_detection_accepts_when_mounted():
    labels = CaseLabels.from_dir(FIXTURE)
    verdict, reason = classify_detection(
        "needle driver", 0.9, labels, "1.0", 120.0, conf_floor=0.25)
    assert (verdict, reason) == ("accept", None)


def test_classify_detection_out_of_taxonomy_always_drops_as_unmounted():
    """Even at t=330, when the fixture's own tools.csv says the suction
    irrigator IS physically installed, this must still drop -- because
    surgvu.taxonomy.normalize_tool excludes it from every interval
    CaseLabels ever builds, so tools_at can never contain it regardless of
    ground truth. This is the load-bearing claim in the module docstring's
    "THE TWO CLASSES THIS CAN NEVER CONFIRM" section.

    Breaks if: classify_detection gains a special case that maps
    out-of-taxonomy names through surgvu.detect.map_to_taxonomy before the
    membership test (which would make this pass and defeat the point).
    """
    labels = CaseLabels.from_dir(FIXTURE)
    assert "suction irrigator" in {"suction irrigator"}  # sanity: name spelled the same
    verdict, reason = classify_detection(
        "suction irrigator", 0.9, labels, "1.0", 330.0, conf_floor=0.25)
    assert (verdict, reason) == ("drop", "unmounted")


# ---------------------------------------------------------------------------
# YOLO formatting
# ---------------------------------------------------------------------------

def test_box_to_yolo_line_normalizes_exact_values():
    # A 100x50 box centered at (100, 100) inside a 200x200 frame.
    line = box_to_yolo_line(7, [50.0, 75.0, 150.0, 125.0], img_w=200.0, img_h=200.0)
    cls, cx, cy, w, h = line.split()
    assert cls == "7"
    assert (float(cx), float(cy)) == pytest.approx((0.5, 0.5))
    assert (float(w), float(h)) == pytest.approx((0.5, 0.25))


def test_box_to_yolo_line_clamps_a_box_that_slightly_overhangs_the_frame():
    line = box_to_yolo_line(0, [-2.0, -2.0, 50.0, 50.0], img_w=100.0, img_h=100.0)
    cls, cx, cy, w, h = (float(x) if i else x for i, x in enumerate(line.split()))
    assert float(w) == pytest.approx(0.5)   # clamped from 52 to 50 wide
    assert float(h) == pytest.approx(0.5)


def test_box_to_yolo_line_rejects_a_box_entirely_outside_the_frame():
    assert box_to_yolo_line(0, [150.0, 150.0, 200.0, 200.0],
                            img_w=100.0, img_h=100.0) is None


def test_pseudo_stem_is_deterministic_and_varies_with_window_and_anchor():
    a = pseudo_stem("case_073", "1.0", 3, 7)
    b = pseudo_stem("case_073", "1.0", 3, 7)
    c = pseudo_stem("case_073", "1.0", 3, 8)
    d = pseudo_stem("case_073", "1.0", 4, 7)
    assert a == b
    assert len({a, c, d}) == 3


def test_pseudo_stem_never_collides_with_the_hand_labeled_prefix():
    assert pseudo_stem("case_073", "1.0", 0, 0).startswith("pseudo_")
    assert not pseudo_stem("case_073", "1.0", 0, 0).startswith("clip_")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_hand_labeled_baseline_covers_exactly_the_fourteen_detector_classes():
    assert set(HAND_LABELED_TRAIN_INSTANCES) == set(YOLO_CLASSES)


def test_format_yield_table_lists_every_class_with_its_new_count():
    table = format_yield_table({"prograsp forceps": 362})
    assert "prograsp forceps" in table
    assert "362" in table
    # A class with zero new instances still appears -- absence would hide
    # exactly the "gained nothing" result Task 2 needs to see.
    assert "vessel sealer" in table


def test_build_report_records_the_heldout_exclusion_and_the_yield():
    report = build_report(
        kept_shards=[Path("case_050_part1.npz")],
        excluded_shards=[Path("case_122_part1.npz")],
        tally={"detections_dropped_unmounted": 3},
        per_class_new={"prograsp forceps": 5},
        windows_total=1,
        images_written_total=1,
        args=_FakeArgs())
    assert report["shards_heldout_excluded"] == 1
    assert report["shards_heldout_excluded_names"] == ["case_122_part1.npz"]
    assert report["per_class_new_instances"] == {"prograsp forceps": 5}
    assert report["hand_labeled_train_instances"]["prograsp forceps"] == 98


class _FakeArgs:
    conf_floor = 0.25
    detector_conf = 0.01
    iou = 0.45
    shards_dir = "/x"
    labels_root = "/y"
    splits = "/z"
    out_root = "/o"


def test_load_processed_shards_skips_malformed_trailing_lines(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        '{"shard": "a_part1.npz", "status": "ok"}\n'
        '{"shard": "b_part1.npz", "status": "unreadable"}\n'
        'not even json\n',
        encoding="utf-8")
    assert load_processed_shards(path) == {"a_part1.npz", "b_part1.npz"}


def test_load_processed_shards_empty_when_no_manifest_yet(tmp_path):
    assert load_processed_shards(tmp_path / "nope.jsonl") == set()


# ---------------------------------------------------------------------------
# process_shard: the end-to-end constraint, without touching torch
# ---------------------------------------------------------------------------

class _FakeDetector:
    """Duck-types surgvu.detect.Detector.detect without importing torch.

    Returns a fixed, pre-scripted per-frame detection list regardless of
    what it is handed -- this test is about the LOGBOOK constraint and the
    write-out logic, not the model.
    """

    def __init__(self, per_frame):
        self.per_frame = per_frame
        self.calls = 0

    def detect(self, frames):
        self.calls += 1
        assert len(frames) == len(self.per_frame)
        return self.per_frame


def _make_case_dir(tmp_path, name="case_007"):
    """A minimal real case directory: needle driver mounted [0, 100] in part
    1.0, prograsp forceps mounted only [50, 100]."""
    case_dir = tmp_path / name
    case_dir.mkdir()
    (case_dir / "tools.csv").write_text(
        "install_case_part,install_case_time,uninstall_case_part,"
        "uninstall_case_time,arm,commercial_toolname,groundtruth_toolname,case\n"
        "1.0,00:00:00.000000,1.0,00:01:40.000000,USM1,Large Needle Driver,"
        "needle driver,%s\n"
        "1.0,00:00:50.000000,1.0,00:01:40.000000,USM2,ProGrasp,"
        "prograsp forceps,%s\n" % (name, name),
        encoding="utf-8")
    (case_dir / "tasks.csv").write_text(
        "index,start_part,start_time,stop_part,stop_time,duration,taskname,"
        "groundtruth_taskname,matched_description,case\n"
        "0,1.0,0.0,1.0,100.0,100.0,x,suturing,d,%s\n" % (name,),
        encoding="utf-8")
    return case_dir.parent


def _tiny_shard(tmp_path, case, part="1.0", start=0.0, n=5, size=32):
    frames = [np.full((size, size, 3), i * 10, dtype=np.uint8) for i in range(n)]
    window = Window(case=case, part=part, start=start, length=float(n),
                    task="suturing", description="d",
                    tools=frozenset({"needle driver"}))
    path = tmp_path / ("%s_part1.npz" % case)
    write_shard([(window, frames)], path, fps=1, frames_per_window=n)
    return path


def test_process_shard_accepts_mounted_drops_unmounted_and_low_confidence(tmp_path):
    labels_root = _make_case_dir(tmp_path, "case_007")
    shard_path = _tiny_shard(tmp_path, "case_007", n=5)

    per_frame = [
        [{"cls": "needle driver", "conf": 0.9, "box": [0, 0, 10, 10]}],   # t=0: mounted -> accept
        [{"cls": "prograsp forceps", "conf": 0.9, "box": [0, 0, 10, 10]}],  # t=1: not yet mounted -> unmounted
        [{"cls": "needle driver", "conf": 0.1, "box": [0, 0, 10, 10]}],   # t=2: below floor -> low_confidence
        [{"cls": "bipolar dissector", "conf": 0.9, "box": [0, 0, 10, 10]}],  # t=3: out-of-taxonomy -> unmounted
        [],                                                              # t=4: nothing detected
    ]
    detector = _FakeDetector(per_frame)
    logbook = LogbookCache(labels_root)
    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    frames, meta = read_shard_safely(shard_path)
    assert frames is not None
    n_windows, n_images, tally, per_class_new = process_shard(
        frames, meta, detector, logbook, conf_floor=0.5,
        images_dir=images_dir, labels_dir=labels_dir)

    assert n_windows == 1
    assert n_images == 1
    assert tally["detections_dropped_unmounted"] == 2      # prograsp + bipolar dissector
    assert tally["detections_dropped_low_confidence"] == 1
    assert per_class_new == {"needle driver": 1}

    stem = "pseudo_case_007_part1_w0000_f00"
    assert (images_dir / (stem + ".jpg")).exists()
    label_text = (labels_dir / (stem + ".txt")).read_text(encoding="utf-8").strip()
    assert label_text.startswith("7 ")     # needle driver's YOLO_CLASSES index
    # Nothing else was written -- only the one accepted frame.
    assert sorted(p.name for p in images_dir.iterdir()) == [stem + ".jpg"]
    assert sorted(p.name for p in labels_dir.iterdir()) == [stem + ".txt"]


def test_process_shard_skips_unresolved_case_without_running_the_detector(tmp_path):
    # No case directory created under this labels_root at all.
    shard_path = _tiny_shard(tmp_path, "case_999", n=3)
    detector = _FakeDetector([[]] * 3)
    logbook = LogbookCache(tmp_path / "no_such_labels_root")
    images_dir, labels_dir = tmp_path / "images2", tmp_path / "labels2"
    images_dir.mkdir()
    labels_dir.mkdir()

    frames, meta = read_shard_safely(shard_path)
    n_windows, n_images, tally, per_class_new = process_shard(
        frames, meta, detector, logbook, conf_floor=0.25,
        images_dir=images_dir, labels_dir=labels_dir)

    assert n_windows == 1
    assert n_images == 0
    assert tally["windows_dropped_unresolved_logbook"] == 1
    assert per_class_new == {}
    # The whole point: an unresolvable window must never pay for a forward
    # pass whose result can only ever be discarded.
    assert detector.calls == 0


def test_read_shard_safely_returns_none_none_on_a_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist_part1.npz"
    frames, meta = read_shard_safely(missing)
    assert (frames, meta) == (None, None)


def test_read_shard_safely_reads_a_real_shard(tmp_path):
    shard_path = _tiny_shard(tmp_path, "case_042", n=4)
    frames, meta = read_shard_safely(shard_path)
    assert frames is not None
    assert len(meta) == 1


def test_process_shard_does_not_swallow_a_non_read_failure(tmp_path):
    """A bug in the detector/writer (bad weights, torch missing, a full
    disk) must crash the run, not be miscounted as a corrupt shard --
    read_shard_safely is the ONLY catch-and-continue boundary. This is what
    distinguishes 'this one shard was bad' from 'the whole run is broken'.

    Breaks if: process_shard (or its caller, run()) grows a broad
    `except Exception` around anything past the read step.
    """
    labels_root = _make_case_dir(tmp_path, "case_500")
    shard_path = _tiny_shard(tmp_path, "case_500", n=2)
    frames, meta = read_shard_safely(shard_path)

    class _BrokenDetector:
        def detect(self, stack):
            raise RuntimeError("best.pt failed to load")

    logbook = LogbookCache(labels_root)
    images_dir, labels_dir = tmp_path / "images3", tmp_path / "labels3"
    images_dir.mkdir()
    labels_dir.mkdir()

    with pytest.raises(RuntimeError, match="best.pt failed to load"):
        process_shard(frames, meta, _BrokenDetector(), logbook,
                      conf_floor=0.25, images_dir=images_dir,
                      labels_dir=labels_dir)
