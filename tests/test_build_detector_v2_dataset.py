"""Tests for scripts/build_detector_v2_dataset.py.

Deliberately torch-free, same discipline as tests/test_pseudo_label.py: this
module never touches the real 601K-image pseudo pool or the real 886-image
hand-labeled set (both live on /staging and this build is meant to run on a
compute node, not the login node). Every fixture below is a handful of tiny
synthetic files under tmp_path.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from surgvu.detect import YOLO_CLASSES  # noqa: E402

from build_detector_v2_dataset import (  # noqa: E402
    assert_no_heldout_pseudo_images,
    classify_pseudo_pool,
    compute_multipliers,
    hand_oversample_factors,
    image_path_to_label_path,
    list_pseudo_pairs,
    oversample_hand_labeled,
    protected_classes_from_multipliers,
    pseudo_case_id_from_path,
    read_label_classes,
    read_txt_list,
    resolve_hand_paths,
    run,
    select_pseudo_images,
    verify_pseudo_harvest_excluded_heldout,
    write_dataset_yaml,
    zero_gain_classes_from_report,
)

# Real per-class baseline/new counts from the real pseudo_label_report.json
# (2026-08-25 harvest, cluster 9698244) -- used so the multiplier/protection
# tests exercise the same numbers the plan document and this build's own
# defaults were reasoned about, not invented round numbers.
REAL_BASELINE = {
    "bipolar dissector": 79, "bipolar forceps": 366, "cadiere forceps": 204,
    "clip applier": 90, "force bipolar": 126, "grasping retractor": 102,
    "monopolar curved scissors": 159, "needle driver": 257,
    "permanent cautery hook/spatula": 92, "prograsp forceps": 98,
    "stapler": 94, "suction irrigator": 99,
    "tip-up fenestrated grasper": 91, "vessel sealer": 104,
}
REAL_NEW = {
    "bipolar forceps": 209568, "cadiere forceps": 157484, "clip applier": 13525,
    "force bipolar": 25806, "grasping retractor": 6587,
    "monopolar curved scissors": 242464, "needle driver": 484021,
    "permanent cautery hook/spatula": 31500, "prograsp forceps": 20647,
    "stapler": 2248, "tip-up fenestrated grasper": 530, "vessel sealer": 21016,
    # bipolar dissector / suction irrigator absent -- zero new instances.
}


def _yolo_line(index):
    return "%d 0.500000 0.500000 0.200000 0.200000" % index


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def test_read_txt_list_strips_blank_lines(tmp_path):
    path = tmp_path / "train.txt"
    path.write_text("a.jpg\n\n  \nb.jpg\n", encoding="utf-8")
    assert read_txt_list(path) == ["a.jpg", "b.jpg"]


def test_resolve_hand_paths_returns_absolute_existing_paths(tmp_path):
    (tmp_path / "yolo_dataset" / "images").mkdir(parents=True)
    img = tmp_path / "yolo_dataset" / "images" / "clip_1.jpg"
    img.write_bytes(b"fake")
    resolved = resolve_hand_paths(["yolo_dataset/images/clip_1.jpg"], tmp_path)
    assert resolved == [img.resolve()]


def test_resolve_hand_paths_raises_on_a_missing_file(tmp_path):
    # Breaks if: the exists() check is removed, letting a bogus path through.
    with pytest.raises(FileNotFoundError):
        resolve_hand_paths(["yolo_dataset/images/nope.jpg"], tmp_path)


def test_image_path_to_label_path_replaces_last_images_segment():
    # Mirrors yolov5's own img2label_paths rsplit convention.
    p = image_path_to_label_path("/a/images/b/images/clip_1.jpg")
    assert str(p) == "/a/images/b/labels/clip_1.txt"


def test_image_path_to_label_path_raises_without_images_segment():
    # Breaks if: the sep-check `if sep not in text: raise` is removed.
    with pytest.raises(ValueError):
        image_path_to_label_path("/a/frames/clip_1.jpg")


# ---------------------------------------------------------------------------
# Label reading
# ---------------------------------------------------------------------------

def test_read_label_classes_maps_indices_to_names(tmp_path):
    label = tmp_path / "x.txt"
    label.write_text("%s\n%s\n" % (_yolo_line(7), _yolo_line(10)), encoding="utf-8")
    assert read_label_classes(label) == frozenset({"needle driver", "stapler"})


def test_read_label_classes_raises_on_out_of_range_index(tmp_path):
    # Breaks if: the `0 <= idx < len(class_names)` range check is removed.
    label = tmp_path / "x.txt"
    label.write_text(_yolo_line(99) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="out of range"):
        read_label_classes(label)


def test_list_pseudo_pairs_finds_matching_image_label_pairs(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "pseudo_case_001_part1_w0000_f00.jpg").write_bytes(b"x")
    (labels / "pseudo_case_001_part1_w0000_f00.txt").write_text(_yolo_line(7), encoding="utf-8")
    pairs = list_pseudo_pairs(images, labels)
    assert len(pairs) == 1
    assert pairs[0][0].name == "pseudo_case_001_part1_w0000_f00.jpg"


def test_list_pseudo_pairs_raises_when_image_missing(tmp_path):
    # Breaks if: the image_path.exists() check is removed and a missing
    # image is silently skipped instead of raised.
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (labels / "pseudo_case_001_part1_w0000_f00.txt").write_text(_yolo_line(7), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        list_pseudo_pairs(images, labels)


# ---------------------------------------------------------------------------
# R30 -- two independent layers
# ---------------------------------------------------------------------------

def test_verify_pseudo_harvest_excluded_heldout_passes_on_real_report():
    verify_pseudo_harvest_excluded_heldout({"shards_heldout_excluded": 14})


def test_verify_pseudo_harvest_excluded_heldout_raises_on_zero():
    # Breaks if: the `if not excluded: raise` guard is deleted or weakened.
    with pytest.raises(ValueError, match="R30"):
        verify_pseudo_harvest_excluded_heldout({"shards_heldout_excluded": 0})


def test_verify_pseudo_harvest_excluded_heldout_raises_when_field_absent():
    with pytest.raises(ValueError):
        verify_pseudo_harvest_excluded_heldout({})


def test_pseudo_case_id_from_path_parses_the_harvest_stem_format():
    assert pseudo_case_id_from_path(
        "/x/pseudo_case_073_part1_w0000_f02.jpg") == "case_073"


def test_pseudo_case_id_from_path_raises_on_an_unrecognised_name():
    with pytest.raises(ValueError):
        pseudo_case_id_from_path("/x/clip_000036_2.jpg")


def test_assert_no_heldout_pseudo_images_passes_when_clean():
    assert_no_heldout_pseudo_images(
        ["/x/pseudo_case_050_part1_w0000_f00.jpg"], ["case_122"]) is None


def test_assert_no_heldout_pseudo_images_raises_on_contamination_across_spelling():
    # The exact bug shape this project has hit before: heldout spelled
    # 'case122', pseudo path spelled 'case_122'. Breaks if: normalize_case_id
    # is swapped for raw string equality, which would find no match here and
    # silently pass a contaminated pool.
    with pytest.raises(RuntimeError, match="case_122"):
        assert_no_heldout_pseudo_images(
            ["/x/pseudo_case_122_part1_w0000_f00.jpg",
             "/x/pseudo_case_050_part1_w0000_f00.jpg"],
            ["case122"])


# ---------------------------------------------------------------------------
# Multipliers / protection
# ---------------------------------------------------------------------------

def test_compute_multipliers_matches_the_real_measured_yield():
    multipliers = compute_multipliers(REAL_BASELINE, REAL_NEW)
    assert multipliers["tip-up fenestrated grasper"] == pytest.approx(6.82, abs=0.05)
    assert multipliers["needle driver"] == pytest.approx(1884.0, abs=1.0)
    # Zero-gain classes are exactly 1.0x -- baseline plus zero new.
    assert multipliers["bipolar dissector"] == 1.0
    assert multipliers["suction irrigator"] == 1.0


def test_compute_multipliers_raises_on_zero_baseline():
    # Breaks if: the `if not base: raise` guard is removed.
    with pytest.raises(ValueError):
        compute_multipliers({**REAL_BASELINE, "stapler": 0}, REAL_NEW)


def test_protected_classes_from_multipliers_threshold_is_inclusive():
    multipliers = {"a": 100.0, "b": 100.0001, "c": 6.8}
    protected = protected_classes_from_multipliers(multipliers, 100.0)
    assert protected == frozenset({"a", "c"})


def test_zero_gain_classes_from_report_reads_the_real_two_classes():
    report = {"per_class_new_instances": REAL_NEW}
    assert zero_gain_classes_from_report(report) == frozenset(
        {"bipolar dissector", "suction irrigator"})


# ---------------------------------------------------------------------------
# Pseudo-pool classification and capped selection
# ---------------------------------------------------------------------------

def test_classify_pseudo_pool_splits_protected_from_common(tmp_path):
    protected = frozenset({"stapler"})
    p1 = tmp_path / "p1.jpg"
    l1 = tmp_path / "p1.txt"
    l1.write_text(_yolo_line(10), encoding="utf-8")  # stapler -> protected
    p2 = tmp_path / "p2.jpg"
    l2 = tmp_path / "p2.txt"
    l2.write_text(_yolo_line(7), encoding="utf-8")  # needle driver -> common
    prot, common = classify_pseudo_pool([(p1, l1), (p2, l2)], protected)
    assert prot == [p1]
    assert common == [p2]


def test_select_pseudo_images_keeps_all_protected_uncapped():
    # Breaks if: the cap is applied to protected_images too.
    protected = ["p%d" % i for i in range(50)]
    common = ["c%d" % i for i in range(1000)]
    selected = select_pseudo_images(protected, common, common_cap=10, seed=0)
    assert all(p in selected for p in protected)
    assert len(selected) == 50 + 10


def test_select_pseudo_images_keeps_all_common_when_pool_smaller_than_cap():
    selected = select_pseudo_images([], ["c1", "c2"], common_cap=100, seed=0)
    assert sorted(selected) == ["c1", "c2"]


def test_select_pseudo_images_is_deterministic_for_a_fixed_seed():
    common = ["c%d" % i for i in range(1000)]
    a = select_pseudo_images([], common, common_cap=10, seed=42)
    b = select_pseudo_images([], common, common_cap=10, seed=42)
    assert a == b


def test_select_pseudo_images_raises_on_negative_cap():
    with pytest.raises(ValueError):
        select_pseudo_images([], ["c1"], common_cap=-1, seed=0)


# ---------------------------------------------------------------------------
# Hand-labeled oversampling
# ---------------------------------------------------------------------------

def test_hand_oversample_factors_boosts_zero_gain_images(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    zero_gain_img = images_dir / "a.jpg"
    (labels_dir / "a.txt").write_text(_yolo_line(0), encoding="utf-8")  # bipolar dissector
    plain_img = images_dir / "b.jpg"
    (labels_dir / "b.txt").write_text(_yolo_line(7), encoding="utf-8")  # needle driver
    zero_gain_img.write_bytes(b"x")
    plain_img.write_bytes(b"x")

    factors = hand_oversample_factors(
        [zero_gain_img, plain_img], frozenset({"bipolar dissector"}),
        base_factor=20, boost_factor=20)
    assert factors[str(zero_gain_img)] == 40
    assert factors[str(plain_img)] == 20


def test_hand_oversample_factors_raises_on_invalid_base_factor(tmp_path):
    with pytest.raises(ValueError):
        hand_oversample_factors([], frozenset(), base_factor=0, boost_factor=0)


def test_oversample_hand_labeled_repeats_each_path_per_its_factor():
    lines = oversample_hand_labeled(["a", "b"], {"a": 3, "b": 1})
    assert lines == ["a", "a", "a", "b"]


# ---------------------------------------------------------------------------
# Dataset yaml
# ---------------------------------------------------------------------------

def test_write_dataset_yaml_lists_all_fourteen_classes_in_order(tmp_path):
    path = tmp_path / "surg_14cls_v2.yaml"
    write_dataset_yaml(path, tmp_path / "train.txt", tmp_path / "val.txt")
    text = path.read_text(encoding="utf-8")
    assert "nc: 14" in text
    for name in YOLO_CLASSES:
        assert ("- %s" % name) in text


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _build_fixture_tree(tmp_path):
    """A tiny hand-labeled set (2 train / 1 val) plus a tiny pseudo pool (one
    protected-class image, several common-class images, one heldout-cased
    image) -- enough to exercise every path `run()` takes without any real
    data.
    """
    hand_root = tmp_path / "hand"
    (hand_root / "yolo_dataset" / "images").mkdir(parents=True)
    (hand_root / "yolo_dataset" / "labels").mkdir(parents=True)
    train_imgs = []
    for i, cls_idx in enumerate([7, 0]):  # needle driver, bipolar dissector
        img = hand_root / "yolo_dataset" / "images" / ("clip_%d.jpg" % i)
        lbl = hand_root / "yolo_dataset" / "labels" / ("clip_%d.txt" % i)
        img.write_bytes(b"x")
        lbl.write_text(_yolo_line(cls_idx), encoding="utf-8")
        train_imgs.append("yolo_dataset/images/clip_%d.jpg" % i)
    val_img = hand_root / "yolo_dataset" / "images" / "clip_val.jpg"
    val_lbl = hand_root / "yolo_dataset" / "labels" / "clip_val.txt"
    val_img.write_bytes(b"x")
    val_lbl.write_text(_yolo_line(7), encoding="utf-8")
    (hand_root / "yolo_dataset" / "train.txt").write_text(
        "\n".join(train_imgs) + "\n", encoding="utf-8")
    (hand_root / "yolo_dataset" / "val.txt").write_text(
        "yolo_dataset/images/clip_val.jpg\n", encoding="utf-8")

    pseudo_images = tmp_path / "pseudo" / "images"
    pseudo_labels = tmp_path / "pseudo" / "labels"
    pseudo_images.mkdir(parents=True)
    pseudo_labels.mkdir(parents=True)
    # One protected (stapler) image, three common (needle driver) images.
    stems = {
        "pseudo_case_050_part1_w0000_f00": 10,   # stapler -> protected
        "pseudo_case_050_part1_w0001_f00": 7,    # needle driver -> common
        "pseudo_case_051_part1_w0000_f00": 7,
        "pseudo_case_052_part1_w0000_f00": 7,
    }
    for stem, cls_idx in stems.items():
        (pseudo_images / (stem + ".jpg")).write_bytes(b"x")
        (pseudo_labels / (stem + ".txt")).write_text(_yolo_line(cls_idx), encoding="utf-8")

    splits_path = tmp_path / "splits_v2.json"
    splits_path.write_text(json.dumps({"heldout": ["case122"]}), encoding="utf-8")

    report_path = tmp_path / "pseudo_label_report.json"
    report_path.write_text(json.dumps({
        "shards_heldout_excluded": 14,
        "hand_labeled_train_instances": REAL_BASELINE,
        "per_class_new_instances": REAL_NEW,
    }), encoding="utf-8")

    return hand_root, pseudo_images, pseudo_labels, splits_path, report_path


def test_run_end_to_end_builds_a_combined_manifest(tmp_path):
    hand_root, pseudo_images, pseudo_labels, splits_path, report_path = \
        _build_fixture_tree(tmp_path)
    out_dir = tmp_path / "out"
    args = _Args(
        hand_root=str(hand_root),
        hand_train=str(hand_root / "yolo_dataset" / "train.txt"),
        hand_val=str(hand_root / "yolo_dataset" / "val.txt"),
        pseudo_images_dir=str(pseudo_images),
        pseudo_labels_dir=str(pseudo_labels),
        pseudo_report=str(report_path),
        splits=str(splits_path),
        out_dir=str(out_dir),
        report_out=None,
        hand_oversample=3,
        zero_gain_oversample_boost=2,
        pseudo_common_cap=1,  # cap the 3 common images down to 1
        protect_below_multiplier=100.0,
        seed=0,
    )
    rep = run(args)

    train_lines = read_txt_list(out_dir / "train.txt")
    val_lines = read_txt_list(out_dir / "val.txt")

    # 2 hand images oversampled (one at 3x, the zero-gain one at 3+2=5x) +
    # 1 protected pseudo + 1 capped common pseudo.
    assert len(train_lines) == 3 + 5 + 1 + 1
    assert len(val_lines) == 1
    assert "clip_val" in val_lines[0]
    # Val must never contain a pseudo-labeled path.
    assert not any("pseudo_" in line for line in val_lines)
    assert rep["pseudo_images_selected"] == 2
    assert rep["protected_classes"] == ["bipolar dissector", "stapler",
                                        "suction irrigator",
                                        "tip-up fenestrated grasper",
                                        "grasping retractor"] or \
        set(rep["protected_classes"]) == {
            "bipolar dissector", "suction irrigator",
            "tip-up fenestrated grasper", "stapler", "grasping retractor"}


def test_run_raises_when_pseudo_pool_contains_a_heldout_case(tmp_path):
    hand_root, pseudo_images, pseudo_labels, splits_path, report_path = \
        _build_fixture_tree(tmp_path)
    # Plant a heldout-cased pseudo image (splits.json's heldout is
    # 'case122' -- differently spelled, same case as 'case_122').
    (pseudo_images / "pseudo_case_122_part1_w0000_f00.jpg").write_bytes(b"x")
    (pseudo_labels / "pseudo_case_122_part1_w0000_f00.txt").write_text(
        _yolo_line(7), encoding="utf-8")

    out_dir = tmp_path / "out"
    args = _Args(
        hand_root=str(hand_root),
        hand_train=str(hand_root / "yolo_dataset" / "train.txt"),
        hand_val=str(hand_root / "yolo_dataset" / "val.txt"),
        pseudo_images_dir=str(pseudo_images),
        pseudo_labels_dir=str(pseudo_labels),
        pseudo_report=str(report_path),
        splits=str(splits_path),
        out_dir=str(out_dir),
        report_out=None,
        hand_oversample=1,
        zero_gain_oversample_boost=0,
        pseudo_common_cap=100,
        protect_below_multiplier=100.0,
        seed=0,
    )
    # Breaks if: assert_no_heldout_pseudo_images's raise is replaced with a
    # log-and-drop (this test would then need to assert exclusion instead of
    # a raised error, and a real regression would ship silently).
    with pytest.raises(RuntimeError, match="case_122"):
        run(args)


def test_run_raises_when_report_claims_zero_heldout_exclusion(tmp_path):
    hand_root, pseudo_images, pseudo_labels, splits_path, report_path = \
        _build_fixture_tree(tmp_path)
    report_path.write_text(json.dumps({
        "shards_heldout_excluded": 0,
        "hand_labeled_train_instances": REAL_BASELINE,
        "per_class_new_instances": REAL_NEW,
    }), encoding="utf-8")

    out_dir = tmp_path / "out"
    args = _Args(
        hand_root=str(hand_root),
        hand_train=str(hand_root / "yolo_dataset" / "train.txt"),
        hand_val=str(hand_root / "yolo_dataset" / "val.txt"),
        pseudo_images_dir=str(pseudo_images),
        pseudo_labels_dir=str(pseudo_labels),
        pseudo_report=str(report_path),
        splits=str(splits_path),
        out_dir=str(out_dir),
        report_out=None,
        hand_oversample=1,
        zero_gain_oversample_boost=0,
        pseudo_common_cap=100,
        protect_below_multiplier=100.0,
        seed=0,
    )
    with pytest.raises(ValueError, match="R30"):
        run(args)


# ---------------------------------------------------------------------------
# classify_pseudo_pool reads the pool concurrently. These pin the property
# that makes that a PERFORMANCE change and not a behaviour change.
# ---------------------------------------------------------------------------

def _write_pool(tmp_path, specs):
    """(pairs, images_dir, labels_dir) for {stem: [class_index, ...]}."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    for stem, class_indices in specs.items():
        (images_dir / (stem + ".jpg")).write_bytes(b"x")
        (labels_dir / (stem + ".txt")).write_text(
            "\n".join("%d 0.5 0.5 0.1 0.1" % i for i in class_indices),
            encoding="utf-8")
    return list_pseudo_pairs(images_dir, labels_dir), images_dir, labels_dir


def test_classify_pseudo_pool_preserves_input_order_across_threads(tmp_path):
    """ORDER IS THE WHOLE CONTRACT. classify_pseudo_pool now reads the
    601,261-file pool through a ThreadPoolExecutor. `executor.map` yields in
    INPUT order; `as_completed` would not, and switching to it would scramble
    the partition and silently de-reproducibilise select_pseudo_images'
    seeded sample -- a change no other test would catch, because both
    orderings contain exactly the same elements.
    """
    protected_index = YOLO_CLASSES.index("stapler")
    common_index = YOLO_CLASSES.index("needle driver")
    specs = {}
    for i in range(60):
        specs["frame_%03d" % i] = [protected_index if i % 2 == 0 else common_index]
    pairs, _images_dir, _labels_dir = _write_pool(tmp_path, specs)

    protected, common = classify_pseudo_pool(
        pairs, protected_classes=frozenset({"stapler"}))

    expected_protected = [p for p, _l in pairs
                          if int(p.stem.split("_")[1]) % 2 == 0]
    expected_common = [p for p, _l in pairs
                       if int(p.stem.split("_")[1]) % 2 == 1]
    assert protected == expected_protected
    assert common == expected_common
    assert len(protected) == 30 and len(common) == 30


def test_classify_pseudo_pool_handles_an_empty_pool():
    """The threaded path short-circuits on an empty pair list rather than
    standing up a pool to do nothing."""
    assert classify_pseudo_pool([], protected_classes=frozenset({"stapler"})) == ([], [])
