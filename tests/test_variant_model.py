"""Runs VariantHead.predict against real trained weights, once training has
produced them.

This is the container half of Task 10, mirroring how tests/test_detect.py
(torch-free) and tests/test_detect_weights.py (real weights, real forward
pass) split for the detector. tests/test_variant.py already covers
`variant_record` -- the decision logic -- completely and needs no torch;
this file is the one place `VariantHead._load`/`.predict` actually run.

SKIPPED, NOT FAILED, until `scripts/train_variant.py` has been run and its
output committed. This task explicitly does not run training on this
machine or on the login node -- the controller submits
condor/train_variant.sub -- so `config/variant_head.json` and the weights it
points at do not exist yet anywhere this suite can see. A skip here says so
truthfully; a failure would say something is broken when nothing has been
attempted yet.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "variant_head.json"

# Same sample clip test_detect_weights.py uses, for the same reason: it
# matches the exact geometry (1280x720 @ 60 fps, black side margins)
# `prepare_frame` was written for, with the surgvu24 corpus clip as a
# fallback if the public sample set ever moves.
_SAMPLE_VIDEO = Path(
    "/staging/groups/bhaskar_opscribe/surgvu/cat2_sample/case122/case122.mp4")
_FALLBACK_VIDEO = Path(
    "/staging/groups/bhaskar_opscribe/surgvu/videos/surgvu24/case_000/"
    "case_000_video_part_001.mp4")

_DETECTOR_ROOT = Path("/staging/groups/bhaskar_opscribe/surgvu_yolo_detector")
_DETECTOR_WEIGHTS = _DETECTOR_ROOT / "best.pt"
_YOLOV5_DIR = _DETECTOR_ROOT / "yolov5"


def _video_path():
    return _SAMPLE_VIDEO if _SAMPLE_VIDEO.exists() else _FALLBACK_VIDEO


def _load_config():
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _config_and_weights_available():
    cfg = _load_config()
    if cfg is None:
        return False
    weights = Path(cfg.get("weights", ""))
    return weights.exists() and _video_path().exists()


_SKIP_REASON = (
    "config/variant_head.json (%s), its weights, or a sample video are not "
    "present. This suite only runs after scripts/train_variant.py has been "
    "run (condor/train_variant.sub) and its output committed; it is skipped "
    "rather than failed everywhere else so the suite stays green before "
    "that has happened." % (_CONFIG_PATH,))


def _make_head():
    from surgvu.variant import VariantHead

    cfg = _load_config()
    return VariantHead(cfg["weights"], cfg["cutoff"]), cfg


@pytest.mark.skipif(not _config_and_weights_available(), reason=_SKIP_REASON)
def test_variant_head_predicts_on_whole_frames_no_boxes():
    """The degraded path: no detector box, whole preprocessed frame. This is
    what VariantHead.predict falls back to whenever the detector finds
    nothing, so it has to work on its own, not merely as a code path nobody
    exercises."""
    from surgvu.perceive import decode_clip
    from surgvu.variant import FAMILIES

    head, cfg = _make_head()
    frames = decode_clip(_video_path())

    record = head.predict(frames)

    assert set(record) == {"version", "family", "p_large", "p_mega",
                           "cutoff", "decided"}
    assert record["cutoff"] == pytest.approx(float(cfg["cutoff"]))
    assert record["family"] in (None,) + FAMILIES
    assert record["p_large"] + record["p_mega"] == pytest.approx(1.0, abs=1e-3)
    assert record["decided"] == (
        max(record["p_large"], record["p_mega"]) >= record["cutoff"])
    # variant_record's own guard: this cutoff must already be legal, since
    # it came from the same config file the decision layer will read at
    # serving time.
    assert record["cutoff"] > 0.5


def _detector_available():
    return (_config_and_weights_available() and _DETECTOR_WEIGHTS.exists()
            and _YOLOV5_DIR.exists())


@pytest.mark.slow  # needs yolov5's import-time deps -- see
# condor/detect_smoke.sh's docstring for the exact list (pandas, requests,
# tqdm, matplotlib, seaborn) -- which condor/pytest.sub's container does not
# install, same reasoning as tests/test_detect_weights.py's own marker.
@pytest.mark.skipif(not _detector_available(), reason=_SKIP_REASON + (
    " Also needs the yolov5 checkout and best.pt for the cropped path."))
def test_variant_head_predicts_with_detector_boxes():
    """The intended path: crop to the detector's needle-driver box before
    classifying. Exercises the box-clamping/degenerate-box arithmetic in
    VariantHead.predict against real detections, not synthetic boxes."""
    from surgvu.detect import Detector
    from surgvu.perceive import decode_clip

    head, _cfg = _make_head()
    frames = decode_clip(_video_path())

    detector = Detector(_DETECTOR_WEIGHTS, _YOLOV5_DIR)
    detections = detector.detect(frames)
    boxes = {}
    for index, found in enumerate(detections):
        needle = [item for item in found if item["cls"] == "needle driver"]
        if needle:
            boxes[index] = max(needle, key=lambda item: item["conf"])["box"]

    record = head.predict(frames, boxes=boxes or None)

    assert record["family"] in (None, "large", "mega")
    assert 0.0 <= record["p_large"] <= 1.0
    assert 0.0 <= record["p_mega"] <= 1.0
