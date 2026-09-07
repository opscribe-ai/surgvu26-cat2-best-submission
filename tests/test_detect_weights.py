"""Runs Detector.detect against the real best.pt weights and a real frame.

This is the ONLY place `Detector.detect` (src/surgvu/detect.py) is actually
executed. `tests/test_detect.py` is deliberately torch-free -- see its own
docstring -- and only covers `map_to_taxonomy` and `detections_to_record`,
neither of which touches torch, the yolov5 checkout, or the checkpoint. That
suite would have passed unchanged if `Detector.detect` had never been able to
import (it did: the original code imported `scale_coords` under the name
`scale_boxes`, which does not exist in this yolov5 checkout, and no test
anywhere would have caught that -- see the Task 6b writeup).

WHY THIS RUNS AGAINST A REAL FRAME, NOT SYNTHETIC NOISE. The geometry check
below (every returned box inside the frame) is the empirical half of
verifying the letterbox/scale_coords fix described in detect.py's own
docstring: a `cv2.resize`-to-square upstream (`prepare_frame`) followed by a
letterbox-aware `scale_coords` downstream only round-trips correctly if the
ratio/pad math matches what was actually done to the frame. A synthetic
all-noise frame can exercise the same code path but tells you nothing about
whether the *model* fires -- a broken class-index mapping or a checkpoint
that silently loaded the wrong architecture would still "pass" a shape check
on noise. A real, tool-bearing surgical frame is the only input that can
also go quiet in a way worth investigating.

SKIPPED, NOT FAILED, when the weights or the yolov5 checkout are absent --
this suite must stay green on a machine (a laptop, this login node, CI
without a staging mount) that has neither, and it must not need network
access. Only condor/detect_smoke.sub, which runs inside surgvu26-train.sif
with +WantStagingMount, actually exercises the body of the test below.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The controller verified these paths present on staging before assigning
# this task; skipif below re-checks at test time so a machine without the
# +WantStagingMount mount (or a future move of the corpus) gets a clean skip
# rather than a collection-time crash.
_DETECTOR_ROOT = Path(
    "/staging/groups/bhaskar_opscribe/surgvu_yolo_detector")
WEIGHTS = _DETECTOR_ROOT / "best.pt"
YOLOV5_DIR = _DETECTOR_ROOT / "yolov5"

# A real Cat 2 sample clip, preferred because it matches the exact geometry
# (1280x720 @ 60 fps, black side margins) `prepare_frame` was written for.
# Falls back to the surgvu24 corpus clip the controller named, if the public
# sample set is ever moved or renamed.
_SAMPLE_VIDEO = Path(
    "/staging/groups/bhaskar_opscribe/surgvu/cat2_sample/case122/case122.mp4")
_FALLBACK_VIDEO = Path(
    "/staging/groups/bhaskar_opscribe/surgvu/videos/surgvu24/case_000/"
    "case_000_video_part_001.mp4")


def _video_path():
    return _SAMPLE_VIDEO if _SAMPLE_VIDEO.exists() else _FALLBACK_VIDEO


def _staging_available():
    return WEIGHTS.exists() and YOLOV5_DIR.exists() and _video_path().exists()


_SKIP_REASON = (
    "weights (%s), yolov5 checkout (%s) or a sample video not present -- "
    "this test only runs where /staging/groups/bhaskar_opscribe is mounted "
    "(condor/detect_smoke.sub); it is skipped rather than failed everywhere "
    "else so the suite stays green off-staging." % (WEIGHTS, YOLOV5_DIR))


@pytest.mark.slow  # see condor/pytest.sh: "-m 'not slow'" excludes this from
# the main suite there. That matters beyond runtime: condor/pytest.sub's
# container ALSO has +WantStagingMount, so weights/checkout/video all exist
# in that job too -- skipif alone would let the main suite try to actually
# run this, and condor/pytest.sh never installs yolov5's extra dependencies
# (pandas, requests, tqdm, matplotlib, seaborn -- see condor/detect_smoke.sh),
# so it would ImportError there instead of skipping. The marker is what keeps
# that job green; only condor/detect_smoke.sub, which runs `pytest
# tests/test_detect_weights.py` directly with no `-m` filter, executes this.
@pytest.mark.skipif(not _staging_available(), reason=_SKIP_REASON)
def test_detector_runs_on_a_real_frame_and_stays_in_bounds():
    # Imported here, not at module scope: torch and surgvu.perceive (which
    # imports torch directly) must never be required to even COLLECT this
    # file on a machine without torch. The skipif above has already decided
    # whether this body runs before any of these names are touched.
    from surgvu.detect import YOLO_CLASSES, Detector
    from surgvu.perceive import decode_clip

    video = _video_path()
    # Defaults (16 frames, 512x512), not overridden: this must exercise the
    # exact preprocessing (`prepare_frame`'s crop-margins + blur-UI-band +
    # square resize) the serving path actually uses, not a test-only shape.
    frames = decode_clip(video)

    detector = Detector(WEIGHTS, YOLOV5_DIR)
    result = detector.detect(frames)

    # One entry per input frame. A length mismatch here is exactly what
    # `detections_to_record` guards against (it raises on a length
    # mismatch against timestamps) -- if `detect()` ever dropped or merged
    # a frame, every detection after the drop would be stamped with the
    # wrong anchor's timestamp downstream, silently.
    assert isinstance(result, list)
    assert len(result) == len(frames)

    height, width = frames.shape[1], frames.shape[2]
    found_any = False
    for per_frame in result:
        assert isinstance(per_frame, list)
        for item in per_frame:
            found_any = True
            assert isinstance(item, dict)
            assert set(item) == {"cls", "conf", "box"}

            # A name outside the 14-class contract means the weights and
            # detect.py's YOLO_CLASSES table disagree (wrong checkpoint,
            # wrong class-index order, or a stale copy of one of the two).
            assert item["cls"] in YOLO_CLASSES

            # NMS-kept detections carry an objectness*class confidence in
            # (0, 1]; 0 or negative, or anything > 1, means the raw model
            # output is being read as something it is not (e.g. an
            # unnormalized logit slipping through untouched).
            conf = item["conf"]
            assert 0 < conf <= 1, "conf %r out of (0, 1] for %r" % (
                conf, item["cls"])

            # THE IMPORTANT ONE. This is the empirical check on the
            # letterbox/scale_coords geometry fix (see detect.py's
            # docstring above `detect()`): a stretch-vs-letterbox mismatch,
            # a `ratio_pad` computed from the wrong shapes, or a transposed
            # width/height would push boxes outside the frame or shrink
            # them toward a corner -- and every one of those is invisible
            # to a torch-free test that never runs the model. Reading the
            # code only tells you the math was INTENDED to round-trip;
            # this is whether it actually does, against this checkpoint,
            # against this exact preprocessing.
            box = item["box"]
            assert len(box) == 4
            x1, y1, x2, y2 = box
            assert 0 <= x1 < x2 <= width, (
                "box %r has x1/x2 outside [0, %d]" % (box, width))
            assert 0 <= y1 < y2 <= height, (
                "box %r has y1/y2 outside [0, %d]" % (box, height))

    # Not part of the interface contract (an empty record is valid JSON and
    # detections_to_record handles it fine), but a REAL surgical clip with
    # nothing detected across all 16 sampled anchors, at the default
    # conf=0.25 this checkpoint was validated at (precision ~0.80, recall
    # ~0.75 on its own val split per surg_14cls_run/results.csv), is much
    # more likely to mean "the checkpoint, the class table or the
    # preprocessing silently disagree" than "this clip truly shows no tool
    # for 16 evenly spaced samples across 30 seconds of active surgery".
    # Fail loudly here rather than let that look like a passing smoke test.
    assert found_any, (
        "Detector found zero detections across all %d frames sampled from "
        "%s at conf=%.2f. Investigate before trusting this checkpoint: a "
        "silent no-op here is exactly the failure Task 11's try/except "
        "would otherwise swallow as one log line." % (
            len(frames), video, detector.conf))
