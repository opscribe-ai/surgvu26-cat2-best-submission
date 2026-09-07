"""Tests for scripts/build_gi_stage1.py -- v6's stage-1 corpus builder."""
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_gi_stage1",
    Path(__file__).resolve().parents[1] / "scripts" / "build_gi_stage1.py")
gi = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gi)


def test_bucketing_is_stable_across_processes():
    """THE ONE THAT WOULD SILENTLY RESHUFFLE TRAIN AND VAL.

    Python's built-in hash() is randomised per process (PYTHONHASHSEED), so
    using it here would put the same directory in a different split on every
    run -- train/val membership changing between the manifest build and any
    later rebuild, with nothing saying so.
    """
    import subprocess
    import sys

    path = "/staging/x/seq_3/left_frames"
    expected = gi.stable_bucket(path)
    out = subprocess.run(
        [sys.executable, "-c",
         "import importlib.util,sys;"
         "spec=importlib.util.spec_from_file_location('g',%r);"
         "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
         "print(m.stable_bucket(%r))"
         % (str(Path(__file__).resolve().parents[1] / "scripts" / "build_gi_stage1.py"), path)],
        capture_output=True, text=True, check=True)
    assert int(out.stdout.strip()) == expected


def test_bucket_is_in_range():
    for i in range(200):
        assert 0 <= gi.stable_bucket("dir/%d" % i) < gi.N_BUCKETS


def test_case_ids_cannot_collide_with_surgvu_or_ssgvqa():
    """SurgVU holds case_000..154; the SSG-VQA builder holds case_900+."""
    from surgvu.sampling import normalize_case_id

    for i in range(300):
        case = gi.case_for_dir("some/dir/%d" % i)
        n = int(case.split("_")[1])
        assert normalize_case_id(case) == case
        assert 154 < n < 900


# -- conversation parsing ---------------------------------------------------

def test_extract_pair_takes_the_first_human_and_gpt_turns():
    record = {"conversations": [
        {"from": "human", "value": "Q1"}, {"from": "gpt", "value": "A1"},
        {"from": "human", "value": "Q2"}, {"from": "gpt", "value": "A2"}]}
    assert gi.extract_pair(record) == ("Q1", "A1")


@pytest.mark.parametrize("convo", [
    [],
    [{"from": "human", "value": "Q"}],                       # no answer
    [{"from": "gpt", "value": "A"}],                         # no question
    [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "   "}],
    [{"from": "human", "value": ""}, {"from": "gpt", "value": "A"}],
])
def test_records_without_a_usable_pair_are_dropped(convo):
    """An empty target teaches the model to emit nothing, and a missing
    response scores 0 at grading."""
    assert gi.extract_pair({"conversations": convo}) is None


# -- destination paths ------------------------------------------------------

def test_destinations_are_collision_free_across_sources():
    """The nine corpora reuse basenames -- frame000.png appears under many
    sequences -- so keying the destination by basename would silently
    overwrite one source's frames with another's."""
    a = gi.out_path_for("/staging/a/seq_1/left_frames/frame000.png", "/out")
    b = gi.out_path_for("/staging/b/seq_9/left_frames/frame000.png", "/out")
    assert a != b
    assert a.name.endswith(".jpg") and b.name.endswith(".jpg")


def test_destination_is_stable_for_the_same_source():
    """Stability is what makes the conversion resumable."""
    p = "/staging/a/seq_1/left_frames/frame000.png"
    assert gi.out_path_for(p, "/out") == gi.out_path_for(p, "/out")


# -- image conversion -------------------------------------------------------

def test_converted_image_matches_prepare_frame_geometry(tmp_path):
    import cv2
    import numpy as np

    from surgvu.preprocess import prepare_frame

    src = tmp_path / "in.png"
    cv2.imwrite(str(src), (np.random.rand(720, 1280, 3) * 255).astype("uint8"))
    dst = tmp_path / "out.jpg"
    assert gi.convert_image(src, dst, 512)
    got = cv2.imread(str(dst))
    assert got.shape == prepare_frame(np.zeros((720, 1280, 3), "uint8"), 512).shape
    assert got.shape == (512, 512, 3)


def test_unreadable_image_returns_false_rather_than_raising(tmp_path):
    """87k images from nine corpora: one bad file must not kill the build."""
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    assert gi.convert_image(bad, tmp_path / "o.jpg", 512) is False


def test_uses_the_shared_jpeg_quality():
    from surgvu.extract import JPEG_QUALITY

    assert gi.JPEG_QUALITY == JPEG_QUALITY == 90
