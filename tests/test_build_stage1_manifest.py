"""Tests for scripts/build_stage1_manifest.py.

Torch-free and zip-free where it matters: the four functions that can be
silently wrong -- answer formatting, case-id mapping, window clamping, and
record construction -- are pure, and they are what is exercised here.
"""
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_stage1_manifest",
    Path(__file__).resolve().parents[1] / "scripts" / "build_stage1_manifest.py")
b1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(b1)


# -- case ids ---------------------------------------------------------------

def test_case_ids_cannot_collide_with_surgvu():
    """SurgVU occupies case_000..case_154. A stage-1 id that landed in that
    range could be silently assigned to a SurgVU split."""
    from surgvu.sampling import normalize_case_id

    for video in ("VID01", "VID45", "VID80"):
        case = b1.case_id_for_video(video)
        assert normalize_case_id(case) == case          # train_vlm accepts it
        assert int(case.split("_")[1]) > 154            # cannot be a SurgVU case


def test_case_id_rejects_an_unparseable_video():
    with pytest.raises(ValueError, match="unparseable video id"):
        b1.case_id_for_video("not-a-video")


# -- answer formatting ------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("grasper", "Grasper"),
    ("grasper, hook", "Grasper and Hook"),
    ("liver, gut, omentum", "Liver, Gut and Omentum"),
    ("cystic_plate", "Cystic Plate"),
    ("abdominal_wall_cavity, liver", "Abdominal Wall Cavity and Liver"),
])
def test_answer_form_matches_surgvu_shape(raw, expected):
    """SurgVU's own list form: Title Case, comma-separated, final 'and'.
    The NOUNS are left alone -- CholecT45's taxonomy is not SurgVU's, and
    rewriting them would teach wrong names."""
    assert b1.format_answer(raw) == expected


def test_empty_annotation_is_dropped_not_emitted_as_empty():
    """SSG-VQA writes a bare '' when no tool is present (2,780 of the kept
    lines). An empty answer is not a training target: it teaches the model to
    emit nothing, and a missing response scores 0 at grading."""
    assert b1.format_answer("") is None
    assert b1.format_answer("   ") is None
    assert b1.format_answer(" , ,  ") is None


# -- window clamping --------------------------------------------------------

def test_window_is_sixteen_frames():
    available = list(range(100))
    assert len(b1.window_frame_indices(50, available)) == 16


def test_window_at_the_start_slides_inward_rather_than_wrapping():
    """Wrapping would put frames from the END of the operation -- a completely
    different phase -- into the same clip."""
    available = list(range(100))
    got = b1.window_frame_indices(0, available)
    assert got == list(range(16))
    assert max(got) < 90            # nothing from the far end


def test_window_at_the_end_slides_inward():
    available = list(range(100))
    got = b1.window_frame_indices(99, available)
    assert got == list(range(84, 100))


def test_window_is_contiguous_in_the_available_index():
    """Not merely 16 frames -- 16 CONSECUTIVE ones, or the clip is not a clip."""
    available = list(range(0, 200, 2))          # a video with gaps
    got = b1.window_frame_indices(100, available)
    position = available.index(got[0])
    assert got == available[position:position + 16]


def test_window_returns_none_for_a_video_too_short_to_fill_one():
    assert b1.window_frame_indices(2, [0, 1, 2, 3]) is None


# -- record construction ----------------------------------------------------

def _annotations():
    return [("VID01", 50, "tool_identity_open", "Grasper and Hook"),
            ("VID01", 51, "organ_open", "Liver")]


def test_records_carry_sixteen_real_frame_paths():
    records = b1.build_records(_annotations(), {"VID01": list(range(100))}, "/frames")
    assert len(records) == 2
    for record in records:
        assert len(record["frame_paths"]) == b1.FRAMES_PER_WINDOW
        assert len(set(record["frame_paths"])) == b1.FRAMES_PER_WINDOW
        assert all(p.endswith(".jpg") for p in record["frame_paths"])


def test_records_have_every_key_train_vlm_reads():
    records = b1.build_records(_annotations(), {"VID01": list(range(100))}, "/frames")
    for key in ("case", "part", "t_start", "t_stop", "question", "answer",
                "intent", "frame_paths", "frame_dir"):
        assert key in records[0], key


def test_a_centre_frame_with_no_image_is_skipped():
    """An annotation whose frame is absent from the zip must drop, not emit a
    record pointing at a file that will never exist."""
    records = b1.build_records(_annotations(), {"VID01": [0, 1, 2]}, "/frames")
    assert records == []


def test_question_forms_vary_within_an_intent():
    """A single question string for 27k records would train the model on one
    phrasing -- the exact brittleness stage 1 exists to fix."""
    annotations = [("VID01", i, "tool_identity_open", "Grasper") for i in range(30, 40)]
    records = b1.build_records(annotations, {"VID01": list(range(100))}, "/frames")
    assert len({r["question"] for r in records}) > 1


def test_only_the_two_transferable_shapes_are_kept():
    """The other 94.5% of SSG-VQA is templated spatial reasoning whose register
    stage 2 would have to unteach."""
    assert {intent for _, intent in b1.KEPT_SHAPES} == {"tool_identity_open", "organ_open"}


# --------------------------------------------------------------------------
# stage-1 frame conversion: geometry and encoding must match stage 2
# --------------------------------------------------------------------------

_CSPEC = importlib.util.spec_from_file_location(
    "convert_stage1_frames",
    Path(__file__).resolve().parents[1] / "scripts" / "convert_stage1_frames.py")
conv = importlib.util.module_from_spec(_CSPEC)
_CSPEC.loader.exec_module(conv)


def _png_bytes(h, w):
    import cv2
    import numpy as np
    img = (np.random.rand(h, w, 3) * 255).astype("uint8")
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.mark.parametrize("h,w", [(1080, 1920), (480, 854), (512, 512), (900, 900)])
def test_converted_frames_match_prepare_frame_geometry(h, w):
    """THE PARITY THAT MATTERS ACROSS THE CURRICULUM.

    Stage 2's frames come out of preprocess.prepare_frame, which SQUASHES to
    a square (its last line is cv2.resize(frame, (size, size))). A stage 1
    that letterboxed instead -- the obvious implementation, and the one this
    started as -- would hand the model two different frame geometries and
    teach it the difference carried meaning.
    """
    import cv2
    import numpy as np

    from surgvu.preprocess import prepare_frame

    payload = _png_bytes(h, w)
    out = conv.convert_one(payload, 512)
    assert out is not None
    mine = cv2.imdecode(np.frombuffer(out, dtype=np.uint8), cv2.IMREAD_COLOR)
    theirs = prepare_frame(np.zeros((h, w, 3), dtype="uint8"), 512)
    assert mine.shape == theirs.shape == (512, 512, 3)


def test_conversion_uses_the_same_jpeg_quality_as_stage_two():
    """Imported from surgvu.extract, not restated. Written as a literal first,
    it said 92; the real value is 90."""
    from surgvu.extract import JPEG_QUALITY

    assert conv.JPEG_QUALITY == JPEG_QUALITY == 90


def test_undecodable_frame_returns_none_rather_than_raising():
    """One corrupt member in 90,728 must not kill a multi-hour job."""
    assert conv.convert_one(b"this is not a png", 512) is None


def test_converter_reads_in_bounded_batches_not_one_eager_map():
    """THE BUG THIS ENCODES, MEASURED NOT GUESSED.

    ThreadPoolExecutor.map() drains its iterable EAGERLY -- it submits every
    item before yielding the first result. The first version handed it a
    generator that read each zip member as it went, which pulled all 90,728
    PNGs (59GB) into memory at once: the job reached 9,766MB against an 8GB
    request and was held by the scheduler.

    Asserted on the source because the failure is a memory profile, not a
    return value -- there is nothing to assert on the output, which is why it
    got through review the first time.
    """
    import inspect

    source = inspect.getsource(conv.main)
    assert "BATCH_SIZE" in source
    assert "for start in range(0, len(members), BATCH_SIZE)" in source


def test_batch_size_is_bounded_and_small_enough_to_fit():
    """256 x ~500KB is ~128MB live, comfortably inside an 8GB request."""
    assert 1 <= conv.BATCH_SIZE <= 1024
