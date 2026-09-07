"""decode_clip_multiscale must not disturb what already ships.

The existing serving path decodes centres one way and the appearance model
has been measured on exactly those pixels. A new sampler that returns even
slightly different centres would move shipped answers for a reason unrelated
to the evidence being added, so the first assertion here is equality with
decode_clip -- the same guarantee tests/test_perceive.py already makes for
decode_clip_bursts.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.frames import sample_frame_indices              # noqa: E402
from surgvu.perceive import decode_clip, decode_clip_multiscale  # noqa: E402
from surgvu.preprocess import prepare_frame                  # noqa: E402


@pytest.fixture()
def clip(tmp_path):
    """A 120-frame 30 fps synthetic clip with a moving bright square."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             30.0, (256, 256))
    rng = np.random.default_rng(0)
    background = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    for i in range(120):
        frame = background.copy()
        x = 10 + i
        frame[100:140, x:x + 40] = 255
        writer.write(frame)
    writer.release()
    return path


def test_centres_are_identical_to_decode_clip(clip):
    plain = decode_clip(clip, n_frames=8, size=128)
    centres, _ = decode_clip_multiscale(clip, n_frames=8, size=128)
    assert np.array_equal(plain, centres)


def test_one_probe_entry_per_centre(clip):
    centres, probes = decode_clip_multiscale(clip, n_frames=8, size=128)
    assert len(probes) == len(centres)


def test_each_probe_covers_every_requested_offset(clip):
    offsets = (133, 400)
    _, probes = decode_clip_multiscale(clip, n_frames=8, size=128,
                                       offsets_ms=offsets)
    for entry in probes:
        assert set(entry) == set(offsets)


def test_unavailable_offset_is_none_not_a_duplicated_frame(clip):
    """At the clip edges a wide offset runs off the end.

    Clamping would difference a frame against itself and read as stillness --
    a lie with a plausible value. The measurement must be absent instead.
    """
    _, probes = decode_clip_multiscale(clip, n_frames=8, size=128,
                                       offsets_ms=(1200,))
    assert probes[0][1200] is None


def test_probe_pairs_are_preprocessed_frames(clip):
    _, probes = decode_clip_multiscale(clip, n_frames=8, size=128,
                                       offsets_ms=(133,))
    pair = probes[len(probes) // 2][133]
    assert pair is not None
    before, after = pair
    assert before.shape == (128, 128, 3)
    assert after.shape == (128, 128, 3)
    assert before.dtype == np.uint8


# ------------------------------------------------------------- index_range
#
# Ruling R15: decode_clip_multiscale gained an ADDITIVE `index_range`
# parameter so scripts/dump_motion_v2.py can seek directly to one stratified
# window of a multi-hour source video, instead of cutting a temporary clip
# with cv2.VideoWriter (which re-encodes, and this whole task calibrates a
# pixel-magnitude statistic against a threshold meant for the ORIGINAL h264).
# The first test below is the one guarantee that makes this safe to add:
# `index_range=None` (every existing call site) must be byte-identical to
# today.

def test_index_range_full_span_matches_the_default(clip):
    """(0, total-1) must be indistinguishable from index_range=None -- the
    equivalence the additive parameter's default is supposed to preserve."""
    capture = cv2.VideoCapture(str(clip))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    default_centres, _ = decode_clip_multiscale(clip, n_frames=8, size=128)
    ranged_centres, _ = decode_clip_multiscale(
        clip, n_frames=8, size=128, index_range=(0, total - 1))
    assert np.array_equal(default_centres, ranged_centres)


def test_index_range_restricts_sampling_to_the_given_window(clip):
    """Centres from a restricted range must equal frames read directly at
    the indices sample_frame_indices would pick WITHIN that range, offset by
    its start -- not the indices a whole-file sample would have picked."""
    first, last = 60, 89   # a 30-frame window in the back half of the clip
    n = 4
    centres, _ = decode_clip_multiscale(clip, n_frames=n, size=128,
                                        index_range=(first, last))
    assert centres.shape[0] == n

    expected_indices = [first + i for i in
                        sample_frame_indices(last - first + 1, n)]
    capture = cv2.VideoCapture(str(clip))
    try:
        expected = []
        for index in expected_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            assert ok
            expected.append(prepare_frame(frame, size=128))
    finally:
        capture.release()
    assert np.array_equal(centres, np.stack(expected))


def test_index_range_probes_may_reach_outside_the_range_but_not_the_file(clip):
    """A probe offset is bounded by the FILE's own edges, not the requested
    range -- a window near the start of a task interval should still be able
    to probe slightly before it into real, decodable footage."""
    _, probes = decode_clip_multiscale(clip, n_frames=4, size=128,
                                       offsets_ms=(67,), index_range=(50, 90))
    # None of these anchors sits within 67ms-in-frames of the FILE's edges
    # (0 or total-1), so every probe should be available.
    assert all(entry[67] is not None for entry in probes)


def test_index_range_out_of_bounds_raises(clip):
    capture = cv2.VideoCapture(str(clip))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    with pytest.raises(ValueError):
        decode_clip_multiscale(clip, n_frames=4, size=128,
                               index_range=(0, total))  # last == total: OOB


def test_index_range_inverted_bounds_raises(clip):
    with pytest.raises(ValueError):
        decode_clip_multiscale(clip, n_frames=4, size=128,
                               index_range=(50, 10))
