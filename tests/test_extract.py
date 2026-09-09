import json
import os

import numpy as np
import cv2
import pytest

from surgvu.extract import (
    JPEG_QUALITY, extract_window, read_shard, shard_filename, write_shard,
)
from surgvu.sampling import Window


@pytest.fixture
def synthetic_video(tmp_path):
    """60 seconds at 30 fps, frame index encoded in pixel brightness."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             30.0, (320, 240))
    for i in range(1800):
        frame = np.full((240, 320, 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _window(start, part="1.0", length=30.0):
    return Window(case="c", part=part, start=start, length=length,
                  task="suturing", description="d",
                  tools=frozenset({"needle driver"}))


def test_extract_window_returns_one_frame_per_second(synthetic_video):
    frames = extract_window(synthetic_video, _window(10.0), "1.0", fps=1, size=64)
    assert len(frames) == 30
    assert frames[0].shape == (64, 64, 3)


def test_extract_window_starts_at_the_right_place(synthetic_video):
    early = extract_window(synthetic_video, _window(0.0), "1.0", fps=1, size=64)
    late = extract_window(synthetic_video, _window(30.0), "1.0", fps=1, size=64)
    assert early[0].mean() != late[0].mean()


# The synthetic clip is written through a lossy mp4v encoder, so a frame
# authored at brightness 30 decodes at roughly 27.3, not exactly 30. The
# observed round-trip error across the whole clip is under 4 levels; 6 leaves
# headroom without admitting a single frame of time-base slip, which is worth
# at least 30 levels at 30 fps.
_CODEC_TOLERANCE = 6.0
_SOURCE_FPS = 30.0


@pytest.mark.parametrize("start", [0.0, 10.0, 20.0, 40.0])
def test_extract_window_frames_land_on_the_exact_source_frame(synthetic_video, start):
    """The frame at time t must be source frame round(t * 30), exactly.

    Every label in this project is DERIVED from a timestamp rather than read
    off an image, so a wrong time base mislabels the entire corpus and nothing
    downstream can see it. The fixture encodes the source frame index in pixel
    brightness precisely so this can be asserted rather than approximated:
    asserting only that two different starts give different means passes just
    as happily if every index is scaled by 0.5.
    """
    frames = extract_window(synthetic_video, _window(start), "1.0", fps=1, size=64)
    assert frames

    for j, frame in enumerate(frames):
        expected = int(round((start + j) * _SOURCE_FPS)) % 256
        actual = float(frame.mean())
        assert abs(actual - expected) <= _CODEC_TOLERANCE, (
            "frame %d of the window starting at %.1fs decoded as brightness "
            "%.2f, i.e. source frame ~%d, but should be source frame %d"
            % (j, start, actual, int(round(actual)),
               int(round((start + j) * _SOURCE_FPS))))


def test_consecutive_extracted_frames_are_exactly_one_second_apart(synthetic_video):
    """Independent of where the window starts: at fps=1 over a 30 fps source,
    successive frames must be 30 source frames apart. This pins the SCALE of
    the time base, where the test above pins its ORIGIN."""
    frames = extract_window(synthetic_video, _window(5.0), "1.0", fps=1, size=64)
    assert len(frames) >= 2

    for a, b in zip(frames, frames[1:]):
        step = (float(b.mean()) - float(a.mean())) % 256
        assert abs(step - _SOURCE_FPS) <= _CODEC_TOLERANCE, step


def test_extract_window_honours_a_non_default_window_length(synthetic_video):
    """Window length round-trips from enumeration through extraction. It used
    to come from a module constant, so a 15-second window was decoded as 30."""
    short = extract_window(synthetic_video, _window(0.0, length=15.0), "1.0",
                           fps=1, size=64)
    assert len(short) == 15

    long = extract_window(synthetic_video, _window(0.0, length=30.0), "1.0",
                          fps=1, size=64)
    assert len(long) == 30


def test_extract_window_refuses_a_window_from_a_different_part(synthetic_video):
    """Timestamps reset at the part boundary, so decoding a part-2 window
    from the part-1 video returns real frames from the wrong moment -- the
    single most damaging silent failure available in this pipeline."""
    part_two_window = _window(10.0, part="2.0")

    with pytest.raises(ValueError, match="part mismatch"):
        extract_window(synthetic_video, part_two_window, "1.0", fps=1, size=64)

    # And the matching part decodes normally.
    assert extract_window(synthetic_video, part_two_window, "2.0", fps=1, size=64)


def test_extract_window_accepts_equivalent_spellings_of_the_same_part(synthetic_video):
    """CSVs say '1.0', filenames say '001', humans type '1'."""
    for spelling in ("1", "1.0", "001", 1):
        assert extract_window(synthetic_video, _window(0.0), spelling,
                              fps=1, size=64)


def test_shard_filename_carries_the_part():
    """Both of a case's jobs used to write case_056.npz, so whichever landed
    last won and reruns were non-deterministic."""
    assert shard_filename("case_056", "1.0") == "case_056_part1.npz"
    assert shard_filename("case_056", "2.0") == "case_056_part2.npz"
    assert shard_filename("case_056", "1.0") != shard_filename("case_056", "2.0")


def test_extract_window_past_end_returns_short_or_empty(synthetic_video):
    frames = extract_window(synthetic_video, _window(200.0), "1.0", fps=1, size=64)
    assert len(frames) < 30


_REAL_VIDEO_CAPTURE = cv2.VideoCapture


class _ZeroFPSCapture:
    """Wraps a real cv2.VideoCapture but reports CAP_PROP_FPS as unreadable,
    to exercise the fallback path without needing an actually-corrupt file."""

    def __init__(self, path):
        self._real = _REAL_VIDEO_CAPTURE(path)

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 0.0
        return self._real.get(prop)

    def set(self, prop, value):
        return self._real.set(prop, value)

    def read(self):
        return self._real.read()

    def release(self):
        return self._real.release()


def test_extract_window_warns_when_fps_unreadable_and_uses_default(synthetic_video, monkeypatch, capsys):
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: _ZeroFPSCapture(path))

    frames = extract_window(synthetic_video, _window(0.0), "1.0", fps=1, size=64, default_fps=30.0)

    out = capsys.readouterr().out
    assert "falling back to default_fps=30.0" in out
    assert str(synthetic_video) in out
    assert len(frames) > 0


def test_shard_roundtrip(tmp_path):
    frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(3)]
    payload = [(_window(0.0), frames)]
    path = tmp_path / "case_x.npz"
    # NOTE: write_shard requires an EXACT frame count rather than a minimum,
    # and derives it from `window.length * fps` (here 30 * 1 = 30). This
    # fixture window holds 3 frames, so it needs the frames_per_window
    # override to survive. This test exercises npz save/load roundtrip
    # mechanics; the drop/report/exact-length behavior is covered below.
    write_shard(payload, path, frames_per_window=3)

    stack, meta = read_shard(path)
    assert stack.shape == (1, 3, 8, 8, 3)
    assert meta[0]["task"] == "suturing"
    assert meta[0]["tools"] == ["needle driver"]
    assert meta[0]["case"] == "c"


def test_write_shard_drops_windows_not_matching_exact_length_and_reports_it(tmp_path, capsys):
    """A short window must not truncate the survivors, and the drop must be
    reported, not silent. Surviving metadata must line up 1:1 with the
    surviving frame stack -- if a dropped window's frames vanish but its
    metadata entry lingers (or vice versa), every downstream label is
    attached to the wrong frames."""
    full_a = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(5)]
    short = [np.full((4, 4, 3), 99, dtype=np.uint8) for _ in range(3)]
    full_b = [np.full((4, 4, 3), i + 10, dtype=np.uint8) for i in range(5)]
    payload = [
        (_window(0.0), full_a),
        (_window(30.0), short),
        (_window(60.0), full_b),
    ]
    path = tmp_path / "case_mix.npz"
    write_shard(payload, path, frames_per_window=5)

    out = capsys.readouterr().out
    assert "dropped 1/3" in out

    stack, meta = read_shard(path)
    assert stack.shape == (2, 5, 4, 4, 3)          # neither survivor truncated
    assert [m["start"] for m in meta] == [0.0, 60.0]   # aligned, short one skipped
    assert len(meta) == stack.shape[0]


def test_write_shard_raises_when_no_window_has_the_exact_length(tmp_path):
    short = [np.full((4, 4, 3), 1, dtype=np.uint8) for _ in range(3)]
    payload = [(_window(0.0), short)]
    with pytest.raises(ValueError):
        write_shard(payload, tmp_path / "empty.npz", frames_per_window=30)


def test_write_shard_derives_expected_depth_from_the_window_length(tmp_path):
    """The expected frame count used to be hardcoded at 30, so a 15-second
    window's 15 frames were dropped as 'short'. It now follows the window."""
    fifteen = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(15)]
    payload = [(_window(0.0, length=15.0), fifteen)]

    write_shard(payload, tmp_path / "short.npz", fps=1)

    stack, meta = read_shard(tmp_path / "short.npz")
    assert stack.shape == (1, 15, 4, 4, 3)
    assert meta[0]["length"] == 15.0


def test_shard_metadata_records_how_the_frames_were_produced(tmp_path):
    """A shard must carry a machine-checkable claim about its own provenance
    rather than relying on convention. `ui_blurred` is the one that matters:
    blurring the UI band is a challenge-rules requirement, and a consumer
    should be able to refuse a shard that does not assert it."""
    frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(3)]
    path = tmp_path / "case_prov.npz"
    write_shard([(_window(0.0), frames)], path, fps=2, frames_per_window=3)

    _stack, meta = read_shard(path)
    row = meta[0]
    assert row["ui_blurred"] is True
    assert row["frame_size"] == 8
    assert row["fps"] == 2
    assert row["length"] == 30.0
    assert row["part"] == "1.0"


def test_write_shard_refuses_windows_of_mixed_length(tmp_path):
    """Windows of different lengths cannot share one stacked array; silently
    stacking them is impossible and silently dropping one class of them would
    be worse."""
    frames = [np.full((4, 4, 3), 1, dtype=np.uint8) for _ in range(30)]
    payload = [
        (_window(0.0, length=30.0), frames),
        (_window(60.0, length=15.0), frames[:15]),
    ]
    with pytest.raises(ValueError, match="same length"):
        write_shard(payload, tmp_path / "mixed.npz", fps=1)


def _edge_frame():
    """A bright, hard vertical edge against a dark field.

    Stands in for an instrument shaft against tissue: fine metal detail
    against a darker background is exactly what the classifier needs and
    exactly what JPEG smears first.
    """
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, 32:] = 240
    return frame


def test_shard_frames_survive_the_jpeg_roundtrip_with_bounded_loss(tmp_path):
    """Frames are stored as JPEG now, so the roundtrip is lossy on purpose.

    Bounded loss only -- edge preservation is a separate property and gets
    its own test, so that a mutation to either one is provably caught rather
    than masked by whichever assertion happens to run first.
    """
    frame = _edge_frame()
    frames = [frame.copy() for _ in range(3)]
    path = tmp_path / "case_jpeg.npz"

    write_shard([(_window(0.0), frames)], path, frames_per_window=3)
    stack, meta = read_shard(path)

    assert stack.shape == (1, 3, 64, 64, 3)
    assert stack.dtype == np.uint8
    assert np.abs(stack[0][0].astype(np.int16)
                  - frame.astype(np.int16)).mean() < 2.0


def test_jpeg_roundtrip_keeps_instrument_edges_crisp(tmp_path):
    """The edge must not migrate or soften into a ramp.

    Deliberately asserts nothing about overall error: this test exists to
    die when the frame is blurred, and a blur that leaves mean error small
    must still fail here.
    """
    frame = _edge_frame()
    path = tmp_path / "case_edge.npz"

    write_shard([(_window(0.0), [frame.copy() for _ in range(3)])], path,
                frames_per_window=3)
    decoded = read_shard(path)[0][0][0].astype(np.int16)

    assert decoded[32, 20].mean() < 40          # dark side stays dark
    assert decoded[32, 44].mean() > 200         # bright side stays bright
    # The transition must complete in ONE pixel. Measured at q90 the profile
    # is 0,0 | 240,240 across columns 30-33; a 7x7 blur ramps it to 33,86 |
    # 154,205. Asserting a loose gradient across columns 30->33 passes under
    # that blur, so the thresholds sit hard against the measured values.
    assert decoded[32, 31].mean() < 30
    assert decoded[32, 32].mean() > 200


def test_shard_preserves_frame_order_within_a_window(tmp_path):
    """Frames must come back in the order they went in.

    The shard flattens every window's frames into one buffer and rebuilds
    them by offset arithmetic, so a reversed or rotated index would still
    produce correctly-shaped output carrying the right metadata -- and the
    action labels would silently describe time running backwards.
    """
    frames = [np.full((8, 8, 3), value, dtype=np.uint8)
              for value in (10, 20, 30, 40)]
    path = tmp_path / "case_order.npz"

    write_shard([(_window(0.0), frames)], path, frames_per_window=4)
    stack, _meta = read_shard(path)

    assert [round(float(f.mean())) for f in stack[0]] == [10, 20, 30, 40]


def test_shard_keeps_windows_in_order_and_aligned_with_their_metadata(tmp_path):
    """Window order matters as much as frame order: metadata row i must
    describe frame stack i, or every label lands on the wrong footage."""
    payload = [
        (_window(0.0), [np.full((8, 8, 3), 10, dtype=np.uint8)] * 3),
        (_window(30.0), [np.full((8, 8, 3), 20, dtype=np.uint8)] * 3),
        (_window(60.0), [np.full((8, 8, 3), 30, dtype=np.uint8)] * 3),
    ]
    path = tmp_path / "case_walign.npz"

    write_shard(payload, path, frames_per_window=3)
    stack, meta = read_shard(path)

    assert [m["start"] for m in meta] == [0.0, 30.0, 60.0]
    assert [round(float(s[0].mean())) for s in stack] == [10, 20, 30]


def test_shard_records_the_default_jpeg_quality(tmp_path):
    """Provenance: the recorded quality must track the encoder's actual one."""
    frames = [np.full((8, 8, 3), i * 20, dtype=np.uint8) for i in range(3)]
    path = tmp_path / "case_q90.npz"

    write_shard([(_window(0.0), frames)], path, frames_per_window=3)

    assert read_shard(path)[1][0]["jpeg_quality"] == JPEG_QUALITY


def test_shard_records_a_non_default_jpeg_quality(tmp_path):
    """Provenance must state the quality actually used, not the default."""
    frames = [np.full((8, 8, 3), i * 20, dtype=np.uint8) for i in range(3)]
    path = tmp_path / "case_q.npz"

    write_shard([(_window(0.0), frames)], path, frames_per_window=3,
                jpeg_quality=60)

    _stack, meta = read_shard(path)
    assert meta[0]["jpeg_quality"] == 60


# --------------------------------------------------------------------------
# read_shard decodes on demand
# --------------------------------------------------------------------------
# It used to cv2.imdecode every frame of the shard up front and np.stack the
# result. A real shard is ~1.3 GB decoded against ~165 MB of JPEG, the stack
# doubles that at its peak, and this runs once per DataLoader worker -- four
# workers is what took job 9618015 past a 32 GB cgroup limit. Training also
# only ever uses 8 of the 16 frames in a window, so half of that decode was
# thrown away.
#
# This is the TRAINING path only. Serving decodes video with `decode_clip`
# and never opens a shard.

@pytest.fixture
def counted_imdecode(monkeypatch):
    """Every cv2.imdecode call, counted."""
    calls = []
    real = cv2.imdecode

    def counting(buffer, flags):
        calls.append(1)
        return real(buffer, flags)

    monkeypatch.setattr(cv2, "imdecode", counting)
    return calls


def _fds_pointing_at(path):
    """Open file descriptors in this process resolving to `path`."""
    fds = []
    for entry in os.listdir("/proc/self/fd"):
        try:
            fds.append(os.readlink("/proc/self/fd/%s" % entry))
        except OSError:
            continue
    return [f for f in fds if f == str(path)]


def _shard_of(tmp_path, windows=4, depth=5, name="case_lazy.npz"):
    payload = [(_window(30.0 * w),
                [np.full((8, 8, 3), (w * depth + f) % 256, dtype=np.uint8)
                 for f in range(depth)])
               for w in range(windows)]
    path = tmp_path / name
    write_shard(payload, path, frames_per_window=depth)
    return path


def test_read_shard_decodes_nothing_until_a_frame_is_asked_for(
        tmp_path, counted_imdecode):
    path = _shard_of(tmp_path)
    del counted_imdecode[:]                     # writing the shard ENCODES only

    stack, meta = read_shard(path)

    assert len(meta) == 4
    assert not counted_imdecode, (
        "read_shard decoded %d frames before anything asked for one"
        % len(counted_imdecode))


def test_reading_one_frame_decodes_exactly_one_frame(tmp_path, counted_imdecode):
    path = _shard_of(tmp_path)
    stack, _meta = read_shard(path)
    del counted_imdecode[:]

    frame = stack[2][3]

    assert frame.shape == (8, 8, 3)
    assert len(counted_imdecode) == 1


def test_the_shape_is_available_without_decoding_the_shard(
        tmp_path, counted_imdecode):
    """`shape` is what the loader reads to find the window depth. Answering it
    by decoding 20 frames would put the eager cost back under a property
    access."""
    path = _shard_of(tmp_path, windows=4, depth=5)
    stack, _meta = read_shard(path)
    del counted_imdecode[:]

    assert stack.shape == (4, 5, 8, 8, 3)
    assert stack.dtype == np.uint8
    assert len(counted_imdecode) <= 1, (
        "shape/dtype cost %d decodes; at most one probe frame is defensible"
        % len(counted_imdecode))


def test_a_lazy_frame_is_the_same_array_the_eager_reader_returned(tmp_path):
    """The point of laziness is WHEN the decode happens, not what it returns."""
    path = _shard_of(tmp_path, windows=3, depth=4)
    stack, meta = read_shard(path)

    with np.load(path, allow_pickle=False) as payload:
        blob, offsets = payload["jpeg"], payload["offsets"]
        windows, depth = (int(x) for x in payload["shape"])
    eager = np.stack([np.stack([
        cv2.imdecode(blob[offsets[w * depth + f]:offsets[w * depth + f + 1]],
                     cv2.IMREAD_COLOR)
        for f in range(depth)]) for w in range(windows)])

    assert stack.shape == eager.shape
    assert stack.dtype == eager.dtype
    for w in range(windows):
        for f in range(depth):
            assert np.array_equal(stack[w][f], eager[w][f]), (w, f)
    assert [round(float(f.mean())) for f in stack[1]] == \
        [round(float(f.mean())) for f in eager[1]]
    assert [round(float(s[0].mean())) for s in stack] == \
        [round(float(s[0].mean())) for s in eager]


def test_a_lazy_shard_keeps_no_open_handle_on_the_npz(tmp_path):
    """Lazy means the DECODE is deferred, not the read.

    The compressed JPEG buffer is loaded once and the npz is closed. Deferring
    the read instead would keep a file descriptor per worker per shard open for
    the length of training, and would re-read from shared storage on every
    frame -- which is the cost this loader's shard-at-a-time design exists to
    avoid. What is saved is the 8x expansion from decoding, not the I/O.
    """
    path = _shard_of(tmp_path)

    stack, _meta = read_shard(path)

    assert not _fds_pointing_at(path), "read_shard left the npz open"
    path.unlink()
    assert stack[1][2].shape == (8, 8, 3)


def test_an_out_of_range_window_or_frame_raises_indexerror(tmp_path):
    path = _shard_of(tmp_path, windows=2, depth=3)
    stack, _meta = read_shard(path)

    for bad in (2, -3):
        with pytest.raises(IndexError):
            stack[bad]
    for bad in (3, -4):
        with pytest.raises(IndexError):
            stack[0][bad]


def test_negative_indices_count_from_the_end_as_they_did_on_the_array(tmp_path):
    """The eager reader handed back an ndarray, where `stack[-1][-1]` is the
    last frame of the last window. Silently decoding some other frame instead
    would be indistinguishable from correct output."""
    path = _shard_of(tmp_path, windows=2, depth=3)
    stack, _meta = read_shard(path)

    assert np.array_equal(stack[-1][-1], stack[1][2])
    assert np.array_equal(stack[-2][-3], stack[0][0])
    assert [round(float(f.mean())) for f in stack[-1]] == \
        [round(float(f.mean())) for f in stack[1]]


def test_a_corrupt_jpeg_names_the_window_and_frame_it_failed_on(tmp_path):
    """The eager reader raised on the offending window/frame at load time.
    Lazily it raises when that frame is reached, and must still say which one:
    "cv2.imdecode returned None" with no coordinates is not a diagnosis."""
    path = _shard_of(tmp_path, windows=2, depth=3)
    with np.load(path, allow_pickle=False) as payload:
        parts = {name: payload[name] for name in payload.files}
    parts["jpeg"] = np.zeros_like(parts["jpeg"])        # nothing decodes now
    np.savez(path, **parts)

    stack, _meta = read_shard(path)

    with pytest.raises(ValueError, match="window 1 frame 2"):
        stack[1][2]


def test_a_pre_jpeg_shard_comes_back_as_the_array_it_was_stored_as(tmp_path):
    """Shards written before JPEG storage hold raw frames under `frames`.

    There is nothing compressed there, so there is nothing to decode and
    nothing to defer -- the branch returns the ndarray unchanged. Untested,
    the laziness rewrite could have routed these through the JPEG path and
    raised KeyError on the first old shard anyone handed it.
    """
    frames = np.arange(2 * 3 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 4, 3)
    path = tmp_path / "case_legacy.npz"
    np.savez(path, frames=frames,
             meta=json.dumps([{"start": 0.0}, {"start": 30.0}]))

    stack, meta = read_shard(path)

    assert isinstance(stack, np.ndarray)
    assert np.array_equal(stack, frames)
    assert stack.shape == (2, 3, 4, 4, 3)
    assert [m["start"] for m in meta] == [0.0, 30.0]
