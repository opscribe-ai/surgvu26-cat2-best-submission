"""Decode each video once and emit training shards.

Decoding 344 GB of 60 fps video is the expensive step in this project, so it
happens exactly once and all three models are trained from the same frame
pool -- they differ only in labels and in how many consecutive frames they use.

The unit of work is a `(case, part)` pair, never a case. Cases split into up
to two video files and **timestamps reset at the part boundary**, so a part-2
window's `start` is meaningless against the part-1 video. `extract_window`
therefore demands to be told which part the video it was handed represents,
and refuses to decode a window from the wrong one -- a mismatch there produces
frames that look perfectly plausible and are labelled with someone else's
timestamps, which no loss curve would ever reveal.
"""
import json
import re

import cv2
import numpy as np

from .labels import normalize_part
from .preprocess import prepare_frame


def shard_filename(case_id, part):
    """Canonical shard name for one `(case, part)` job: `case_056_part1.npz`.

    The part must be in the name. Keying shards by case alone let both of a
    case's Condor jobs write the same staging path, so whichever landed last
    won and reruns were non-deterministic.
    """
    return "%s_part%s.npz" % (case_id, part_number(part))


def part_number(part):
    """'2.0' / '2' / 2 / '002' -> '2'. Raises on anything else."""
    canonical = normalize_part(part)
    match = re.fullmatch(r"(\d+)\.0", canonical or "")
    if not match:
        raise ValueError("cannot read a part number from %r" % (part,))
    return str(int(match.group(1)))


def extract_window(video_path, window, part, fps=1, size=512, default_fps=60.0):
    """Decode one window as `fps` frames per second, preprocessed.

    `part` names the part that `video_path` contains, and it is required. If
    it disagrees with `window.part` this raises immediately rather than
    decoding: timestamps reset at the part boundary, so a part-2 window read
    from the part-1 video silently yields real frames from the wrong moment.

    The number of frames comes from `window.length`, not from a module
    constant, so a window enumerated at a non-default length is decoded at
    that length.

    `default_fps` is the fallback used when the container doesn't report a
    readable CAP_PROP_FPS. It defaults to 60.0 to match this corpus's real
    frame rate -- silently falling back to some other rate would compute
    every frame index at the wrong offset and mislabel the whole video
    without any error. Callers working with already-1-fps test-format clips
    should pass `default_fps=1.0` explicitly. The fallback firing at all is
    unusual enough to warn about, naming the video, every time it happens.
    """
    video_part = normalize_part(part)
    if video_part != normalize_part(window.part):
        raise ValueError(
            "part mismatch: window %s/%s start=%s belongs to part %s but the "
            "video supplied (%s) is part %s. Timestamps reset at the part "
            "boundary, so decoding this would return frames from the wrong "
            "moment." % (window.case, window.part, window.start,
                         normalize_part(window.part), video_path, video_part))

    capture = cv2.VideoCapture(str(video_path))
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not source_fps:
            print("extract_window: %s reported no readable fps, "
                  "falling back to default_fps=%s" % (video_path, default_fps))
            source_fps = default_fps
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        step = 1.0 / float(fps)
        frames = []
        for i in range(expected_frame_count(window, fps)):
            index = int(round((window.start + i * step) * source_fps))
            if index >= total:
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(prepare_frame(frame, size=size))
        return frames
    finally:
        capture.release()


def dense_window(window, seconds=2.0, anchor="center"):
    """The same window, narrowed to a short burst that can carry MOTION.

    The 1 fps shards mirror what serving samples, not what the video holds:
    the source is 60 fps and the graded test clips are 1800 frames over 30
    seconds. A 3D convolution over frames a second apart sees scene changes,
    not movement -- 59 of every 60 frames of actual motion are already gone
    before the model looks. So a temporal model needs frames spaced in
    HUNDREDTHS of a second, which means a shorter span, not more frames.

    The window's LABELS are carried over unchanged, and that is the point of
    doing it this way rather than enumerating short windows directly:

      * the label is a task segment covering the full 30 seconds, so any
        sub-span inside it inherits the same task and the same tool set;
      * the window count, the case split and the stratification all stay
        exactly as they are, so a dense shard is a drop-in A/B against the
        sparse one rather than a different dataset that also happens to be
        denser.

    `anchor` is "center" because a task segment's edges are where the label is
    least trustworthy -- segments overlap, and `task_at` resolves a moment by
    the shortest covering segment, so the middle of a window is the part most
    likely to actually contain the labelled activity.
    """
    if seconds > window.length:
        raise ValueError("a %.1fs burst does not fit in a %.1fs window"
                         % (seconds, window.length))
    if anchor == "center":
        start = window.start + (window.length - seconds) / 2.0
    elif anchor == "start":
        start = window.start
    else:
        raise ValueError("unknown anchor %r" % (anchor,))
    return window._replace(start=start, length=seconds)


def extract_window_dense(video_path, window, part, fps=15, size=512,
                         default_fps=60.0):
    """Decode `window` densely, reading SEQUENTIALLY instead of seeking.

    `extract_window` seeks per frame, which is right when frames are a second
    apart -- the decoder would otherwise walk 60 frames it throws away. Here
    the spacing is 4 frames, and a seek costs far more than decoding the 3
    frames in between: `CAP_PROP_POS_FRAMES` on a compressed stream forces the
    decoder back to the preceding keyframe and re-decodes forward. Seeking 30
    times per window across ~24,500 windows is the difference between a night
    and a weekend.

    So: seek ONCE to the burst start, then read straight through and keep
    every k-th frame. Same output as the seeking version, minus the cost.
    """
    video_part = normalize_part(part)
    if video_part != normalize_part(window.part):
        raise ValueError(
            "part mismatch: window %s/%s start=%s belongs to part %s but the "
            "video supplied (%s) is part %s. Timestamps reset at the part "
            "boundary, so decoding this would return frames from the wrong "
            "moment." % (window.case, window.part, window.start,
                         normalize_part(window.part), video_path, video_part))

    capture = cv2.VideoCapture(str(video_path))
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not source_fps:
            print("extract_window_dense: %s reported no readable fps, "
                  "falling back to default_fps=%s" % (video_path, default_fps))
            source_fps = default_fps
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        want = expected_frame_count(window, fps)

        # How many source frames apart the kept frames are. Clamped at 1: a
        # requested fps at or above the source rate cannot invent frames, and
        # a step of 0 would return the same frame `want` times -- a stack with
        # no motion in it at all, which trains silently and means nothing.
        stride = max(1, int(round(source_fps / float(fps))))
        first = int(round(window.start * source_fps))
        if first >= total:
            return []

        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        frames = []
        index = 0
        while len(frames) < want:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                frames.append(prepare_frame(frame, size=size))
            index += 1
        return frames
    finally:
        capture.release()


def multi_burst_windows(window, bursts=4, seconds=0.5334, spread=True):
    """One window -> several short bursts SPREAD ACROSS it, labels unchanged.

    WHY THIS EXISTS. `dense_window` narrows a 30-second window to a single
    burst at its centre, and every 3D result we have was measured on that. It
    hands the model 16 frames -- 1.07 s at 15 fps -- and asks it a question
    about 30 seconds, so the model answers from 3.6% of the labelled span
    while the 2D path samples 16 frames spread across the whole of it. Any
    comparison between them confounds "temporal modelling" with "saw 28x less
    of the window", and the 3D null we measured cannot distinguish the two.

    Spreading the bursts fixes exactly that and nothing else:

        centre only   [              ####              ]
        spread        [   ##      ##      ##      ##   ]

    Same total frame count per window, same labels, same window set, same case
    split. What changes is WHERE in the window the frames come from.

    IT ALSO PROVIDES THE MULTI-CLIP TRAINING FIX FOR FREE. `ShardClips` yields
    one clip per window per epoch while `ShardFrames` yields thirty frames, so
    at equal epochs the 3D models got ~30x fewer gradient samples per window
    than the 2D model did. Four bursts are four genuinely different clips of
    the same window, so a loader can yield four per epoch -- and at evaluation
    they average into a prediction that covers the window instead of a sliver
    of it.

    BURST LENGTH IS NOT ARBITRARY. 8 frames at 15 fps is 0.53 s, which is what
    the Kinetics-400 weights were trained on (16 frames at 25 fps = 0.64 s).
    A burst much shorter than that feeds the pretrained temporal filters an
    input outside the regime they were fitted in.

    The frame SPACING deliberately stays at 15 fps, the same as the centre-only
    pool. Changing spread and spacing together would leave a win unattributable
    to either.
    """
    if bursts < 1:
        raise ValueError("bursts must be >= 1, got %r" % (bursts,))
    span = bursts * seconds
    if span > window.length:
        raise ValueError(
            "%d bursts of %.3fs need %.2fs and the window is only %.2fs"
            % (bursts, seconds, span, window.length))
    if bursts == 1 or not spread:
        return [dense_window(window, seconds=seconds)]

    # Burst i is centred at (i + 0.5) / bursts through the window: evenly
    # spaced, symmetric, and never touching the edges -- where overlapping
    # task segments make the label least trustworthy, which is the same
    # reason `dense_window` anchors at the centre.
    out = []
    for index in range(bursts):
        centre = window.start + window.length * (index + 0.5) / bursts
        out.append(window._replace(start=centre - seconds / 2.0,
                                   length=seconds))
    return out


def extract_window_multi(video_path, window, part, bursts=4, seconds=0.5334,
                         fps=15, size=512, default_fps=60.0):
    """Frames for every burst of `window`, concatenated in time order.

    Returned as ONE list so a shard row still holds a single window: the
    bursts are contiguous runs inside it, `bursts` frames per run, and the
    boundaries are at multiples of len(frames)//bursts. A loader that wants
    one burst slices; a loader that wants them all has them in order.

    A short read anywhere makes the whole window ragged and unusable -- the
    runs would no longer be equal length and the boundary arithmetic above
    would silently point at the wrong frames -- so this returns [] rather than
    a partial window, and `write_shard` drops it with the others.
    """
    frames = []
    per_burst = int(round(seconds * fps))
    for burst in multi_burst_windows(window, bursts=bursts, seconds=seconds):
        got = extract_window_dense(video_path, burst, part, fps=fps, size=size,
                                   default_fps=default_fps)
        if len(got) != per_burst:
            return []
        frames.extend(got)
    return frames


def expected_frame_count(window, fps=1):
    """How many frames a window of `window.length` seconds holds at `fps`.

    One definition, used by both the decoder and the shard writer, so the two
    cannot drift apart.
    """
    return int(round(window.length * fps))


JPEG_QUALITY = 90


#: A JPEG smaller than this is treated as never written. A 512x512 frame at
#: quality 90 is several KB; 100 bytes cannot be one.
MIN_WRITTEN_JPEG_BYTES = 100


def looks_like_a_written_jpeg(path):
    """True when `path` is a plausible, completely-written JPEG.

    WHY `Path.exists()` IS NOT ENOUGH, MEASURED.
    Every resumable frame writer in this repo skipped work whose destination
    already existed. `Path.write_bytes` is NOT atomic, so a job killed
    mid-write leaves a ZERO-BYTE file -- which exists, so the re-run skips it,
    so it stays zero bytes forever.

    That is not hypothetical: the GI conversion was killed three times on
    2026-08-27, and v6 stage 1 died 3h49m into training with

        PIL.UnidentifiedImageError: cannot identify image file
        '.../gi_frames/78/7845c99806e10d20e035d075bb83b5edebe5746d.jpg'

    on a 0-byte file. `filter_records_with_frames` did not catch it either --
    it also only calls `.exists()`. Nothing between the writer and the
    DataLoader ever opened the image.

    The check is size-based rather than a full decode on purpose: it runs once
    per frame across hundreds of thousands of files on cephfs, where a stat is
    already ~54/second and a decode would be far worse. Size catches the
    zero-byte and severely-truncated cases, which is what an interrupted write
    actually produces.
    """
    try:
        return path.stat().st_size >= MIN_WRITTEN_JPEG_BYTES
    except OSError:
        return False



def write_shard(windows_and_frames, out_path, fps=1, frames_per_window=None,
                jpeg_quality=JPEG_QUALITY):
    """One .npz per (case, part): JPEG-encoded frames plus JSON metadata.

    Frames are stored as JPEG, not raw uint8. Raw, the selected corpus is
    ~580 GB (24,578 windows x 30 frames x 512x512x3); as JPEG it is ~38 GB.
    Both fit the allocation, but raw is 15x the I/O for no benefit.

    This is a LOSSY step, so `jpeg_quality` is recorded in every metadata row
    alongside `ui_blurred` / `frame_size` / `fps`: a shard states what models
    will actually see rather than leaving it to convention.

    JPEG blobs are ragged, so they cannot go into one rectangular array. They
    are concatenated into a single uint8 buffer with an offsets index, which
    keeps `read_shard` on `allow_pickle=False` -- an object array would have
    forced pickle back on just to store bytes.

    A ragged shard cannot be stacked. Rather than truncating every window
    down to whatever the shortest survivor happens to be -- which would clip
    real frames off windows that decoded completely fine, just because some
    other window in the same case ran short -- a window is kept only if it
    has exactly the frame count its own `length` implies at `fps`.
    `enumerate_windows` already guarantees a window fits entirely inside its
    label segment, so a window that comes back short means the video ended
    early or a read failed -- both genuinely unusable, not something to
    salvage by shortening its neighbours. Any drop is reported, not silent.

    `frames_per_window` overrides that derivation and exists for tests that
    build fixture windows by hand; production callers should leave it unset
    so the expected depth follows `window.length`.

    Each metadata row also records how the frames were produced --
    `ui_blurred`, `frame_size`, `fps`, `length` -- so a shard carries a
    machine-checkable claim rather than relying on convention. `ui_blurred`
    is True because `prepare_frame` is the only path frames take into a
    shard and it always blurs the bottom UI band; blurring is required by
    challenge rules, so a shard that cannot assert it is not usable.
    """
    def expected(window):
        if frames_per_window is not None:
            return frames_per_window
        return expected_frame_count(window, fps)

    depths = {expected(w) for w, _ in windows_and_frames}
    if len(depths) > 1:
        raise ValueError(
            "windows in one shard must all be the same length; got frame "
            "counts %s" % sorted(depths))

    usable = [(w, f) for w, f in windows_and_frames if len(f) == expected(w)]
    dropped = len(windows_and_frames) - len(usable)
    if dropped:
        seen = sorted({len(f) for w, f in windows_and_frames
                       if len(f) != expected(w)})
        print("write_shard: dropped %d/%d windows not exactly %s frames "
              "(lengths seen: %s)"
              % (dropped, len(windows_and_frames), sorted(depths), seen))
    if not usable:
        seen = sorted({len(f) for _, f in windows_and_frames})
        raise ValueError(
            "no windows with exactly %s frames (lengths seen: %s)"
            % (sorted(depths), seen))

    frame_size = int(usable[0][1][0].shape[0])
    depth = len(usable[0][1])
    meta = [{
        "case": w.case,
        "part": w.part,
        "start": w.start,
        "length": w.length,
        "task": w.task,
        "description": w.description,
        "tools": sorted(w.tools),
        "ui_blurred": True,
        "frame_size": frame_size,
        "fps": fps,
        "jpeg_quality": int(jpeg_quality),
    } for w, _ in usable]

    encoded, offsets = [], [0]
    for _, frames in usable:
        for frame in frames:
            ok, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            if not ok:
                raise ValueError(
                    "cv2.imencode failed on a frame bound for %s" % (out_path,))
            encoded.append(buffer.reshape(-1))
            offsets.append(offsets[-1] + int(buffer.size))

    # np.savez, not savez_compressed: JPEG is already compressed, so deflating
    # it again costs CPU on every read and write for ~nothing.
    np.savez(
        out_path,
        jpeg=np.concatenate(encoded),
        offsets=np.asarray(offsets, dtype=np.int64),
        shape=np.asarray([len(usable), depth], dtype=np.int64),
        meta=json.dumps(meta),
    )
    return len(usable)


class LazyWindow:
    """One window's frames. `window[position]` decodes that frame and no other.

    Indexes like the second axis of the array `read_shard` used to return:
    integer indexing (negatives included), `len`, and iteration in frame order.
    It is not an ndarray and deliberately does not pretend to be one -- the
    frames it hands back are.
    """

    __slots__ = ("_shard", "_window")

    def __init__(self, shard, window):
        self._shard = shard
        self._window = window

    def __len__(self):
        return self._shard.depth

    def __getitem__(self, position):
        if isinstance(position, slice):
            return [self[i] for i in range(*position.indices(len(self)))]
        return self._shard.frame(self._window, position)

    def __iter__(self):
        for position in range(len(self)):
            yield self[position]

    def __repr__(self):
        return "LazyWindow(window=%d, depth=%d)" % (self._window, len(self))


class LazyShard:
    """A shard's frames, JPEG-resident and decoded one frame at a time.

    WHY THIS IS NOT AN ARRAY. The eager reader `cv2.imdecode`d every frame of
    a shard and `np.stack`ed the result: ~1.3 GB resident for a real shard
    against ~165 MB of JPEG, with the stack briefly doubling that at its peak.
    `ShardFrames` calls it once per shard per DataLoader worker, and four
    workers of that is what took job 9618015 past a 32 GB cgroup limit.

    It was also mostly wasted work. A window holds 30 frames on this corpus and
    training samples 8 of them, so roughly three quarters of every decode was
    discarded immediately.

    WHAT IS AND IS NOT DEFERRED. The compressed buffer IS read eagerly and the
    npz is closed. Deferring the read as well would hold a file descriptor per
    worker per shard and re-read shared storage per frame, which is exactly
    what this loader's shard-at-a-time design exists to avoid. The 8x that
    decoding adds is what gets deferred; the I/O stays where it was.

    SERVING IS UNAFFECTED. Nothing in the container reads a shard --
    `perceive.decode_clip` reads the video directly. This is the training path
    only.
    """

    __slots__ = ("_blob", "_offsets", "_windows", "_depth", "_path", "_probe")

    def __init__(self, blob, offsets, windows, depth, path):
        self._blob = blob
        self._offsets = offsets
        self._windows = int(windows)
        self._depth = int(depth)
        self._path = path
        self._probe = None

    @property
    def depth(self):
        return self._depth

    def frame(self, window, position):
        """Decode exactly one frame. The only place a JPEG is expanded."""
        if not -self._windows <= window < self._windows:
            raise IndexError("window %d out of range for %d window(s) in %s"
                             % (window, self._windows, self._path))
        if not -self._depth <= position < self._depth:
            raise IndexError("frame %d out of range for depth %d in %s"
                             % (position, self._depth, self._path))
        window %= self._windows
        position %= self._depth
        index = window * self._depth + position
        frame = cv2.imdecode(
            self._blob[self._offsets[index]:self._offsets[index + 1]],
            cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(
                "cv2.imdecode failed for window %d frame %d of %s"
                % (window, position, self._path))
        return frame

    def _probe_frame(self):
        """Frame 0's shape and dtype, decoded once, for `shape` and `dtype`.

        The metadata rows carry `frame_size`, but that is the writer's claim
        about the frames rather than an observation of them. One decode makes
        `shape` describe what is actually in the shard, and it is cached so
        the cost is paid at most once per shard however often it is asked.
        """
        if self._probe is None:
            if not self._windows or not self._depth:
                self._probe = ((0, 0, 0), np.uint8)
            else:
                frame = self.frame(0, 0)
                self._probe = (tuple(frame.shape), frame.dtype)
        return self._probe

    @property
    def shape(self):
        return (self._windows, self._depth) + self._probe_frame()[0]

    @property
    def dtype(self):
        return self._probe_frame()[1]

    def __len__(self):
        return self._windows

    def __getitem__(self, window):
        if isinstance(window, slice):
            return [self[i] for i in range(*window.indices(self._windows))]
        if not -self._windows <= window < self._windows:
            raise IndexError("window %d out of range for %d window(s) in %s"
                             % (window, self._windows, self._path))
        # Held as given, negative or not. `frame` is the one place an index
        # becomes an offset, so it is the one place that wraps -- normalising
        # here as well would be a second copy of the rule that no test can
        # tell apart from the first.
        return LazyWindow(self, window)

    def __iter__(self):
        for window in range(self._windows):
            yield self[window]

    def __repr__(self):
        return "LazyShard(%s, windows=%d, depth=%d)" % (
            self._path, self._windows, self._depth)


def read_shard(path):
    """(frames, meta), `frames` indexing as (windows, depth, size, size, 3).

    `frames` is a `LazyShard`, not an ndarray: `frames[w][f]` is the same
    uint8 frame the eager reader returned, decoded at the moment it is asked
    for rather than at load time. See `LazyShard` for why.
    """
    with np.load(path, allow_pickle=False) as payload:
        meta = json.loads(str(payload["meta"]))
        if "jpeg" not in payload:
            # A shard written before JPEG storage: raw frames, already an
            # array, nothing to decode and nothing to defer.
            return payload["frames"], meta
        blob = payload["jpeg"]
        offsets = payload["offsets"]
        windows, depth = (int(x) for x in payload["shape"])
    return LazyShard(blob, offsets, windows, depth, path), meta
