"""Decode motion-v2 anchors from the SOURCE videos, for calibrate_motion_v2.py.

THE GAP THIS FILLS (controller ruling R13). calibrate_motion_v2.py's --dump
consumes a JSON list of {"case", "part", "t", "vector"} records built from
real timestamps, and no such producer existed: scripts/sample_motion.py
emits a different, v1 shape ({case, micro_mean}) over the eleven public
SAMPLE clips only, one 30 s clip per case. The real source is 155 cases of
up to two ~5-hour, 60 fps video parts each
(/staging/groups/bhaskar_opscribe/surgvu/videos/surgvu24/<case>/
<case>_video_part_<NNN>.mp4), and `t` must be an ABSOLUTE second offset
within that video part, because activity_labels() in calibrate_motion_v2.py
compares it against tasks.csv start_time/stop_time, which are in those same
units and are themselves scoped to a `part` (a case can have two video
files, and timestamps reset at the part boundary). That is why every
emitted record carries `part` alongside `t` -- so the calibrator can pass it
straight to activity_labels(part=...) and never compare a part-2 timestamp
against part-1 task intervals.

STRATIFIED, NOT UNIFORM. A task-annotated case can be almost entirely
"active" or almost entirely "idle" by tasks.csv's own bookkeeping; a
timestamp drawn uniformly across a whole 5-hour part therefore risks landing
in one class only, and separability() in calibrate_motion_v2.py correctly
raises ValueError when that happens -- which would then look like a code bug
rather than what it is, a sampling artefact. So sampling happens at the SPAN
level, before any individual timestamp is chosen: `plan_span_requests`
splits [0, duration) into task-covered ("active") spans and their complement
("idle") from tasks.csv, then allocates roughly half of --windows-per-case
to each side, using surgvu.labels.CaseLabels for the real, part-aware
start_part/start_time/stop_part/stop_time schema -- the same schema
calibrate_motion_v2.py's own activity_labels() reads as of controller ruling
R14 (an earlier draft of that function read simplified start/stop columns
that do not exist in the real corpus; both scripts now agree).

HOW A WINDOW IS ACTUALLY DECODED (controller ruling R15). decode_clip_multiscale
has no start-time argument on its own -- it always samples `n_frames`
evenly spaced across the ENTIRE file handed to it via
`sample_frame_indices(total, n_frames)`. It gained an ADDITIVE `index_range`
parameter (src/surgvu/perceive.py) precisely so this script can point it at
one stratified window instead: `index_range=(first, last)` samples
`n_frames` evenly across just that inclusive frame-index range of the
SOURCE video, by seeking, with `index_range=None` (every other call site)
staying byte-identical to today. So `_decode_span` opens the real
`video_path` directly and calls `decode_clip_multiscale(video_path,
n_frames=..., index_range=(first, last), ...)` -- no temporary file, no
re-encode.

An earlier draft of this script cut a temporary clip per span with
cv2.VideoWriter (mp4v) and decoded THAT. R15 rejected it: mp4v re-encodes,
and this whole task calibrates a mean-absolute-difference statistic in
pixel units -- a cut fitted on re-encoded frames need not transfer to the
serving decoder reading the original h264. Seeking directly in the source
file removes that risk entirely, and is also cheaper: decode_clip_multiscale
only seeks to the ~n_frames anchors and their probe flanks, rather than
reading every frame of a multi-second span sequentially first.

Every frame decode_clip_multiscale returns still goes through
`preprocess.prepare_frame` automatically, with nothing in this script that
bypasses it: `_decode_span` never calls cv2.resize/crop/blur itself, and
never reads a frame it hands onward without going through
decode_clip_multiscale first.

WHY macro_prev/macro_next ARE MEANINGFUL HERE (not always None). Anchors
requested from one `index_range` are evenly spaced across it, exactly as a
production 30 s clip's `DEFAULT_FRAMES` anchors are -- just typically over a
shorter span, because `--window-seconds` (default 8s) bounds how wide a
single request's range is. `micro_short/mid/long` and the flow slots are
unaffected by that spacing -- they are read at fixed offsets (133/400/1200
ms) regardless of how far apart the anchors themselves are.

WHAT THIS DOES NOT DO. It does not run the appearance model, does not
compute a threshold, and is not itself the calibration -- it only supplies
calibrate_motion_v2.py's --dump input. It does not touch every case by
default in one run; --cases limits a smoke run to one or two cases before a
full 155-case Condor job is submitted.

TORCH IS DEFERRED. This module needs `surgvu.perceive.decode_clip_multiscale`,
which imports torch at module scope, and torch is not installed on the
machine this was authored on. Only `_decode_span` -- the one function that
actually decodes video -- imports it, and only inside the function body, the
same pattern `src/surgvu/detect.py` will use. Everything else here (argument
parsing, span planning, `_anchor_records`) is plain csv/re/random/cv2 and is
unit-tested directly in tests/test_dump_motion_v2.py, without importing
perceive and without touching the 326GB corpus.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.frames import sample_frame_indices              # noqa: E402
from surgvu.labels import CaseLabels, normalize_part         # noqa: E402
from surgvu.motion import (PROBE_OFFSETS_MS, _VECTOR_KEYS,  # noqa: E402
                           motion_record_v2)

VIDEO_ROOT = "/staging/groups/bhaskar_opscribe/surgvu/videos/surgvu24"
LABELS_ROOT = "/staging/groups/bhaskar_opscribe/surgvu/labels_cat2/SURGVU25_train_labels"

#: The single source of truth for these offsets is
#: `surgvu.motion.PROBE_OFFSETS_MS` (controller ruling R16) -- imported
#: rather than re-typed, unlike before. `surgvu.motion` is torch-free (only
#: `surgvu.perceive`, which this module defers importing, is not), so this
#: import costs nothing the TORCH IS DEFERRED note above needs to guard
#: against.
OFFSETS_MS = PROBE_OFFSETS_MS

#: Longest index_range span requested per decode_clip_multiscale call, in
#: seconds. At 60fps that's 480 frames; comfortably more than the largest
#: probe offset (1200ms = 72 frames) needs as edge margin for every anchor
#: but the first and last of a batch, while keeping one request's seeking
#: cheap.
WINDOW_SECONDS = 8.0

#: Anchors requested per index_range span, capped at
#: surgvu.perceive.DEFAULT_FRAMES so a single span's decode costs no more
#: than one production clip's worth of decode_clip_multiscale work.
BATCH_FRAMES = 16

_PART_RE = re.compile(r"part[_\-]?(\d+)", re.IGNORECASE)


def find_case_videos(video_root, case):
    """Sorted (part_number, path) for one case's video files.

    Filenames are `<case>_video_part_<NNN>.mp4`; the part number is read
    from the name rather than assumed, mirroring
    scripts/build_cases_txt.py's regex for the same reason -- a video with
    no readable part number is skipped rather than silently called part 1,
    which would decode real frames and label them with someone else's
    timestamps.
    """
    case_dir = Path(video_root) / case
    if not case_dir.is_dir():
        return []
    videos = []
    for path in sorted(case_dir.glob("%s_video_part_*.mp4" % case)):
        match = _PART_RE.search(path.name)
        if not match:
            continue
        videos.append((int(match.group(1)), path))
    return sorted(videos)


def resolve_cases(value, available):
    """`--cases` is either an int limit or a comma-separated list of case ids.

    An int always wins when `value` parses as one. Every real case id in
    this corpus is `case_NNN`, never a bare digit string, so there is no
    real case name this could misinterpret as a limit.
    """
    if value is None:
        return list(available)
    text = str(value).strip()
    try:
        limit = int(text)
    except ValueError:
        requested = [c.strip() for c in text.split(",") if c.strip()]
        missing = [c for c in requested if c not in available]
        if missing:
            raise ValueError(
                "requested cases not found under the labels root: %s" % missing)
        return requested
    if limit <= 0:
        raise ValueError("--cases limit must be positive, got %d" % limit)
    return list(available)[:limit]


def _merge_intervals(intervals):
    """Sorted, non-overlapping (start, stop) spans from possibly-overlapping ones."""
    spans = sorted((s, e) for s, e in intervals if e > s)
    merged = []
    for start, stop in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def _complement(merged, duration):
    """The spans of [0, duration) NOT covered by `merged` (sorted, disjoint).

    Includes the time before the first span and after the last, not only
    the gaps strictly between two spans -- a case with a single task segment
    covering its first hour still has four hours of un-annotated "idle" time
    after it, and that time is exactly as usable for the idle draw as a gap
    between two segments would be.
    """
    gaps = []
    cursor = 0.0
    for start, stop in merged:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, stop)
    if cursor < duration:
        gaps.append((cursor, duration))
    return gaps


def _budget_requests(spans, want, rng, window_seconds, batch_frames):
    """Chunk `want` anchors into (start, length, n_frames) requests over `spans`.

    Each request draws ONE span (weighted by length, so a 600s segment is
    not swamped by ten 1s ones), picks a `window_seconds`-long sub-window
    inside it (the whole span if it is shorter), and asks for up to
    `batch_frames` anchors from that sub-window -- so one decode_clip_multiscale
    call can supply several anchors at once instead of one call per anchor.
    """
    requests = []
    if not spans or want <= 0:
        return requests
    weights = [stop - start for start, stop in spans]
    remaining = want
    guard = 0
    while remaining > 0:
        guard += 1
        if guard > 4 * want + 10:
            # The spans on offer are too short/degenerate to fill the
            # budget (e.g. every span under a second); stop rather than
            # spin forever chasing a count that cannot be reached.
            break
        start, stop = rng.choices(spans, weights=weights, k=1)[0]
        span_len = stop - start
        length = min(window_seconds, span_len)
        if length <= 0:
            continue
        anchor_at = (start if span_len <= length
                    else start + rng.uniform(0, span_len - length))
        n = min(batch_frames, remaining)
        requests.append((anchor_at, length, n))
        remaining -= n
    return requests


def plan_span_requests(duration, intervals, windows_per_case, rng,
                       window_seconds=WINDOW_SECONDS, batch_frames=BATCH_FRAMES):
    """Stratified (start, length, n_frames) requests covering [0, duration).

    Splits `windows_per_case` roughly half-and-half between spans covered by
    `intervals` (active) and their complement (idle, in the loose sense: not
    annotated, not necessarily quiet -- see the module docstring). If one
    side has no spans at all its whole budget moves to the other side rather
    than being silently dropped, so a case with (say) zero annotated gaps
    still returns `windows_per_case` anchors, all of the one class that
    exists -- calibrate_motion_v2.py's separability() is what is entitled to
    complain about a single-class case, not this planner.
    """
    if duration <= 0:
        raise ValueError("duration must be positive, got %r" % (duration,))
    if windows_per_case <= 0:
        raise ValueError(
            "windows_per_case must be positive, got %r" % (windows_per_case,))
    active = _merge_intervals(intervals)
    idle = _complement(active, duration)
    n_active = windows_per_case // 2
    n_idle = windows_per_case - n_active
    if not active:
        n_idle += n_active
        n_active = 0
    if not idle:
        n_active += n_idle
        n_idle = 0
    return (_budget_requests(active, n_active, rng, window_seconds, batch_frames)
           + _budget_requests(idle, n_idle, rng, window_seconds, batch_frames))


def _anchor_records(case, part, first, sample_span, fps, per_anchor,
                    offsets_ms=OFFSETS_MS):
    """{"case", "part", "t", "vector", "offsets_ms"} records for one decoded
    index_range.

    `first`/`sample_span` are the SAME `(first, last)` -> `(first, last -
    first + 1)` values passed to decode_clip_multiscale's `index_range`, and
    `per_anchor` is `motion_record_v2(...)["per_anchor"]` for that same
    call -- both are required so the indices `sample_frame_indices`
    recomputes here line up with the ones decode_clip_multiscale used
    internally to pick centres. Pure: no cv2, no torch, exercised directly in
    tests/test_dump_motion_v2.py with a fabricated `per_anchor`.

    Every vector value is cast through `float(...)` (None stays None) so the
    written JSON is strict-JSON-safe -- a numpy scalar surviving into
    json.dumps raises, and this must not depend on numpy's dtype only being
    caught downstream.

    `offsets_ms` -- the offsets this span was ACTUALLY decoded with -- is
    recorded on every record (controller ruling R16). Before this, nothing
    in the dump carried its own provenance, and calibrate_motion_v2.py wrote
    a literal `[133, 400, 1200]` into config/motion_v2.json that asserted
    what the dump was ASSUMED to have used rather than what it was handed --
    confidently wrong provenance if the two ever diverged, which is worse
    than no provenance at all. Now the calibrator reads this field instead
    of guessing.
    """
    indices = sample_frame_indices(sample_span, len(per_anchor))
    records = []
    for relative_index, vector in zip(indices, per_anchor):
        absolute_index = first + relative_index
        t = absolute_index / float(fps)
        records.append({
            "case": case,
            "part": part,
            "t": float(t),
            "vector": {key: (None if vector[key] is None else float(vector[key]))
                      for key in _VECTOR_KEYS},
            "offsets_ms": list(offsets_ms),
        })
    return records


def _decode_span(case, part, video_path, fps, total, start, length, n_frames,
                 size, offsets_ms=OFFSETS_MS):
    """One span's anchor records, decoded by SEEKING directly in the source
    video via decode_clip_multiscale's additive `index_range` parameter
    (ruling R15) -- no temporary file, no re-encode. Earlier drafts of this
    function cut a temporary clip with cv2.VideoWriter/mp4v and decoded
    that; R15 rejected that approach because mp4v re-encodes, and the
    motion statistic this whole task calibrates is a mean absolute
    difference in pixel units -- a threshold fitted on re-encoded frames
    need not transfer to the serving decoder reading the original h264.

    The torch-touching import lives here, inside the function body, so
    everything above it in this module stays importable without torch.
    """
    from surgvu.perceive import decode_clip_multiscale

    first = int(round(start * fps))
    span_frames = max(1, int(round(length * fps)))
    last = min(first + span_frames - 1, total - 1)
    if first > last:
        print("skipping span at %.1fs of %s: out of range "
              "(first=%d last=%d total=%d)"
              % (start, video_path, first, last, total))
        return []

    sample_span = last - first + 1
    want = min(n_frames, sample_span)
    centres, probes = decode_clip_multiscale(
        video_path, n_frames=want, offsets_ms=offsets_ms, size=size,
        index_range=(first, last))
    if len(centres) != want:
        # A dropped read here would desynchronise `_anchor_records`'
        # recovered indices from the anchors decode_clip_multiscale
        # actually returned -- better to drop the whole span's anchors
        # than to attribute a vector to the wrong timestamp.
        print("skipping span at %.1fs of %s: decoded %d of %d requested "
              "anchors, which would misalign the recovered timestamps"
              % (start, video_path, len(centres), want))
        return []
    record = motion_record_v2(centres, probes)
    return _anchor_records(case, part, first, sample_span, fps,
                           record["per_anchor"], offsets_ms=offsets_ms)


def dump_case(case, video_root, labels_root, windows_per_case, size, rng,
              window_seconds=WINDOW_SECONDS, batch_frames=BATCH_FRAMES):
    """All anchor records for one case, across every video part it has.

    Uses `surgvu.labels.CaseLabels` for the real, part-aware tasks.csv
    schema (start_part/start_time/stop_part/stop_time) to plan the
    stratified sample -- the same schema calibrate_motion_v2.py's own
    activity_labels() reads as of ruling R14.
    """
    labels_dir = Path(labels_root) / case
    if not (labels_dir / "tasks.csv").exists():
        print("skipping %s: no tasks.csv under %s" % (case, labels_dir))
        return []
    case_labels = CaseLabels.from_dir(labels_dir)
    videos = find_case_videos(video_root, case)
    if not videos:
        print("skipping %s: no video parts found under %s/%s"
              % (case, video_root, case))
        return []

    per_part_budget = max(1, windows_per_case // len(videos))
    records = []
    for part_number, video_path in videos:
        part = normalize_part(part_number)
        capture = cv2.VideoCapture(str(video_path))
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = capture.get(cv2.CAP_PROP_FPS) or 60.0
        finally:
            capture.release()
        if total <= 0:
            print("skipping %s part %s: unreadable video %s"
                  % (case, part, video_path))
            continue
        duration = total / fps
        intervals = [(seg.start, seg.stop) for seg in case_labels.task_segments()
                    if seg.part == part]
        requests = plan_span_requests(duration, intervals, per_part_budget, rng,
                                      window_seconds, batch_frames)
        for start, length, n in requests:
            records.extend(
                _decode_span(case, part, video_path, fps, total, start,
                            length, n, size))
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video-root", default=VIDEO_ROOT)
    parser.add_argument("--labels-root", default=LABELS_ROOT)
    parser.add_argument("--cases", default=None,
                        help="an int limit, or a comma-separated list of "
                             "case ids; default is every case under "
                             "--labels-root (155 on the real corpus)")
    parser.add_argument("--windows-per-case", type=int, default=20)
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--batch-frames", type=int, default=BATCH_FRAMES)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0,
                        help="span selection is randomised; fixed so a rerun "
                             "over the same cases is reproducible")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    labels_root = Path(args.labels_root)
    available = sorted(p.name for p in labels_root.iterdir() if p.is_dir())
    if not available:
        raise SystemExit("no case directories under %s" % labels_root)
    cases = resolve_cases(args.cases, available)
    print("%d case(s) selected: %s"
          % (len(cases), cases[:5] + (["..."] if len(cases) > 5 else [])))

    rng = random.Random(args.seed)
    records = []
    for case in cases:
        rows = dump_case(case, args.video_root, args.labels_root,
                        args.windows_per_case, args.size, rng,
                        args.window_seconds, args.batch_frames)
        print("%-9s %d anchor(s)" % (case, len(rows)), flush=True)
        records.extend(rows)

    if not records:
        print("WARNING: zero anchors decoded across %d case(s); "
              "calibrate_motion_v2.py would have nothing to fit against"
              % len(cases))

    Path(args.out).write_text(json.dumps(records, indent=2) + "\n",
                              encoding="utf-8")
    print("wrote %d anchor(s) across %d case(s) to %s"
          % (len(records), len(cases), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
