"""Raw clip -> the per-class probability record the question router reads.

This is the perception half of inference. It ends at a JSON record per case;
nothing here knows what question was asked, and nothing downstream re-opens a
video. That seam is why the record's shape, its class ordering and the
thresholds behind `tools_present` are treated as a contract and pinned by
tests rather than left to whatever the checkpoint happens to contain.

FRAME RATE. `extract.py` converts wall-clock window offsets into frame
indices and therefore has to know the source rate -- it defaults to 60 fps
for the training corpus and warns when it has to guess. Nothing here needs
that. The graded unit is the entire clip, so the sample is "N frames evenly
spaced across the file", expressed in frame indices from the file's own frame
count. The Cat 2 sample clips report 60 fps / 1800 frames / 1280x720 and the
Cat 1 test clips are 1 fps / 640x512; both give the same 30 seconds of
coverage under an index-based sampler, and neither can be silently decoded at
the wrong time base. Copying `default_fps=60.0` here would have been wrong on
the second format and invisible on the first.

Every frame passes through `preprocess.prepare_frame`, which crops the black
side margins and blurs the bottom UI band. That is a challenge rule -- "using
the information available in the UI to make predictions is not allowed" --
not an optimisation, so there is deliberately no path into a model that
bypasses it.
"""
import cv2
import numpy as np
import torch

from .frames import sample_frame_indices
from .models import build_model
from .motion import PROBE_OFFSETS_MS
from .predict import predict_window
from .preprocess import prepare_frame
from .taxonomy import TASK_CLASSES, TOOL_CLASSES
from .train import load_checkpoint

DEFAULT_FRAMES = 16


# Re-exported, not redefined: the rule now lives in `surgvu.frames`, which
# imports nothing, so callers without torch (the v2 offline sweeps, and
# `surgvu.vlm`, which carries a hand-copy for exactly this reason) can share
# the one definition. Every existing
# `from surgvu.perceive import sample_frame_indices` is unaffected.
sample_frame_indices = sample_frame_indices  # noqa: F811 - see the import above


def _frame_count(capture, video_path):
    """Frame count from the container, counted by hand if it is not reported."""
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 0:
        return total
    count = 0
    while capture.grab():
        count += 1
    if count:
        print("decode_clip: %s reported no frame count; counted %d by decoding"
              % (video_path, count))
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return count


def decode_clip(video_path, n_frames=DEFAULT_FRAMES, size=512):
    """`n_frames` evenly spaced frames of a clip, preprocessed, as one array.

    Returns uint8 (n, size, size, 3) in OpenCV BGR order -- exactly the shape
    `predict_window` expects, and exactly what a training shard yields, so the
    serving path and the training path hand their models the same thing.

    A frame whose seek-and-read fails is skipped rather than ending the clip:
    one bad index late in a file should cost one frame, not every frame after
    it. The count that survives is what the caller records as `n_frames`, so a
    degraded clip is visible in the output rather than being asserted away.
    """
    capture = cv2.VideoCapture(str(video_path))
    try:
        total = _frame_count(capture, video_path)
        if total <= 0:
            raise ValueError(
                "decoded no frames from %s: the file is unreadable or empty. "
                "A missing clip must stop the run -- an empty record would be "
                "answered from as if it were a prediction." % (video_path,))
        frames = []
        missed = []
        for index in sample_frame_indices(total, n_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                missed.append(index)
                continue
            frames.append(prepare_frame(frame, size=size))
        if missed:
            print("decode_clip: %s failed to read %d of %d sampled frames "
                  "(indices %s)" % (video_path, len(missed), n_frames, missed))
        if not frames:
            raise ValueError(
                "decoded no frames from %s: all %d sampled indices failed to "
                "read." % (video_path, n_frames))
        return np.stack(frames)
    finally:
        capture.release()


#: Frame spacing inside a burst, in frames per second. 15 fps -- 67 ms between
#: frames -- because that is what shards_multi16 was extracted at, and the
#: serving path must measure motion the same way the pool it was calibrated on
#: does. A different spacing here would produce a statistic in different units
#: from the threshold fitted against it.
BURST_FPS = 15.0
DEFAULT_FRAMES_PER_BURST = 3


def decode_clip_bursts(video_path, n_frames=DEFAULT_FRAMES,
                       per_burst=DEFAULT_FRAMES_PER_BURST,
                       burst_fps=BURST_FPS, size=512):
    """The same frames `decode_clip` returns, plus their immediate neighbours.

    Returns `(centres, bursts)`.

    THE CONTRACT THAT MAKES THIS SAFE TO SHIP. `centres` is EXACTLY what
    `decode_clip` returns for the same arguments: the same sampled indices,
    the same preprocessing, and the same skip-a-failed-read behaviour. The
    appearance model therefore sees byte-identical input to what it sees
    today, and the shipped answers cannot move because motion was added.
    tests/test_perceive.py asserts that equality rather than trusting it.

    `bursts` is a list the same length as `centres`. Each entry is a
    (per_burst, size, size, 3) stack covering t-67ms, t, t+67ms -- or None if
    any of its frames failed to read. None is NOT an array of zeros and not a
    still burst: it means the measurement is unavailable, and
    motion.motion_record_from_bursts excludes it rather than counting it as
    stillness.

    A FAILED FLANK NEVER COSTS A CENTRE. The centre is appended whether or not
    its neighbours read, so a degraded clip loses motion evidence and keeps
    every bit of the appearance evidence it has today. That asymmetry is
    deliberate: the appearance path is what ships.

    Cost is roughly three times `decode_clip`, which the budget absorbs --
    serving measured 4.2 s per case against a 600 s limit.
    """
    if per_burst < 1:
        raise ValueError("per_burst must be >= 1, got %r" % (per_burst,))
    capture = cv2.VideoCapture(str(video_path))
    try:
        total = _frame_count(capture, video_path)
        if total <= 0:
            raise ValueError(
                "decoded no frames from %s: the file is unreadable or empty."
                % (video_path,))
        # The clip's own frame rate decides how many frames 67 ms is. A
        # hardcoded offset would mean a different real duration on every
        # differently-encoded video, and the statistic would not be comparable
        # across cases -- which is the one property it has to have.
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 0:      # 0, None, or NaN
            fps = 60.0
        offset = max(1, int(round(fps / float(burst_fps))))

        centres, bursts = [], []
        missed = []
        for index in sample_frame_indices(total, n_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                missed.append(index)
                continue
            centre = prepare_frame(frame, size=size)
            centres.append(centre)

            # Offsets around the centre, in time order, with the centre in the
            # middle -- the same layout shards_multi16 writes, so micro
            # activity means the same thing in both places.
            half = per_burst // 2
            wanted = [index + (k - half) * offset for k in range(per_burst)]
            if min(wanted) < 0 or max(wanted) >= total:
                # Clamping would difference a frame against itself and read as
                # stillness at the clip edges, which is a lie with a plausible
                # value. Better to have no measurement there.
                bursts.append(None)
                continue
            stack, complete = [], True
            for want in wanted:
                if want == index:
                    stack.append(centre)
                    continue
                capture.set(cv2.CAP_PROP_POS_FRAMES, want)
                ok, flank = capture.read()
                if not ok:
                    complete = False
                    break
                stack.append(prepare_frame(flank, size=size))
            bursts.append(np.stack(stack) if complete else None)

        if missed:
            print("decode_clip_bursts: %s failed to read %d of %d sampled "
                  "frames (indices %s)"
                  % (video_path, len(missed), n_frames, missed))
        if not centres:
            raise ValueError(
                "decoded no frames from %s: all %d sampled indices failed."
                % (video_path, n_frames))
        unmeasured = sum(1 for b in bursts if b is None)
        if unmeasured:
            print("decode_clip_bursts: %s has %d of %d bursts unmeasurable "
                  "(clip edge or failed flank read); motion excludes them"
                  % (video_path, unmeasured, len(bursts)))
        return np.stack(centres), bursts
    finally:
        capture.release()


#: Probe offsets in milliseconds, log-spaced to fill the gap the existing
#: sampler leaves. decode_clip_bursts measures at 67 ms and macro activity at
#: 1875 ms; between those two nothing is sampled at all, and that is precisely
#: the range a tool stroke occupies. These are DEFAULTS and are overridden by
#: whatever scripts/calibrate_motion_v2.py fits and writes to config.
#:
#: Imported from `surgvu.motion.PROBE_OFFSETS_MS` (controller ruling R16)
#: rather than repeated as a literal here, so this and `VECTOR_SLOTS` -- and
#: scripts/dump_motion_v2.OFFSETS_MS, which imports the same name -- can
#: never disagree about what the shipped default actually is.
DEFAULT_PROBE_OFFSETS_MS = PROBE_OFFSETS_MS


def decode_clip_multiscale(video_path, n_frames=DEFAULT_FRAMES,
                           offsets_ms=DEFAULT_PROBE_OFFSETS_MS, size=512,
                           index_range=None):
    """`decode_clip`'s frames plus symmetric probes at several timescales.

    Returns `(centres, probes)`.

    `centres` is EXACTLY what `decode_clip` returns for the same arguments --
    same indices, same preprocessing, same skip-a-failed-read behaviour. The
    appearance model must see byte-identical input to what it sees today, so
    adding evidence cannot move a shipped answer.
    tests/test_perceive_multiscale.py asserts that rather than trusting it.

    `probes` has one entry per centre: {offset_ms: (before, after)} or
    {offset_ms: None} when either flank is unreadable or runs off the clip.
    None means UNAVAILABLE. It is never a duplicated centre, because
    differencing a frame against itself reads as perfect stillness -- a false
    measurement that looks entirely reasonable, which is the failure mode this
    codebase has paid for most.

    THIS DOES NOT REPLACE decode_clip_bursts. That function's uniform
    (per_burst, ...) layout is the shard format `surgvu/dataset.py` and
    `surgvu/temporal.py` read, and a non-uniform spacing would silently change
    what "micro activity" means in both. The two coexist.

    `index_range`, if given, is an inclusive `(first, last)` pair of frame
    indices. `n_frames` centres are then sampled evenly across THAT range
    (`sample_frame_indices(last - first + 1, n_frames)`, offset by `first`)
    instead of evenly across the whole file. This is ADDITIVE (ruling R15):
    `index_range=None`, the default, is BYTE-IDENTICAL to today -- same
    indices, same everything -- because it degrades to sampling across
    `(0, total - 1)`, which is exactly what happened before this parameter
    existed. tests/test_perceive_multiscale.py pins that equality.

    It exists so `scripts/dump_motion_v2.py` can target one stratified
    window of a multi-hour source video by SEEKING to it directly, with no
    temporary file and no re-encode: a temporary clip cut with
    cv2.VideoWriter would re-encode (e.g. mp4v) the frames the motion
    statistic is computed from, and a threshold fitted on re-encoded pixels
    need not transfer to the serving decoder reading the original h264 --
    exactly the property calibration exists to get right.
    """
    if not offsets_ms:
        raise ValueError("no probe offsets requested; a multiscale decode "
                         "with no scales is a plain decode_clip call")
    capture = cv2.VideoCapture(str(video_path))
    try:
        total = _frame_count(capture, video_path)
        if total <= 0:
            raise ValueError(
                "decoded no frames from %s: the file is unreadable or empty."
                % (video_path,))
        if index_range is None:
            sample_base, sample_span = 0, total
        else:
            first, last = index_range
            if first < 0 or last >= total or first > last:
                raise ValueError(
                    "index_range %r is out of bounds for a %d-frame file %s"
                    % (index_range, total, video_path))
            sample_base, sample_span = first, last - first + 1
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 0:      # 0, None, or NaN
            fps = 60.0
        # Milliseconds -> frames using the CLIP's own rate. A hardcoded frame
        # offset would mean a different real duration on every differently
        # encoded video, and the statistic has to be comparable across cases.
        strides = {ms: max(1, int(round(fps * ms / 1000.0)))
                   for ms in offsets_ms}

        centres, probes = [], []
        missed = []
        for index in (sample_base + i
                     for i in sample_frame_indices(sample_span, n_frames)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                missed.append(index)
                continue
            centres.append(prepare_frame(frame, size=size))

            entry = {}
            for ms, stride in strides.items():
                lo, hi = index - stride, index + stride
                if lo < 0 or hi >= total:
                    entry[ms] = None
                    continue
                pair = []
                for want in (lo, hi):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, want)
                    ok, flank = capture.read()
                    if not ok:
                        pair = None
                        break
                    pair.append(prepare_frame(flank, size=size))
                entry[ms] = tuple(pair) if pair else None
            probes.append(entry)

        if missed:
            print("decode_clip_multiscale: %s failed to read %d of %d sampled "
                  "frames (indices %s)" % (video_path, len(missed), n_frames,
                                           missed))
        if not centres:
            raise ValueError(
                "decoded no frames from %s: all %d sampled indices failed to "
                "read." % (video_path, n_frames))
        return np.stack(centres), probes
    finally:
        capture.release()


def tools_present(probs, thresholds, classes):
    """The classes whose clip probability meets their own tuned threshold.

    Per class, never a shared 0.5. `train_tools.py` tunes one threshold per
    class on validation and freezes them into the checkpoint precisely because
    the corpus is 90x imbalanced; the tuned values span 0.05 to 0.95, so a
    global cutoff both drops rare classes and admits common ones while still
    producing a list that looks entirely reasonable.

    `>=`, matching the `probs >= thresholds` that selected the checkpoint --
    a strict `>` would score it differently than it was tuned.
    """
    probs = list(probs)
    thresholds = list(thresholds)
    classes = list(classes)
    if len(thresholds) != len(classes) or len(probs) != len(classes):
        raise ValueError(
            "threshold/probability/class lengths must match; got %d "
            "probabilities, %d thresholds, %d classes. They are positional, "
            "so a short list would zip silently and leave the tail of the "
            "taxonomy unthresholded."
            % (len(probs), len(thresholds), len(classes)))
    return sorted(name for name, prob, threshold
                  in zip(classes, probs, thresholds) if prob >= threshold)


def _checked_classes(meta, expected, role):
    """The checkpoint's class list, proven to be the taxonomy the router uses.

    The record is keyed by class name, so a checkpoint trained on a different
    ordering produces a record whose keys are all correct and whose values
    belong to other classes. This also catches the plausible operator error of
    swapping the two checkpoint paths: both are EfficientNetV2-S files in one
    directory whose names differ by four characters.
    """
    classes = list(meta["classes"])
    if classes != list(expected):
        raise ValueError(
            "%s checkpoint classes do not match the taxonomy: got %r, expected "
            "%r. The record is keyed by name, so a mismatched head would "
            "report every class as another one." % (role, classes, list(expected)))
    return classes


def _as_probability_map(probs, classes, role):
    # No dtype coercion here: the float() below is the single place a numpy
    # scalar becomes JSON-writable, and widening to float64 first would hide
    # a dropped float() -- np.float64 is a subclass of float and serialises
    # fine, while the float32 that predict_window actually returns does not.
    probs = np.asarray(probs).reshape(-1)
    if probs.shape[0] != len(classes):
        raise ValueError(
            "%s head produced %d probabilities for %d classes"
            % (role, probs.shape[0], len(classes)))
    return {name: float(value) for name, value in zip(classes, probs)}


def clip_record(tool_probs, tool_meta, task_probs, task_meta, n_frames,
                motion=None, motion_v2=None, yolo=None, variant=None,
                agree=None):
    """One case's entry in the perception JSON.

    Floats are plain Python floats, not numpy scalars: this record is written
    with `json.dumps`, which refuses float32, and discovering that after
    paying for a full inference run would throw the run away.

    Every optional block is PURELY ADDITIVE, and that is the safety property
    the whole evidence pipeline rests on. Omitted, this returns a dict
    byte-identical to what it returned before the block existed -- same
    keys, same order, same values -- so enabling a new evidence source
    cannot move a shipped answer by itself. The router reads perception only
    through accessors, none of which look at an unopened key, so adding one
    cannot change an answer until a calibrated gate is wired to read it.
    Asserted in tests/test_perceive.py and tests/test_inference_motion_v2.py
    rather than assumed.
    """
    tool_classes = _checked_classes(tool_meta, TOOL_CLASSES, "tool")
    task_classes = _checked_classes(task_meta, TASK_CLASSES, "task")
    if "thresholds" not in tool_meta:
        raise KeyError(
            "the tool checkpoint carries no 'thresholds'; refusing to invent "
            "one. Defaulting to 0.5 would produce a full, plausible "
            "tools_present list from an untuned cutoff.")

    tools = _as_probability_map(tool_probs, tool_classes, "tool")
    task = _as_probability_map(task_probs, task_classes, "task")
    record = {
        "tools": tools,
        "tools_present": tools_present(
            [tools[name] for name in tool_classes],
            tool_meta["thresholds"], tool_classes),
        "task": task,
        "task_top": max(task_classes, key=lambda name: task[name]),
        "n_frames": int(n_frames),
    }
    # Every optional block is PURELY ADDITIVE, and that is the safety
    # property the whole evidence pipeline rests on. Omitted, this returns a
    # dict byte-identical to what it returned before the block existed --
    # same keys, same order, same values -- so enabling a new evidence source
    # cannot move a shipped answer by itself. Asserted in
    # tests/test_perceive.py and tests/test_inference_motion_v2.py.
    for key, block in (("motion", motion), ("motion_v2", motion_v2),
                       ("yolo", yolo), ("variant", variant),
                       ("agree", agree)):
        if block is not None:
            record[key] = block
    return record


def perceive_clip(frames, tool_model, tool_meta, task_model, task_meta,
                  device="cpu"):
    """Both experts over one clip's frames -> one record.

    The activations are not interchangeable: the tool head is multi-label
    (several instruments are installed at once, so they must not compete for
    one unit of mass) and the task head is multi-class. Each is served at its
    own checkpoint's `image_size` rather than a shared constant -- both are
    384 today, so a hardcoded value would pass every test until one expert is
    retrained and then be silently wrong.
    """
    tool_probs = predict_window(tool_model, frames, device,
                                tool_meta["image_size"], activation="sigmoid")
    task_probs = predict_window(task_model, frames, device,
                                task_meta["image_size"], activation="softmax")
    return clip_record(tool_probs, tool_meta, task_probs, task_meta, len(frames))


def find_clips(root):
    """[(case_id, path)] for every clip under `root`, sorted by case id.

    Matches the sample layout `root/caseNNN/caseNNN.mp4` and a flat directory
    of `.mp4` files equally; the case id is the file stem either way.
    """
    from pathlib import Path

    root = Path(root)
    clips = {}
    for path in sorted(root.rglob("*.mp4")):
        if path.stem in clips:
            raise ValueError(
                "two clips under %s share the case id %r: %s and %s"
                % (root, path.stem, clips[path.stem], path))
        clips[path.stem] = path
    if not clips:
        raise ValueError("found no .mp4 clips under %s" % (root,))
    return [(case, clips[case]) for case in sorted(clips)]


def load_expert(path, device="cpu"):
    """(model, meta) for one checkpoint, sized from its own recorded classes.

    `pretrained=False`: the ImageNet weights are about to be overwritten by
    the checkpoint anyway, and fetching them would make serving depend on a
    torchvision download that an offline node has no path to.
    """
    meta = torch.load(str(path), map_location="cpu", weights_only=False)["meta"]
    model = build_model(len(meta["classes"]), meta["backbone"], pretrained=False)
    load_checkpoint(path, model)
    return model.to(device).eval(), meta
