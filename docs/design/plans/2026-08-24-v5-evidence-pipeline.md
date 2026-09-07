# v5 Plan 1 — Evidence Pipeline Implementation Plan


**Goal:** Give the pipeline richer, calibrated, time-resolved evidence — multi-scale motion with optical flow, a YOLO second opinion, and a Large/Mega needle-driver variant head — all additive to the existing perception record and all behind flags.

**Architecture:** Nothing is replaced. `perceive.clip_record()` already returns a JSON record with a proven byte-identity property (omit an optional block and the dict is unchanged); every new evidence source becomes another optional block on that record. Motion v2 runs alongside, never instead of, the existing `decode_clip_bursts` contract, because `shards_multi16` and `surgvu/temporal.py` depend on the current uniform-burst layout. Thresholds are fitted and recorded, never guessed.

**Tech Stack:** Python 3, numpy, OpenCV (`cv2.calcOpticalFlowFarneback`), PyTorch (YOLOv5 via `torch.hub`-free local repo load), pytest.

**Spec:** `docs/design/specs/2026-08-24-v5-parallel-build-design.md`

## Global Constraints

- **Runtime contract:** 10 min per case, one case per container invocation, 32 GB DRAM, no internet.
- **GPU draw is not guaranteed:** either No GPU or a single **T4 (16 GiB, sm_75)**. sm_75 means **no bf16 and no FlashAttention-2**. Every component in this plan must produce a correct answer with **zero GPU**. Optical flow is therefore CPU-only by construction.
- **UI band:** `preprocess.prepare_frame` crops black side margins and blurs the bottom UI band. This is a challenge rule — "using the information available in the UI to make predictions is not allowed". **No code path may bypass `prepare_frame`**, including label generation.
- **Independence:** no imports from `opscribe_pipeline`, and no use of the OpScribe container, venv, HF cache, or `pypkgs`.
- **Login node `ap2001` is for editing and numpy/cv2 tests only.** numpy 2.0.2 and cv2 4.13.0 are importable there; **torch is not**. Any task touching torch runs its tests inside the container via `python3 scripts/run_tests.py`, submitted as a compute job. Never build containers or train on the login node.
- **Additivity:** every new record block is optional. With all flags off, `clip_record()` must return a dict byte-identical to today's. This is asserted, not assumed.
- **Provenance:** every computed statistic carries a version integer in its record. A number whose definition is not written down gets compared against a number it is not comparable with.
- **Branch:** `fix/staging-verification-jpeg-shards`. Commit per task. Do not open a PR, merge to main, make the repo public, or submit to Grand Challenge.
- **No hardcoded decisions.** Any cutoff that decides an answer is fitted on a training split and written to `config/`, with the fitting script committed alongside it.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/surgvu/flow.py` *(new)* | Farnebäck optical flow → magnitude and coherence scalars. Pure cv2/numpy, no torch. |
| `src/surgvu/motion.py` *(modify)* | Add `MOTION_V2_VERSION`, `motion_vector`, `motion_record_v2`. Existing v1 functions untouched. |
| `src/surgvu/perceive.py` *(modify)* | Add `decode_clip_multiscale`; extend `clip_record` with optional `yolo`/`variant`/`agree` blocks. |
| `src/surgvu/detect.py` *(new)* | YOLOv5 adapter: load `best.pt`, run on anchors, emit timestamped detections, map 14→12. |
| `src/surgvu/variant.py` *(new)* | Large/Mega needle-driver head: model definition, inference, config-driven cutoff. |
| `src/surgvu/agreement.py` *(new)* | CNN↔YOLO agreement statistics. |
| `scripts/calibrate_motion_v2.py` *(new)* | Fit burst offsets and thresholds against the `tasks.csv` activity proxy. |
| `scripts/build_variant_labels.py` *(new)* | Generate Large/Mega frame labels from `tools.csv` install intervals. |
| `scripts/train_variant.py` *(new)* | Train the variant head. |
| `scripts/flag_matrix.py` *(new)* | Score every flag combination on the 11-case sample; the attribution instrument. |
| `scripts/inference.py` *(modify)* | New flags: `--motion-v2`, `--yolo`, `--variant-head`. |
| `src/surgvu/router.py` *(modify)* | Capture `large`/`mega` as a variant-qualifier **slot** without registering them as class identifiers. |

---

### Task 1: Optical flow features

Farnebäck dense flow gives what frame differencing structurally cannot: whether the motion is *coherent* (the camera moved, every vector points the same way) or *incoherent* (an instrument moved inside a static scene). That single distinction is the reason to add flow at all.

**Files:**
- Create: `src/surgvu/flow.py`
- Test: `tests/test_flow.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `flow_features(frame_a, frame_b, work_size=128) -> dict` with keys `mag_mean` (float), `mag_p90` (float), `coherence` (float in 0..1). Consumed by Task 3.

- [ ] **Step 1: Write the failing test**

```python
"""Direct tests for surgvu/flow.py.

Coherence is the only reason this module exists, so it is tested by
CONSTRUCTION: a pure translation of the whole frame is what a camera pan
looks like and must read as coherent; independent motion of a small patch
against a static background is what an instrument looks like and must read
as incoherent. Magnitudes alone cannot tell those apart, which is the point.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.flow import flow_features  # noqa: E402


def _texture(size=192, seed=0):
    """A textured frame. Flow needs gradient; a flat field has no solution."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    return np.repeat(base[:, :, None], 3, axis=2)


def test_still_frames_have_near_zero_magnitude():
    frame = _texture()
    out = flow_features(frame, frame.copy())
    assert out["mag_mean"] < 0.1


def test_global_translation_is_coherent():
    """A camera pan: every pixel moves the same way."""
    frame = _texture()
    shifted = np.roll(frame, 6, axis=1)
    out = flow_features(frame, shifted)
    assert out["mag_mean"] > 1.0
    assert out["coherence"] > 0.8


def test_local_motion_is_incoherent():
    """An instrument: a patch moves, the rest of the scene does not."""
    frame = _texture()
    moved = frame.copy()
    patch = frame[40:90, 40:90].copy()
    moved[40:90, 60:110] = patch
    out = flow_features(frame, moved)
    assert out["coherence"] < 0.6


def test_coherence_is_bounded():
    out = flow_features(_texture(), np.roll(_texture(), 4, axis=0))
    assert 0.0 <= out["coherence"] <= 1.0


def test_output_is_json_safe():
    import json
    out = flow_features(_texture(), np.roll(_texture(), 3, axis=1))
    json.loads(json.dumps(out, allow_nan=False))


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        flow_features(_texture(size=192), _texture(size=128))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_flow.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'surgvu.flow'`

- [ ] **Step 3: Write the implementation**

```python
"""Dense optical flow, reduced to three scalars per frame pair.

WHY THIS EXISTS, stated against what it replaces. `surgvu/motion.py` measures
mean absolute frame difference. That statistic answers "did pixels change"
and cannot answer "did the CAMERA move or did an INSTRUMENT move" -- its own
docstring says so. A scope push changes every pixel and reads as high
activity; so does a dissection. The router cannot tell those apart from a
difference scalar, and the difference matters for exactly the questions the
motion gate was opened for.

Flow separates them without a model. If the field is a translation, the
vectors are parallel: the camera moved. If a small region moves against a
static background, the vectors disagree: something in the scene moved. That
is COHERENCE, and it is the only new information here -- the two magnitude
statistics are reported because they are free once the field is computed.

CPU BY CONSTRUCTION. Grand Challenge may allocate no GPU at all, and evidence
that vanishes on half the draws is not evidence. Farneback at 128x128 is
milliseconds and needs nothing but OpenCV.
"""
import cv2
import numpy as np

#: Flow is computed at this resolution. Large enough that an instrument tip
#: spans several pixels between frames, small enough to be free. Larger sizes
#: mostly buy resolution on JPEG noise.
FLOW_SIZE = 128

#: A vector counts as agreeing with the field's dominant direction if it lies
#: within this many degrees of it. 30 degrees is wide enough to tolerate the
#: aperture-problem wobble a real pan produces and narrow enough that
#: independent motion falls outside it.
COHERENCE_DEGREES = 30.0

#: Vectors shorter than this carry no reliable direction -- the angle of a
#: near-zero vector is noise -- so they are excluded from the coherence
#: fraction rather than counted as disagreeing. A still frame therefore has
#: no opinion about coherence instead of a random one.
MIN_MAGNITUDE = 0.25

FLOW_VERSION = 1


def _to_gray(frame):
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[-1] == 3:
        gray = (0.299 * array[..., 2] + 0.587 * array[..., 1]
                + 0.114 * array[..., 0])
    elif array.ndim == 2:
        gray = array
    else:
        raise ValueError("expected (H, W) or (H, W, 3), got %r"
                         % (array.shape,))
    return cv2.resize(gray.astype(np.float32), (FLOW_SIZE, FLOW_SIZE),
                      interpolation=cv2.INTER_AREA)


def flow_features(frame_a, frame_b, work_size=FLOW_SIZE):
    """Three scalars describing the motion field between two frames.

    Returns {"mag_mean", "mag_p90", "coherence"}. All plain Python floats:
    this lands in a record written with json.dumps, which refuses float32.

    `coherence` is the fraction of MOVING vectors within COHERENCE_DEGREES of
    the field's dominant direction, where the dominant direction is the angle
    of the summed vector rather than a mean of angles -- averaging angles
    across the +/-pi wrap gives a direction no vector points in.
    """
    a = np.asarray(frame_a)
    b = np.asarray(frame_b)
    if a.shape != b.shape:
        raise ValueError(
            "flow needs two frames of the same shape, got %r and %r. "
            "Resizing one to match would invent motion at the seams."
            % (a.shape, b.shape))
    gray_a, gray_b = _to_gray(a), _to_gray(b)
    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0)
    dx, dy = flow[..., 0], flow[..., 1]
    magnitude = np.sqrt(dx * dx + dy * dy)

    moving = magnitude >= MIN_MAGNITUDE
    if not moving.any():
        coherence = 0.0
    else:
        sum_x, sum_y = float(dx[moving].sum()), float(dy[moving].sum())
        if sum_x == 0.0 and sum_y == 0.0:
            coherence = 0.0
        else:
            norm = np.hypot(sum_x, sum_y)
            unit_x, unit_y = sum_x / norm, sum_y / norm
            # cos of the angle between each moving vector and the dominant
            # direction, via the normalised dot product.
            dot = (dx[moving] * unit_x + dy[moving] * unit_y) / magnitude[moving]
            coherence = float(
                (dot >= np.cos(np.deg2rad(COHERENCE_DEGREES))).mean())

    return {
        "mag_mean": float(magnitude.mean()),
        "mag_p90": float(np.percentile(magnitude, 90)),
        "coherence": coherence,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_flow.py -q`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add src/surgvu/flow.py tests/test_flow.py
git commit -m "feat(flow): Farneback flow features with camera/tool coherence

Frame differencing cannot separate camera motion from instrument motion --
motion.py's own docstring records that limitation. Coherence (fraction of
moving vectors within 30 deg of the dominant direction) does, with no model
and no GPU. Tested by construction: a global roll must read coherent, a
moved patch against a static background must not."
```

---

### Task 2: Multi-scale burst decode

The existing burst samples t−67 ms, t, t+67 ms; macro compares burst centres 1.875 s apart. **Nothing is sampled between 67 ms and 1875 ms — a 28× span.** This adds probes inside that gap without touching the existing burst, because `shards_multi16`, `surgvu/dataset.py` and `surgvu/temporal.py` all depend on the current uniform layout.

**Files:**
- Modify: `src/surgvu/perceive.py` (add after `decode_clip_bursts`, ~line 200)
- Test: `tests/test_perceive_multiscale.py`

**Interfaces:**
- Consumes: `sample_frame_indices`, `prepare_frame` (already in `perceive.py`).
- Produces: `decode_clip_multiscale(video_path, n_frames=16, offsets_ms=(133, 400, 1200), size=512) -> (centres, probes)` where `centres` is `(n, size, size, 3)` uint8 identical to `decode_clip`'s output, and `probes` is a list of length `n`; each entry is `{offset_ms: (before, after)}` mapping each offset to a pair of frames, or `None` for any offset whose frames could not be read. Consumed by Task 3.

- [ ] **Step 1: Write the failing test**

```python
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

from surgvu.perceive import decode_clip, decode_clip_multiscale  # noqa: E402


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_perceive_multiscale.py -q`
Expected: FAIL — `ImportError: cannot import name 'decode_clip_multiscale'`

- [ ] **Step 3: Write the implementation**

Append to `src/surgvu/perceive.py`:

```python
#: Probe offsets in milliseconds, log-spaced to fill the gap the existing
#: sampler leaves. decode_clip_bursts measures at 67 ms and macro activity at
#: 1875 ms; between those two nothing is sampled at all, and that is precisely
#: the range a tool stroke occupies. These are DEFAULTS and are overridden by
#: whatever scripts/calibrate_motion_v2.py fits and writes to config.
DEFAULT_PROBE_OFFSETS_MS = (133, 400, 1200)


def decode_clip_multiscale(video_path, n_frames=DEFAULT_FRAMES,
                           offsets_ms=DEFAULT_PROBE_OFFSETS_MS, size=512):
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
        for index in sample_frame_indices(total, n_frames):
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_perceive_multiscale.py tests/test_perceive.py -q`
Expected: PASS. `tests/test_perceive.py` must still pass — it holds the existing `decode_clip_bursts` contract.

- [ ] **Step 5: Commit**

```bash
git add src/surgvu/perceive.py tests/test_perceive_multiscale.py
git commit -m "feat(perceive): multi-scale probe decode alongside the 67ms burst

The shipped sampler measures at 67ms and 1875ms and nothing between, which
is where a tool stroke lives. This adds symmetric probes at configurable
offsets while leaving decode_clip_bursts untouched, because its uniform
layout is the shard format dataset.py and temporal.py read. Centres are
asserted byte-identical to decode_clip."
```

---

### Task 3: Motion vector and the v2 record

Turns the probes and flow into the eight-element vector the spec defines, and packages it with a version integer so a v2 record can never be mistaken for a v1 one.

**Files:**
- Modify: `src/surgvu/motion.py`
- Test: `tests/test_motion_v2.py`

**Interfaces:**
- Consumes: `flow_features` (Task 1); the `probes` structure (Task 2).
- Produces:
  - `motion_vector(centre_before, centre, centre_after, probe_entry) -> dict` with keys `micro_short`, `micro_mid`, `micro_long`, `macro_prev`, `macro_next`, `flow_mag_mean`, `flow_mag_p90`, `flow_coherence`. Missing measurements are `None`, never `0.0`.
  - `motion_record_v2(centres, probes) -> dict` with keys `version`, `anchors`, `per_anchor` (list of vectors), `summary`.
  Consumed by Tasks 4, 5, 6.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the v2 motion record.

The property that matters most is the one v1 already established and v2 must
not lose: an unavailable measurement is None, not zero. A missing flank is
not evidence that nothing moved, and a record that says 0.0 where it means
"unknown" will be averaged into a threshold as though it were a measurement.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.motion import (MOTION_V2_VERSION, motion_record_v2,  # noqa: E402
                           motion_vector)


def _frame(value, size=96, seed=None):
    if seed is None:
        return np.full((size, size, 3), value, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)


def test_still_scene_reads_near_zero_micro():
    still = _frame(100)
    probe = {133: (still.copy(), still.copy()),
             400: (still.copy(), still.copy())}
    vector = motion_vector(still.copy(), still, still.copy(), probe)
    assert vector["micro_short"] == pytest.approx(0.0, abs=1e-5)


def test_moving_scene_reads_above_still():
    base = _frame(0, seed=1)
    moved = np.roll(base, 12, axis=1)
    probe = {133: (base.copy(), moved.copy())}
    active = motion_vector(base.copy(), base, moved, probe)

    still = _frame(100)
    quiet = motion_vector(still.copy(), still, still.copy(),
                          {133: (still.copy(), still.copy())})
    assert active["micro_short"] > quiet["micro_short"]


def test_unavailable_offset_is_none_not_zero():
    still = _frame(100)
    vector = motion_vector(still.copy(), still, still.copy(), {133: None})
    assert vector["micro_short"] is None


def test_missing_neighbour_leaves_macro_none():
    still = _frame(100)
    vector = motion_vector(None, still, None, {133: (still, still)})
    assert vector["macro_prev"] is None
    assert vector["macro_next"] is None


def test_record_carries_its_version():
    frames = np.stack([_frame(0, seed=i) for i in range(4)])
    probes = [{133: (frames[0], frames[1])} for _ in range(4)]
    record = motion_record_v2(frames, probes)
    assert record["version"] == MOTION_V2_VERSION
    assert record["version"] != 1        # must not collide with the v1 record


def test_record_is_valid_strict_json():
    frames = np.stack([_frame(0, seed=i) for i in range(4)])
    probes = [{133: None} for _ in range(4)]
    record = motion_record_v2(frames, probes)
    json.loads(json.dumps(record, allow_nan=False))


def test_summary_ignores_none_rather_than_counting_it_as_zero():
    frames = np.stack([_frame(0, seed=i) for i in range(4)])
    probes = [{133: None}, {133: None}, {133: None}, {133: None}]
    record = motion_record_v2(frames, probes)
    assert record["summary"]["micro_short"]["mean"] is None
    assert record["summary"]["micro_short"]["measured"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_motion_v2.py -q`
Expected: FAIL — `ImportError: cannot import name 'MOTION_V2_VERSION'`

- [ ] **Step 3: Write the implementation**

Append to `src/surgvu/motion.py`:

```python
from .flow import flow_features

#: Distinct from MOTION_VERSION so a v2 record can never be read as a v1 one.
#: The v4 lesson, applied again: a number whose provenance is not written down
#: gets compared to a number it is not comparable with.
MOTION_V2_VERSION = 2

#: Which probe offset feeds which slot of the vector. Chosen so the three
#: micro slots span the 67ms-1875ms gap the shipped sampler leaves. The values
#: are DEFAULTS; scripts/calibrate_motion_v2.py fits them and writes the fitted
#: set to config, and the record names the offsets it actually used.
VECTOR_SLOTS = (("micro_short", 133), ("micro_mid", 400),
                ("micro_long", 1200))


def _mad(frame_a, frame_b):
    """Mean absolute difference between two frames, in 0-255 units."""
    work = _to_work(np.stack([np.asarray(frame_a), np.asarray(frame_b)]))
    return float(np.abs(work[1] - work[0]).mean())


def motion_vector(centre_before, centre, centre_after, probe_entry):
    """The eight-element motion description of one anchor.

    Every element is a float or None. NONE MEANS UNAVAILABLE, never still --
    the distinction v1 already makes and the one a threshold fitted over these
    numbers depends on. A missing flank averaged in as 0.0 would drag a case
    toward "quiet" on the strength of a failed disk read.

    `centre_before` / `centre_after` are the neighbouring ANCHORS (1.875 s
    away at the shipped sampling rate), so the macro slots describe how the
    scene changes across the moments the appearance model reports on. Either
    may be None at the ends of the window.

    Flow is computed on the SHORTEST available probe pair. Flow's assumption
    is small displacement; at 1.2 s a tool can cross the frame and the field
    stops meaning anything.
    """
    vector = {}
    for name, offset in VECTOR_SLOTS:
        pair = (probe_entry or {}).get(offset)
        vector[name] = None if pair is None else _mad(pair[0], pair[1])

    vector["macro_prev"] = (None if centre_before is None
                            else _mad(centre_before, centre))
    vector["macro_next"] = (None if centre_after is None
                            else _mad(centre, centre_after))

    flow_pair = None
    for _, offset in VECTOR_SLOTS:
        pair = (probe_entry or {}).get(offset)
        if pair is not None:
            flow_pair = pair
            break
    if flow_pair is None:
        vector.update(flow_mag_mean=None, flow_mag_p90=None,
                      flow_coherence=None)
    else:
        flow = flow_features(flow_pair[0], flow_pair[1])
        vector.update(flow_mag_mean=flow["mag_mean"],
                      flow_mag_p90=flow["mag_p90"],
                      flow_coherence=flow["coherence"])
    return vector


_VECTOR_KEYS = ("micro_short", "micro_mid", "micro_long",
                "macro_prev", "macro_next",
                "flow_mag_mean", "flow_mag_p90", "flow_coherence")


def motion_record_v2(centres, probes):
    """Per-anchor motion vectors plus a summary that excludes what it lacks.

    `centres` is the (n, H, W, 3) stack from decode_clip_multiscale; `probes`
    is its companion list. Returns a record whose "summary" reports, for every
    slot, the mean/max over the anchors that HAVE a measurement together with
    how many that was. A mean over three of sixteen anchors is not the same
    measurement as a mean over sixteen and a reader must be able to tell --
    the same reasoning behind v1's `bursts_measured`.
    """
    array = np.asarray(centres)
    if array.ndim != 4:
        raise ValueError("expected (N, H, W, 3) centres, got %r"
                         % (array.shape,))
    if len(probes) != array.shape[0]:
        raise ValueError(
            "%d probe entries for %d centres; a vector would be built from "
            "another anchor's neighbours"
            % (len(probes), array.shape[0]))

    per_anchor = []
    for index in range(array.shape[0]):
        before = array[index - 1] if index > 0 else None
        after = array[index + 1] if index + 1 < array.shape[0] else None
        per_anchor.append(
            motion_vector(before, array[index], after, probes[index]))

    summary = {}
    for key in _VECTOR_KEYS:
        values = [v[key] for v in per_anchor if v[key] is not None]
        summary[key] = {
            "mean": float(np.mean(values)) if values else None,
            "max": float(np.max(values)) if values else None,
            "measured": len(values),
        }

    return {
        "version": MOTION_V2_VERSION,
        "anchors": int(array.shape[0]),
        "offsets_ms": [offset for _, offset in VECTOR_SLOTS],
        "per_anchor": per_anchor,
        "summary": summary,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_motion_v2.py tests/test_motion.py -q`
Expected: PASS. `tests/test_motion.py` must still pass — v1 is untouched.

- [ ] **Step 5: Commit**

```bash
git add src/surgvu/motion.py tests/test_motion_v2.py
git commit -m "feat(motion): v2 per-anchor motion vectors with flow

Eight slots per anchor across three timescales plus flow magnitude and
coherence, versioned 2 so it cannot be confused with the v1 record. Keeps
v1's hardest-won property: an unavailable measurement is None, never 0.0,
and the summary reports how many anchors it actually saw."
```

---

### Task 4: Calibrate the probe offsets against a real objective

The current ±67 ms was never swept against anything. `tasks.csv` gives a free objective: timestamps inside an annotated task interval are active, timestamps between intervals are idle. That is a real label over 155 cases.

**Files:**
- Create: `scripts/calibrate_motion_v2.py`
- Create: `config/motion_v2.json` (produced by the script, committed)
- Test: `tests/test_calibrate_motion_v2.py`

**Interfaces:**
- Consumes: `motion_record_v2` (Task 3).
- Produces: `activity_labels(tasks_csv_path, timestamps) -> list[bool]`, and `separability(values, labels) -> float` (AUC). The written config has shape `{"version": 2, "offsets_ms": [...], "thresholds": {"<slot>": {"cut": float, "auc": float}}, "fitted_on": "<split name>", "n_cases": int}`. Consumed by Task 5.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the motion-v2 calibration objective.

The point of this file is that the objective is REAL -- a statistic that
cannot separate annotated task intervals from the gaps between them is not
measuring surgical activity, whatever its docstring claims. AUC is used
rather than accuracy because the classes are unbalanced and a threshold has
not been chosen yet.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from calibrate_motion_v2 import activity_labels, separability  # noqa: E402


def _tasks_csv(tmp_path):
    path = tmp_path / "tasks.csv"
    path.write_text(
        "start,stop,taskname,groundtruth_taskname,matched_description\n"
        "10.0,20.0,suturing,suturing,sews tissue\n"
        "40.0,50.0,dissection,dissection,separates tissue\n",
        encoding="utf-8")
    return path


def test_timestamp_inside_an_interval_is_active(tmp_path):
    labels = activity_labels(_tasks_csv(tmp_path), [15.0])
    assert labels == [True]


def test_timestamp_between_intervals_is_idle(tmp_path):
    labels = activity_labels(_tasks_csv(tmp_path), [30.0])
    assert labels == [False]


def test_interval_boundaries_are_inclusive(tmp_path):
    assert activity_labels(_tasks_csv(tmp_path), [10.0, 20.0]) == [True, True]


def test_perfect_separation_scores_one():
    assert separability([0.1, 0.2, 5.0, 6.0],
                        [False, False, True, True]) == pytest.approx(1.0)


def test_reversed_separation_scores_zero():
    assert separability([5.0, 6.0, 0.1, 0.2],
                        [False, False, True, True]) == pytest.approx(0.0)


def test_no_signal_scores_one_half():
    assert separability([1.0, 1.0, 1.0, 1.0],
                        [False, True, False, True]) == pytest.approx(0.5)


def test_none_values_are_dropped_not_counted_as_zero():
    """An unavailable measurement must not be scored as a quiet one."""
    assert separability([None, 0.1, 5.0], [True, False, True]) == pytest.approx(1.0)


def test_refuses_a_single_class():
    with pytest.raises(ValueError):
        separability([1.0, 2.0], [True, True])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_calibrate_motion_v2.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'calibrate_motion_v2'`

- [ ] **Step 3: Write the implementation**

```python
"""Fit motion-v2 probe offsets and thresholds against annotated task intervals.

WHY A PROXY OBJECTIVE. The shipped burst offset of 67 ms was chosen and never
swept against anything, so "is 67 ms right" has no answer on record. It does
not need a new annotation to get one: tasks.csv already marks when surgical
work is happening. A timestamp inside an annotated interval is active; one in
the gap between intervals is idle. That is a real label over 155 cases and it
costs nothing.

WHAT THIS OBJECTIVE IS NOT. It is not the answer metric, and a probe offset
that separates task intervals best is not thereby proven to raise BERTScore.
The gaps between annotated tasks also contain real surgery that simply was
not annotated, so "idle" is noisy in a known direction. Recorded here rather
than discovered later: this picks between candidate offsets, it does not
prove the winner helps.

AUC rather than accuracy, because the two classes are unbalanced and the
point of the sweep is to compare statistics before any threshold exists.
"""
import argparse
import csv
import json
from pathlib import Path


def activity_labels(tasks_csv, timestamps):
    """True where a timestamp falls inside an annotated task interval.

    Boundaries are inclusive: a frame at the annotated start of a task is
    part of that task.
    """
    intervals = []
    with open(tasks_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                intervals.append((float(row["start"]), float(row["stop"])))
            except (KeyError, TypeError, ValueError):
                continue
    return [any(start <= t <= stop for start, stop in intervals)
            for t in timestamps]


def separability(values, labels):
    """AUC of `values` against boolean `labels`, by rank.

    None values are DROPPED with their label rather than substituted. An
    unavailable measurement scored as 0.0 would look like a quiet frame and
    would be counted as evidence the statistic works.
    """
    pairs = [(v, bool(l)) for v, l in zip(values, labels) if v is not None]
    positives = [v for v, l in pairs if l]
    negatives = [v for v, l in pairs if not l]
    if not positives or not negatives:
        raise ValueError(
            "AUC needs both classes; got %d active and %d idle. A sweep over "
            "one class would report a number that means nothing."
            % (len(positives), len(negatives)))
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positives) * len(negatives))


def best_cut(values, labels):
    """The threshold maximising balanced accuracy, and that accuracy."""
    pairs = sorted((v, bool(l)) for v, l in zip(values, labels)
                   if v is not None)
    if not pairs:
        raise ValueError("no measured values to fit a cut against")
    candidates = sorted({v for v, _ in pairs})
    positives = sum(1 for _, l in pairs if l)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        raise ValueError("a cut needs both classes present")
    best, best_score = candidates[0], -1.0
    for cut in candidates:
        tp = sum(1 for v, l in pairs if l and v >= cut)
        tn = sum(1 for v, l in pairs if not l and v < cut)
        score = 0.5 * (tp / positives + tn / negatives)
        if score > best_score:
            best, best_score = cut, score
    return float(best), float(best_score)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump", required=True,
                        help="JSON produced by scripts/sample_motion.py: a "
                             "list of {case, t, vector} records")
    parser.add_argument("--labels-root", required=True,
                        help="SURGVU25_train_labels root, one dir per case")
    parser.add_argument("--split", default="train",
                        help="recorded in the output so a threshold can never "
                             "be quoted without the split it was fitted on")
    parser.add_argument("--out", default="config/motion_v2.json")
    args = parser.parse_args(argv)

    records = json.loads(Path(args.dump).read_text(encoding="utf-8"))
    by_case = {}
    for record in records:
        by_case.setdefault(record["case"], []).append(record)

    slots = ("micro_short", "micro_mid", "micro_long", "macro_prev",
             "macro_next", "flow_mag_mean", "flow_mag_p90", "flow_coherence")
    columns = {slot: [] for slot in slots}
    labels = []
    for case, rows in sorted(by_case.items()):
        tasks_csv = Path(args.labels_root) / case / "tasks.csv"
        if not tasks_csv.exists():
            print("skipping %s: no tasks.csv" % (case,))
            continue
        stamps = [row["t"] for row in rows]
        labels.extend(activity_labels(tasks_csv, stamps))
        for slot in slots:
            columns[slot].extend(row["vector"].get(slot) for row in rows)

    thresholds = {}
    for slot in slots:
        try:
            auc = separability(columns[slot], labels)
            cut, balanced = best_cut(columns[slot], labels)
        except ValueError as exc:
            print("%-16s unusable: %s" % (slot, exc))
            continue
        thresholds[slot] = {"cut": cut, "auc": auc,
                            "balanced_accuracy": balanced,
                            "measured": sum(1 for v in columns[slot]
                                            if v is not None)}
        print("%-16s auc=%.4f cut=%.4f balacc=%.4f  n=%d"
              % (slot, auc, cut, balanced, thresholds[slot]["measured"]))

    out = {
        "version": 2,
        "offsets_ms": [133, 400, 1200],
        "thresholds": thresholds,
        "fitted_on": args.split,
        "n_cases": len(by_case),
        "objective": "tasks.csv interval membership (proxy, not the metric)",
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n",
                              encoding="utf-8")
    print("wrote %s" % (args.out,))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_calibrate_motion_v2.py -q`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_motion_v2.py tests/test_calibrate_motion_v2.py
git commit -m "feat(motion): fit probe offsets against tasks.csv activity proxy

The shipped 67ms offset was never swept against any objective. tasks.csv
interval membership is a free label over 155 cases. AUC not accuracy: the
classes are unbalanced and no threshold exists yet. The docstring records
what the proxy is NOT -- unannotated surgery makes 'idle' noisy in a known
direction, so this ranks candidates, it does not prove the winner helps."
```

---

### Task 5: Wire `--motion-v2` into serving

**Files:**
- Modify: `scripts/inference.py` (flag near `--motion`, branch at ~line 665)
- Modify: `src/surgvu/perceive.py` (`clip_record`, ~line 270)
- Test: `tests/test_inference_motion_v2.py`

**Interfaces:**
- Consumes: `decode_clip_multiscale` (Task 2), `motion_record_v2` (Task 3).
- Produces: `clip_record(..., motion_v2=None)` — an optional `"motion_v2"` key on the record. Consumed by Tasks 6–9 and by Plan 2's VLM.

- [ ] **Step 1: Write the failing test**

```python
"""Additivity of the motion_v2 block.

clip_record's guarantee is that an omitted optional block leaves the dict
byte-identical to what it was before that block existed. That property is why
new evidence can be added without re-measuring every shipped answer, and it
holds only as long as every new block is tested for it.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.perceive import clip_record  # noqa: E402
from surgvu.taxonomy import TASK_CLASSES, TOOL_CLASSES  # noqa: E402


def _meta(classes):
    return {"classes": list(classes),
            "thresholds": [0.5] * len(classes)}


def _args():
    return ([0.1] * len(TOOL_CLASSES), _meta(TOOL_CLASSES),
            [0.1] * len(TASK_CLASSES), _meta(TASK_CLASSES), 16)


def test_omitting_motion_v2_changes_nothing():
    assert clip_record(*_args()) == clip_record(*_args(), motion_v2=None)


def test_supplying_motion_v2_only_appends_one_key():
    plain = clip_record(*_args())
    block = {"version": 2, "anchors": 16, "per_anchor": [], "summary": {}}
    with_v2 = clip_record(*_args(), motion_v2=block)
    assert set(with_v2) - set(plain) == {"motion_v2"}
    for key in plain:
        assert with_v2[key] == plain[key]


def test_motion_v2_survives_strict_json():
    block = {"version": 2, "anchors": 1,
             "per_anchor": [{"micro_short": None}], "summary": {}}
    record = clip_record(*_args(), motion_v2=block)
    json.loads(json.dumps(record, allow_nan=False))


def test_v1_and_v2_blocks_coexist():
    record = clip_record(*_args(),
                         motion={"version": 1, "bursts": 16},
                         motion_v2={"version": 2, "anchors": 16})
    assert record["motion"]["version"] == 1
    assert record["motion_v2"]["version"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_inference_motion_v2.py -q`
Expected: FAIL — `TypeError: clip_record() got an unexpected keyword argument 'motion_v2'`

- [ ] **Step 3: Write the implementation**

In `src/surgvu/perceive.py`, change the `clip_record` signature and tail:

```python
def clip_record(tool_probs, tool_meta, task_probs, task_meta, n_frames,
                motion=None, motion_v2=None, yolo=None, variant=None,
                agree=None):
```

and replace the closing block:

```python
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
```

In `scripts/inference.py`, add the flag beside `--motion`:

```python
    parser.add_argument("--motion-v2", action="store_true",
                        help="multi-scale motion probes plus optical-flow "
                             "coherence. Independent of --motion: the v1 "
                             "block feeds a calibrated router gate that is "
                             "already shipping, and removing it here would "
                             "change answers for a reason unrelated to this "
                             "flag.")
```

and extend the decode branch:

```python
        motion = None
        motion_v2 = None
        if args.motion_v2:
            from surgvu.motion import motion_record_v2
            from surgvu.perceive import decode_clip_multiscale
            with timed("decode", timings):
                frames, probes = decode_clip_multiscale(
                    video, n_frames=frames_wanted, size=size)
            with timed("motion_v2", timings):
                motion_v2 = motion_record_v2(frames, probes)
            log("motion_v2 anchors=%d flow_coherence=%s micro_mid=%s"
                % (motion_v2["anchors"],
                   motion_v2["summary"]["flow_coherence"]["mean"],
                   motion_v2["summary"]["micro_mid"]["mean"]))
            if args.motion:
                # Both blocks requested: the v1 gate needs the uniform burst
                # layout, which the multiscale decode does not produce. Pay
                # the second decode rather than approximate one from the other.
                from surgvu.motion import motion_record_from_bursts
                with timed("decode_bursts", timings):
                    _, bursts = decode_clip_bursts(
                        video, n_frames=frames_wanted, size=size)
                with timed("motion", timings):
                    motion = motion_record_from_bursts(bursts, frames)
        elif args.motion:
            ...  # existing branch, unchanged
        else:
            ...  # existing branch, unchanged
```

and thread it through: `infer_with_retry(..., motion=motion, motion_v2=motion_v2)` → `infer(...)` → `clip_record(..., motion_v2=motion_v2)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_inference_motion_v2.py tests/test_perceive.py -q`
Expected: PASS. `tests/test_perceive.py` holds the pre-existing additivity contract and must not regress.

- [ ] **Step 5: Commit**

```bash
git add src/surgvu/perceive.py scripts/inference.py tests/test_inference_motion_v2.py
git commit -m "feat(inference): --motion-v2 flag, additive motion_v2 record block

clip_record gains four optional blocks (motion_v2, yolo, variant, agree)
under one additive loop, preserving the byte-identity guarantee. --motion-v2
is independent of --motion because the v1 gate ships today and needs the
uniform burst layout the multiscale decode does not produce."
```

---

### Task 6: YOLO detector adapter

`best.pt` is a 14-class YOLOv5s trained by a groupmate on 886 clip-disjoint images: P 0.773 / R 0.740 / mAP@0.5 0.773. Per-class recall is what matters here, and it separates bipolar↔cadiere at 0.01 confusion each way — the pair that costs us case124, the single largest recoverable item on the sample.

**Files:**
- Create: `src/surgvu/detect.py`
- Test: `tests/test_detect.py`

**Interfaces:**
- Consumes: `TOOL_CLASSES` from `surgvu.taxonomy`.
- Produces:
  - `YOLO_CLASSES` — the 14 names in her `surg_14cls.yaml` index order.
  - `map_to_taxonomy(name) -> str | None` — 14→12, returning `None` for the two out-of-taxonomy classes.
  - `detections_to_record(detections, timestamps) -> dict` with keys `version`, `per_anchor`, `by_class`.
  - `Detector.detect(frames) -> list[list[dict]]`, each dict `{cls, conf, box, anchor_idx, t_seconds}`.
  Consumed by Tasks 7 and 9.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the YOLO adapter.

Deliberately torch-free: the mapping, the record shape and the timestamping
are the parts that decide answers, and they must be testable on the login
node where torch does not exist. Detector.detect itself is exercised in the
container by tests/test_detect_weights.py.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.detect import (YOLO_CLASSES, detections_to_record,  # noqa: E402
                           map_to_taxonomy)
from surgvu.taxonomy import TOOL_CLASSES  # noqa: E402


def test_fourteen_classes_in_yaml_order():
    assert len(YOLO_CLASSES) == 14
    assert YOLO_CLASSES[0] == "bipolar dissector"


def test_twelve_of_fourteen_map_into_the_taxonomy():
    mapped = [map_to_taxonomy(name) for name in YOLO_CLASSES]
    assert sorted(n for n in mapped if n) == sorted(TOOL_CLASSES)


def test_out_of_taxonomy_classes_map_to_none():
    """Kept as evidence, never emitted as an answer."""
    assert map_to_taxonomy("bipolar dissector") is None
    assert map_to_taxonomy("suction irrigator") is None


def test_unknown_name_raises_rather_than_silently_dropping():
    with pytest.raises(KeyError):
        map_to_taxonomy("laser sword")


def test_record_preserves_time_not_just_presence():
    detections = [
        [{"cls": "needle driver", "conf": 0.9, "box": [0, 0, 1, 1]}],
        [],
        [{"cls": "needle driver", "conf": 0.8, "box": [0, 0, 1, 1]}],
    ]
    record = detections_to_record(detections, [0.0, 1.875, 3.75])
    times = [d["t_seconds"] for d in record["by_class"]["needle driver"]]
    assert times == [0.0, 3.75]


def test_record_reports_max_confidence_per_class():
    detections = [
        [{"cls": "needle driver", "conf": 0.4, "box": [0, 0, 1, 1]}],
        [{"cls": "needle driver", "conf": 0.9, "box": [0, 0, 1, 1]}],
    ]
    record = detections_to_record(detections, [0.0, 1.875])
    assert record["max_conf"]["needle driver"] == pytest.approx(0.9)


def test_empty_detections_give_an_empty_record_not_a_crash():
    record = detections_to_record([[], []], [0.0, 1.875])
    assert record["by_class"] == {}
    assert record["max_conf"] == {}


def test_record_is_strict_json():
    record = detections_to_record(
        [[{"cls": "stapler", "conf": 0.5, "box": [1, 2, 3, 4]}]], [0.0])
    json.loads(json.dumps(record, allow_nan=False))


def test_mismatched_timestamps_raise():
    with pytest.raises(ValueError):
        detections_to_record([[], []], [0.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_detect.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'surgvu.detect'`

- [ ] **Step 3: Write the implementation**

```python
"""YOLOv5 tool detection as a second opinion on the CNN heads.

WHAT THIS ADDS THAT THE CNNs DO NOT. The tool heads are whole-frame
multi-label classifiers: they report that a needle driver is present, not
where or when. A detector reports both, and the "when" is the part this
pipeline has never had -- pooled confidences are a single number for a 30 s
clip, so "needle driver at 3.7 s and 9.4 s but not between" is a statement
the record could not previously make.

WHY IT IS ADDITIVE AND NOT A REPLACEMENT. The CNN path is measured: it earns
0.8766 on the sample against 0.8294 for a blind router. This detector has
never been scored against BERTScore at all. It joins as evidence, and its
DISAGREEMENT with the CNNs is itself a signal (see surgvu/agreement.py) --
arguably the more valuable half, because two independently-wrong models
rarely fail the same way.

FOURTEEN CLASSES, TWELVE ANSWERS. The detector's vocabulary is our twelve
plus `bipolar dissector` and `suction irrigator`. Those two are KEPT in the
record and never emitted as an answer: a confident suction-irrigator
detection constrains what else is in the frame, which is useful even though
it can never be the reply. Mapping happens at answer-formatting time, not
here, so the record stays a faithful account of what was seen.
"""
from .taxonomy import TOOL_CLASSES

DETECT_VERSION = 1

#: Index order from surg_14cls.yaml in the groupmate's yolo_dataset. The
#: ORDER IS THE CONTRACT -- a checkpoint's class indices mean nothing without
#: it, and reordering this silently relabels every detection.
YOLO_CLASSES = (
    "bipolar dissector",             # 0   out of taxonomy
    "bipolar forceps",               # 1
    "cadiere forceps",               # 2
    "clip applier",                  # 3
    "force bipolar",                 # 4
    "grasping retractor",            # 5
    "monopolar curved scissors",     # 6
    "needle driver",                 # 7
    "permanent cautery hook/spatula",# 8
    "prograsp forceps",              # 9
    "stapler",                       # 10
    "suction irrigator",             # 11  out of taxonomy
    "tip-up fenestrated grasper",    # 12
    "vessel sealer",                 # 13
)

#: Present in the detector, absent from the answer taxonomy. Evidence only.
OUT_OF_TAXONOMY = frozenset({"bipolar dissector", "suction irrigator"})

_TOOL_SET = frozenset(TOOL_CLASSES)


def map_to_taxonomy(name):
    """14-class detector name -> 12-class answer name, or None.

    Raises on a name the detector cannot produce. Returning None for an
    unknown string would let a typo in a config silently delete a class's
    detections, and the record would look merely empty rather than wrong.
    """
    if name not in YOLO_CLASSES:
        raise KeyError(
            "%r is not one of the detector's 14 classes. The class list is a "
            "contract with the checkpoint; a name outside it means the "
            "weights and this table disagree." % (name,))
    if name in OUT_OF_TAXONOMY:
        return None
    if name not in _TOOL_SET:
        raise KeyError(
            "%r is in the detector vocabulary but not in TOOL_CLASSES and not "
            "declared out-of-taxonomy. Refusing to guess which it is."
            % (name,))
    return name


def detections_to_record(detections, timestamps):
    """Per-anchor detections -> the record block, time preserved.

    `detections[i]` is the list for anchor i; `timestamps[i]` is that anchor's
    time in seconds within the clip. Every detection is stamped, so downstream
    can ask when a tool appeared and not merely whether.
    """
    if len(detections) != len(timestamps):
        raise ValueError(
            "%d anchors of detections against %d timestamps; a detection "
            "would be stamped with another anchor's time."
            % (len(detections), len(timestamps)))

    per_anchor, by_class, max_conf = [], {}, {}
    for index, (found, when) in enumerate(zip(detections, timestamps)):
        stamped = []
        for item in found:
            entry = {
                "cls": item["cls"],
                "conf": float(item["conf"]),
                "box": [float(v) for v in item["box"]],
                "anchor_idx": index,
                "t_seconds": float(when),
            }
            stamped.append(entry)
            by_class.setdefault(item["cls"], []).append(entry)
            max_conf[item["cls"]] = max(max_conf.get(item["cls"], 0.0),
                                        entry["conf"])
        per_anchor.append(stamped)

    return {
        "version": DETECT_VERSION,
        "classes": list(YOLO_CLASSES),
        "per_anchor": per_anchor,
        "by_class": by_class,
        "max_conf": max_conf,
    }


class Detector:
    """Loads `best.pt` once and runs it over a clip's anchors.

    Torch is imported INSIDE the methods, not at module scope, so the mapping
    and record functions above stay importable on a machine without torch --
    which is where most of their tests run.
    """

    def __init__(self, weights, repo_dir, conf=0.25, iou=0.45, device="cpu"):
        self.weights = str(weights)
        self.repo_dir = str(repo_dir)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        import sys
        import torch
        # The local yolov5 checkout, not torch.hub: the container has no
        # internet, and a hub fetch would fail at serving time on a machine
        # nobody can log into.
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)
        from models.common import DetectMultiBackend
        model = DetectMultiBackend(self.weights, device=torch.device(self.device))
        model.eval()
        self._model = model
        return model

    def detect(self, frames, size=640):
        """(N, H, W, 3) uint8 BGR -> list of N detection lists."""
        import numpy as np
        import torch
        from utils.general import non_max_suppression, scale_boxes

        model = self._load()
        out = []
        for frame in np.asarray(frames):
            import cv2
            resized = cv2.resize(frame, (size, size))
            tensor = torch.from_numpy(
                resized[:, :, ::-1].copy()).permute(2, 0, 1).float()
            tensor = (tensor / 255.0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                raw = model(tensor)
            kept = non_max_suppression(raw, self.conf, self.iou)[0]
            found = []
            if kept is not None and len(kept):
                boxes = scale_boxes(tensor.shape[2:], kept[:, :4].clone(),
                                    frame.shape).round()
                for box, row in zip(boxes.tolist(), kept.tolist()):
                    found.append({"cls": YOLO_CLASSES[int(row[5])],
                                  "conf": float(row[4]),
                                  "box": [float(v) for v in box]})
            out.append(found)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_detect.py -q`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add src/surgvu/detect.py tests/test_detect.py
git commit -m "feat(detect): YOLOv5 adapter with timestamped detections

Wraps the groupmate's 14-class best.pt as a second opinion. Detections keep
their anchor index and time in seconds, which the pooled CNN confidences
never carried. The two out-of-taxonomy classes are kept as evidence and
mapped to None at answer time, not dropped at read time. Torch imports are
function-local so the mapping and record logic stay testable without torch."
```

---

### Task 7: CNN↔YOLO agreement

Two independently-trained models that agree are better evidence than either alone; when they disagree, that is the honest uncertainty channel — and unlike self-consistency, it cannot be confidently wrong in unison. The groupmate's own temperature sweep found the VLM agreeing 2/2 on wrong answers at temperature 0.1, which is exactly why agreement between *different* models is the signal worth having.

**Files:**
- Create: `src/surgvu/agreement.py`
- Test: `tests/test_agreement.py`

**Interfaces:**
- Consumes: `detections_to_record` output (Task 6); the `tools` probability map from `clip_record`.
- Produces: `agreement_record(tool_probs, tool_thresholds, yolo_record) -> dict` with keys `version`, `tool_agreement` (float 0..1), `both_present` (list), `cnn_only` (list), `yolo_only` (list), `top_disagreement` (`[str, str]` or `None`). Consumed by Task 9 and by Plan 2's VLM.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for CNN/YOLO agreement.

The asymmetry matters: the detector has two classes the CNNs cannot name, and
a detection of one of those is not a disagreement -- the CNN was never asked.
Counting it as one would make every frame containing a suction irrigator look
uncertain.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.agreement import agreement_record  # noqa: E402


def _yolo(max_conf):
    return {"version": 1, "max_conf": dict(max_conf), "by_class": {},
            "per_anchor": []}


def test_full_agreement_scores_one():
    out = agreement_record({"needle driver": 0.9, "stapler": 0.1},
                           {"needle driver": 0.5, "stapler": 0.5},
                           _yolo({"needle driver": 0.8}))
    assert out["tool_agreement"] == pytest.approx(1.0)
    assert out["both_present"] == ["needle driver"]


def test_cnn_only_detection_is_recorded_as_disagreement():
    out = agreement_record({"needle driver": 0.9},
                           {"needle driver": 0.5},
                           _yolo({}))
    assert out["cnn_only"] == ["needle driver"]
    assert out["tool_agreement"] < 1.0


def test_yolo_only_detection_is_recorded_as_disagreement():
    out = agreement_record({"needle driver": 0.1},
                           {"needle driver": 0.5},
                           _yolo({"needle driver": 0.8}))
    assert out["yolo_only"] == ["needle driver"]


def test_out_of_taxonomy_detection_is_not_a_disagreement():
    """The CNNs were never asked about suction irrigator."""
    out = agreement_record({"needle driver": 0.9},
                           {"needle driver": 0.5},
                           _yolo({"needle driver": 0.8,
                                  "suction irrigator": 0.9}))
    assert out["yolo_only"] == []
    assert out["tool_agreement"] == pytest.approx(1.0)


def test_neither_finds_anything_is_agreement_not_a_zero():
    out = agreement_record({"needle driver": 0.1},
                           {"needle driver": 0.5},
                           _yolo({}))
    assert out["tool_agreement"] == pytest.approx(1.0)


def test_top_disagreement_names_the_widest_gap():
    out = agreement_record({"needle driver": 0.95, "stapler": 0.55},
                           {"needle driver": 0.5, "stapler": 0.5},
                           _yolo({"stapler": 0.52}))
    assert out["top_disagreement"] == ["needle driver", "cnn_only"]


def test_no_disagreement_leaves_top_none():
    out = agreement_record({"needle driver": 0.9},
                           {"needle driver": 0.5},
                           _yolo({"needle driver": 0.8}))
    assert out["top_disagreement"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agreement.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'surgvu.agreement'`

- [ ] **Step 3: Write the implementation**

```python
"""Where the CNN heads and the detector disagree, and by how much.

WHY DISAGREEMENT IS THE USEFUL SIGNAL. Self-consistency -- sampling one model
several times and trusting it when the samples match -- is the confidence
proxy the VLM branch inherited, and it has a measured failure mode: at low
temperature the model agreed with itself 2/2 on answers that were wrong.
Agreement WITHIN one model measures determinism, not correctness.

Two models trained on different objectives from different label formats fail
differently. When they agree, the evidence is genuinely stronger; when they
diverge, something is actually hard about the frame. That is a confidence
signal a single model cannot produce at any temperature.

THE ASYMMETRY IS DELIBERATE. The detector knows `bipolar dissector` and
`suction irrigator`; the CNN heads do not have those classes at all. A
detection of one is NOT a disagreement -- the CNNs were never asked -- and
counting it as one would make every frame containing a suction irrigator look
uncertain for no reason.
"""
from .detect import OUT_OF_TAXONOMY

AGREEMENT_VERSION = 1


def agreement_record(tool_probs, tool_thresholds, yolo_record,
                     yolo_conf_floor=0.25):
    """Compare CNN presence calls against detector presence calls.

    `tool_probs` and `tool_thresholds` are name-keyed maps over the 12-class
    taxonomy. `yolo_record` is the block from detect.detections_to_record.

    `tool_agreement` is the Jaccard similarity of the two presence sets, with
    the empty-vs-empty case defined as 1.0: both models saying "nothing here"
    is agreement, not an undefined ratio. A 0.0 there would flag every quiet
    frame as maximally uncertain.
    """
    cnn = {name for name, prob in tool_probs.items()
           if prob >= tool_thresholds.get(name, 1.0)}
    yolo = {name for name, conf in (yolo_record.get("max_conf") or {}).items()
            if conf >= yolo_conf_floor and name not in OUT_OF_TAXONOMY}

    both = sorted(cnn & yolo)
    cnn_only = sorted(cnn - yolo)
    yolo_only = sorted(yolo - cnn)

    union = cnn | yolo
    agreement = 1.0 if not union else len(cnn & yolo) / len(union)

    # The single widest divergence, named so a prompt or a log can quote it
    # instead of a ratio. Ranked by how far past its own bar the lone model
    # went: a class the CNN calls at 0.95 against a 0.5 threshold is a
    # stronger disagreement than one it calls at 0.51.
    candidates = []
    for name in cnn_only:
        margin = tool_probs[name] - tool_thresholds.get(name, 0.5)
        candidates.append((margin, name, "cnn_only"))
    for name in yolo_only:
        margin = yolo_record["max_conf"][name] - yolo_conf_floor
        candidates.append((margin, name, "yolo_only"))
    top = max(candidates)[1:] if candidates else None

    return {
        "version": AGREEMENT_VERSION,
        "tool_agreement": float(agreement),
        "both_present": both,
        "cnn_only": cnn_only,
        "yolo_only": yolo_only,
        "top_disagreement": list(top) if top else None,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_agreement.py -q`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add src/surgvu/agreement.py tests/test_agreement.py
git commit -m "feat(agreement): CNN vs YOLO presence agreement as a confidence signal

Agreement within one model measures determinism, not correctness -- the
groupmate's sweep recorded 2/2 self-agreement on wrong answers. Two models
with different objectives fail differently, so their divergence is a real
uncertainty channel. Out-of-taxonomy detections are excluded: the CNN heads
were never asked about those classes, so a detection is not a disagreement."
```

---

### Task 8: Router captures the variant qualifier as a slot

`router.py:651` lists `"large"` and `"mega"` in `_BRAND_TOKEN_STOPLIST`, so **"large needle driver" collapses to "needle driver"**. The stoplist is right about what it does — those tokens must not *register a class*, or a specific question would widen into a generic one. The fix is not to remove them; it is to capture them separately as a qualifier slot. This affects 3 of 11 sample questions (27%), currently scored 1/3.

**Files:**
- Modify: `src/surgvu/router.py`
- Test: `tests/test_router_variant_slot.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `variant_qualifier(question) -> str | None`, returning `"large"`, `"mega"`, or `None`. Consumed by Task 11 and by Plan 2's arbiter.

- [ ] **Step 1: Write the failing test**

```python
"""The variant qualifier must be readable without becoming an identifier.

_BRAND_TOKEN_STOPLIST is correct about its own job: "large" must not register
a class, or "the large forceps" would widen into a generic forceps question.
But the qualifier is still the difference between two gold answers, so it has
to be captured somewhere. A separate slot, not a change to the stoplist.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.router import variant_qualifier  # noqa: E402


def test_large_is_captured():
    assert variant_qualifier("Is a large needle driver being used?") == "large"


def test_mega_is_captured():
    assert variant_qualifier("Is the mega needle driver in view?") == "mega"


def test_case_and_punctuation_are_ignored():
    assert variant_qualifier("Is a LARGE, needle driver present?") == "large"


def test_absent_qualifier_is_none():
    assert variant_qualifier("Is a needle driver being used?") is None


def test_suturecut_counts_as_large_family():
    """Large SutureCut is 624 of 1629 needle drivers -- the largest single
    commercial name in the corpus, and it is a Large-family instrument."""
    assert variant_qualifier("Is the large suturecut driver used?") == "large"


def test_a_question_naming_both_returns_none_rather_than_guessing():
    q = "Is a large or mega needle driver being used?"
    assert variant_qualifier(q) is None


def test_the_stoplist_is_unchanged():
    """The fix must not widen specific questions as a side effect."""
    from surgvu.router import _BRAND_TOKEN_STOPLIST
    assert "large" in _BRAND_TOKEN_STOPLIST
    assert "mega" in _BRAND_TOKEN_STOPLIST
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_router_variant_slot.py -q`
Expected: FAIL — `ImportError: cannot import name 'variant_qualifier'`

- [ ] **Step 3: Write the implementation**

Add to `src/surgvu/router.py`, immediately after `_BRAND_TOKEN_STOPLIST`:

```python
#: Commercial-name families for needle drivers, from config/commercial_names.json:
#: Large SutureCut 624 + Large 398 = 1022 (62.7%), Mega 318 + Mega SutureCut
#: 285 + Mega Suturecut 2 = 605 (37.1%). Two families, and every question that
#: names one is asking a question the other answer gets wrong.
_VARIANT_FAMILIES = {
    "large": ("large", "suturecut"),
    "mega": ("mega",),
}


def variant_qualifier(question):
    """The size family a question names, or None.

    WHY THIS IS SEPARATE FROM THE STOPLIST. `_BRAND_TOKEN_STOPLIST` keeps
    "large" and "mega" from REGISTERING A CLASS, and it is right to: a brand
    token that identifies nothing on its own would widen "the large forceps"
    into a generic forceps question. But the qualifier is still the whole
    difference between two gold answers -- it appears in 3 of the 11 sample
    questions -- so it is captured here as a SLOT and consumed by the variant
    head, without ever being treated as a tool identifier.

    A question naming BOTH families returns None. There is no single right
    answer to give it, and picking one would be a guess wearing the costume
    of a measurement.
    """
    text = _normalize(question or "")
    tokens = set(text.replace(",", " ").replace(".", " ").split())
    hit = [family for family, words in _VARIANT_FAMILIES.items()
           if tokens & set(words)]
    return hit[0] if len(hit) == 1 else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_router_variant_slot.py tests/test_router.py -q`
Expected: PASS. `tests/test_router.py` is the large pre-existing router suite and must not regress.

- [ ] **Step 5: Commit**

```bash
git add src/surgvu/router.py tests/test_router_variant_slot.py
git commit -m "feat(router): capture large/mega as a variant slot, stoplist unchanged

'large needle driver' collapsed to 'needle driver' because the brand-token
stoplist strips size qualifiers -- correctly, since a token that identifies
nothing must not register a class. The qualifier is nonetheless the whole
difference between two gold answers in 3 of 11 sample questions, so it is
now a separate slot. A question naming both families returns None rather
than guessing."
```

---

### Task 9: Variant labels from the logbook

`tools.csv` records `commercial_toolname` per install interval, so every frame inside an interval is automatically a Large-family or Mega-family label. No annotation, no boxes. **Case-level priors are useless here and must not be substituted:** 137 of 154 cases contain *both* families; only 16 (10.4%) are single-family.

**Files:**
- Create: `scripts/build_variant_labels.py`
- Test: `tests/test_build_variant_labels.py`

**Interfaces:**
- Consumes: `tools.csv` under `SURGVU25_train_labels/<case>/`.
- Produces: `family_of(commercial_name) -> str | None` (`"large"`, `"mega"`, or `None`), and `intervals_for_case(tools_csv) -> list[dict]` with keys `start`, `stop`, `family`, `arm`. Writes `variant_labels.json`: `{"version": 1, "cases": {case: [{"start", "stop", "family"}]}}`. Consumed by Task 10.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for logbook-derived Large/Mega labels.

These labels are free and there are a lot of them, which makes it especially
important that an unrecognised commercial name is DROPPED rather than
assigned to the larger family. A silent default would put hundreds of
mislabelled frames into training and the head would learn the prior instead
of the appearance.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_variant_labels import family_of, intervals_for_case  # noqa: E402


def test_large_suturecut_is_large_family():
    assert family_of("Large SutureCut") == "large"


def test_plain_large_is_large_family():
    assert family_of("Large") == "large"


def test_mega_suturecut_is_mega_family_regardless_of_case():
    assert family_of("Mega Suturecut") == "mega"
    assert family_of("Mega SutureCut") == "mega"


def test_unknown_name_is_dropped_not_defaulted():
    assert family_of("DeBakey Forceps") is None
    assert family_of("") is None
    assert family_of(None) is None


def test_intervals_cover_only_needle_drivers(tmp_path):
    path = tmp_path / "tools.csv"
    path.write_text(
        "install_part,install_time,uninstall_part,uninstall_time,arm,"
        "commercial_toolname,groundtruth_toolname\n"
        "1,10.0,1,20.0,1,Large SutureCut,needle driver\n"
        "1,25.0,1,35.0,2,Cadiere,cadiere forceps\n",
        encoding="utf-8")
    intervals = intervals_for_case(path)
    assert len(intervals) == 1
    assert intervals[0]["family"] == "large"
    assert intervals[0]["start"] == pytest.approx(10.0)


def test_both_families_in_one_case_are_both_kept(tmp_path):
    """137 of 154 cases contain both. A per-case prior cannot resolve this."""
    path = tmp_path / "tools.csv"
    path.write_text(
        "install_part,install_time,uninstall_part,uninstall_time,arm,"
        "commercial_toolname,groundtruth_toolname\n"
        "1,10.0,1,20.0,1,Large,needle driver\n"
        "1,30.0,1,40.0,1,Mega,needle driver\n",
        encoding="utf-8")
    assert {i["family"] for i in intervals_for_case(path)} == {"large", "mega"}


def test_unparseable_time_drops_the_row(tmp_path):
    path = tmp_path / "tools.csv"
    path.write_text(
        "install_part,install_time,uninstall_part,uninstall_time,arm,"
        "commercial_toolname,groundtruth_toolname\n"
        "1,,1,20.0,1,Large,needle driver\n",
        encoding="utf-8")
    assert intervals_for_case(path) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_build_variant_labels.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_variant_labels'`

- [ ] **Step 3: Write the implementation**

```python
"""Large-vs-Mega needle-driver labels, free, from the logbook.

WHY THIS IS THE ONLY WAY TO FIX case132. The question "is a large needle
driver being used" has a different gold answer from the same question about a
mega needle driver, and no model in the pipeline can currently tell them
apart -- the tool heads have one `needle driver` class and so does the
detector.

A PER-CASE PRIOR CANNOT SUBSTITUTE, and this was measured rather than
assumed: 137 of 154 cases contain BOTH families, and only 16 (10.4%) are
single-family. Guessing from the case is guessing. The variant has to be
resolved visually, per clip, which needs labels.

The labels already exist. tools.csv records `commercial_toolname` for every
install interval, so every frame between an install and its uninstall carries
a family label for free -- no annotation, no boxes, and at the scale of the
whole 155-case corpus.

AN UNRECOGNISED NAME IS DROPPED. Defaulting to the larger family would be
defensible on frequency (Large is 62.7%) and would be exactly wrong: it would
put mislabelled frames into training and teach the head the corpus prior
instead of the appearance, which is the failure this whole task exists to
avoid.
"""
import argparse
import csv
import json
from pathlib import Path

#: Substring tests against the lowercased commercial name. "suturecut" alone
#: implies Large: the corpus has `Large SutureCut` (624) and
#: `Mega SutureCut` (285), and the mega ones are caught by the mega rule
#: first, so the remaining SutureCut names are Large-family.
_MEGA_MARKERS = ("mega",)
_LARGE_MARKERS = ("large", "suturecut")

NEEDLE_DRIVER = "needle driver"
VARIANT_LABELS_VERSION = 1


def family_of(commercial_name):
    """"large", "mega", or None for a commercial tool name."""
    if not commercial_name:
        return None
    text = str(commercial_name).strip().lower()
    if not text:
        return None
    if any(marker in text for marker in _MEGA_MARKERS):
        return "mega"
    if any(marker in text for marker in _LARGE_MARKERS):
        return "large"
    return None


def intervals_for_case(tools_csv):
    """Needle-driver install intervals with a resolved family.

    Rows that are not needle drivers, or whose family cannot be resolved, or
    whose times will not parse, are dropped. Every drop is a row that would
    otherwise contribute mislabelled frames.
    """
    out = []
    with open(tools_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row.get("groundtruth_toolname") or "").strip().lower() \
                    != NEEDLE_DRIVER:
                continue
            family = family_of(row.get("commercial_toolname"))
            if family is None:
                continue
            try:
                start = float(row["install_time"])
                stop = float(row["uninstall_time"])
            except (KeyError, TypeError, ValueError):
                continue
            if stop <= start:
                continue
            out.append({"start": start, "stop": stop, "family": family,
                        "arm": (row.get("arm") or "").strip()})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--out", default="config/variant_labels.json")
    args = parser.parse_args(argv)

    cases = {}
    both, large_only, mega_only = 0, 0, 0
    for case_dir in sorted(Path(args.labels_root).iterdir()):
        tools_csv = case_dir / "tools.csv"
        if not tools_csv.exists():
            continue
        intervals = intervals_for_case(tools_csv)
        if not intervals:
            continue
        cases[case_dir.name] = intervals
        families = {i["family"] for i in intervals}
        if families == {"large", "mega"}:
            both += 1
        elif families == {"large"}:
            large_only += 1
        else:
            mega_only += 1

    print("cases with both families: %d   large-only: %d   mega-only: %d"
          % (both, large_only, mega_only))
    print("single-family cases: %d of %d (%.1f%%) -- a per-case prior "
          "resolves only these" % (large_only + mega_only, len(cases),
                                   100.0 * (large_only + mega_only)
                                   / max(1, len(cases))))
    out = {"version": VARIANT_LABELS_VERSION, "cases": cases}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n",
                              encoding="utf-8")
    print("wrote %s (%d cases)" % (args.out, len(cases)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_build_variant_labels.py -q`
Expected: PASS, 7 tests.

Then generate the real labels and confirm the both-families count reproduces the measured 137/154:

```bash
python3 scripts/build_variant_labels.py \
  --labels-root /staging/groups/bhaskar_opscribe/surgvu/labels_cat2/SURGVU25_train_labels \
  --out config/variant_labels.json
```

- [ ] **Step 5: Commit**

```bash
git add scripts/build_variant_labels.py tests/test_build_variant_labels.py config/variant_labels.json
git commit -m "feat(variant): free Large/Mega labels from tools.csv install intervals

commercial_toolname per install interval labels every frame in that interval
with no annotation and no boxes. An unrecognised name is dropped, never
defaulted to the 62.7% majority: a frequency default would teach the head
the corpus prior instead of the appearance. Prints the both-families count,
which is the evidence that a per-case prior cannot do this job."
```

---

### Task 10: Train and serve the variant head

**Files:**
- Create: `src/surgvu/variant.py`
- Create: `scripts/train_variant.py`
- Test: `tests/test_variant.py` (torch-free parts), `tests/test_variant_model.py` (container)

**Interfaces:**
- Consumes: `config/variant_labels.json` (Task 9); YOLO needle-driver boxes (Task 6).
- Produces: `variant_record(probs, cutoff) -> dict` with keys `version`, `family`, `p_large`, `p_mega`, `decided`; `VariantHead.predict(frames, boxes=None) -> dict`. Consumed by Task 11.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the variant head's decision layer.

The head must be allowed to ABSTAIN. A forced binary choice on an ambiguous
clip converts a 0.7015 polar answer into a coin flip between 1.0000 and
0.7015, which is only worth taking when the head is actually better than
chance on that clip -- and the cutoff is what encodes "actually better".
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.variant import variant_record  # noqa: E402


def test_confident_large_decides_large():
    out = variant_record({"large": 0.9, "mega": 0.1}, cutoff=0.65)
    assert out["family"] == "large"
    assert out["decided"] is True


def test_confident_mega_decides_mega():
    out = variant_record({"large": 0.2, "mega": 0.8}, cutoff=0.65)
    assert out["family"] == "mega"


def test_below_cutoff_abstains_with_family_none():
    out = variant_record({"large": 0.55, "mega": 0.45}, cutoff=0.65)
    assert out["family"] is None
    assert out["decided"] is False


def test_abstention_still_reports_both_probabilities():
    """Downstream may weigh a 0.55 differently from a 0.51."""
    out = variant_record({"large": 0.55, "mega": 0.45}, cutoff=0.65)
    assert out["p_large"] == pytest.approx(0.55)
    assert out["p_mega"] == pytest.approx(0.45)


def test_record_is_strict_json():
    json.loads(json.dumps(variant_record({"large": 0.5, "mega": 0.5}, 0.65),
                          allow_nan=False))


def test_rejects_probabilities_that_do_not_sum_to_one():
    with pytest.raises(ValueError):
        variant_record({"large": 0.9, "mega": 0.9}, cutoff=0.65)


def test_rejects_a_cutoff_at_or_below_chance():
    """A 0.5 cutoff never abstains, which defeats the point of having one."""
    with pytest.raises(ValueError):
        variant_record({"large": 0.9, "mega": 0.1}, cutoff=0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_variant.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'surgvu.variant'`

- [ ] **Step 3: Write the implementation**

```python
"""Large vs Mega needle driver: the one distinction nothing else can make.

WHAT IT IS WORTH. On the 11-case sample, "large needle driver" appears in 3
questions (27%) and we score 1 of 3. A polar answer is 1.0000 right and
0.7015 wrong, so each of those is 0.2985 -- and the family is the entire
difference between the two gold answers.

WHY IT ABSTAINS. Forcing a binary call on an ambiguous clip trades a certain
0.7015 for a coin flip. That trade is only worth taking when the head is
genuinely better than chance on THIS clip, and the cutoff is where that
judgement is written down. A cutoff of 0.5 never abstains, which is why this
module refuses one: it would be a decision layer that makes no decision.

THE CUTOFF IS FITTED, NOT CHOSEN. scripts/train_variant.py writes it to
config alongside the validation accuracy it achieved. A number typed into
source here would be a guess with the authority of code.
"""
VARIANT_VERSION = 1

FAMILIES = ("large", "mega")


def variant_record(probs, cutoff):
    """Turn a two-class distribution into a decision, or an abstention."""
    if not 0.5 < float(cutoff) <= 1.0:
        raise ValueError(
            "cutoff must be above 0.5 and at most 1.0, got %r. At or below "
            "chance the head never abstains, which removes the only reason "
            "this layer exists." % (cutoff,))
    p_large = float(probs.get("large", 0.0))
    p_mega = float(probs.get("mega", 0.0))
    total = p_large + p_mega
    if abs(total - 1.0) > 1e-3:
        raise ValueError(
            "large+mega must be a distribution, got %.4f. Unnormalised "
            "scores compared against a probability cutoff would abstain or "
            "decide for arithmetic reasons." % (total,))

    top = "large" if p_large >= p_mega else "mega"
    confidence = max(p_large, p_mega)
    decided = confidence >= float(cutoff)
    return {
        "version": VARIANT_VERSION,
        "family": top if decided else None,
        "p_large": p_large,
        "p_mega": p_mega,
        "cutoff": float(cutoff),
        "decided": bool(decided),
    }


class VariantHead:
    """ResNet-18 two-class head over needle-driver crops.

    Torch is imported inside the methods so `variant_record` stays importable
    without it -- that is where the decision logic lives and where the tests
    that matter run.

    CROPS WHEN AVAILABLE, WHOLE FRAME OTHERWISE. The detector's needle-driver
    box is what makes this tractable: a size distinction between two otherwise
    identical instruments is a fine-grained appearance problem, and a
    whole-frame classifier has to find the tool before it can compare it. When
    no box is available the whole frame is used rather than skipping the
    clip -- a degraded measurement beats none, and the abstention path exists
    precisely to catch the cases where that degradation matters.
    """

    def __init__(self, weights, cutoff, device="cpu", size=224):
        self.weights = str(weights)
        self.cutoff = float(cutoff)
        self.device = device
        self.size = int(size)
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        import torch
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(FAMILIES))
        state = torch.load(self.weights, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        model.eval().to(self.device)
        self._model = model
        return model

    def predict(self, frames, boxes=None):
        """Mean two-class distribution over the supplied frames."""
        import cv2
        import numpy as np
        import torch

        model = self._load()
        crops = []
        for index, frame in enumerate(np.asarray(frames)):
            box = (boxes or {}).get(index)
            if box is not None:
                x1, y1, x2, y2 = (int(round(v)) for v in box)
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                if x2 - x1 >= 8 and y2 - y1 >= 8:
                    frame = frame[y1:y2, x1:x2]
            crops.append(cv2.resize(frame, (self.size, self.size)))
        if not crops:
            raise ValueError("no frames to classify")

        batch = torch.from_numpy(
            np.stack(crops)[:, :, :, ::-1].copy()).permute(0, 3, 1, 2).float()
        batch = (batch / 255.0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1).mean(dim=0).tolist()
        return variant_record(dict(zip(FAMILIES, probs)), self.cutoff)
```

`scripts/train_variant.py` trains it: sample frames from `config/variant_labels.json` intervals via `surgvu.extract`, hold out by CASE (never by frame — frames from one interval are near-duplicates and a frame-level split would report memorisation as accuracy), train ResNet-18 for 15 epochs with Adam at 1e-4, sweep the cutoff on the held-out cases for the accuracy-vs-coverage point where accuracy first exceeds 0.75, and write `config/variant_head.json` with `{"weights", "cutoff", "val_accuracy", "val_coverage", "held_out_cases"}`.

- [ ] **Step 4: Run test to verify it passes**

Run (login node, decision logic only): `python3 -m pytest tests/test_variant.py -q` → PASS, 7 tests.
Run (container, full suite incl. torch paths): `condor_submit condor/run_tests.sub` then check the JSON output for 0 failures.

- [ ] **Step 5: Commit**

```bash
git add src/surgvu/variant.py scripts/train_variant.py tests/test_variant.py
git commit -m "feat(variant): Large/Mega needle-driver head with a fitted abstention

Worth 0.2985 per polar question and it appears in 27% of the sample. Refuses
a cutoff at or below 0.5 -- a decision layer that never abstains is not one.
The cutoff is fitted on held-out CASES and written to config with the
accuracy it achieved; splitting by frame would report memorisation, since
frames inside one install interval are near-duplicates."
```

---

### Task 11: Wire YOLO, agreement and variant into serving

**Files:**
- Modify: `scripts/inference.py`
- Test: `tests/test_inference_evidence.py`

**Interfaces:**
- Consumes: everything from Tasks 5–10.
- Produces: `--yolo` and `--variant-head` flags; the `yolo`, `agree` and `variant` blocks on the perception record. Consumed by Plan 2.

- [ ] **Step 1: Write the failing test**

```python
"""The evidence flags must be inert until they are switched on, and must
never be able to take the pipeline down when they are.

Every one of these components is new and unmeasured against BERTScore. The
container's existing contract is that a failure anywhere still writes an
answer, because a missing response scores zero while a wrong polar answer
still scores 0.7015. New evidence does not get to weaken that.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference import parse_args  # noqa: E402


def test_evidence_flags_default_off():
    args = parse_args(["--input-dir", "/in", "--output-dir", "/out"])
    assert args.yolo is False
    assert args.variant_head is False
    assert args.motion_v2 is False


def test_flags_are_independent():
    args = parse_args(["--input-dir", "/in", "--output-dir", "/out", "--yolo"])
    assert args.yolo is True
    assert args.variant_head is False


def test_variant_head_requires_yolo_or_says_why():
    """--variant-head without --yolo is legal (whole-frame fallback) and must
    not raise; the crop is an improvement, not a precondition."""
    args = parse_args(["--input-dir", "/in", "--output-dir", "/out",
                       "--variant-head"])
    assert args.variant_head is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_inference_evidence.py -q`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'yolo'`

- [ ] **Step 3: Write the implementation**

Add the flags:

```python
    parser.add_argument("--yolo", action="store_true",
                        help="run the 14-class detector as a second opinion. "
                             "Additive: its detections and its disagreement "
                             "with the CNN heads are recorded, and nothing "
                             "reads them unless a later flag does.")
    parser.add_argument("--yolo-weights",
                        default="/opt/algorithm/models/yolo_best.pt")
    parser.add_argument("--yolo-repo", default="/opt/algorithm/yolov5")
    parser.add_argument("--variant-head", action="store_true",
                        help="resolve Large vs Mega needle driver. Uses the "
                             "detector's box when --yolo is on and the whole "
                             "frame otherwise.")
    parser.add_argument("--variant-weights",
                        default="/opt/algorithm/models/variant_head.pt")
    parser.add_argument("--variant-config",
                        default="config/variant_head.json",
                        help="carries the FITTED cutoff and the validation "
                             "accuracy it achieved. Read at serving time so "
                             "the abstention point cannot drift from the "
                             "number it was measured at.")
```

and, after `perception` is built, a block that cannot take the run down:

```python
        # EVERY BLOCK HERE IS OPTIONAL AND EVERY FAILURE IS SWALLOWED. These
        # are unmeasured components on a pipeline whose one hard guarantee is
        # that it always writes an answer -- a missing response scores zero,
        # a wrong polar answer still scores 0.7015. Evidence that cannot be
        # gathered is evidence the record simply does not carry.
        yolo_record = None
        if args.yolo:
            try:
                from surgvu.detect import Detector, detections_to_record
                with timed("yolo", timings):
                    detector = Detector(args.yolo_weights, args.yolo_repo,
                                        device=resolve_devices(args.device)[0])
                    found = detector.detect(frames)
                    stamps = [i * (30.0 / max(1, len(frames)))
                              for i in range(len(frames))]
                    yolo_record = detections_to_record(found, stamps)
                perception["yolo"] = yolo_record
                log("yolo max_conf=%s" % (yolo_record["max_conf"],))
            except Exception:                   # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                log("WARNING: the detector failed; continuing without it")

        if yolo_record is not None:
            try:
                from surgvu.agreement import agreement_record
                thresholds = dict(zip(config["experts"]["tools"]["classes"],
                                      config["experts"]["tools"]["thresholds"]))
                perception["agree"] = agreement_record(
                    perception["tools"], thresholds, yolo_record)
                log("agreement=%.3f top_disagreement=%s"
                    % (perception["agree"]["tool_agreement"],
                       perception["agree"]["top_disagreement"]))
            except Exception:                   # noqa: BLE001
                traceback.print_exc(file=sys.stderr)

        if args.variant_head:
            try:
                from surgvu.variant import VariantHead
                head_config = json.loads(
                    Path(args.variant_config).read_text(encoding="utf-8"))
                boxes = {}
                if yolo_record is not None:
                    for entry in yolo_record["by_class"].get("needle driver", []):
                        boxes[entry["anchor_idx"]] = entry["box"]
                with timed("variant", timings):
                    head = VariantHead(args.variant_weights,
                                       head_config["cutoff"],
                                       device=resolve_devices(args.device)[0])
                    perception["variant"] = head.predict(frames, boxes or None)
                log("variant family=%s p_large=%.3f decided=%s"
                    % (perception["variant"]["family"],
                       perception["variant"]["p_large"],
                       perception["variant"]["decided"]))
            except Exception:                   # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                log("WARNING: the variant head failed; continuing without it")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_inference_evidence.py tests/test_inference.py -q`
Expected: PASS. `tests/test_inference.py` holds the existing serving contract.

- [ ] **Step 5: Commit**

```bash
git add scripts/inference.py tests/test_inference_evidence.py
git commit -m "feat(inference): --yolo and --variant-head, every failure swallowed

Three optional evidence blocks on the perception record. Each is wrapped so
a failure logs and continues: the container's one hard guarantee is that it
always writes an answer, and unmeasured components do not get to weaken it.
The variant head uses the detector's box when available and the whole frame
otherwise -- the crop is an improvement, not a precondition."
```

---

### Task 12: The flag matrix

The attribution instrument. v4 already shipped two changes at once and cannot be read; v5 ships six. This is what makes the eventual leaderboard delta interpretable, and it has to exist **before** the submission, not after.

**Files:**
- Create: `scripts/flag_matrix.py`
- Test: `tests/test_flag_matrix.py`

**Interfaces:**
- Consumes: `scripts/score_sample.py`'s scoring entry point.
- Produces: `combinations(flags) -> list[tuple]`, and a written `baselines/flag_matrix.json` of `{"<flag combination>": {"mean": float, "per_case": {...}}}`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the flag-combination enumerator.

The baseline (no flags) MUST be in the matrix. A matrix of only the enabled
combinations measures them against each other and not against what ships,
which is the exact mistake that made v4 unreadable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from flag_matrix import combinations  # noqa: E402


def test_baseline_is_present():
    assert () in combinations(["--yolo", "--variant-head"])


def test_every_single_flag_is_present():
    combos = combinations(["--yolo", "--variant-head"])
    assert ("--yolo",) in combos
    assert ("--variant-head",) in combos


def test_the_full_set_is_present():
    combos = combinations(["--yolo", "--variant-head"])
    assert ("--yolo", "--variant-head") in combos


def test_count_is_two_to_the_n():
    assert len(combinations(["--a", "--b", "--c"])) == 8


def test_ordering_is_stable_so_keys_are_comparable_across_runs():
    flags = ["--yolo", "--variant-head", "--motion-v2"]
    assert combinations(flags) == combinations(flags)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_flag_matrix.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'flag_matrix'`

- [ ] **Step 3: Write the implementation**

```python
"""Score every flag combination on the 11-case sample.

WHY THIS EXISTS AND WHY IT RUNS BEFORE THE SUBMISSION, NOT AFTER. v4 carried
both the Aug-13 router batch and the motion gate relative to the last scored
submission, so whatever v4 scores, the cause is unresolved -- a move up
cannot be credited to the motion gate and a move down cannot be blamed on it.
v5 ships six workstreams at once, by explicit decision. The flags are how
attribution is recovered, and they only work if the matrix was recorded.

WHAT THIS IS NOT. Eleven cases is a small sample and the user's instruction
is not to over-weight it. This is a TRIPWIRE, not a gate: it does not decide
whether anything ships. It exists so that when the leaderboard moves, there
is something to read the move against.
"""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def combinations(flags):
    """Every subset of `flags`, baseline first, in a stable order.

    The empty combination is included deliberately. A matrix that only
    compares enabled combinations to each other never measures any of them
    against what actually ships today.
    """
    ordered = list(flags)
    out = []
    for size in range(len(ordered) + 1):
        out.extend(itertools.combinations(ordered, size))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--flags", nargs="+",
                        default=["--motion-v2", "--yolo", "--variant-head"])
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--out", default="baselines/flag_matrix.json")
    args = parser.parse_args(argv)

    results = {}
    for combo in combinations(args.flags):
        key = " ".join(combo) or "(baseline)"
        cmd = [sys.executable, "scripts/score_sample.py",
               "--sample-dir", args.sample_dir, "--json"] + list(combo)
        print("=== %s" % (key,))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr[-2000:], file=sys.stderr)
            results[key] = {"error": proc.returncode}
            continue
        scored = json.loads(proc.stdout)
        results[key] = {"mean": scored["mean"],
                        "per_case": scored["per_case"]}
        print("    mean %.4f" % (scored["mean"],))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n",
                              encoding="utf-8")
    print("wrote %s (%d combinations)" % (args.out, len(results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_flag_matrix.py -q`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/flag_matrix.py tests/test_flag_matrix.py
git commit -m "feat(eval): flag-combination matrix as the attribution instrument

v4 shipped two changes and is unreadable; v5 ships six by decision. This
records every subset's score on the sample, baseline included, so the
leaderboard delta has something to be read against. A tripwire, not a gate --
it does not decide what ships."
```

---

## Self-Review

**Spec coverage.** W1 → Tasks 1–5. W2 → Tasks 6, 7, 9, 10, 11, plus the router slot in Task 8. W3 → Task 5 (`clip_record`'s additive block loop, which is the evidence packet in its existing form). W4, W5 → Plan 2, not this plan. W6 → Plan 3, not this plan. Validation item 1 (flag matrix) → Task 12. Validation items 2–4 (timing budget, T4 validation, YOLO re-validation on our own cases) are **not covered by any task in this plan** and belong in Plan 2, where the VLM makes the timing question binding; recorded here so the gap is visible rather than discovered.

**Corrections against the spec.** Two spec statements are wrong and this plan uses the corrected values: the shipped burst offset is **±67 ms** (`perceive.py:104`, `BURST_FPS = 15.0`), not ±0.67 s; and the evidence packet is not a new type but an extension of `perceive.clip_record()`, which already has the additivity property W3 wanted. The spec should be amended to match.

**Type consistency.** `motion_vector` returns the eight keys `motion_record_v2` summarises and `calibrate_motion_v2.py` sweeps — checked against `_VECTOR_KEYS` and the `slots` tuple. `detections_to_record` produces `max_conf`, which `agreement_record` reads. `variant_record`'s `cutoff` is written by `train_variant.py` into `config/variant_head.json` and read by `VariantHead`. `map_to_taxonomy` and `OUT_OF_TAXONOMY` are defined in `detect.py` and imported by `agreement.py`.

**Placeholder scan.** No TBDs, no "add error handling", no "similar to Task N". Every code step carries the code. One gap found and fixed inline: Task 11 referenced `args.variant_config` without adding the flag; the flag is now in Task 11's own flag block.
