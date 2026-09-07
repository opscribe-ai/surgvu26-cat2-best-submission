"""One 30-second clip -> the JSON record the question router consumes.

Everything here guards the seam between the two halves of inference: the
router never sees a video, only this record, so its shape, its class
ordering, and the thresholds that decide `tools_present` are a contract and
not an implementation detail.
"""
import json

import cv2
import numpy as np
import pytest
import torch

from surgvu.perceive import (
    clip_record, decode_clip, find_clips, load_expert, perceive_clip,
    sample_frame_indices, tools_present,
)
from surgvu.taxonomy import TASK_CLASSES, TOOL_CLASSES
from surgvu.train import save_checkpoint


# ---------------------------------------------------------------- fixtures

def _write_video(path, frames, fps):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(fps), (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def _ramp(n=30, height=64, width=64, step=7):
    """Frames whose brightness climbs with index, so a decoded frame's
    position in the clip is readable off its mean pixel value.

    `step` is explicit because the default only fits a SHORT clip: at n=120,
    20 + 7*119 is 853 and numpy 2 raises OverflowError rather than wrapping.
    Callers wanting a long ramp pass a smaller step, or use `_long_ramp`."""
    top = 20 + step * (n - 1)
    if top > 255:
        raise ValueError(
            "a %d-frame ramp at step %d reaches %d, past uint8. Pass a "
            "smaller step -- silently wrapping would make frame 40 as bright "
            "as frame 4 and every ordering assertion meaningless."
            % (n, step, top))
    return [np.full((height, width, 3), 20 + step * i, dtype=np.uint8)
            for i in range(n)]


def _long_ramp(n, height=64, width=64):
    """The longest ramp that fits in uint8 for `n` frames, still monotonic."""
    step = max(1, (235 - 20) // max(1, n - 1))
    return _ramp(n=n, height=height, width=width, step=step)


@pytest.fixture
def ramp_video(tmp_path):
    return _write_video(tmp_path / "clip.mp4", _ramp(), fps=60.0)


def _meta(classes, image_size=8, thresholds=None, backbone="efficientnet_v2_s"):
    meta = {"classes": list(classes), "image_size": image_size,
            "backbone": backbone}
    if thresholds is not None:
        meta["thresholds"] = list(thresholds)
    return meta


TOOL_META = _meta(TOOL_CLASSES, thresholds=[0.5] * len(TOOL_CLASSES))
TASK_META = _meta(TASK_CLASSES)


class ConstantLogits(torch.nn.Module):
    """Fixed logits for every frame, recording the spatial size it was fed.

    Parameter-free on purpose, matching tests/test_predict.py: a stub with
    parameters would make `requires_grad` true and blow up in `.numpy()`,
    which kills mutations by accident instead of by assertion.
    """

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor([logits], dtype=torch.float32))
        self.seen_size = None

    def forward(self, x):
        self.seen_size = tuple(x.shape[2:])
        return self.logits.expand(x.shape[0], -1)


# ------------------------------------------------------- frame index choice

def test_sample_frame_indices_spans_the_whole_clip():
    """A 30-second clip is one graded unit and the tool set can change what is
    VISIBLE within it, so the sample has to cover the clip end to end. Taking
    the first N frames would answer from the first half-second."""
    indices = sample_frame_indices(1800, 16)

    assert len(indices) == 16
    assert indices == sorted(indices)
    assert len(set(indices)) == 16
    assert all(0 <= i < 1800 for i in indices)
    assert indices[0] < 1800 // 16                 # starts near the beginning
    assert indices[-1] > 1800 * 15 // 16           # reaches the end


def test_sample_frame_indices_are_evenly_spaced():
    """Even spacing, not clustering: a sampler that put 15 of 16 frames in the
    first second would still pass a span check by including one late frame."""
    indices = sample_frame_indices(1800, 16)
    gaps = np.diff(indices)

    assert gaps.min() >= 1800 // 16 - 1
    assert gaps.max() <= 1800 // 16 + 1


def test_sample_frame_indices_avoids_the_very_first_and_last_frame():
    """Bin centres, not endpoints. Frame 0 and frame N-1 are where fades,
    black frames and truncated final packets live, and at 16 samples the two
    of them would be an eighth of the evidence."""
    indices = sample_frame_indices(1800, 16)

    assert indices[0] > 0
    assert indices[-1] < 1799


def test_sample_frame_indices_returns_every_frame_of_a_short_clip():
    """Asking for more frames than exist yields each frame once, never a
    duplicate: a repeated frame would be double-counted by the mean over
    frames and quietly weight one moment twice."""
    assert sample_frame_indices(5, 16) == [0, 1, 2, 3, 4]


def test_sample_frame_indices_rejects_an_empty_clip():
    with pytest.raises(ValueError, match="no frames"):
        sample_frame_indices(0, 16)


def test_sample_frame_indices_rejects_asking_for_no_frames():
    with pytest.raises(ValueError, match="at least one frame"):
        sample_frame_indices(1800, 0)


# --------------------------------------------------------------- decoding

def test_decode_clip_returns_the_requested_number_of_prepared_frames(ramp_video):
    frames = decode_clip(ramp_video, n_frames=6, size=32)

    assert frames.shape == (6, 32, 32, 3)
    assert frames.dtype == np.uint8


def test_decode_clip_covers_the_clip_end_to_end(ramp_video):
    """The fixture's brightness climbs with frame index, so a decoder that
    read the first six frames would return six nearly identical means."""
    frames = decode_clip(ramp_video, n_frames=6, size=32)
    means = [float(f.mean()) for f in frames]

    assert means == sorted(means)
    assert means[-1] - means[0] > 100        # ramp spans 20 -> 223


def test_decode_clip_ignores_the_declared_frame_rate(tmp_path):
    """THE fps decision, pinned.

    The training corpus is 60 fps and the Cat 1 test clips are 1 fps, so any
    decoder that converts a wall-clock offset into a frame index has to be
    told which one it is holding -- and is silently wrong, not broken, when
    told the wrong thing. Sampling by frame index over the whole file removes
    the question: the same 30 pictures produce the same sample whether the
    container claims 1 fps or 60.
    """
    pictures = _ramp()
    slow = _write_video(tmp_path / "slow.mp4", pictures, fps=1.0)
    fast = _write_video(tmp_path / "fast.mp4", pictures, fps=60.0)

    assert cv2.VideoCapture(str(slow)).get(cv2.CAP_PROP_FPS) == 1.0
    assert cv2.VideoCapture(str(fast)).get(cv2.CAP_PROP_FPS) == 60.0
    np.testing.assert_array_equal(decode_clip(slow, n_frames=8, size=32),
                                  decode_clip(fast, n_frames=8, size=32))


def test_decode_clip_returns_every_frame_when_asked_for_more_than_exist(ramp_video):
    frames = decode_clip(ramp_video, n_frames=500, size=32)
    assert frames.shape[0] == 30


def test_decode_clip_blurs_the_ui_band(tmp_path):
    """Compliance, not tuning: the challenge forbids predicting from the UI
    overlay, so no frame may reach a model without going through
    prepare_frame. The whole fixture frame is striped, top to bottom; only
    the bottom band may come back smoothed, which pins that the blur happened
    AND that it did not eat the image."""
    striped = np.zeros((64, 64, 3), dtype=np.uint8)
    for x in range(0, 64, 16):
        striped[:, x:x + 8] = 220            # 8-px stripes, survive mp4v
    video = _write_video(tmp_path / "ui.mp4", [striped] * 10, fps=60.0)

    frames = decode_clip(video, n_frames=4, size=64)

    # 64 * UI_BAND_FRACTION (0.08) = 5 rows, so row 63 is inside the band and
    # row 10 is nowhere near it.
    contrast = lambda row: int(np.abs(np.diff(row[:, 0].astype(np.int32))).max())
    assert contrast(frames[0][10]) > 150     # picture survives outside the band
    assert contrast(frames[0][63]) < 40      # UI band is destroyed


def test_decode_clip_rejects_a_file_it_cannot_read(tmp_path):
    """A missing or unreadable clip must stop the run, not contribute an empty
    record that the router would answer from."""
    broken = tmp_path / "nope.mp4"
    broken.write_bytes(b"not a video")

    with pytest.raises(ValueError, match="no frames"):
        decode_clip(broken, n_frames=4, size=32)


# ---------------------------------------------------------- thresholding

def test_tools_present_uses_the_per_class_threshold_not_a_half():
    """The tuned thresholds range from 0.05 to 0.95 in the real checkpoint, so
    a hardcoded 0.5 is wrong in both directions: it drops force bipolar at
    0.10 (threshold 0.05) and admits bipolar forceps at 0.60 (threshold
    0.77). Macro-F1 was tuned on those numbers; ignoring them throws the
    tuning away while still producing a plausible-looking list."""
    probs = [0.0] * 12
    probs[TOOL_CLASSES.index("bipolar forceps")] = 0.60      # under its 0.77
    probs[TOOL_CLASSES.index("force bipolar")] = 0.10        # over its 0.05
    thresholds = [0.5] * 12
    thresholds[TOOL_CLASSES.index("bipolar forceps")] = 0.77
    thresholds[TOOL_CLASSES.index("force bipolar")] = 0.05

    assert tools_present(probs, thresholds, TOOL_CLASSES) == ["force bipolar"]


def test_tools_present_includes_a_probability_exactly_on_the_threshold():
    """`>=`, matching how train_tools.py tuned them (`probs >= thresholds`).
    A strict `>` would score the checkpoint differently than it was selected."""
    probs = [0.0] * 12
    probs[TOOL_CLASSES.index("stapler")] = 0.66
    thresholds = [0.5] * 12
    thresholds[TOOL_CLASSES.index("stapler")] = 0.66

    assert tools_present(probs, thresholds, TOOL_CLASSES) == ["stapler"]


def test_tools_present_is_sorted():
    probs = [0.9] * 12
    assert tools_present(probs, [0.5] * 12, TOOL_CLASSES) == sorted(TOOL_CLASSES)


def test_tools_present_can_be_empty():
    """No tool clearing its threshold is a real answer -- the router has to be
    able to say "none" rather than receive a spurious most-likely tool."""
    assert tools_present([0.01] * 12, [0.5] * 12, TOOL_CLASSES) == []


def test_tools_present_rejects_a_threshold_list_of_the_wrong_length():
    """Thresholds are positional against `classes`. A short list would zip
    silently, thresholding the first few classes and dropping the rest."""
    with pytest.raises(ValueError, match="threshold"):
        tools_present([0.9] * 12, [0.5] * 8, TOOL_CLASSES)


# ------------------------------------------------------------- the record

def _record(tool_probs=None, task_probs=None, n_frames=16):
    tool_probs = np.full(12, 0.1) if tool_probs is None else np.asarray(tool_probs)
    task_probs = (np.full(8, 0.125) if task_probs is None
                  else np.asarray(task_probs))
    return clip_record(tool_probs, TOOL_META, task_probs, TASK_META, n_frames)


def test_clip_record_matches_the_published_contract():
    """The router is being written against exactly these five keys."""
    record = _record(n_frames=16)

    assert set(record) == {"tools", "tools_present", "task", "task_top", "n_frames"}
    assert list(record["tools"]) == list(TOOL_CLASSES)
    assert list(record["task"]) == list(TASK_CLASSES)
    assert record["n_frames"] == 16


def test_clip_record_is_json_serialisable_with_plain_floats():
    """numpy float32 is not JSON-serialisable, and the record is written to a
    file the router reads. Failing here at write time would lose the whole
    run's inference after paying for it."""
    record = _record(tool_probs=np.full(12, 0.3, dtype=np.float32),
                     task_probs=np.full(8, 0.125, dtype=np.float32))

    reloaded = json.loads(json.dumps(record))
    assert all(isinstance(v, float) for v in reloaded["tools"].values())
    assert isinstance(reloaded["n_frames"], int)


def test_clip_record_maps_probabilities_to_the_right_class_names():
    """Positional alignment between the head's outputs and the class names.
    An off-by-one here reports every tool as its neighbour and nothing about
    the numbers would look wrong."""
    tool_probs = np.zeros(12)
    tool_probs[TOOL_CLASSES.index("needle driver")] = 0.97
    task_probs = np.zeros(8)
    task_probs[TASK_CLASSES.index("uterine horn")] = 0.9

    record = _record(tool_probs, task_probs)

    assert record["tools"]["needle driver"] == pytest.approx(0.97)
    assert record["tools"]["bipolar forceps"] == pytest.approx(0.0)
    assert record["task"]["uterine horn"] == pytest.approx(0.9)


def test_clip_record_task_top_is_the_argmax():
    task_probs = np.full(8, 0.01)
    task_probs[TASK_CLASSES.index("suturing")] = 0.5
    assert _record(task_probs=task_probs)["task_top"] == "suturing"


def test_clip_record_tools_present_applies_the_checkpoint_thresholds():
    """The thresholds come off the checkpoint meta, not from a constant in
    this file -- they are what the tuned macro-F1 was measured at."""
    meta = _meta(TOOL_CLASSES, thresholds=[0.9] * 12)
    meta["thresholds"][TOOL_CLASSES.index("clip applier")] = 0.12
    probs = np.full(12, 0.2)

    record = clip_record(probs, meta, np.full(8, 0.125), TASK_META, 16)

    assert record["tools_present"] == ["clip applier"]


def test_clip_record_rejects_a_checkpoint_whose_classes_are_not_the_taxonomy():
    """The router indexes this record by taxonomy name. A checkpoint trained
    on a different ordering would produce a record whose keys are right and
    whose values belong to other classes."""
    shuffled = _meta(list(reversed(TOOL_CLASSES)), thresholds=[0.5] * 12)

    with pytest.raises(ValueError, match="classes"):
        clip_record(np.full(12, 0.2), shuffled, np.full(8, 0.125), TASK_META, 16)


def test_clip_record_rejects_the_two_checkpoints_swapped():
    """The plausible operator error: the task checkpoint passed where the tool
    checkpoint belongs. Both are EfficientNetV2-S files in one directory with
    names differing by four characters."""
    with pytest.raises(ValueError, match="classes"):
        clip_record(np.full(8, 0.125), TASK_META, np.full(12, 0.2), TOOL_META, 16)


def test_clip_record_rejects_a_tool_checkpoint_without_thresholds():
    """The task checkpoint has no `thresholds` key because it is multi-class.
    Defaulting to 0.5 when the key is missing would let a mis-wired run
    produce a full, plausible `tools_present` from an untuned cutoff."""
    with pytest.raises(KeyError, match="thresholds"):
        clip_record(np.full(12, 0.2), _meta(TOOL_CLASSES),
                    np.full(8, 0.125), TASK_META, 16)


def test_clip_record_rejects_a_probability_vector_of_the_wrong_width():
    with pytest.raises(ValueError, match="12"):
        clip_record(np.full(11, 0.2), TOOL_META, np.full(8, 0.125), TASK_META, 16)


# ----------------------------------------------------- the two-head serve

def test_perceive_clip_serves_each_head_with_its_own_activation():
    """Multi-label tools, multi-class task. Sigmoid on the task head gives
    eight numbers that do not sum to one; softmax on the tool head makes two
    simultaneously-installed tools split one unit of mass and each read 0.5.
    Both produce well-formed output."""
    frames = np.zeros((3, 16, 16, 3), dtype=np.uint8)
    tool_model = ConstantLogits([2.0] * 12)
    task_model = ConstantLogits([2.0] * 8)

    record = perceive_clip(frames, tool_model, TOOL_META,
                           task_model, TASK_META, device="cpu")

    assert record["tools"]["cadiere forceps"] == pytest.approx(0.8807971, rel=1e-5)
    assert sum(record["tools"].values()) > 10          # NOT a distribution
    assert sum(record["task"].values()) == pytest.approx(1.0, rel=1e-6)


def test_perceive_clip_serves_each_head_at_its_own_recorded_resolution():
    """image_size comes from each checkpoint's own meta. Both experts happen
    to be trained at 384 today, so a hardcoded 384 would pass every run and
    break silently the day one is retrained."""
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    tool_model = ConstantLogits([0.0] * 12)
    task_model = ConstantLogits([0.0] * 8)

    perceive_clip(frames, tool_model, _meta(TOOL_CLASSES, image_size=16,
                                            thresholds=[0.5] * 12),
                  task_model, _meta(TASK_CLASSES, image_size=24), device="cpu")

    assert tool_model.seen_size == (16, 16)
    assert task_model.seen_size == (24, 24)


def test_perceive_clip_records_the_frames_it_actually_used():
    """n_frames is the count that was aggregated, not the count requested. A
    clip that came back short is a weaker prediction and the record has to
    say so."""
    frames = np.zeros((5, 16, 16, 3), dtype=np.uint8)

    record = perceive_clip(frames, ConstantLogits([0.0] * 12), TOOL_META,
                           ConstantLogits([0.0] * 8), TASK_META, device="cpu")

    assert record["n_frames"] == 5


# ------------------------------------------------------------- discovery

def test_find_clips_reads_the_sample_layout(tmp_path):
    """Ordered by case id, not by path. The enclosing directories here sort
    in the opposite order to the case ids on purpose: the records are written
    to one JSON object and read back by case, so "whatever order the
    filesystem walked" is not a guarantee anyone should depend on."""
    for parent, case in (("z_batch", "case122"), ("m_batch", "case124"),
                         ("a_batch", "case131")):
        directory = tmp_path / parent / case
        directory.mkdir(parents=True)
        (directory / ("%s.mp4" % case)).write_bytes(b"")
        (directory / ("%s.json" % case)).write_bytes(b"[]")

    found = find_clips(tmp_path)

    assert [case for case, _ in found] == ["case122", "case124", "case131"]
    assert found[0][1].name == "case122.mp4"


def test_find_clips_rejects_a_directory_with_no_clips(tmp_path):
    with pytest.raises(ValueError, match="no .mp4"):
        find_clips(tmp_path)


# ------------------------------------------------------- checkpoint load

def test_load_expert_sizes_the_head_from_the_checkpoint(tmp_path):
    """Head width comes from meta["classes"], not from a constant: the two
    experts are 12-way and 8-way and go through this one loader."""
    from surgvu.models import build_model

    path = tmp_path / "task.pt"
    save_checkpoint(path, build_model(len(TASK_CLASSES), pretrained=False),
                    _meta(TASK_CLASSES, image_size=384))

    model, meta = load_expert(path)

    assert meta["classes"] == list(TASK_CLASSES)
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 64, 64)).shape == (1, len(TASK_CLASSES))


@pytest.mark.slow
def test_perceive_a_real_sample_clip_end_to_end():
    """The whole path on a real 30-second clip and the real checkpoints."""
    from pathlib import Path

    clip = Path("/staging/groups/bhaskar_opscribe/surgvu/cat2_sample/"
                "case124/case124.mp4")
    tools_ckpt = Path("/staging/n/nkalthoff/surgvu26/models/"
                      "tools_efficientnet_v2_s.pt")
    task_ckpt = Path("/staging/n/nkalthoff/surgvu26/models/"
                     "task_efficientnet_v2_s.pt")
    for path in (clip, tools_ckpt, task_ckpt):
        if not path.exists():
            pytest.skip("%s is not available here" % path)

    tool_model, tool_meta = load_expert(tools_ckpt)
    task_model, task_meta = load_expert(task_ckpt)
    frames = decode_clip(clip, n_frames=4)
    record = perceive_clip(frames, tool_model, tool_meta,
                           task_model, task_meta, device="cpu")

    assert record["n_frames"] == 4
    assert set(record) == {"tools", "tools_present", "task", "task_top", "n_frames"}
    assert sum(record["task"].values()) == pytest.approx(1.0, rel=1e-5)
    assert json.loads(json.dumps(record))


# ------------------------------------------------- burst decoding for motion
#
# The contract that makes motion safe to add to a shipped serving path: the
# CENTRES must be byte-identical to what decode_clip returns, so the
# appearance model's input -- and therefore every shipped answer -- cannot
# move because motion was computed alongside it.

def test_burst_centres_are_byte_identical_to_decode_clip(ramp_video):
    """THE safety property. Asserted, not trusted."""
    from surgvu.perceive import decode_clip, decode_clip_bursts

    plain = decode_clip(ramp_video, n_frames=8)
    centres, bursts = decode_clip_bursts(ramp_video, n_frames=8)
    assert centres.shape == plain.shape
    assert np.array_equal(centres, plain), (
        "the appearance model would see different pixels than it does today")
    assert len(bursts) == len(centres)


def test_a_burst_brackets_its_centre_in_time(tmp_path):
    """t-67ms, t, t+67ms -- in order, with the centre in the middle.

    The ramp fixture's brightness climbs with frame index, so the ORDER of a
    burst is readable off its pixel means: a burst that came back reversed, or
    centred on the wrong frame, shows up here rather than as a motion
    statistic that is quietly measuring the wrong interval.
    """
    from surgvu.perceive import decode_clip_bursts

    path = _write_video(tmp_path / "ramp.mp4", _long_ramp(60), fps=60.0)
    centres, bursts = decode_clip_bursts(path, n_frames=4, per_burst=3)
    measured = [b for b in bursts if b is not None]
    assert measured, "every burst was unmeasurable; the offsets are wrong"
    for burst, centre in zip(bursts, centres):
        if burst is None:
            continue
        means = [float(f.mean()) for f in burst]
        assert means[0] < means[1] < means[2], (
            "burst is not in time order: %s" % means)
        assert np.array_equal(burst[1], centre), (
            "the middle frame of the burst is not the sampled moment")


def test_the_burst_offset_follows_the_clips_own_frame_rate(tmp_path):
    """67 ms is a duration, not a frame count.

    At 60 fps that is 4 frames and at 30 fps it is 2. A hardcoded offset would
    measure a different real interval on every differently-encoded video, and
    the statistic would not be comparable across cases -- which is the one
    property it must have, since a single threshold is fitted against it.
    """
    from surgvu.perceive import decode_clip_bursts

    fast = _write_video(tmp_path / "fast.mp4", _long_ramp(60), fps=60.0)
    slow = _write_video(tmp_path / "slow.mp4", _long_ramp(60), fps=30.0)
    _, fast_bursts = decode_clip_bursts(fast, n_frames=4, per_burst=3)
    _, slow_bursts = decode_clip_bursts(slow, n_frames=4, per_burst=3)

    def span(bursts):
        for burst in bursts:
            if burst is not None:
                return float(burst[2].mean()) - float(burst[0].mean())
        return None

    fast_span, slow_span = span(fast_bursts), span(slow_bursts)
    assert fast_span is not None and slow_span is not None
    # The ramp climbs 7 grey levels per FRAME, so a wider frame offset shows
    # up as a larger brightness span. 60fps -> 8 frames apart, 30fps -> 4.
    assert fast_span > slow_span * 1.5, (
        "60fps span %.1f vs 30fps span %.1f: the offset is not tracking fps"
        % (fast_span, slow_span))


def test_an_unmeasurable_burst_is_none_and_never_zeros(ramp_video):
    """At the clip edges there is no t-67ms, and clamping would lie.

    Differencing a frame against itself reads as perfect stillness -- a
    plausible value that is not a measurement. None says so.
    """
    from surgvu.perceive import decode_clip_bursts

    centres, bursts = decode_clip_bursts(ramp_video, n_frames=16, per_burst=3)
    assert len(centres) == len(bursts)
    for burst in bursts:
        assert burst is None or burst.shape[0] == 3


def test_a_failed_flank_read_never_costs_the_centre(ramp_video, monkeypatch):
    """The appearance path must be exactly as robust as it is today.

    Motion is the thing that degrades when a read fails, never the evidence
    that ships.

    DELEGATION, NOT SUBCLASSING. The first version of this test subclassed
    cv2.VideoCapture and called super().read(), and pytest took a SIGSEGV at
    69% of the suite -- cv2's bindings are a C extension whose types are not
    built to be subclassed and re-entered that way. Worse, run_tests.py
    reported that crash as "0 failures, 0 real" and exited 0, because every
    count it produces is parsed from lines a crashed pytest never prints.
    Both were fixed; this wrapper holds a real capture rather than inheriting
    from one, which stays entirely in Python.
    """
    from surgvu.perceive import decode_clip, decode_clip_bursts

    plain = decode_clip(ramp_video, n_frames=8)

    real_capture = cv2.VideoCapture
    state = {"reads": 0}

    class FlakyCapture:
        """Forwards everything to a real capture, but drops every third read.

        In this decoder the centre of each sample is read FIRST, so the reads
        this refuses are overwhelmingly flanks -- which is the case under
        test. The assertion below does not depend on that being exact: it
        requires the centres to survive whatever was dropped.
        """

        def __init__(self, *args, **kwargs):
            self._inner = real_capture(*args, **kwargs)

        def read(self):
            state["reads"] += 1
            if state["reads"] % 3 == 0:
                return False, None
            return self._inner.read()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(cv2, "VideoCapture", FlakyCapture)
    centres, bursts = decode_clip_bursts(ramp_video, n_frames=8)

    assert np.array_equal(centres, plain), (
        "a flank read failure cost the appearance path a frame")
    assert any(b is None for b in bursts), "expected some bursts to be lost"


def test_a_record_without_motion_is_byte_identical_to_the_old_one():
    """The additive guarantee, at the level the file is actually written.

    Not "the same keys" -- the same BYTES. The perception JSON is compared
    against shipped copies and consumed by a router whose behaviour is pinned
    to it, so a reordered key or a stray field is a diff someone has to
    explain. Omitting `motion` must produce exactly the document it produced
    before motion existed.
    """
    from surgvu.perceive import clip_record

    tool_meta = _meta(TOOL_CLASSES, thresholds=[0.5] * len(TOOL_CLASSES))
    task_meta = _meta(TASK_CLASSES)
    probs_t = np.linspace(0.1, 0.9, len(TOOL_CLASSES)).astype(np.float32)
    probs_k = np.linspace(0.1, 0.9, len(TASK_CLASSES)).astype(np.float32)

    plain = clip_record(probs_t, tool_meta, probs_k, task_meta, 16)
    explicit_none = clip_record(probs_t, tool_meta, probs_k, task_meta, 16,
                                motion=None)
    assert json.dumps(plain) == json.dumps(explicit_none)
    assert "motion" not in plain
    assert list(plain) == ["tools", "tools_present", "task", "task_top",
                           "n_frames"], (
        "key ORDER changed, which is a diff in every shipped perception file")


def test_motion_is_appended_and_disturbs_nothing_else():
    from surgvu.perceive import clip_record

    tool_meta = _meta(TOOL_CLASSES, thresholds=[0.5] * len(TOOL_CLASSES))
    task_meta = _meta(TASK_CLASSES)
    probs_t = np.linspace(0.1, 0.9, len(TOOL_CLASSES)).astype(np.float32)
    probs_k = np.linspace(0.1, 0.9, len(TASK_CLASSES)).astype(np.float32)

    block = {"version": 1, "bursts": 16, "bursts_measured": 16,
             "micro": {"per_burst": [1.0] * 16, "mean": 1.0, "max": 1.0,
                       "std": 0.0},
             "macro": {"per_gap": [2.0] * 15, "mean": 2.0, "max": 2.0,
                       "std": 0.0}}
    plain = clip_record(probs_t, tool_meta, probs_k, task_meta, 16)
    withm = clip_record(probs_t, tool_meta, probs_k, task_meta, 16,
                        motion=block)

    assert withm["motion"] == block
    assert {k: v for k, v in withm.items() if k != "motion"} == plain
    json.dumps(withm)          # must stay JSON-native


def _record_pair(activity):
    """(without motion, with motion at `activity`) -- otherwise identical."""
    from surgvu.perceive import clip_record

    tool_meta = _meta(TOOL_CLASSES, thresholds=[0.5] * len(TOOL_CLASSES))
    task_meta = _meta(TASK_CLASSES)
    probs_t = np.linspace(0.1, 0.9, len(TOOL_CLASSES)).astype(np.float32)
    probs_k = np.linspace(0.1, 0.9, len(TASK_CLASSES)).astype(np.float32)
    block = {"version": 1, "bursts": 16, "bursts_measured": 16,
             "micro": {"per_burst": [activity] * 16, "mean": activity,
                       "max": activity, "std": 0.0},
             "macro": {"per_gap": [activity] * 15, "mean": activity,
                       "max": activity, "std": 0.0}}
    return (clip_record(probs_t, tool_meta, probs_k, task_meta, 16),
            clip_record(probs_t, tool_meta, probs_k, task_meta, 16,
                        motion=block))


#: Every question in this file's routing set EXCEPT the cutting one. The gate
#: is wired into `_answer_cutting` alone, so these must be untouched by motion
#: at any activity -- that is the blast radius, asserted rather than assumed.
_UNGATED_QUESTIONS = ("Is suturing being performed?",
                      "What is happening in this clip?",
                      "What tool is being used?")


def test_motion_changes_nothing_outside_the_cutting_question():
    """The blast radius of the gate, through the real router.

    This test used to assert that motion changed NOTHING, which was true while
    STATIC_ACTIVITY_THRESHOLD was None. The gate opened on 2026-08-16 and this
    correctly failed -- rewritten to the invariant that survives opening it,
    rather than deleted. A near-zero activity is used deliberately: if any
    ungated rule ever starts consulting motion, this is where it shows up.
    """
    from surgvu.router import answer_question

    plain, withm = _record_pair(0.01)
    for question in _UNGATED_QUESTIONS:
        assert answer_question(question, withm) == answer_question(
            question, plain), (
            "motion changed the answer to %r, which no rule should gate"
            % question)


def test_motion_above_the_threshold_changes_no_answer_at_all():
    """An ACTIVE scene must route exactly as it did before the gate existed."""
    from surgvu.router import answer_question
    from surgvu.router import STATIC_ACTIVITY_THRESHOLD

    plain, withm = _record_pair(STATIC_ACTIVITY_THRESHOLD * 4)
    for question in ("Is tissue being cut in this clip?",) + _UNGATED_QUESTIONS:
        assert answer_question(question, withm) == answer_question(
            question, plain), (
            "motion changed %r on an active scene, where it must be inert"
            % question)


def test_a_static_scene_downgrades_only_the_cutting_answer():
    """The feature itself: below threshold, cutting flips Yes -> No."""
    from surgvu.router import answer_question
    from surgvu.router import STATIC_ACTIVITY_THRESHOLD

    question = "Is tissue being cut in this clip?"
    plain, withm = _record_pair(STATIC_ACTIVITY_THRESHOLD / 2.0)
    assert answer_question(question, plain) == "Yes", (
        "fixture no longer answers Yes without motion, so this test would "
        "pass without exercising the gate")
    assert answer_question(question, withm) == "No"
