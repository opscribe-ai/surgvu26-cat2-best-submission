import json
import numpy as np
import pytest

from surgvu.dataset import (
    encode_tools, encode_task, shard_paths_for_split, ShardFrames,
    ShardTemporal,
)
from surgvu.extract import write_shard
from surgvu.sampling import Window
from surgvu.taxonomy import TOOL_CLASSES, TASK_CLASSES


def _window(case, start, task="suturing", tools=("needle driver",)):
    return Window(case=case, part="1.0", start=start, length=30.0,
                  task=task, description="d", tools=frozenset(tools))


def _make_shard(tmp_path, case, n_windows=2, depth=3):
    frames = [np.full((8, 8, 3), i * 10, dtype=np.uint8) for i in range(depth)]
    payload = [(_window(case, 30.0 * w), list(frames)) for w in range(n_windows)]
    path = tmp_path / f"{case}_part1.npz"
    write_shard(payload, path, frames_per_window=depth)
    return path


def test_encode_tools_is_multi_hot_over_the_twelve_classes():
    vector = encode_tools(["needle driver", "stapler"])
    assert vector.shape == (12,)
    assert vector.dtype == np.float32
    assert vector.sum() == 2.0
    assert vector[TOOL_CLASSES.index("needle driver")] == 1.0
    assert vector[TOOL_CLASSES.index("stapler")] == 1.0


def test_encode_tools_rejects_a_class_outside_the_twelve():
    """Out-of-scope classes must never reach a prediction vector."""
    with pytest.raises(KeyError):
        encode_tools(["suction irrigator"])


def test_encode_task_maps_to_a_stable_index():
    assert encode_task("suturing") == TASK_CLASSES.index("suturing")
    with pytest.raises(KeyError):
        encode_task("dissection")


def test_split_selects_shards_by_case_and_never_mixes(tmp_path):
    _make_shard(tmp_path, "case_000")
    _make_shard(tmp_path, "case_001")
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": ["case_000"], "val": ["case_001"]}))

    train = shard_paths_for_split(tmp_path, splits, "train")
    val = shard_paths_for_split(tmp_path, splits, "val")

    assert [p.name for p in train] == ["case_000_part1.npz"]
    assert [p.name for p in val] == ["case_001_part1.npz"]
    assert not set(train) & set(val)


def test_split_raises_when_it_selects_nothing(tmp_path):
    """An empty split is the silent-nothing failure: training would run,
    converge on zero batches, and report a meaningless loss."""
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": ["case_999"], "val": []}))
    with pytest.raises(ValueError, match="no shards"):
        shard_paths_for_split(tmp_path, splits, "train")


def test_dataset_yields_frames_with_their_window_labels(tmp_path):
    path = _make_shard(tmp_path, "case_000", n_windows=2, depth=3)
    dataset = ShardFrames([path], frames_per_window=3, shuffle=False)
    items = list(dataset)

    assert len(items) == 6                      # 2 windows x 3 frames
    frame, tools, task = items[0]
    assert frame.shape == (8, 8, 3)
    assert tools.shape == (12,)
    assert tools[TOOL_CLASSES.index("needle driver")] == 1.0
    assert task == TASK_CLASSES.index("suturing")


def test_dataset_subsamples_frames_per_window(tmp_path):
    path = _make_shard(tmp_path, "case_000", n_windows=2, depth=3)
    dataset = ShardFrames([path], frames_per_window=2, shuffle=False)
    assert len(list(dataset)) == 4              # 2 windows x 2 frames


def test_dataset_rejects_an_empty_shard_list():
    """An empty ShardFrames is the same silent-nothing failure as an empty
    split: it would iterate to zero items and let training report a
    meaningless loss as if nothing were wrong. This guard must hold even
    when a caller builds ShardFrames directly, bypassing
    shard_paths_for_split's own guard."""
    with pytest.raises(ValueError, match="empty shard list"):
        ShardFrames([])


# --------------------------------------------------------------------------
# the loader decodes only the frames it uses
# --------------------------------------------------------------------------
# Windows hold 30 frames on the real corpus and training samples 8 of them, so
# an eager shard reader threw away most of what it decoded -- while holding
# the whole decoded shard resident, once per DataLoader worker. Four workers
# of that is what took job 9618015 past its 32 GB cgroup limit.

def test_the_loader_decodes_only_the_frames_it_yields(tmp_path, monkeypatch):
    import cv2

    path = _make_shard(tmp_path, "case_000", n_windows=3, depth=6)
    calls = []
    real = cv2.imdecode
    monkeypatch.setattr(cv2, "imdecode",
                        lambda buffer, flags: calls.append(1) or real(buffer, flags))

    items = list(ShardFrames([path], frames_per_window=2, shuffle=False))

    assert len(items) == 6                      # 3 windows x 2 frames
    # Exactly the frames it yielded. The shard holds 18, and the loader does
    # not touch the shard's `shape` (which would cost one probe decode).
    assert len(calls) == 6, (
        "decoded %d frames to yield 6 of the shard's 18" % len(calls))


def test_the_loader_yields_the_same_frames_it_did_when_reading_eagerly(tmp_path):
    """Laziness changes WHEN a frame is decoded, not which frame or what is
    in it. Same seed, same shard, same stream of (frame, tools, task)."""
    path = _make_shard(tmp_path, "case_000", n_windows=3, depth=6)

    ordered = list(ShardFrames([path], frames_per_window=6, shuffle=False))
    shuffled_a = list(ShardFrames([path], frames_per_window=3, seed=5))
    shuffled_b = list(ShardFrames([path], frames_per_window=3, seed=5))

    # The unshuffled stream is the whole shard in file order: window 0's six
    # frames, then window 1's, then window 2's.
    assert [round(float(f.mean())) for f, _t, _k in ordered] == \
        [0, 10, 20, 30, 40, 50] * 3
    # And a seeded shuffle is reproducible, which is what makes the eager and
    # lazy readers comparable at all.
    assert [round(float(f.mean())) for f, _t, _k in shuffled_a] == \
        [round(float(f.mean())) for f, _t, _k in shuffled_b]
    assert len(shuffled_a) == 9


def _burst_shard(tmp_path, case="case_000", bursts=4, per_burst=3):
    """A shard whose frames encode WHICH frame they are, so picks are readable.

    Frame i is filled with i*10, and `write_shard` JPEG-encodes it at quality
    90 -- a flat patch survives that with an error well under 5, so dividing
    the decoded mean by ten recovers the index exactly.
    """
    depth = bursts * per_burst
    frames = [np.full((8, 8, 3), i * 10, dtype=np.uint8) for i in range(depth)]
    payload = [(_window(case, 0.0), list(frames))]
    path = tmp_path / f"{case}_part1.npz"
    write_shard(payload, path, frames_per_window=depth)
    return path


def _indices(clip):
    return [int(round(float(f.mean()) / 10.0)) for f in clip]


def test_the_bursts_layout_yields_every_frame_of_every_burst_in_order(tmp_path):
    path = _burst_shard(tmp_path, bursts=4, per_burst=3)
    clips = [c for c, _t, _k in
             ShardTemporal([path], frames=12, layout="bursts", bursts=4,
                           shuffle=False)]
    assert len(clips) == 1
    assert _indices(clips[0]) == list(range(12))


def test_the_bursts_layout_never_jitters_across_a_burst_boundary(tmp_path):
    """Unlike `spread`, which jitters by a frame while training.

    ResidualTemporal splits the clip at multiples of frames_per_burst, so a
    one-frame jitter would put the tail of one burst and the head of the next
    into a single 'motion' -- a 7.5-second jump cut called a 67 ms step.
    """
    path = _burst_shard(tmp_path, bursts=4, per_burst=3)
    shuffled = [c for c, _t, _k in
                ShardTemporal([path], frames=12, layout="bursts", bursts=4,
                              seed=11, shuffle=True)]
    assert _indices(shuffled[0]) == list(range(12))


def test_the_bursts_layout_spans_the_window_when_it_takes_a_subset(tmp_path):
    """Two bursts of four must be bin centres, not the first two.

    The memory lever is dropping bursts, and dropping them off the front of
    the window would quietly reintroduce exactly the coverage handicap this
    pool was extracted to remove.
    """
    path = _burst_shard(tmp_path, bursts=4, per_burst=3)
    clips = [c for c, _t, _k in
             ShardTemporal([path], frames=6, layout="bursts", bursts=4,
                           shuffle=False)]
    # Bursts 1 and 3 of 0..3: frames 3,4,5 and 9,10,11.
    assert _indices(clips[0]) == [3, 4, 5, 9, 10, 11]


def test_the_bursts_layout_refuses_a_ragged_frame_count(tmp_path):
    path = _burst_shard(tmp_path, bursts=4, per_burst=3)
    with pytest.raises(ValueError, match="multiple of the pool's 3"):
        list(ShardTemporal([path], frames=8, layout="bursts", bursts=4,
                           shuffle=False))


def test_the_bursts_layout_is_the_same_clip_every_epoch(tmp_path):
    """Deterministic by construction: there is nothing left to sample."""
    path = _burst_shard(tmp_path, bursts=4, per_burst=3)
    dataset = ShardTemporal([path], frames=12, layout="bursts", bursts=4,
                            seed=3, shuffle=True)
    dataset.set_epoch(0)
    first = _indices(next(iter(dataset))[0])
    dataset.set_epoch(7)
    assert _indices(next(iter(dataset))[0]) == first


def test_an_unknown_layout_is_refused(tmp_path):
    path = _burst_shard(tmp_path)
    with pytest.raises(ValueError, match="unknown layout"):
        ShardTemporal([path], layout="whatever")
