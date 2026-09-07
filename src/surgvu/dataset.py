"""Turn shards into training batches, split strictly by case.

A window's label is a property of the whole 30-second window, so every frame
in it carries the same label. That is correct here and not a shortcut: the
target is INSTALLATION STATE, which does not change within a window by
construction, unlike visibility which changes constantly.
"""
import json
import random
from pathlib import Path

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from .extract import read_shard
from .frames import sample_frame_indices
from .taxonomy import TASK_CLASSES, TOOL_CLASSES

_TOOL_INDEX = {name: i for i, name in enumerate(TOOL_CLASSES)}
_TASK_INDEX = {name: i for i, name in enumerate(TASK_CLASSES)}


def encode_tools(tools):
    """Multi-hot over the 12 emitted classes. Raises on anything else."""
    vector = np.zeros(len(TOOL_CLASSES), dtype=np.float32)
    for tool in tools:
        vector[_TOOL_INDEX[tool]] = 1.0
    return vector


def encode_task(task):
    return _TASK_INDEX[task]


def shard_paths_for_split(shard_dir, splits_path, split):
    """Shards whose case is in `split`. Empty is an error, not a result."""
    shard_dir = Path(shard_dir)
    cases = set(json.loads(Path(splits_path).read_text(encoding="utf-8"))[split])
    paths = sorted(p for p in shard_dir.glob("*.npz")
                   if p.name.rsplit("_part", 1)[0] in cases)
    if not paths:
        raise ValueError(
            "no shards matched split %r over %d case(s) under %s. Training on "
            "an empty split silently reports a meaningless loss."
            % (split, len(cases), shard_dir))
    return paths


class ShardFrames(IterableDataset):
    """Frames from shards, one shard resident at a time.

    Shard-at-a-time is deliberate: a random-access sampler over 235 npz files
    would reopen and re-read a shard for nearly every item. Shards are shuffled
    each epoch, and frames are subsampled per window, so a given frame is seen
    on some epochs and not others rather than the same 8 every time.

    WARNING — `set_epoch` is silently inert under `DataLoader(...,
    persistent_workers=True)`. Persistent workers are forked once and never
    re-pickle this dataset, so a worker process's copy of `self.epoch` stays
    0 for the life of training no matter how many times the main process
    calls `set_epoch` on its own copy. The per-epoch RNG seed is
    `self.seed + 1000 * self.epoch`, so a stuck epoch means every epoch draws
    the identical frame subsample and shard order — the epoch-to-epoch frame
    diversity this design depends on quietly disappears while training loss
    curves look completely normal. Callers must either leave
    `persistent_workers` False, or rebuild the `DataLoader` (not just call
    `set_epoch`) at the start of every epoch.
    """

    def __init__(self, shards, frames_per_window=8, seed=0, shuffle=True):
        self.shards = list(shards)
        if not self.shards:
            raise ValueError(
                "ShardFrames got an empty shard list. Iterating this would "
                "silently yield zero items, so a training run built on it "
                "would run to completion, converge on zero batches, and "
                "report a meaningless loss as if nothing were wrong. Pass a "
                "non-empty list (e.g. from shard_paths_for_split, which "
                "already guards its own empty case).")
        self.frames_per_window = frames_per_window
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    # WARNING: inert under DataLoader(..., persistent_workers=True) — see the
    # class docstring. Mutating self.epoch here only reaches a persistent
    # worker's copy of this dataset if the DataLoader (and its workers) are
    # rebuilt; otherwise the worker keeps using epoch 0 forever and the
    # failure is silent, not an exception.
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        worker = get_worker_info()
        shards = list(self.shards)
        if worker is not None:
            shards = shards[worker.id::worker.num_workers]
        rng = random.Random(self.seed + 1000 * self.epoch)
        if self.shuffle:
            rng.shuffle(shards)

        for path in shards:
            frames, meta = read_shard(path)
            order = list(range(len(meta)))
            if self.shuffle:
                rng.shuffle(order)
            for w in order:
                row = meta[w]
                tools = encode_tools(row["tools"])
                task = encode_task(row["task"])
                # len(), not frames.shape[1]: a LazyShard has to decode one
                # frame to answer `shape` (it reports what is in the shard
                # rather than what the metadata claims), and the loader does
                # not need that -- the depth is a length, not a measurement.
                depth = len(frames[w])
                k = min(self.frames_per_window, depth)
                picks = (rng.sample(range(depth), k) if self.shuffle
                         else list(range(k)))
                for f in picks:
                    yield frames[w][f], tools, task


class ShardClips(IterableDataset):
    """Whole CLIPS rather than loose frames, for a temporal model.

    `ShardFrames` yields frames independently and the model never learns that
    they are ordered -- which is the entire limitation a 3D network exists to
    remove. This yields a contiguous run of `clip_length` frames in order.

    CONTIGUOUS AND IN ORDER, both load-bearing. Sampling frames at random the
    way `ShardFrames` does would hand a 3D convolution a shuffled stack, and
    it would dutifully learn from temporal structure that is not there. The
    start offset is what varies between epochs, not the ordering.

    Intended for the DENSE shards (2 s at 15 fps), where consecutive frames
    are 67 ms apart. Pointed at the sparse 1 fps pool it still runs, and the
    "motion" it would model is a second of elapsed surgery per step -- which
    is exactly the null this experiment is trying to distinguish itself from.
    The caller is trusted to pass the right pool; the shard metadata records
    `fps`, so the trainer prints it rather than assuming.
    """

    def __init__(self, shards, clip_length=16, seed=0, shuffle=True):
        self.shards = list(shards)
        if not self.shards:
            raise ValueError(
                "ShardClips got an empty shard list. Iterating this would "
                "silently yield zero items and train to a meaningless loss.")
        self.clip_length = clip_length
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    # Same persistent_workers caveat as ShardFrames -- see that docstring.
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        worker = get_worker_info()
        shards = list(self.shards)
        if worker is not None:
            shards = shards[worker.id::worker.num_workers]
        rng = random.Random(self.seed + 1000 * self.epoch)
        if self.shuffle:
            rng.shuffle(shards)

        for path in shards:
            frames, meta = read_shard(path)
            order = list(range(len(meta)))
            if self.shuffle:
                rng.shuffle(order)
            for w in order:
                row = meta[w]
                tools = encode_tools(row["tools"])
                task = encode_task(row["task"])
                depth = len(frames[w])
                take = min(self.clip_length, depth)
                # Random start while training so epochs see different spans of
                # the burst; the CENTRE at eval so the number is reproducible
                # and does not move with the seed.
                if self.shuffle:
                    start = rng.randint(0, depth - take)
                else:
                    start = (depth - take) // 2
                clip = np.stack([frames[w][start + i] for i in range(take)])
                # A short window would produce a ragged batch that torch
                # cannot collate; pad by repeating the last frame, which adds
                # no motion rather than inventing some.
                if take < self.clip_length:
                    pad = np.repeat(clip[-1:], self.clip_length - take, axis=0)
                    clip = np.concatenate([clip, pad], axis=0)
                yield clip, tools, task


class ShardTemporal(IterableDataset):
    """Clips with an explicit LAYOUT and several clips per window per epoch.

    TWO THINGS `ShardClips` GOT WRONG, both of which shaped every 3D result so
    far.

    ONE CLIP PER WINDOW PER EPOCH. `ShardFrames` yields `frames_per_window`
    samples for every window -- thirty of them -- while `ShardClips` yields
    exactly one. At the same epoch count the 3D models therefore received
    about 30x fewer gradient samples per window than the 2D model did. That is
    a property of the loader, not of 3D convolution, and `clips_per_window`
    fixes it directly.

    ONE PLACE IN THE WINDOW. A contiguous run of 16 frames out of a 2-second
    centre burst is 3.6% of the 30 seconds the label describes. With the
    multi-burst pool the frames are spread across the window, and the layout
    decides how a clip draws on that:

        contiguous   one burst, frames adjacent in time. Real motion at 67 ms,
                     which is what a 3D convolution is for. Each burst is a
                     different clip, so `clips_per_window` up to `bursts`
                     costs nothing extra.

        spread       evenly spaced across the whole window. What TSM wants:
                     its shift relates frames wherever they sit, and spreading
                     them covers the labelled span rather than a sliver.

        bursts       every frame of several whole bursts, in time order, as
                     ONE clip: coverage AND local motion, which the two above
                     make you choose between. What `ResidualTemporal` needs,
                     because it reads the burst boundaries out of the clip.

    Both layouts work on the centre-only dense pool too -- "spread" there
    spreads within the burst -- so an arm can be measured on the old pool
    while the new one is still extracting.
    """

    def __init__(self, shards, frames=8, layout="contiguous", bursts=1,
                 clips_per_window=1, seed=0, shuffle=True):
        self.shards = list(shards)
        if not self.shards:
            raise ValueError("ShardTemporal got an empty shard list.")
        if layout not in ("contiguous", "spread", "bursts"):
            raise ValueError("unknown layout %r" % (layout,))
        self.frames = frames
        self.layout = layout
        self.bursts = max(1, bursts)
        self.clips_per_window = clips_per_window
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _starts(self, depth, rng):
        """Which clips to emit from a window of `depth` frames."""
        per_burst = depth // self.bursts
        if self.layout == "bursts":
            # WHOLE BURSTS, in time order, as ONE clip. `ResidualTemporal`
            # reads the clip as (bursts x frames_per_burst) and splits it at
            # multiples of frames_per_burst, so the boundaries in the clip have
            # to be the boundaries in the pool -- which is why this never
            # slices inside a burst and never jitters, unlike the layouts above.
            #
            # Fewer bursts than the pool holds is allowed and is the memory
            # lever: 8 of 16 bursts halves the frames per window. Which 8 are
            # bin centres over the burst INDICES, so a subset still spans the
            # window rather than taking its first half.
            if self.frames % per_burst:
                raise ValueError(
                    "the 'bursts' layout emits whole bursts, so frames (%d) "
                    "must be a multiple of the pool's %d frames per burst. "
                    "Asking for %d would hand the model a ragged final burst "
                    "and split every later one at the wrong frame."
                    % (self.frames, per_burst, self.frames))
            take = min(self.bursts, max(1, self.frames // per_burst))
            chosen = sample_frame_indices(self.bursts, take)
            picks = []
            for burst in chosen:
                base = burst * per_burst
                picks.extend(range(base, base + per_burst))
            return [picks]

        if self.layout == "spread":
            # BIN CENTRES via sample_frame_indices, the same helper the 2D path
            # uses -- not `i * step`. With depth 30 and 16 frames, step is
            # max(1, 30//16) = 1 and the naive version returns frames 0..15:
            # the first sixteen seconds of a thirty-second window, called
            # "spread". It happened to be harmless on the 32-frame multi pool
            # (step 4, two frames per burst) and would have quietly wrecked any
            # run over the 30-frame sparse pool.
            picks = list(sample_frame_indices(depth, self.frames))
            if self.shuffle and len(picks) > 1:
                # Jitter by one frame while training so epochs differ, clamped
                # so the order and the coverage are unchanged.
                shift = rng.randint(-1, 1)
                picks = [min(depth - 1, max(0, i + shift)) for i in picks]
            return [picks]

        take = min(self.frames, per_burst)
        order = list(range(self.bursts))
        if self.shuffle:
            rng.shuffle(order)
        else:
            # Eval: the FIRST burst, deterministically. Averaging across bursts
            # belongs in the evaluator, where it can be reported per arm,
            # rather than hidden inside the loader.
            order = sorted(order)
        picks = []
        for burst in order[:max(1, self.clips_per_window)]:
            base = burst * per_burst
            slack = per_burst - take
            start = base + (rng.randint(0, slack) if (self.shuffle and slack)
                            else slack // 2)
            picks.append(list(range(start, start + take)))
        return picks

    def __iter__(self):
        worker = get_worker_info()
        shards = list(self.shards)
        if worker is not None:
            shards = shards[worker.id::worker.num_workers]
        rng = random.Random(self.seed + 1000 * self.epoch)
        if self.shuffle:
            rng.shuffle(shards)

        for path in shards:
            frames, meta = read_shard(path)
            order = list(range(len(meta)))
            if self.shuffle:
                rng.shuffle(order)
            for w in order:
                row = meta[w]
                tools = encode_tools(row["tools"])
                task = encode_task(row["task"])
                depth = len(frames[w])
                for picks in self._starts(depth, rng):
                    clip = np.stack([frames[w][i] for i in picks])
                    if len(clip) < self.frames:
                        # Repeat the last frame rather than wrapping: wrapping
                        # splices a jump cut into the middle of a motion.
                        clip = np.concatenate(
                            [clip, np.repeat(clip[-1:],
                                             self.frames - len(clip), axis=0)])
                    yield clip, tools, task
