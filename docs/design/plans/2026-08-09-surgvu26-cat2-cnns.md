# SurgVU 2026 Cat 2 — Perception Models Implementation Plan (Plan 2)


**Goal:** Train the three perception experts — tool install-state recogniser, task classifier, and action recogniser — against the extracted 24,578-window shard corpus, and emit a validation report that Plan 3's router consumes.

**Architecture:** One shared training substrate (`dataset`, `models`, `metrics`, `train`) plus three thin driver scripts, so all three models share the same loader, seeding, checkpoint format, and metric code. Each model is an ImageNet-pretrained EfficientNetV2-S over blurred 30-second window frames; per-frame predictions are averaged across a window's 30 frames into a clip-level prediction, because the label being predicted is *installation state*, which is constant across the window by construction. Training runs as HTCondor GPU jobs inside a container we build ourselves.

**Tech Stack:** Python 3.11, PyTorch 2.5.1 + torchvision 0.20.1 (CUDA 12.1), pytest, numpy, OpenCV. Backbones come from torchvision (**BSD-3**, deliberately — no AGPL enters this repo).

## Global Constraints

- **12 tool classes only**, emitted as 12 independent sigmoids: `bipolar forceps`, `cadiere forceps`, `clip applier`, `force bipolar`, `grasping retractor`, `monopolar curved scissors`, `needle driver`, `permanent cautery hook/spatula`, `prograsp forceps`, `stapler`, `tip-up fenestrated grasper`, `vessel sealer`. Never emit anything else.
- **8 task classes**: `other`, `range of motion`, `rectal artery/vein`, `retraction and collision avoidance`, `skills application`, `suspensory ligaments`, `suturing`, `uterine horn`.
- **NOT top-3.** The output is 12 independent sigmoids with tuned thresholds. "Three instruments installed" is a fact about *arm slots*, not classes: two arms carry the same class in nearly half of sampled moments, and the window-weighted distribution over distinct classes peaks at 3 with only **44.6%** share. Hard-constraining to three would be wrong closer to one time in two.
- **The target is installation state, not visibility.** Tools may be installed yet obscured. A model that gets better at *seeing* instruments can move away from ground truth. Never add a visibility-based objective.
- **Split by case, never by window or frame.** Use `config/splits.json` verbatim: 124 train / 31 val. Windows from one session are near-duplicates.
- **Frames are already UI-blurred** in the shards (`ui_blurred: true` on all 24,578). Never disable it; it is challenge compliance, not a hyperparameter.
- **Never train or tune on `cat1_test_set_public.zip`.**
- **No artifact whose training history includes private clinical data.** Public checkpoints only (ImageNet weights are fine).
- **Independent of OpScribe.** Build our own containers under `/staging/n/nkalthoff/surgvu26/`. Never use `/staging/groups/bhaskar_opscribe/**` containers or `pypkgs`. Reading the shard corpus from group staging is fine and expected.
- **Any stage that can produce zero output must treat zero as an error** unless explicitly told otherwise. This project's characteristic failure is plausible-looking nothing.
- **When adding a test, mutate the thing it protects and confirm the test dies.** A test that passes against its own defect is not a test.
- **Deployment target is a T4 (Turing, sm_75): no bf16, no FlashAttention-2.** Train in fp16/fp32; never default to bf16. CHTC has no T4 — validate on `NVIDIA GeForce RTX 2080 Ti` (31 available, identical compute capability 7.5).

## Corpus Facts (measured, do not re-derive)

- Shards: `/staging/groups/bhaskar_opscribe/surgvu/shards/*.npz`, **235 shards, 24,578 windows, 40.5 GB**.
- Every window is **30 frames at 512x512x3**, `fps=1`, `jpeg_quality=90`, `ui_blurred=True`.
- Shard filename is `case_NNN_partP.npz`. `read_shard(path)` returns `(frames, meta)` where `frames` is `(windows, 30, 512, 512, 3)` uint8 and `meta` is a list of dicts with `case, part, start, length, task, description, tools, ui_blurred, frame_size, fps, jpeg_quality`.
- Tool class frequencies over train windows (`config/tool_frequency.json`): `cadiere forceps` 15,600 down to `stapler` 137. A **90x** imbalance — `pos_weight` is mandatory, not optional.
- Task distribution over all windows: suturing 8,633; uterine horn 4,058; rectal artery/vein 3,674; suspensory ligaments 3,525; skills application 2,530; retraction and collision avoidance 1,322; other 483; range of motion 353.
- **The 7 out-of-scope tool classes are NOT in the shards.** `labels._load_tools` drops them at load time via `normalize_tool`. The spec offers them as auxiliary signal; recovering them means re-querying `tools.csv` using each window's `case`/`part`/`start`. Deferred to ablation — do not block on it.

## File Structure

| File | Responsibility |
|---|---|
| `src/surgvu/dataset.py` | Shard indexing, case-level split, frame sampling, label encoding |
| `src/surgvu/models.py` | Backbone factory and the two head types |
| `src/surgvu/metrics.py` | Macro-F1, per-class F1, per-class threshold tuning |
| `src/surgvu/train.py` | Seeding, train/val loop, checkpoint format |
| `scripts/train_tools.py` | Driver: tool recogniser |
| `scripts/train_task.py` | Driver: task classifier |
| `scripts/train_action.py` | Driver: action recogniser (gated) |
| `scripts/report_val.py` | Combined validation report consumed by Plan 3 |
| `containers/surgvu26-train.def` | GPU training container definition |
| `condor/train.sub`, `condor/train.sh` | GPU job submission |
| `tests/test_dataset.py`, `tests/test_models.py`, `tests/test_metrics.py`, `tests/test_train.py` | Unit tests |

---

### Task 1: GPU training container

**Files:**
- Create: `containers/surgvu26-train.def`
- Create: `containers/build_train_sif.sh`
- Create: `containers/build_train.sub`

**Interfaces:**
- Consumes: nothing.
- Produces: `/staging/n/nkalthoff/surgvu26/surgvu26-train.sif`, an image where `python3` has `torch`, `torchvision`, `numpy`, `cv2` and CUDA.

- [ ] **Step 1: Write the container definition**

```
Bootstrap: docker
From: pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Training container for SurgVU26 Cat 2 perception models.
#
# Independent by construction: a public PyTorch base image plus two pip
# packages. Nothing from /staging/groups/bhaskar_opscribe.
#
# cu121 includes sm_75, so this image also runs on the RTX 2080 Ti we use to
# validate against the T4 deployment target.

%post
    pip install --no-cache-dir \
        "opencv-python-headless==4.11.0.86" \
        "PyYAML==6.0.2"
    python -c "import torch, torchvision, cv2, yaml; print(torch.__version__, torchvision.__version__, cv2.__version__)"

%environment
    export PYTHONUNBUFFERED=1

%labels
    Purpose SurgVU26 Cat2 perception training
```

- [ ] **Step 2: Write the build script**

```bash
#!/bin/bash
set -u
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin
echo "host: $(hostname)  start: $(date)"
command -v apptainer || { echo "FATAL: no apptainer"; exit 1; }

OUT=/staging/n/nkalthoff/surgvu26/surgvu26-train.sif
mkdir -p /staging/n/nkalthoff/surgvu26
export APPTAINER_CACHEDIR="$PWD/.apptainer_cache"
export APPTAINER_TMPDIR="$PWD/.apptainer_tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

if ! apptainer build surgvu26-train.sif surgvu26-train.def; then
  echo "plain build failed, retrying with --fakeroot"
  apptainer build --fakeroot surgvu26-train.sif surgvu26-train.def || exit 1
fi

apptainer exec surgvu26-train.sif python -c "
import torch, torchvision, cv2
print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| cv2', cv2.__version__)
" || exit 1

cp surgvu26-train.sif "$OUT" && chmod 664 "$OUT"
echo "wrote $OUT ($(stat -c %s "$OUT") bytes)"
echo "end: $(date)"
```

- [ ] **Step 3: Write the build submit file**

```
universe   = vanilla
executable = containers/build_train_sif.sh
transfer_input_files = containers/surgvu26-train.def
output = logs/build_train_$(Cluster).out
error  = logs/build_train_$(Cluster).err
log    = logs/build_train_$(Cluster).log
+WantStagingMount = true
request_cpus   = 4
request_memory = 16GB
request_disk   = 40GB
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
queue 1
```

- [ ] **Step 4: Build it (never on the login node)**

Run: `mkdir -p logs && chmod +x containers/build_train_sif.sh && condor_submit containers/build_train.sub`
Wait for completion, then confirm the smoke test printed versions and the `.sif` exists.
Expected: `wrote /staging/n/nkalthoff/surgvu26/surgvu26-train.sif`

- [ ] **Step 5: Verify a GPU is actually visible from inside the image**

Submit a one-off job with `container_image = file:///staging/n/nkalthoff/surgvu26/surgvu26-train.sif`, `request_gpus = 1`, `require_gpus = (DeviceName == "NVIDIA GeForce RTX 2080 Ti")`, running:

```bash
python3 -c "
import torch
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('capability', torch.cuda.get_device_capability(0))
assert torch.cuda.get_device_capability(0) == (7, 5), 'not Turing — wrong validation target'
"
```

Expected: `cuda True NVIDIA GeForce RTX 2080 Ti`, `capability (7, 5)`.

- [ ] **Step 6: Commit**

```bash
git add containers/ && git commit -m "feat: GPU training container, independent of OpScribe"
```

---

### Task 2: Shard dataset and label encoding

**Files:**
- Create: `src/surgvu/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `surgvu.extract.read_shard`, `surgvu.taxonomy.TOOL_CLASSES`, `TASK_CLASSES`, `tool_index`.
- Produces:
  - `encode_tools(tools) -> np.ndarray` shape `(12,)` float32 multi-hot.
  - `encode_task(task) -> int` in `[0, 8)`.
  - `shard_paths_for_split(shard_dir, splits_path, split) -> list[Path]`
  - `ShardFrames(shards, frames_per_window=8, seed=0, shuffle=True)` — a `torch.utils.data.IterableDataset` yielding `(frame_uint8_hwc, tools_multihot, task_index)`.

- [ ] **Step 1: Write the failing test**

```python
import json
import numpy as np
import pytest

from surgvu.dataset import (
    encode_tools, encode_task, shard_paths_for_split, ShardFrames,
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'surgvu.dataset'`

- [ ] **Step 3: Implement the module**

```python
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
    """

    def __init__(self, shards, frames_per_window=8, seed=0, shuffle=True):
        self.shards = list(shards)
        self.frames_per_window = frames_per_window
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

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
                depth = frames.shape[1]
                k = min(self.frames_per_window, depth)
                picks = (rng.sample(range(depth), k) if self.shuffle
                         else list(range(k)))
                for f in picks:
                    yield frames[w][f], tools, task
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_dataset.py -v`
Expected: 7 passed

- [ ] **Step 5: Mutate and confirm the tests die**

Run each mutation, confirm the named test FAILS, then revert:

1. In `shard_paths_for_split`, delete the `if not paths: raise` block → `test_split_raises_when_it_selects_nothing` must fail.
2. In `encode_tools`, replace `vector[_TOOL_INDEX[tool]] = 1.0` with `vector[_TOOL_INDEX.get(tool, 0)] = 1.0` → `test_encode_tools_rejects_a_class_outside_the_twelve` must fail.
3. In `ShardFrames.__iter__`, change `k = min(self.frames_per_window, depth)` to `k = depth` → `test_dataset_subsamples_frames_per_window` must fail.

If any mutation leaves the suite green, the test is not protecting what it claims. Fix the test before continuing.

- [ ] **Step 6: Commit**

```bash
git add src/surgvu/dataset.py tests/test_dataset.py
git commit -m "feat: shard dataset with case-level split and label encoding"
```

---

### Task 3: Model definitions

**Files:**
- Create: `src/surgvu/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `torchvision.models.efficientnet_v2_s`.
- Produces: `build_model(num_outputs, backbone="efficientnet_v2_s", pretrained=True) -> torch.nn.Module` whose `forward(x)` takes `(B, 3, H, W)` float32 in `[0, 1]` and returns raw logits `(B, num_outputs)`.

- [ ] **Step 1: Write the failing test**

```python
import pytest
import torch

from surgvu.models import build_model


def test_model_emits_one_logit_per_class():
    model = build_model(num_outputs=12, pretrained=False)
    out = model(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 12)


def test_model_emits_raw_logits_not_probabilities():
    """The loss functions apply their own sigmoid/softmax. A model that
    already squashed its output would be trained through it twice, which
    flattens gradients and looks like slow convergence rather than a bug."""
    torch.manual_seed(0)
    model = build_model(num_outputs=12, pretrained=False)
    out = model(torch.randn(8, 3, 64, 64))
    assert out.min() < 0.0 or out.max() > 1.0


def test_task_head_width_is_independent_of_tool_head_width():
    assert build_model(num_outputs=8, pretrained=False)(
        torch.zeros(1, 3, 64, 64)).shape == (1, 8)


def test_unknown_backbone_fails_loudly():
    with pytest.raises(ValueError, match="unknown backbone"):
        build_model(num_outputs=12, backbone="not_a_real_net", pretrained=False)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'surgvu.models'`

- [ ] **Step 3: Implement**

```python
"""Backbones for the perception experts.

EfficientNetV2-S is the default because the 2025 winner reached 97% macro-F1
with it on this exact data. That is a strong empirical prior, not a principled
choice, so `backbone` stays a parameter and the ablation includes ResNet-50.

Weights come from torchvision, which is BSD-3. This matters: the repo goes
public at submission, and an AGPL detector dependency would force AGPL on the
whole released work.
"""
import torch.nn as nn
import torchvision

_BACKBONES = {
    "efficientnet_v2_s": (
        torchvision.models.efficientnet_v2_s,
        torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
    ),
    "resnet50": (
        torchvision.models.resnet50,
        torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
    ),
}


def build_model(num_outputs, backbone="efficientnet_v2_s", pretrained=True):
    """A backbone with its classifier replaced by a `num_outputs` head.

    Returns RAW LOGITS. BCEWithLogitsLoss and CrossEntropyLoss both apply
    their own squashing; a model that pre-applied it would be squashed twice.
    """
    if backbone not in _BACKBONES:
        raise ValueError("unknown backbone %r; expected one of %s"
                         % (backbone, sorted(_BACKBONES)))
    factory, weights = _BACKBONES[backbone]
    model = factory(weights=weights if pretrained else None)

    if backbone.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_outputs)
    else:
        model.fc = nn.Linear(model.fc.in_features, num_outputs)
    return model
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_models.py -v`
Expected: 4 passed

- [ ] **Step 5: Mutate and confirm the tests die**

1. Append `nn.Sigmoid()` after the head → `test_model_emits_raw_logits_not_probabilities` must fail.
2. Replace the `raise ValueError` with a silent default to EfficientNet → `test_unknown_backbone_fails_loudly` must fail.

- [ ] **Step 6: Commit**

```bash
git add src/surgvu/models.py tests/test_models.py
git commit -m "feat: EfficientNetV2-S / ResNet-50 backbones with configurable head"
```

---

### Task 4: Metrics and threshold tuning

**Files:**
- Create: `src/surgvu/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: numpy only (no sklearn — it is not in the container).
- Produces:
  - `per_class_f1(y_true, y_pred) -> np.ndarray` shape `(C,)`
  - `macro_f1(y_true, y_pred) -> float`
  - `tune_thresholds(y_true, probs, grid=None) -> np.ndarray` shape `(C,)`
  - `multiclass_accuracy(y_true_idx, logits) -> float`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from surgvu.metrics import (
    macro_f1, multiclass_accuracy, per_class_f1, tune_thresholds,
)


def test_per_class_f1_is_one_for_perfect_predictions():
    y = np.array([[1, 0], [0, 1]], dtype=np.float32)
    assert per_class_f1(y, y).tolist() == [1.0, 1.0]


def test_a_class_never_predicted_scores_zero_not_nan():
    """A rare class the model never fires on must drag macro-F1 down. If it
    silently becomes NaN or is skipped, macro-F1 flatters the model exactly
    where the corpus is weakest -- stapler has 137 windows against cadiere's
    15,600."""
    y = np.array([[1, 1], [1, 0]], dtype=np.float32)
    pred = np.array([[1, 0], [1, 0]], dtype=np.float32)
    scores = per_class_f1(y, pred)
    assert scores[1] == 0.0
    assert not np.isnan(scores).any()


def test_macro_f1_weights_every_class_equally():
    y = np.array([[1, 0]] * 99 + [[0, 1]], dtype=np.float32)
    # Class 0 perfect (99 TP, 0 FP, 0 FN); class 1 never predicted.
    # The last row predicts NOTHING -- predicting class 0 there would add a
    # false positive and drop class 0's F1 to 198/199, which is not the point
    # being made.
    pred = np.array([[1, 0]] * 99 + [[0, 0]], dtype=np.float32)
    assert macro_f1(y, pred) == pytest.approx(0.5)   # (1.0 + 0.0) / 2


def test_tune_thresholds_finds_a_better_cut_than_one_half():
    """A class whose probabilities all sit below 0.5 is invisible at the
    default threshold. Rare classes behave exactly like this."""
    y = np.array([[1], [1], [0], [0]], dtype=np.float32)
    probs = np.array([[0.4], [0.35], [0.1], [0.05]], dtype=np.float32)
    thresholds = tune_thresholds(y, probs)
    assert thresholds[0] < 0.5
    assert macro_f1(y, (probs >= thresholds).astype(np.float32)) == 1.0


def test_tune_thresholds_returns_one_threshold_per_class():
    y = np.zeros((4, 3), dtype=np.float32)
    y[0] = 1
    probs = np.random.RandomState(0).rand(4, 3).astype(np.float32)
    assert tune_thresholds(y, probs).shape == (3,)


def test_multiclass_accuracy_uses_argmax():
    logits = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    assert multiclass_accuracy(np.array([1, 0]), logits) == 1.0
    assert multiclass_accuracy(np.array([0, 0]), logits) == 0.5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'surgvu.metrics'`

- [ ] **Step 3: Implement**

```python
"""Metrics, implemented directly rather than pulled from sklearn.

sklearn is not in the training container and is not worth adding for four
functions. More importantly, `f1_score(..., zero_division=...)` defaults bite
here: a rare class the model never predicts must score 0 and drag macro-F1
down, not vanish into a NaN or be silently skipped.
"""
import numpy as np


def per_class_f1(y_true, y_pred):
    """F1 per column. A class with no true positives scores 0.0, never NaN."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1 - y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1 - y_pred)).sum(axis=0)
    denominator = 2 * tp + fp + fn
    return np.where(denominator > 0, 2 * tp / np.maximum(denominator, 1e-12), 0.0)


def macro_f1(y_true, y_pred):
    return float(per_class_f1(y_true, y_pred).mean())


def tune_thresholds(y_true, probs, grid=None):
    """Per-class threshold maximising that class's F1 on the given data.

    Tuned on VALIDATION and then frozen. Tuning on train would pick cuts that
    fit noise the model already memorised.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.01, dtype=np.float32)

    thresholds = np.full(probs.shape[1], 0.5, dtype=np.float32)
    for c in range(probs.shape[1]):
        best, best_threshold = -1.0, 0.5
        for threshold in grid:
            pred = (probs[:, c] >= threshold).astype(np.float32)
            score = per_class_f1(y_true[:, c:c + 1], pred[:, None])[0]
            if score > best:
                best, best_threshold = score, float(threshold)
        thresholds[c] = best_threshold
    return thresholds


def multiclass_accuracy(y_true_idx, logits):
    return float((np.asarray(logits).argmax(axis=1) ==
                  np.asarray(y_true_idx)).mean())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_metrics.py -v`
Expected: 6 passed

- [ ] **Step 5: Mutate and confirm the tests die**

1. In `per_class_f1`, change the `np.where` fallback from `0.0` to `1.0` → `test_a_class_never_predicted_scores_zero_not_nan` and `test_macro_f1_weights_every_class_equally` must fail.
2. In `tune_thresholds`, return `np.full(probs.shape[1], 0.5)` unconditionally → `test_tune_thresholds_finds_a_better_cut_than_one_half` must fail.

- [ ] **Step 6: Commit**

```bash
git add src/surgvu/metrics.py tests/test_metrics.py
git commit -m "feat: macro-F1 and per-class threshold tuning without sklearn"
```

---

### Task 5: Training loop

**Files:**
- Create: `src/surgvu/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `surgvu.dataset.ShardFrames`, `surgvu.models.build_model`, `surgvu.metrics`.
- Produces:
  - `seed_everything(seed) -> None`
  - `prepare_batch(frames_uint8, device, image_size=None) -> torch.Tensor` — `(B, 3, H, W)` float32 scaled to `[0, 1]`, channel-first, BGR→RGB, bilinearly resized to `image_size` when given. **Training and inference both go through this one function**, so the resolution a model trained at cannot drift from the resolution it is served at.
  - `run_epoch(model, loader, loss_fn, target_fn, optimizer=None, device="cpu", image_size=None) -> dict` with keys `loss`, `probs`, `targets`. `target_fn(tools, task) -> Tensor` selects which label this model trains against; it is explicit because all three models share one loader but consume different labels, and inferring it from the loss class would silently hand the action model the tool labels.
  - `save_checkpoint(path, model, meta) -> None` / `load_checkpoint(path, model) -> dict`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import torch

from surgvu.train import (
    load_checkpoint, prepare_batch, save_checkpoint, seed_everything,
)


def test_prepare_batch_is_channel_first_and_unit_scaled():
    frames = np.full((2, 8, 8, 3), 255, dtype=np.uint8)
    batch = prepare_batch(frames, device="cpu")
    assert batch.shape == (2, 3, 8, 8)
    assert batch.dtype == torch.float32
    assert float(batch.max()) == 1.0


def test_prepare_batch_resizes_when_asked():
    """Shards are 512; the backbone was pretrained at 384. If training and
    inference resize differently the model silently loses accuracy, so the
    resize lives in one function used by both."""
    frames = np.zeros((2, 512, 512, 3), dtype=np.uint8)
    assert prepare_batch(frames, device="cpu", image_size=384).shape == (2, 3, 384, 384)
    assert prepare_batch(frames, device="cpu").shape == (2, 3, 512, 512)


def test_prepare_batch_converts_bgr_to_rgb():
    """Shards hold OpenCV BGR. ImageNet weights expect RGB. Swapping the
    channels silently costs accuracy that looks like an architecture problem."""
    frame = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    frame[..., 0] = 255                      # blue in BGR
    batch = prepare_batch(frame, device="cpu")
    assert float(batch[0, 2].max()) == 1.0   # lands in the RED channel
    assert float(batch[0, 0].max()) == 0.0


def test_seed_everything_makes_two_runs_identical():
    seed_everything(7)
    a = torch.randn(4)
    seed_everything(7)
    assert torch.equal(a, torch.randn(4))


def test_run_epoch_uses_the_target_fn_it_was_given(tmp_path):
    """All three models share one loader but consume different labels. If the
    target were inferred from the loss class, the action model would silently
    train against the 12-way tool vector and still report a falling loss."""
    from surgvu.train import run_epoch

    frames = torch.zeros(2, 4, 4, 3, dtype=torch.uint8)
    tools = torch.zeros(2, 12)
    task = torch.tensor([1, 0])
    loader = [(frames, tools, task)]
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 8))

    stats = run_epoch(model, loader, torch.nn.CrossEntropyLoss(),
                      target_fn=lambda tools, task: task, device="cpu")

    assert stats["targets"].tolist() == [1, 0]


def test_checkpoint_roundtrips_weights_and_metadata(tmp_path):
    model = torch.nn.Linear(4, 2)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, {"macro_f1": 0.5, "thresholds": [0.3] * 2})

    restored = torch.nn.Linear(4, 2)
    meta = load_checkpoint(path, restored)

    assert meta["macro_f1"] == 0.5
    assert torch.equal(model.weight, restored.weight)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'surgvu.train'`

- [ ] **Step 3: Implement**

```python
"""Shared training substrate for all three perception experts.

One loop, one checkpoint format, one seeding routine, so the three models
differ only in their head width, loss, and labels -- and so a metric measured
on one is comparable to a metric measured on another.
"""
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_batch(frames_uint8, device, image_size=None):
    """(B, H, W, 3) uint8 BGR -> (B, 3, H, W) float32 RGB in [0, 1].

    The resize lives here, not in the caller, so training and inference
    cannot disagree about it. Shards hold 512x512; EfficientNetV2-S was
    pretrained at 384. Training at one resolution and serving at another is
    a silent accuracy loss that looks like a bad architecture choice.
    """
    array = np.asarray(frames_uint8)
    rgb = array[..., ::-1].copy()             # OpenCV BGR -> RGB
    tensor = torch.from_numpy(rgb).to(device)
    batch = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    if image_size:
        batch = torch.nn.functional.interpolate(
            batch, size=(image_size, image_size),
            mode="bilinear", align_corners=False)
    return batch


def run_epoch(model, loader, loss_fn, target_fn, optimizer=None, device="cpu",
              image_size=None):
    """One pass. Training when `optimizer` is given, evaluation otherwise.

    `target_fn(tools, task)` chooses the label. It is a required argument, not
    inferred from the loss: all three models share this loader but consume
    different labels, and guessing from the loss class would hand the action
    model the 12-way tool vector while still training and reporting happily.
    """
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    probs, targets = [], []

    with torch.set_grad_enabled(training):
        for frames, tools, task in loader:
            batch = prepare_batch(frames.numpy(), device, image_size)
            target = target_fn(tools, task).to(device)
            logits = model(batch)
            loss = loss_fn(logits, target)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total += float(loss) * batch.shape[0]
            count += batch.shape[0]
            probs.append(logits.detach().float().cpu().numpy())
            targets.append(target.detach().cpu().numpy())

    if count == 0:
        raise ValueError(
            "the loader produced zero batches. Training would report a "
            "meaningless loss of 0.0 and exit successfully.")
    return {"loss": total / count,
            "probs": np.concatenate(probs),
            "targets": np.concatenate(targets)}


def save_checkpoint(path, model, meta):
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path, model):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return payload["meta"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_train.py -v`
Expected: 4 passed

- [ ] **Step 5: Mutate and confirm the tests die**

1. In `prepare_batch`, drop the `[..., ::-1]` reversal → `test_prepare_batch_converts_bgr_to_rgb` must fail.
2. In `prepare_batch`, drop `.div_(255.0)` → `test_prepare_batch_is_channel_first_and_unit_scaled` must fail.
3. In `run_epoch`, delete the `if count == 0: raise` block and return zeros → add a test feeding an empty loader and confirm it fails without the guard.

- [ ] **Step 6: Commit**

```bash
git add src/surgvu/train.py tests/test_train.py
git commit -m "feat: shared training loop, seeding, and checkpoint format"
```

---

### Task 6: Train the tool install-state recogniser

**Files:**
- Create: `scripts/train_tools.py`
- Create: `condor/train.sh`, `condor/train.sub`

**Interfaces:**
- Consumes: everything from Tasks 1–5, `config/splits.json`, `config/tool_frequency.json`.
- Produces: `/staging/n/nkalthoff/surgvu26/models/tools_<backbone>.pt` — checkpoint whose `meta` carries `{"macro_f1", "per_class_f1", "thresholds", "classes", "backbone", "epochs", "frames_per_window", "image_size"}`.

**Why this model first:** it is the only source of tool identity at test time, so its accuracy caps roughly two-thirds of the questions.

- [ ] **Step 1: Write the driver**

```python
"""Train the tool install-state recogniser: 12 independent sigmoids.

NOT top-3. "Three instruments installed" counts arm slots, and two arms carry
the same class in nearly half of sampled moments -- the window-weighted
distribution over distinct classes peaks at 3 with only 44.6% share, so a hard
three-class constraint would be wrong close to half the time.

Thresholds are tuned on validation after training and frozen into the
checkpoint. The corpus is 90x imbalanced (cadiere forceps 15,600 windows,
stapler 137), so pos_weight is mandatory: without it the rare classes are
never predicted and macro-F1 collapses while accuracy looks fine.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.dataset import ShardFrames, shard_paths_for_split   # noqa: E402
from surgvu.metrics import macro_f1, per_class_f1, tune_thresholds  # noqa: E402
from surgvu.models import build_model                            # noqa: E402
from surgvu.taxonomy import TOOL_CLASSES                         # noqa: E402
from surgvu.train import (                                       # noqa: E402
    run_epoch, save_checkpoint, seed_everything,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", default="/staging/groups/bhaskar_opscribe/surgvu/shards")
    parser.add_argument("--splits", default="config/splits.json")
    parser.add_argument("--frequency", default="config/tool_frequency.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--backbone", default="efficientnet_v2_s")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--frames-per-window", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    train_shards = shard_paths_for_split(args.shards, args.splits, "train")
    val_shards = shard_paths_for_split(args.shards, args.splits, "val")
    print("shards: %d train / %d val" % (len(train_shards), len(val_shards)))

    train_set = ShardFrames(train_shards, args.frames_per_window, args.seed, True)
    val_set = ShardFrames(val_shards, args.frames_per_window, args.seed, False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4)

    # pos_weight = (negatives / positives) per class, from the corpus-wide table.
    frequency = json.loads(Path(args.frequency).read_text(encoding="utf-8"))
    total = max(frequency.values())
    weights = np.array([max(total - frequency[c], 1) / max(frequency[c], 1)
                        for c in TOOL_CLASSES], dtype=np.float32)
    weights = np.clip(weights, 1.0, 50.0)      # unclipped, stapler reaches ~110
    print("pos_weight:", dict(zip(TOOL_CLASSES, weights.round(1).tolist())))

    model = build_model(len(TOOL_CLASSES), args.backbone, pretrained=True).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = -1.0
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        take_tools = lambda tools, task: tools
        train_stats = run_epoch(model, train_loader, loss_fn, take_tools,
                                optimizer, device, args.image_size)
        val_stats = run_epoch(model, val_loader, loss_fn, take_tools, None,
                              device, args.image_size)

        probs = 1.0 / (1.0 + np.exp(-val_stats["probs"]))
        thresholds = tune_thresholds(val_stats["targets"], probs)
        pred = (probs >= thresholds).astype(np.float32)
        score = macro_f1(val_stats["targets"], pred)
        print("epoch %d  train_loss %.4f  val_loss %.4f  val_macroF1 %.4f"
              % (epoch, train_stats["loss"], val_stats["loss"], score), flush=True)

        if score > best:
            best = score
            per_class = per_class_f1(val_stats["targets"], pred)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(args.out, model, {
                "macro_f1": score,
                "per_class_f1": dict(zip(TOOL_CLASSES, per_class.tolist())),
                "thresholds": thresholds.tolist(),
                "classes": list(TOOL_CLASSES),
                "backbone": args.backbone,
                "epochs": epoch + 1,
                "frames_per_window": args.frames_per_window,
                "image_size": args.image_size,
            })
            print("  saved (best so far)", flush=True)

    if best <= 0.0:
        raise SystemExit(
            "macro-F1 never exceeded 0.0 across %d epochs. Something is wrong "
            "with the labels or the loader; do not proceed." % args.epochs)
    print("BEST val macro-F1: %.4f" % best)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the Condor job script**

```bash
#!/bin/bash
set -u
# Runs inside surgvu26-train.sif; do not override PATH.
echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 -c "import torch; print('cuda', torch.cuda.is_available())"
umask 002
mkdir -p /staging/n/nkalthoff/surgvu26/models
python3 "$@"
RC=$?
echo "exit: $RC  end: $(date)"
exit $RC
```

- [ ] **Step 3: Write the submit file**

```
universe        = container
container_image = file:///staging/n/nkalthoff/surgvu26/surgvu26-train.sif

executable   = condor/train.sh
arguments    = "$(script) $(args)"

+WantStagingMount = true
transfer_input_files = src, scripts, config

request_cpus   = 8
request_memory = 32GB
request_disk   = 20GB
request_gpus   = 1
# Any modern card trains fine; deployment is a T4, so the FINAL timing and
# numeric validation must be re-run under require_gpus on an RTX 2080 Ti
# (compute capability 7.5, identical to T4).
require_gpus   = (GlobalMemoryMb >= 16000)

output = logs/train_$(Cluster).out
error  = logs/train_$(Cluster).err
log    = logs/train_$(Cluster).log

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
max_retries = 1

queue 1
```

- [ ] **Step 4: Smoke-test on two shards before committing GPU hours**

Run a 1-epoch job with `--epochs 1 --frames-per-window 2` against a `splits.json` copy holding two train cases and one val case. Confirm it prints a device, non-zero shard counts, a finite loss, and saves a checkpoint.
Expected: no exception, `saved (best so far)` appears at least once.

- [ ] **Step 5: Run the real training job**

Run: `condor_submit condor/train.sub` with
`script = scripts/train_tools.py` and
`args = --out /staging/n/nkalthoff/surgvu26/models/tools_efficientnet_v2_s.pt --epochs 8`

Expected: `BEST val macro-F1` printed. **The 2025 winner reported 97% macro-F1 on this data with this backbone** — treat anything below ~0.85 as a bug hunt rather than a tuning exercise, and check first that `pos_weight` is applied and that rare classes are not uniformly zero in `per_class_f1`.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_tools.py condor/train.sh condor/train.sub
git commit -m "feat: train the tool install-state recogniser"
```

---

### Task 7: Train the task classifier and wire description retrieval

**Files:**
- Create: `scripts/train_task.py`
- Modify: none

**Interfaces:**
- Consumes: Tasks 1–5, `surgvu.descriptions.DescriptionRetriever`, `config/descriptions.yaml`.
- Produces: `/staging/n/nkalthoff/surgvu26/models/task_<backbone>.pt` with `meta` carrying `{"accuracy", "macro_f1", "per_class_f1", "description_accuracy", "classes", "backbone"}`.

**Why this matters most for score:** the task class selects the `matched_description`, which is verbatim the text the ground-truth answers were generated from. Two of the 21 descriptions are shared across task classes (one by three, one by four), so **description accuracy is strictly higher than task accuracy** — confusing two classes that share a description still retrieves the right text. Report both.

- [ ] **Step 1: Write the driver**

```python
"""Train the 8-way task classifier and measure DESCRIPTION accuracy too.

Task accuracy is not the number that matters. The task class exists to
retrieve a `matched_description`, and two of the 21 descriptions are shared
across task classes -- so a confusion between two classes sharing a
description costs nothing downstream. Reporting only task accuracy understates
the model exactly where it is already good enough.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.dataset import ShardFrames, shard_paths_for_split   # noqa: E402
from surgvu.descriptions import load_corpus                     # noqa: E402
from surgvu.metrics import macro_f1, multiclass_accuracy, per_class_f1  # noqa: E402
from surgvu.models import build_model                           # noqa: E402
from surgvu.taxonomy import TASK_CLASSES                        # noqa: E402
from surgvu.train import run_epoch, save_checkpoint, seed_everything  # noqa: E402


def description_accuracy(true_idx, pred_idx, corpus):
    """Fraction whose RETRIEVED DESCRIPTION SET is right, not whose class is."""
    hits = 0
    for t, p in zip(true_idx, pred_idx):
        true_set = set(corpus.get(TASK_CLASSES[t], []) or [])
        pred_set = set(corpus.get(TASK_CLASSES[p], []) or [])
        if true_set and true_set == pred_set:
            hits += 1
        elif t == p:
            hits += 1
    return hits / max(len(true_idx), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", default="/staging/groups/bhaskar_opscribe/surgvu/shards")
    parser.add_argument("--splits", default="config/splits.json")
    parser.add_argument("--descriptions", default="config/descriptions.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--backbone", default="efficientnet_v2_s")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--frames-per-window", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = load_corpus(args.descriptions)

    train_shards = shard_paths_for_split(args.shards, args.splits, "train")
    val_shards = shard_paths_for_split(args.shards, args.splits, "val")
    train_set = ShardFrames(train_shards, args.frames_per_window, args.seed, True)
    val_set = ShardFrames(val_shards, args.frames_per_window, args.seed, False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4)

    model = build_model(len(TASK_CLASSES), args.backbone, pretrained=True).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = -1.0
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        take_task = lambda tools, task: task
        train_stats = run_epoch(model, train_loader, loss_fn, take_task,
                                optimizer, device)
        val_stats = run_epoch(model, val_loader, loss_fn, take_task, None, device)

        logits, truth = val_stats["probs"], val_stats["targets"]
        pred = logits.argmax(axis=1)
        accuracy = multiclass_accuracy(truth, logits)
        onehot_t = np.eye(len(TASK_CLASSES), dtype=np.float32)[truth]
        onehot_p = np.eye(len(TASK_CLASSES), dtype=np.float32)[pred]
        score = macro_f1(onehot_t, onehot_p)
        described = description_accuracy(truth, pred, corpus)
        print("epoch %d  train_loss %.4f  val_acc %.4f  macroF1 %.4f  desc_acc %.4f"
              % (epoch, train_stats["loss"], accuracy, score, described), flush=True)

        if score > best:
            best = score
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(args.out, model, {
                "accuracy": accuracy,
                "macro_f1": score,
                "per_class_f1": dict(zip(
                    TASK_CLASSES, per_class_f1(onehot_t, onehot_p).tolist())),
                "description_accuracy": described,
                "classes": list(TASK_CLASSES),
                "backbone": args.backbone,
            })
            print("  saved (best so far)", flush=True)

    if best <= 0.0:
        raise SystemExit("macro-F1 never exceeded 0.0; do not proceed.")
    print("BEST val macro-F1 %.4f" % best)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on two shards**

Same reduced-split procedure as Task 6, Step 4.
Expected: a printed `desc_acc` **greater than or equal to** `val_acc` on every epoch. If it is ever lower, `description_accuracy` is wrong — the shared-description classes can only help.

- [ ] **Step 3: Run the real training job**

Run: `condor_submit condor/train.sub` with `script = scripts/train_task.py` and `args = --out /staging/n/nkalthoff/surgvu26/models/task_efficientnet_v2_s.pt --epochs 8`

- [ ] **Step 4: Commit**

```bash
git add scripts/train_task.py
git commit -m "feat: train the task classifier and report description accuracy"
```

---

### Task 8: Clip-level aggregation

**Files:**
- Create: `src/surgvu/predict.py`
- Test: `tests/test_predict.py`

**Interfaces:**
- Consumes: Tasks 2–5.
- Produces:
  - `aggregate_window(frame_probs) -> np.ndarray` — mean over a window's frames.
  - `predict_window(model, frames, device, image_size) -> np.ndarray` of per-class probabilities for one window.

**Why:** the graded unit is a clip, not a frame. Installation state is constant across a window, so averaging per-frame probabilities is both correct and a free variance reduction over ~30 views of the same label.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from surgvu.predict import aggregate_window


def test_aggregate_window_averages_over_frames():
    frame_probs = np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float32)
    assert aggregate_window(frame_probs).tolist() == [
        pytest.approx(0.3), pytest.approx(0.7)]


def test_aggregate_window_suppresses_a_single_outlier_frame():
    """One frame where the tool is fully occluded must not flip the clip.
    Occlusion is the known blind spot; averaging is the cheap defence."""
    frame_probs = np.vstack([np.full((29, 1), 0.9, dtype=np.float32),
                             np.zeros((1, 1), dtype=np.float32)])
    assert float(aggregate_window(frame_probs)[0]) > 0.85


def test_aggregate_window_rejects_an_empty_stack():
    with pytest.raises(ValueError, match="no frames"):
        aggregate_window(np.zeros((0, 12), dtype=np.float32))
```


- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_predict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'surgvu.predict'`

- [ ] **Step 3: Implement**

```python
"""Frame predictions -> clip predictions.

The graded unit is a 30-second clip. Installation state is constant across a
window by construction, so the 30 frames are 30 independent views of one
label and averaging them is a free variance reduction -- and specifically a
defence against the known blind spot, a tool briefly occluded in some frames.
"""
import numpy as np
import torch

from .train import prepare_batch


def aggregate_window(frame_probs):
    frame_probs = np.asarray(frame_probs, dtype=np.float32)
    if frame_probs.shape[0] == 0:
        raise ValueError("no frames to aggregate; a window must have frames")
    return frame_probs.mean(axis=0)


def predict_window(model, frames, device, image_size=None, activation="sigmoid"):
    """Per-class probabilities for one window's frame stack.

    `activation` is explicit: the tool head is multi-label (sigmoid) and the
    task head is multi-class (softmax). Defaulting one of them silently would
    produce well-formed numbers that do not sum the way the caller assumes.
    """
    model.eval()
    with torch.no_grad():
        batch = prepare_batch(frames, device, image_size)
        logits = model(batch)
        if activation == "sigmoid":
            probs = torch.sigmoid(logits)
        elif activation == "softmax":
            probs = torch.softmax(logits, dim=1)
        else:
            raise ValueError("unknown activation %r" % (activation,))
        probs = probs.float().cpu().numpy()
    return aggregate_window(probs)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_predict.py -v`
Expected: 3 passed

- [ ] **Step 5: Mutate and confirm the tests die**

1. Change `.mean(axis=0)` to `.min(axis=0)` → `test_aggregate_window_suppresses_a_single_outlier_frame` must fail (one occluded frame would sink the clip).
2. Change `.mean(axis=0)` to `frame_probs[0]` → `test_aggregate_window_averages_over_frames` must fail.
3. Delete the empty-stack guard → `test_aggregate_window_rejects_an_empty_stack` must fail.

- [ ] **Step 6: Commit**

```bash
git add src/surgvu/predict.py tests/test_predict.py
git commit -m "feat: clip-level aggregation over a window's frames"
```

---

### Task 9: Action recogniser — and the honest case that it is redundant

**Files:**
- Create: `scripts/train_action.py`

**Interfaces:**
- Consumes: Tasks 1–6, 8, plus the tool checkpoint from Task 6.
- Produces: `/staging/n/nkalthoff/surgvu26/models/action_<backbone>.pt` with `meta` carrying `{"verdict": "keep"|"drop", "accuracy", "majority_baseline", "tool_channel_baseline"}`.

**Read this before writing any code — the design has a problem worth surfacing.**

The spec calls for a temporal CNN answering *"is tissue being cut"*, with labels derived "from the task segments in `tasks.csv` combined with tool presence, since cutting co-occurs with monopolar curved scissors."

But nothing in the corpus states that tissue is being cut. The only available label is a **proxy computed from installed tools** — and `monopolar curved scissors ∈ tools` is *exactly one output channel of the tool recogniser already trained in Task 6*. So:

1. **A model trained on this proxy cannot beat the tool recogniser's own channel**, because it is trained on that channel's label with strictly less capacity devoted to it.
2. **A temporal architecture cannot help.** The proxy label is a function of installation state, which is constant across the window by construction. There is no motion signal in the label for a temporal model to learn. Building a temporal CNN here would be modelling machinery pointed at a label that does not vary in time.

That does not mean skip the task — it means **the gate is against the tool channel, not against a majority-class baseline.** A majority baseline is too weak and would license keeping a model that adds nothing.

**Proxy definition, stated plainly:** `cutting = monopolar curved scissors ∈ tools`. Those scissors are the only cutting instrument among the 12. `vessel sealer` and `stapler` divide tissue but seal rather than cut, so they are excluded — a judgement call, not a fact, recorded here so it is reviewable.

**Expected outcome: `drop`.** Action questions then route to dense re-sampling in Plan 3, and the tool channel answers "is cutting happening" directly for free. Reaching `drop` quickly is the cheapest good result in this plan.

- [ ] **Step 1: Write the driver**

```python
"""Train the action recogniser and decide whether it earns its place.

The label is a PROXY: nothing in the corpus says tissue is being cut, so
`cutting` is derived from installed tools. That proxy is exactly one channel
of the tool recogniser, which is why the gate below compares against that
channel rather than a majority-class baseline. Beating "always predict the
majority" would prove nothing; beating a model we already have is the only
result that justifies a fourth model in the container.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.dataset import ShardFrames, shard_paths_for_split   # noqa: E402
from surgvu.models import build_model                           # noqa: E402
from surgvu.taxonomy import TOOL_CLASSES                        # noqa: E402
from surgvu.train import (                                      # noqa: E402
    load_checkpoint, run_epoch, save_checkpoint, seed_everything,
)

CUTTER = TOOL_CLASSES.index("monopolar curved scissors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", default="/staging/groups/bhaskar_opscribe/surgvu/shards")
    parser.add_argument("--splits", default="config/splits.json")
    parser.add_argument("--tool-checkpoint", required=True,
                        help="Task 6 output; supplies the baseline to beat")
    parser.add_argument("--out", required=True)
    parser.add_argument("--backbone", default="efficientnet_v2_s")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--frames-per-window", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_shards = shard_paths_for_split(args.shards, args.splits, "train")
    val_shards = shard_paths_for_split(args.shards, args.splits, "val")
    train_set = ShardFrames(train_shards, args.frames_per_window, args.seed, True)
    val_set = ShardFrames(val_shards, args.frames_per_window, args.seed, False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4)

    # The proxy label: one column of the tool vector, kept 2-D for BCE.
    take_cutting = lambda tools, task: tools[:, CUTTER:CUTTER + 1]

    model = build_model(1, args.backbone, pretrained=True).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best, best_stats = -1.0, None
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        train_stats = run_epoch(model, train_loader, loss_fn, take_cutting,
                                optimizer, device)
        val_stats = run_epoch(model, val_loader, loss_fn, take_cutting, None, device)

        probs = 1.0 / (1.0 + np.exp(-val_stats["probs"]))
        truth = val_stats["targets"].reshape(-1)
        accuracy = float(((probs.reshape(-1) >= 0.5) == (truth >= 0.5)).mean())
        print("epoch %d  train_loss %.4f  val_acc %.4f"
              % (epoch, train_stats["loss"], accuracy), flush=True)
        if accuracy > best:
            best, best_stats = accuracy, (truth, probs.reshape(-1))

    truth, _ = best_stats
    majority = float(max(truth.mean(), 1.0 - truth.mean()))

    # The baseline that actually matters: read the answer off the tool
    # recogniser we already have, on the same validation windows.
    tool_model = build_model(len(TOOL_CLASSES), args.backbone, pretrained=False)
    tool_meta = load_checkpoint(args.tool_checkpoint, tool_model)
    tool_model = tool_model.to(device)
    tool_stats = run_epoch(tool_model, val_loader,
                           torch.nn.BCEWithLogitsLoss(),
                           lambda tools, task: tools, None, device)
    tool_probs = 1.0 / (1.0 + np.exp(-tool_stats["probs"][:, CUTTER]))
    tool_truth = tool_stats["targets"][:, CUTTER]
    threshold = tool_meta["thresholds"][CUTTER]
    tool_channel = float(((tool_probs >= threshold) == (tool_truth >= 0.5)).mean())

    verdict = "keep" if best > tool_channel + 0.02 else "drop"
    print("\naction model      : %.4f" % best)
    print("majority baseline : %.4f" % majority)
    print("tool channel      : %.4f   <- the baseline that matters" % tool_channel)
    print("VERDICT: %s" % verdict.upper())
    if verdict == "drop":
        print("The tool recogniser already answers this. Action questions route "
              "to dense re-sampling in Plan 3. This is an expected outcome, not "
              "a failure -- it removes a model from the 16 GiB budget.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.out, model, {
        "verdict": verdict,
        "accuracy": best,
        "majority_baseline": majority,
        "tool_channel_baseline": tool_channel,
        "proxy_label": "monopolar curved scissors in tools",
        "backbone": args.backbone,
    })


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `condor_submit condor/train.sub` with
`script = scripts/train_action.py` and
`args = --tool-checkpoint /staging/n/nkalthoff/surgvu26/models/tools_efficientnet_v2_s.pt --out /staging/n/nkalthoff/surgvu26/models/action_efficientnet_v2_s.pt`

Expected: all three numbers printed and a verdict. **If the verdict is `keep`, be suspicious** — check that the tool checkpoint loaded (its `meta["thresholds"]` should have 12 entries) and that both models saw the same validation windows. A genuine `keep` would mean the dedicated model extracts something the shared backbone did not, which is possible but is the surprising result, not the expected one.

- [ ] **Step 3: Record the verdict where Plan 3 will read it**

Add one line to `docs/OUTSTANDING.md` giving the accuracy, the tool-channel baseline, and the decision. Plan 3's router branches on this, and the methodology report should state that a component was measured and dropped — that is a stronger result than never having tried it.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_action.py docs/OUTSTANDING.md
git commit -m "feat: action recogniser gated against the tool recogniser channel"
```

---

### Task 10: Combined validation report

**Files:**
- Create: `scripts/report_val.py`

**Interfaces:**
- Consumes: the three checkpoints.
- Produces: `docs/plan2-validation.md` plus `config/perception.json` — the frozen artefact Plan 3's router reads: `{"tools": {"checkpoint", "thresholds", "macro_f1", "per_class_f1"}, "task": {...}, "action": {"verdict", ...}}`.

- [ ] **Step 1: Write the reporter**

It must load each checkpoint, read `meta`, and emit both files. It must **fail loudly** if a checkpoint is missing rather than writing a partial config:

```python
    for name, path in checkpoints.items():
        if not Path(path).exists():
            raise SystemExit(
                "%s checkpoint missing at %s. Refusing to write a partial "
                "perception config: Plan 3 would route to a model that does "
                "not exist." % (name, path))
```

- [ ] **Step 2: Run it and read the output**

Run: `python3 scripts/report_val.py`
Expected: `config/perception.json` exists and names all three experts; `docs/plan2-validation.md` shows per-class F1 for all 12 tool classes.

- [ ] **Step 3: Check the rare classes explicitly**

Look at `per_class_f1` for `stapler` (137 train windows) and `tip-up fenestrated grasper` (157). If either is 0.0, the threshold tuning or `pos_weight` failed and macro-F1 is being carried by the common classes. This is the single most likely silent failure in Plan 2.

- [ ] **Step 4: Commit**

```bash
git add scripts/report_val.py config/perception.json docs/plan2-validation.md
git commit -m "feat: combined perception validation report for Plan 3"
```

---

## What Plan 2 deliberately does not do

- **No detector / YOLO.** A detector predicts *visible* instruments; ground truth is *installed*. The eyeball check confirmed three-listed/two-visible is routine, so a perfect detector would systematically under-report. It is also the only component needing hand-drawn boxes. Build the CNN first, measure how often the router would even call a detector, and annotate only if that number is large. See the licensing note below if it is.
- **No VLM work.** Plan 3.
- **No auxiliary out-of-scope classes.** They are not in the shards and recovering them means re-querying `tools.csv`. Ablation only.
- **No per-arm formulation.** The heads are not identifiable — nothing in the image says which arm an instrument is on — and the exactly-three premise it relies on does not hold. Ablation only.

## Licensing note for any later detector work

Ultralytics YOLOv5/v8/YOLO11 are **AGPL-3.0**. This repo goes public at submission, so publishing is fine, but AGPL would require the entire distributed work to be AGPL and commercial use needs a paid licence. Permissive alternatives: **YOLOX** (Apache-2.0), **RT-DETR** (Apache-2.0), torchvision detection (BSD-3). Settle the licence question before benchmarking versions — it may eliminate most candidates. Everything in Plan 2 uses torchvision, which is BSD-3.
