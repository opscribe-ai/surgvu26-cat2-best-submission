# SurgVU 2026 Cat 2 — Foundation Implementation Plan


**Goal:** Build the label engine, preprocessing, training-corpus extractor, and scoring harness — everything the CNNs and the inference container both depend on.

**Architecture:** A small Python package `surgvu` with four independent modules. `taxonomy` owns the closed vocabularies. `labels` turns the two challenge CSVs into a queryable interval structure. `preprocess` handles frame geometry adaptively. `extract` samples 30-second windows and writes training shards. `scoring` wraps the official evaluation metric so every later measurement is the real number.

**Tech Stack:** Python 3.10, pytest, PyYAML, opencv-python-headless, numpy. `bert-score` + `roberta-large` for the scoring harness only.

## Global Constraints

- **Python 3.10** — matches the challenge's reference container.
- **12 tool classes only.** `needle driver`, `monopolar curved scissors`, `force bipolar`, `clip applier`, `cadiere forceps`, `bipolar forceps`, `vessel sealer`, `permanent cautery hook/spatula`, `prograsp forceps`, `stapler`, `grasping retractor`, `tip-up fenestrated grasper`. Everything else is train-only signal, never predicted.
- **8 task classes.** `suturing`, `uterine horn`, `suspensory ligaments`, `rectal artery/vein`, `skills application`, `range of motion`, `retraction and collision avoidance`, `other`. Raw labels vary in case; normalize by lowercasing.
- **Using the UI overlay to make predictions is prohibited by challenge rules.** Every frame that reaches a model must have the bottom UI band blurred. This is compliance, not a hyperparameter.
- **Never train or tune on `cat1_test_set_public.zip` contents.** It may be read for geometry measurement only.
- **No artifact whose training history includes private clinical data may enter this repo.** Public datasets only.
- **Split by case, never by frame.** Frames from one session are near-duplicates.
- `tools.csv` times are `HH:MM:SS.ffffff` **strings**; `tasks.csv` times are **float seconds**. They are different formats in the same dataset.
- Cases have up to two video **parts** and timestamps reset at the boundary. Both CSVs carry `*_part` columns.

---

### Task 1: Package scaffolding and the closed vocabularies

**Files:**
- Create: `src/surgvu/__init__.py`
- Create: `src/surgvu/taxonomy.py`
- Create: `tests/test_taxonomy.py`
- Create: `pyproject.toml`
- Create: `.gitignore`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `TOOL_CLASSES: tuple[str, ...]` — the 12, in fixed index order
  - `TASK_CLASSES: tuple[str, ...]` — the 8, in fixed index order
  - `normalize_task(raw: str) -> str | None`
  - `normalize_tool(raw: str) -> str | None` — returns `None` for out-of-scope and for the endoscope
  - `tool_index(name: str) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_taxonomy.py
import pytest
from surgvu.taxonomy import (
    TOOL_CLASSES, TASK_CLASSES, normalize_task, normalize_tool, tool_index,
)


def test_twelve_tool_classes_in_fixed_order():
    assert len(TOOL_CLASSES) == 12
    assert TOOL_CLASSES[0] == "bipolar forceps"
    assert TOOL_CLASSES == tuple(sorted(TOOL_CLASSES))


def test_eight_task_classes():
    assert len(TASK_CLASSES) == 8
    assert "suturing" in TASK_CLASSES
    assert "other" in TASK_CLASSES


def test_normalize_task_lowercases():
    assert normalize_task("Suturing") == "suturing"
    assert normalize_task("suturing") == "suturing"
    assert normalize_task("Rectal artery/vein") == "rectal artery/vein"


def test_normalize_task_rejects_unknown():
    assert normalize_task("not a task") is None
    assert normalize_task("") is None


def test_normalize_tool_passes_in_scope():
    assert normalize_tool("needle driver") == "needle driver"
    assert normalize_tool("Needle Driver") == "needle driver"


def test_normalize_tool_rejects_endoscope():
    # The camera is not a predictable class and must never enter a label.
    assert normalize_tool("nan(camera in)") is None


def test_normalize_tool_rejects_out_of_scope_rare_classes():
    for rare in ["suction irrigator", "synchroseal", "curved scissors",
                 "potts scissors", "tenaculum forceps", "bipolar dissector",
                 "crocodile grasper"]:
        assert normalize_tool(rare) is None, rare


def test_tool_index_is_stable():
    for i, name in enumerate(TOOL_CLASSES):
        assert tool_index(name) == i


def test_tool_index_raises_on_unknown():
    with pytest.raises(KeyError):
        tool_index("suction irrigator")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_taxonomy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'surgvu'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/surgvu/__init__.py
"""SurgVU 2026 Category 2 — surgical VQA."""
__version__ = "0.1.0"
```

```python
# src/surgvu/taxonomy.py
"""Closed vocabularies for SurgVU Category 2.

Only these 12 tool classes appear in the test set. The label tables contain
seven rarer classes plus the endoscope; the challenge states those "will not
be part of the testing set", so they are training signal only and must never
be emitted as a prediction.
"""

TOOL_CLASSES = (
    "bipolar forceps",
    "cadiere forceps",
    "clip applier",
    "force bipolar",
    "grasping retractor",
    "monopolar curved scissors",
    "needle driver",
    "permanent cautery hook/spatula",
    "prograsp forceps",
    "stapler",
    "tip-up fenestrated grasper",
    "vessel sealer",
)

TASK_CLASSES = (
    "other",
    "range of motion",
    "rectal artery/vein",
    "retraction and collision avoidance",
    "skills application",
    "suspensory ligaments",
    "suturing",
    "uterine horn",
)

# Present in the label tables, excluded from the test set.
OUT_OF_SCOPE_TOOLS = frozenset({
    "bipolar dissector",
    "crocodile grasper",
    "curved scissors",
    "potts scissors",
    "suction irrigator",
    "synchroseal",
    "tenaculum forceps",
})

_TOOL_INDEX = {name: i for i, name in enumerate(TOOL_CLASSES)}
_TOOL_SET = frozenset(TOOL_CLASSES)
_TASK_SET = frozenset(TASK_CLASSES)


def normalize_tool(raw):
    """Map a raw groundtruth_toolname to a test-relevant class, or None.

    Returns None for the endoscope (recorded as 'nan(camera in)'), for the
    seven out-of-scope rare classes, and for anything unrecognised.
    """
    if not raw:
        return None
    name = raw.strip().lower()
    if name.startswith("nan"):          # endoscope / camera
        return None
    if name in OUT_OF_SCOPE_TOOLS:
        return None
    return name if name in _TOOL_SET else None


def normalize_task(raw):
    """Map a raw groundtruth_taskname to one of the 8 classes, or None."""
    if not raw:
        return None
    name = raw.strip().lower()
    return name if name in _TASK_SET else None


def tool_index(name):
    """Index of a tool class in TOOL_CLASSES. Raises KeyError if out of scope."""
    return _TOOL_INDEX[name]
```

```toml
# pyproject.toml
[project]
name = "surgvu"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "PyYAML",
    "opencv-python-headless",
]

[project.optional-dependencies]
scoring = ["bert-score", "torch"]
dev = ["pytest"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

```gitignore
# .gitignore
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
data/
shards/
model_cache/
*.zip
*.mp4
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_taxonomy.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore src/surgvu/__init__.py src/surgvu/taxonomy.py tests/test_taxonomy.py
git commit -m "feat: package scaffolding and closed tool/task vocabularies"
```

---

### Task 2: Label engine

**Files:**
- Create: `src/surgvu/labels.py`
- Create: `tests/test_labels.py`
- Create: `tests/fixtures/case_test/tools.csv`
- Create: `tests/fixtures/case_test/tasks.csv`

**Interfaces:**
- Consumes: `surgvu.taxonomy.normalize_tool`, `normalize_task`
- Produces:
  - `parse_hms(text: str) -> float | None` — `"00:41:01.008000"` → seconds
  - `CaseLabels.from_dir(path) -> CaseLabels`
  - `CaseLabels.tools_at(part: str, seconds: float) -> set[str]`
  - `CaseLabels.task_at(part: str, seconds: float) -> tuple[str, str] | None` — `(task_class, matched_description)`
  - `CaseLabels.task_segments() -> list[Segment]` where `Segment` is a NamedTuple `(part: str, start: float, stop: float, task: str, description: str)`
  - `load_all_cases(root) -> dict[str, CaseLabels]`

- [ ] **Step 1: Write the failing test**

Create the fixtures first. These reproduce the real quirks: duplicate task rows, a part-2 segment, an out-of-scope tool, and the endoscope.

```csv
# tests/fixtures/case_test/tools.csv
install_case_part,install_case_time,uninstall_case_part,uninstall_case_time,arm,commercial_toolname,groundtruth_toolname,case
1.0,00:00:10.000000,1.0,00:10:00.000000,USM1,Large Needle Driver,needle driver,case_test
1.0,00:00:10.000000,1.0,00:10:00.000000,USM3,Monopolar Curved Scissors,monopolar curved scissors,case_test
1.0,00:00:10.000000,1.0,00:10:00.000000,USM4,Cadiere Forceps,cadiere forceps,case_test
1.0,00:00:00.000000,1.0,00:20:00.000000,USM2,30 Endoscope,nan(camera in),case_test
1.0,00:05:00.000000,1.0,00:06:00.000000,USM1,EndoWrist Suction Irrigator,suction irrigator,case_test
2.0,00:00:05.000000,2.0,00:09:00.000000,USM1,Large Clip Applier,clip applier,case_test
```

```csv
# tests/fixtures/case_test/tasks.csv
index,start_part,start_time,stop_part,stop_time,duration,taskname,groundtruth_taskname,matched_description,case
0,1.0,60.0,1.0,300.0,240.0,Skills Drills,Suturing,"Excess bleeding occurs which requires surgeon to pause",case_test
0,1.0,60.0,1.0,300.0,240.0,Skills Drills,Suturing,"Excess bleeding occurs which requires surgeon to pause",case_test
1,2.0,100.0,2.0,400.0,300.0,Training Topic 4,uterine horn,"Surgeon begins by tracing the right ovary",case_test
2,1.0,500.0,2.0,50.0,0.0,Spanning,suturing,"Excess bleeding occurs which requires surgeon to pause",case_test
```

```python
# tests/test_labels.py
from pathlib import Path
from surgvu.labels import parse_hms, CaseLabels

FIXTURE = Path(__file__).parent / "fixtures" / "case_test"


def test_parse_hms():
    assert parse_hms("00:41:01.008000") == 41 * 60 + 1.008
    assert parse_hms("01:00:00.000000") == 3600.0
    assert parse_hms("") is None
    assert parse_hms("garbage") is None


def test_tools_at_returns_only_in_scope_classes():
    c = CaseLabels.from_dir(FIXTURE)
    tools = c.tools_at("1.0", 120.0)
    # endoscope excluded, suction irrigator not yet installed at t=120
    assert tools == {"needle driver", "monopolar curved scissors", "cadiere forceps"}


def test_tools_at_excludes_out_of_scope_even_when_installed():
    c = CaseLabels.from_dir(FIXTURE)
    # at t=330s the suction irrigator IS installed but is out of scope
    tools = c.tools_at("1.0", 330.0)
    assert "suction irrigator" not in tools
    assert tools == {"needle driver", "monopolar curved scissors", "cadiere forceps"}


def test_tools_at_respects_part_boundary():
    c = CaseLabels.from_dir(FIXTURE)
    # clip applier is installed in part 2 only; t=100 in part 1 must not see it
    assert "clip applier" not in c.tools_at("1.0", 100.0)
    assert c.tools_at("2.0", 100.0) == {"clip applier"}


def test_tools_at_outside_any_interval_is_empty():
    c = CaseLabels.from_dir(FIXTURE)
    assert c.tools_at("1.0", 5.0) == set()


def test_task_at_normalizes_case():
    c = CaseLabels.from_dir(FIXTURE)
    task, desc = c.task_at("1.0", 120.0)
    assert task == "suturing"          # raw was "Suturing"
    assert desc.startswith("Excess bleeding")


def test_task_at_outside_segments_is_none():
    c = CaseLabels.from_dir(FIXTURE)
    assert c.task_at("1.0", 10.0) is None


def test_task_segments_are_deduplicated():
    c = CaseLabels.from_dir(FIXTURE)
    segs = c.task_segments()
    # 4 raw rows -> 1 duplicate removed, 1 part-spanning row dropped
    assert len(segs) == 2
    assert sum(1 for s in segs if s.part == "1.0") == 1
    assert sum(1 for s in segs if s.part == "2.0") == 1


def test_task_segments_drop_part_spanning():
    c = CaseLabels.from_dir(FIXTURE)
    for s in c.task_segments():
        assert s.stop > s.start
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'surgvu.labels'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/surgvu/labels.py
"""Turn the two challenge CSVs into queryable, part-aware interval structures.

Three quirks in the real data drive this module's design:
  1. tools.csv times are 'HH:MM:SS.ffffff' strings; tasks.csv times are float
     seconds. Same dataset, different formats.
  2. tasks.csv contains ~50% duplicate rows (3,673 raw -> 1,850 unique).
     tools.csv is clean by comparison.
  3. Cases split into up to two video parts and timestamps RESET at the
     boundary. Segments spanning the boundary have stop < start and are
     unusable without part durations, so they are dropped.
"""
import csv
from collections import namedtuple
from pathlib import Path

from .taxonomy import normalize_task, normalize_tool

Segment = namedtuple("Segment", "part start stop task description")
_Interval = namedtuple("_Interval", "part start stop tool")


def parse_hms(text):
    """'00:41:01.008000' -> 2461.008 seconds. Returns None if unparseable."""
    if not text:
        return None
    try:
        hours, minutes, seconds = text.strip().split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (ValueError, AttributeError):
        return None


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _part(value):
    return (value or "").strip()


class CaseLabels:
    """Labels for a single case, queryable by (part, seconds)."""

    def __init__(self, case_id, intervals, segments):
        self.case_id = case_id
        self._intervals = intervals
        self._segments = segments

    @classmethod
    def from_dir(cls, path):
        path = Path(path)
        return cls(
            case_id=path.name,
            intervals=cls._load_tools(path / "tools.csv"),
            segments=cls._load_tasks(path / "tasks.csv"),
        )

    @staticmethod
    def _load_tools(path):
        intervals = []
        seen = set()
        for row in _read_csv(path):
            key = (row.get("install_case_part"), row.get("install_case_time"),
                   row.get("uninstall_case_part"), row.get("uninstall_case_time"),
                   row.get("arm"), row.get("groundtruth_toolname"))
            if key in seen:
                continue
            seen.add(key)

            tool = normalize_tool(row.get("groundtruth_toolname"))
            if tool is None:                     # endoscope or out of scope
                continue
            part_in = _part(row.get("install_case_part"))
            if part_in != _part(row.get("uninstall_case_part")):
                continue                         # spans a part boundary
            start = parse_hms(row.get("install_case_time"))
            stop = parse_hms(row.get("uninstall_case_time"))
            if start is None or stop is None or stop <= start:
                continue
            intervals.append(_Interval(part_in, start, stop, tool))
        return intervals

    @staticmethod
    def _load_tasks(path):
        segments = []
        seen = set()
        for row in _read_csv(path):
            key = (row.get("start_part"), row.get("start_time"),
                   row.get("stop_part"), row.get("stop_time"),
                   row.get("groundtruth_taskname"))
            if key in seen:
                continue
            seen.add(key)

            task = normalize_task(row.get("groundtruth_taskname"))
            if task is None:
                continue
            part = _part(row.get("start_part"))
            if part != _part(row.get("stop_part")):
                continue                         # spans a part boundary
            try:
                start = float(row["start_time"])
                stop = float(row["stop_time"])
            except (KeyError, TypeError, ValueError):
                continue
            if stop <= start:
                continue
            description = (row.get("matched_description") or "").strip()
            segments.append(Segment(part, start, stop, task, description))
        return segments

    def tools_at(self, part, seconds):
        """Set of in-scope tool classes installed at this moment."""
        part = _part(part)
        return {iv.tool for iv in self._intervals
                if iv.part == part and iv.start <= seconds <= iv.stop}

    def task_at(self, part, seconds):
        """(task_class, matched_description) covering this moment, or None.

        Where segments overlap, the shortest wins — it is the most specific.
        """
        part = _part(part)
        hits = [s for s in self._segments
                if s.part == part and s.start <= seconds <= s.stop]
        if not hits:
            return None
        best = min(hits, key=lambda s: s.stop - s.start)
        return best.task, best.description

    def task_segments(self):
        return list(self._segments)


def load_all_cases(root):
    """Load every case directory under root into {case_id: CaseLabels}."""
    root = Path(root)
    cases = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if not (child / "tools.csv").exists():
            continue
        cases[child.name] = CaseLabels.from_dir(child)
    return cases
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_labels.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Verify against the real label set**

Run:
```bash
python -c "
from surgvu.labels import load_all_cases
cases = load_all_cases('data/SURGVU25_train_labels')
print('cases:', len(cases))
segs = sum(len(c.task_segments()) for c in cases.values())
print('deduped task segments:', segs)
descs = {s.description for c in cases.values() for s in c.task_segments()}
print('unique descriptions:', len(descs))
"
```
Expected: `cases: 155`, roughly `1500-1900` segments, and **`unique descriptions: 21`** — the 21 is the load-bearing number. If it is not 21, the dedupe or normalization is wrong; stop and fix before continuing.

- [ ] **Step 6: Commit**

```bash
git add src/surgvu/labels.py tests/test_labels.py tests/fixtures/
git commit -m "feat: part-aware label engine over tools.csv and tasks.csv"
```

---

### Task 3: Description corpus and retrieval

**Files:**
- Create: `src/surgvu/descriptions.py`
- Create: `tests/test_descriptions.py`
- Create: `scripts/build_descriptions.py`

**Interfaces:**
- Consumes: `surgvu.labels.load_all_cases`, `surgvu.taxonomy.TASK_CLASSES`
- Produces:
  - `build_corpus(cases: dict) -> dict` — `{task_class: [description, ...]}` ordered by frequency
  - `write_corpus(corpus, path)` / `load_corpus(path) -> dict`
  - `DescriptionRetriever(corpus).retrieve(task_class: str) -> str` — the modal description for that task

- [ ] **Step 1: Write the failing test**

```python
# tests/test_descriptions.py
from surgvu.descriptions import build_corpus, DescriptionRetriever, write_corpus, load_corpus
from surgvu.labels import CaseLabels
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "case_test"


def test_build_corpus_groups_by_task():
    cases = {"case_test": CaseLabels.from_dir(FIXTURE)}
    corpus = build_corpus(cases)
    assert "suturing" in corpus
    assert "uterine horn" in corpus
    assert corpus["suturing"][0].startswith("Excess bleeding")


def test_retriever_returns_modal_description():
    corpus = {"suturing": ["most common", "rare one"]}
    r = DescriptionRetriever(corpus)
    assert r.retrieve("suturing") == "most common"


def test_retriever_unknown_task_returns_empty_string():
    r = DescriptionRetriever({"suturing": ["x"]})
    assert r.retrieve("range of motion") == ""


def test_corpus_roundtrips_through_yaml(tmp_path):
    corpus = {"suturing": ["a", "b"], "other": ["c"]}
    path = tmp_path / "descriptions.yaml"
    write_corpus(corpus, path)
    assert load_corpus(path) == corpus
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_descriptions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'surgvu.descriptions'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/surgvu/descriptions.py
"""The 21-string description corpus.

Across all 155 cases there are exactly 21 unique matched_description values.
This is not a training set — it is a closed retrieval corpus, and it is the
verbatim text the challenge's ground-truth answers were generated from.
Classify the task, retrieve the string, and you hold the source material.
"""
from collections import Counter, defaultdict

import yaml


def build_corpus(cases):
    """{task_class: [description, ...]} ordered most-frequent first."""
    counters = defaultdict(Counter)
    for case in cases.values():
        for segment in case.task_segments():
            if segment.description:
                counters[segment.task][segment.description] += 1
    return {task: [desc for desc, _ in counter.most_common()]
            for task, counter in counters.items()}


def write_corpus(corpus, path):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(corpus, handle, allow_unicode=True, sort_keys=True,
                       default_flow_style=False, width=100)


def load_corpus(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class DescriptionRetriever:
    """Task class -> the description the graders most likely read."""

    def __init__(self, corpus):
        self._corpus = corpus

    def retrieve(self, task_class):
        entries = self._corpus.get(task_class) or []
        return entries[0] if entries else ""
```

```python
# scripts/build_descriptions.py
"""Generate config/descriptions.yaml from the challenge label tables."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.descriptions import build_corpus, write_corpus  # noqa: E402
from surgvu.labels import load_all_cases  # noqa: E402

if __name__ == "__main__":
    labels_root = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("config/descriptions.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)

    cases = load_all_cases(labels_root)
    corpus = build_corpus(cases)
    write_corpus(corpus, out)

    total = sum(len(v) for v in corpus.values())
    print("cases: %d" % len(cases))
    print("tasks: %d" % len(corpus))
    print("unique descriptions: %d" % total)
    for task in sorted(corpus):
        print("  %-38s %d" % (task, len(corpus[task])))
    if total != 21:
        print("\nWARNING: expected 21 unique descriptions, got %d" % total)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_descriptions.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Generate the real corpus**

Run: `python scripts/build_descriptions.py data/SURGVU25_train_labels config/descriptions.yaml`
Expected: `unique descriptions: 21` with no warning. `suturing` should have the most entries (around 12); `uterine horn`, `suspensory ligaments`, and `range of motion` should have 1 each.

- [ ] **Step 6: Commit**

```bash
git add src/surgvu/descriptions.py tests/test_descriptions.py scripts/build_descriptions.py config/descriptions.yaml
git commit -m "feat: 21-string description corpus and retriever"
```

---

### Task 4: Adaptive preprocessing

**Files:**
- Create: `src/surgvu/preprocess.py`
- Create: `tests/test_preprocess.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `detect_side_margins(frame: np.ndarray, threshold: int = 12) -> tuple[int, int]`
  - `crop_side_margins(frame) -> np.ndarray`
  - `blur_ui_band(frame, band_fraction: float = 0.0625, kernel: int = 51) -> np.ndarray`
  - `prepare_frame(frame, size: int = 512) -> np.ndarray` — the single entry point every model uses

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preprocess.py
import numpy as np
from surgvu.preprocess import (
    UI_BAND_FRACTION, detect_side_margins, crop_side_margins, blur_ui_band,
    prepare_frame,
)


def _frame_with_margins(width=1280, height=720, margin=192):
    f = np.zeros((height, width, 3), dtype=np.uint8)
    f[:, margin:width - margin, :] = 200          # bright content
    return f


def test_detect_side_margins_finds_black_bars():
    left, right = detect_side_margins(_frame_with_margins())
    assert left == 192
    assert right == 192


def test_detect_side_margins_returns_zero_when_already_cropped():
    f = np.full((512, 640, 3), 200, dtype=np.uint8)
    assert detect_side_margins(f) == (0, 0)


def test_crop_removes_margins_only_when_present():
    cropped = crop_side_margins(_frame_with_margins())
    assert cropped.shape[1] == 1280 - 384

    already = np.full((512, 640, 3), 200, dtype=np.uint8)
    assert crop_side_margins(already).shape[1] == 640


def test_blur_ui_band_changes_bottom_and_preserves_top():
    f = np.random.RandomState(0).randint(0, 255, (720, 896, 3), dtype=np.uint8)
    out = blur_ui_band(f)
    band = int(round(720 * UI_BAND_FRACTION))
    assert np.array_equal(out[: 720 - band], f[: 720 - band])   # top untouched
    assert not np.array_equal(out[720 - band:], f[720 - band:])  # bottom changed


def test_blur_ui_band_reduces_bottom_variance():
    f = np.random.RandomState(1).randint(0, 255, (720, 896, 3), dtype=np.uint8)
    out = blur_ui_band(f)
    band = int(round(720 * UI_BAND_FRACTION))
    assert out[720 - band:].var() < f[720 - band:].var() / 2


def test_band_is_wider_than_the_measured_overlay():
    # The bar is ~45 px of 720 (0.0625). We deliberately blur more, because
    # under-blurring leaves readable text and reintroduces shortcut learning,
    # while over-blurring costs a sliver of mostly-black bottom edge.
    assert UI_BAND_FRACTION > 0.0625


def test_reblurring_is_stable():
    # Test clips arrive already blurred by the organizers, so our blur runs on
    # top of theirs. Re-applying must not error, reshape, or keep degrading:
    # the second application should change far less than the first.
    f = np.random.RandomState(2).randint(0, 255, (512, 640, 3), dtype=np.uint8)
    once = blur_ui_band(f)
    twice = blur_ui_band(once)
    assert twice.shape == once.shape
    assert twice.dtype == once.dtype
    first_delta = np.abs(once.astype(int) - f.astype(int)).mean()
    second_delta = np.abs(twice.astype(int) - once.astype(int)).mean()
    assert second_delta < first_delta / 2


def test_prepare_frame_is_square_and_blurred():
    out = prepare_frame(_frame_with_margins(), size=512)
    assert out.shape == (512, 512, 3)
    assert out.dtype == np.uint8


def test_prepare_frame_handles_test_geometry():
    # 640x512 at 1 fps is the Cat 1 test-set geometry; must not crash.
    f = np.full((512, 640, 3), 180, dtype=np.uint8)
    assert prepare_frame(f).shape == (512, 512, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'surgvu.preprocess'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/surgvu/preprocess.py
"""Frame preparation, identical at training and inference.

Blurring the bottom UI band is REQUIRED BY CHALLENGE RULES, not a tuning
choice: "using the information available in the UI to make predictions is not
allowed. To enforce this, the UI will be blurred from the test set". A model
trained on unblurred frames uses UI information whether or not that was the
intent, and would collapse at test where the band is gone.

Geometry is detected rather than assumed. Observed formats differ:
  Cat 2 sample clips   1280x720 @ 60 fps, black side margins present
  Cat 1 test clips      640x512 @  1 fps, already cropped
so cropping is conditional and the band is expressed as a fraction of height.
"""
import cv2
import numpy as np

UI_BAND_FRACTION = 0.08        # 45 px of 720 is 0.0625; widened for margin
BLUR_KERNEL = 51


def detect_side_margins(frame, threshold=12):
    """(left, right) width in pixels of near-black vertical margins."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    column_mean = grey.mean(axis=0)
    bright = column_mean > threshold
    if not bright.any():
        return 0, 0
    left = int(np.argmax(bright))
    right = int(np.argmax(bright[::-1]))
    return left, right


def crop_side_margins(frame):
    """Remove black side bars if present; return unchanged if already cropped."""
    left, right = detect_side_margins(frame)
    if left == 0 and right == 0:
        return frame
    width = frame.shape[1]
    return frame[:, left:width - right, :]


def blur_ui_band(frame, band_fraction=UI_BAND_FRACTION, kernel=BLUR_KERNEL):
    """Gaussian-blur the bottom band where the instrument UI is rendered.

    Safe to apply to an already-blurred frame — blurring is idempotent enough
    that re-applying costs nothing and guarantees compliance regardless of
    what the organizers shipped.
    """
    height = frame.shape[0]
    band = max(1, int(round(height * band_fraction)))
    out = frame.copy()
    k = kernel if kernel % 2 == 1 else kernel + 1
    out[height - band:] = cv2.GaussianBlur(out[height - band:], (k, k), 0)
    return out


def prepare_frame(frame, size=512):
    """The single entry point. Crop, blur, resize — in that order, always."""
    frame = crop_side_margins(frame)
    frame = blur_ui_band(frame)
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_CUBIC)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_preprocess.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Visual verification on a real clip**

Run:
```bash
python -c "
import cv2, sys; sys.path.insert(0,'src')
from surgvu.preprocess import prepare_frame
cap = cv2.VideoCapture('data/samples/case122.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 900); ok, f = cap.read(); cap.release()
cv2.imwrite('/tmp/prepared.png', prepare_frame(f))
print('wrote /tmp/prepared.png', f.shape)
"
```
Open the image. **The instrument names at the bottom must be unreadable.** If any text is legible, increase `UI_BAND_FRACTION` until it is not. This is a rules-compliance check, not an aesthetic one.

- [ ] **Step 6: Commit**

```bash
git add src/surgvu/preprocess.py tests/test_preprocess.py
git commit -m "feat: adaptive frame preprocessing with mandatory UI blur"
```

---

### Task 5: Window sampling and canonical splits

**Files:**
- Create: `src/surgvu/sampling.py`
- Create: `tests/test_sampling.py`
- Create: `scripts/build_splits.py`

**Interfaces:**
- Consumes: `surgvu.labels.CaseLabels`, `surgvu.taxonomy.TOOL_CLASSES`
- Produces:
  - `Window = namedtuple("Window", "case part start task description tools")`
  - `enumerate_windows(case_id, labels, length: float = 30.0, stride: float = 30.0) -> list[Window]`
  - `stratify(windows, per_case_cap: int, seed: int) -> list[Window]`
  - `make_splits(case_ids, val_fraction: float, seed: int) -> dict` — `{"train": [...], "val": [...]}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sampling.py
from pathlib import Path

from surgvu.labels import CaseLabels
from surgvu.sampling import enumerate_windows, stratify, make_splits

FIXTURE = Path(__file__).parent / "fixtures" / "case_test"


def test_windows_lie_inside_task_segments():
    labels = CaseLabels.from_dir(FIXTURE)
    windows = enumerate_windows("case_test", labels)
    assert windows
    for w in windows:
        assert labels.task_at(w.part, w.start + 15.0) is not None


def test_windows_carry_tools_and_description():
    labels = CaseLabels.from_dir(FIXTURE)
    windows = [w for w in enumerate_windows("case_test", labels) if w.part == "1.0"]
    assert windows
    w = windows[0]
    assert w.task == "suturing"
    assert "needle driver" in w.tools
    assert w.description.startswith("Excess bleeding")


def test_windows_never_span_a_part_boundary():
    labels = CaseLabels.from_dir(FIXTURE)
    for w in enumerate_windows("case_test", labels):
        assert w.part in {"1.0", "2.0"}


def test_stratify_is_deterministic_for_a_seed():
    labels = CaseLabels.from_dir(FIXTURE)
    windows = enumerate_windows("case_test", labels)
    a = stratify(windows, per_case_cap=3, seed=42)
    b = stratify(windows, per_case_cap=3, seed=42)
    assert [w.start for w in a] == [w.start for w in b]


def test_stratify_respects_the_cap():
    labels = CaseLabels.from_dir(FIXTURE)
    windows = enumerate_windows("case_test", labels)
    assert len(stratify(windows, per_case_cap=2, seed=0)) <= 2


def test_splits_are_disjoint_and_cover_everything():
    ids = ["case_%03d" % i for i in range(155)]
    splits = make_splits(ids, val_fraction=0.2, seed=7)
    assert set(splits["train"]) | set(splits["val"]) == set(ids)
    assert not set(splits["train"]) & set(splits["val"])
    assert 28 <= len(splits["val"]) <= 34


def test_splits_are_deterministic():
    ids = ["case_%03d" % i for i in range(155)]
    assert make_splits(ids, 0.2, seed=7) == make_splits(ids, 0.2, seed=7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sampling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'surgvu.sampling'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/surgvu/sampling.py
"""Choose which 30-second windows become training data.

The unit is a 30-second window sampled at 1 fps, because that is exactly the
test-time format. Windows are enumerated only inside task segments — that is
where the evaluation clips come from, and it is where the labels are defined.

Stratification is by TOOL CLASS rather than duration. Intervals and hours
disagree sharply in this dataset: clip applier appears in 895 intervals but
only 12.5 hours, because it is installed briefly and often. Sampling by
duration would under-represent it badly.
"""
import random
from collections import namedtuple

from .taxonomy import TOOL_CLASSES

Window = namedtuple("Window", "case part start task description tools")

WINDOW_SECONDS = 30.0


def enumerate_windows(case_id, labels, length=WINDOW_SECONDS, stride=WINDOW_SECONDS):
    """Every non-overlapping window that fits inside a task segment."""
    windows = []
    for segment in labels.task_segments():
        t = segment.start
        while t + length <= segment.stop:
            midpoint = t + length / 2.0
            tools = labels.tools_at(segment.part, midpoint)
            windows.append(Window(
                case=case_id,
                part=segment.part,
                start=t,
                task=segment.task,
                description=segment.description,
                tools=frozenset(tools),
            ))
            t += stride
    return windows


def stratify(windows, per_case_cap, seed=0):
    """Down-sample to per_case_cap, favouring rare tool classes.

    Windows are visited in ascending order of their rarest tool's global count,
    so a window containing a tip-up fenestrated grasper is taken before one
    containing only needle drivers.
    """
    if len(windows) <= per_case_cap:
        return list(windows)

    frequency = {name: 0 for name in TOOL_CLASSES}
    for w in windows:
        for tool in w.tools:
            frequency[tool] += 1

    def rarity(window):
        if not window.tools:
            return 10 ** 9
        return min(frequency[tool] for tool in window.tools)

    rng = random.Random(seed)
    shuffled = list(windows)
    rng.shuffle(shuffled)                       # break ties reproducibly
    shuffled.sort(key=rarity)
    return shuffled[:per_case_cap]


def make_splits(case_ids, val_fraction=0.2, seed=7):
    """Split by CASE. Frames from one session are near-duplicates, so a
    frame-level split leaks and every number measured afterwards is fiction."""
    ids = sorted(case_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_fraction))
    return {"val": sorted(ids[:n_val]), "train": sorted(ids[n_val:])}
```

```python
# scripts/build_splits.py
"""Write the canonical case-level split. Generate once; never regenerate."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.labels import load_all_cases  # noqa: E402
from surgvu.sampling import make_splits  # noqa: E402

if __name__ == "__main__":
    labels_root = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("config/splits.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        print("REFUSING to overwrite %s — the split must stay stable." % out)
        raise SystemExit(1)

    cases = load_all_cases(labels_root)
    splits = make_splits(list(cases), val_fraction=0.2, seed=7)
    out.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print("train %d  val %d" % (len(splits["train"]), len(splits["val"])))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sampling.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Generate the canonical split and check window supply**

Run:
```bash
python scripts/build_splits.py data/SURGVU25_train_labels config/splits.json
python -c "
import sys; sys.path.insert(0,'src')
from collections import Counter
from surgvu.labels import load_all_cases
from surgvu.sampling import enumerate_windows
cases = load_all_cases('data/SURGVU25_train_labels')
total, card, tools = 0, Counter(), Counter()
for cid, lb in cases.items():
    ws = enumerate_windows(cid, lb)
    total += len(ws)
    for w in ws:
        card[len(w.tools)] += 1
        for t in w.tools: tools[t] += 1
print('windows:', total)
print('cardinality:', dict(sorted(card.items())))
print('rarest tools:', tools.most_common()[-4:])
"
```
Expected: `train 124  val 31`. Windows in the tens of thousands. **Cardinality should peak at 3 with roughly 78% of mass** — that reproduces the measurement the design rests on. A very different shape means the label engine and the earlier analysis disagree; investigate before extracting frames.

- [ ] **Step 6: Commit**

```bash
git add src/surgvu/sampling.py tests/test_sampling.py scripts/build_splits.py config/splits.json
git commit -m "feat: window enumeration, rarity-weighted stratification, canonical splits"
```

---

### Task 6: Frame extraction to shards

**Files:**
- Create: `src/surgvu/extract.py`
- Create: `tests/test_extract.py`
- Create: `scripts/extract_case.py`
- Create: `condor/extract.sub`
- Create: `condor/extract.sh`

**Interfaces:**
- Consumes: `surgvu.preprocess.prepare_frame`, `surgvu.sampling.Window`
- Produces:
  - `extract_window(video_path, window, fps: int = 1, size: int = 512) -> list[np.ndarray]`
  - `write_shard(windows_and_frames, out_path)` — one `.npz` per case
  - `read_shard(path) -> tuple[np.ndarray, list[dict]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extract.py
import numpy as np
import cv2
import pytest

from surgvu.extract import extract_window, write_shard, read_shard
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


def _window(start):
    return Window(case="c", part="1.0", start=start, task="suturing",
                  description="d", tools=frozenset({"needle driver"}))


def test_extract_window_returns_one_frame_per_second(synthetic_video):
    frames = extract_window(synthetic_video, _window(10.0), fps=1, size=64)
    assert len(frames) == 30
    assert frames[0].shape == (64, 64, 3)


def test_extract_window_starts_at_the_right_place(synthetic_video):
    early = extract_window(synthetic_video, _window(0.0), fps=1, size=64)
    late = extract_window(synthetic_video, _window(30.0), fps=1, size=64)
    assert early[0].mean() != late[0].mean()


def test_extract_window_past_end_returns_short_or_empty(synthetic_video):
    frames = extract_window(synthetic_video, _window(200.0), fps=1, size=64)
    assert len(frames) < 30


def test_shard_roundtrip(tmp_path):
    frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(3)]
    payload = [(_window(0.0), frames)]
    path = tmp_path / "case_x.npz"
    write_shard(payload, path)

    stack, meta = read_shard(path)
    assert stack.shape == (1, 3, 8, 8, 3)
    assert meta[0]["task"] == "suturing"
    assert meta[0]["tools"] == ["needle driver"]
    assert meta[0]["case"] == "c"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_extract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'surgvu.extract'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/surgvu/extract.py
"""Decode each video once and emit training shards.

Decoding 344 GB of 60 fps video is the expensive step in this project, so it
happens exactly once and all three models are trained from the same frame
pool — they differ only in labels and in how many consecutive frames they use.
"""
import json

import cv2
import numpy as np

from .preprocess import prepare_frame
from .sampling import WINDOW_SECONDS


def extract_window(video_path, window, fps=1, size=512):
    """Decode one window as `fps` frames per second, preprocessed."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        step = 1.0 / float(fps)
        frames = []
        for i in range(int(WINDOW_SECONDS * fps)):
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


def write_shard(windows_and_frames, out_path, min_frames=25):
    """One compressed .npz per case: a frame stack plus JSON metadata.

    A ragged shard cannot be stacked, so every window is truncated to the
    shortest one present. That truncation is reported rather than silent —
    a single short window would otherwise quietly shorten the whole case.
    Windows below min_frames are dropped instead of dragging the rest down.
    """
    usable = [(w, f) for w, f in windows_and_frames if len(f) >= min_frames]
    if not usable:
        raise ValueError("no windows with at least %d frames" % min_frames)

    lengths = [len(f) for _, f in usable]
    length = min(lengths)
    if length != max(lengths):
        print("write_shard: truncating %d windows from max %d to %d frames"
              % (len(usable), max(lengths), length))
    stack = np.stack([np.stack(f[:length]) for _, f in usable])
    meta = [{
        "case": w.case,
        "part": w.part,
        "start": w.start,
        "task": w.task,
        "description": w.description,
        "tools": sorted(w.tools),
    } for w, _ in usable]

    np.savez_compressed(out_path, frames=stack, meta=json.dumps(meta))


def read_shard(path):
    payload = np.load(path, allow_pickle=False)
    return payload["frames"], json.loads(str(payload["meta"]))
```

```python
# scripts/extract_case.py
"""Extract one case to a shard. One HTCondor job runs one of these."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.extract import extract_window, write_shard  # noqa: E402
from surgvu.labels import CaseLabels  # noqa: E402
from surgvu.sampling import enumerate_windows, stratify  # noqa: E402

PER_CASE_CAP = 160

if __name__ == "__main__":
    case_id, labels_dir, video_path, out_path = sys.argv[1:5]

    labels = CaseLabels.from_dir(labels_dir)
    windows = stratify(enumerate_windows(case_id, labels),
                       per_case_cap=PER_CASE_CAP, seed=7)
    print("%s: %d windows selected" % (case_id, len(windows)))

    payload = []
    for i, window in enumerate(windows):
        frames = extract_window(video_path, window)
        payload.append((window, frames))        # write_shard drops short ones
        if (i + 1) % 20 == 0:
            print("  %d/%d" % (i + 1, len(windows)), flush=True)

    write_shard(payload, out_path)
    print("%s: wrote %d windows -> %s" % (case_id, len(payload), out_path))
```

```bash
# condor/extract.sh
#!/bin/bash
set -euo pipefail
CASE_ID="$1"
VIDEO="$2"
tar xzf surgvu_src.tar.gz
python3 scripts/extract_case.py "$CASE_ID" "labels/${CASE_ID}" "$VIDEO" "${CASE_ID}.npz"
```

```
# condor/extract.sub
universe                = vanilla
executable              = condor/extract.sh
arguments               = $(case_id) $(video)
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = surgvu_src.tar.gz, labels/, $(video)
transfer_output_files   = $(case_id).npz
transfer_output_remaps  = "$(case_id).npz = /staging/n/nkalthoff/surgvu/shards/$(case_id).npz"
request_cpus            = 2
request_memory          = 8GB
request_disk            = 20GB
log                     = logs/extract_$(case_id).log
error                   = logs/extract_$(case_id).err
output                  = logs/extract_$(case_id).out
queue case_id, video from cases.txt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_extract.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Generate the Condor queue file**

`extract.sub` reads `cases.txt`, one `case_id, video_path` pair per line. Generate it from what actually landed on staging rather than from the label list, so a missing video fails loudly here instead of silently mid-run:

```python
# scripts/build_cases_txt.py
"""Emit condor/cases.txt from the videos present on staging."""
import re
import sys
from pathlib import Path

if __name__ == "__main__":
    video_root = Path(sys.argv[1])
    labels_root = Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("condor/cases.txt")
    out.parent.mkdir(parents=True, exist_ok=True)

    known = {d.name for d in labels_root.iterdir() if d.is_dir()}
    lines, skipped = [], []
    for video in sorted(video_root.glob("*.mp4")):
        match = re.match(r"(case_\d+)", video.name)
        if not match:
            skipped.append(video.name)
            continue
        case_id = match.group(1)
        if case_id not in known:
            skipped.append(video.name)
            continue
        lines.append("%s, %s" % (case_id, video))

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("queued %d videos" % len(lines))
    if skipped:
        print("SKIPPED %d with no matching labels:" % len(skipped), skipped[:5])
```

Run: `python scripts/build_cases_txt.py /staging/n/nkalthoff/surgvu/videos data/SURGVU25_train_labels condor/cases.txt`
Expected: around 280 queued, 0 skipped. A non-zero skip count means the video filenames do not follow `case_NNN*`, so inspect them before submitting 280 jobs.

- [ ] **Step 6: Extract one real case and eyeball it**

Run:
```bash
python scripts/extract_case.py case_056 data/SURGVU25_train_labels/case_056 \
    data/videos/case_056_video_part_001.mp4 /tmp/case_056.npz

python -c "
import sys; sys.path.insert(0,'src')
import cv2, numpy as np
from surgvu.extract import read_shard
stack, meta = read_shard('/tmp/case_056.npz')
print(stack.shape)
for i in (0, 5, 10):
    print(meta[i]['task'], meta[i]['tools'])
    cv2.imwrite('/tmp/check_%d.png' % i, stack[i][15])
"
```

**This is the check not to skip.** Open `/tmp/check_0.png`, `check_5.png`, `check_10.png` and confirm: (a) the instruments visible match the `tools` list printed for that window, and (b) the bottom UI band is unreadable. Every label in this project is *derived* from timestamps rather than read off a frame, so a single off-by-one in part handling or timestamp parsing would mislabel the whole corpus in a way no loss curve would reveal.

- [ ] **Step 7: Commit**

```bash
git add src/surgvu/extract.py tests/test_extract.py scripts/extract_case.py scripts/build_cases_txt.py condor/
git commit -m "feat: window extraction to shards, with Condor fan-out"
```

---

### Task 7: Official scoring harness

**Files:**
- Create: `src/surgvu/scoring.py`
- Create: `tests/test_scoring.py`
- Create: `scripts/score_answers.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `normalize(text: str) -> str` — matches the organizers' BLEU/ROUGE normalization exactly
  - `Scorer(device: str = "cpu")` with `.score_one(candidate: str, references: list[str]) -> dict`
  - `.score_many(pairs) -> dict` — per-case results plus the mean aggregate

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scoring.py
import pytest
from surgvu.scoring import normalize


def test_normalize_matches_organizer_implementation():
    # Verbatim from the challenge's evaluate.py: strip punctuation, lowercase, strip
    assert normalize("Yes, a large needle driver was utilized.") == \
        "yes a large needle driver was utilized"
    assert normalize("  Uterine horn  ") == "uterine horn"
    assert normalize("") == ""


@pytest.mark.slow
def test_identical_string_scores_one():
    from surgvu.scoring import Scorer
    scorer = Scorer()
    result = scorer.score_one("Yes", ["Yes", "Yes, a tool was used."])
    assert result["bertscore_f1"] > 0.99


@pytest.mark.slow
def test_wrong_polarity_scores_far_below_correct():
    from surgvu.scoring import Scorer
    scorer = Scorer()
    refs = ["Yes", "Yes, a large needle driver was utilized."]
    right = scorer.score_one("Yes", refs)["bertscore_f1"]
    wrong = scorer.score_one("No", refs)["bertscore_f1"]
    assert right > wrong + 0.2, (right, wrong)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring.py -v -m "not slow"`
Expected: FAIL with `ModuleNotFoundError: No module named 'surgvu.scoring'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/surgvu/scoring.py
"""Wrapper around the official SurgVU 2026 Category 2 metric.

The evaluation container is public, so this reproduces it rather than
approximating it. Primary metric: BERTScore-F1, roberta-large, rescaled with
baseline, MAX over the five references, meaned across cases.

Note the asymmetry, copied deliberately from the organizers' evaluate.py:
BLEU and ROUGE are computed on normalized text, but BERTScore and NLI receive
the RAW candidate and references. Casing and punctuation therefore affect the
metric that ranks us.
"""
import string
from statistics import mean

_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text):
    """Lowercase and strip punctuation — the organizers' BLEU/ROUGE path."""
    return (text or "").translate(_PUNCT).lower().strip()


class Scorer:
    """Lazily loads roberta-large; reuse one instance across many calls."""

    def __init__(self, device=None, model_type="roberta-large"):
        self._device = device
        self._model_type = model_type
        self._bert = None

    def _bert_scorer(self):
        if self._bert is None:
            from bert_score import BERTScorer
            self._bert = BERTScorer(
                model_type=self._model_type,
                lang="en",
                rescale_with_baseline=True,
                device=self._device,
            )
        return self._bert

    def score_one(self, candidate, references):
        """Max BERTScore-F1 of candidate against every reference."""
        if not references:
            return {"bertscore_f1": 0.0}
        scorer = self._bert_scorer()
        expanded = [candidate] * len(references)
        _p, _r, f1 = scorer.score(expanded, list(references))
        return {"bertscore_f1": float(f1.max().item())}

    def score_many(self, pairs):
        """pairs: iterable of (case_id, candidate, references)."""
        results = []
        for case_id, candidate, references in pairs:
            row = self.score_one(candidate, references)
            row["case_id"] = case_id
            results.append(row)
        aggregate = mean(r["bertscore_f1"] for r in results) if results else 0.0
        return {"results": results, "aggregates": {"bertscore_f1": aggregate}}
```

```python
# scripts/score_answers.py
"""Score a predictions file against the Cat 2 sample set.

Usage: python scripts/score_answers.py data/samples predictions.json
predictions.json: {"case122": "Yes", "case123": "No", ...}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.scoring import Scorer  # noqa: E402

if __name__ == "__main__":
    samples_root = Path(sys.argv[1])
    predictions = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

    pairs = []
    for case_dir in sorted(samples_root.iterdir()):
        if not case_dir.is_dir():
            continue
        gt_file = case_dir / ("%s.json" % case_dir.name)
        if not gt_file.exists() or case_dir.name not in predictions:
            continue
        references = json.loads(gt_file.read_text(encoding="utf-8"))
        pairs.append((case_dir.name, predictions[case_dir.name], references))

    report = Scorer().score_many(pairs)
    for row in report["results"]:
        print("%-10s %.4f" % (row["case_id"], row["bertscore_f1"]))
    print("-" * 22)
    print("%-10s %.4f" % ("MEAN", report["aggregates"]["bertscore_f1"]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scoring.py -v -m "not slow"`
Expected: PASS, 1 test

Then, once `pip install bert-score torch` has completed and roberta-large has downloaded (~1.4 GB):

Run: `pytest tests/test_scoring.py -v`
Expected: PASS, 3 tests. The polarity test is the important one — it quantifies how much a wrong yes/no actually costs, which is the number the whole answer-form design rests on.

- [ ] **Step 5: Establish the bar**

Run:
```bash
python -c "
import json
from pathlib import Path
preds = {d.name: 'Yes' for d in Path('data/samples').iterdir() if d.is_dir()}
Path('/tmp/all_yes.json').write_text(json.dumps(preds))
"
python scripts/score_answers.py data/samples /tmp/all_yes.json
```
Record the number. Answering `"Yes"` to everything is the floor any real system must beat, and it takes two minutes to know.

- [ ] **Step 6: Commit**

```bash
git add src/surgvu/scoring.py tests/test_scoring.py scripts/score_answers.py
git commit -m "feat: official BERTScore harness and sample-set scoring script"
```

---

## What this plan deliberately leaves out

- **Model training.** Plan 2. It needs this plan's shards and splits.
- **The router, question parser, answer-form selector, and submission container.** Plan 3.
- **Synthetic validation-set generation.** Plan 3 — it needs the scoring harness from Task 7 and the description corpus from Task 3, both of which land here.
- **The detector.** Off the shelf, no training, and only if measurement shows it is needed.

## Verification that the whole plan worked

After Task 7, all of the following must hold:

```bash
pytest -v                                    # 44 tests, all passing (42 without -m slow)
python -c "
import sys, json; sys.path.insert(0,'src')
from surgvu.descriptions import load_corpus
c = load_corpus('config/descriptions.yaml')
assert sum(len(v) for v in c.values()) == 21, 'description corpus is not 21 strings'
s = json.load(open('config/splits.json'))
assert not set(s['train']) & set(s['val']), 'split leaks between train and val'
assert len(s['train']) + len(s['val']) == 155, 'split does not cover all cases'
print('foundation OK')
"
```

Plus the two human checks that no assertion can replace: **extracted frames match their derived labels**, and **the UI band is unreadable after preprocessing**.
