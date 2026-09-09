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


def normalize_part(value):
    """Canonical part label. `'1'`, `1`, `'1.0'`, `'001'` all become `'1.0'`.

    The CSVs write parts as `'1.0'`/`'2.0'`, video filenames write them as
    `001`/`002`, and a human on a command line writes `1`. Part identity
    decides which video a timestamp is measured against, so every comparison
    in this package goes through one canonicalisation rather than trusting
    that the two spellings happen to match. Anything unparseable is returned
    stripped, so it simply fails to match rather than matching the wrong part.
    """
    text = str(value if value is not None else "").strip()
    try:
        return "%.1f" % float(text)
    except (TypeError, ValueError):
        return text


_part = normalize_part


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

        Where segments overlap, the shortest wins -- it is the most specific.
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
    """Load every case directory under root into {case_id: CaseLabels}.

    A directory holding neither CSV is not a case and is skipped quietly. A
    directory holding one but not the other IS a case, and a broken one: with
    `tasks.csv` missing it yields zero segments, therefore zero windows,
    therefore an empty shard -- silently, and only at the far end of a
    day-scale extraction run. Half a case is an error, and it is raised here
    at queue-build time where it costs nothing to fix.
    """
    root = Path(root)
    cases = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        tools, tasks = child / "tools.csv", child / "tasks.csv"
        if not tools.exists() and not tasks.exists():
            continue                       # not a case directory at all
        missing = [f.name for f in (tools, tasks) if not f.exists()]
        if missing:
            raise ValueError(
                "%s is missing %s. A case with one CSV and not the other "
                "produces zero windows and an empty shard without ever "
                "failing. Fix the case or remove the directory."
                % (child, " and ".join(missing)))
        cases[child.name] = CaseLabels.from_dir(child)
    return cases
