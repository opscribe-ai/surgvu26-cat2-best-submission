from pathlib import Path

from surgvu.labels import CaseLabels, Segment
from surgvu.sampling import Window, enumerate_windows, stratify, make_splits
from surgvu.taxonomy import TOOL_CLASSES

FIXTURE = Path(__file__).parent / "fixtures" / "case_test"


def _w(start, tools, length=30.0):
    return Window(case="c", part="1.0", start=start, length=length,
                  task="suturing", description="", tools=frozenset(tools))


def test_windows_lie_inside_task_segments():
    labels = CaseLabels.from_dir(FIXTURE)
    windows = enumerate_windows("case_test", labels)
    assert windows
    for w in windows:
        assert labels.task_at(w.part, w.start + 15.0) is not None


def test_windows_record_their_own_length():
    """Window length was an unrecorded convention: `enumerate_windows` took a
    `length`, the extractor read a module constant, and the shard writer
    hardcoded 30. `enumerate_windows(length=15)` therefore produced windows
    that were decoded as 30 seconds. The window now carries the length that
    produced it, and it is the only definition downstream reads."""
    labels = CaseLabels.from_dir(FIXTURE)

    default = enumerate_windows("case_test", labels)
    assert default
    assert {w.length for w in default} == {30.0}

    short = enumerate_windows("case_test", labels, length=15.0, stride=15.0)
    assert short
    assert {w.length for w in short} == {15.0}
    assert len(short) > len(default)          # more windows fit in a segment


def test_windows_carry_tools_and_description():
    labels = CaseLabels.from_dir(FIXTURE)
    windows = [w for w in enumerate_windows("case_test", labels) if w.part == "1.0"]
    assert windows
    w = windows[0]
    assert w.task == "suturing"
    assert "needle driver" in w.tools
    assert w.description.startswith("Excess bleeding")


def test_every_window_task_agrees_with_task_at_its_own_midpoint():
    """`task_at` resolves overlapping segments by taking the SHORTEST covering
    one. `enumerate_windows` used to copy the task off whichever segment it was
    iterating, so a window inside an overlap could carry a task that
    contradicted `task_at` at its own midpoint — 155 such windows across the
    real corpus. There is exactly one answer to "what task is happening here",
    and it is `task_at`'s."""
    labels = CaseLabels.from_dir(FIXTURE)
    windows = enumerate_windows("case_test", labels)
    assert windows
    for w in windows:
        resolved = labels.task_at(w.part, w.start + w.length / 2.0)
        assert resolved is not None
        assert (w.task, w.description) == resolved


def test_overlapping_segments_resolve_to_the_shortest_segment():
    """A long 'other' segment containing a short 'suturing' one. EVERY window
    whose midpoint falls in the short segment must be labelled suturing — not
    just the one enumerated from the short segment. The window enumerated from
    the LONG segment at the same moment is the one that used to be wrong."""
    labels = _OverlappingLabels()
    windows = enumerate_windows("c", labels)
    assert windows

    for w in windows:
        assert (w.task, w.description) == labels.task_at(w.part, w.start + w.length / 2.0)

    inside_short = [w for w in windows if 60.0 <= w.start + w.length / 2.0 <= 90.0]
    assert inside_short
    assert {w.task for w in inside_short} == {"suturing"}
    assert {w.task for w in windows if w not in inside_short} == {"other"}


def test_duplicate_windows_are_not_emitted():
    """Overlapping segments can enumerate the identical clip twice. Decoding
    and storing it twice wastes the expensive step and over-weights that clip
    in training, so exact duplicates are dropped."""
    labels = _DuplicateProducingLabels()
    windows = enumerate_windows("c", labels)

    assert len(windows) == len(set(windows))
    assert len(windows) == len({(w.part, w.start) for w in windows})


class _OverlappingLabels:
    """Minimal CaseLabels stand-in: a short segment nested in a long one."""

    _SEGMENTS = [
        Segment(part="1.0", start=0.0, stop=240.0, task="other", description="long"),
        Segment(part="1.0", start=60.0, stop=90.0, task="suturing", description="short"),
    ]

    def task_segments(self):
        return list(self._SEGMENTS)

    def tools_at(self, part, seconds):
        return set()

    def task_at(self, part, seconds):
        hits = [s for s in self._SEGMENTS
                if s.part == part and s.start <= seconds <= s.stop]
        if not hits:
            return None
        best = min(hits, key=lambda s: s.stop - s.start)
        return best.task, best.description


class _DuplicateProducingLabels(_OverlappingLabels):
    """Two identical-start segments of different lengths: without dedupe,
    enumeration emits the same window twice."""

    _SEGMENTS = [
        Segment(part="1.0", start=0.0, stop=120.0, task="suturing", description="d"),
        Segment(part="1.0", start=0.0, stop=60.0, task="suturing", description="d"),
    ]


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


def test_stratify_prefers_locally_rare_tool_over_common_tool():
    rare = _w(0.0, {"clip applier"})              # appears once
    common = [_w(30.0 * i, {"needle driver"}) for i in range(1, 6)]  # appears 5x
    windows = [rare] + common

    result = stratify(windows, per_case_cap=1, seed=0)

    assert result == [rare]


def test_stratify_explicit_frequency_overrides_case_local_default():
    rare = _w(0.0, {"clip applier"})               # locally rare (count 1)
    common = [_w(30.0 * i, {"needle driver"}) for i in range(1, 6)]  # locally common
    windows = [rare] + common

    default_result = stratify(windows, per_case_cap=1, seed=0)
    assert default_result == [rare]        # case-local: clip applier "rarer" here

    # Dataset-wide reality is the opposite: needle driver is actually rare,
    # clip applier is actually common.
    global_frequency = {name: 1000 for name in TOOL_CLASSES}
    global_frequency["needle driver"] = 1
    global_frequency["clip applier"] = 1000

    global_result = stratify(windows, per_case_cap=1, seed=0, frequency=global_frequency)

    assert global_result != default_result
    assert global_result[0] in common


def test_stratify_toolless_window_sorts_last():
    tool_bearing = [_w(0.0, {"clip applier"}), _w(30.0, {"needle driver"})]
    toolless = [_w(60.0 * i, set()) for i in range(1, 4)]
    windows = tool_bearing + toolless

    # Cap equal to the number of tool-bearing windows: none of the
    # toolless windows should survive.
    tight = stratify(windows, per_case_cap=2, seed=0)
    assert len(tight) == 2
    assert all(w.tools for w in tight)

    # Cap one higher than the number of tool-bearing windows: exactly one
    # toolless window is admitted, only once the tool-bearing ones are
    # already in.
    looser = stratify(windows, per_case_cap=3, seed=0)
    assert len(looser) == 3
    assert sum(1 for w in looser if not w.tools) == 1


def test_splits_are_disjoint_and_cover_everything():
    ids = ["case_%03d" % i for i in range(155)]
    splits = make_splits(ids, val_fraction=0.2, seed=7)
    assert set(splits["train"]) | set(splits["val"]) == set(ids)
    assert not set(splits["train"]) & set(splits["val"])
    assert 28 <= len(splits["val"]) <= 34


def test_splits_are_deterministic():
    ids = ["case_%03d" % i for i in range(155)]
    assert make_splits(ids, 0.2, seed=7) == make_splits(ids, 0.2, seed=7)
