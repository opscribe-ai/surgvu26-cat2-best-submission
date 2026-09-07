from pathlib import Path

import pytest

from surgvu.labels import parse_hms, CaseLabels, load_all_cases

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


def test_tools_at_drops_part_spanning_tool():
    c = CaseLabels.from_dir(FIXTURE)
    # force bipolar installs in part 1.0 and uninstalls in part 2.0 (900s ->
    # 1200s numerically, so it would survive a stop<=start guard on its own);
    # only the part-mismatch check can drop it. It must never appear in
    # either part.
    assert "force bipolar" not in c.tools_at("1.0", 1000.0)
    assert "force bipolar" not in c.tools_at("2.0", 1000.0)


def test_task_at_normalizes_case():
    c = CaseLabels.from_dir(FIXTURE)
    task, desc = c.task_at("1.0", 120.0)
    assert task == "suturing"          # raw was "Suturing"
    assert desc.startswith("Excess bleeding")


def test_task_at_outside_segments_is_none():
    c = CaseLabels.from_dir(FIXTURE)
    assert c.task_at("1.0", 10.0) is None


def test_task_at_overlap_prefers_shorter_segment():
    c = CaseLabels.from_dir(FIXTURE)
    # 2000-2100 ("other") and 2010-2050 ("range of motion") both cover
    # t=2030; the shorter, more specific segment must win.
    task, desc = c.task_at("1.0", 2030.0)
    assert task == "range of motion"
    assert desc.startswith("Short segment")


def test_task_segments_are_deduplicated():
    c = CaseLabels.from_dir(FIXTURE)
    segs = c.task_segments()
    # Raw rows in the fixture, by outcome:
    #   [60, 300]     part 1.0  suturing        -- appears twice, 1 duplicate removed -> kept
    #   [100, 400]    part 2.0  uterine horn     -- kept
    #   [500, 50]     spans part 1.0->2.0, stop < start numerically       -- dropped
    #   [700, 800]    part 1.0  other, blank description                 -- kept
    #   [1400, 1700]  spans part 1.0->2.0, stop > start numerically       -- dropped
    #   [2000, 2100]  part 1.0  other (overlap-long)                     -- kept
    #   [2010, 2050]  part 1.0  range of motion (overlap-short)          -- kept
    #   [3000, 3100]  part 1.0  skills application, rare description     -- kept
    #   [3200, 3300]  part 1.0  skills application, common description A -- kept
    #   [3400, 3500]  part 1.0  skills application, common description B -- kept
    #   [3600, 3700]  part 1.0  skills application, common description C -- kept
    # -> 9 surviving segments: 8 in part 1.0, 1 in part 2.0
    assert len(segs) == 9
    assert sum(1 for s in segs if s.part == "1.0") == 8
    assert sum(1 for s in segs if s.part == "2.0") == 1


def test_task_segments_drop_part_spanning():
    c = CaseLabels.from_dir(FIXTURE)
    segs = c.task_segments()
    # The 500->50 row is also caught by the unrelated stop<=start guard, so
    # it alone would not prove the part-mismatch check does anything. The
    # 1400->1700 row has stop > start numerically and would survive a
    # stop<=start-only guard; only the part-mismatch check can drop it.
    assert not any(s.start == 1400.0 and s.stop == 1700.0 for s in segs)
    assert not any(s.start == 500.0 for s in segs)
    for s in segs:
        assert s.stop > s.start


def test_blank_description_segments_are_kept():
    # A segment with a valid task label but no description text is still
    # usable training data for the task classifier. Only the description
    # corpus filters blanks, and it does so at its own layer.
    labels = CaseLabels.from_dir(FIXTURE)
    segs = labels.task_segments()
    assert any(s.description == "" for s in segs)
    assert all(s.task for s in segs)


def test_load_all_cases_raises_when_a_case_has_only_one_csv(tmp_path):
    """Half a case is an error, not an empty result.

    With tasks.csv missing the case yields zero segments, zero windows and an
    empty shard -- silently, and only at the far end of a day-scale extraction
    run. This is the project's characteristic failure and it must surface at
    queue-build time.
    """
    good = tmp_path / "case_000"
    good.mkdir()
    (good / "tools.csv").write_text(
        "install_case_part,install_case_time,uninstall_case_part,"
        "uninstall_case_time,arm,commercial_toolname,groundtruth_toolname,case\n",
        encoding="utf-8")
    (good / "tasks.csv").write_text(
        "index,start_part,start_time,stop_part,stop_time,duration,taskname,"
        "groundtruth_taskname,matched_description,case\n", encoding="utf-8")

    half = tmp_path / "case_001"
    half.mkdir()
    (half / "tools.csv").write_text("install_case_part\n", encoding="utf-8")

    with pytest.raises(ValueError, match="tasks.csv"):
        load_all_cases(tmp_path)


def test_load_all_cases_skips_directories_that_are_not_cases(tmp_path):
    """A directory with neither CSV is not a broken case, just not a case."""
    (tmp_path / ".ipynb_checkpoints").mkdir()
    (tmp_path / "notes").mkdir()
    case = tmp_path / "case_000"
    case.mkdir()
    (case / "tools.csv").write_text(
        "install_case_part,install_case_time,uninstall_case_part,"
        "uninstall_case_time,arm,commercial_toolname,groundtruth_toolname,case\n",
        encoding="utf-8")
    (case / "tasks.csv").write_text(
        "index,start_part,start_time,stop_part,stop_time,duration,taskname,"
        "groundtruth_taskname,matched_description,case\n", encoding="utf-8")

    assert sorted(load_all_cases(tmp_path)) == ["case_000"]
