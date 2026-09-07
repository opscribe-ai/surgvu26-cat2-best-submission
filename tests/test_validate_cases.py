"""Tests for the container-path validation harness.

`scripts/validate_cases.py` is what proves the submission entrypoint works, so
it may not import torch (it must be runnable and testable outside the training
container) and it may not quietly turn a broken run into a passing row. Both
properties are pinned here.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import validate_cases as vc  # noqa: E402


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def make_sample(root, case_ids, question="Are there forceps being used here?"):
    """Build the nested cat2_sample layout: <root>/caseNNN/{mp4,json,_question}."""
    for case_id in case_ids:
        case_dir = root / case_id
        case_dir.mkdir(parents=True)
        (case_dir / ("%s.mp4" % case_id)).write_bytes(b"not-a-real-mp4")
        (case_dir / ("%s.json" % case_id)).write_text(json.dumps(["Yes"]))
        (case_dir / ("%s_question.json" % case_id)).write_text(json.dumps(question))
    return root


# --------------------------------------------------------------------------
# discover_cases
# --------------------------------------------------------------------------

def test_discovers_every_case_sorted(tmp_path):
    make_sample(tmp_path, ["case130", "case122", "case127"])
    cases = vc.discover_cases(tmp_path)
    assert [case.case_id for case in cases] == ["case122", "case127", "case130"]


def test_discovered_case_points_at_video_and_question(tmp_path):
    make_sample(tmp_path, ["case122"])
    case = vc.discover_cases(tmp_path)[0]
    assert case.video == tmp_path / "case122" / "case122.mp4"
    assert case.question == tmp_path / "case122" / "case122_question.json"


def test_question_file_is_not_mistaken_for_a_case(tmp_path):
    """`caseNNN_question.json` is metadata, not a twelfth case."""
    make_sample(tmp_path, ["case122"])
    assert [case.case_id for case in vc.discover_cases(tmp_path)] == ["case122"]


def test_missing_video_is_an_error_not_a_skip(tmp_path):
    make_sample(tmp_path, ["case122"])
    (tmp_path / "case122" / "case122.mp4").unlink()
    with pytest.raises(ValueError, match="case122"):
        vc.discover_cases(tmp_path)


def test_missing_question_is_an_error_not_a_skip(tmp_path):
    make_sample(tmp_path, ["case122"])
    (tmp_path / "case122" / "case122_question.json").unlink()
    with pytest.raises(ValueError, match="case122"):
        vc.discover_cases(tmp_path)


def test_empty_sample_root_raises(tmp_path):
    """Eleven cases silently becoming zero is the failure mode that would make
    an empty report look like a clean one."""
    with pytest.raises(ValueError):
        vc.discover_cases(tmp_path)


# --------------------------------------------------------------------------
# stage_case -- the /input layout is the contract
# --------------------------------------------------------------------------

def test_stage_case_uses_the_contract_filenames(tmp_path):
    make_sample(tmp_path / "sample", ["case122"])
    case = vc.discover_cases(tmp_path / "sample")[0]
    input_dir, output_dir = vc.stage_case(case, tmp_path / "work")
    assert (input_dir / "endoscopic-robotic-surgery-video.mp4").exists()
    assert (input_dir / "visual-context-question.json").exists()
    assert sorted(p.name for p in input_dir.iterdir()) == [
        "endoscopic-robotic-surgery-video.mp4", "visual-context-question.json"]
    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


def test_stage_case_copies_bytes_verbatim(tmp_path):
    """The question must arrive still JSON-encoded; re-encoding it here would
    hide the very bug read_question exists to survive."""
    make_sample(tmp_path / "sample", ["case122"], question="Is this a test?")
    case = vc.discover_cases(tmp_path / "sample")[0]
    input_dir, _ = vc.stage_case(case, tmp_path / "work")
    assert (input_dir / "visual-context-question.json").read_text() \
        == '"Is this a test?"'
    assert (input_dir / "endoscopic-robotic-surgery-video.mp4").read_bytes() \
        == b"not-a-real-mp4"


def test_stage_case_is_isolated_per_case(tmp_path):
    make_sample(tmp_path / "sample", ["case122", "case123"])
    cases = vc.discover_cases(tmp_path / "sample")
    first, _ = vc.stage_case(cases[0], tmp_path / "work")
    second, _ = vc.stage_case(cases[1], tmp_path / "work")
    assert first != second


# --------------------------------------------------------------------------
# parse_timings
# --------------------------------------------------------------------------

TIMING_LINE = ("[surgvu] timings question=0.00s decode=1.20s load_tools=2.30s "
               "load_task=0.40s tools_infer=9.10s task_infer=8.70s route=0.01s "
               "total=21.90s")


def test_parses_every_stage_in_order():
    timings = vc.parse_timings("noise\n" + TIMING_LINE + "\nmore noise\n")
    assert list(timings) == ["question", "decode", "load_tools", "load_task",
                            "tools_infer", "task_infer", "route", "total"]
    assert timings["total"] == pytest.approx(21.90)
    assert timings["tools_infer"] == pytest.approx(9.10)


def test_absent_timings_line_yields_nothing_rather_than_zeroes():
    """A run that died before printing timings must not report total=0.00s --
    that reads as an instant success."""
    assert vc.parse_timings("[surgvu] torch=2.5.1\nTraceback...\n") == {}


def test_last_timings_line_wins():
    first = "[surgvu] timings total=1.00s"
    second = "[surgvu] timings total=2.00s"
    assert vc.parse_timings(first + "\n" + second)["total"] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# read_response -- valid JSON containing a NON-EMPTY string
# --------------------------------------------------------------------------

def test_valid_response_reports_no_problem(tmp_path):
    path = tmp_path / "visual-context-response.json"
    path.write_text(json.dumps("Cadiere Forceps"))
    answer, problem = vc.read_response(path)
    assert answer == "Cadiere Forceps"
    assert problem is None


def test_missing_response_is_a_problem(tmp_path):
    answer, problem = vc.read_response(tmp_path / "visual-context-response.json")
    assert answer is None
    assert "missing" in problem


def test_bare_unquoted_text_is_a_problem(tmp_path):
    """Writing `Yes` instead of `"Yes"` is malformed JSON and fails the case
    however right the answer was."""
    path = tmp_path / "visual-context-response.json"
    path.write_text("Yes")
    answer, problem = vc.read_response(path)
    assert answer is None
    assert "JSON" in problem


def test_empty_string_response_is_a_problem(tmp_path):
    """An empty string crashes the official scorer outright."""
    path = tmp_path / "visual-context-response.json"
    path.write_text(json.dumps(""))
    answer, problem = vc.read_response(path)
    assert answer is None
    assert "empty" in problem


def test_whitespace_only_response_is_a_problem(tmp_path):
    path = tmp_path / "visual-context-response.json"
    path.write_text(json.dumps("   "))
    _, problem = vc.read_response(path)
    assert "empty" in problem


def test_non_string_response_is_a_problem(tmp_path):
    """The interface says String. An object that happens to be valid JSON is
    not an answer."""
    path = tmp_path / "visual-context-response.json"
    path.write_text(json.dumps({"answer": "Yes"}))
    answer, problem = vc.read_response(path)
    assert answer is None
    assert "string" in problem


# --------------------------------------------------------------------------
# build_command
# --------------------------------------------------------------------------

def test_command_passes_the_directories_and_models_dir():
    command = vc.build_command("/usr/bin/python3", "/repo/scripts/inference.py",
                               "/w/in", "/w/out", models_dir="/w/models",
                               device="cpu", frames=8)
    assert command[:2] == ["/usr/bin/python3", "/repo/scripts/inference.py"]
    assert "--input-dir" in command and "/w/in" in command
    assert "--output-dir" in command and "/w/out" in command
    assert "--models-dir" in command and "/w/models" in command
    assert "--device" in command and "cpu" in command
    assert "--frames" in command and "8" in command


def test_command_omits_frames_when_not_overridden():
    """No --frames means the config's own value is used; passing a default here
    would silently detach the measurement from config/perception.json."""
    command = vc.build_command("python3", "inference.py", "in", "out",
                               models_dir="m", device="auto", frames=None)
    assert "--frames" not in command


def test_command_omits_models_dir_when_not_given():
    command = vc.build_command("python3", "inference.py", "in", "out",
                               models_dir=None, device="auto", frames=None)
    assert "--models-dir" not in command


def test_extra_arguments_reach_the_entrypoint():
    """The VLM off/on comparison is the same 11 cases run twice, differing in
    nothing but these."""
    command = vc.build_command("python3", "inference.py", "in", "out",
                               extra=["--vlm", "--vlm-model", "/w/vlm"])

    assert command[-3:] == ["--vlm", "--vlm-model", "/w/vlm"]


def test_extra_arguments_go_last_so_they_cannot_displace_our_own():
    """Appended, never interleaved: a `--device` arriving through `extra`
    should lose to the one this harness sets, not silently win."""
    command = vc.build_command("python3", "inference.py", "in", "out",
                               device="cpu", extra=["--device", "cuda"])

    assert command.index("cpu") < command.index("cuda")


def test_no_extra_arguments_changes_nothing():
    assert (vc.build_command("python3", "inference.py", "in", "out")
            == vc.build_command("python3", "inference.py", "in", "out",
                                extra=[]))


# --------------------------------------------------------------------------
# summarise -- the harness must not call a failed run a pass
# --------------------------------------------------------------------------

def test_summary_counts_a_problem_case_as_invalid():
    rows = [{"case_id": "case122", "problem": None, "returncode": 0},
            {"case_id": "case123", "problem": "empty answer", "returncode": 0}]
    summary = vc.summarise(rows)
    assert summary["valid"] == 1
    assert summary["total"] == 2
    assert summary["ok"] is False


def test_summary_is_ok_only_when_every_case_is_valid():
    rows = [{"case_id": "case122", "problem": None, "returncode": 0},
            {"case_id": "case123", "problem": None, "returncode": 0}]
    assert vc.summarise(rows)["ok"] is True


def test_nonzero_exit_is_not_ok_even_with_a_valid_answer():
    """inference.py is built to exit 0 always; if it ever does not, that is a
    finding, not a detail to average away."""
    rows = [{"case_id": "case122", "problem": None, "returncode": 1}]
    assert vc.summarise(rows)["ok"] is False


def test_candidates_mapping_only_includes_answers_we_actually_read():
    rows = [{"case_id": "case122", "answer": "Yes", "problem": None},
            {"case_id": "case123", "answer": None, "problem": "missing"}]
    assert vc.candidates(rows) == {"case122": "Yes"}
