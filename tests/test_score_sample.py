# tests/test_score_sample.py
"""Pure-logic tests for the sample-set scoring harness and the trivial baselines.

Nothing here loads roberta-large. The one test that would is marked `slow`,
matching the convention already used in tests/test_scoring.py.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from answer_baselines import (  # noqa: E402
    BASELINES, GENERIC_SENTENCE, build_all_baselines, is_yes_no_question,
)
from score_sample import (  # noqa: E402
    build_pairs, format_table, load_candidates, load_sample_cases,
)


def _write_case(root, case_id, question, references, nested=True):
    case_dir = root / case_id if nested else root
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / ("%s_question.json" % case_id)).write_text(
        json.dumps(question), encoding="utf-8")
    (case_dir / ("%s.json" % case_id)).write_text(
        json.dumps(references), encoding="utf-8")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def test_load_sample_cases_reads_nested_layout(tmp_path):
    _write_case(tmp_path, "case122", "Are there forceps being used here?",
                ["No", "No forceps are being used."])
    _write_case(tmp_path, "case123", "What organ is being manipulated?",
                ["Uterine horn"])

    cases = load_sample_cases(tmp_path)

    assert sorted(cases) == ["case122", "case123"]
    assert cases["case122"].question == "Are there forceps being used here?"
    assert cases["case122"].references == ["No", "No forceps are being used."]
    assert cases["case123"].references == ["Uterine horn"]


def test_load_sample_cases_reads_flat_layout(tmp_path):
    _write_case(tmp_path, "case122", "Q one?", ["A one"], nested=False)
    _write_case(tmp_path, "case123", "Q two?", ["A two"], nested=False)

    cases = load_sample_cases(tmp_path)

    assert sorted(cases) == ["case122", "case123"]
    assert cases["case123"].question == "Q two?"


def test_load_sample_cases_does_not_treat_question_file_as_a_case(tmp_path):
    _write_case(tmp_path, "case122", "Q?", ["A"], nested=False)

    cases = load_sample_cases(tmp_path)

    assert "case122_question" not in cases


def test_load_sample_cases_rejects_an_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="no cases"):
        load_sample_cases(tmp_path)


def test_load_candidates_reads_case_to_answer_mapping(tmp_path):
    path = tmp_path / "cands.json"
    path.write_text(json.dumps({"case122": "Yes", "case123": ""}),
                    encoding="utf-8")

    assert load_candidates(path) == {"case122": "Yes", "case123": ""}


def test_load_candidates_rejects_non_string_answers(tmp_path):
    path = tmp_path / "cands.json"
    path.write_text(json.dumps({"case122": ["Yes"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="case122"):
        load_candidates(path)


# --------------------------------------------------------------------------
# the missing-candidate guard
# --------------------------------------------------------------------------

def test_build_pairs_matches_candidates_to_references(tmp_path):
    _write_case(tmp_path, "case122", "Q?", ["No", "No forceps."])
    cases = load_sample_cases(tmp_path)

    pairs = build_pairs(cases, {"case122": "Yes"})

    assert pairs == [("case122", "Yes", ["No", "No forceps."])]


def test_build_pairs_fails_loudly_when_a_case_has_no_candidate(tmp_path):
    _write_case(tmp_path, "case122", "Q?", ["A"])
    _write_case(tmp_path, "case123", "Q?", ["A"])
    cases = load_sample_cases(tmp_path)

    with pytest.raises(ValueError, match="case123"):
        build_pairs(cases, {"case122": "Yes"})


def test_build_pairs_accepts_the_empty_string_as_a_real_candidate(tmp_path):
    # "" is a baseline we deliberately measure; it must not read as "missing".
    _write_case(tmp_path, "case122", "Q?", ["A"])
    cases = load_sample_cases(tmp_path)

    pairs = build_pairs(cases, {"case122": ""})

    assert pairs == [("case122", "", ["A"])]


def test_build_pairs_rejects_candidates_for_unknown_cases(tmp_path):
    _write_case(tmp_path, "case122", "Q?", ["A"])
    cases = load_sample_cases(tmp_path)

    with pytest.raises(ValueError, match="case999"):
        build_pairs(cases, {"case122": "Yes", "case999": "Yes"})


def test_build_pairs_orders_pairs_by_case_id(tmp_path):
    for cid in ["case131", "case122", "case129"]:
        _write_case(tmp_path, cid, "Q?", ["A"])
    cases = load_sample_cases(tmp_path)

    pairs = build_pairs(cases, {c: "Yes" for c in ["case131", "case122", "case129"]})

    assert [p[0] for p in pairs] == ["case122", "case129", "case131"]


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def test_format_table_reports_the_best_reference_not_the_first(tmp_path):
    _write_case(tmp_path, "case122", "Are there forceps?", ["No", "Not at all"])
    cases = load_sample_cases(tmp_path)
    report = {
        "results": [{"case_id": "case122", "bertscore_f1": 0.42}],
        "aggregates": {"bertscore_f1": 0.42},
    }

    text = format_table(cases, {"case122": "Nope"}, report,
                        best_refs={"case122": "Not at all"})

    assert "Not at all" in text
    assert "0.4200" in text
    assert "MEAN" in text


# --------------------------------------------------------------------------
# the yes/no question detector
# --------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "Is a suture required in this surgical step?",
    "Are there forceps being used here?",
    "Was a large needle driver used in this clip?",
    "Does the clip show cutting?",
    "Do the tools include a needle driver?",
])
def test_is_yes_no_question_accepts_polar_openers(question):
    assert is_yes_no_question(question) is True


@pytest.mark.parametrize("question", [
    "What type of forceps is mentioned?",
    "What organ is being manipulated?",
    "What is the purpose of using forceps in this procedure?",
    "How many instruments are visible?",
    "",
])
def test_is_yes_no_question_rejects_open_questions(question):
    assert is_yes_no_question(question) is False


def test_is_yes_no_question_ignores_case_and_leading_space():
    assert is_yes_no_question("  is a suture required?") is True


def test_is_yes_no_question_matches_whole_words_only():
    # "Isolating"/"Doesn't-a-word" must not be read as the opener "Is"/"Does".
    assert is_yes_no_question("Isolating which vessel is shown?") is False
    assert is_yes_no_question("Arent questions like this open?") is False


# --------------------------------------------------------------------------
# the baselines themselves
# --------------------------------------------------------------------------

def test_baseline_names_are_the_six_we_calibrate_against():
    assert sorted(BASELINES) == [
        "always_no", "always_yes", "echo_question", "empty",
        "generic", "yesno_aware",
    ]


def test_trivial_baselines_ignore_the_question():
    assert BASELINES["always_yes"]("anything?") == "Yes"
    assert BASELINES["always_no"]("anything?") == "No"
    assert BASELINES["empty"]("anything?") == ""
    assert BASELINES["generic"]("anything?") == GENERIC_SENTENCE


def test_echo_question_returns_the_question_verbatim():
    assert BASELINES["echo_question"]("What organ is being manipulated?") == \
        "What organ is being manipulated?"


def test_yesno_aware_answers_yes_only_to_polar_questions():
    assert BASELINES["yesno_aware"]("Is a suture required?") == "Yes"
    assert BASELINES["yesno_aware"]("What organ is shown?") == GENERIC_SENTENCE


def test_build_all_baselines_covers_every_case_for_every_strategy():
    questions = {"case122": "Are there forceps?", "case124": "What forceps?"}

    out = build_all_baselines(questions)

    assert sorted(out) == sorted(BASELINES)
    for name, answers in out.items():
        assert sorted(answers) == ["case122", "case124"], name
    assert out["yesno_aware"] == {
        "case122": "Yes", "case124": GENERIC_SENTENCE,
    }


# --------------------------------------------------------------------------
# slow: needs roberta-large
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_score_sample_end_to_end(tmp_path):
    from surgvu.scoring import Scorer

    _write_case(tmp_path, "case122", "Are there forceps?", ["No", "No forceps."])
    cases = load_sample_cases(tmp_path)
    pairs = build_pairs(cases, {"case122": "No"})
    report = Scorer().score_many(pairs)

    assert report["aggregates"]["bertscore_f1"] > 0.9
