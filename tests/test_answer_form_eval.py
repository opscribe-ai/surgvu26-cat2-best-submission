# tests/test_answer_form_eval.py
"""Pure-logic tests for the answer-FORM experiment and its fallback arms.

Nothing here loads roberta-large. The two tests that read the real gold are
skipped when /staging is not mounted rather than marked slow, because reading
eleven small JSON files is not slow -- it is just unavailable on some nodes.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from answer_form_eval import (  # noqa: E402
    BARE_TOKEN_MAX_WORDS, CONDITIONS, aggregate, condition_indices,
    condition_score, fallback_arms, is_bare_first_reference, load_forms,
    load_perception, per_reference_scores, score_rows, select_references,
)
from score_sample import SampleCase  # noqa: E402
from surgvu.router import (  # noqa: E402
    FALLBACK_OPEN, perception_sentence,
)

FORMS_FIXTURE = REPO / "tests" / "fixtures" / "answer_forms.json"
# THE TRACKED COPY, not the one in the repo root. Both exist and are
# byte-identical, but only outputs/vlm/ is under version control -- the root
# one is a job artefact that landed in the submit directory and is now
# gitignored with the rest of them. Pointing here means the test can actually
# RUN in the container, where only tracked files are transferred; it had been
# failing as "environmental" since it was written, which is two of the five
# permanently-red results that this suite's own comment warns teach people to
# ignore it.
SHIPPED = REPO / "outputs" / "vlm" / "shipped_candidates.json"
SAMPLE_ROOT = Path("/staging/groups/bhaskar_opscribe/surgvu/cat2_sample")

FIVE = ["No", "No, forceps are not mentioned.", "No forceps are being used.",
        "No, there's no indication of forceps.", "No forceps are listed."]


# --------------------------------------------------------------------------
# the reference conditions
# --------------------------------------------------------------------------

def test_a_full_keeps_every_reference():
    assert condition_indices("A_full", 5) == (0, 1, 2, 3, 4)


def test_b_drops_only_the_first_reference():
    assert condition_indices("B_drop_terse", 5) == (1, 2, 3, 4)


def test_c_keeps_exactly_one_sentence_reference():
    assert condition_indices("C_one_sentence", 5) == (1,)


def test_d_keeps_only_the_terse_reference():
    assert condition_indices("D_terse_only", 5) == (0,)


def test_every_named_condition_is_implemented():
    for condition in CONDITIONS:
        assert condition_indices(condition, 5)


def test_an_unknown_condition_is_an_error_not_a_default():
    with pytest.raises(ValueError):
        condition_indices("B-drop-terse", 5)


def test_a_case_with_no_references_is_an_error():
    with pytest.raises(ValueError):
        condition_indices("A_full", 0)


def test_select_references_returns_the_chosen_strings():
    assert select_references(FIVE, "C_one_sentence") == [FIVE[1]]
    assert select_references(FIVE, "B_drop_terse") == FIVE[1:]


def test_selecting_nothing_raises_instead_of_scoring_zero():
    """A single-reference case under B_drop_terse must stop the run.

    Scorer.score_one returns 0.0 for an empty reference list, so a silent
    empty selection would be reported as a form that scored 0.0000.
    """
    with pytest.raises(ValueError):
        select_references(["Yes"], "B_drop_terse")


# --------------------------------------------------------------------------
# grouping and derivation
# --------------------------------------------------------------------------

def test_a_bare_token_first_reference_is_recognised():
    assert is_bare_first_reference(["Yes", "Yes, sutures are required."])
    assert is_bare_first_reference(["Uterine horn", "..."])


def test_a_sentence_first_reference_is_not_bare():
    assert not is_bare_first_reference(
        ["To grasp and hold tissues or objects during the surgery.", "..."])


def test_the_bare_boundary_is_inclusive_and_the_next_word_crosses_it():
    words = ["w"] * BARE_TOKEN_MAX_WORDS
    assert is_bare_first_reference([" ".join(words)])
    assert not is_bare_first_reference([" ".join(words + ["w"])])


def test_no_references_is_not_bare():
    assert not is_bare_first_reference([])


def test_condition_score_is_the_max_over_the_kept_references_only():
    vector = [0.9, 0.4, 0.5, 0.3, 0.2]
    assert condition_score(vector, "A_full") == pytest.approx(0.9)
    assert condition_score(vector, "B_drop_terse") == pytest.approx(0.5)
    assert condition_score(vector, "C_one_sentence") == pytest.approx(0.4)
    assert condition_score(vector, "D_terse_only") == pytest.approx(0.9)


# --------------------------------------------------------------------------
# the scoring loop and its independence self-check
# --------------------------------------------------------------------------

class _FakeScorer:
    """Scores reference "rN" as vector[N]; `batch` overrides the max call."""

    def __init__(self, vector, batch=None):
        self.vector = list(vector)
        self.batch = batch
        self.calls = []

    def _value(self, reference):
        return self.vector[int(str(reference).split()[0][1:])]

    def score_one(self, candidate, references):
        self.calls.append((candidate, tuple(references)))
        if len(references) == 1:
            return {"bertscore_f1": self._value(references[0])}
        if self.batch is not None:
            return {"bertscore_f1": self.batch}
        return {"bertscore_f1": max(self._value(r) for r in references)}


def _indexed_case(case_id="case001", n=5, first=None):
    """A case whose references name their own index, for the fake scorer."""
    references = ["r%d" % i for i in range(n)]
    if first is not None:
        references[0] = first
    return {case_id: SampleCase(case_id, "Q?", references)}


def test_score_rows_derives_every_condition_from_one_vector():
    cases = _indexed_case()
    scorer = _FakeScorer([0.9, 0.4, 0.5, 0.3, 0.2])
    rows = score_rows(scorer, {"terse": {"case001": "Yes"}}, cases)
    assert len(rows) == 1
    assert rows[0]["scores"]["B_drop_terse"] == pytest.approx(0.5)
    assert rows[0]["best_index"] == 0
    assert rows[0]["group"] == "bare"


def test_score_rows_groups_a_case_by_the_shape_of_its_first_reference():
    cases = _indexed_case(first="r0 spelled out at some length indeed")
    scorer = _FakeScorer([0.9, 0.4, 0.5, 0.3, 0.2])
    rows = score_rows(scorer, {"terse": {"case001": "Yes"}}, cases)
    assert rows[0]["group"] == "phrase"


def test_score_rows_fails_loudly_when_pairs_are_not_independent():
    cases = _indexed_case()
    scorer = _FakeScorer([0.9, 0.4, 0.5, 0.3, 0.2], batch=0.2)
    with pytest.raises(AssertionError):
        score_rows(scorer, {"terse": {"case001": "Yes"}}, cases)


def test_score_rows_refuses_a_candidate_set_missing_a_case():
    cases = _indexed_case()
    cases.update(_indexed_case("case002"))
    scorer = _FakeScorer([0.9, 0.4, 0.5, 0.3, 0.2])
    with pytest.raises(ValueError):
        score_rows(scorer, {"terse": {"case001": "Yes"}}, cases)


def test_per_reference_scores_asks_for_one_reference_at_a_time():
    scorer = _FakeScorer([0.1, 0.2, 0.3, 0.4, 0.5])
    assert per_reference_scores(scorer, "Yes", ["r0", "r1", "r2", "r3", "r4"]) == [
        0.1, 0.2, 0.3, 0.4, 0.5]
    assert all(len(refs) == 1 for _cand, refs in scorer.calls)


def test_aggregate_means_within_a_group_only():
    rows = [
        {"set": "terse", "group": "bare",
         "scores": {c: 1.0 for c in CONDITIONS}},
        {"set": "terse", "group": "phrase",
         "scores": {c: 0.0 for c in CONDITIONS}},
    ]
    assert aggregate(rows)["terse"]["A_full"] == pytest.approx(0.5)
    assert aggregate(rows, "bare")["terse"]["A_full"] == pytest.approx(1.0)
    assert aggregate(rows, "phrase")["terse"]["A_full"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# the hand-written fixture
# --------------------------------------------------------------------------

def test_the_terse_form_is_byte_identical_to_the_shipped_answers():
    """The experiment's baseline arm has to BE the thing we ship."""
    forms = load_forms(FORMS_FIXTURE)
    assert forms["terse"] == json.loads(SHIPPED.read_text(encoding="utf-8"))


def test_every_form_covers_exactly_the_same_cases():
    forms = load_forms(FORMS_FIXTURE)
    expected = set(forms["terse"])
    assert len(expected) == 11
    for name, answers in forms.items():
        assert set(answers) == expected, name


def test_no_form_carries_an_empty_or_multiline_answer():
    for name, answers in load_forms(FORMS_FIXTURE).items():
        for case_id, answer in answers.items():
            assert answer.strip(), (name, case_id)
            assert answer == " ".join(answer.split()), (name, case_id)


def test_the_hedged_form_leads_with_the_terse_token():
    """Terse-plus-clause means the terse answer is still the first thing said."""
    forms = load_forms(FORMS_FIXTURE)
    for case_id, terse in forms["terse"].items():
        if len(terse.split()) > BARE_TOKEN_MAX_WORDS:
            continue
        assert forms["hedged"][case_id].startswith(terse), case_id


def test_the_sentence_form_does_not_lead_with_a_polar_token():
    """Otherwise 'sentence' and 'hedged' would be the same arm on polar cases."""
    forms = load_forms(FORMS_FIXTURE)
    for case_id, terse in forms["terse"].items():
        if terse not in ("Yes", "No"):
            continue
        assert not forms["sentence"][case_id].lower().startswith(
            ("yes", "no,", "no ")), case_id


def test_every_form_preserves_the_polarity_of_the_shipped_answer():
    """Form, not correctness. A 'No' case must stay negative in all three."""
    forms = load_forms(FORMS_FIXTURE)
    for case_id, terse in forms["terse"].items():
        if terse not in ("Yes", "No"):
            continue
        negated = terse == "No"
        for name in ("hedged", "sentence"):
            text = forms[name][case_id].lower()
            assert (" not " in text or text.startswith("no,")) is negated, (
                name, case_id)


@pytest.mark.skipif(not SAMPLE_ROOT.exists(), reason="sample set not mounted")
def test_no_hand_written_answer_copies_a_gold_reference_verbatim():
    """A form that reproduces the answer key would score 1.0000 by cheating.

    The terse form is exempt: it is the SHIPPED answer, and 8 of the 11 really
    are byte-identical to reference[0]. That is the finding under test, not a
    leak.
    """
    from score_sample import load_sample_cases

    cases = load_sample_cases(SAMPLE_ROOT)
    forms = load_forms(FORMS_FIXTURE)
    for name in ("hedged", "sentence"):
        for case_id, answer in forms[name].items():
            assert answer not in cases[case_id].references, (name, case_id)


@pytest.mark.skipif(not SAMPLE_ROOT.exists(), reason="sample set not mounted")
def test_the_public_sample_really_does_lead_with_a_bare_token():
    """The premise of the whole terse bet, checked rather than remembered."""
    from score_sample import load_sample_cases

    cases = load_sample_cases(SAMPLE_ROOT)
    bare = [c for c in cases.values() if is_bare_first_reference(c.references)]
    assert len(bare) == 9
    assert all(len(c.references) == 5 for c in cases.values())


# --------------------------------------------------------------------------
# the perception-derived fallback
# --------------------------------------------------------------------------

def test_a_single_tool_takes_a_singular_verb():
    assert perception_sentence(
        {"tools_present": ["needle driver"], "task_top": "suturing"}
    ) == "Needle driver is in use during suturing."


def test_several_tools_are_joined_and_take_a_plural_verb():
    assert perception_sentence(
        {"tools_present": ["needle driver", "cadiere forceps",
                           "monopolar curved scissors"],
         "task_top": "suturing"}
    ) == ("Cadiere forceps, monopolar curved scissors and needle driver "
          "are in use during suturing.")


def test_tools_lead_with_the_most_likely_class_not_the_alphabetical_one():
    """The pair matters: prior order and alphabetical order must DISAGREE.

    A first attempt used stapler + cadiere forceps, where both orderings give
    the same string, and a mutation swapping the prior for `sorted` survived.
    Needle driver (corpus prior .398) outranks grasping retractor (.141) while
    sorting after it.
    """
    sentence = perception_sentence(
        {"tools_present": ["grasping retractor", "needle driver"],
         "task_top": "suturing"})
    assert sentence.startswith("Needle driver and grasping retractor")


def test_a_tool_name_with_a_slash_is_written_out_for_a_reader():
    """The one class where the display table and the taxonomy key differ."""
    assert perception_sentence(
        {"tools_present": ["permanent cautery hook/spatula"]}
    ) == "Permanent cautery hook is in use in this procedure."


def test_the_task_name_is_spelled_out_not_the_taxonomy_key():
    assert "rectal artery and vein dissection" in perception_sentence(
        {"tools_present": ["bipolar forceps"], "task_top": "rectal artery/vein"})


def test_no_task_still_names_the_tools():
    assert perception_sentence({"tools_present": ["needle driver"]}) == (
        "Needle driver is in use in this procedure.")


def test_no_tools_still_names_the_task():
    assert perception_sentence(
        {"tools_present": [], "task_top": "suturing"}
    ) == "The procedure involves suturing."


def test_an_empty_record_degrades_to_the_shipped_constant():
    assert perception_sentence({}) == FALLBACK_OPEN


def test_the_ablation_arm_drops_the_tool_nouns_and_keeps_the_task():
    record = {"tools_present": ["needle driver"], "task_top": "suturing"}
    assert perception_sentence(record, include_tools=False) == (
        "The procedure involves suturing.")


def test_the_sentence_is_clip_specific():
    """The whole point: two different clips must not produce one string."""
    a = perception_sentence({"tools_present": ["needle driver"],
                             "task_top": "suturing"})
    b = perception_sentence({"tools_present": ["stapler"],
                             "task_top": "uterine horn"})
    assert a != b
    assert a != FALLBACK_OPEN and b != FALLBACK_OPEN


def test_fallback_arms_pin_the_generic_to_the_shipped_constant():
    arms = fallback_arms({"case001": {"tools_present": ["needle driver"],
                                      "task_top": "suturing"}}, ["case001"])
    assert arms["fb_generic"]["case001"] == FALLBACK_OPEN
    assert arms["fb_perception"]["case001"] == (
        "Needle driver is in use during suturing.")
    assert arms["fb_task_only"]["case001"] == "The procedure involves suturing."


def test_fallback_arms_tolerate_a_case_with_no_perception_record():
    arms = fallback_arms({}, ["case001"])
    assert arms["fb_perception"]["case001"] == FALLBACK_OPEN


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------

def test_load_forms_rejects_a_file_with_no_forms(tmp_path):
    path = tmp_path / "forms.json"
    path.write_text(json.dumps({"_note": "nothing here"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_forms(path)


def test_load_forms_rejects_an_empty_form(tmp_path):
    path = tmp_path / "forms.json"
    path.write_text(json.dumps({"forms": {"terse": {}}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_forms(path)


def test_load_perception_accepts_both_a_bare_map_and_a_records_wrapper(tmp_path):
    record = {"case001": {"task_top": "suturing"}}
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(record), encoding="utf-8")
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"provenance": "job 1", "records": record}),
                       encoding="utf-8")
    assert load_perception(bare) == record
    assert load_perception(wrapped) == record


# --------------------------------------------------------------------------
# the entry point, exercised end to end without roberta-large
# --------------------------------------------------------------------------

def test_main_runs_the_whole_report_against_a_stub_scorer(tmp_path, monkeypatch,
                                                          capsys):
    """Catches a formatting or wiring bug in one second instead of one job.

    The real run costs several minutes of roberta-large start-up on a compute
    node; every crash it could suffer that is not the metric's fault is
    reachable from here.
    """
    import surgvu.scoring as scoring

    class _Stub:
        def score_one(self, candidate, references):
            return {"bertscore_f1": 0.5 if len(references) == 1
                    else 0.5}

    monkeypatch.setattr(scoring, "Scorer", _Stub)

    root = tmp_path / "sample"
    for case_id in load_forms(FORMS_FIXTURE)["terse"]:
        case_dir = root / case_id
        case_dir.mkdir(parents=True)
        (case_dir / ("%s_question.json" % case_id)).write_text(
            json.dumps("Q?"), encoding="utf-8")
        (case_dir / ("%s.json" % case_id)).write_text(
            json.dumps(["Yes", "s1", "s2", "s3", "s4"]), encoding="utf-8")

    out = tmp_path / "results.json"
    from answer_form_eval import main

    assert main([str(root), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    for name in ("terse", "hedged", "sentence", "fb_generic", "fb_perception",
                 "fb_task_only"):
        assert name in printed
    for condition in CONDITIONS:
        assert condition in printed
    written = json.loads(out.read_text(encoding="utf-8"))
    assert len(written["rows"]) == 6 * 11
    assert set(written["aggregate_all"]) == {
        "terse", "hedged", "sentence", "fb_generic", "fb_perception",
        "fb_task_only"}


@pytest.mark.skipif(not SAMPLE_ROOT.exists(), reason="sample set not mounted")
def test_no_sample_question_reaches_either_unknown_fallback():
    """The licence to change the fallback without re-running the container.

    `_answer_unknown_open` and `_answer_unknown_polar` are the only forms a
    fallback change can touch, so if no sample question classifies into them,
    every one of the 11 shipped answers is byte-identical whatever the
    fallback becomes. Pinned here rather than argued in a report, because it
    is the premise the whole fallback experiment rests on.
    """
    from surgvu.router import (
        INTENT_UNKNOWN_OPEN, INTENT_UNKNOWN_POLAR, classify_question,
    )
    from score_sample import load_sample_cases

    cases = load_sample_cases(SAMPLE_ROOT)
    assert len(cases) == 11
    for case_id, case in cases.items():
        assert classify_question(case.question) not in (
            INTENT_UNKNOWN_OPEN, INTENT_UNKNOWN_POLAR), case_id
