# tests/test_router.py
"""Pure-logic tests for the question router. No torch, no video, no model.

Every test here runs in milliseconds on a login node, which is the point: the
router is the half of the system that can be iterated on without a GPU.
"""
import json

import pytest

from surgvu.router import (
    CUTTING_TOOLS, DIVIDING_TOOLS, FALLBACK_OPEN, FALLBACK_POLAR,
    GENERIC_TOOL_TERMS, INTENT_APPROACH, INTENT_COUNT, INTENT_CUTTING,
    INTENT_ORGAN,
    INTENT_PROCEDURE, INTENT_PURPOSE, INTENT_SUTURE, INTENT_TASK,
    INTENT_TASK_CONFIRM,
    INTENT_TOOL_IDENTITY, INTENT_TOOL_PRESENCE, INTENT_UNKNOWN_OPEN,
    INTENT_UNKNOWN_POLAR, MODAL_TOOL_COUNT, PERCEPTION_INDEPENDENT_INTENTS,
    SOFT_PRESENCE_MAX_CLASSES,
    ANSWER_FORMS, PROCEDURE_ANSWER, TASK_ORGANS, TOOL_PRIOR_ORDER, answer_question,
    choose_display_name, classify_question, credible_tools, display_name,
    finalize_answer, is_polar_question, load_commercial_names,
    load_variant_priors, mentioned_tool_classes,
    mentioned_tool_terms, organ_for_task, task_top, tools_present,
)
from surgvu.taxonomy import TASK_CLASSES, TOOL_CLASSES


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def perception(tools_present=(), task_top=None, tools=None, task=None,
               n_frames=30):
    """A minimal perception dict in the fixed contract's shape."""
    out = {
        "tools": tools if tools is not None else {c: 0.0 for c in TOOL_CLASSES},
        "tools_present": list(tools_present),
        "task": task if task is not None else {c: 0.0 for c in TASK_CLASSES},
        "task_top": task_top,
        "n_frames": n_frames,
    }
    return out


# --------------------------------------------------------------------------
# polarity detection
# --------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "Is a suture required in this surgical step?",
    "Are there forceps being used here?",
    "Was a large needle driver used in this clip?",
    "Were forceps used?",
    "Does the clip show cutting?",
    "Do the tools include a needle driver?",
    "Did the surgeon use scissors?",
    "Has a clip applier been used?",
    "Can you see a stapler?",
])
def test_is_polar_question_accepts_polar_openers(question):
    assert is_polar_question(question) is True


@pytest.mark.parametrize("question", [
    "What type of forceps is mentioned?",
    "What organ is being manipulated?",
    "Which instrument is in view?",
    "How many instruments are visible?",
    "",
])
def test_is_polar_question_rejects_open_questions(question):
    assert is_polar_question(question) is False


def test_is_polar_question_matches_whole_words_only():
    # "Isolating"/"Doesn't" must not read as the openers "Is"/"Does".
    assert is_polar_question("Isolating which vessel is shown?") is False
    assert is_polar_question("Arent questions like this open?") is False


def test_is_polar_question_tolerates_leading_space_and_case():
    assert is_polar_question("  iS a suture required?") is True


def test_is_polar_question_survives_none():
    assert is_polar_question(None) is False


# --------------------------------------------------------------------------
# A politeness frame in front of an OPEN question.
#
# This is the most expensive mistake this module can make, and it is measured
# rather than assumed. Substituting "Yes" for the answer on the three open
# sample questions and scoring with the real metric (cluster 9624729):
#
#     case127  "What organ is being manipulated?"        1.0000 ->  0.0478
#     case129  "What procedure is this summary ...?"     1.0000 -> -0.0649
#     case130  "What is the purpose of using forceps?"   1.0000 -> -0.0585
#
# versus 0.3497-0.4790 for the generic open fallback on the same three. Two of
# the three go NEGATIVE. Reading a genuinely polar question as open is
# cheaper: it replaces a 1.0000 "Yes" with that ~0.43 sentence. So the guard
# below errs toward "open", and the two directions are pinned separately.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "Can you identify the organ being manipulated?",
    "Could you describe the procedure being performed?",
    "Can you name the instruments used in this clip?",
    "Can you tell me what type of forceps is mentioned?",
    "Do you know what task is being performed?",
    "Would you say what organ is shown?",
    "Could you list the tools in view?",
])
def test_a_politeness_frame_does_not_make_an_open_question_polar(question):
    assert is_polar_question(question) is False


@pytest.mark.parametrize("question", [
    # From the held-out battery, and genuinely polar: "confirm" asks yes/no.
    "Can you confirm there is no monopolar curved scissors cutting here?",
    "Can you see a stapler?",
    "Do you see a needle driver?",
    "Will you use the stapler?",
    # The verb is an information-request verb, but " as " makes it yes/no.
    "Would you describe this as suturing?",
    "Can you tell if a stapler was used?",
])
def test_a_politeness_frame_over_a_yes_no_question_stays_polar(question):
    assert is_polar_question(question) is True


def test_the_politeness_frame_reaches_the_answer(perception=None):
    """The whole point: the organ question keeps its organ answer."""
    record = {"task_top": "uterine horn", "tools_present": []}
    assert answer_question("Can you identify the organ being manipulated?",
                           record) == "Uterine horn"
    assert answer_question("Could you describe the procedure being performed?",
                           record) == PROCEDURE_ANSWER


# --------------------------------------------------------------------------
# tool-term resolution: generic terms map to a SET of classes
# --------------------------------------------------------------------------

def test_generic_forceps_resolves_to_every_forceps_class():
    assert mentioned_tool_classes("Are there forceps being used here?") == frozenset({
        "bipolar forceps", "cadiere forceps", "force bipolar", "prograsp forceps",
    })


def test_a_specific_forceps_does_not_also_drag_in_the_generic_set():
    # "cadiere forceps" must consume the word "forceps" so the generic rule
    # cannot re-add bipolar/prograsp/force bipolar.
    assert mentioned_tool_classes("Are cadiere forceps used here?") == \
        frozenset({"cadiere forceps"})


def test_a_commercial_variant_name_resolves_to_its_class():
    assert mentioned_tool_classes("Was a large needle driver used in this clip?") == \
        frozenset({"needle driver"})
    assert mentioned_tool_classes("Was a Maryland Bipolar Forceps used?") == \
        frozenset({"bipolar forceps"})


def test_longer_phrases_win_over_the_shorter_ones_they_contain():
    # "force bipolar" is its own class and must not be read as generic "bipolar".
    assert mentioned_tool_classes("Is force bipolar in use?") == \
        frozenset({"force bipolar"})


def test_scissors_resolves_to_the_only_scissors_class():
    assert mentioned_tool_classes("Are scissors being used?") == \
        frozenset({"monopolar curved scissors"})


def test_no_tool_mentioned_returns_an_empty_set():
    assert mentioned_tool_classes("Is tissue being cut during this clip?") == frozenset()


def test_tool_terms_are_matched_on_word_boundaries():
    assert mentioned_tool_classes("Is the stapleriffic device used?") == frozenset()


def test_tool_mentions_ignore_case_and_punctuation():
    assert mentioned_tool_classes("FORCEPS, are they used?") == \
        mentioned_tool_classes("forceps are they used")


def test_mentioned_tool_terms_reports_the_phrase_that_matched():
    terms = mentioned_tool_terms("What is the purpose of using forceps here?")
    assert [phrase for phrase, _classes in terms] == ["forceps"]


def test_every_generic_term_maps_only_to_real_tool_classes():
    for term, classes in GENERIC_TOOL_TERMS.items():
        assert classes, term
        for cls in classes:
            assert cls in TOOL_CLASSES, (term, cls)


# --------------------------------------------------------------------------
# intent classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize("question,intent", [
    # the 11 public sample questions
    ("Are there forceps being used here?", INTENT_TOOL_PRESENCE),
    ("Is a large needle driver among the listed tools?", INTENT_TOOL_PRESENCE),
    ("What type of forceps is mentioned?", INTENT_TOOL_IDENTITY),
    ("Is a suture required in this surgical step?", INTENT_SUTURE),
    ("Was a large needle driver used in this clip?", INTENT_TOOL_PRESENCE),
    ("What organ is being manipulated?", INTENT_ORGAN),
    ("Is a needle driver involved in the procedure?", INTENT_TOOL_PRESENCE),
    ("What procedure is this summary describing?", INTENT_PROCEDURE),
    ("What is the purpose of using forceps in this procedure?", INTENT_PURPOSE),
    ("Is tissue being cut during this clip?", INTENT_CUTTING),
    ("Was a large needle driver used during the surgery?", INTENT_TOOL_PRESENCE),
])
def test_classify_the_eleven_sample_questions(question, intent):
    assert classify_question(question) == intent


def test_purpose_is_classified_before_procedure():
    # A compound question that genuinely trips BOTH rules: "what procedure"
    # matches the procedure pattern and "purpose" matches the purpose one.
    # Purpose wins, because naming the procedure answers neither half while a
    # purpose sentence at least answers one.
    assert classify_question("What procedure is this, and what is its purpose?") \
        == INTENT_PURPOSE
    # And the sample question, which contains the word "procedure" in passing.
    assert classify_question("What is the purpose of using forceps in this procedure?") \
        == INTENT_PURPOSE


def test_a_tool_mention_outranks_the_suture_rule_on_a_polar_question():
    # "needle driver" contains "needle"; it is a tool question, not a suturing one.
    assert classify_question("Is a needle driver involved in the procedure?") == \
        INTENT_TOOL_PRESENCE


def test_task_questions_are_classified_as_task():
    assert classify_question("What task is being performed?") == INTENT_TASK


def test_unknown_questions_fall_back_by_polarity():
    assert classify_question("Would the trainee benefit?") == INTENT_UNKNOWN_POLAR
    # RETARGETED. This assertion used to read "How many instruments are
    # visible?", which was the router's deliberate counting gap. The gap is
    # closed (see INTENT_COUNT and test_counting_questions_get_their_own_intent
    # below); the example was replaced, the rule it protects was not.
    assert classify_question("What is the patient's diagnosis?") == INTENT_UNKNOWN_OPEN


def test_classify_survives_empty_and_none():
    assert classify_question("") == INTENT_UNKNOWN_OPEN
    assert classify_question(None) == INTENT_UNKNOWN_OPEN


# --------------------------------------------------------------------------
# the surface form of a tool name
# --------------------------------------------------------------------------

def test_choose_display_name_takes_a_dominant_variant():
    variants = [{"name": "Large Clip Applier", "p_present": 0.984},
                {"name": "Small Clip Applier", "p_present": 0.003}]
    assert choose_display_name(variants, "clip applier") == "Large Clip Applier"


def test_choose_display_name_refuses_a_coin_flip_and_uses_the_class_name():
    variants = [{"name": "Maryland Bipolar Forceps", "p_present": 0.540},
                {"name": "Fenestrated Bipolar Forceps", "p_present": 0.456}]
    assert choose_display_name(variants, "bipolar forceps") == "Bipolar Forceps"


def test_choose_display_name_prefers_the_shorter_of_two_near_certain_variants():
    # Both are present in ~85% of clips holding a needle driver, so betting on
    # the longer one buys nothing and costs precision if the gold is the short
    # one. Terseness breaks the tie.
    variants = [{"name": "Large SutureCut Needle Driver", "p_present": 0.866},
                {"name": "Large Needle Driver", "p_present": 0.824}]
    assert choose_display_name(variants, "needle driver") == "Large Needle Driver"


def test_choose_display_name_falls_back_when_there_are_no_variants():
    assert choose_display_name([], "vessel sealer") == "Vessel Sealer"


def test_display_name_uses_the_measured_priors_shipped_in_config():
    assert display_name("cadiere forceps") == "Cadiere Forceps"
    assert display_name("needle driver") == "Large Needle Driver"
    assert display_name("bipolar forceps") == "Bipolar Forceps"


def test_display_name_preserves_vendor_casing_that_title_case_would_destroy():
    # str.title() would produce "Prograsp Forceps"; casing alone is worth ~0.2
    # BERTScore, so the capital G is not cosmetic.
    assert display_name("prograsp forceps") == "ProGrasp Forceps"


def test_display_name_never_returns_an_empty_string_for_any_class():
    for cls in TOOL_CLASSES:
        assert display_name(cls).strip()


def test_display_name_of_an_unknown_class_is_still_usable():
    assert display_name("chainsaw").strip()


def test_load_variant_priors_returns_an_empty_mapping_when_the_file_is_absent(tmp_path):
    assert load_variant_priors(tmp_path / "nope.json") == {}


def test_load_variant_priors_returns_an_empty_mapping_on_malformed_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_variant_priors(path) == {}


def test_load_variant_priors_reads_the_classes_block(tmp_path):
    path = tmp_path / "priors.json"
    path.write_text(json.dumps({"classes": {"stapler": {"variants": []}}}),
                    encoding="utf-8")
    assert load_variant_priors(path) == {"stapler": {"variants": []}}


# --------------------------------------------------------------------------
# the answer-form layer
# --------------------------------------------------------------------------

def test_finalize_answer_collapses_whitespace():
    assert finalize_answer("  Cadiere   Forceps \n") == "Cadiere Forceps"


def test_finalize_answer_capitalises_the_first_character():
    assert finalize_answer("yes") == "Yes"


def test_finalize_answer_does_not_lowercase_the_rest():
    assert finalize_answer("proGrasp Forceps") == "ProGrasp Forceps"


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_finalize_answer_never_returns_an_empty_string(bad):
    # An empty candidate crashes the official scorer in our environment.
    assert finalize_answer(bad) == FALLBACK_OPEN
    assert finalize_answer(bad).strip()


# --------------------------------------------------------------------------
# tool-presence questions
# --------------------------------------------------------------------------

def test_tool_presence_says_yes_when_the_named_class_is_present():
    answer = answer_question("Is a needle driver involved in the procedure?",
                             perception(tools_present=["needle driver"]))
    assert answer == "Yes"


def test_tool_presence_says_no_when_the_named_class_is_absent():
    answer = answer_question("Is a needle driver involved in the procedure?",
                             perception(tools_present=["cadiere forceps"]))
    assert answer == "No"


def test_a_generic_term_is_satisfied_by_any_member_of_its_set():
    answer = answer_question("Are there forceps being used here?",
                             perception(tools_present=["prograsp forceps"]))
    assert answer == "Yes"


def test_a_generic_term_is_not_satisfied_by_a_non_member():
    # A grasping retractor is a grasper, not a forceps.
    answer = answer_question("Are there forceps being used here?",
                             perception(tools_present=["grasping retractor",
                                                       "monopolar curved scissors"]))
    assert answer == "No"


def test_the_large_needle_driver_variant_is_answered_from_the_class():
    # Measured: given the needle-driver CLASS in a clip, a literal "Large
    # Needle Driver" is installed on some arm 0.824 of the time.
    assert answer_question("Was a large needle driver used in this clip?",
                           perception(tools_present=["needle driver"])) == "Yes"
    assert answer_question("Was a large needle driver used in this clip?",
                           perception(tools_present=["cadiere forceps"])) == "No"


def test_a_polar_question_about_an_unknown_tool_still_answers_yes_or_no():
    # A scalpel is not in the 12-class taxonomy, so there is nothing to check.
    # The right move is the calibrated polar guess, not an open-question
    # sentence: the reference list for a polar question leads with "Yes"/"No".
    assert answer_question("Is a scalpel being used here?",
                           perception(tools_present=["cadiere forceps"])) == \
        FALLBACK_POLAR


def test_the_presence_answer_form_guesses_when_handed_no_recognisable_tool():
    # The answer forms are addressable on their own, so this exercises the
    # form directly rather than through classify_question -- which currently
    # only routes here when a tool WAS recognised. If that routing rule is
    # ever relaxed, this branch is what stops the router answering "No" to
    # every question about an instrument outside the 12 classes.
    form = ANSWER_FORMS[INTENT_TOOL_PRESENCE]
    assert form("Is a scalpel being used?", perception()) == FALLBACK_POLAR


def test_every_intent_has_an_answer_form():
    for intent in [INTENT_TOOL_PRESENCE, INTENT_TOOL_IDENTITY, INTENT_ORGAN,
                   INTENT_CUTTING, INTENT_SUTURE, INTENT_PROCEDURE,
                   INTENT_PURPOSE, INTENT_TASK, INTENT_UNKNOWN_POLAR,
                   INTENT_UNKNOWN_OPEN]:
        assert intent in ANSWER_FORMS


def test_tool_presence_answers_are_bare_tokens_not_sentences():
    answer = answer_question("Are there forceps being used here?",
                             perception(tools_present=["cadiere forceps"]))
    assert answer in ("Yes", "No")


# --------------------------------------------------------------------------
# cutting questions
# --------------------------------------------------------------------------

def test_cutting_is_yes_when_a_cutting_instrument_is_present():
    assert answer_question("Is tissue being cut during this clip?",
                           perception(tools_present=["monopolar curved scissors"])) == "Yes"


def test_cutting_is_no_when_only_grasping_instruments_are_present():
    assert answer_question("Is tissue being cut during this clip?",
                           perception(tools_present=["cadiere forceps",
                                                     "prograsp forceps"])) == "No"


def test_a_dividing_instrument_counts_as_cutting():
    # A vessel sealer seals and then divides; the tissue does end up cut.
    assert answer_question("Is tissue being cut during this clip?",
                           perception(tools_present=["vessel sealer"])) == "Yes"


def test_bipolar_energy_alone_is_not_cutting():
    # Bipolar forceps cauterise but do not divide.
    assert answer_question("Is tissue being cut during this clip?",
                           perception(tools_present=["bipolar forceps"])) == "No"


def test_the_cutting_and_dividing_sets_are_disjoint_and_in_vocabulary():
    assert not (CUTTING_TOOLS & DIVIDING_TOOLS)
    for cls in CUTTING_TOOLS | DIVIDING_TOOLS:
        assert cls in TOOL_CLASSES


def test_transection_counts_as_a_cutting_question():
    assert classify_question("Is the vessel being transected?") == INTENT_CUTTING


# --------------------------------------------------------------------------
# suturing questions
# --------------------------------------------------------------------------

def test_suture_is_yes_when_the_task_is_suturing():
    assert answer_question("Is a suture required in this surgical step?",
                           perception(task_top="suturing")) == "Yes"


def test_suture_is_yes_when_a_needle_driver_is_installed():
    # Measured: P(task = suturing | needle driver present) = 0.883.
    assert answer_question("Is a suture required in this surgical step?",
                           perception(tools_present=["needle driver"],
                                      task_top="uterine horn")) == "Yes"


def test_suture_is_no_without_either_signal():
    assert answer_question("Is a suture required in this surgical step?",
                           perception(tools_present=["monopolar curved scissors"],
                                      task_top="uterine horn")) == "No"


# --------------------------------------------------------------------------
# organ questions
# --------------------------------------------------------------------------

def test_organ_comes_from_the_task_class():
    assert answer_question("What organ is being manipulated?",
                           perception(task_top="uterine horn")) == "Uterine horn"


def test_the_organ_answer_is_sentence_case_not_title_case():
    # The gold reference is "Uterine horn". "Uterine Horn" is a different
    # string to BERTScore and casing measurably costs score.
    assert answer_question("What organ is being manipulated?",
                           perception(task_top="uterine horn")) != "Uterine Horn"


def test_every_task_class_maps_to_a_non_empty_organ():
    for task in TASK_CLASSES:
        assert organ_for_task(task).strip()


def test_an_unknown_task_still_yields_a_related_generic_organ():
    assert organ_for_task(None).strip()
    assert organ_for_task("nonsense").strip()


# --------------------------------------------------------------------------
# tool-identity questions
# --------------------------------------------------------------------------

def test_tool_identity_names_the_present_member_of_the_generic_set():
    answer = answer_question(
        "What type of forceps is mentioned?",
        perception(tools_present=["cadiere forceps", "monopolar curved scissors"]))
    assert answer == "Cadiere Forceps"


def test_tool_identity_picks_the_highest_scoring_candidate():
    scores = {c: 0.0 for c in TOOL_CLASSES}
    scores["prograsp forceps"] = 0.9
    scores["cadiere forceps"] = 0.3
    answer = answer_question(
        "What type of forceps is mentioned?",
        perception(tools_present=["cadiere forceps", "prograsp forceps"],
                   tools=scores))
    assert answer == "ProGrasp Forceps"


def test_tool_identity_falls_back_inside_the_asked_for_family():
    # Nothing crossed the threshold, but the question presupposes forceps.
    # Answering with the corpus-modal forceps beats answering something
    # unrelated: a wrong OPEN answer can score negative.
    answer = answer_question("What type of forceps is mentioned?", perception())
    assert answer == "Cadiere Forceps"


def test_tool_identity_with_no_family_named_reports_the_top_tool():
    answer = answer_question("What instrument is being used?",
                             perception(tools_present=["vessel sealer"]))
    assert answer == "Vessel Sealer Extend"


def test_tool_identity_with_nothing_at_all_stays_generic():
    assert answer_question("What instrument is being used?", perception()).strip()


# --------------------------------------------------------------------------
# procedure, purpose, task
# --------------------------------------------------------------------------

def test_procedure_is_a_corpus_constant():
    # Pinned to the literal: this is the gold FIRST reference of the public
    # sample, so it scores 1.0000 and is not a phrase to reword casually.
    assert PROCEDURE_ANSWER == "Endoscopic surgery or a laparoscopic surgery"
    assert answer_question("What procedure is this summary describing?",
                           perception()) == PROCEDURE_ANSWER


def test_procedure_ignores_perception_entirely():
    assert answer_question("What procedure is this summary describing?",
                           perception(tools_present=["stapler"])) == \
        answer_question("What procedure is this summary describing?", perception())


def test_purpose_of_forceps_is_the_canned_domain_answer():
    answer = answer_question("What is the purpose of using forceps in this procedure?",
                             perception())
    assert answer == "To grasp and hold tissues or objects during the surgery."


def test_purpose_is_tool_specific():
    answer = answer_question("What is the purpose of the monopolar curved scissors?",
                             perception())
    assert "cut" in answer.lower()


def test_purpose_of_an_unnamed_tool_is_still_a_plausible_sentence():
    answer = answer_question("What is the purpose of this step?", perception())
    assert answer.strip() and answer.endswith(".")


def test_task_question_names_the_task():
    assert answer_question("What task is being performed?",
                           perception(task_top="suturing")) == "Suturing"


# --------------------------------------------------------------------------
# fallbacks and robustness — the router must never crash, never return ""
# --------------------------------------------------------------------------

def test_unknown_polar_question_gets_the_polar_fallback():
    assert answer_question("Would a trainee find this difficult?",
                           perception()) == FALLBACK_POLAR


def test_unknown_open_question_gets_the_open_fallback():
    # RETARGETED, same reason as test_unknown_questions_fall_back_by_polarity:
    # a counting question is no longer an unknown one.
    assert answer_question("What is the patient's diagnosis?",
                           perception()) == FALLBACK_OPEN


def test_the_fallbacks_themselves_are_non_empty():
    assert FALLBACK_POLAR.strip()
    assert FALLBACK_OPEN.strip()


def test_the_polar_fallback_guesses_yes_not_no():
    # 4 of the 7 polar samples are "Yes", and a wrong polar guess costs only
    # 0.2985. Guessing "No" would be the worse side of the coin.
    assert FALLBACK_POLAR == "Yes"


@pytest.mark.parametrize("bad_perception", [
    None, {}, {"tools_present": None}, {"tools_present": []},
    {"task_top": None}, {"tools": None, "task": None},
    {"tools_present": ["not a real tool"]},
])
def test_the_router_survives_a_degenerate_perception_dict(bad_perception):
    for question in ["Are there forceps being used here?",
                     "What organ is being manipulated?",
                     "Is tissue being cut during this clip?",
                     "What type of forceps is mentioned?"]:
        answer = answer_question(question, bad_perception)
        assert isinstance(answer, str) and answer.strip()


@pytest.mark.parametrize("question", [None, "", "   ", "?", "banana"])
def test_the_router_survives_a_degenerate_question(question):
    answer = answer_question(question, perception())
    assert isinstance(answer, str) and answer.strip()


def test_tools_present_filters_anything_outside_the_taxonomy():
    # The perception half is a separate agent's code. Junk in the list must not
    # reach the answer tables; the taxonomy is the contract.
    assert tools_present({"tools_present": ["needle driver", "scalpel", 7, None]}) == \
        frozenset({"needle driver"})


def test_tools_present_normalises_casing_and_whitespace():
    assert tools_present({"tools_present": ["  Cadiere Forceps "]}) == \
        frozenset({"cadiere forceps"})


def test_task_top_reports_none_for_an_all_zero_distribution():
    # An all-zero softmax carries no information. Picking an argmax anyway
    # would silently name an organ on the strength of a tie-break.
    assert task_top({"task": {c: 0.0 for c in TASK_CLASSES}}) is None
    assert answer_question("What organ is being manipulated?",
                           {"task": {c: 0.0 for c in TASK_CLASSES}}) == "Tissue"


def test_task_top_ignores_a_task_top_outside_the_vocabulary():
    assert task_top({"task_top": "appendectomy"}) is None


def test_tool_prior_order_is_exactly_the_taxonomy():
    # A missing class would silently fall to the bottom of every tie-break.
    assert sorted(TOOL_PRIOR_ORDER) == sorted(TOOL_CLASSES)
    assert len(set(TOOL_PRIOR_ORDER)) == len(TOOL_PRIOR_ORDER)


def test_ties_are_broken_towards_the_more_common_class():
    # Cadiere is present in 58.7% of clip windows and bipolar in 45.0%, so an
    # unresolved tie between them resolves to Cadiere.
    answer = answer_question(
        "What type of forceps is mentioned?",
        perception(tools_present=["bipolar forceps", "cadiere forceps"]))
    assert answer == "Cadiere Forceps"


def test_tools_present_is_derived_from_scores_when_the_key_is_missing():
    # A missing tools_present would otherwise make every presence question
    # answer "No" silently. Fall back to thresholding the score dict.
    scores = {c: 0.0 for c in TOOL_CLASSES}
    scores["needle driver"] = 0.99
    assert answer_question("Is a needle driver involved in the procedure?",
                           {"tools": scores}) == "Yes"


def test_task_top_is_derived_from_scores_when_the_key_is_missing():
    scores = {c: 0.0 for c in TASK_CLASSES}
    scores["uterine horn"] = 0.8
    assert answer_question("What organ is being manipulated?",
                           {"task": scores}) == "Uterine horn"


def test_an_explicit_tools_present_is_not_second_guessed_by_the_scores():
    # The contract says thresholds are already applied upstream. An empty
    # list means "nothing present", not "no information".
    scores = {c: 0.99 for c in TOOL_CLASSES}
    assert answer_question("Is a needle driver involved in the procedure?",
                           {"tools": scores, "tools_present": []}) == "No"


def test_every_answer_for_every_sample_question_is_non_empty_and_terse():
    questions = [
        "Are there forceps being used here?",
        "Is a large needle driver among the listed tools?",
        "What type of forceps is mentioned?",
        "Is a suture required in this surgical step?",
        "Was a large needle driver used in this clip?",
        "What organ is being manipulated?",
        "Is a needle driver involved in the procedure?",
        "What procedure is this summary describing?",
        "What is the purpose of using forceps in this procedure?",
        "Is tissue being cut during this clip?",
        "Was a large needle driver used during the surgery?",
    ]
    for question in questions:
        answer = answer_question(question, perception(
            tools_present=["cadiere forceps", "needle driver"],
            task_top="suturing"))
        assert answer.strip()
        # Only the world-knowledge question is allowed to be a sentence.
        if classify_question(question) != INTENT_PURPOSE:
            assert len(answer.split()) <= 6, (question, answer)


def test_task_organs_covers_every_task_class():
    for task in TASK_CLASSES:
        assert task in TASK_ORGANS


# ==========================================================================
# COVERAGE HARDENING
#
# Everything below was written against tests/fixtures/question_variants.json,
# a 159-item paraphrase battery measured by scripts/router_coverage.py. The
# eleven public sample questions are the only phrasings we have ever seen; the
# battery is the phrasings a challenge annotator could plausibly have written
# instead. Each test here protects one GENERAL mechanism, not one paraphrase.
# ==========================================================================

# --------------------------------------------------------------------------
# morphology: a class name in the plural is still that class
# --------------------------------------------------------------------------

def test_a_plural_class_name_resolves_to_its_class():
    assert mentioned_tool_classes("Are staplers deployed in this clip?") == \
        frozenset({"stapler"})
    assert mentioned_tool_classes("Are clip appliers present?") == \
        frozenset({"clip applier"})
    assert mentioned_tool_classes("Are vessel sealers being used?") == \
        frozenset({"vessel sealer"})


def test_a_plural_generic_term_resolves_to_the_same_set_as_the_singular():
    assert mentioned_tool_classes("Are retractors being used?") == \
        mentioned_tool_classes("Is a retractor being used?")


def test_pluralisation_never_manufactures_a_bare_clip_term():
    # The whole point of having no "clip" entry is that two of the eleven real
    # questions say "in this clip". Pluralising the phrase table must not
    # smuggle one in through the back door.
    assert mentioned_tool_classes("Are these clips shown?") == frozenset()


# --------------------------------------------------------------------------
# commercial names: config/commercial_names.json is a synonym table
# --------------------------------------------------------------------------

def test_a_distinctive_brand_token_resolves_to_its_class():
    # "SureForm" appears only in stapler rows, "SutureCut" only in needle
    # driver rows, across all 1,629 installations in config/commercial_names.json.
    assert mentioned_tool_classes("Was a SureForm used?") == frozenset({"stapler"})
    assert mentioned_tool_classes("Is a SutureCut driver among the tools?") == \
        frozenset({"needle driver"})


def test_a_brand_name_outranks_the_generic_head_noun_it_carries():
    # "Maryland Bipolar Forceps" is a bipolar forceps and nothing else; the
    # bare generic "bipolar" would also drag in `force bipolar`.
    assert mentioned_tool_classes("Is a Maryland bipolar in use?") == \
        frozenset({"bipolar forceps"})
    # The single needle-driver row commercially named "DeBakey Forceps" must
    # not be read as the four FORCEPS classes.
    assert mentioned_tool_classes("Is a DeBakey Forceps involved?") == \
        frozenset({"needle driver"})


def test_a_brand_token_shared_by_two_classes_is_not_registered():
    # "Fenestrated" names both a bipolar forceps and the tip-up grasper, so it
    # identifies nothing on its own.
    assert mentioned_tool_classes("Is a fenestrated instrument used?") == frozenset()


def test_ordinary_english_words_inside_commercial_names_are_not_tool_terms():
    # "Small Grasping Retractor" and "Vessel Sealer Extend" contain words that
    # a question uses in their ordinary sense far more often than as a brand.
    assert mentioned_tool_classes("Is the surgeon grasping the tissue?") == frozenset()
    assert mentioned_tool_classes("Does the arm extend fully?") == frozenset()


def test_load_commercial_names_degrades_to_an_empty_mapping(tmp_path):
    assert load_commercial_names(tmp_path / "nope.json") == {}


# --------------------------------------------------------------------------
# polarity: contractions and fronted adjuncts
# --------------------------------------------------------------------------

def test_contracted_openers_are_polar():
    assert is_polar_question("Isn't a needle driver used?") is True
    assert is_polar_question("Aren't forceps being used here?") is True
    assert is_polar_question("Doesn't this step require a suture?") is True


def test_a_polar_opener_after_a_fronted_adjunct_is_still_polar():
    assert is_polar_question("In this clip, was a large needle driver used?") is True
    assert is_polar_question("During the surgery, was a driver used?") is True
    assert is_polar_question("Needle driver - is one being used?") is True


def test_a_non_polar_later_clause_does_not_make_a_question_polar():
    assert is_polar_question("What procedure is this, and what is its purpose?") \
        is False


# --------------------------------------------------------------------------
# negation: the answer flips, the intent does not
# --------------------------------------------------------------------------

def test_existential_negation_flips_a_presence_answer():
    assert classify_question("Is there no needle driver present?") == \
        INTENT_TOOL_PRESENCE
    assert answer_question("Is there no needle driver present?",
                           perception(tools_present=["needle driver"])) == "No"
    assert answer_question("Is there no needle driver present?",
                           perception(tools_present=["cadiere forceps"])) == "Yes"


@pytest.mark.parametrize("question", [
    "Are there no forceps being used?",
    "Was a large needle driver absent from this clip?",
    "Was a needle driver never used during the surgery?",
    "Is a needle driver not involved in this procedure?",
])
def test_every_existential_negation_form_flips(question):
    assert answer_question(question, perception(
        tools_present=["cadiere forceps", "needle driver"])) == "No"


def test_a_contracted_auxiliary_is_not_existential_negation():
    # "Isn't X used?" is answered Yes when X is used. Only negation of the
    # EXISTENCE flips; negation of the auxiliary is rhetorical.
    assert answer_question("Isn't a needle driver being used?",
                           perception(tools_present=["needle driver"])) == "Yes"
    assert answer_question("Aren't forceps being used here?",
                           perception(tools_present=["cadiere forceps"])) == "Yes"


def test_negation_flips_cutting_and_suturing_as_well():
    assert answer_question("Is no tissue being cut in this clip?",
                           perception(tools_present=["monopolar curved scissors"])) \
        == "No"
    assert answer_question("Is there no suturing in this step?",
                           perception(task_top="suturing")) == "No"


def test_negation_in_an_open_question_names_an_ABSENT_tool():
    """REVERSES A PRIOR DECISION, ON CORPUS EVIDENCE.

    This asserted the opposite -- that "not" leaves an open answer alone, so
    "Which instrument is not in view?" with a vessel sealer present answered
    "Vessel Sealer Extend". The reasoning ("there is nothing to invert in a
    noun") is true about answer FORM and does not license naming a tool that
    IS present.

    The corpus settles it: all 2,000 `tool_absence_open` records have a gold
    answer that is a tool NOT present ("What instrument class does not appear
    in this segment?" -> "Stapler", "Vessel Sealer", ...). Naming a present
    tool is therefore wrong by construction, and measured at 4.0% exact
    against 10.7% for naming an absent one.

    The gain is small and honestly so -- roughly +0.004 overall, BELOW the
    resolution of the graded-11 predictor. This is a correctness fix, not a
    score play.
    """
    answer = answer_question("Which instrument is not in view?",
                             perception(tools_present=["vessel sealer"]))
    assert "vessel sealer" not in answer.lower(), (
        "named the tool the question said is NOT there")


def test_a_non_negated_open_question_still_names_a_PRESENT_tool():
    """The other half of the coupling: only NEGATED questions invert."""
    assert answer_question("Which instrument is in view?",
                           perception(tools_present=["vessel sealer"])) == \
        "Vessel Sealer Extend"


# --------------------------------------------------------------------------
# counting -- previously routed to the generic open fallback
# --------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "How many instruments are installed?",
    "How many tools are being used?",
    "Count the number of tools visible.",
    "What is the number of instruments installed?",
])
def test_counting_questions_get_their_own_intent(question):
    assert classify_question(question) == INTENT_COUNT


def test_how_long_is_not_a_counting_question():
    assert classify_question("How long is this clip?") == INTENT_UNKNOWN_OPEN


def test_a_count_is_answered_as_a_single_english_word():
    answer = answer_question("How many instruments are installed?",
                             perception(tools_present=["cadiere forceps",
                                                       "needle driver"]))
    assert answer == "Two"


def test_a_count_is_scoped_to_the_family_the_question_names():
    assert answer_question("How many forceps are in use?",
                           perception(tools_present=["cadiere forceps",
                                                     "needle driver"])) == "One"


def test_a_count_with_no_evidence_at_all_falls_back_to_the_modal_count():
    # Measured over the 23,515 clip-sized windows of the training corpus:
    # 3 distinct in-scope classes is the mode (44.9%).
    assert MODAL_TOOL_COUNT == 3
    assert answer_question("How many instruments are installed?",
                           perception()) == "Three"


# --------------------------------------------------------------------------
# open-question vocabulary
# --------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "What tissue is being manipulated?",
    "What structure is being grasped?",
    "Which part of the body is being operated on?",
])
def test_an_organ_question_need_not_use_the_word_organ(question):
    assert classify_question(question) == INTENT_ORGAN


def test_merely_containing_the_word_tissue_is_not_an_organ_question():
    # The organ cue has to be the head of the wh-phrase, or "what colour is
    # the tissue" would be answered with the name of an organ.
    assert classify_question("What colour is the tissue?") == INTENT_UNKNOWN_OPEN


def test_purpose_survives_a_stranded_preposition():
    assert classify_question("What are forceps for?") == INTENT_PURPOSE
    assert classify_question("What do surgeons use forceps for?") == INTENT_PURPOSE


def test_purpose_recognises_intended_use():
    assert classify_question("What is the intended use of the cautery hook?") == \
        INTENT_PURPOSE


def test_a_named_step_outranks_the_procedure_rule():
    # "What phase of the procedure is this?" matches the procedure pattern
    # verbatim, but the question is about the phase.
    assert classify_question("What phase of the procedure is this?") == INTENT_TASK


@pytest.mark.parametrize("question", [
    "What tools are used in this step?",
    "Which instruments are used in this step?",
    "Which forceps is used in this step?",
    "What type of forceps is used in this task?",
    "Name the instrument used in this step.",
    "Which tool is used during this activity?",
])
def test_a_named_tool_outranks_an_incidental_task_noun(question):
    """A task noun in an adjunct does not make a tool question a task question.

    This is the same rule the procedure family already follows -- the narrow
    "what procedure is this" pattern runs before the tool rules and the broad
    "mentions a procedure" pattern runs after them, so "which instrument is
    used in this procedure" still names an instrument. "in this step" and "in
    this task" are the same kind of adjunct and must not outrank the tool the
    question is actually asking about: answering "Suturing" to "which forceps
    is used in this step?" is a wrong noun in the wrong category, which is the
    expensive end of this metric.
    """
    assert classify_question(question) == INTENT_TOOL_IDENTITY


@pytest.mark.parametrize("question", [
    "Which surgical task is underway?",
    "What step is the surgeon on?",
    "What activity is shown in this clip?",
    "Describe the current surgical step.",
    "What exercise is the trainee performing?",
    "Which training task does this clip show?",
    "What surgical step does this clip belong to?",
    "Name the activity the surgeon is practicing.",
    "What phase of the procedure is shown?",
    # A task noun under a wh-head still wins even when a tool is named: the
    # question asks WHICH TASK, and the tool is the adjunct this time.
    "What task is the needle driver performing?",
])
def test_a_task_question_is_still_a_task_question(question):
    assert classify_question(question) == INTENT_TASK


@pytest.mark.parametrize("question", [
    "Which procedure is this summary describing?",
    "Describe the surgical procedure shown.",
])
def test_procedure_questions_that_do_not_open_with_what(question):
    assert classify_question(question) == INTENT_PROCEDURE


def test_a_who_question_about_the_surgery_is_not_a_procedure_question():
    assert classify_question("Who is performing this surgery?") == INTENT_UNKNOWN_OPEN


# --------------------------------------------------------------------------
# TASK 3 -- low-confidence perception (case129 has tools_present == [])
# --------------------------------------------------------------------------

def _case129():
    """case129 verbatim from outputs/perception_sample.json."""
    return {
        "tools": {
            "bipolar forceps": 0.5189, "cadiere forceps": 0.6011,
            "clip applier": 0.0553, "force bipolar": 0.0056,
            "grasping retractor": 0.2318,
            "monopolar curved scissors": 0.7456, "needle driver": 0.4698,
            "permanent cautery hook/spatula": 0.0077, "prograsp forceps": 0.2924,
            "stapler": 0.0129, "tip-up fenestrated grasper": 0.0009,
            "vessel sealer": 0.0263,
        },
        "tools_present": [],
        "task": {c: 0.0 for c in TASK_CLASSES},
        "task_top": "suturing",
        "n_frames": 16,
    }


def test_credible_tools_prefers_the_authoritative_list():
    assert credible_tools(perception(tools_present=["needle driver"])) == \
        frozenset({"needle driver"})


def test_soft_evidence_answers_presence_when_every_threshold_was_missed():
    # Nothing cleared its tuned threshold, but a cadiere at 0.60 is still more
    # likely present than not, and a wrong polar answer costs only 0.2985.
    assert credible_tools(_case129()) == frozenset({
        "bipolar forceps", "cadiere forceps", "monopolar curved scissors"})
    assert answer_question("Are there forceps being used here?", _case129()) == "Yes"
    assert answer_question("Is tissue being cut during this clip?", _case129()) == "Yes"


def test_soft_evidence_stops_at_one_half():
    # The needle driver is at 0.4698 -- under a half, so even the permissive
    # reading says No. This is the line the policy will not cross.
    assert answer_question("Is a needle driver involved in the procedure?",
                           _case129()) == "No"


def test_soft_evidence_requires_a_physically_credible_number_of_tools():
    # Three instrument arms plus an endoscope: 97.9% of the 23,515 training
    # windows hold at most 3 distinct in-scope classes. Twelve classes over
    # threshold with an empty presence list is not a shy classifier, it is an
    # incoherent record, and the authoritative empty list stands.
    assert SOFT_PRESENCE_MAX_CLASSES == 3
    everything = {"tools": {c: 0.99 for c in TOOL_CLASSES}, "tools_present": []}
    assert credible_tools(everything) == frozenset()
    assert answer_question("Is a needle driver involved in the procedure?",
                           everything) == "No"


def test_soft_evidence_never_invents_a_tool_out_of_nothing():
    assert credible_tools(perception()) == frozenset()
    assert answer_question("Is a stapler being used?", perception()) == "No"


def test_soft_evidence_does_not_make_an_open_answer_more_specific():
    # The asymmetry runs the other way on open questions: a wrong noun can
    # score negative. Identity already reads the raw scores and must be
    # unchanged by the presence policy.
    assert answer_question("What type of forceps is mentioned?", _case129()) == \
        "Cadiere Forceps"


def test_the_counting_gap_that_two_older_tests_pinned_is_closed_deliberately():
    """The one behaviour change that required editing an existing test.

    Two tests used "How many instruments are visible?" as their example of an
    unknown open question. Counting is no longer unknown: the perception half
    already reports which classes are installed, and a number scored against a
    number beats a sentence about surgical instruments scored against a
    number. Both tests kept their rule and swapped their example; this test
    holds the retired example so the change stays visible in the suite.
    """
    assert classify_question("How many instruments are visible?") == INTENT_COUNT
    assert answer_question("How many instruments are visible?",
                           perception(tools_present=["cadiere forceps",
                                                     "needle driver",
                                                     "monopolar curved scissors"])) \
        == "Three"


# --------------------------------------------------------------------------
# tag questions -- found by the HELD-OUT battery, not by this author
# (tests/fixtures/question_variants_heldout.json)
# --------------------------------------------------------------------------

def test_a_tag_question_is_polar_even_with_no_opener_anywhere():
    question = "Tip-up fenestrated grasper -- present or not?"
    assert is_polar_question(question) is True
    assert classify_question(question) == INTENT_TOOL_PRESENCE


def test_the_or_not_tag_is_not_existential_negation():
    # "or not" is what MAKES this a yes/no question; reading its "not" as
    # negation would inverte the answer of every tag question ever asked.
    assert answer_question("Tip-up fenestrated grasper -- present or not?",
                           perception(tools_present=["tip-up fenestrated grasper"])) \
        == "Yes"


# --------------------------------------------------------------------------
# which intents can be answered with no perception at all
# --------------------------------------------------------------------------
# The serving entrypoint needs this set to decide what a perception FAILURE
# costs. For most intents the answer is "everything", and the calibrated
# fallback string is the floor. For the intents whose answer form never reads
# the record, a failure costs NOTHING: routing them against {} produces the
# same answer a healthy run would have produced. See
# scripts/inference.py::fallback_answer.

# One question per intent, and the parametrisation doubles as a classifier
# pin: every case asserts classify_question maps it where it says it does.
INTENT_PROBES = {
    INTENT_TOOL_PRESENCE: "Is a needle driver being used?",
    INTENT_TOOL_IDENTITY: "What instrument is being used?",
    INTENT_ORGAN: "What organ is being manipulated?",
    INTENT_CUTTING: "Is tissue being cut?",
    INTENT_SUTURE: "Is suturing being performed?",
    INTENT_PROCEDURE: "What type of procedure is being performed?",
    INTENT_PURPOSE: "What is the purpose of using forceps?",
    INTENT_TASK: "What task is being performed?",
    INTENT_COUNT: "How many instruments are visible?",
    INTENT_TASK_CONFIRM: "Is uterine horn mobilization taking place in this clip?",
    INTENT_APPROACH: "Is this an open surgery?",
    INTENT_UNKNOWN_POLAR: "Is the trainer box level?",
    INTENT_UNKNOWN_OPEN: "Describe the widget.",
}

# Records chosen to move every form that reads one: an empty dict, a
# needle-driver/suturing clip, and a two-tool uterine-horn clip.
_RECORDS = (
    {},
    perception(tools_present=["needle driver"], task_top="suturing",
               tools={"needle driver": 0.95}, task={"suturing": 0.95}),
    perception(tools_present=["clip applier", "monopolar curved scissors"],
               task_top="uterine horn",
               tools={"clip applier": 0.9, "monopolar curved scissors": 0.8},
               task={"uterine horn": 0.9}),
)


def test_every_intent_has_a_probe():
    """A new intent must be classified here, or the two tests below would
    silently stop covering it."""
    assert set(INTENT_PROBES) == set(ANSWER_FORMS)


@pytest.mark.parametrize("intent,question", sorted(INTENT_PROBES.items()))
def test_the_probe_questions_classify_where_they_claim(intent, question):
    assert classify_question(question) == intent


@pytest.mark.parametrize("intent,question", sorted(INTENT_PROBES.items()))
def test_perception_independence_is_declared_exactly(intent, question):
    """PERCEPTION_INDEPENDENT_INTENTS holds every intent that ignores the
    record, and nothing else.

    Both directions matter. A missing entry throws away an answer we already
    had on a perception failure; a spurious one routes a question against an
    empty record, and an empty record answers "No" to every presence question
    and invents a modal count out of nothing.
    """
    answers = {answer_question(question, record) for record in _RECORDS}
    ignores_perception = len(answers) == 1

    assert ignores_perception == (intent in PERCEPTION_INDEPENDENT_INTENTS), (
        "%s produced %r across the probe records" % (intent, sorted(answers)))


@pytest.mark.parametrize("intent,question", sorted(INTENT_PROBES.items()))
def test_routing_against_an_empty_record_never_raises(intent, question):
    """The router tolerates {} for EVERY intent -- that is what makes the
    fallback safe to write. It is not that the other intents crash; it is
    that their answers get worse, which the test above pins."""
    answer = answer_question(question, {})

    assert isinstance(answer, str) and answer.strip()


def test_an_empty_record_answers_no_to_a_presence_question():
    """The measured reason the fallback may not simply route everything
    against {}: gold polar answers skew Yes and a wrong polar costs 0.2985."""
    assert answer_question("Is a needle driver being used?", {}) == "No"


def test_a_purpose_answer_survives_a_total_perception_failure():
    """The gold first reference from the public sample, from world knowledge
    alone. Worth 1.0000 against 0.35-0.48 for the generic sentence."""
    assert answer_question("What is the purpose of using forceps?", {}) \
        == "To grasp and hold tissues or objects during the surgery."


def test_a_procedure_answer_survives_a_total_perception_failure():
    assert answer_question("What type of procedure is being performed?", {}) \
        == PROCEDURE_ANSWER


# --------------------------------------------------------------------------
# questions whose answer is not in the record at all
# --------------------------------------------------------------------------
#
# These pin a CATEGORY guarantee rather than a score. The record holds tool
# probabilities and a task distribution -- no clock, no arm assignment, no
# spatial layout, no agent -- so a duration question must not come back with a
# task name and an arm question must not come back with an instrument. Without
# the guard each of these routed on the question's OTHER words and answered
# confidently: "how long does this step take?" saw "step" and said "Suturing".

@pytest.mark.parametrize("question", [
    "How long does this step take?",
    "How long is the needle driver used?",
    "How much time does the suturing take?",
    "How many seconds is the stapler used?",
    "How many minutes does this clip last?",
    "How many times is the stapler fired?",
    "When is the stapler fired?",
    "Where is the needle driver?",
    "Who is operating the robot?",
    "Which arm holds the needle driver?",
    "What arm is the bipolar forceps on?",
    "Which hand is the surgeon using?",
])
def test_a_question_the_record_cannot_answer_falls_through(question):
    assert classify_question(question) == INTENT_UNKNOWN_OPEN


@pytest.mark.parametrize("question", [
    "How long does this step take?",
    "Which arm holds the needle driver?",
    "How many seconds is the stapler used?",
])
def test_an_unanswerable_question_gets_the_generic_sentence(question):
    record = {"tools_present": ["needle driver"], "task_top": "suturing",
              "tools": {"needle driver": 0.9}}
    assert answer_question(question, record) == FALLBACK_OPEN


@pytest.mark.parametrize("question,intent", [
    # The guard is narrow on purpose: it fires on the question HEAD, so
    # neighbouring question families must be untouched.
    ("How many instruments are listed?", INTENT_COUNT),
    ("How many tools are being used?", INTENT_COUNT),
    ("What surgical step is shown?", INTENT_TASK),
    ("Which instrument is used most in this clip?", INTENT_TOOL_IDENTITY),
    ("What organ is being manipulated?", INTENT_ORGAN),
    ("What is the purpose of using forceps in this procedure?", INTENT_PURPOSE),
])
def test_the_guard_does_not_swallow_neighbouring_families(question, intent):
    assert classify_question(question) == intent


def test_a_polar_question_is_exempt_from_the_guard():
    # Wrong polarity still scores 0.7015, so a polar guess is cheap and worth
    # making. Only the open path -- where a wrong specific noun can score
    # NEGATIVE -- declines to answer.
    assert classify_question("Is the arm moving?") != INTENT_UNKNOWN_OPEN
    assert answer_question("Is the arm moving?", {}) == FALLBACK_POLAR


# --------------------------------------------------------------------------
# GENERIC INSTRUMENT QUESTIONS -- the record answers them, the constant did not
# --------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "Is any instrument being used?",
    "Are there any instruments in view?",
    "Is an instrument installed?",
    "Are any tools visible?",
    "Is there a device in the field of view?",
])
def test_a_generic_instrument_question_is_a_presence_question(question):
    """It names no class, but it is still asking what the record knows.

    These used to fall to unknown_polar, whose answer is the constant "Yes".
    That is right about 96% of the time for the POSITIVE phrasings -- 3.95% of
    validation windows have no tool installed -- so the bug was invisible
    until the negated phrasings below.
    """
    assert classify_question(question) == INTENT_TOOL_PRESENCE


@pytest.mark.parametrize("question", [
    "Is no instrument being used?",
    "Are no tools installed?",
    "Is there not a single instrument in view?",
])
def test_a_negated_generic_instrument_question_is_answered_no(question):
    """The half of the fix that actually changes an answer.

    unknown_polar is deliberately NOT in NEGATABLE_INTENTS -- flipping a
    calibrated guess only moves the coin to the side the corpus disfavours --
    so while these routed there, nothing inverted them and every one was
    answered "Yes" about a clip full of instruments. Routing them to presence
    puts them under the existing negation rule.
    """
    record = {"tools_present": ["needle driver"], "task_top": "suturing"}
    assert answer_question(question, record) == "No"


def test_a_generic_instrument_question_still_follows_the_record():
    empty = {"tools_present": [], "task_top": "other"}
    assert answer_question("Is any instrument being used?", empty) == "No"
    assert answer_question("Is no instrument being used?", empty) == "Yes"


def test_an_unrecognised_instrument_keeps_the_guess():
    """A scalpel is not one of the twelve, so our record is not about it.

    This is the line between the two unnamed cases: "is any instrument in use"
    asks about something we can see, "is a scalpel in use" asks about
    something we cannot. Answering the second from our record would say No
    with confidence about an instrument outside the taxonomy.
    """
    record = {"tools_present": ["needle driver"], "task_top": "suturing"}
    assert answer_question("Is a scalpel being used?", record) == FALLBACK_POLAR


def test_a_specific_polar_reading_outranks_the_generic_one():
    # "instrument" appears in both, but the cutting rule is the more specific
    # evidence and runs first.
    assert classify_question("Is the instrument cutting tissue?") == INTENT_CUTTING
    assert classify_question("Is the instrument suturing?") == INTENT_SUTURE


# --------------------------------------------------------------------------
# "DESCRIBE WHAT IS HAPPENING" -- a task question with no task noun in it
# --------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "Describe what is happening in this clip.",
    "Describe the surgical scene.",
    "What is going on in this video?",
    "Tell me what the surgeon is doing.",
    "What is happening here?",
    "Explain what is being performed.",
])
def test_a_scene_level_question_is_answered_with_the_task(question):
    record = {"task_top": "suturing", "tools_present": ["needle driver"]}
    assert classify_question(question) == INTENT_TASK
    assert answer_question(question, record) == "Suturing"


@pytest.mark.parametrize("question,intent", [
    # The activity rule runs LAST, so everything that has already said what it
    # is about keeps its own reading.
    ("Could you describe the procedure being performed?", INTENT_PROCEDURE),
    ("Describe the instruments in view.", INTENT_TOOL_IDENTITY),
    ("Can you identify the organ being manipulated?", INTENT_ORGAN),
    ("Describe how many instruments are installed.", INTENT_COUNT),
    ("Describe what step is being performed.", INTENT_TASK),
])
def test_the_activity_rule_runs_last(question, intent):
    assert classify_question(question) == intent


# --------------------------------------------------------------------------
# PLURAL IDENTITY QUESTIONS -- the single noun was the minority answer
# --------------------------------------------------------------------------

def _three_tool_record():
    return {"tools": {"needle driver": 0.95, "cadiere forceps": 0.80,
                      "monopolar curved scissors": 0.62,
                      "bipolar forceps": 0.10},
            "tools_present": ["needle driver", "cadiere forceps",
                              "monopolar curved scissors"],
            "task_top": "suturing"}


@pytest.mark.parametrize("question", [
    "What tools are used in this step?",
    "Which instruments are used in this step?",
    "Could you list the tools in view?",
    "Name the instruments.",
    "What devices are visible?",
])
def test_a_plural_question_lists_the_instruments(question):
    """Measured at +0.4964 over the single name with real perception.

    Only 13.9% of validation windows have one instrument installed, so naming
    one against a reference that lists the set scores 0.4982 at m=2 and 0.2976
    at m=3.
    """
    answer = answer_question(question, _three_tool_record())
    assert answer == ("Large Needle Driver, Cadiere Forceps and "
                      "Monopolar Curved Scissors")


@pytest.mark.parametrize("question", [
    "What instrument is being used?",
    "Which tool is used during this activity?",
    "What type of forceps is mentioned?",
    # "forceps" and "scissors" are invariant in English, so a plural VERB is
    # the only signal and it is too weak to act on. These keep the single name.
    "Which forceps are used in this step?",
])
def test_a_singular_question_still_names_one(question):
    assert " and " not in answer_question(question, _three_tool_record())


def test_the_list_is_capped_and_ordered_by_confidence():
    """Three names, most credible first -- not everything above threshold.

    "top three" scored 0.8355 and "all above threshold" 0.8340, a difference
    of 0.0015. With the two tied, the bounded policy is the one to take.
    """
    record = {"tools": {"needle driver": 0.95, "cadiere forceps": 0.90,
                        "monopolar curved scissors": 0.85,
                        "bipolar forceps": 0.80, "clip applier": 0.75},
              "tools_present": ["needle driver", "cadiere forceps",
                                "monopolar curved scissors", "bipolar forceps",
                                "clip applier"]}
    answer = answer_question("What tools are used in this step?", record)
    assert answer == ("Large Needle Driver, Cadiere Forceps and "
                      "Monopolar Curved Scissors")


def test_a_plural_question_with_one_credible_tool_names_one():
    record = {"tools": {"needle driver": 0.95}, "tools_present": ["needle driver"]}
    assert answer_question("What tools are used in this step?",
                           record) == "Large Needle Driver"


def test_a_plural_question_inside_a_family_stays_inside_it():
    # The family restriction outranks the plural rule: a question about
    # forceps gets forceps, however many are named.
    record = {"tools": {"needle driver": 0.99, "cadiere forceps": 0.80,
                        "bipolar forceps": 0.70},
              "tools_present": ["needle driver", "cadiere forceps",
                                "bipolar forceps"]}
    answer = answer_question("Which forceps instruments are used?", record)
    assert "Needle Driver" not in answer
    assert answer == "Cadiere Forceps and Bipolar Forceps"


# --------------------------------------------------------------------------
# motion evidence -- the gate is closed, and that must be provable
# --------------------------------------------------------------------------
#
# `_answer_cutting` answers "is tissue being cut?" with "are scissors
# visible?" -- an EVENT question answered by a proxy for PRESENCE. Motion is
# the missing evidence, and surgvu/motion.py computes it. What these tests
# protect is the order of operations: the mechanism ships INERT, and no answer
# may move until a measured threshold opens the gate deliberately.

def with_motion(record, micro_mean=0.0, macro_mean=0.0, bursts=16):
    """The same record, plus a motion block in the contract's shape."""
    out = dict(record)
    out["motion"] = {
        "version": 1, "frames_per_burst": 3, "bursts": bursts,
        "micro": {"per_burst": [micro_mean] * bursts, "mean": micro_mean,
                  "max": micro_mean, "std": 0.0},
        "macro": {"per_gap": [macro_mean] * (bursts - 1), "mean": macro_mean,
                  "max": macro_mean, "std": 0.0},
    }
    return out


MOTION_QUESTIONS = [
    "Is tissue being cut in this clip?",
    "Is suturing being performed?",
    "What tool is being used?",
    "What is happening in this clip?",
    "Is there a needle driver present?",
    "What organ is visible?",
    "How many instruments are in view?",
    "What is the purpose of the needle driver?",
]


@pytest.mark.parametrize("question", MOTION_QUESTIONS)
@pytest.mark.parametrize("activity", [0.0, 0.5, 40.0])
def test_a_motion_block_changes_no_answer_while_the_gate_is_closed(question,
                                                                   activity):
    """THE safety property, and the reason it is structural rather than lucky.

    STATIC_ACTIVITY_THRESHOLD is None, so every motion accessor reports "no
    evidence" and every rule falls through to what it does today. Adding the
    block to a record must therefore be a no-op for every intent at every
    activity level -- including an activity high enough to be obviously
    active and one low enough to be obviously still.

    If this ever fails, a rule started reading motion without a calibrated
    threshold, and the 11-case sample check would be the next thing to notice
    -- after the change had already shipped.
    """
    from surgvu.router import STATIC_ACTIVITY_THRESHOLD
    assert STATIC_ACTIVITY_THRESHOLD == 1.283, (
        "the gate was opened at 1.283 on 2026-08-16; if it moves again, the "
        "expectations below move with it")

    record = perception(tools_present=["monopolar curved scissors"],
                        task_top="suturing")
    before = answer_question(question, record)
    after = answer_question(question, with_motion(record, activity))

    # THE ONLY INTENDED CHANGE: a cutting question, a cutting tool credible,
    # and activity BELOW the threshold. Everything else must still be
    # untouched -- the motion block is additive and the gate reaches exactly
    # one rule.
    cutting_q = "cut" in question.lower()
    should_flip = cutting_q and activity < STATIC_ACTIVITY_THRESHOLD
    if should_flip:
        assert before == "Yes" and after == "No", (
            "a still scene with scissors credible must downgrade to No: "
            "%r -> %r" % (before, after))
    else:
        assert after == before, (
            "motion changed %r for a question it must not reach: %r -> %r"
            % (question, before, after))


def test_motion_evidence_is_none_when_there_is_no_block():
    from surgvu.router import motion_evidence
    assert motion_evidence(perception()) is None


@pytest.mark.parametrize("junk", [None, 42, "lots", {}, {"micro": {}},
                                  {"micro": {"mean": "high"}}])
def test_a_malformed_motion_block_reads_as_no_evidence(junk):
    """Never as zero. A missing key answering 'nothing is moving' would answer
    'No, nothing is being cut' on the strength of a serialisation bug."""
    from surgvu.router import motion_evidence
    record = perception()
    record["motion"] = junk
    assert motion_evidence(record) is None


def test_scene_activity_reads_the_measured_value():
    from surgvu.router import scene_activity
    assert scene_activity(with_motion(perception(), 3.25)) == pytest.approx(3.25)
    assert scene_activity(perception()) is None


def test_scene_is_static_is_none_when_there_is_no_measurement():
    """Three states, and conflating two of them is the failure mode.

    With the gate OPEN, a record carrying no motion block must still answer
    None rather than False -- "we did not measure" is not "nothing moved", and
    a container running without --motion must fall back to the presence proxy
    rather than answering No to everything.
    """
    from surgvu.router import scene_is_static
    assert scene_is_static(perception()) is None
    assert scene_is_static(with_motion(perception(), 0.2)) is True
    assert scene_is_static(with_motion(perception(), 5.0)) is False


def test_scene_is_static_answers_once_a_threshold_exists(monkeypatch):
    """The mechanism works -- it is only the constant that is withheld."""
    import surgvu.router as router
    monkeypatch.setattr(router, "STATIC_ACTIVITY_THRESHOLD", 1.0)
    assert router.scene_is_static(with_motion(perception(), 0.2)) is True
    assert router.scene_is_static(with_motion(perception(), 5.0)) is False
    # Still None when the record carries no measurement at all.
    assert router.scene_is_static(perception()) is None
