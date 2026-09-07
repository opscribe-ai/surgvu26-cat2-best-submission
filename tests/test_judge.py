"""The decision VLM's contract.

Every test here guards a way the judge could QUIETLY make the pipeline worse:
by running when it cannot help (cost), by paraphrasing an answer whose exact
wording was tuned for BERTScore, or by returning nothing at all -- which
scores 0, worse than any wrong answer.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu import judge                                        # noqa: E402


# --- should_consult: the cost control -------------------------------------

def test_agreement_does_not_consult_the_judge():
    """A second VLM pass costs ~42s warm and ~171s cold, and every Grand
    Challenge invocation is cold. When both candidates already agree there is
    nothing to arbitrate, so paying that is pure loss."""
    assert judge.should_consult("Yes", "Yes") is False


def test_cosmetic_differences_are_not_disagreements():
    """'Yes' vs 'yes.' must not buy a 171-second consultation. Case and
    trailing punctuation are exactly the differences the scorer ignores."""
    assert judge.should_consult("Yes", "yes.") is False
    assert judge.should_consult("Uterine horn", "  uterine horn  ") is False


def test_a_real_disagreement_consults():
    assert judge.should_consult("No", "Yes") is True
    assert judge.should_consult("Bipolar Forceps", "Clip applier") is True


def test_an_empty_candidate_does_not_consult():
    """Nothing to compare against; the caller's own fallback is better than
    asking a model to choose between an answer and a blank."""
    assert judge.should_consult("", "Yes") is False
    assert judge.should_consult("Yes", None) is False


# --- the prompt ------------------------------------------------------------

def test_the_prompt_never_names_the_router_or_the_vlm():
    """Neutral labels are load-bearing. Telling the model which candidate came
    from 'the neural model' invites choosing by reputation rather than by the
    evidence, and the router is the MORE accurate of the two on the graded
    polar questions (7/7 vs 5/7)."""
    prompt = judge.build_judge_prompt(
        "Is a needle driver visible?", "Tools: needle driver.", ("Yes", "No"))
    lowered = prompt.lower()
    assert "router" not in lowered
    assert "vlm" not in lowered
    assert "Answer 1" in prompt and "Answer 2" in prompt


def test_the_prompt_allows_a_third_option():
    """BOTH candidates were wrong on case124 (router 'Bipolar Forceps', VLM
    'Clip applier', gold 'Cadiere Forceps'). A judge restricted to picking one
    of two could only have chosen the less wrong."""
    prompt = judge.build_judge_prompt("What forceps?", "", ("A", "B"))
    assert "If both are wrong" in prompt


def test_the_evidence_text_is_passed_through_not_reformatted():
    """This module must not re-implement evidence_vlm's renderers -- a second
    copy drifts, and the model would then see two different descriptions of
    the same detector output depending on which stage rendered it."""
    evidence = "Detected tools: needle driver (0.91), cadiere forceps (0.32)."
    prompt = judge.build_judge_prompt("q?", evidence, ("Yes", "No"))
    assert evidence in prompt


def test_the_question_is_whitespace_collapsed():
    prompt = judge.build_judge_prompt("Is   a\ntool\tvisible?", "", ("Yes", "No"))
    assert "Question: Is a tool visible?" in prompt


# --- parsing the reply -----------------------------------------------------

def test_choosing_a_candidate_returns_it_verbatim():
    """VERBATIM MATTERS. The router's phrasing is tuned to the reference
    answers; a judge that paraphrases its choice loses BERTScore for no
    reason. case130 lost 0.0288 to a single missing full stop."""
    answer, source = judge.parse_judgement("Answer 2", ("Yes", "No"))
    assert (answer, source) == ("No", "choice")


def test_a_labelled_restatement_still_counts_as_a_choice():
    answer, source = judge.parse_judgement("Answer 1: Yes", ("Yes", "No"))
    assert (answer, source) == ("Yes", "choice")


def test_free_text_is_SUPPRESSED_by_default_because_of_the_judges_register():
    """MEASURED, not cautious. The judge is a BASE Qwen3-VL-4B and its own
    verification run answered "3" where gold was "Three" (cluster 9707616) --
    right, in the wrong register. BERTScore punishes register, and the
    router's phrasing was tuned against the references while the judge's was
    not. Letting it rewrite an answer can lose points while being more
    correct, the same way case130 lost 0.0288 to a missing full stop."""
    answer, source = judge.parse_judgement(
        "Cadiere Forceps", ("Bipolar Forceps", "Clip applier"))
    assert answer is None and source == "freetext-suppressed"


def test_free_text_can_be_re_enabled_deliberately():
    """The case for it is real and stays reachable: on case124 BOTH candidates
    were wrong, and a judge restricted to picking could only choose the less
    wrong. Worth revisiting if the judge is ever fine-tuned on this corpus."""
    answer, source = judge.parse_judgement(
        "Cadiere Forceps", ("Bipolar Forceps", "Clip applier"),
        allow_freetext=True)
    assert (answer, source) == ("Cadiere Forceps", "freetext")


def test_an_empty_reply_returns_None_so_the_caller_can_fall_back():
    """An empty response scores 0 -- strictly worse than any wrong answer.
    Returning None lets the caller keep a real answer instead."""
    answer, source = judge.parse_judgement("   ", ("Yes", "No"))
    assert answer is None and source == "unparseable"


def test_an_out_of_range_label_is_unparseable_not_an_IndexError():
    answer, source = judge.parse_judgement("Answer 3", ("Yes", "No"))
    assert answer is None and source == "unparseable"


def test_an_out_of_range_labelled_restatement_also_falls_back():
    """'Answer 5: something' names a candidate that does not exist. Shipping
    the restatement would be shipping a reply the judge itself signalled it
    was confused about."""
    answer, source = judge.parse_judgement("Answer 5: Yes", ("Yes", "No"))
    assert answer is None and source == "unparseable"


def test_a_real_answer_that_merely_starts_with_the_word_answer_is_not_a_choice():
    """The label pattern must not swallow a legitimate answer. Guarding the
    out-of-range fix from over-reaching."""
    _answer, source = judge.parse_judgement(
        "Answering forceps are not present", ("Yes", "No"), allow_freetext=True)
    assert source == "freetext"
