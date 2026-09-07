"""The probe that decides whether the evidence-conditioned adapter ships.

If this partitioning is wrong, the probe reports a confident verdict about a
model based on mislabelled records -- which is worse than not running it.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import evidence_reliance_probe as probe                          # noqa: E402
from surgvu.detect import YOLO_CLASSES                           # noqa: E402


def _variant(family, decided=True):
    return {"variant": {"family": family, "decided": decided,
                        "p_large": 0.5, "p_mega": 0.5}}


# --- variant lens ----------------------------------------------------------

def test_variant_evidence_agrees_when_decided_family_matches_a_yes():
    """Asked about mega, head decided mega, gold says Yes -- consistent."""
    assert probe.variant_verdict(
        "Does this clip show a mega needle driver?", "Yes",
        _variant("mega")) == probe.AGREES


def test_variant_evidence_agrees_when_a_different_family_matches_a_no():
    """Asked about mega, head decided LARGE, gold says No. The evidence
    supports the gold answer, so this is agreement -- not contradiction."""
    assert probe.variant_verdict(
        "Does this clip show a mega needle driver?", "No",
        _variant("large")) == probe.AGREES


def test_variant_evidence_contradicts_a_yes_it_does_not_support():
    """THE CASE132 SHAPE. The head decided large; gold says a mega IS shown.
    A parroting model follows the evidence and answers No."""
    assert probe.variant_verdict(
        "Does this clip show a mega needle driver?", "Yes",
        _variant("large")) == probe.CONTRADICTS


def test_variant_evidence_contradicts_a_no_it_does_not_support():
    assert probe.variant_verdict(
        "Does this clip show a mega needle driver?", "No",
        _variant("mega")) == probe.CONTRADICTS


def test_an_undecided_variant_head_is_undecidable_not_contradicting():
    """ABSENT evidence is not WRONG evidence. Counting an undecided head as a
    contradiction would put records carrying no signal into the bucket whose
    whole purpose is measuring behaviour under wrong signal, blurring the one
    distinction this probe exists to make."""
    assert probe.variant_verdict(
        "Does this clip show a mega needle driver?", "Yes",
        _variant("large", decided=False)) == probe.UNDECIDABLE


def test_a_non_polar_answer_is_undecidable():
    assert probe.variant_verdict(
        "Does this clip show a mega needle driver?", "Uterine horn",
        _variant("mega")) == probe.UNDECIDABLE


# --- tool lens -------------------------------------------------------------

def test_tool_evidence_agrees_when_the_named_tool_is_present_and_gold_is_yes():
    ev = {"tools_present": ["needle driver", "prograsp forceps"]}
    assert probe.tool_presence_verdict(
        "Is a needle driver visible?", "Yes", ev, YOLO_CLASSES) == probe.AGREES


def test_tool_evidence_contradicts_when_it_missed_a_tool_that_is_there():
    ev = {"tools_present": ["prograsp forceps"]}
    assert probe.tool_presence_verdict(
        "Is a needle driver visible?", "Yes", ev, YOLO_CLASSES) == probe.CONTRADICTS


def test_the_longest_tool_name_wins_the_match():
    """'bipolar forceps' must not be matched as 'forceps'. A short-name match
    would classify against the WRONG instrument and silently mislabel records."""
    assert probe.mentioned_tool("Are bipolar forceps visible?", YOLO_CLASSES) \
        == "bipolar forceps"


def test_a_question_naming_no_known_tool_is_undecidable():
    ev = {"tools_present": ["needle driver"]}
    assert probe.tool_presence_verdict(
        "Is anything visible?", "Yes", ev, YOLO_CLASSES) == probe.UNDECIDABLE


# --- partition / summary ---------------------------------------------------

def test_partition_routes_each_intent_to_its_own_lens():
    records = [
        {"intent": "variant_presence_polar",
         "question": "Does this clip show a mega needle driver?",
         "answer": "Yes", "evidence": _variant("large")},
        {"intent": "organ_open", "question": "What organ?",
         "answer": "Gallbladder", "evidence": _variant("mega")},
    ]
    buckets = probe.partition(records, YOLO_CLASSES)
    assert len(buckets[probe.CONTRADICTS]) == 1
    assert len(buckets[probe.UNDECIDABLE]) == 1
    assert buckets[probe.CONTRADICTS][0]["_probe_lens"] == "variant"


def test_summary_warns_when_the_contradicting_bucket_is_too_small():
    """Below ~30 the two scores are not separable, and a probe that reports a
    verdict it cannot support is worse than one that says so."""
    buckets = {probe.AGREES: [{}] * 500, probe.CONTRADICTS: [{}] * 4,
               probe.UNDECIDABLE: []}
    assert "WARNING" in probe.summarize(buckets)


def test_summary_is_quiet_when_the_bucket_is_big_enough():
    buckets = {probe.AGREES: [{}] * 500, probe.CONTRADICTS: [{}] * 80,
               probe.UNDECIDABLE: []}
    assert "WARNING" not in probe.summarize(buckets)


# --- the verdict --------------------------------------------------------
# The gap thresholds are the whole decision. If these invert or drift, the
# probe confidently recommends the opposite of what its data says.

def _res(agrees_mean, contradicts_mean, n=45):
    return {probe.AGREES: {"n": 500, "mean": agrees_mean},
            probe.CONTRADICTS: {"n": n, "mean": contradicts_mean}}


def test_verdict_calls_parroting_when_wrong_evidence_hurts_badly():
    """The shape that must not ship: near-perfect where evidence is right,
    polar-wrong where it is wrong -- the model is reading the evidence text."""
    out = probe.verdict(_res(0.98, 0.76))
    assert "PARROTING" in out
    assert "Do NOT ship" in out


def test_a_near_zero_gap_is_the_PASS_not_an_inconclusive_result():
    """THE CORRECTION THIS TEST EXISTS FOR. A model reading the PIXELS scores
    the same whether the evidence beside them is right or wrong -- so success
    is gap ~= 0. The first draft of verdict() called that band
    "EVIDENCE-NEUTRAL ... not obviously gaining", i.e. it reported the desired
    outcome as a non-answer, which would have stalled a shippable adapter."""
    out = probe.verdict(_res(0.95, 0.92))
    assert "NOT PARROTING" in out and "PASS" in out
    assert "0.9092" in out, "must still point at the score it has to beat"


def test_scoring_much_better_on_wrong_evidence_is_flagged_as_anomalous():
    """Not a triumph -- fusion is gap ~= 0, so a large NEGATIVE gap means the
    two buckets differ in something other than evidence correctness."""
    out = probe.verdict(_res(0.85, 0.99))
    assert "ANOMALOUS" in out


def test_verdict_refuses_when_a_bucket_is_empty():
    assert "INCONCLUSIVE" in probe.verdict(
        {probe.AGREES: {"n": 0, "mean": None},
         probe.CONTRADICTS: {"n": 45, "mean": 0.9}})


def test_the_parroting_threshold_is_half_the_measured_behaviour_gap():
    """0.09 is half the ~0.18 that separates parroting (0.7612) from fusion
    (0.9735) at n=45. Pinned because a threshold nobody can derive later gets
    'tuned' until it says what someone wanted."""
    assert "PARROTING" in probe.verdict(_res(0.98, 0.98 - 0.10))
    assert "NOT PARROTING" in probe.verdict(_res(0.98, 0.98 - 0.08))


def test_the_scorer_returns_a_dict_not_a_float():
    """PINNED AFTER A REAL FAILURE. surgvu.scoring.Scorer.score_one returns
    {"bertscore_f1": float}; the probe summed the dicts directly and raised
    "unsupported operand type(s) for +: 'int' and 'dict'" -- AFTER all 195
    generations had run, throwing away the expensive half of the job for a
    missing key lookup.

    Asserted against the real source rather than a mock, so a change to the
    Scorer's return shape breaks this test rather than a GPU job."""
    src = (Path(__file__).resolve().parents[1] / "src" / "surgvu" / "scoring.py").read_text()
    assert 'return {"bertscore_f1"' in src, (
        "Scorer.score_one no longer returns a dict keyed 'bertscore_f1'; "
        "evidence_reliance_probe.score_partition unwraps that key explicitly")
