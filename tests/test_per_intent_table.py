"""Tests for scripts/per_intent_table.py -- the script that turns an eval
report into `config/arbiter.json`'s `vlm_intents` list.

Torch-free: `paired_stats`, `case_key` and `verify_alignment` are pure, and
they are the three places this script can be silently wrong. `main` is not
exercised here (it needs a real report, a real manifest and BERTScore); what
IS exercised is every guard that stands between a drifted sample and a
`vlm_intents` list nobody can attribute.
"""
import importlib.util
import math
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "per_intent_table",
    Path(__file__).resolve().parents[1] / "scripts" / "per_intent_table.py")
pit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pit)


def record(case="case_007", part="1.0", t_start=12.5):
    return {"case": case, "part": part, "t_start": t_start}


def row(case_id, f1=0.5):
    return {"case_id": case_id, "bertscore_f1": f1}


# --------------------------------------------------------------------------
# the alignment check -- the guard that makes an INDEX join legitimate
# --------------------------------------------------------------------------

def test_case_key_matches_the_format_train_vlm_writes():
    """Pinned to the exact f-string in train_vlm.run_eval. If that changes,
    this fails rather than the join silently pairing wrong rows."""
    assert pit.case_key(record()) == "case_007|1.0|12.500"


def test_case_key_rounds_t_start_to_three_places():
    """The report's key is %.3f, so two windows 0.0001s apart collapse to one
    string. That is exactly why the join is by index and this is only a
    CHECK -- asserted here so the limitation is recorded, not discovered."""
    assert pit.case_key(record(t_start=12.5001)) == pit.case_key(record(t_start=12.5002))


def test_verify_alignment_accepts_a_matching_sample():
    records = [record(t_start=1.0), record(t_start=2.0)]
    pit.verify_alignment(records, [row("case_007|1.0|1.000"),
                                   row("case_007|1.0|2.000")])


def test_verify_alignment_rejects_a_length_mismatch():
    """A drifted --max-eval-examples or --seed must abort, not truncate."""
    with pytest.raises(SystemExit, match="rebuilt 2 eval record"):
        pit.verify_alignment([record(t_start=1.0), record(t_start=2.0)],
                             [row("case_007|1.0|1.000")])


def test_verify_alignment_rejects_a_reordered_sample():
    """THE FAILURE THIS SCRIPT EXISTS TO PREVENT. Same records, same count,
    different order -- an unchecked index join would pair every question with
    another question's score and produce a per-intent table that looks
    entirely reasonable."""
    records = [record(t_start=1.0), record(t_start=2.0)]
    with pytest.raises(SystemExit, match="index 0"):
        pit.verify_alignment(records, [row("case_007|1.0|2.000"),
                                       row("case_007|1.0|1.000")])


def test_verify_alignment_reports_the_first_bad_index():
    records = [record(t_start=1.0), record(t_start=2.0), record(t_start=3.0)]
    with pytest.raises(SystemExit, match="index 2"):
        pit.verify_alignment(records, [row("case_007|1.0|1.000"),
                                       row("case_007|1.0|2.000"),
                                       row("case_007|1.0|9.000")])


def test_verify_alignment_rejects_a_report_row_with_no_case_id():
    with pytest.raises(SystemExit, match="index 0"):
        pit.verify_alignment([record(t_start=1.0)], [{"bertscore_f1": 0.5}])


# --------------------------------------------------------------------------
# paired_stats -- the arithmetic the arming decision rests on
# --------------------------------------------------------------------------

def test_paired_stats_is_paired_not_two_sample():
    """THE PROPERTY THAT MAKES THE BAR MEANINGFUL. Both lists have huge
    between-record spread (0.1 to 0.9) but the VLM is uniformly +0.10 on
    every single record. Paired, that is a zero-variance difference and the
    standard error is 0. A two-sample calculation would report a large one
    and refuse to arm an intent the VLM wins outright."""
    router_scores = [0.1, 0.5, 0.9, 0.3, 0.7]
    vlm_scores = [r + 0.10 for r in router_scores]
    delta, se, n = pit.paired_stats(router_scores, vlm_scores)
    assert n == 5
    assert delta == pytest.approx(0.10)
    assert se == pytest.approx(0.0, abs=1e-12)


def test_paired_stats_sign_is_vlm_minus_router():
    """A flipped sign would arm exactly the intents the VLM LOSES."""
    delta, _, _ = pit.paired_stats([0.9, 0.9], [0.2, 0.2])
    assert delta < 0


def test_paired_stats_reports_infinite_se_for_a_single_record():
    """One record cannot support a variance. Infinity makes `delta > sigma*se`
    false, so a one-record intent is never armed -- the safe direction."""
    delta, se, n = pit.paired_stats([0.4], [0.9])
    assert n == 1 and delta == pytest.approx(0.5) and math.isinf(se)


def test_paired_stats_handles_an_empty_intent():
    delta, se, n = pit.paired_stats([], [])
    assert n == 0 and delta == 0.0 and math.isinf(se)


def test_paired_stats_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="differ in length"):
        pit.paired_stats([0.1, 0.2], [0.3])


def test_paired_stats_se_shrinks_with_n():
    """Sanity: the same difference measured on more records is more certain,
    so a marginal intent can become armable purely by evaluating more."""
    _, se_small, _ = pit.paired_stats([0.1, 0.9], [0.3, 0.8])
    _, se_large, _ = pit.paired_stats([0.1, 0.9] * 50, [0.3, 0.8] * 50)
    assert se_large < se_small


def test_default_sigma_is_a_two_sided_95_percent_bar():
    """Pinned so the arming bar cannot be loosened without a deliberate edit
    -- every intent armed costs a submission to find out about."""
    assert pit.DEFAULT_SIGMA == 2.0
