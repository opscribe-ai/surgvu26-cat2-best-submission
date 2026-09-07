# tests/test_scoring.py
import pytest
from surgvu.scoring import normalize


def test_normalize_matches_organizer_implementation():
    # Verbatim from the challenge's evaluate.py: strip punctuation, lowercase, strip
    assert normalize("Yes, a large needle driver was utilized.") == \
        "yes a large needle driver was utilized"
    assert normalize("  Uterine horn  ") == "uterine horn"
    assert normalize("") == ""


@pytest.mark.slow
def test_identical_string_scores_one():
    from surgvu.scoring import Scorer
    scorer = Scorer()
    result = scorer.score_one("Yes", ["Yes", "Yes, a tool was used."])
    assert result["bertscore_f1"] > 0.99


@pytest.mark.slow
def test_wrong_polarity_scores_far_below_correct():
    from surgvu.scoring import Scorer
    scorer = Scorer()
    refs = ["Yes", "Yes, a large needle driver was utilized."]
    right = scorer.score_one("Yes", refs)["bertscore_f1"]
    wrong = scorer.score_one("No", refs)["bertscore_f1"]
    assert right > wrong + 0.2, (right, wrong)
