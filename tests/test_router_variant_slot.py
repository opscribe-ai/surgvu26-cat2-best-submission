"""The variant qualifier must be readable without becoming an identifier.

_BRAND_TOKEN_STOPLIST is correct about its own job: "large" must not register
a class, or "the large forceps" would widen into a generic forceps question.
But the qualifier is still the difference between two gold answers, so it has
to be captured somewhere. A separate slot, not a change to the stoplist.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.router import variant_qualifier  # noqa: E402


def test_large_is_captured():
    assert variant_qualifier("Is a large needle driver being used?") == "large"


def test_mega_is_captured():
    assert variant_qualifier("Is the mega needle driver in view?") == "mega"


def test_case_and_punctuation_are_ignored():
    assert variant_qualifier("Is a LARGE, needle driver present?") == "large"


def test_absent_qualifier_is_none():
    assert variant_qualifier("Is a needle driver being used?") is None


def test_bare_suturecut_is_ambiguous():
    """SutureCut spans both families -- train-split counts from
    config/commercial_names.json: Large SutureCut Needle Driver 624, Mega
    SutureCut Needle Driver 285 + 2 -- so the bare word identifies neither
    on its own."""
    assert variant_qualifier("Is the suturecut driver used?") is None


def test_mega_suturecut_is_captured_as_mega():
    """Regression guard: 'mega suturecut' must not be lost to the
    both-families guard just because 'suturecut' used to be mis-marked
    as large-only."""
    assert variant_qualifier("Is the mega suturecut driver used?") == "mega"


def test_a_question_naming_both_returns_none_rather_than_guessing():
    q = "Is a large or mega needle driver being used?"
    assert variant_qualifier(q) is None


def test_the_stoplist_is_unchanged():
    """The fix must not widen specific questions as a side effect."""
    from surgvu.router import _BRAND_TOKEN_STOPLIST
    assert "large" in _BRAND_TOKEN_STOPLIST
    assert "mega" in _BRAND_TOKEN_STOPLIST
