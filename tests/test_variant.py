"""Tests for the variant head's decision layer.

The head must be allowed to ABSTAIN. A forced binary choice on an ambiguous
clip converts a 0.7015 polar answer into a coin flip between 1.0000 and
0.7015, which is only worth taking when the head is actually better than
chance on that clip -- and the cutoff is what encodes "actually better".
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.variant import variant_record  # noqa: E402


def test_confident_large_decides_large():
    out = variant_record({"large": 0.9, "mega": 0.1}, cutoff=0.65)
    assert out["family"] == "large"
    assert out["decided"] is True


def test_confident_mega_decides_mega():
    out = variant_record({"large": 0.2, "mega": 0.8}, cutoff=0.65)
    assert out["family"] == "mega"


def test_below_cutoff_abstains_with_family_none():
    out = variant_record({"large": 0.55, "mega": 0.45}, cutoff=0.65)
    assert out["family"] is None
    assert out["decided"] is False


def test_abstention_still_reports_both_probabilities():
    """Downstream may weigh a 0.55 differently from a 0.51."""
    out = variant_record({"large": 0.55, "mega": 0.45}, cutoff=0.65)
    assert out["p_large"] == pytest.approx(0.55)
    assert out["p_mega"] == pytest.approx(0.45)


def test_record_is_strict_json():
    json.loads(json.dumps(variant_record({"large": 0.5, "mega": 0.5}, 0.65),
                          allow_nan=False))


def test_rejects_probabilities_that_do_not_sum_to_one():
    with pytest.raises(ValueError):
        variant_record({"large": 0.9, "mega": 0.9}, cutoff=0.65)


def test_rejects_a_cutoff_at_or_below_chance():
    """A 0.5 cutoff never abstains, which defeats the point of having one."""
    with pytest.raises(ValueError):
        variant_record({"large": 0.9, "mega": 0.1}, cutoff=0.5)
