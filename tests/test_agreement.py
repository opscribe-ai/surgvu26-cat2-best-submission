"""Tests for CNN/YOLO agreement.

The asymmetry matters: the detector has two classes the CNNs cannot name, and
a detection of one of those is not a disagreement -- the CNN was never asked.
Counting it as one would make every frame containing a suction irrigator look
uncertain.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.agreement import agreement_record  # noqa: E402


def _yolo(max_conf):
    return {"version": 1, "max_conf": dict(max_conf), "by_class": {},
            "per_anchor": []}


def test_full_agreement_scores_one():
    out = agreement_record({"needle driver": 0.9, "stapler": 0.1},
                           {"needle driver": 0.5, "stapler": 0.5},
                           _yolo({"needle driver": 0.8}))
    assert out["tool_agreement"] == pytest.approx(1.0)
    assert out["both_present"] == ["needle driver"]


def test_cnn_only_detection_is_recorded_as_disagreement():
    out = agreement_record({"needle driver": 0.9},
                           {"needle driver": 0.5},
                           _yolo({}))
    assert out["cnn_only"] == ["needle driver"]
    assert out["tool_agreement"] < 1.0


def test_yolo_only_detection_is_recorded_as_disagreement():
    out = agreement_record({"needle driver": 0.1},
                           {"needle driver": 0.5},
                           _yolo({"needle driver": 0.8}))
    assert out["yolo_only"] == ["needle driver"]


def test_out_of_taxonomy_detection_is_not_a_disagreement():
    """The CNNs were never asked about suction irrigator."""
    out = agreement_record({"needle driver": 0.9},
                           {"needle driver": 0.5},
                           _yolo({"needle driver": 0.8,
                                  "suction irrigator": 0.9}))
    assert out["yolo_only"] == []
    assert out["tool_agreement"] == pytest.approx(1.0)


def test_neither_finds_anything_is_agreement_not_a_zero():
    out = agreement_record({"needle driver": 0.1},
                           {"needle driver": 0.5},
                           _yolo({}))
    assert out["tool_agreement"] == pytest.approx(1.0)


def test_top_disagreement_names_the_widest_gap():
    out = agreement_record({"needle driver": 0.95, "stapler": 0.55},
                           {"needle driver": 0.5, "stapler": 0.5},
                           _yolo({"stapler": 0.52}))
    assert out["top_disagreement"] == ["needle driver", "cnn_only"]


def test_no_disagreement_leaves_top_none():
    out = agreement_record({"needle driver": 0.9},
                           {"needle driver": 0.5},
                           _yolo({"needle driver": 0.8}))
    assert out["top_disagreement"] is None
