"""Tests for the YOLO adapter.

Deliberately torch-free: the mapping, the record shape and the timestamping
are the parts that decide answers, and they must be testable on the login
node where torch does not exist. Detector.detect itself is exercised in the
container by tests/test_detect_weights.py.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.detect import (YOLO_CLASSES, detections_to_record,  # noqa: E402
                           map_to_taxonomy)
from surgvu.taxonomy import TOOL_CLASSES  # noqa: E402


def test_fourteen_classes_in_yaml_order():
    assert len(YOLO_CLASSES) == 14
    assert YOLO_CLASSES[0] == "bipolar dissector"


def test_twelve_of_fourteen_map_into_the_taxonomy():
    mapped = [map_to_taxonomy(name) for name in YOLO_CLASSES]
    assert sorted(n for n in mapped if n) == sorted(TOOL_CLASSES)


def test_out_of_taxonomy_classes_map_to_none():
    """Kept as evidence, never emitted as an answer."""
    assert map_to_taxonomy("bipolar dissector") is None
    assert map_to_taxonomy("suction irrigator") is None


def test_unknown_name_raises_rather_than_silently_dropping():
    with pytest.raises(KeyError):
        map_to_taxonomy("laser sword")


def test_record_preserves_time_not_just_presence():
    detections = [
        [{"cls": "needle driver", "conf": 0.9, "box": [0, 0, 1, 1]}],
        [],
        [{"cls": "needle driver", "conf": 0.8, "box": [0, 0, 1, 1]}],
    ]
    record = detections_to_record(detections, [0.0, 1.875, 3.75])
    times = [d["t_seconds"] for d in record["by_class"]["needle driver"]]
    assert times == [0.0, 3.75]


def test_record_reports_max_confidence_per_class():
    # Higher confidence FIRST: a last-write-wins regression (no max()) would
    # end at 0.4, not 0.9, so this actually distinguishes "true max" from
    # "last write" -- ascending order does not.
    detections = [
        [{"cls": "needle driver", "conf": 0.9, "box": [0, 0, 1, 1]}],
        [{"cls": "needle driver", "conf": 0.4, "box": [0, 0, 1, 1]}],
    ]
    record = detections_to_record(detections, [0.0, 1.875])
    assert record["max_conf"]["needle driver"] == pytest.approx(0.9)


def test_empty_detections_give_an_empty_record_not_a_crash():
    record = detections_to_record([[], []], [0.0, 1.875])
    assert record["by_class"] == {}
    assert record["max_conf"] == {}


def test_record_is_strict_json():
    record = detections_to_record(
        [[{"cls": "stapler", "conf": 0.5, "box": [1, 2, 3, 4]}]], [0.0])
    json.loads(json.dumps(record, allow_nan=False))


def test_mismatched_timestamps_raise():
    with pytest.raises(ValueError):
        detections_to_record([[], []], [0.0])
