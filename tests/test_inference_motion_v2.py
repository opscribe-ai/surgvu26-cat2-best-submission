"""Additivity of the motion_v2 block.

clip_record's guarantee is that an omitted optional block leaves the dict
byte-identical to what it was before that block existed. That property is why
new evidence can be added without re-measuring every shipped answer, and it
holds only as long as every new block is tested for it.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.perceive import clip_record  # noqa: E402
from surgvu.taxonomy import TASK_CLASSES, TOOL_CLASSES  # noqa: E402


def _meta(classes):
    return {"classes": list(classes),
            "thresholds": [0.5] * len(classes)}


def _args():
    return ([0.1] * len(TOOL_CLASSES), _meta(TOOL_CLASSES),
            [0.1] * len(TASK_CLASSES), _meta(TASK_CLASSES), 16)


def test_omitting_motion_v2_changes_nothing():
    assert clip_record(*_args()) == clip_record(*_args(), motion_v2=None)


def test_supplying_motion_v2_only_appends_one_key():
    plain = clip_record(*_args())
    block = {"version": 2, "anchors": 16, "per_anchor": [], "summary": {}}
    with_v2 = clip_record(*_args(), motion_v2=block)
    assert set(with_v2) - set(plain) == {"motion_v2"}
    for key in plain:
        assert with_v2[key] == plain[key]


def test_motion_v2_survives_strict_json():
    block = {"version": 2, "anchors": 1,
             "per_anchor": [{"micro_short": None}], "summary": {}}
    record = clip_record(*_args(), motion_v2=block)
    json.loads(json.dumps(record, allow_nan=False))


def test_v1_and_v2_blocks_coexist():
    record = clip_record(*_args(),
                         motion={"version": 1, "bursts": 16},
                         motion_v2={"version": 2, "anchors": 16})
    assert record["motion"]["version"] == 1
    assert record["motion_v2"]["version"] == 2


# ------------------------------------------------------- R16: no drift
#
# The probe offsets (133, 400, 1200ms) used to be typed as a literal in four
# places: surgvu.perceive.DEFAULT_PROBE_OFFSETS_MS, surgvu.motion.
# VECTOR_SLOTS, scripts/dump_motion_v2.OFFSETS_MS, and (worse) written
# unconditionally into config/motion_v2.json by scripts/calibrate_motion_v2.py
# regardless of what the dump it read was actually produced with. Now
# surgvu.motion.PROBE_OFFSETS_MS is the one definition and the others import
# it. A single-line edit to any ONE of the three module-level consumers below
# (e.g. changing DEFAULT_PROBE_OFFSETS_MS back to a hand-typed tuple with a
# different value) makes this fail.

def test_probe_offsets_do_not_drift_between_consumers():
    import sys as _sys
    from pathlib import Path as _Path

    from surgvu import motion, perceive

    scripts_dir = str(_Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in _sys.path:
        _sys.path.insert(0, scripts_dir)
    import dump_motion_v2  # noqa: E402 - deliberately imported late

    assert perceive.DEFAULT_PROBE_OFFSETS_MS == motion.PROBE_OFFSETS_MS
    assert tuple(dump_motion_v2.OFFSETS_MS) == motion.PROBE_OFFSETS_MS
    assert (tuple(offset for _, offset in motion.VECTOR_SLOTS)
           == motion.PROBE_OFFSETS_MS)
