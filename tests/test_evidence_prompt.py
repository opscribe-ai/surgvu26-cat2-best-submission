"""Tests for the Evidence VLM's prompt renderer: `build_sampling_prompt`
rendering `surgvu.perceive.clip_record`-shaped context into text.

Torch-free, same discipline as tests/test_evidence_vlm.py: nothing here
imports torch, and `surgvu.evidence_vlm` must stay importable without it.
`surgvu.router` is imported in ONE test (the calibration cross-check) --
router.py is itself torch-free at module scope, confirmed by that import
succeeding here.
"""
import json

import pytest

from surgvu import evidence_vlm
from surgvu.evidence_vlm import build_sampling_prompt


# --------------------------------------------------------------- YOLO block

def test_yolo_detections_are_rendered_with_their_timestamps():
    """The statement pooled confidences could never make: WHEN a tool was
    seen, not just whether."""
    context = {
        "yolo": {
            "version": 1,
            "classes": ["needle driver"],
            "by_class": {
                "needle driver": [
                    {"cls": "needle driver", "conf": 0.9, "box": [0, 0, 1, 1],
                     "anchor_idx": 2, "t_seconds": 3.7},
                    {"cls": "needle driver", "conf": 0.8, "box": [0, 0, 1, 1],
                     "anchor_idx": 5, "t_seconds": 9.4},
                ],
            },
            "max_conf": {"needle driver": 0.9},
        },
    }
    prompt = build_sampling_prompt("Which tool is used?", context)
    assert "needle driver" in prompt
    assert "3.7s" in prompt
    assert "9.4s" in prompt


def test_yolo_block_absent_entirely_when_key_missing():
    prompt = build_sampling_prompt("Q?", {})
    assert "detector" not in prompt.lower()


def test_yolo_present_but_empty_is_an_explicit_sentence_not_omitted():
    """The detector RAN and found nothing -- different information from the
    block never having run, and must not render as a null."""
    prompt = build_sampling_prompt(
        "Q?", {"yolo": {"by_class": {}, "max_conf": {}, "classes": []}})
    assert "detector" in prompt.lower()
    assert "none" not in prompt.lower()


def test_yolo_timestamps_are_capped_for_a_busy_clip():
    """Budget awareness: a clip with many classes and many detections must
    not make the prompt grow without bound."""
    by_class = {}
    max_conf = {}
    for i in range(20):
        name = "class-%d" % i
        by_class[name] = [
            {"cls": name, "conf": 0.5, "box": [0, 0, 1, 1],
             "anchor_idx": j, "t_seconds": float(j)}
            for j in range(30)
        ]
        max_conf[name] = 0.5 + i * 0.001
    context = {"yolo": {"by_class": by_class, "max_conf": max_conf,
                        "classes": list(by_class)}}
    prompt = build_sampling_prompt("Q?", context)
    yolo_line = next(l for l in prompt.splitlines() if "Detector timestamps" in l)
    # At most MAX_YOLO_CLASSES_IN_PROMPT classes named, plus one "+N more".
    assert yolo_line.count(" at ") <= evidence_vlm.MAX_YOLO_CLASSES_IN_PROMPT
    assert "more" in yolo_line
    assert len(prompt) < 2000


# ------------------------------------------------------------- motion block

def test_motion_v2_renders_calibrated_words_not_raw_floats():
    context = {
        "motion_v2": {
            "version": 2,
            "summary": {
                "macro_prev": {"mean": 5.0449, "max": 6.0, "measured": 10},
                "macro_next": {"mean": None, "max": None, "measured": 0},
                "micro_short": {"mean": None, "max": None, "measured": 0},
                "flow_moving_fraction": {"mean": 0.0822, "max": 0.2,
                                         "measured": 10},
            },
        },
    }
    prompt = build_sampling_prompt("Q?", context)
    assert "active" in prompt.lower()
    assert "tool-dominant" in prompt.lower()
    assert "5.0449" not in prompt
    assert "0.0822" not in prompt


def test_motion_v2_camera_dominant_when_moving_fraction_is_high():
    context = {
        "motion_v2": {
            "summary": {
                "macro_prev": {"mean": 5.0, "max": 6.0, "measured": 10},
                "flow_moving_fraction": {"mean": 1.0, "max": 1.0,
                                         "measured": 10},
            },
        },
    }
    prompt = build_sampling_prompt("Q?", context)
    assert "camera-dominant" in prompt.lower()


def test_motion_v2_still_below_the_active_threshold():
    context = {
        "motion_v2": {
            "summary": {
                "macro_prev": {"mean": 0.1, "max": 0.2, "measured": 10},
                "flow_moving_fraction": {"mean": 0.9, "max": 1.0,
                                         "measured": 10},
            },
        },
    }
    prompt = build_sampling_prompt("Q?", context)
    assert "still" in prompt.lower()


def test_motion_block_absent_entirely_when_no_key_present():
    prompt = build_sampling_prompt("Q?", {})
    assert "motion" not in prompt.lower()


def test_motion_v2_with_nothing_measured_is_an_explicit_sentence():
    context = {
        "motion_v2": {
            "summary": {
                "macro_prev": {"mean": None, "max": None, "measured": 0},
                "macro_next": {"mean": None, "max": None, "measured": 0},
                "micro_short": {"mean": None, "max": None, "measured": 0},
                "flow_moving_fraction": {"mean": None, "max": None,
                                         "measured": 0},
            },
        },
    }
    prompt = build_sampling_prompt("Q?", context)
    assert "no measurable motion" in prompt.lower()
    assert "none" not in prompt.lower()


def test_motion_v1_fallback_used_when_v2_absent():
    context = {
        "motion": {
            "bursts_measured": 5,
            "micro": {"mean": 5.0, "max": 6.0, "per_burst": []},
            "macro": {"mean": 5.0, "max": 6.0, "per_gap": []},
        },
    }
    prompt = build_sampling_prompt("Q?", context)
    assert "active" in prompt.lower()
    assert "camera" not in prompt.lower() or "not distinguishable" in prompt.lower()


def test_motion_v1_zero_bursts_measured_is_explicit_not_omitted():
    context = {"motion": {"bursts_measured": 0, "micro": {}, "macro": {}}}
    prompt = build_sampling_prompt("Q?", context)
    assert "no measurable motion" in prompt.lower()


def test_motion_active_threshold_matches_router_calibration():
    """Pinned so the two constants -- this module's borrowed anchor and
    router.py's fitted one -- cannot silently drift apart."""
    from surgvu.router import STATIC_ACTIVITY_THRESHOLD
    assert evidence_vlm.MOTION_ACTIVE_THRESHOLD == STATIC_ACTIVITY_THRESHOLD


# ------------------------------------------------------------ variant block

def test_variant_block_decided_names_the_family():
    context = {"variant": {"family": "large", "p_large": 0.91, "p_mega": 0.09,
                           "cutoff": 0.75, "decided": True}}
    prompt = build_sampling_prompt("Q?", context)
    assert "large" in prompt.lower()
    assert "0.91" in prompt


def test_variant_block_abstention_is_distinct_from_absence():
    """decided=False ('the head could not tell') must read differently from
    the block never having run at all."""
    decided_false = build_sampling_prompt(
        "Q?", {"variant": {"family": None, "p_large": 0.5, "p_mega": 0.5,
                           "cutoff": 0.75, "decided": False}})
    absent = build_sampling_prompt("Q?", {})
    assert "could not decide" in decided_false.lower()
    assert "variant" not in absent.lower()
    assert decided_false != absent


def test_variant_block_absent_entirely_when_no_key_present():
    prompt = build_sampling_prompt("Q?", {})
    assert "variant" not in prompt.lower()


# --------------------------------------------------------------- UI band

def test_prompt_never_contains_anything_derived_from_the_ui_band():
    """The rules prohibit using UI-visible information; `preprocess.
    prepare_frame` blurs the band on every frame for exactly this reason.
    Nothing this renderer does may reintroduce it, even if a caller's
    context dict happens to carry a plausible-looking UI string under some
    key this function does not know about."""
    context = {
        "tools_present": ["needle driver"],
        "tools": {"needle driver": 0.9},
        "task_top": "suturing",
        "task": {"suturing": 0.8},
        "yolo": {"by_class": {"needle driver": [
            {"cls": "needle driver", "conf": 0.9, "box": [0, 0, 1, 1],
             "anchor_idx": 0, "t_seconds": 1.0}]},
                "max_conf": {"needle driver": 0.9}},
        "motion_v2": {"summary": {
            "macro_prev": {"mean": 5.0, "max": 6.0, "measured": 1},
            "flow_moving_fraction": {"mean": 0.1, "max": 0.1, "measured": 1}}},
        "variant": {"family": "large", "p_large": 0.9, "p_mega": 0.1,
                   "cutoff": 0.75, "decided": True},
        # A rogue key mimicking UI-band-derived text. No renderer knows this
        # name, so it must never reach the prompt -- proving the function
        # reads only known evidence keys rather than dumping `context`
        # wholesale.
        "ui_band_text": "tool list: needle driver, cadiere forceps",
        "overlay_tool_list": ["needle driver", "cadiere forceps"],
    }
    prompt = build_sampling_prompt("What tool is mounted?", context)
    assert "overlay" not in prompt.lower()
    assert "numbered" not in prompt.lower()
    assert "tool list" not in prompt.lower()
    assert "ui_band_text" not in prompt
    assert "overlay_tool_list" not in prompt


def test_build_sampling_prompt_source_never_reads_context_wholesale():
    """A structural guard alongside the behavioural one above: the function
    must consult context by name, not serialise or iterate it wholesale."""
    import inspect
    source = inspect.getsource(evidence_vlm.build_sampling_prompt)
    assert "context.items()" not in source
    assert "context.values()" not in source
    assert "**context" not in source


# ------------------------------------------------------- tools / task blocks

def test_tools_present_rendered_with_probabilities():
    context = {"tools_present": ["needle driver", "cadiere forceps"],
              "tools": {"needle driver": 0.91, "cadiere forceps": 0.67}}
    prompt = build_sampling_prompt("Q?", context)
    assert "needle driver" in prompt
    assert "0.91" in prompt
    assert "cadiere forceps" in prompt
    assert "0.67" in prompt


def test_tools_present_empty_omits_the_section():
    prompt = build_sampling_prompt("Q?", {"tools_present": [], "tools": {}})
    assert "instrument classifier" not in prompt.lower()


def test_task_top_rendered_with_its_probability():
    context = {"task_top": "suturing", "task": {"suturing": 0.74, "other": 0.1}}
    prompt = build_sampling_prompt("Q?", context)
    assert "suturing" in prompt.lower()
    assert "0.74" in prompt


def test_task_top_absent_omits_the_section():
    prompt = build_sampling_prompt("Q?", {})
    assert "activity classifier" not in prompt.lower()


# ------------------------------------------------------- backward compatibility

def test_old_style_context_still_renders_alongside_new_blocks():
    """robot_tools/task_description (the ported original's fields) and the
    new evidence-packet blocks can coexist in one context without either
    clobbering the other."""
    context = {
        "robot_tools": ["needle driver"],
        "task_description": "Suturing",
        "tools_present": ["cadiere forceps"],
        "tools": {"cadiere forceps": 0.8},
    }
    prompt = build_sampling_prompt("Q?", context)
    assert "Tools mounted on the robot: needle driver." in prompt
    assert "Procedure context: Suturing" in prompt
    assert "cadiere forceps" in prompt


def test_fully_empty_context_omits_every_evidence_section():
    """Requirement: an absent block omits its section entirely, never
    'None' and never an empty header, for every block at once."""
    prompt = build_sampling_prompt("Is a tool present?", {})
    lines = prompt.splitlines()
    assert lines[0] == "Question: Is a tool present?"
    for forbidden in ("none", "null", "n/a"):
        assert forbidden not in prompt.lower()
    for section in ("instrument classifier", "activity classifier",
                    "detector", "motion", "variant", "agreement"):
        assert section not in prompt.lower()


def test_prompt_is_strict_text_not_json():
    """Sanity: the prompt is plain text for a chat template, not a
    serialised blob -- guards against a future change that dumps the packet
    as JSON into the prompt instead of rendering it."""
    context = {"tools_present": ["needle driver"], "tools": {"needle driver": 0.9}}
    prompt = build_sampling_prompt("Q?", context)
    with pytest.raises(json.JSONDecodeError):
        json.loads(prompt)


# ------------------------------------------------------------- agreement

def test_evidence_vlm_does_not_import_surgvu_agreement():
    """`_render_agree_block` reads only the `agree` dict's own fields --
    `tool_agreement`, `both_present`, `top_disagreement` -- so this module
    never needs to import `surgvu.agreement` itself to render its output.
    Kept as a guard against a future change coupling the two modules for no
    reason."""
    import inspect
    source = inspect.getsource(evidence_vlm)
    assert "import agreement" not in source
    assert "from .agreement" not in source
    assert "from surgvu.agreement" not in source


def test_agree_block_absent_entirely_when_no_key_present():
    prompt = build_sampling_prompt("Q?", {})
    assert "agreement" not in prompt.lower()


def test_agree_high_renders_as_corroboration_not_a_raw_float():
    context = {"agree": {"version": 1, "tool_agreement": 1.0,
                         "both_present": ["needle driver"],
                         "cnn_only": [], "yolo_only": [],
                         "top_disagreement": None}}
    prompt = build_sampling_prompt("Q?", context)
    assert "high" in prompt.lower()
    assert "needle driver" in prompt.lower()
    assert "1.0" not in prompt


def test_agree_low_names_the_lone_caller_and_which_model():
    context = {"agree": {"version": 1, "tool_agreement": 0.0,
                         "both_present": [], "cnn_only": ["stapler"],
                         "yolo_only": [],
                         "top_disagreement": ["stapler", "cnn_only"]}}
    prompt = build_sampling_prompt("Q?", context)
    assert "low" in prompt.lower()
    assert "stapler" in prompt.lower()
    assert "classifier" in prompt.lower()
    assert "0.0" not in prompt


def test_an_agree_key_with_the_old_mock_shape_is_harmlessly_ignored():
    """A dict under `agree` that does not carry `tool_agreement` (e.g. a
    stale caller using a shape that predates `surgvu.agreement`) must not
    crash and must not appear in the prompt -- the renderer omits the
    section rather than guessing at an unfamiliar shape."""
    context = {"agree": {"version": 1, "disagreement": 0.4}}
    prompt = build_sampling_prompt("Q?", context)
    assert "disagreement" not in prompt.lower()
    assert "agreement" not in prompt.lower()
