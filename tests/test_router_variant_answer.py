# tests/test_router_variant_answer.py
"""The variant gate: the first place this pipeline lets evidence change a
shipped answer, rather than only sitting next to one unread.

`router.variant_qualifier` is a pure lexical scan with NO tool-context
guard by design -- its own docstring says `variant_qualifier("Is a large
organ visible in this clip?")` returns `"large"`. `variant.variant_record`
abstains, but only on the axis it was trained for -- it has no opinion on
whether a needle driver is even in the clip. Neither fact is a bug in
either function; both are why `_variant_gate_answer` exists as a separate,
independently-testable choke point that ANDs four conditions before a
size-family opinion is allowed to flip a Yes/No.

Every scenario here is drawn from a measured case: case126 and case132 are
the two the gate is meant to fix (class-policy WRONG, gate RIGHT); case123
is the one class-policy already had right and the gate must not disturb;
case129 and case131 are the reason condition 3 (needle-driver detected)
exists at all -- both are DECIDED, confident, and wrong to trust, because
the detector never boxed a needle driver in either clip.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.router import (  # noqa: E402
    INTENT_TOOL_PRESENCE, _needle_driver_detected, _variant_gate_answer,
    answer_question, classify_question,
)
from surgvu.taxonomy import TASK_CLASSES, TOOL_CLASSES  # noqa: E402


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def perception(tools_present=(), yolo=None, variant=None):
    """A minimal perception dict, matching tests/test_router.py's shape,
    plus the two optional blocks this gate reads."""
    out = {
        "tools": {c: 0.0 for c in TOOL_CLASSES},
        "tools_present": list(tools_present),
        "task": {c: 0.0 for c in TASK_CLASSES},
        "task_top": None,
        "n_frames": 30,
    }
    if yolo is not None:
        out["yolo"] = yolo
    if variant is not None:
        out["variant"] = variant
    return out


def _yolo_needle_box_found():
    """A `yolo` block where the detector boxed a needle driver at least
    once -- the shape `_needle_driver_detected` reads."""
    return {
        "version": 1,
        "classes": list(TOOL_CLASSES),
        "by_class": {
            "needle driver": [
                {"cls": "needle driver", "conf": 0.87, "box": [10, 10, 90, 90],
                 "anchor_idx": 0, "t_seconds": 1.5},
            ],
        },
        "max_conf": {"needle driver": 0.87},
    }


def _yolo_needle_box_absent():
    """The case129/case131 shape: the detector ran and found other tools,
    but never a needle driver -- `detbox=False` in
    scripts/variant_sample_report.py's per-case table."""
    return {
        "version": 1,
        "classes": list(TOOL_CLASSES),
        "by_class": {
            "cadiere forceps": [
                {"cls": "cadiere forceps", "conf": 0.7, "box": [1, 1, 5, 5],
                 "anchor_idx": 0, "t_seconds": 0.5},
            ],
        },
        "max_conf": {"cadiere forceps": 0.7},
    }


def _variant_block(family, decided, p_large, p_mega, cutoff=0.51):
    return {
        "version": 1,
        "family": family,
        "p_large": p_large,
        "p_mega": p_mega,
        "cutoff": cutoff,
        "decided": decided,
    }


LARGE_QUESTION = "Was a large needle driver used in this clip?"


# --------------------------------------------------------------------------
# 1. the safety property: no "variant" block -> byte-identical
# --------------------------------------------------------------------------

def test_absent_variant_block_the_gate_never_fires():
    """No `variant` key at all -- true of every record until Task 11 -- must
    make `_variant_gate_answer` a no-op.

    BREAKS ON: rewriting `block = perception.get("variant") if
    isinstance(perception, dict) else None` together with the guard
    `if not isinstance(block, dict) or not block.get("decided"): return
    None` so that a missing key is treated as an already-decided block
    (e.g. defaulting to `{"decided": True, "family": family}` instead of
    `None`) -- any change that manufactures a decision from an absent key.
    """
    perc = perception(tools_present=["needle driver"],
                      yolo=_yolo_needle_box_found())
    assert "variant" not in perc
    assert _variant_gate_answer(LARGE_QUESTION, perc) is None


def test_absent_variant_block_answer_matches_class_policy_no_case():
    """Same safety property, exercised end to end. The needle driver is NOT
    installed, so the pre-existing class policy says "No" -- chosen
    specifically because it differs from FALLBACK_POLAR ("Yes"), so a
    regression that makes `_variant_gate_answer` raise (caught by
    `answer_question`'s handler and masked as FALLBACK_POLAR) is still
    visible here rather than silently passing.

    BREAKS ON: deleting the `not isinstance(block, dict) or` clause from
    the guard `if not isinstance(block, dict) or not block.get("decided"):
    return None` -- `None.get("decided")` then raises, which
    `answer_question` swallows into "Yes", flipping this assertion.
    """
    perc = perception(tools_present=[], yolo=_yolo_needle_box_found())
    assert answer_question(LARGE_QUESTION, perc) == "No"


# --------------------------------------------------------------------------
# 2. head undecided -> answer unchanged
# --------------------------------------------------------------------------

def test_undecided_head_the_gate_never_fires_even_with_a_family_present():
    """`decided=False` must block the gate on its own, independent of
    whatever `family` happens to hold -- so this deliberately uses a
    `family` value a real `variant_record` would never pair with
    `decided=False`, to isolate condition 4 from the `family in
    ("large","mega")` check that follows it.

    BREAKS ON: weakening the guard from
    `if not isinstance(block, dict) or not block.get("decided"): return
    None` to `if not isinstance(block, dict): return None` (dropping the
    decided check).
    """
    perc = perception(tools_present=["needle driver"],
                      yolo=_yolo_needle_box_found(),
                      variant=_variant_block("large", decided=False,
                                             p_large=0.55, p_mega=0.45))
    assert _variant_gate_answer(LARGE_QUESTION, perc) is None


def test_undecided_head_answer_falls_back_to_class_policy():
    perc = perception(tools_present=["needle driver"],
                      yolo=_yolo_needle_box_found(),
                      variant=_variant_block(None, decided=False,
                                             p_large=0.55, p_mega=0.45))
    assert answer_question(LARGE_QUESTION, perc) == "Yes"


# --------------------------------------------------------------------------
# 3. no needle driver detected -> answer unchanged, even on a confident head
#    (the case129/case131 scenario)
# --------------------------------------------------------------------------

def test_no_detected_needle_driver_blocks_a_confident_decided_head():
    """Mirrors case129 exactly: the head is confident (0.892) and decided,
    but the detector never boxed a needle driver in this clip. The gate
    must refuse regardless of how confident the head is.

    BREAKS ON: deleting the line
    `if not _needle_driver_detected(perception): return None`.
    """
    perc = perception(yolo=_yolo_needle_box_absent(),
                      variant=_variant_block("large", decided=True,
                                             p_large=0.892, p_mega=0.108))
    assert _needle_driver_detected(perc) is False
    assert _variant_gate_answer(LARGE_QUESTION, perc) is None


def test_no_detected_needle_driver_answer_stays_at_class_policy_no():
    """End to end: class policy says "No" (the CNN's `tools_present` does
    not list a needle driver either), and the confident "large" head must
    not override it to "Yes".
    """
    perc = perception(tools_present=[], yolo=_yolo_needle_box_absent(),
                      variant=_variant_block("large", decided=True,
                                             p_large=0.892, p_mega=0.108))
    assert answer_question(LARGE_QUESTION, perc) == "No"


# --------------------------------------------------------------------------
# 4 & 5. the gate firing -- case126 and case132's shapes
# --------------------------------------------------------------------------

def test_asks_large_head_says_large_detected_decided_yields_yes():
    """case126's shape: class policy alone said "No" (needle driver not in
    `tools_present`), gold was "Yes". The head says large; the gate must
    override to "Yes".

    BREAKS ON: `return "Yes" if head_family == family else "No"` ->
    `return "No" if head_family == family else "Yes"` (swap the branches).
    """
    perc = perception(tools_present=[], yolo=_yolo_needle_box_found(),
                      variant=_variant_block("large", decided=True,
                                             p_large=0.816, p_mega=0.184))
    assert _variant_gate_answer(LARGE_QUESTION, perc) == "Yes"
    assert answer_question(LARGE_QUESTION, perc) == "Yes"


def test_asks_large_head_says_mega_detected_decided_yields_no():
    """case132's shape: class policy alone said "Yes" (needle driver IS in
    `tools_present`), gold was "No". The head says mega; the gate must
    override to "No".

    BREAKS ON: `head_family == family` -> `head_family != family` (invert
    the comparison).
    """
    perc = perception(tools_present=["needle driver"],
                      yolo=_yolo_needle_box_found(),
                      variant=_variant_block("mega", decided=True,
                                             p_large=0.427, p_mega=0.573))
    assert _variant_gate_answer(LARGE_QUESTION, perc) == "No"
    assert answer_question(LARGE_QUESTION, perc) == "No"


# --------------------------------------------------------------------------
# 6. a non-tool question containing "large" -> unchanged, head never
#    consulted
# --------------------------------------------------------------------------

def test_organ_question_never_reaches_the_gate():
    """`variant_qualifier("Is a large organ visible in this clip?")`
    returns `"large"` by that function's own documented design -- it has no
    tool-context guard. The router's OWN intent classification must be
    what stops this question from being answered by a needle-driver-size
    head: it is not even INTENT_TOOL_PRESENCE.

    This is protected twice over -- classify_question routes it away from
    INTENT_TOOL_PRESENCE, AND mentioned_tool_classes(question) is empty --
    so no single line in `_variant_gate_answer` is uniquely load-bearing
    for this particular question; either the intent check or the
    mentioned-tool-classes check alone would already stop it. See the task
    report for the full analysis of which gate tests DO isolate a single
    line and which, like this one, are doubly covered.
    """
    question = "Is a large organ visible in this clip?"
    assert classify_question(question) != INTENT_TOOL_PRESENCE
    perc = perception(tools_present=["needle driver"],
                      yolo=_yolo_needle_box_found(),
                      variant=_variant_block("large", decided=True,
                                             p_large=0.95, p_mega=0.05))
    assert _variant_gate_answer(question, perc) is None
    assert answer_question(question, perc) == "Yes"  # FALLBACK_POLAR


# --------------------------------------------------------------------------
# 7. a question about a DIFFERENT tool containing "large" -> unchanged
# --------------------------------------------------------------------------

def test_a_different_tools_large_question_never_reaches_the_gate():
    """"Large" here modifies the cadiere forceps, not the needle driver --
    a real, plausible confusable, since `variant_qualifier` names a family
    from the bare word alone. Condition 2 restricts the gate to questions
    ABOUT the needle driver specifically. Unlike the organ case above, this
    question DOES clear the intent check (a cadiere forceps IS a
    recognised tool, so classify_question routes it to
    INTENT_TOOL_PRESENCE) -- so this test isolates the
    `mentioned_tool_classes(question) == frozenset({"needle driver"})`
    check on its own.

    BREAKS ON: deleting the line
    `if mentioned_tool_classes(question) != frozenset({"needle driver"}):
    return None`.
    """
    question = "Was a large cadiere forceps used in this clip?"
    assert classify_question(question) == INTENT_TOOL_PRESENCE
    perc = perception(tools_present=["cadiere forceps"],
                      yolo=_yolo_needle_box_found(),
                      variant=_variant_block("large", decided=True,
                                             p_large=0.95, p_mega=0.05))
    assert _variant_gate_answer(question, perc) is None
    # class policy: cadiere forceps is credible, so "Yes" -- unaffected by
    # the needle-driver head's opinion about the needle driver.
    assert answer_question(question, perc) == "Yes"
