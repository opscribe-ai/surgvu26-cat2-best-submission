# tests/test_qa_forms.py
"""Pure-logic tests for the QA shared vocabulary. No torch, no video, no model.

Mirrors tests/test_router.py's contract: everything here runs in
milliseconds on a login node with no torch installed.
"""
import pytest

from surgvu.router import (
    INTENT_COUNT, INTENT_CUTTING, INTENT_ORGAN, INTENT_PROCEDURE,
    INTENT_PURPOSE, INTENT_SUTURE, INTENT_TASK, INTENT_TOOL_IDENTITY,
    INTENT_TOOL_PRESENCE, classify_question,
)
from surgvu.taxonomy import TASK_CLASSES, TOOL_CLASSES
from surgvu.qa_forms import QA_TEMPLATES, render

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

# One question per router intent, copied verbatim from
# tests/test_router.py::INTENT_PROBES (the module that already pins these
# classifications) -- MINUS the two catch-all fallbacks (unknown_polar,
# unknown_open). Those two are what a question gets when it matches NONE of
# the router's real rules; they are not "intents classify_question produces"
# for any question that actually asks something, so they are not part of
# "what classify_question can produce" for the purposes of this module.
ROUTER_PROBE_QUESTIONS = (
    "Is a needle driver being used?",
    "What instrument is being used?",
    "What organ is being manipulated?",
    "Is tissue being cut?",
    "Is suturing being performed?",
    "What type of procedure is being performed?",
    "What is the purpose of using forceps?",
    "What task is being performed?",
    "How many instruments are visible?",
)

EXPECTED_ROUTER_INTENTS = frozenset({
    INTENT_TOOL_PRESENCE, INTENT_TOOL_IDENTITY, INTENT_ORGAN, INTENT_CUTTING,
    INTENT_SUTURE, INTENT_PROCEDURE, INTENT_PURPOSE, INTENT_TASK,
    INTENT_COUNT,
})

# Slot names that name a taxonomy tool/task class in at least one template's
# example_slots. Used to drive the taxonomy-validation and artefact tests
# below without hardcoding which template has which slot.
_TOOL_CLASS_SLOTS = ("tool_class", "absent_class")
_TASK_CLASS_SLOTS = ("task_class", "asked_class", "actual_class")

# Values measured directly off the raw logbook -- see
# docs/design/notes/2026-08-24-label-vocab-hazards.md. Real values that
# must never reach a rendered answer.
LOGBOOK_ARTIFACTS = (
    "nan(camera in)",     # 1277 rows: the endoscope, not a tool
    "clip applier ",      # 882 rows: trailing space
    ' Single Site"',      # 29 rows: malformed CSV row, stray quote
    "",                   # 144 rows: empty string
    "Suturing",           # wrong-case task name (775 vs 609 lowercase)
)


def _templates_with_slot(slot_name):
    return [t for t in QA_TEMPLATES if slot_name in t.example_slots]


# --------------------------------------------------------------------------
# 1. every template renders a non-empty question and a non-empty answer
# --------------------------------------------------------------------------

@pytest.mark.parametrize("template", QA_TEMPLATES, ids=lambda t: t.intent)
def test_every_template_renders_nonempty_question_and_answer(template):
    question, answer = render(template, **template.example_slots)
    assert isinstance(question, str) and question.strip(), template.intent
    assert isinstance(answer, str) and answer.strip(), template.intent


# --------------------------------------------------------------------------
# 2. taxonomy-valued slots are drawn from surgvu.taxonomy, not free text
# --------------------------------------------------------------------------

def test_taxonomy_valued_slots_reject_values_outside_taxonomy():
    """A tool_class/task_class slot outside the taxonomy must raise.

    This is the whole guarantee: the only way render() can produce an answer
    for one of these slots is a value that is an exact member of
    TOOL_CLASSES or TASK_CLASSES. Free text -- a synonym, a raw logbook
    string, a plausible-looking guess -- is rejected before it can become a
    question or an answer.
    """
    checked_any = False
    for slot_name in _TOOL_CLASS_SLOTS:
        for template in _templates_with_slot(slot_name):
            checked_any = True
            bad_slots = dict(template.example_slots)
            bad_slots[slot_name] = "not a real tool class"
            with pytest.raises(ValueError):
                render(template, **bad_slots)
    for slot_name in _TASK_CLASS_SLOTS:
        for template in _templates_with_slot(slot_name):
            checked_any = True
            bad_slots = dict(template.example_slots)
            bad_slots[slot_name] = "not a real task class"
            with pytest.raises(ValueError):
                render(template, **bad_slots)
    assert checked_any, "no template exercised a taxonomy-valued slot"


def test_taxonomy_valued_slots_accept_every_real_class():
    """The flip side: every genuine taxonomy member must be usable."""
    for template in _templates_with_slot("tool_class"):
        for cls in TOOL_CLASSES:
            good_slots = dict(template.example_slots, tool_class=cls)
            render(template, **good_slots)  # must not raise
    for template in _templates_with_slot("task_class"):
        for cls in TASK_CLASSES:
            good_slots = dict(template.example_slots, task_class=cls)
            render(template, **good_slots)  # must not raise


# --------------------------------------------------------------------------
# 3. no template can render an answer containing a raw logbook artefact
# --------------------------------------------------------------------------

def test_no_template_can_render_an_answer_containing_a_raw_logbook_artefact():
    checked_any = False
    for slot_name in _TOOL_CLASS_SLOTS + _TASK_CLASS_SLOTS:
        for template in _templates_with_slot(slot_name):
            for artifact in LOGBOOK_ARTIFACTS:
                checked_any = True
                bad_slots = dict(template.example_slots)
                bad_slots[slot_name] = artifact
                with pytest.raises(ValueError):
                    render(template, **bad_slots)
    assert checked_any, "no template exercised a logbook-derived slot"


def test_valid_renders_carry_no_artifact_shape():
    """Positive-path check: a clean render never LOOKS like a dirty logbook
    value, even by accident (no 'nan(' substring, no leading/trailing
    whitespace, no unpaired double quote)."""
    for template in QA_TEMPLATES:
        question, answer = render(template, **template.example_slots)
        for text in (question, answer):
            assert "nan(" not in text.lower()
            assert text == text.strip()
            assert text.count('"') % 2 == 0
            assert text != ""


# --------------------------------------------------------------------------
# 4. templates cover intents beyond the router's 11
# --------------------------------------------------------------------------

def test_router_probes_classify_exactly_where_expected():
    """Guards the fixture above: if this ever drifts from
    tests/test_router.py::INTENT_PROBES, the superset test below would be
    silently checking the wrong baseline."""
    produced = {classify_question(q) for q in ROUTER_PROBE_QUESTIONS}
    assert produced == EXPECTED_ROUTER_INTENTS


def test_template_intents_are_a_strict_superset_of_router_intents():
    router_intents = {classify_question(q) for q in ROUTER_PROBE_QUESTIONS}
    template_intents = {template.intent for template in QA_TEMPLATES}
    assert router_intents.issubset(template_intents)
    assert template_intents != router_intents, (
        "QA_TEMPLATES must cover open-ended forms the router has no "
        "intent for, not just mirror the router's own 9 real intents")
