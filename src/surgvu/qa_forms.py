"""The shared vocabulary of QA question templates and answer forms.

This module is what `scripts/build_qa_pairs.py` (QA generation, Task 2) and
any future evaluation of the fine-tuned VLM both import. There must be
exactly ONE definition of "what a question about this corpus looks like" and
"what a correct answer looks like" -- two copies drift, silently, and the
drift is invisible until the model is scored. See
docs/design/plans/2026-08-25-v5-plan3-vlm-training.md, Task 1.

WHAT THIS DOES NOT DO. It does not read tools.csv/tasks.csv (that is Task 2)
and it does not touch perception, torch, or the router's classification
logic. It is a tuple of templates plus pure functions over already-clean
slot values, so it imports and its tests pass in milliseconds on a login
node with no torch installed.

REUSE, NOT RECREATION. `src/surgvu/router.py` already carries the canonical
answer vocabulary for the 9 question types it can identify --
CLASS_DISPLAY_NAMES, TASK_DISPLAY, TASK_ORGANS, PURPOSES, PROCEDURE_ANSWER,
COUNT_WORDS, and the INTENT_* constants themselves. This module imports
those tables read-only (router.py is settled and is not modified) so that
"generation and evaluation cannot drift apart" is literally true rather than
a hope: there is exactly one Cadiere Forceps string in this codebase, not
two copies that could someday disagree.

WHITELIST, NEVER FREE TEXT. Every slot that names a tool or task class is
validated against surgvu.taxonomy before it is allowed to touch a question
or an answer. docs/design/notes/2026-08-24-label-vocab-hazards.md
measured the raw logbook directly: roughly 19% of groundtruth_toolname is
not a tool at all ("nan(camera in)" 1277 times, 144 empty strings, a
malformed ' Single Site"' row), every task name appears in both cases
("Suturing" 775 / "suturing" 609), and "clip applier " carries a trailing
space. A generator that calls render() with a raw logbook string and expects
a clean answer back is the exact bug this module exists to make impossible:
_require_tool_class/_require_task_class raise ValueError on anything that is
not an EXACT member of TOOL_CLASSES / TASK_CLASSES, so a caller is forced to
normalise with surgvu.taxonomy.normalize_tool/normalize_task first, and a
malformed value cannot silently ride through into training data.

BEYOND THE ROUTER'S 11 INTENTS. The router always answers something -- that
is its job -- but three question types below have no router intent at all,
because the router was never asked to be right about them:

  * INTENT_VARIANT_PRESENCE -- a specific size-family bet ("was a LARGE
    needle driver used"), answered from the actual installed family rather
    than merely "was a needle driver used". The variant head already
    resolves this distinction in serving (router._variant_gate_answer); this
    is the question-side form the VLM should learn the same thing from.
  * INTENT_TOOL_ABSENCE -- a named absence ("which instrument class is NOT
    in use"). The router has a presence intent and an identity intent, but
    nothing that asks what is missing.
  * INTENT_TASK_CONFIRMATION -- confirm-or-deny a SPECIFIC candidate task.
    The router's task_open only asks WHICH task is underway; asked to
    confirm a proposed one, an unmatched router rule falls through to
    unknown_polar, whose calibrated answer is the constant "Yes" regardless
    of whether the guess is right. That is exactly the shape of failure this
    plan's header demonstrates directly ("Is the patient stable?" -> always
    "Yes").
"""
from collections import namedtuple

from .router import (
    CLASS_DISPLAY_NAMES,
    COUNT_WORDS,
    INTENT_COUNT,
    INTENT_CUTTING,
    INTENT_ORGAN,
    INTENT_PROCEDURE,
    INTENT_PURPOSE,
    INTENT_SUTURE,
    INTENT_TASK,
    INTENT_TOOL_IDENTITY,
    INTENT_TOOL_PRESENCE,
    PROCEDURE_ANSWER,
    PURPOSE_DEFAULT,
    PURPOSES,
    TASK_DISPLAY,
    TASK_ORGANS,
)
from .taxonomy import TASK_CLASSES, TOOL_CLASSES

_TOOL_SET = frozenset(TOOL_CLASSES)
_TASK_SET = frozenset(TASK_CLASSES)

# --------------------------------------------------------------------------
# intents beyond the router's 11 -- see the module docstring for what each
# one teaches that the router structurally cannot answer
# --------------------------------------------------------------------------
INTENT_VARIANT_PRESENCE = "variant_presence_polar"
INTENT_TOOL_ABSENCE = "tool_absence_open"
INTENT_TASK_CONFIRMATION = "task_confirmation_polar"

#: The only two commercial-name families this corpus distinguishes for the
#: needle driver -- see src/surgvu/router.py::variant_qualifier's docstring
#: for the corpus counts (Large family 1281/62.7%, Mega family 759/37.2%
#: across all 155 cases). Not part of surgvu.taxonomy: it is a sub-class
#: distinction within ONE tool class, not a tool or task class itself, so it
#: gets its own tiny closed vocabulary rather than borrowing that module's.
NEEDLE_DRIVER_FAMILIES = ("large", "mega")


# --------------------------------------------------------------------------
# validators -- the whitelist. Every taxonomy-valued slot passes through one
# of these before it can reach a question or an answer.
# --------------------------------------------------------------------------

def _require_tool_class(value, slot_name="tool_class"):
    if value not in _TOOL_SET:
        raise ValueError(
            "%s=%r is not one of the %d surgvu.taxonomy.TOOL_CLASSES -- a "
            "raw logbook value must be normalised with "
            "surgvu.taxonomy.normalize_tool before it reaches this module"
            % (slot_name, value, len(TOOL_CLASSES)))
    return value


def _require_task_class(value, slot_name="task_class"):
    if value not in _TASK_SET:
        raise ValueError(
            "%s=%r is not one of the %d surgvu.taxonomy.TASK_CLASSES -- a "
            "raw logbook value must be normalised with "
            "surgvu.taxonomy.normalize_task before it reaches this module"
            % (slot_name, value, len(TASK_CLASSES)))
    return value


def _require_family(value, slot_name="family"):
    if value not in NEEDLE_DRIVER_FAMILIES:
        raise ValueError("%s=%r is not one of %r"
                         % (slot_name, value, NEEDLE_DRIVER_FAMILIES))
    return value


# --------------------------------------------------------------------------
# answer-form helpers -- each one guards its own taxonomy slot, so every
# answer_fn below that names a tool or task gets the whitelist for free by
# calling through here rather than reading the raw slot directly.
# --------------------------------------------------------------------------

def _tool_display(tool_class):
    """The exact string the router would emit for this class, with no
    commercial-variant guess. CLASS_DISPLAY_NAMES is the router's own
    fallback table (not str.title(), which mangles "prograsp forceps" into
    the wrong casing), so this is byte-identical to what a healthy serving
    run produces when it cannot bet on a specific variant.
    """
    _require_tool_class(tool_class)
    return CLASS_DISPLAY_NAMES[tool_class]


def _task_display(task_class):
    _require_task_class(task_class)
    return TASK_DISPLAY.get(task_class) or task_class.capitalize()


def _task_organ(task_class):
    _require_task_class(task_class)
    return TASK_ORGANS.get(task_class) or "Tissue"


def _bool_answer(value):
    return "Yes" if value else "No"


def _join_and(names):
    """"A" / "A and B" / "A, B and C" -- no Oxford comma, matching the
    corpus prose (see router.join_tool_names, which this mirrors but does
    not import: joining punctuation is not part of the answer VOCABULARY
    that must not drift, only the tool/task/purpose strings are).
    """
    names = list(names)
    if not names:
        raise ValueError("no names to join")
    if len(names) == 1:
        return names[0]
    return "%s and %s" % (", ".join(names[:-1]), names[-1])


def _count_word(count):
    if not isinstance(count, int) or isinstance(count, bool) \
            or not (0 <= count < len(COUNT_WORDS)):
        raise ValueError("count=%r is outside 0..%d"
                         % (count, len(COUNT_WORDS) - 1))
    return COUNT_WORDS[count]


# --------------------------------------------------------------------------
# templates
# --------------------------------------------------------------------------

QATemplate = namedtuple(
    "QATemplate", "intent question_template answer_fn example_slots")


def render(template, **slots):
    """(question, answer) for `template` filled with `slots`.

    `question_template` is either a plain str (filled with `.format(**slots)`)
    or a callable(**slots) -> str, for the few templates whose question needs
    a slot-derived phrase rather than a bare substitution. Either way, a bad
    slot is caught inside `answer_fn` -- and, for the callable question
    templates, inside the question builder too -- so an invalid value raises
    ValueError before it can become a question or an answer string.

    Never returns an empty question or an empty answer; raises instead of
    silently producing one, because a silently empty template would be
    exactly the kind of drift this module exists to prevent from being
    invisible.
    """
    question = (template.question_template(**slots)
                if callable(template.question_template)
                else template.question_template.format(**slots))
    answer = template.answer_fn(**slots)
    question = " ".join(str(question).split())
    answer = " ".join(str(answer).split())
    if not question:
        raise ValueError("template %r rendered an empty question"
                         % (template.intent,))
    if not answer:
        raise ValueError("template %r rendered an empty answer"
                         % (template.intent,))
    return question, answer


# -- tool_presence_polar: mirrors router.INTENT_TOOL_PRESENCE --------------

def _answer_tool_presence(tool_class, present):
    _tool_display(tool_class)  # validate only; the answer is Yes/No
    return _bool_answer(present)


def _answer_any_tool_present(any_present):
    return _bool_answer(any_present)


# -- tool_identity_open: mirrors router.INTENT_TOOL_IDENTITY ---------------

def _answer_tool_identity(tool_class):
    return _tool_display(tool_class)


def _answer_tool_identity_list(tool_classes):
    return _join_and([_tool_display(cls) for cls in tool_classes])


# -- task_open: mirrors router.INTENT_TASK ---------------------------------

def _answer_task(task_class):
    return _task_display(task_class)


# -- organ_open: mirrors router.INTENT_ORGAN -------------------------------

def _answer_organ(task_class):
    return _task_organ(task_class)


# -- purpose_open: mirrors router.INTENT_PURPOSE ---------------------------

def _answer_purpose(tool_class):
    _require_tool_class(tool_class)
    return PURPOSES.get(tool_class, PURPOSE_DEFAULT)


# -- procedure_open: mirrors router.INTENT_PROCEDURE -----------------------

def _answer_procedure():
    return PROCEDURE_ANSWER


# -- count_open: mirrors router.INTENT_COUNT -------------------------------

def _answer_count(count):
    return _count_word(count)


# -- cutting_polar: mirrors router.INTENT_CUTTING --------------------------

def _answer_cutting(cutting):
    return _bool_answer(cutting)


# -- suture_polar: mirrors router.INTENT_SUTURE ----------------------------

def _answer_suture(suturing):
    return _bool_answer(suturing)


# -- variant_presence_polar: BEYOND the router's 11 ------------------------

def _question_variant_presence(family, installed_family):
    _require_family(family)
    return "Was a %s needle driver used in this clip?" % family


def _answer_variant_presence(family, installed_family):
    _require_family(family)
    _require_family(installed_family)
    return _bool_answer(installed_family == family)


# -- tool_absence_open: BEYOND the router's 11 -----------------------------

def _answer_tool_absence(absent_class):
    return _tool_display(absent_class)


# -- task_confirmation_polar: BEYOND the router's 11 -----------------------

def _question_task_confirmation(asked_class, actual_class):
    _require_task_class(asked_class)
    return "Is the surgeon currently performing %s?" % _task_display(
        asked_class).lower()


def _answer_task_confirmation(asked_class, actual_class):
    _require_task_class(asked_class)
    _require_task_class(actual_class)
    return _bool_answer(asked_class == actual_class)


QA_TEMPLATES = (
    QATemplate(
        intent=INTENT_TOOL_PRESENCE,
        question_template="Is a {tool_class} being used in this clip?",
        answer_fn=_answer_tool_presence,
        example_slots={"tool_class": "needle driver", "present": True},
    ),
    QATemplate(
        intent=INTENT_TOOL_PRESENCE,
        question_template="Are any surgical instruments visible in this clip?",
        answer_fn=_answer_any_tool_present,
        example_slots={"any_present": True},
    ),
    QATemplate(
        intent=INTENT_TOOL_IDENTITY,
        question_template="What type of instrument is being used in this clip?",
        answer_fn=_answer_tool_identity,
        example_slots={"tool_class": "cadiere forceps"},
    ),
    QATemplate(
        intent=INTENT_TOOL_IDENTITY,
        question_template="Which instruments are visible in this clip?",
        answer_fn=_answer_tool_identity_list,
        example_slots={"tool_classes": ("cadiere forceps", "needle driver")},
    ),
    QATemplate(
        intent=INTENT_TASK,
        question_template="What task is being performed in this clip?",
        answer_fn=_answer_task,
        example_slots={"task_class": "suturing"},
    ),
    QATemplate(
        intent=INTENT_ORGAN,
        question_template="What organ is being manipulated in this clip?",
        answer_fn=_answer_organ,
        example_slots={"task_class": "uterine horn"},
    ),
    QATemplate(
        intent=INTENT_PURPOSE,
        question_template="What is the purpose of using {tool_class} in this procedure?",
        answer_fn=_answer_purpose,
        example_slots={"tool_class": "prograsp forceps"},
    ),
    QATemplate(
        intent=INTENT_PROCEDURE,
        question_template="What procedure is this clip showing?",
        answer_fn=_answer_procedure,
        example_slots={},
    ),
    QATemplate(
        intent=INTENT_COUNT,
        question_template="How many instruments are visible in this clip?",
        answer_fn=_answer_count,
        example_slots={"count": 2},
    ),
    QATemplate(
        intent=INTENT_CUTTING,
        question_template="Is tissue being cut in this clip?",
        answer_fn=_answer_cutting,
        example_slots={"cutting": True},
    ),
    QATemplate(
        intent=INTENT_SUTURE,
        question_template="Is suturing being performed in this clip?",
        answer_fn=_answer_suture,
        example_slots={"suturing": True},
    ),
    QATemplate(
        intent=INTENT_VARIANT_PRESENCE,
        question_template=_question_variant_presence,
        answer_fn=_answer_variant_presence,
        example_slots={"family": "large", "installed_family": "large"},
    ),
    QATemplate(
        intent=INTENT_TOOL_ABSENCE,
        question_template="Which instrument class is not in use during this clip?",
        answer_fn=_answer_tool_absence,
        example_slots={"absent_class": "stapler"},
    ),
    QATemplate(
        intent=INTENT_TASK_CONFIRMATION,
        question_template=_question_task_confirmation,
        answer_fn=_answer_task_confirmation,
        example_slots={"asked_class": "suturing", "actual_class": "suturing"},
    ),
)
