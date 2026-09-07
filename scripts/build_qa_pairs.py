"""Logbook -> QA pairs, in the router's own answer forms (Task 2 of the v5
plan3 VLM training pipeline; see
docs/design/plans/2026-08-25-v5-plan3-vlm-training.md).

WHAT THIS DOES. `tools.csv` and `tasks.csv` already say, for every second of
every case, which of the 12 test-relevant tool classes are installed and
which of the 8 task classes is underway. That is free, exactly-on-taxonomy
supervision. This script windows each case into the same 30-second clips the
test-time pipeline decodes (`surgvu.sampling.enumerate_windows` -- reused,
not reimplemented), derives a question/answer pair per applicable intent from
each window, and writes one JSON record per example to `qa_pairs.jsonl`.

Every answer is produced by calling an `answer_fn` taken directly from
`surgvu.qa_forms.QA_TEMPLATES` -- never by building an answer string in this
file. That is what makes the taxonomy whitelist in qa_forms unconditional:
`_require_tool_class`/`_require_task_class` raise before a malformed value
can reach a rendered answer, and there is no code path here that bypasses it.

THE THREE RULINGS THIS FILE EXISTS TO HONOUR (repeated because each already
cost a contaminated run or a silent bug elsewhere in this project):

  R30 -- split discipline. `config/splits_v2.json`'s `heldout` list holds the
  11 publicly-graded sample cases. Excluding them is done through
  `surgvu.sampling.normalize_case_id` on BOTH sides, never raw string
  equality: the corpus spells a case `case_122` and a heldout id may be
  spelled `case122`, and a raw membership test between the two reports zero
  overlap while excluding nothing. `select_cases` below raises if the
  exclusion removes zero cases, because a silent zero-exclusion IS the bug --
  it is exactly how the variant head got trained on the graded cases.

  R14 -- time format. `tools.csv` times are `HH:MM:SS.ffffff` strings;
  `tasks.csv` times are float seconds. Both are parsed by
  `surgvu.labels.CaseLabels` (via `parse_hms`), reused here rather than
  hand-rolled a second time.

  R28 -- the part. Timestamps reset at video-part boundaries and 126 of 155
  cases have more than one part, so every record below carries `part`.

LABEL-VOCABULARY HAZARDS (docs/design/notes/2026-08-24-label-vocab-
hazards.md). `surgvu.taxonomy.normalize_tool`/`normalize_task` already strip,
lower-case, and whitelist every raw value against the 12/8-class taxonomy --
that is the single source of truth for which rows this script's data (via
`CaseLabels`) actually uses. `scan_logbook_hazards` below is a SEPARATE,
read-only tally over the raw CSV rows, for REPORTING only: it exists because
normalize_tool/normalize_task collapse several distinct raw-value problems
(the endoscope's `nan(camera in)` row, an empty string, a malformed quoted
row, a name outside the 12/8-class taxonomy) into one `None`, and a training
run must be able to see those reasons separately before trusting the totals.

TASK 3 -- SAMPLING AND FRAME EXTRACTION (--extract-frames). 377,557 records
is far more than a LoRA fine-tune needs and far more than is sensible to
decode (16 frames each would be six million frames). `sample_corpus` below
is the size-control policy: case-stratified so no handful of cases dominates
what the model learns, and -- for `tool_presence_polar` specifically, the
intent the controller measured at 71.3% Yes / 28.7% No across 66,144
records, against a graded distribution of about 57/43 -- answer-balanced
toward parity, because training at 71/29 induces a Yes prior and a Yes bias
is ALREADY case132's error (gold No, answered Yes). Both corrections reuse
ONE primitive, `water_fill_allocate`: stratifying by case and correcting the
Yes/No skew are the same problem (spread a budget evenly across groups,
never proportionally to how many each group happens to have) applied to two
different partitions of the same pool.

Frame extraction then decodes each DISTINCT (case, part, t_start, t_stop)
window in the sampled set exactly once via `surgvu.perceive.decode_clip_
multiscale`'s `index_range` (ruling R15: seek directly in the source file,
no temporary clip, no re-encode) -- never `scripts/build_qa_pairs.py` code
that bypasses `preprocess.prepare_frame`'s crop-and-blur, which stays in
force here exactly as it does at serving. Many QA records share one window
(tool presence, task, organ, count, ... are all asked about the SAME 30 s
clip), so decoding by window rather than by record is both cheaper and the
only sane semantics: two intents about one clip must see the same frames.
Every frame that could not be produced is tallied by REASON (missing video,
unreadable video, the record's own timestamps falling outside the video
that was actually decoded, or a short/failed decode) and attributed to every
QA record that shared the dropped window, not just "one" per window -- a
generator yielding 80% of records is indistinguishable from one yielding
100% until the model is wrong, and undercounting a multi-record window would
be exactly that kind of silent loss.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.extract import (  # noqa: E402
    JPEG_QUALITY, looks_like_a_written_jpeg,
)
from surgvu.labels import CaseLabels, normalize_part  # noqa: E402
from surgvu.qa_forms import (  # noqa: E402
    INTENT_COUNT, INTENT_CUTTING, INTENT_ORGAN, INTENT_PROCEDURE,
    INTENT_PURPOSE, INTENT_SUTURE, INTENT_TASK, INTENT_TASK_CONFIRMATION,
    INTENT_TOOL_ABSENCE, INTENT_TOOL_IDENTITY, INTENT_TOOL_PRESENCE,
    INTENT_VARIANT_PRESENCE, NEEDLE_DRIVER_FAMILIES, QA_TEMPLATES,
    QATemplate, TASK_DISPLAY, render,
)
from surgvu.router import (  # noqa: E402
    COUNT_DIVIDING_AS_CUTTING, CUTTING_TOOLS, DIVIDING_TOOLS,
    SUTURING_TASKS, SUTURING_TOOLS,
)
from surgvu.sampling import (  # noqa: E402
    enumerate_windows, normalize_case_id,
)
from surgvu.taxonomy import (  # noqa: E402
    TASK_CLASSES, TOOL_CLASSES, normalize_task, normalize_tool,
)
from surgvu.vlm import DEFAULT_FRAMES as VLM_DEFAULT_FRAMES  # noqa: E402

_CUT_SET = CUTTING_TOOLS | (DIVIDING_TOOLS if COUNT_DIVIDING_AS_CUTTING else frozenset())

# --------------------------------------------------------------------------
# hazard tallying -- reporting only, see module docstring
# --------------------------------------------------------------------------


def scan_logbook_hazards(tools_csv, tasks_csv):
    """Row-level tally of RAW `groundtruth_toolname`/`groundtruth_taskname`
    values, split out by the specific reason a row is not a clean, on-
    taxonomy value. Returns (tool_reasons, task_reasons), both
    collections.Counter.

    This is independent of, and does not feed, what `CaseLabels` actually
    loads -- `surgvu.taxonomy.normalize_tool`/`normalize_task` remain the one
    accept/reject decision. This function only explains, after the fact, WHY
    normalize_* returned None or accepted a messy raw string, so a drop count
    can be checked against docs/design/notes/2026-08-24-label-vocab-
    hazards.md before a training run.
    """
    tool_reasons = Counter()
    with open(tools_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("groundtruth_toolname") or ""
            text = raw.strip()
            if text == "":
                tool_reasons["empty"] += 1
                continue
            if text.lower().startswith("nan"):
                tool_reasons["nan_camera"] += 1
                continue
            if normalize_tool(raw) is None:
                # one of the 7 out-of-scope rare classes, or a malformed row
                # like the stray-quote ' Single Site"' -- either way, not one
                # of the 12 test-relevant classes.
                tool_reasons["out_of_scope_or_malformed"] += 1
                continue
            tool_reasons["kept"] += 1
            if text != raw or text != text.lower():
                # e.g. "clip applier " (trailing space) survives only
                # because normalize_tool strips+lowers before whitelisting.
                tool_reasons["kept_after_normalisation"] += 1

    task_reasons = Counter()
    with open(tasks_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("groundtruth_taskname") or ""
            text = raw.strip()
            if text == "":
                task_reasons["empty"] += 1
                continue
            if normalize_task(raw) is None:
                task_reasons["unrecognised_or_malformed"] += 1
                continue
            task_reasons["kept"] += 1
            if text != text.lower():
                # e.g. "Suturing" collapses onto "suturing" -- both raw
                # spellings appear in the real corpus (775 vs 609).
                task_reasons["case_normalised"] += 1

    return tool_reasons, task_reasons


# --------------------------------------------------------------------------
# split discipline (ruling R30)
# --------------------------------------------------------------------------


def load_heldout(splits_path):
    """Normalised set of heldout case ids from config/splits_v2.json."""
    data = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    heldout = data.get("heldout")
    if not heldout:
        raise ValueError(
            "%s has no non-empty 'heldout' list -- cannot verify the split "
            "excludes the 11 graded cases" % splits_path)
    return {normalize_case_id(c) for c in heldout}


def select_cases(labels_root, heldout_norm):
    """(eligible, excluded) case directory names under labels_root.

    Comparison is via normalize_case_id on BOTH sides -- never raw string
    equality (ruling R30): `case122` and `case_122` name the same case but
    are never equal as strings, so a raw `name in heldout_ids` test can
    report "excluded 0" while every heldout case sits inside `eligible`.
    Fails loudly in exactly that situation: a zero-case exclusion here is
    indistinguishable from a working one until the graded cases turn up
    inside the training corpus, which is the bug that produced the
    contaminated 0.9011 run.
    """
    all_cases = sorted(p.name for p in Path(labels_root).iterdir() if p.is_dir())
    if not all_cases:
        raise ValueError("no case directories found under %s" % labels_root)
    excluded = [c for c in all_cases if normalize_case_id(c) in heldout_norm]
    eligible = [c for c in all_cases if normalize_case_id(c) not in heldout_norm]
    if not excluded:
        raise RuntimeError(
            "heldout exclusion removed ZERO of %d case(s) under %s. This is "
            "ruling R30's bug: a silent zero-exclusion means the graded "
            "cases are still in the corpus and any model trained on it is "
            "contaminated. Refusing to generate." % (len(all_cases), labels_root))
    return eligible, excluded


# --------------------------------------------------------------------------
# deterministic, hash-based pseudo-choice -- no random.Random state to seed
# or thread through, so the same (case, part, start) always yields the same
# paraphrase and the same present/absent-tool or Yes/No choice, run to run.
# --------------------------------------------------------------------------


def _stable_index(key, n):
    if n <= 0:
        raise ValueError("cannot choose an index from %d options" % n)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n


def _stable_choice(options, key):
    options = list(options)
    return options[_stable_index(key, len(options))]


def _stable_bool(key):
    return _stable_index(key, 2) == 0


def _window_key(w, shape):
    return "%s|%s|%.3f|%s" % (w.case, w.part, w.start, shape)


# --------------------------------------------------------------------------
# QA_TEMPLATES shapes -- resolved by slot signature, not list position, so a
# reordering of QA_TEMPLATES cannot silently swap which shape this script
# thinks it is rendering.
# --------------------------------------------------------------------------


def _find_template(intent, has_slot=None, lacks_slot=None):
    matches = [t for t in QA_TEMPLATES if t.intent == intent
               and (has_slot is None or has_slot in t.example_slots)
               and (lacks_slot is None or lacks_slot not in t.example_slots)]
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one QA_TEMPLATES entry for intent=%r "
            "has_slot=%r lacks_slot=%r; found %d -- qa_forms.QA_TEMPLATES "
            "changed shape under this script" % (intent, has_slot, lacks_slot,
                                                  len(matches)))
    return matches[0]


TPL_TOOL_PRESENCE_SPECIFIC = _find_template(INTENT_TOOL_PRESENCE, has_slot="tool_class")
TPL_TOOL_PRESENCE_ANY = _find_template(INTENT_TOOL_PRESENCE, has_slot="any_present")
TPL_TOOL_IDENTITY_SINGLE = _find_template(INTENT_TOOL_IDENTITY, has_slot="tool_class")
TPL_TOOL_IDENTITY_LIST = _find_template(INTENT_TOOL_IDENTITY, has_slot="tool_classes")
TPL_TASK = _find_template(INTENT_TASK)
TPL_ORGAN = _find_template(INTENT_ORGAN)
TPL_PURPOSE = _find_template(INTENT_PURPOSE)
TPL_PROCEDURE = _find_template(INTENT_PROCEDURE)
TPL_COUNT = _find_template(INTENT_COUNT)
TPL_CUTTING = _find_template(INTENT_CUTTING)
TPL_SUTURE = _find_template(INTENT_SUTURE)
TPL_VARIANT = _find_template(INTENT_VARIANT_PRESENCE)
TPL_ABSENCE = _find_template(INTENT_TOOL_ABSENCE)
TPL_CONFIRM = _find_template(INTENT_TASK_CONFIRMATION)


def _lower_task(task_class):
    return (TASK_DISPLAY.get(task_class) or task_class.capitalize()).lower()


# --------------------------------------------------------------------------
# paraphrases -- register grounded in the 11 public sample questions (see
# scripts/score_sample.py::load_sample_cases against
# /staging/groups/bhaskar_opscribe/surgvu/cat2_sample), e.g. "Are there
# forceps being used here?", "Is a large needle driver among the listed
# tools?", "What is the purpose of using forceps in this procedure?". Every
# shape below has more than one phrasing so the model learns the TASK, not
# one fixed sentence.
# --------------------------------------------------------------------------

PARAPHRASES = {
    "tool_presence_specific": (
        "Is a {tool_class} being used in this clip?",
        "Is a {tool_class} involved in the procedure?",
        "Are there any {tool_class} being used here?",
        "Does this clip show a {tool_class}?",
        "Was a {tool_class} used in this segment?",
    ),
    "tool_presence_any": (
        "Are any surgical instruments visible in this clip?",
        "Is any instrument being used here?",
        "Does this clip show any surgical instruments?",
        "Are there any tools in use during this segment?",
    ),
    "tool_identity_single": (
        "What type of instrument is being used in this clip?",
        "What instrument is shown in this clip?",
        "Which instrument is being used here?",
        "What kind of tool is in use during this segment?",
    ),
    "tool_identity_list": (
        "Which instruments are visible in this clip?",
        "What instruments are being used in this clip?",
        "Which tools appear in this segment?",
        "What instruments does this clip show?",
    ),
    "task": (
        "What task is being performed in this clip?",
        "What surgical task is underway here?",
        "Which task is being carried out in this segment?",
        "What is the surgeon doing in this clip?",
    ),
    "organ": (
        "What organ is being manipulated in this clip?",
        "Which organ is being handled here?",
        "What anatomical structure is being manipulated in this segment?",
    ),
    "purpose": (
        "What is the purpose of using {tool_class} in this procedure?",
        "Why is {tool_class} being used in this procedure?",
        "What is {tool_class} used for during the surgery?",
    ),
    "procedure": (
        "What procedure is this clip showing?",
        "What procedure is this summary describing?",
        "What type of procedure is being performed?",
        "What surgery does this clip depict?",
    ),
    "count": (
        "How many instruments are visible in this clip?",
        "How many surgical tools are in use here?",
        "How many instruments does this clip show?",
    ),
    "cutting": (
        "Is tissue being cut in this clip?",
        "Is tissue being cut during this clip?",
        "Is dissection occurring in this segment?",
        "Is the surgeon cutting tissue here?",
    ),
    "suture": (
        "Is suturing being performed in this clip?",
        "Is a suture required in this surgical step?",
        "Is the surgeon suturing in this segment?",
        "Does this clip show suturing?",
    ),
    "variant_presence": (
        lambda family, installed_family: "Was a %s needle driver used in this clip?" % family,
        lambda family, installed_family: "Is a %s needle driver among the listed tools?" % family,
        lambda family, installed_family: "Was a %s needle driver used during the surgery?" % family,
        lambda family, installed_family: "Does this clip show a %s needle driver?" % family,
    ),
    "tool_absence": (
        "Which instrument class is not in use during this clip?",
        "Which type of instrument is absent from this clip?",
        "What instrument class does not appear in this segment?",
    ),
    "task_confirmation": (
        lambda asked_class, actual_class: "Is the surgeon currently performing %s?" % _lower_task(asked_class),
        lambda asked_class, actual_class: "Is %s taking place in this clip?" % _lower_task(asked_class),
        lambda asked_class, actual_class: "Is this clip showing %s?" % _lower_task(asked_class),
    ),
}

#: intent -> total number of distinct phrasing forms available across every
#: shape that emits that intent (two shapes can share an intent, e.g.
#: INTENT_TOOL_PRESENCE covers both "tool_presence_specific" and
#: "tool_presence_any").
_SHAPE_INTENT = {
    "tool_presence_specific": INTENT_TOOL_PRESENCE,
    "tool_presence_any": INTENT_TOOL_PRESENCE,
    "tool_identity_single": INTENT_TOOL_IDENTITY,
    "tool_identity_list": INTENT_TOOL_IDENTITY,
    "task": INTENT_TASK,
    "organ": INTENT_ORGAN,
    "purpose": INTENT_PURPOSE,
    "procedure": INTENT_PROCEDURE,
    "count": INTENT_COUNT,
    "cutting": INTENT_CUTTING,
    "suture": INTENT_SUTURE,
    "variant_presence": INTENT_VARIANT_PRESENCE,
    "tool_absence": INTENT_TOOL_ABSENCE,
    "task_confirmation": INTENT_TASK_CONFIRMATION,
}


def paraphrase_counts_by_intent():
    counts = Counter()
    for shape, intent in _SHAPE_INTENT.items():
        counts[intent] += len(PARAPHRASES[shape])
    return dict(counts)


def _paraphrase(template, question_form):
    return QATemplate(intent=template.intent, question_template=question_form,
                       answer_fn=template.answer_fn, example_slots=template.example_slots)


def _render_shape(shape, template, key, slots):
    forms = PARAPHRASES[shape]
    idx = _stable_index(key, len(forms))
    question, answer = render(_paraphrase(template, forms[idx]), **slots)
    return question, answer, idx


# --------------------------------------------------------------------------
# variant (needle-driver family) resolution -- config/variant_labels.json,
# built by scripts/build_variant_labels.py from tools.csv's
# commercial_toolname column. Reused rather than re-derived: it already
# solves "which Large/Mega interval covers this window" correctly, including
# the part-boundary handling (ruling R28) and the unrecognised-name drop.
# --------------------------------------------------------------------------


def load_variant_labels(path):
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("cases", {})


def resolve_variant_family(variant_intervals, part, t_start, t_stop):
    """The single needle-driver family whose install interval fully covers
    [t_start, t_stop] on `part`, or None if zero or more-than-one family
    covers the window (missing data, or an ambiguous overlap we should not
    guess through).
    """
    part = normalize_part(part)
    covering = {iv["family"] for iv in variant_intervals
                if normalize_part(iv["part"]) == part
                and iv["start"] <= t_start and t_stop <= iv["stop"]}
    if len(covering) == 1:
        return next(iter(covering))
    return None


# --------------------------------------------------------------------------
# per-window example generation
# --------------------------------------------------------------------------


def generate_examples_for_window(w, variant_intervals, drops):
    """List of (shape, template, slots, provenance_extra) for one window.

    `drops` (a Counter) is tallied whenever an intent is structurally
    inapplicable to this window (e.g. tool_identity_single needs exactly one
    tool; variant_presence needs a resolvable needle-driver family) -- these
    are not data-quality drops, they are "this question does not apply
    here", and they are tallied under a distinct set of reasons so the two
    kinds are never confused in the report.
    """
    t_stop = w.start + w.length
    present = tuple(c for c in TOOL_CLASSES if c in w.tools)
    absent = tuple(c for c in TOOL_CLASSES if c not in w.tools)
    out = []

    # -- tool_presence_specific: ask about a present tool half the time, an
    # absent one the other half, so the corpus is not overwhelmingly "Yes".
    key = _window_key(w, "tool_presence_specific")
    if present and (not absent or _stable_bool(key)):
        tool_class = _stable_choice(present, key + "|pick")
        slots = {"tool_class": tool_class, "present": True}
        out.append(("tool_presence_specific", TPL_TOOL_PRESENCE_SPECIFIC, key, slots))
    elif absent:
        tool_class = _stable_choice(absent, key + "|pick")
        slots = {"tool_class": tool_class, "present": False}
        out.append(("tool_presence_specific", TPL_TOOL_PRESENCE_SPECIFIC, key, slots))
    else:
        drops["tool_presence_specific_impossible"] += 1

    # -- tool_presence_any: always applicable.
    key = _window_key(w, "tool_presence_any")
    out.append(("tool_presence_any", TPL_TOOL_PRESENCE_ANY, key,
                {"any_present": bool(w.tools)}))

    # -- tool_identity_single: only when exactly one tool is installed.
    if len(present) == 1:
        key = _window_key(w, "tool_identity_single")
        out.append(("tool_identity_single", TPL_TOOL_IDENTITY_SINGLE, key,
                    {"tool_class": present[0]}))
    else:
        drops["tool_identity_single_not_applicable"] += 1

    # -- tool_identity_list: whenever at least one tool is installed.
    if present:
        key = _window_key(w, "tool_identity_list")
        out.append(("tool_identity_list", TPL_TOOL_IDENTITY_LIST, key,
                    {"tool_classes": present}))
    else:
        drops["tool_identity_list_not_applicable"] += 1

    # -- task_open / organ_open: always applicable (a window always lies
    # inside a task segment -- see enumerate_windows).
    key = _window_key(w, "task")
    out.append(("task", TPL_TASK, key, {"task_class": w.task}))
    key = _window_key(w, "organ")
    out.append(("organ", TPL_ORGAN, key, {"task_class": w.task}))

    # -- purpose_open: needs a tool to ask the purpose of.
    if present:
        key = _window_key(w, "purpose")
        tool_class = _stable_choice(present, key + "|pick")
        out.append(("purpose", TPL_PURPOSE, key, {"tool_class": tool_class}))
    else:
        drops["purpose_not_applicable"] += 1

    # -- procedure_open: the answer is a corpus-wide constant, so it is
    # deliberately sub-sampled (~1 in 25 windows) rather than emitted for
    # every one -- otherwise this single intent would dwarf every other
    # answer in the corpus with one repeated string.
    key = _window_key(w, "procedure")
    if _stable_index(key, 25) == 0:
        out.append(("procedure", TPL_PROCEDURE, key, {}))

    # -- count_open: always applicable.
    key = _window_key(w, "count")
    out.append(("count", TPL_COUNT, key, {"count": len(present)}))

    # -- cutting_polar / suture_polar: derived from the router's own
    # definitional tool/task sets, so the training data agrees with the
    # router's stance rather than inventing a second one.
    key = _window_key(w, "cutting")
    out.append(("cutting", TPL_CUTTING, key, {"cutting": bool(w.tools & _CUT_SET)}))
    key = _window_key(w, "suture")
    suturing = (w.task in SUTURING_TASKS) or bool(w.tools & SUTURING_TOOLS)
    out.append(("suture", TPL_SUTURE, key, {"suturing": suturing}))

    # -- variant_presence_polar: only when a needle driver is installed AND
    # config/variant_labels.json resolves exactly one family for this
    # window. Ask about the true family half the time (Yes) and the other
    # family the other half (No).
    if "needle driver" in w.tools:
        actual_family = resolve_variant_family(variant_intervals, w.part, w.start, t_stop)
        if actual_family is not None:
            key = _window_key(w, "variant_presence")
            if _stable_bool(key):
                asked_family = actual_family
            else:
                asked_family = next(f for f in NEEDLE_DRIVER_FAMILIES if f != actual_family)
            out.append(("variant_presence", TPL_VARIANT, key,
                        {"family": asked_family, "installed_family": actual_family}))
        else:
            drops["variant_family_unresolved"] += 1

    # -- tool_absence_open: always applicable (12-class taxonomy, a window
    # essentially never has all 12 installed at once).
    if absent:
        key = _window_key(w, "tool_absence")
        # DETERMINISTIC, NOT _stable_choice -- AND THAT IS THE WHOLE POINT.
        #
        # This used to be `_stable_choice(absent, key + "|pick")`: a hash of the
        # window key, picking one of the ~8 absent classes arbitrarily. Measured
        # on the v1 corpus, the resulting gold distribution was almost exactly
        # uniform across all twelve classes (48/47/46/43/42/42/33/29/21/20/19/15
        # in val), i.e. the target carried no information about the window at
        # all. That made 1,595 train and 405 val records -- 8.6% of each -- a
        # RANDOM MAPPING from frames to answer.
        #
        # A model cannot learn a random mapping, but it will happily spend
        # capacity trying, and the damage is not confined to these records: it
        # is 8.6% of the gradient signal teaching that the answer is unrelated
        # to what is in the frames. The ceiling here was ~1/12 for anything, and
        # five different preference orderings measured between 0.057 and 0.123
        # exact -- all of them noise.
        #
        # Picking the first absent class in a FIXED order makes the target a
        # function of the window's PRESENT set, which is exactly the skill worth
        # learning: to answer, the model must identify every instrument in view
        # and name one that is not there. `absent` is already built by filtering
        # TOOL_CLASSES in order, so `absent[0]` is that first-in-fixed-order
        # class and no separate ordering constant is needed.
        #
        # The arbitrariness does not disappear -- it moves into the REFERENCES.
        # Every absent class is an equally correct answer to "which instrument
        # is not in use", so build_records_for_case attaches all of them as
        # alternate references and the scorer takes the MAX, the same way the
        # official harness scores against five human references. The single
        # `answer` is the training target; the reference list is the honest
        # grading key.
        absent_class = absent[0]
        out.append(("tool_absence", TPL_ABSENCE, key, {"absent_class": absent_class}))
    else:
        drops["tool_absence_impossible"] += 1

    # -- task_confirmation_polar: always applicable; ask about the true task
    # half the time (Yes) and a different task the other half (No).
    key = _window_key(w, "task_confirmation")
    if _stable_bool(key):
        asked_class = w.task
    else:
        others = tuple(t for t in TASK_CLASSES if t != w.task)
        asked_class = _stable_choice(others, key + "|pick")
    out.append(("task_confirmation", TPL_CONFIRM, key,
               {"asked_class": asked_class, "actual_class": w.task}))

    return out


def build_records_for_case(case_id, labels, variant_intervals, drops, paraphrase_usage):
    """List of qa_pairs.jsonl records for one case."""
    records = []
    windows = enumerate_windows(case_id, labels)
    for w in windows:
        t_stop = w.start + w.length
        for shape, template, key, slots in generate_examples_for_window(
                w, variant_intervals, drops):
            question, answer, idx = _render_shape(shape, template, key, slots)
            paraphrase_usage[(shape, idx)] += 1
            record = {
                "case": case_id,
                "part": w.part,
                "t_start": w.start,
                "t_stop": t_stop,
                "question": question,
                "answer": answer,
                "intent": template.intent,
                "provenance": {
                    "generator": "scripts/build_qa_pairs.py",
                    "shape": shape,
                    "paraphrase_index": idx,
                    "window_task": w.task,
                    "window_description": (w.description or "")[:200],
                },
            }
            records.append(record)
    return records


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def summarize(records):
    per_intent = Counter(r["intent"] for r in records)
    per_intent_answer = {}
    for r in records:
        bucket = per_intent_answer.setdefault(r["intent"], Counter())
        bucket[r["answer"]] += 1
    return per_intent, per_intent_answer


def _print_report(eligible, excluded, tool_hazards, task_hazards, gen_drops,
                  records, paraphrase_usage):
    print("cases scanned: %d   eligible: %d   excluded (heldout): %d"
          % (len(eligible) + len(excluded), len(eligible), len(excluded)))
    print("excluded case ids: %s" % ", ".join(sorted(excluded)))
    print()
    print("groundtruth_toolname raw-row hazards: %s" % dict(sorted(tool_hazards.items())))
    print("groundtruth_taskname raw-row hazards: %s" % dict(sorted(task_hazards.items())))
    print()
    print("generation-skip counts (question not applicable to a window): %s"
          % dict(sorted(gen_drops.items())))
    print()
    print("total QA records: %d" % len(records))
    per_intent, per_intent_answer = summarize(records)
    print("per-intent totals:")
    for intent, count in sorted(per_intent.items()):
        print("  %-28s %6d" % (intent, count))
    print()
    print("per-intent answer balance:")
    for intent in sorted(per_intent_answer):
        answers = per_intent_answer[intent]
        total = sum(answers.values())
        top = answers.most_common(5)
        rendered = ", ".join("%s=%d (%.1f%%)" % (a, n, 100.0 * n / total) for a, n in top)
        print("  %-28s n=%-6d %s" % (intent, total, rendered))
    print()
    print("paraphrase forms available per intent: %s" % paraphrase_counts_by_intent())
    used_by_shape = Counter()
    for (shape, _idx), n in paraphrase_usage.items():
        used_by_shape[shape] += 0  # ensure shape present even if idx-specific
    distinct_used = Counter()
    for (shape, idx), n in paraphrase_usage.items():
        distinct_used[shape] += 1
    print("distinct phrasings actually used per shape (of those available): %s"
          % {shape: "%d/%d" % (distinct_used[shape], len(PARAPHRASES[shape]))
             for shape in sorted(PARAPHRASES)})


# ============================================================================
# TASK 3 -- sampling policy
#
# ONE primitive, `water_fill_allocate`, used twice: once to spread a budget
# evenly across CASES (the "no handful of cases should dominate" requirement)
# and once to spread it evenly across ANSWER VALUES (the tool_presence_polar
# Yes/No parity fix). Both are "distribute a target evenly across groups,
# capped by each group's own size" -- proportional allocation would simply
# reproduce whatever skew the input already has, which is exactly the bug
# in both cases (a few cases supplying most records; Yes supplying 71%).
# ============================================================================


def water_fill_allocate(capacities, target, rng=None):
    """{key: count} summing to `min(target, sum(capacities.values()))`,
    distributed as evenly as possible across `capacities`'s keys.

    Classic water-filling: every still-open key gets an equal share of what
    remains; any key whose OWN capacity is smaller than that share is
    saturated (given all of it) and removed from the pool; repeat with the
    reduced target and the reduced pool. A key can never receive more than
    its own capacity -- that is what makes this safe to call with
    `capacities` built directly from `len(group)` for a real, uneven corpus.

    `rng`, if given, only decides which key(s) get the final +1 when the
    remainder is smaller than the number of open keys (e.g. distributing 4
    leftover units across 10 equally-open keys) -- picking a random subset
    rather than always the same first N keys by iteration order. Without
    `rng` that subset is the sorted key order, which is deterministic but
    arbitrary; callers that care about not always favouring the
    alphabetically-first keys should pass a seeded `random.Random`.
    """
    if target < 0:
        raise ValueError("target must be >= 0, got %r" % (target,))
    open_caps = {k: v for k, v in capacities.items() if v > 0}
    alloc = {k: 0 for k in capacities}
    remaining = min(target, sum(open_caps.values()))
    while remaining > 0 and open_caps:
        n = len(open_caps)
        share, extra = divmod(remaining, n)
        if share == 0:
            keys = sorted(open_caps) if rng is None else list(open_caps)
            if rng is not None:
                rng.shuffle(keys)
            for k in keys[:extra]:
                alloc[k] += 1
                open_caps[k] -= 1
                if open_caps[k] == 0:
                    del open_caps[k]
            break
        saturating = {k: c for k, c in open_caps.items() if c <= share}
        if saturating:
            for k, c in saturating.items():
                alloc[k] += c
                remaining -= c
                del open_caps[k]
            continue
        for k in open_caps:
            open_caps[k] -= share
            alloc[k] += share
        remaining -= share * n
    return alloc


def stratified_sample(records, target, rng, key_fn):
    """Up to `target` records from `records`, allocated as evenly as
    possible across the groups `key_fn` partitions them into
    (`water_fill_allocate`), then drawn WITHOUT replacement (`rng.sample`)
    inside each group. Deterministic for a seeded `rng`. Returns fewer than
    `target` only if `records` itself has fewer than `target` in total.
    """
    groups = defaultdict(list)
    for r in records:
        groups[key_fn(r)].append(r)
    alloc = water_fill_allocate({k: len(v) for k, v in groups.items()}, target, rng)
    out = []
    for k, want in alloc.items():
        if want:
            out.extend(rng.sample(groups[k], want))
    return out


#: Intents whose ANSWER distribution is corrected toward parity while
#: sampling, rather than left at its natural (logbook-driven) skew.
#: `INTENT_TOOL_PRESENCE` ("tool_presence_polar") is the one the controller
#: measured: 71.3% Yes / 28.7% No across 66,144 records (the largest intent,
#: 17.5% of the corpus) against a graded polar rate of about 57/43. Every
#: OTHER intent is left alone deliberately -- e.g. `procedure_open`'s answer
#: is a corpus-wide constant, and forcing "balance" on a single-valued
#: distribution is meaningless. Extend this tuple, not the branch condition
#: below, if another intent is later found to need the same correction.
BALANCED_INTENTS = (INTENT_TOOL_PRESENCE,)


def sample_intent(records_for_intent, target, rng, balance_by_answer=False):
    """Case-stratified sample of one intent's records, up to `target`.

    `balance_by_answer=True` nests a SECOND water-filling pass in front of
    the case-stratification: the target is first split evenly across the
    intent's distinct `answer` values (parity, via `water_fill_allocate`),
    and only THEN is each answer-value bucket sampled case-stratified. That
    ordering matters -- balancing the answer ratio must not undo the
    per-case spread, so each bucket goes back through `stratified_sample`
    rather than a single flat `rng.sample`.
    """
    if not balance_by_answer:
        return stratified_sample(records_for_intent, target, rng, key_fn=lambda r: r["case"])
    by_answer = defaultdict(list)
    for r in records_for_intent:
        by_answer[r["answer"]].append(r)
    answer_alloc = water_fill_allocate(
        {a: len(rs) for a, rs in by_answer.items()}, target, rng)
    out = []
    for answer, want in answer_alloc.items():
        out.extend(stratified_sample(by_answer[answer], want, rng, key_fn=lambda r: r["case"]))
    return out


#: A few thousand per intent is the right order for LoRA on a 7B model: a
#: LoRA adapter trained on this qa_pairs corpus needs enough examples per
#: intent to cover the taxonomy's combinations (12 tool classes, 8 task
#: classes, 2 needle-driver families) several times over, not the tens of
#: thousands each intent actually has -- 377,557 records across 144 cases
#: would mean decoding upwards of a million frames even at a few frames per
#: window (see the module docstring's "six million frames" arithmetic at 16
#: frames/record). A few thousand per intent, times ~12 intents, is a
#: corpus in the tens of thousands of examples -- ample for a LoRA adapter,
#: decodable in a bounded Condor job, and small enough that per-case /
#: per-answer stratification actually has room to matter.
DEFAULT_MAX_PER_INTENT = 2000


def sample_corpus(records, max_per_intent, seed):
    """Case-stratified, tool_presence_polar-balanced sample of `records`.

    Returns `(sampled_records, report)`. `report["intents"][intent]` always
    carries `available`, `sampled`, `balanced_by_answer`, `per_case` (a
    full `{case: count}` map), and `per_answer` (a full `{answer: count}`
    map) -- Task 3 requires reporting exactly what was sampled per-intent,
    per-case, and per-answer, and a summary that drops any one of those
    would hide exactly the kind of skew this function exists to fix.
    """
    rng = random.Random(seed)
    by_intent = defaultdict(list)
    for r in records:
        by_intent[r["intent"]].append(r)

    sampled = []
    report = {"max_per_intent": max_per_intent, "seed": seed, "intents": {}}
    for intent in sorted(by_intent):
        pool = by_intent[intent]
        balance = intent in BALANCED_INTENTS
        chosen = sample_intent(pool, max_per_intent, rng, balance_by_answer=balance)
        sampled.extend(chosen)
        per_case = Counter(r["case"] for r in chosen)
        per_answer = Counter(r["answer"] for r in chosen)
        report["intents"][intent] = {
            "available": len(pool),
            "sampled": len(chosen),
            "balanced_by_answer": balance,
            "per_case": dict(per_case),
            "per_answer": dict(per_answer),
        }
    return sampled, report


def _print_sampling_report(report):
    print("sampling: max_per_intent=%d seed=%d"
          % (report["max_per_intent"], report["seed"]))
    total_sampled = 0
    for intent, entry in sorted(report["intents"].items()):
        total_sampled += entry["sampled"]
        per_case = entry["per_case"].values()
        lo = min(per_case) if per_case else 0
        hi = max(per_case) if per_case else 0
        print("  %-28s available=%-7d sampled=%-6d balanced=%-5s "
              "cases=%-4d per_case[min=%d max=%d] answers=%s"
              % (intent, entry["available"], entry["sampled"], entry["balanced_by_answer"],
                 len(entry["per_case"]), lo, hi, dict(sorted(entry["per_answer"].items()))))
    print("total sampled across all intents: %d" % total_sampled)


# ============================================================================
# TASK 3 -- frame extraction
# ============================================================================

VIDEO_ROOT = "/staging/groups/bhaskar_opscribe/surgvu/videos/surgvu24"
QA_FRAMES_ROOT = "/staging/n/nkalthoff/surgvu26/qa_frames"

#: Source corpus is 60 fps (challenge spec) -- used only as a FALLBACK when
#: a video's own reported fps is missing/zero/NaN, mirroring
#: scripts/dump_motion_v2.py's identical fallback. The frame-index math
#: always prefers the fps actually read off the file via cv2, never assumes
#: this constant is correct for every part.
SOURCE_FPS = 60.0

#: Frames decoded per DISTINCT window, not per QA record (see the module
#: docstring on why windows are deduplicated first). Four, matching
#: `surgvu.vlm.DEFAULT_FRAMES` -- the serving Evidence VLM already samples
#: 4 frames from a 30 s clip; training on the same frame budget means the
#: fine-tune learns from what it will actually be given at inference time,
#: not a richer context it will never see in production.
DEFAULT_FRAMES_PER_WINDOW = VLM_DEFAULT_FRAMES

#: decode_clip_multiscale requires a non-empty `offsets_ms` but this caller
#: only wants its `centres` return value -- the probes are discarded. So this
#: is the cheapest legal value, not scripts/dump_motion_v2.py's multiscale
#: probe set (which exists for ITS OWN motion statistic and would cost extra
#: flanking reads here for nothing this script uses).
_MINIMAL_OFFSET_MS = (1,)

VideoInfo = namedtuple("VideoInfo", "path total fps")


def part_number(part):
    """`'1.0'` / `'1'` / `1` -> `1`, via `surgvu.labels.normalize_part` --
    never re-derived by parsing/guessing, so this and every other place in
    the codebase that reads a part label agree by construction.
    """
    return int(round(float(normalize_part(part))))


def video_path_for_record(video_root, case, part):
    """The one source file this record's frames must come from, resolved
    ONLY from the record's own `case`/`part` fields -- never by probing
    video durations to guess which of a case's (possibly two) parts a
    timestamp belongs to. Ruling R28: timestamps reset at part boundaries
    and guessing put ~6% of a previous training set on the wrong video.
    """
    nnn = "%03d" % part_number(part)
    return Path(video_root) / case / ("%s_video_part_%s.mp4" % (case, nnn))


def frame_dir_for_window(frames_root, case, part, t_start, t_stop):
    """The directory one window's decoded frames are written under.

    Millisecond integers, not a float repr, name the leaf directory: a raw
    `t_start`/`t_stop` like `1773.300869` risks two different-looking
    strings for what should be the same path across platforms/precisions,
    and a directory name is exactly the place that ambiguity becomes a
    silent duplicate-decode or a silent miss.
    """
    nnn = "%03d" % part_number(part)
    start_ms = int(round(float(t_start) * 1000.0))
    stop_ms = int(round(float(t_stop) * 1000.0))
    return Path(frames_root) / case / ("part_%s" % nnn) / ("%d_%d" % (start_ms, stop_ms))


def distinct_windows(records):
    """{(case, part, t_start, t_stop): [index into `records`, ...]}.

    Many QA records come from `generate_examples_for_window` on the SAME
    window -- tool presence, task, organ, count, ... are all asked about one
    30 s clip. Grouping here is what lets the caller decode that clip
    exactly once and hand every one of those records the same frame set,
    rather than decoding it once per record.
    """
    groups = defaultdict(list)
    for i, r in enumerate(records):
        key = (r["case"], r["part"], r["t_start"], r["t_stop"])
        groups[key].append(i)
    return groups


def resolve_window_video(video_root, case, part, cache):
    """`(VideoInfo, None)` or `(None, reason)` for one (case, part), cached
    by `(case, normalize_part(part))` so a video file already opened once
    for a batch of windows sharing it is never reopened -- real cases can
    have thousands of windows against one video part.
    """
    key = (case, normalize_part(part))
    if key in cache:
        return cache[key]
    path = video_path_for_record(video_root, case, part)
    if not path.exists():
        result = (None, "video_missing")
        cache[key] = result
        return result
    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 0:      # 0, None, or NaN
            fps = SOURCE_FPS
    finally:
        capture.release()
    if total <= 0:
        result = (None, "video_unreadable")
    else:
        result = (VideoInfo(path=path, total=total, fps=fps), None)
    cache[key] = result
    return result


def frame_index_range(t_start, t_stop, fps, total):
    """Inclusive `(first, last)` frame indices for `[t_start, t_stop)` at
    `fps`, or `None` if that range falls outside `[0, total)`.

    `None`, never a clamped substitute: a clamped `(first, total - 1)` would
    decode a DIFFERENT, shorter/shifted span of video than the question was
    actually written against, and that has to surface as a reported drop,
    not a silently different clip standing in for the right one.
    """
    first = int(round(t_start * fps))
    last = int(round(t_stop * fps)) - 1
    if last < first:
        last = first
    if first < 0 or last >= total:
        return None
    return first, last


def _decode_window_frames(video_path, first, last, n_frames, size):
    """`n_frames` `prepare_frame`-processed centre frames spanning frame
    indices `[first, last]` of `video_path`.

    Uses `surgvu.perceive.decode_clip_multiscale`'s additive `index_range`
    parameter (ruling R15) to SEEK directly in the source video -- no
    temporary clip is cut and nothing is re-encoded, so the frames this
    function returns are the exact bytes the serving decoder would read at
    the same indices. The `probes` half of the return value is discarded;
    this caller only wants centres.

    The torch-touching import lives inside this function body (surgvu.
    perceive imports torch at module scope) so every other function in this
    module -- including every other piece of Task 3's logic above --
    stays importable, and testable, without torch installed. Mirrors
    scripts/dump_motion_v2.py's identical pattern for the identical reason.
    """
    from surgvu.perceive import decode_clip_multiscale
    centres, _probes = decode_clip_multiscale(
        str(video_path), n_frames=n_frames, offsets_ms=_MINIMAL_OFFSET_MS,
        size=size, index_range=(first, last))
    return centres


def extract_frames_for_windows(records, video_root, frames_root, n_frames,
                               size, decode_fn=None, dry_run=False, workers=1):
    """Decode+write frames for every distinct window among `records`.

    Returns `(manifest, drops)`. `manifest` is a list of QA records (each
    the ORIGINAL record dict, plus `frame_paths` and `frame_dir`) for every
    record whose window decoded successfully. `drops` is a `Counter`
    tallying, by reason, how many RECORDS (not windows) lost their frames --
    a window shared by several records that fails must cost all of them,
    or "how many records lost their frames" (Task 3's explicit reporting
    requirement) silently undercounts.

    `decode_fn(video_path, first, last, n_frames, size) -> frames` defaults
    to `_decode_window_frames`; tests inject a fake so every path here
    except the actual pixel decode is exercised without torch.

    `workers > 1` decodes and writes windows on a thread pool. THE DECODE IS
    THE WALL CLOCK: measured at 247 frames/min single-threaded, a 16-frame
    rebuild of ~22,000 windows projects to ~24 hours, and the job requests 4
    CPUs it was using one of. cv2's decode and imencode both release the GIL,
    so threads -- not processes -- are enough, and they keep `video_cache`
    and the manifest in one address space.

    THE MANIFEST STAYS IN SERIAL ORDER REGARDLESS OF `workers`. Windows are
    resolved and ordered first, dispatched second, and the manifest is
    assembled third from the ORIGINAL ordering -- never from completion
    order. A manifest whose row order depended on thread scheduling would
    make `train_vlm.sample_eval_records` (a seeded `random.sample` over the
    record list) draw a different eval set on every rebuild, which is the
    kind of irreproducibility that is invisible until two runs disagree and
    nobody can say why.

    `dry_run=True` skips both the decode call AND the filesystem write, but
    still resolves every video, computes every frame-index range, and
    reports drops for real -- it is the one mode of this function that runs
    on a machine with no torch installed, and its manifest's `frame_paths`
    are exactly the paths a real run would have written to.
    """
    if decode_fn is None:
        decode_fn = _decode_window_frames
    frames_root = Path(frames_root)
    video_cache = {}
    drops = Counter()

    # -- PASS 1 (serial): resolve every window's video and frame span.
    # Serial on purpose -- it is metadata only (no decode), and it is what
    # populates `video_cache`, which would otherwise need a lock and would
    # re-open the same video from several threads at once.
    planned = []
    for (case, part, t_start, t_stop), idxs in distinct_windows(records).items():
        info, reason = resolve_window_video(video_root, case, part, video_cache)
        if info is None:
            drops[reason] += len(idxs)
            continue
        span = frame_index_range(t_start, t_stop, info.fps, info.total)
        if span is None:
            drops["window_out_of_bounds"] += len(idxs)
            continue
        out_dir = frame_dir_for_window(frames_root, case, part, t_start, t_stop)
        planned.append({
            "case": case, "part": part, "t_start": t_start, "t_stop": t_stop,
            "idxs": idxs, "path": info.path, "span": span, "out_dir": out_dir,
            "frame_paths": [str(out_dir / ("frame_%02d.jpg" % i))
                            for i in range(n_frames)],
        })

    def _do(job):
        """Decode+write ONE window. Returns (job, drop_reason_or_None).

        Raises nothing a worker thread would swallow: a genuinely unexpected
        error propagates out of `future.result()` below and kills the run,
        which is correct -- a silent per-window exception is how a rebuild
        ends up with a partial frames tree and a manifest that claims
        otherwise.
        """
        # RESUMABLE. A 16-frame rebuild runs for hours and CHTC preemption on
        # a job that long is a real, observed risk (see condor/train_vlm.sub on
        # cluster 9629572). A window whose frames are all present is skipped,
        # so a re-run continues instead of starting over. Checked per FILE, not
        # per directory: a job killed mid-window leaves a partial directory,
        # and treating that as done would put a manifest entry on frames that
        # were never written.
        # Size-checked, not just existence -- see
        # surgvu.extract.looks_like_a_written_jpeg. A window left half-written
        # by a killed job would otherwise be skipped forever and surface as
        # UnidentifiedImageError inside a DataLoader worker hours into training.
        if all(looks_like_a_written_jpeg(Path(p)) for p in job["frame_paths"]):
            return job, None
        first, last = job["span"]
        try:
            frames = decode_fn(job["path"], first, last, n_frames, size)
        except ValueError as exc:
            print("extract_frames_for_windows: %s" % exc)
            return job, "decode_failed"
        if len(frames) != n_frames:
            print("extract_frames_for_windows: %s window (%.3f, %.3f) "
                  "decoded %d of %d requested frames -- dropping"
                  % (job["case"], job["t_start"], job["t_stop"],
                     len(frames), n_frames))
            return job, "decode_short"
        job["out_dir"].mkdir(parents=True, exist_ok=True)
        for path_str, frame in zip(job["frame_paths"], frames):
            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
            if not ok:
                raise ValueError("cv2.imencode failed for %s" % path_str)
            Path(path_str).write_bytes(buf.tobytes())
        return job, None

    # -- PASS 2: decode+write. `failed` is keyed by id(job) so PASS 3 can ask
    # "did THIS window fail" in original order, whatever order results landed.
    failed = {}
    if not dry_run and planned:
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for job, reason in pool.map(_do, planned):
                    if reason is not None:
                        failed[id(job)] = reason
        else:
            for job in planned:
                _, reason = _do(job)
                if reason is not None:
                    failed[id(job)] = reason

    # -- PASS 3 (serial): assemble the manifest in the ORIGINAL window order,
    # so the output is byte-identical whatever `workers` was.
    manifest = []
    for job in planned:
        reason = failed.get(id(job))
        if reason is not None:
            drops[reason] += len(job["idxs"])
            continue
        for i in job["idxs"]:
            manifest.append(dict(records[i], frame_paths=job["frame_paths"],
                                 frame_dir=str(job["out_dir"])))
    return manifest, drops


def _print_extraction_report(records, manifest, drops):
    total_records = len(records)
    lost = sum(drops.values())
    print("frame extraction: %d record(s) in, %d survived, %d lost"
          % (total_records, len(manifest), lost))
    print("drop reasons (in RECORDS, not windows): %s" % dict(sorted(drops.items())))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def load_qa_pairs(path):
    """The records `main`'s generation mode wrote -- one JSON object per
    line, exactly the shape `build_records_for_case` produces.
    """
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _run_generate(args):
    """Logbook -> qa_pairs.jsonl (Task 2). Unchanged from before Task 3."""
    heldout_norm = load_heldout(args.splits)
    eligible, excluded = select_cases(args.labels_root, heldout_norm)

    variant_by_case = load_variant_labels(args.variant_labels)

    tool_hazards_total = Counter()
    task_hazards_total = Counter()
    gen_drops = Counter()
    paraphrase_usage = Counter()
    all_records = []

    labels_root = Path(args.labels_root)
    for case_id in eligible:
        case_dir = labels_root / case_id
        tool_h, task_h = scan_logbook_hazards(case_dir / "tools.csv", case_dir / "tasks.csv")
        tool_hazards_total.update(tool_h)
        task_hazards_total.update(task_h)

        labels = CaseLabels.from_dir(case_dir)
        variant_intervals = variant_by_case.get(case_id, [])
        all_records.extend(build_records_for_case(
            case_id, labels, variant_intervals, gen_drops, paraphrase_usage))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    _print_report(eligible, excluded, tool_hazards_total, task_hazards_total,
                  gen_drops, all_records, paraphrase_usage)
    print()
    print("wrote %s (%d records)" % (out_path, len(all_records)))
    return 0


def _run_extract_frames(args):
    """qa_pairs.jsonl -> sampled corpus -> frames + manifest (Task 3)."""
    records = load_qa_pairs(args.qa_pairs)
    if not records:
        raise SystemExit("no records read from %s" % args.qa_pairs)
    print("read %d record(s) from %s across %d case(s)"
          % (len(records), args.qa_pairs, len({r["case"] for r in records})))

    sampled, report = sample_corpus(records, args.max_per_intent, args.seed)
    print()
    _print_sampling_report(report)

    manifest, drops = extract_frames_for_windows(
        sampled, args.video_root, args.frames_root, args.frames_per_window,
        args.size, dry_run=args.dry_run, workers=args.workers)
    print()
    _print_extraction_report(sampled, manifest, drops)
    report["drops"] = dict(drops)
    report["records_in"] = len(sampled)
    report["records_out"] = len(manifest)
    report["distinct_windows"] = len(distinct_windows(sampled))

    manifest_out = Path(args.manifest_out)
    # A DRY RUN MUST NOT LEAVE A MANIFEST AT THE REAL PATH.
    #
    # --dry-run skips the decode, so every `frame_paths` entry it records names
    # a JPEG that was never written. Writing that to the production manifest
    # path leaves a file that is the right size, the right shape, and parses
    # cleanly -- but whose every frame is missing. Downstream,
    # train_vlm.filter_records_with_frames silently drops all of it and reports
    # "0 records", which reads as a data problem rather than as "you are
    # pointed at a dry run's output".
    #
    # It also cost a real submission: a dry run wrote qa_frames_manifest_v2
    # .jsonl, and the next REAL build refused to start because that path
    # already existed (build_qa_frames.sh's overwrite guard, working as
    # intended, on an artefact that should never have been there).
    #
    # `.dryrun` suffix, not "skip the write": the manifest is the dry run's
    # actual product -- it is how you check which windows resolved and what the
    # frame index ranges came out as -- it just must not masquerade as a real
    # one.
    if args.dry_run:
        manifest_out = manifest_out.with_suffix(manifest_out.suffix + ".dryrun")
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_out, "w", encoding="utf-8") as handle:
        for record in manifest:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    # The FULL per-intent/per-case/per-answer breakdown -- printed above only
    # as min/max summaries, because 144 cases x 12 intents is too much to
    # usefully scan in a Condor log -- is written here in full so "report
    # exactly what you sampled" has a machine-readable, auditable artefact
    # and does not rely on anyone re-deriving it from the manifest by hand.
    report_out = Path(args.report_out)
    if args.dry_run:      # same reason as the manifest above
        report_out = report_out.with_suffix(report_out.suffix + ".dryrun")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")

    print()
    print("wrote %s (%d records%s)"
          % (manifest_out, len(manifest), ", DRY RUN -- no frames written" if args.dry_run else ""))
    print("wrote %s (full sampling+extraction report)" % report_out)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels-root",
                        help="Directory of case_NNN/ dirs holding tools.csv "
                             "and tasks.csv. Required unless --extract-frames.")
    parser.add_argument("--splits",
                        help="config/splits_v2.json -- its 'heldout' list is "
                             "excluded. Required unless --extract-frames.")
    parser.add_argument("--variant-labels", default="config/variant_labels.json",
                        help="Needle-driver Large/Mega family intervals "
                             "(scripts/build_variant_labels.py's output). "
                             "If missing, variant_presence_polar examples "
                             "are simply not generated.")
    parser.add_argument("--out", help="qa_pairs.jsonl path to WRITE (generation "
                             "mode). Required unless --extract-frames.")

    parser.add_argument("--extract-frames", action="store_true",
                        help="Switch to Task 3's mode: sample --qa-pairs and "
                             "decode frames for the sampled records, instead "
                             "of generating qa_pairs.jsonl from the logbook.")
    parser.add_argument("--qa-pairs",
                        help="[--extract-frames] qa_pairs.jsonl to READ (this "
                             "script's own generation-mode output).")
    parser.add_argument("--video-root", default=VIDEO_ROOT,
                        help="[--extract-frames] root of <case>/<case>_video_"
                             "part_<NNN>.mp4 source videos.")
    parser.add_argument("--frames-root", default=QA_FRAMES_ROOT,
                        help="[--extract-frames] directory JPEG frames are "
                             "written under.")
    parser.add_argument("--manifest-out", default=None,
                        help="[--extract-frames] manifest JSONL path; default "
                             "is <frames-root>_manifest.jsonl.")
    parser.add_argument("--report-out", default=None,
                        help="[--extract-frames] full sampling+extraction "
                             "report JSON path (per-intent/per-case/"
                             "per-answer counts, drop tallies); default is "
                             "<frames-root>_report.json.")
    parser.add_argument("--max-per-intent", type=int, default=DEFAULT_MAX_PER_INTENT,
                        help="[--extract-frames] per-intent sample cap.")
    parser.add_argument("--seed", type=int, default=0,
                        help="[--extract-frames] sampling is randomised; fixed "
                             "so a rerun is reproducible.")
    parser.add_argument("--frames-per-window", type=int, default=DEFAULT_FRAMES_PER_WINDOW,
                        help="[--extract-frames] frames decoded per distinct "
                             "window.")
    parser.add_argument("--size", type=int, default=512,
                        help="[--extract-frames] decoded frame side length.")
    parser.add_argument("--workers", type=int, default=1,
                        help="[--extract-frames] threads decoding+writing "
                             "windows. cv2 releases the GIL in both decode "
                             "and imencode, so threads are enough. Default 1 "
                             "(serial) so behaviour is unchanged unless asked "
                             "for; the manifest is identical either way.")
    parser.add_argument("--dry-run", action="store_true",
                        help="[--extract-frames] resolve videos, compute frame "
                             "index ranges, and report drops for real, but "
                             "skip the actual decode/write -- the mode this "
                             "script can run without torch installed.")
    args = parser.parse_args(argv)

    if args.extract_frames:
        if not args.qa_pairs:
            parser.error("--extract-frames requires --qa-pairs")
        if args.manifest_out is None:
            args.manifest_out = str(Path(args.frames_root).with_name(
                Path(args.frames_root).name + "_manifest.jsonl"))
        if args.report_out is None:
            args.report_out = str(Path(args.frames_root).with_name(
                Path(args.frames_root).name + "_report.json"))
        return _run_extract_frames(args)

    missing = [name for name, val in (("--labels-root", args.labels_root),
                                      ("--splits", args.splits),
                                      ("--out", args.out)) if not val]
    if missing:
        parser.error("%s required (unless --extract-frames)" % ", ".join(missing))
    return _run_generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
