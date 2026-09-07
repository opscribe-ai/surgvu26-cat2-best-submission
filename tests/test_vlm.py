"""The gated VLM fallback, tested without a GPU, torch or 6 GB of weights.

Everything here is about the two properties that decide whether this module
is safe to ship at all:

  * it can only ever produce a string for a question the router could not
    answer, and
  * every way it can fail -- an import error, an OOM, a timeout, an empty
    generation, a refusal -- ends in None, which means the router's calibrated
    sentence is what gets written.

The parts that need real weights (the chat template, the generate call, the
device placement) are exercised in tests/test_inference_vlm.py inside the
container and by the measured GPU run; nothing here pretends to cover them.
"""
import time

import pytest

from surgvu import vlm


# ----------------------------------------------------------- prompt building

def test_the_prompt_carries_the_question_verbatim():
    prompt = vlm.build_prompt("What is the purpose of using forceps?", {})

    assert "What is the purpose of using forceps?" in prompt


def test_the_prompt_asks_for_a_short_answer():
    """The metric scores against references that are mostly a bare noun
    phrase, so an unbounded answer is a losing answer."""
    prompt = vlm.build_prompt("Anything at all?", {})

    assert "12 words" in prompt
    assert "no explanation" in prompt


def test_the_prompt_states_what_the_cnns_found():
    prompt = vlm.build_prompt(
        "Anything?", {"tools_present": ["needle driver"], "task_top": "suturing"})

    assert "needle driver" in prompt
    assert "suturing" in prompt


def test_the_cnn_evidence_is_offered_as_a_report_not_as_truth():
    """`tools_present` is a thresholded list with known false negatives
    (case124's cadiere, case126's needle driver). A prompt asserting it as
    fact teaches the model to repeat our errors more confidently than we hold
    them."""
    prompt = vlm.build_prompt("Anything?", {"tools_present": ["stapler"]})

    assert "classifier reports" in prompt
    assert "may be wrong" in prompt


def test_an_empty_perception_record_adds_no_context_line():
    assert vlm.perception_context({}) == ""
    assert vlm.perception_context({"tools_present": [], "task_top": ""}) == ""
    assert vlm.perception_context(None) == ""


def test_context_can_be_turned_off_without_changing_the_question():
    perception = {"tools_present": ["stapler"], "task_top": "suturing"}

    with_context = vlm.build_prompt("Why?", perception, use_context=True)
    without = vlm.build_prompt("Why?", perception, use_context=False)

    assert "stapler" in with_context
    assert "stapler" not in without
    assert "Question: Why?" in without


def test_a_task_alone_still_produces_a_context_line():
    assert "suturing" in vlm.perception_context({"task_top": "suturing"})


def test_whitespace_in_the_question_is_collapsed_into_one_line():
    """A question arriving with a newline in it would otherwise split the
    prompt's own line structure and hide the instruction that follows it."""
    prompt = vlm.build_prompt("What  organ\nis  shown?", {})

    assert "Question: What organ is shown?" in prompt


# --------------------------------------------------------------- sanitising

@pytest.mark.parametrize("raw,expected", [
    ("Uterine horn", "Uterine horn"),
    ("  Uterine horn  ", "Uterine horn"),
    ('"Uterine horn"', "Uterine horn"),
    ("Answer: Uterine horn", "Uterine horn"),
    ("answer - Uterine horn", "Uterine horn"),
    ("Uterine horn\nBecause the frames show it.", "Uterine horn"),
    ("\n\nUterine  horn\n", "Uterine horn"),
])
def test_a_usable_generation_is_cleaned_but_not_rewritten(raw, expected):
    assert vlm.sanitize(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "\n\n", None, 7, b"bytes"])
def test_nothing_usable_sanitises_to_none(raw):
    """None means "keep the router's answer". An empty string would be
    collapsed back to the generic fallback downstream anyway; returning None
    is what makes that decision legible."""
    assert vlm.sanitize(raw) is None


@pytest.mark.parametrize("raw", [
    "I cannot determine the organ from these frames.",
    "I'm unable to tell.",
    "As an AI, I do not have enough information.",
    "Sorry, the frames are too blurry.",
    "Unable to identify the structure.",
    "Unfortunately there is not enough detail.",
])
def test_a_refusal_is_declined_in_favour_of_the_calibrated_fallback(raw):
    """The generic sentence is a MEASURED 0.35-0.48 on an open question. A
    first-person hedge is not measured, is long, and shares no vocabulary with
    a reference that is usually a bare noun phrase."""
    assert vlm.sanitize(raw) is None


def test_a_refusal_word_later_in_the_answer_is_kept():
    """The guard is anchored on purpose: an answer that MENTIONS being unable
    is not the same as an answer that declines."""
    assert vlm.sanitize("The clip is unable to show the ureter") is not None


def test_an_over_long_answer_is_truncated_rather_than_dropped():
    raw = " ".join("word%d" % i for i in range(60))

    cleaned = vlm.sanitize(raw)

    assert len(cleaned.split()) == vlm.MAX_ANSWER_WORDS
    assert cleaned.startswith("word0 word1")


def test_a_sentence_length_answer_survives_intact():
    """The purpose family's gold answers ARE sentences -- "To grasp and hold
    tissues or objects during the surgery." -- so the truncation bound must
    sit above them, not on them."""
    gold = "To grasp and hold tissues or objects during the surgery."

    assert vlm.sanitize(gold) == gold


# ----------------------------------------------------------- frame selection

def test_frames_are_spread_across_the_clip_not_taken_from_its_head():
    assert vlm.even_indices(16, 4) == [2, 6, 10, 14]


def test_asking_for_more_frames_than_exist_returns_each_one_once():
    assert vlm.even_indices(3, 8) == [0, 1, 2]


@pytest.mark.parametrize("total,wanted", [(0, 4), (4, 0), (-1, 4), (4, -1)])
def test_a_degenerate_frame_request_is_empty_rather_than_an_exception(
        total, wanted):
    assert vlm.even_indices(total, wanted) == []


# ------------------------------------------------------- the fail-safe gate

class _Recorder(object):
    def __init__(self):
        self.lines = []

    def __call__(self, message):
        self.lines.append(message)

    def said(self, fragment):
        return any(fragment in line for line in self.lines)


def _fallback(monkeypatch, generation, available=True, **kwargs):
    """A QwenVlmFallback whose only real machinery is `_generate`."""
    log = _Recorder()
    fallback = vlm.QwenVlmFallback(model_dir="/nonexistent", log=log, **kwargs)
    monkeypatch.setattr(fallback, "available", lambda: available)
    monkeypatch.setattr(fallback, "_images", lambda frames: list(frames))
    monkeypatch.setattr(fallback, "_generate", generation)
    return fallback, log


def test_a_clean_generation_becomes_the_answer(monkeypatch):
    fallback, _log = _fallback(
        monkeypatch, lambda images, prompt, deadline: "Uterine horn")

    assert fallback.answer("What organ?", {}, ["frame"]) == "Uterine horn"


def test_a_vlm_that_raises_returns_none_rather_than_propagating(monkeypatch):
    """A crash inside the graded container scores 0 for the case. Declining
    scores 0.35-0.48. The exception may not escape."""
    def explode(images, prompt, deadline):
        raise RuntimeError("CUDA out of memory")

    fallback, log = _fallback(monkeypatch, explode)

    assert fallback.answer("What organ?", {}, ["frame"]) is None
    assert log.said("declined")


def test_an_import_failure_deep_in_the_stack_is_absorbed(monkeypatch):
    def explode(images, prompt, deadline):
        raise ImportError("No module named 'transformers'")

    fallback, _log = _fallback(monkeypatch, explode)

    assert fallback.answer("What organ?", {}, ["frame"]) is None


def test_an_empty_generation_returns_none(monkeypatch):
    fallback, _log = _fallback(monkeypatch, lambda images, prompt, deadline: "")

    assert fallback.answer("What organ?", {}, ["frame"]) is None


def test_a_none_generation_returns_none(monkeypatch):
    fallback, _log = _fallback(monkeypatch,
                               lambda images, prompt, deadline: None)

    assert fallback.answer("What organ?", {}, ["frame"]) is None


def test_no_cuda_device_means_declining_not_crashing(monkeypatch):
    """The NF4 weights are a bitsandbytes artifact and need CUDA. The
    deployment instance may legitimately be No GPU, and that must be a quiet
    decline rather than a stack trace on every case."""
    called = []
    fallback, log = _fallback(
        monkeypatch,
        lambda images, prompt, deadline: called.append(1) or "Uterine horn",
        available=False)

    assert fallback.answer("What organ?", {}, ["frame"]) is None
    assert called == []
    assert log.said("no CUDA")


def test_no_frames_means_declining_before_anything_is_loaded(monkeypatch):
    called = []
    fallback, log = _fallback(
        monkeypatch,
        lambda images, prompt, deadline: called.append(1) or "Uterine horn")

    assert fallback.answer("What organ?", {}, []) is None
    assert called == []
    assert log.said("no frames")


def test_an_empty_question_is_declined(monkeypatch):
    called = []
    fallback, _log = _fallback(
        monkeypatch,
        lambda images, prompt, deadline: called.append(1) or "Uterine horn")

    assert fallback.answer("   ", {}, ["frame"]) is None
    assert called == []


def test_the_deadline_handed_to_generate_reflects_the_budget(monkeypatch):
    seen = {}

    def capture(images, prompt, deadline):
        seen["deadline"] = deadline
        return "Uterine horn"

    fallback, _log = _fallback(monkeypatch, capture, budget_seconds=30.0)
    before = time.time()

    fallback.answer("What organ?", {}, ["frame"])

    assert 29.0 <= seen["deadline"] - before <= 31.0


def test_a_refusal_from_a_working_model_still_keeps_the_router_answer(
        monkeypatch):
    fallback, _log = _fallback(
        monkeypatch,
        lambda images, prompt, deadline: "I cannot tell from these frames.")

    assert fallback.answer("What organ?", {}, ["frame"]) is None


def test_the_prompt_that_reaches_generate_is_the_built_one(monkeypatch):
    seen = {}

    def capture(images, prompt, deadline):
        seen["prompt"] = prompt
        return "Uterine horn"

    fallback, _log = _fallback(monkeypatch, capture)

    fallback.answer("What organ?", {"task_top": "suturing"}, ["frame"])

    assert seen["prompt"] == vlm.build_prompt(
        "What organ?", {"task_top": "suturing"}, True)


def test_construction_touches_nothing_expensive():
    """Enabling this module and never hitting an unrouted question must cost
    nothing: no import, no file read, no CUDA context."""
    fallback = vlm.QwenVlmFallback(model_dir="/definitely/not/here")

    assert fallback._model is None
    assert fallback._processor is None
