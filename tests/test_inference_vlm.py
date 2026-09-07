"""The VLM seam in the submission entrypoint (Plan 2, Task 4).

Separate from tests/test_inference.py only to keep two concurrent workstreams
out of each other's way; the fixtures are imported from it so there is one
definition of what a case looks like.

WHAT CHANGED FROM THE PRE-PLAN-2 SEAM THIS FILE USED TO TEST. The old seam
(`vlm_answer`/`try_vlm`, `surgvu.vlm.QwenVlmFallback`) only ever offered
INTENT_UNKNOWN_OPEN questions to the VLM, on the measured grounds that the
CNN+router path (0.8766) beat a VLM-as-sole-answerer (0.4923-0.5743) on every
question the router could classify. Plan 2 replaces that gate with
`surgvu.arbiter`: the VLM (`surgvu.evidence_vlm`) now drafts an answer for
EVERY question, and the arbiter -- not an intent filter in this file -- is
what decides whether that draft ships. `config/arbiter.json`'s shipped mode
is `challenger`; see `arbiter.py`'s own module docstring for the measurement
this rests on (`fallback`'s ceiling on the graded sample is exactly zero).

WHAT THIS FILE STILL PROTECTS. The container's one hard guarantee: it always
writes an answer.

  * `--vlm` off (the shipped default until this is deliberately turned on) ->
    `build_vlm` returns None -> `route()` is byte-identical to the router
    alone, exactly as `arbiter.arbitrate`'s own fall-through property
    guarantees.
  * No CUDA device -> the VLM is skipped BEFORE any `transformers` import,
    not attempted and caught -- this is a correctness requirement (the
    shipped weights are 4-bit NF4, a bitsandbytes/CUDA-only format), not a
    performance nicety.
  * Every other VLM failure (construction, a missing model directory, a
    crash mid-generation, a malformed return) is absorbed at the seam,
    R18-style: a traceback to stderr, a WARNING, and the router's answer
    stands -- never an exception that reaches `write_response` with nothing
    to write.
  * A question the router DOES route now also reaches the VLM -- pinned
    explicitly below, because that is the one property a careless read of
    the old test file's name ("the gate is one intent") would expect this
    file to still assert, and it is now the opposite.
  * `EvidenceVlmHandle.sample` passes `{}`, not the real `perception`
    packet, to `evidence_vlm.build_sampling_prompt` unless
    `config/arbiter.json`'s `vlm_evidence_context` (or a per-run
    `--vlm-evidence-context` override) says otherwise -- see the "train/serve
    prompt parity switch" tests below. `scripts/train_vlm.py` trains this
    adapter against that same empty context; passing `perception`
    unconditionally would silently serve prompt text the model never saw
    during training.

What is NOT re-tested here: the arbiter's own policy logic (which mode
overrides when) is `tests/test_arbiter.py`'s job, and the prompt/evidence
rendering is `tests/test_evidence_prompt.py`'s. This file is only about the
serving wiring -- constructing the VLM, gating it on CUDA, and never letting
it cost the case its answer.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import inference                                                    # noqa: E402
from surgvu.evidence_vlm import ConfidenceResult                    # noqa: E402
from surgvu.router import FALLBACK_OPEN, FALLBACK_POLAR             # noqa: E402

# The fixtures. Imported rather than redefined: a second `case` fixture that
# drifted from the first would test a container nobody ships.
from test_inference import (                                        # noqa: E402,F401
    Case, IMAGE_SIZE, _config, _write_video, checkpoints, case, tiny_backbone,
)

# The 11 public sample questions, verbatim from
# /staging/groups/bhaskar_opscribe/surgvu/cat2_sample/caseNNN/caseNNN_question.json.
# Embedded rather than read so this file is hermetic on a node with no
# /staging; the set is last year's frozen test data and cannot change.
SAMPLE_QUESTIONS = {
    "case122": "Are there forceps being used here?",
    "case123": "Is a large needle driver among the listed tools?",
    "case124": "What type of forceps is mentioned?",
    "case125": "Is a suture required in this surgical step?",
    "case126": "Was a large needle driver used in this clip?",
    "case127": "What organ is being manipulated?",
    "case128": "Is a needle driver involved in the procedure?",
    "case129": "What procedure is this summary describing?",
    "case130": "What is the purpose of using forceps in this procedure?",
    "case131": "Is tissue being cut during this clip?",
    "case132": "Was a large needle driver used during the surgery?",
}

# A question no rule in the router matches, so it classifies unknown_open and
# is answered with one generic sentence absent a VLM.
UNROUTED_OPEN = "Describe what you can see in the upper left corner."


def usable_result(answer="OVERRULED BY THE VLM", confidence=0.99):
    """A `ConfidenceResult` `arbiter._is_usable_vlm_result` accepts, and high
    enough confidence to clear `challenger`'s default 0.66 ceiling -- so a
    fake built from this is guaranteed to be able to override the router's
    answer, making any failure to do so the seam's fault, not the policy's.
    """
    return ConfidenceResult(answer=answer, confidence=confidence,
                            n_calls_used=1, all_answers=[answer], agreed=True)


class _Loud(object):
    """A VLM whose `sample()` always returns a confident, usable draft, so
    any silence in the final answer is the SEAM's doing, not the arbiter
    declining to override for its own policy reasons."""

    def __init__(self, answer="OVERRULED BY THE VLM", confidence=0.99,
                arbiter_mode=None):
        self.answer_text = answer
        self.confidence = confidence
        self.arbiter_mode = arbiter_mode
        self.calls = []

    def available(self):
        return True

    def sample(self, video, question, perception, budget_seconds=None):
        self.calls.append((video, question, perception))
        return usable_result(self.answer_text, self.confidence)


def _with_vlm(monkeypatch, fake):
    monkeypatch.setattr(inference, "build_vlm", lambda args: fake)


# ----------------------------------------------------- off unless asked for

def test_the_vlm_is_off_by_default():
    """Shipping it has to be a decision. The flag is the decision."""
    assert inference.parse_args([]).vlm is False
    assert inference.build_vlm(inference.parse_args([])) is None


def test_the_flag_is_what_turns_it_on():
    built = inference.build_vlm(inference.parse_args(["--vlm"]))

    assert isinstance(built, inference.EvidenceVlmHandle)


def test_the_model_dir_defaults_to_the_in_container_path():
    """Not evidence_vlm.DEFAULT_MODEL_DIR's bare HF hub id -- that would try
    a network fetch a no-internet, HF_HUB_OFFLINE=1 container refuses."""
    built = inference.build_vlm(inference.parse_args(["--vlm"]))

    assert built.model_dir == str(inference.DEFAULT_VLM_MODEL_DIR)


def test_the_model_dir_default_is_repo_derived_not_a_bare_literal():
    """R32, same defect class as --variant-config's (see
    tests/test_inference_evidence.py's
    test_variant_config_default_is_absolute_not_relative). The old default
    was the bare literal "/opt/algorithm/models/qwen25vl-7b-nf4" -- correct
    INSIDE the container by luck, not derivation, and with no local
    counterpart at all: it could never resolve to anything on this login
    node, or on a machine testing scripts/inference.py from a checkout
    elsewhere. REPO-derived (Path(__file__).resolve().parents[1], exactly
    like DEFAULT_CONFIG/DEFAULT_VARIANT_CONFIG) gives it one: <repo_root>/
    models/qwen25vl-7b-nf4 locally, /opt/algorithm/models/qwen25vl-7b-nf4 in
    the container, by the same mechanism, not two hardcoded strings that
    could drift apart.

    BREAKS ON: reverting DEFAULT_VLM_MODEL_DIR to the bare literal string.
    """
    path = Path(inference.DEFAULT_VLM_MODEL_DIR)
    assert path.is_absolute(), (
        "R32: DEFAULT_VLM_MODEL_DIR must be absolute, and derived from "
        "REPO rather than hardcoded, so it resolves correctly both locally "
        "and in the container")
    assert path.parts[-2:] == ("models", "qwen25vl-7b-nf4")
    assert inference.DEFAULT_VLM_MODEL_DIR == inference.REPO / "models" / "qwen25vl-7b-nf4"


def test_arbiter_mode_defaults_to_none_not_a_hardcoded_string():
    """None means "read config/arbiter.json's own mode" -- see route()."""
    built = inference.build_vlm(inference.parse_args(["--vlm"]))

    assert inference.parse_args(["--vlm"]).arbiter_mode is None
    assert built.arbiter_mode is None


def test_the_arbiter_mode_flag_is_carried_onto_the_handle():
    built = inference.build_vlm(
        inference.parse_args(["--vlm", "--arbiter-mode", "primary"]))

    assert built.arbiter_mode == "primary"


# ------------------------- the train/serve prompt parity switch (Plan 2b) --
#
# NOTE: this whole file requires `torch` (imported at `scripts/inference.py`
# module scope) and cannot be collected or run on a login node without it --
# these tests are written to run under this project's real training/serving
# environment, not here. See tests/test_train_vlm.py's
# test_serving_and_training_prompts_match_under_the_shipped_default for the
# torch-free half of this same invariant, which CAN run and does run on this
# login node.

def test_evidence_context_defaults_to_none_not_a_hardcoded_bool():
    """None means "read config/arbiter.json's own vlm_evidence_context" --
    the same "None defers to the config file" contract `--arbiter-mode`
    already uses (see test_arbiter_mode_defaults_to_none_not_a_hardcoded_
    string above)."""
    assert inference.parse_args(["--vlm"]).vlm_evidence_context is None


def test_evidence_context_resolves_from_the_shipped_config_not_the_fallback():
    """config/arbiter.json's OWN `vlm_evidence_context` is the value that must
    reach the handle when no CLI flag overrides it -- not
    DEFAULT_VLM_EVIDENCE_CONTEXT's in-code fallback, which would agree by
    accident whenever the two happen to match.

    Asserted against the config rather than a literal: this test hardcoded
    `False` until 2026-08-29, and shipping v6 (trained WITH evidence) flipped
    the config to `true` and broke it, even though the behaviour under test --
    "the file wins over the in-code default" -- had not changed at all.
    """
    from surgvu import arbiter

    shipped = bool(arbiter.load_config().get("vlm_evidence_context", False))
    built = inference.build_vlm(inference.parse_args(["--vlm"]))

    assert built.evidence_context is shipped


def test_evidence_context_flag_turns_it_on_for_one_run():
    built = inference.build_vlm(
        inference.parse_args(["--vlm", "--vlm-evidence-context"]))

    assert built.evidence_context is True


def test_no_evidence_context_flag_forces_it_off():
    """Useful only if config/arbiter.json's default were ever flipped to
    true -- proves the override direction works both ways, not just on."""
    built = inference.build_vlm(
        inference.parse_args(["--vlm", "--no-vlm-evidence-context"]))

    assert built.evidence_context is False


def test_evidence_context_falls_back_to_default_when_config_key_absent(
        monkeypatch):
    """A config file that predates this switch (no `vlm_evidence_context`
    key at all) must degrade to DEFAULT_VLM_EVIDENCE_CONTEXT (bare), not
    raise a KeyError -- the same "a malformed config must degrade the
    policy, never take down the run" contract `arbiter.load_config` uses."""
    monkeypatch.setattr(inference, "load_arbiter_config",
                        lambda: {"mode": "challenger"})

    built = inference.build_vlm(inference.parse_args(["--vlm"]))

    assert built.evidence_context == inference.DEFAULT_VLM_EVIDENCE_CONTEXT
    assert built.evidence_context is False


def test_sample_renders_a_bare_prompt_when_evidence_context_is_off(
        monkeypatch):
    """THE BUG THIS FILE EXISTS TO CATCH: `EvidenceVlmHandle.sample` must
    hand `adaptive_confidence_sample` an EMPTY context (`{}`), matching
    `scripts/train_vlm.py`'s `render_training_prompt(question) ==
    build_sampling_prompt(question, {})`, when `evidence_context=False` --
    even though a real, non-empty `perception` packet is available and was
    passed in. Passing `perception` straight through unconditionally here
    (the pre-fix behaviour) is exactly the train/serve mismatch this task
    fixes."""
    import surgvu.evidence_vlm as evidence_vlm_module

    seen = {}

    def fake_adaptive_confidence_sample(video, question, context, **kwargs):
        seen["context"] = context
        return None

    monkeypatch.setattr(evidence_vlm_module, "adaptive_confidence_sample",
                        fake_adaptive_confidence_sample)

    handle = inference.EvidenceVlmHandle(
        "/any/dir", 5, 3, inference.log, evidence_context=False)
    real_perception = {"tools_present": ["needle_driver"],
                       "tools": {"needle_driver": 0.9}}

    handle.sample("video.mp4", "q?", real_perception)

    assert seen["context"] == {}


def test_sample_renders_the_full_evidence_packet_when_evidence_context_is_on(
        monkeypatch):
    """The other half: with the switch explicitly on, the real `perception`
    packet reaches the renderer unchanged -- proving the gate does not
    always force `{}` regardless of the flag."""
    import surgvu.evidence_vlm as evidence_vlm_module

    seen = {}

    def fake_adaptive_confidence_sample(video, question, context, **kwargs):
        seen["context"] = context
        return None

    monkeypatch.setattr(evidence_vlm_module, "adaptive_confidence_sample",
                        fake_adaptive_confidence_sample)

    handle = inference.EvidenceVlmHandle(
        "/any/dir", 5, 3, inference.log, evidence_context=True)
    real_perception = {"tools_present": ["needle_driver"],
                       "tools": {"needle_driver": 0.9}}

    handle.sample("video.mp4", "q?", real_perception)

    assert seen["context"] == real_perception


def test_enabling_it_imports_nothing_expensive_until_a_question_needs_it(
        monkeypatch):
    """Construction stores a path and two ints; it must not import
    transformers, touch the filesystem, or create a CUDA context."""
    def boom(*_args, **_kwargs):
        raise AssertionError("adaptive_confidence_sample must not run "
                             "during construction")

    import surgvu.evidence_vlm as evidence_vlm_module
    monkeypatch.setattr(evidence_vlm_module, "adaptive_confidence_sample", boom)

    built = inference.build_vlm(
        inference.parse_args(["--vlm", "--vlm-model", "/definitely/not/here"]))

    assert built.model_dir == "/definitely/not/here"


def test_a_vlm_that_cannot_even_be_constructed_is_declined(capsys, monkeypatch):
    """`build_vlm` runs OUTSIDE the pipeline's try block, so an exception here
    would escape everything and the case would get no response file at all --
    a 0 where the router's answer is a real number."""
    def explode(*args, **kwargs):
        raise RuntimeError("transformers is not installed")

    monkeypatch.setattr(inference, "EvidenceVlmHandle", explode)

    assert inference.build_vlm(inference.parse_args(["--vlm"])) is None
    assert "running without it" in capsys.readouterr().err


def test_the_shipped_seam_is_inert_without_a_vlm():
    assert inference.try_vlm_result("video.mp4", "anything", {}, None, []) is None


def test_the_sample_count_is_configurable():
    """Item 6's lever against a runaway wall-clock: fewer allowed samples is
    the one knob available without touching surgvu.evidence_vlm itself."""
    default_built = inference.build_vlm(inference.parse_args(["--vlm"]))
    capped_built = inference.build_vlm(
        inference.parse_args(["--vlm", "--vlm-max-samples", "1"]))

    from surgvu.evidence_vlm import DEFAULT_MAX_SAMPLES
    assert default_built.max_samples == DEFAULT_MAX_SAMPLES
    assert capped_built.max_samples == 1


# --------------------------------------- it now reaches EVERY question ----
#
# THE OPPOSITE OF THE OLD GATE, DELIBERATELY. Pinned so a future edit cannot
# silently reintroduce "if classify_question(question) == INTENT_UNKNOWN_OPEN"
# around the call to try_vlm_result -- that WAS this file's contract before
# Plan 2, and it is not any more.

@pytest.mark.parametrize("case_id", sorted(SAMPLE_QUESTIONS))
def test_every_sample_question_reaches_the_vlm_when_enabled(
        case_id, case, monkeypatch):
    loud = _Loud()
    _with_vlm(monkeypatch, loud)
    case.ask(SAMPLE_QUESTIONS[case_id])

    assert case.run("--vlm") == 0

    assert len(loud.calls) == 1, (
        "%r must reach the VLM now -- the arbiter decides whether it "
        "overrides, not an intent gate in inference.py" % (case_id,))


def test_a_routed_question_can_be_overridden_by_a_confident_vlm(
        case, monkeypatch):
    """The measured, deliberate `challenger` behaviour this task wires:
    unlike the old seam, a routed question is not immune. Deterministic, not
    just plausible -- `get_router_confidence` always returns None
    (`arbiter.py`'s own documented "structurally inert" fact), so
    `challenger` treats every router answer as unvouched-for and a VLM draft
    at 0.99 confidence always clears the 0.66 override ceiling. What OTHER
    modes do with a confident draft is `test_arbiter.py`'s job to pin in
    detail; this only confirms the WIRING lets `challenger` do what its own
    docstring says it will."""
    # arbiter_mode ON THE FAKE, not on the command line. `_with_vlm` replaces
    # build_vlm outright, so the CLI's --arbiter-mode never reaches anything:
    # route() reads the mode from `vlm.arbiter_mode`, which is why _Loud takes
    # that argument at all. Passing the flag instead looks right and does
    # nothing -- a first attempt at this fix did exactly that and failed
    # identically.
    #
    # Requesting challenger EXPLICITLY rather than relying on the shipped
    # default: config/arbiter.json moved to `fallback` on 2026-08-26 (v5
    # shipped challenger and scored 0.7737 against v4's 0.8015), under which a
    # routed intent is immune by design. A test of "can challenger override"
    # should ask for challenger; coupling it to whichever mode currently ships
    # makes it break whenever that product decision changes.
    _with_vlm(monkeypatch, _Loud("No", confidence=0.99,
                                 arbiter_mode="challenger"))
    # tool_presence_polar -- a genuinely ROUTED intent (not unknown_open or
    # unknown_polar), verified directly against surgvu.router.classify_question.
    case.ask("Is a needle driver being used?")

    assert case.run("--vlm") == 0

    assert case.answer() == "No"


# ------------------------------------------ what it is allowed to do, once

def test_a_confident_draft_can_replace_the_generic_fallback(case, monkeypatch):
    _with_vlm(monkeypatch, _Loud("A trocar"))
    case.ask(UNROUTED_OPEN)

    assert case.run("--vlm") == 0

    assert case.answer() == "A trocar"


def test_the_vlms_answer_goes_through_the_routers_final_form(case, monkeypatch):
    """Casing is worth ~0.2 BERTScore, so an overriding answer is finalised
    the same way every other answer is rather than being written raw."""
    _with_vlm(monkeypatch, _Loud("  a trocar   is   visible  "))
    case.ask(UNROUTED_OPEN)

    case.run("--vlm")

    assert case.answer() == "A trocar is visible"


def test_the_vlm_is_handed_the_video_path_not_decoded_cnn_frames(
        case, monkeypatch):
    """The Evidence VLM re-decodes its own (smaller) frame set from the path
    via evidence_vlm.sample_frames -- see EvidenceVlmHandle.sample -- rather
    than reusing the CNN path's `frames` array, so the seam hands it the
    path, not an ndarray."""
    loud = _Loud()
    _with_vlm(monkeypatch, loud)
    case.ask(UNROUTED_OPEN)

    case.run("--vlm")

    video, _question, perception = loud.calls[0]
    assert str(video).endswith(inference.VIDEO_NAME)
    assert perception["task_top"] in perception["task"]


# ------------------------------------------------------------- fail safe --

def test_a_vlm_that_raises_still_leaves_the_generic_answer(
        case, monkeypatch, capsys):
    class Exploding(object):
        arbiter_mode = None

        def available(self):
            return True

        def sample(self, video, question, perception, budget_seconds=None):
            raise RuntimeError("CUDA out of memory")

    _with_vlm(monkeypatch, Exploding())
    case.ask(UNROUTED_OPEN)

    assert case.run("--vlm") == 0

    assert case.answer() == FALLBACK_OPEN
    err = capsys.readouterr().err
    assert "keeping the router's answer" in err
    # The whole-pipeline fallback would produce the same string here. The log
    # is what says the perception half was fine and only the VLM died.
    assert "FALLBACK:" not in err


@pytest.mark.parametrize("returned", [None, "not a ConfidenceResult", 7, object()])
def test_a_vlm_that_returns_something_unusable_still_answers(
        case, monkeypatch, returned):
    class Unusable(object):
        arbiter_mode = None

        def available(self):
            return True

        def sample(self, video, question, perception, budget_seconds=None):
            return returned

    _with_vlm(monkeypatch, Unusable())
    case.ask(UNROUTED_OPEN)

    assert case.run("--vlm") == 0

    assert case.answer() == FALLBACK_OPEN


def test_a_missing_model_directory_costs_nothing_but_its_own_answer(case):
    """The real class, the real gate, no weights anywhere -- the
    controller's finding that the NF4 checkpoint is not staged yet made
    real. Exit 0 and a valid non-empty answer either way: a crash here
    would convert the router's answer into a 0."""
    case.ask(UNROUTED_OPEN)

    assert case.run("--vlm", "--vlm-model", "/definitely/not/here") == 0

    answer = json.loads(case.response.read_text(encoding="utf-8"))
    assert answer == FALLBACK_OPEN


def test_a_vlm_failure_does_not_stop_the_response_being_written(case):
    case.ask(UNROUTED_OPEN)
    case.run("--vlm", "--vlm-model", "/definitely/not/here")

    raw = case.response.read_bytes()

    assert raw.startswith(b'"') and raw.endswith(b'"')
    assert json.loads(raw.decode("utf-8")).strip()


def test_a_perception_failure_never_reaches_the_vlm(case, monkeypatch):
    """The fallback path answers from the question alone -- it never even
    builds a `perception` dict, so there is nothing for the VLM to read
    from and it must not be asked."""
    loud = _Loud()
    _with_vlm(monkeypatch, loud)
    case.ask(UNROUTED_OPEN)
    case.video.write_bytes(b"")

    assert case.run("--vlm") == 0

    assert loud.calls == []
    assert case.answer() == FALLBACK_OPEN


# --------------------------------------------------------------- no CUDA --
#
# THE NO-GPU PATH IS A CORRECTNESS REQUIREMENT (item 3), not a nicety: the
# shipped weights are 4-bit NF4, a bitsandbytes/CUDA-only format, so on a
# No-GPU deployment draw the VLM is structurally unable to run. This must be
# a skip, decided BEFORE any transformers import, never an attempt that is
# merely caught.

def test_no_cuda_declares_the_handle_unavailable(monkeypatch):
    monkeypatch.setattr(inference.torch.cuda, "is_available", lambda: False)

    handle = inference.EvidenceVlmHandle("/any/dir", 5, 3, inference.log)

    assert handle.available() is False


def test_cuda_present_declares_the_handle_available(monkeypatch):
    monkeypatch.setattr(inference.torch.cuda, "is_available", lambda: True)

    handle = inference.EvidenceVlmHandle("/any/dir", 5, 3, inference.log)

    assert handle.available() is True


def test_no_cuda_means_sample_is_never_called(monkeypatch, capsys):
    """The gate lives in try_vlm_result, checked before `sample()` -- not
    inside `sample()` itself, and not something a caller could bypass by
    calling it directly with a badly-behaved fake."""
    monkeypatch.setattr(inference.torch.cuda, "is_available", lambda: False)

    class BoomIfSampled(object):
        arbiter_mode = None

        def available(self):
            return inference.torch.cuda.is_available()

        def sample(self, video, question, perception, budget_seconds=None):
            raise AssertionError("sample() must not be called with no CUDA")

    result = inference.try_vlm_result("video.mp4", "q", {}, BoomIfSampled(), [])

    assert result is None
    assert "no CUDA device available" in capsys.readouterr().err


def test_a_full_run_with_vlm_enabled_but_no_cuda_still_answers(
        case, monkeypatch):
    """The real `EvidenceVlmHandle`, the real gate, no CUDA -- exactly the
    No-GPU deployment draw the challenge documents as possible. Must exit 0
    with the same answer the router alone would have given, never a hang or
    an exception reaching the top level."""
    monkeypatch.setattr(inference.torch.cuda, "is_available", lambda: False)
    case.ask("Is the camera being moved?")

    assert case.run() == 0
    without_vlm = case.answer()

    assert case.run("--vlm") == 0
    with_vlm_no_cuda = case.answer()

    assert with_vlm_no_cuda == without_vlm


# --------------------------------------------------- the model-dir pin ----

def test_the_model_dir_is_pinned_for_the_call_and_restored_after(monkeypatch):
    """`_vlm_model_dir_pinned` must not leave `evidence_vlm.call_vlm`
    permanently repointed -- a later question (or a later test in the same
    process) must see the ORIGINAL function again."""
    import surgvu.evidence_vlm as evidence_vlm_module

    original = evidence_vlm_module.call_vlm
    seen = {}

    def fake_call_vlm(frames, question, context, temperature=None,
                      model_dir=None, device=None):
        seen["model_dir"] = model_dir
        return "an answer"

    monkeypatch.setattr(evidence_vlm_module, "call_vlm", fake_call_vlm)
    # `_vlm_model_dir_pinned` reads `evidence_vlm.call_vlm` at entry, so it
    # wraps the monkeypatched fake here -- proving the pin composes with
    # whatever the module's current `call_vlm` is, not a cached reference
    # taken at inference.py's import time.
    with inference._vlm_model_dir_pinned("/pinned/dir"):
        assert evidence_vlm_module.call_vlm is not fake_call_vlm
        evidence_vlm_module.call_vlm(["frame"], "q", {}, temperature=0.4)

    assert seen["model_dir"] == "/pinned/dir"
    assert evidence_vlm_module.call_vlm is fake_call_vlm

    monkeypatch.undo()
    assert evidence_vlm_module.call_vlm is original


# ---------------------------------------------------------------------------
# The per-case VLM budget. evidence_vlm anchors its deadline at VLM start with
# no knowledge of what the case already spent; these pin the arithmetic that
# gives it that knowledge.
# ---------------------------------------------------------------------------

def test_remaining_vlm_budget_shrinks_as_the_case_burns_time(monkeypatch):
    """A slow router must not be able to push the case past 600 s.

    Real numbers: validation 9698732 measured 317 s for the router path alone
    on a contended node. A flat 240 s VLM budget on top of that is 557 s plus
    teardown, against a hard 600 s -- and a missing response scores 0, worse
    than any wrong answer.
    """
    monkeypatch.setattr(inference, "_CASE_STARTED", 1000.0)
    monkeypatch.setattr(inference.time, "time", lambda: 1000.0 + 317.0)
    budget = inference.remaining_vlm_budget()
    assert budget == pytest.approx(540.0 - 317.0 - 45.0)
    assert 317.0 + budget + 45.0 <= 600.0


def test_remaining_vlm_budget_grows_on_a_fast_node_up_to_the_cap(monkeypatch):
    """The same arithmetic must also USE the window, not just protect it.

    Validation 9698714 measured 17-22 s per case on an uncontended node. The
    old flat 240 s left over half the allowance unspent; adaptive_confidence_
    sample turns extra budget into extra self-consistency samples.
    """
    monkeypatch.setattr(inference, "_CASE_STARTED", 1000.0)
    monkeypatch.setattr(inference.time, "time", lambda: 1000.0 + 20.0)
    budget = inference.remaining_vlm_budget()
    assert budget == pytest.approx(inference.VLM_MAX_BUDGET_SECONDS)
    assert budget > 240.0, "a fast case should get MORE than the old constant"


def test_remaining_vlm_budget_never_goes_negative(monkeypatch):
    """Clamped at 0, because the caller treats the value as a duration."""
    monkeypatch.setattr(inference, "_CASE_STARTED", 1000.0)
    monkeypatch.setattr(inference.time, "time", lambda: 1000.0 + 5000.0)
    assert inference.remaining_vlm_budget() == 0.0


def test_elapsed_this_case_is_zero_when_main_never_ran(monkeypatch):
    """Direct callers (tests, probes) get the full budget rather than a crash."""
    monkeypatch.setattr(inference, "_CASE_STARTED", None)
    assert inference.elapsed_this_case() == 0.0


def test_try_vlm_result_declines_when_too_little_budget_remains(monkeypatch):
    """A generation cut off after a few tokens is not a cheap partial answer --
    it is a truncated string the arbiter might prefer over the router's correct
    one. Declining is the safe move, and must be logged rather than silent."""
    monkeypatch.setattr(inference, "_CASE_STARTED", 1000.0)
    monkeypatch.setattr(inference.time, "time", lambda: 1000.0 + 530.0)

    class _Vlm:
        arbiter_mode = None
        def available(self):
            return True
        def sample(self, *a, **k):
            raise AssertionError("must not start the VLM with no budget left")

    assert inference.try_vlm_result("v.mp4", "q?", {}, _Vlm(), []) is None
