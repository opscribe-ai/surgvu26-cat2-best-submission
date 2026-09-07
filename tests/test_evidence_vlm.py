"""Tests for the ported adaptive confidence sampler.

Deliberately torch-free: the agreement/routing logic is what decides which
answer ships, and it must be testable on this login node, where `torch` is
not installed at all (confirmed: `import torch` fails here). If
`surgvu.evidence_vlm` imported torch at module scope, EVERY test below would
fail at collection, not just the ones that exercise it -- that is the
strongest available proof the module-scope import discipline holds, on top
of the explicit source-inspection tests further down.

Frame decoding (`sample_frames`) and the model call (`call_vlm`) are
replaced with fakes via `monkeypatch.setattr` on the module's own globals,
mirroring how `tests/test_vlm.py` replaces `QwenVlmFallback._generate` --
the only difference is this module is written as functions rather than a
class, so the seam is a module attribute rather than a bound method.
"""
import json

import pytest

from surgvu import evidence_vlm
from surgvu.evidence_vlm import ConfidenceResult, adaptive_confidence_sample, route


# --------------------------------------------------------- what must be gone

def test_no_overlay_prompt_under_any_name():
    """The rules prohibit reading the UI band; the ported OVERLAY_PROMPT
    instructed exactly that. It must not exist here, under this name or a
    disguised one -- checked directly rather than trusted."""
    assert not hasattr(evidence_vlm, "OVERLAY_PROMPT")
    for name in dir(evidence_vlm):
        assert "overlay" not in name.lower()


def test_no_opscribe_pipeline_import_anywhere_in_the_module_source():
    """SurgVU26 Cat 2 is independent of OpScribe: its own containers, its
    own venv, never OpScribe's .sif or pypkgs. A single surviving import
    would reintroduce that dependency invisibly. The docstring is allowed to
    NAME opscribe_pipeline in prose, explaining what was severed and why --
    what must never appear is an actual import of it."""
    import inspect
    source = inspect.getsource(evidence_vlm)
    assert "import opscribe_pipeline" not in source
    assert "from opscribe_pipeline" not in source


def test_is_correct_was_not_ported():
    """Substring matching (`pred == a_clean or pred in a_clean`) inflates
    accuracy; this project scores with surgvu.scoring.Scorer instead."""
    assert not hasattr(evidence_vlm, "is_correct")


def test_parse_question_type_was_not_ported():
    """The router already ships 11 intents; a second, cruder RECORDS/
    LOOK_HARDER classifier in front of it is not wanted."""
    assert not hasattr(evidence_vlm, "parse_question_type")


# ------------------------------------------------ documentation is a contract

def test_module_docstring_records_the_hypothesis_status_of_the_temperature():
    doc = (evidence_vlm.__doc__ or "").lower()
    assert "hypothesis" in doc
    assert "n=4" in doc or "4 real" in doc


def test_module_docstring_keeps_the_qualitative_case_finding():
    """The three case IDs and the "agreement is not calibration" framing are
    the one finding the plan explicitly requires survive the port,
    independent of whatever happens to the temperature default."""
    doc = evidence_vlm.__doc__ or ""
    for case_id in ("case122", "case127", "case130"):
        assert case_id in doc
    assert "agreement is not calibration" in doc.lower()


def test_default_temperature_keeps_her_value_but_is_not_asserted_as_correct():
    assert evidence_vlm.DEFAULT_SAMPLING_TEMPERATURE == 0.4


# ----------------------------------------------------------- normalize_answer

@pytest.mark.parametrize("raw,expected", [
    ("Yes", "yes"),
    ("yes.", "yes"),
    ("  Yes.  ", "yes"),
    ("NO", "no"),
])
def test_normalize_answer_folds_case_and_trailing_period(raw, expected):
    assert evidence_vlm.normalize_answer(raw) == expected


def test_differently_worded_answers_do_not_normalize_equal():
    assert evidence_vlm.normalize_answer("Yes") != evidence_vlm.normalize_answer("No")


# --------------------------------------------------------------- route()

def _result(confidence, answer="Yes", n_calls_used=2):
    return ConfidenceResult(answer=answer, confidence=confidence,
                            n_calls_used=n_calls_used, all_answers=[answer] * n_calls_used,
                            agreed=True)


def test_high_confidence_routes_to_accept():
    decision = route(_result(0.9), confidence_threshold=0.66)
    assert decision["decision"] == evidence_vlm.ACCEPT
    assert decision["answer"] == "Yes"
    assert decision["calls_used"] == 2


def test_low_confidence_routes_to_escalate():
    decision = route(_result(0.5), confidence_threshold=0.66)
    assert decision["decision"] == evidence_vlm.ESCALATE
    assert decision["partial_answer"] == "Yes"


def test_confidence_exactly_at_threshold_accepts():
    """`>=`, not `>` -- matches the ported comparison exactly."""
    decision = route(_result(0.66), confidence_threshold=0.66)
    assert decision["decision"] == evidence_vlm.ACCEPT


def test_route_rejects_a_non_confidence_result():
    with pytest.raises(TypeError):
        route({"confidence": 0.9, "answer": "Yes", "n_calls_used": 1})


def test_route_output_is_strict_json_and_versioned():
    decision = route(_result(0.9))
    json.loads(json.dumps(decision, allow_nan=False))
    assert decision["version"] == evidence_vlm.EVIDENCE_VLM_VERSION


def test_confidence_result_to_dict_is_strict_json_and_versioned():
    result = _result(0.75, answer="Cadiere forceps", n_calls_used=3)
    record = result.to_dict()
    json.loads(json.dumps(record, allow_nan=False))
    assert record["version"] == evidence_vlm.EVIDENCE_VLM_VERSION
    assert record["confidence"] == pytest.approx(0.75)
    assert record["all_answers"] == ["Cadiere forceps"] * 3


# --------------------------------------------------- adaptive_confidence_sample

def _install_fakes(monkeypatch, answers, frames=("frame",)):
    """Replace `sample_frames`/`call_vlm` with counters over a fixed script.

    `answers` is the scripted sequence of raw model outputs, one per call;
    calling past the end of the script is a test bug, not a code path, so it
    raises rather than looping.
    """
    calls = {"sample_frames": [], "call_vlm": []}
    remaining = list(answers)

    def fake_sample_frames(video, n=evidence_vlm.DEFAULT_FRAMES_PER_CALL):
        calls["sample_frames"].append({"video": video, "n": n})
        return frames

    def fake_call_vlm(frames_arg, question, context, temperature=None, **kwargs):
        calls["call_vlm"].append({
            "frames": frames_arg, "question": question,
            "context": context, "temperature": temperature,
            "deadline": kwargs.get("deadline"),
        })
        if not remaining:
            raise AssertionError("call_vlm invoked more times than scripted")
        return remaining.pop(0)

    monkeypatch.setattr(evidence_vlm, "sample_frames", fake_sample_frames)
    monkeypatch.setattr(evidence_vlm, "call_vlm", fake_call_vlm)
    return calls


def test_two_agreeing_samples_accept_after_exactly_two_calls(monkeypatch):
    calls = _install_fakes(monkeypatch, ["Yes", "Yes.", "No"])

    result = adaptive_confidence_sample(
        {"path": "case.mp4"}, "Is the tool present?", {})

    assert result.answer == "yes"
    assert result.confidence == pytest.approx(1.0)
    assert result.agreed is True
    assert result.n_calls_used == 2
    assert len(calls["call_vlm"]) == 2   # never reaches the scripted 3rd answer


def test_frames_are_decoded_exactly_once_per_run(monkeypatch):
    """The design decision this port makes explicitly: reuse one frame set
    across every sample in a run, rather than re-decoding per attempt --
    see sample_frames's docstring on why (no jitter primitive in
    decode_clip, so a retry would just be an expensive no-op)."""
    calls = _install_fakes(monkeypatch, ["Yes", "No", "Yes"])

    adaptive_confidence_sample({"path": "case.mp4"}, "Q?", {}, max_samples=3)

    assert len(calls["sample_frames"]) == 1


def test_disagreement_escalates_to_a_third_sample_then_majority_votes(
        monkeypatch):
    calls = _install_fakes(monkeypatch, ["Yes", "No", "Yes"])

    result = adaptive_confidence_sample(
        {"path": "case.mp4"}, "Is the tool present?", {}, max_samples=3)

    assert result.answer == "yes"
    assert result.n_calls_used == 3
    assert result.confidence == pytest.approx(2 / 3)
    assert result.agreed is False
    assert len(calls["call_vlm"]) == 3


def test_a_single_sample_cannot_self_verify_even_though_it_computes_full_agreement(
        monkeypatch):
    """max_samples=1 is a real edge, not a special case bolted on: the loop
    never reaches the len(answers) >= 2 check, so `agreed` stays False even
    though `top_count / len(normalized)` is 1/1 = 1.0. A caller must read
    `agreed`, not just `confidence`, to know whether the model had any
    chance to disagree with itself."""
    _install_fakes(monkeypatch, ["Yes"])

    result = adaptive_confidence_sample(
        {"path": "case.mp4"}, "Q?", {}, max_samples=1)

    assert result.n_calls_used == 1
    assert result.confidence == pytest.approx(1.0)
    assert result.agreed is False


def test_full_disagreement_across_the_cap_still_returns_a_usable_answer(
        monkeypatch):
    """Three distinct answers, none repeated: must not crash, and must still
    produce an answer plus a confidence that reflects the weak agreement."""
    _install_fakes(monkeypatch, ["Yes", "No", "Maybe"])

    result = adaptive_confidence_sample(
        {"path": "case.mp4"}, "Q?", {}, max_samples=3, agreement_threshold=1.0)

    assert result.n_calls_used == 3
    assert result.confidence == pytest.approx(1 / 3)
    assert result.agreed is False
    assert result.answer in {"yes", "no", "maybe"}


def test_agreement_threshold_below_one_can_accept_a_majority_early(
        monkeypatch):
    """With a 0.6 threshold, two disagreeing samples (agreement 0.5) must
    NOT stop early; a third landing 2-of-3 (agreement 0.667) must."""
    calls = _install_fakes(monkeypatch, ["Yes", "No", "Yes", "No"])

    result = adaptive_confidence_sample(
        {"path": "case.mp4"}, "Q?", {}, max_samples=4, agreement_threshold=0.6)

    assert result.n_calls_used == 3
    assert result.confidence == pytest.approx(2 / 3)
    assert result.agreed is True
    assert len(calls["call_vlm"]) == 3


def test_sampling_temperature_reaches_call_vlm_on_every_call(monkeypatch):
    calls = _install_fakes(monkeypatch, ["Yes", "Yes"])

    adaptive_confidence_sample(
        {"path": "case.mp4"}, "Q?", {}, sampling_temperature=0.7)

    assert all(c["temperature"] == 0.7 for c in calls["call_vlm"])


def test_question_and_context_reach_call_vlm_unchanged(monkeypatch):
    calls = _install_fakes(monkeypatch, ["Yes", "Yes"])
    context = {"robot_tools": ["needle driver"]}

    adaptive_confidence_sample(
        {"path": "case.mp4"}, "Is a needle driver mounted?", context)

    assert calls["call_vlm"][0]["question"] == "Is a needle driver mounted?"
    assert calls["call_vlm"][0]["context"] == context


def test_all_answers_are_recorded_in_call_order_not_just_the_winner(
        monkeypatch):
    _install_fakes(monkeypatch, ["Yes", "No", "Yes"])

    result = adaptive_confidence_sample(
        {"path": "case.mp4"}, "Q?", {}, max_samples=3)

    assert result.all_answers == ["Yes", "No", "Yes"]


def test_video_and_frames_per_call_reach_sample_frames(monkeypatch):
    calls = _install_fakes(monkeypatch, ["Yes", "Yes"])
    video = {"path": "case131.mp4", "case_id": "case131"}

    adaptive_confidence_sample(video, "Q?", {}, frames_per_call=7)

    assert calls["sample_frames"][0]["video"] == video
    assert calls["sample_frames"][0]["n"] == 7


# --------------------------------------------------- build_sampling_prompt

def test_prompt_never_contains_anything_derived_from_the_ui_band():
    """No amount of context can smuggle overlay-derived text in through
    this function -- it only ever renders `robot_tools`/`task_description`/
    the question, none of which originate from the UI band. Guarded here
    the same way tests/test_vlm.py guards `build_prompt`."""
    prompt = evidence_vlm.build_sampling_prompt(
        "What tool is mounted?",
        {"robot_tools": ["needle driver"], "task_description": "Suturing"})
    assert "overlay" not in prompt.lower()
    assert "numbered" not in prompt.lower()


def test_prompt_carries_the_question():
    prompt = evidence_vlm.build_sampling_prompt("What organ is shown?", {})
    assert "What organ is shown?" in prompt


def test_prompt_folds_in_context_when_present():
    prompt = evidence_vlm.build_sampling_prompt(
        "Q?", {"robot_tools": ["stapler"], "task_description": "Clipping"})
    assert "stapler" in prompt
    assert "Clipping" in prompt


def test_prompt_omits_context_lines_when_context_is_empty():
    prompt = evidence_vlm.build_sampling_prompt("Q?", {})
    assert "Tools mounted" not in prompt
    assert "Procedure context" not in prompt


# --------------------------------------------------- the wall-clock deadline
#
# `_clock` is looked up from the module's own globals at call time (the same
# seam `sample_frames`/`call_vlm` already use), so a test can
# `monkeypatch.setattr(evidence_vlm, "_clock", fake)` and get a fully
# deterministic, non-sleeping clock -- no real elapsed time is ever load-
# bearing below.

def _fake_clock(times):
    """A callable that returns each value in `times` in turn, then repeats
    the last value forever (so a call past the end of the script is a flat
    line, not an IndexError)."""
    values = list(times)

    def _clock():
        value = values[0] if len(values) == 1 else values.pop(0)
        return value

    return _clock


def test_default_budget_seconds_leaves_margin_against_measured_pipeline_time():
    """Locks in the chosen default against the numbers its docstring cites:
    the shipped image's worst measured CPU-only time (196s) plus this
    budget must stay comfortably under the 600s per-case ceiling. This is a
    guard against silently raising the default past the point where the
    justification in the docstring still holds -- it is not a test of the
    sampling logic."""
    assert evidence_vlm.DEFAULT_BUDGET_SECONDS > 0
    assert 196.0 + evidence_vlm.DEFAULT_BUDGET_SECONDS < 600.0


def test_deadline_is_checked_between_samples_and_stops_asking_for_more(
        monkeypatch):
    """Three disagreeing answers would normally run all the way to
    max_samples=3 (see test_disagreement_escalates_to_a_third_sample_...).
    Here the fake clock reports the deadline as already passed by the time
    the loop is about to request the THIRD sample, so only 2 calls happen."""
    calls = _install_fakes(monkeypatch, ["Yes", "No", "Yes"])
    # start=0.0; deadline = 0.0 + 10.0 = 10.0
    # checked before sample 1 (0.0, ok), before sample 2 (1.0, ok),
    # before sample 3 (11.0, past deadline) -> stop.
    monkeypatch.setattr(evidence_vlm, "_clock",
                        _fake_clock([0.0, 0.0, 1.0, 11.0, 11.0]))

    result = adaptive_confidence_sample(
        {"path": "case.mp4"}, "Q?", {}, max_samples=3,
        budget_seconds=10.0)

    assert len(calls["call_vlm"]) == 2
    assert result.n_calls_used == 2
    assert result.curtailed is True
    assert result.agreed is False


def test_deadline_exhausted_before_any_sample_returns_none(monkeypatch):
    """Zero samples collected -> None, so the caller (arbiter/router) falls
    through to its own answer rather than being handed a hollow result."""
    calls = _install_fakes(monkeypatch, ["Yes", "Yes"])
    # start=0.0; deadline = 0.0 + 5.0 = 5.0; checked before sample 1 at 6.0.
    monkeypatch.setattr(evidence_vlm, "_clock", _fake_clock([0.0, 6.0]))

    result = adaptive_confidence_sample(
        {"path": "case.mp4"}, "Q?", {}, max_samples=3, budget_seconds=5.0)

    assert result is None
    assert calls["call_vlm"] == []


def test_deadline_exhausted_after_exactly_one_sample_still_returns_it(
        monkeypatch):
    """One sample collected before the budget ran out must come back as a
    real, usable result -- not None (that would throw away the one answer
    actually collected) and not silently marked as if it agreed with
    itself."""
    calls = _install_fakes(monkeypatch, ["Yes", "No"])
    # deadline check passes before sample 1 (0.0), fails before sample 2 (6.0).
    monkeypatch.setattr(evidence_vlm, "_clock", _fake_clock([0.0, 0.0, 6.0]))

    result = adaptive_confidence_sample(
        {"path": "case.mp4"}, "Q?", {}, max_samples=3, budget_seconds=5.0)

    assert result is not None
    assert len(calls["call_vlm"]) == 1
    assert result.n_calls_used == 1
    assert result.answer == "yes"
    assert result.confidence == pytest.approx(1.0)
    assert result.curtailed is True
    assert result.agreed is False   # a single sample never self-verified


def test_agreement_reached_before_the_deadline_is_never_marked_curtailed(
        monkeypatch):
    """Two samples agree on the second call; the deadline has, by
    construction, already passed by then -- if it had kept sampling, the
    3rd call would be refused. Reaching agreement first must win: the
    result is `agreed=True, curtailed=False`, not the other way round."""
    calls = _install_fakes(monkeypatch, ["Yes", "Yes", "No"])
    # deadline = 0.0 + 1.0 = 1.0. Checked before sample 1 (0.0, ok) and
    # before sample 2 (0.5, ok); agreement is found right after sample 2
    # (clock never consulted again), even though a 3rd check would fail.
    monkeypatch.setattr(evidence_vlm, "_clock",
                        _fake_clock([0.0, 0.0, 0.5, 99.0]))

    result = adaptive_confidence_sample(
        {"path": "case.mp4"}, "Q?", {}, max_samples=3, budget_seconds=1.0)

    assert len(calls["call_vlm"]) == 2
    assert result.agreed is True
    assert result.curtailed is False


def test_normal_completion_within_budget_is_never_marked_curtailed(
        monkeypatch):
    """The ordinary exhaust-max_samples-without-agreement path (already
    covered functionally by
    test_full_disagreement_across_the_cap_still_returns_a_usable_answer)
    must not pick up `curtailed=True` merely because the loop ran to its
    cap -- curtailed means the BUDGET cut it short, not the sample cap."""
    _install_fakes(monkeypatch, ["Yes", "No", "Maybe"])
    monkeypatch.setattr(evidence_vlm, "_clock", _fake_clock([0.0] * 10))

    result = adaptive_confidence_sample(
        {"path": "case.mp4"}, "Q?", {}, max_samples=3,
        agreement_threshold=1.0, budget_seconds=600.0)

    assert result.n_calls_used == 3
    assert result.curtailed is False


def test_call_vlm_receives_the_computed_deadline(monkeypatch):
    """`call_vlm` is hand the absolute deadline (start + budget), not the
    budget itself, so a single generation call (see the module docstring's
    `_deadline_stopping_criteria`) can bound itself with a plain
    `time.time() >= deadline` check without knowing when the run started."""
    calls = _install_fakes(monkeypatch, ["Yes", "Yes"])
    monkeypatch.setattr(evidence_vlm, "_clock", _fake_clock([100.0] * 10))

    adaptive_confidence_sample(
        {"path": "case.mp4"}, "Q?", {}, budget_seconds=50.0)

    assert calls["call_vlm"][0]["deadline"] == pytest.approx(150.0)
    assert calls["call_vlm"][1]["deadline"] == pytest.approx(150.0)


def test_elapsed_seconds_reflects_the_clock_not_a_constant(monkeypatch):
    """A future measurement needs to be able to see how close to the budget
    a real case ran -- this reports the actual elapsed time on the result,
    not a hardcoded value."""
    _install_fakes(monkeypatch, ["Yes", "Yes"])
    # start=1000.0; second sample's post-call reading is 1017.5.
    monkeypatch.setattr(evidence_vlm, "_clock",
                        _fake_clock([1000.0, 1000.0, 1005.0, 1017.5]))

    result = adaptive_confidence_sample(
        {"path": "case.mp4"}, "Q?", {}, budget_seconds=600.0)

    assert result.elapsed_seconds == pytest.approx(17.5)


def test_confidence_result_default_curtailed_is_false():
    """Additive field, additive default: a caller (arbiter.py,
    scripts/inference.py) constructing/reading a `ConfidenceResult` without
    ever knowing about the budget feature must see the pre-existing,
    non-curtailed behaviour."""
    result = ConfidenceResult(answer="Yes", confidence=0.9, n_calls_used=1)
    assert result.curtailed is False
    assert result.elapsed_seconds == pytest.approx(0.0)


def test_confidence_result_to_dict_carries_curtailed_and_elapsed_seconds():
    result = ConfidenceResult(answer="Yes", confidence=0.9, n_calls_used=1,
                              curtailed=True, elapsed_seconds=123.4)
    record = result.to_dict()
    json.loads(json.dumps(record, allow_nan=False))
    assert record["curtailed"] is True
    assert record["elapsed_seconds"] == pytest.approx(123.4)


def test_budget_seconds_parameter_overrides_the_default(monkeypatch):
    """A caller can tighten the budget below the default; this exercises
    that the parameter, not just the module constant, is what the deadline
    is computed from."""
    calls = _install_fakes(monkeypatch, ["Yes", "No", "Yes"])
    monkeypatch.setattr(evidence_vlm, "_clock",
                        _fake_clock([0.0, 0.0, 100.0, 100.0]))

    result = adaptive_confidence_sample(
        {"path": "case.mp4"}, "Q?", {}, max_samples=3, budget_seconds=1.0)

    assert len(calls["call_vlm"]) == 1
    assert result.curtailed is True


# ---------------------------------------------------------------------------
# release_models: the judge is a SECOND model, and both resident is 11.45 GiB
# of weights on a 16 GiB T4 before prefill activations.
# ---------------------------------------------------------------------------

def test_release_models_empties_the_cache():
    evidence_vlm._MODEL_CACHE[("/a", "cuda")] = ("model", "processor")
    evidence_vlm._MODEL_CACHE[("/b", "cuda")] = ("model", "processor")
    evidence_vlm.release_models()
    assert evidence_vlm._MODEL_CACHE == {}


def test_release_models_can_keep_one_directory():
    """The judge keeps its OWN weights while dropping the answering VLM's --
    it is about to use them."""
    evidence_vlm._MODEL_CACHE.clear()
    evidence_vlm._MODEL_CACHE[("/vlm", "cuda")] = ("v", "p")
    evidence_vlm._MODEL_CACHE[("/judge", "cuda")] = ("j", "p")
    evidence_vlm.release_models(keep="/judge")
    assert list(evidence_vlm._MODEL_CACHE) == [("/judge", "cuda")]
    evidence_vlm._MODEL_CACHE.clear()


def test_release_models_is_safe_on_an_empty_cache():
    evidence_vlm._MODEL_CACHE.clear()
    evidence_vlm.release_models()
    assert evidence_vlm._MODEL_CACHE == {}


def test_temperature_zero_selects_greedy_not_a_degenerate_sampler():
    """transformers REJECTS temperature=0.0 with do_sample=True ("has to be a
    strictly positive float") -- it killed the judge probe (cluster 9707948).
    A caller asking for zero wants determinism, which is do_sample=False.

    The judge needs exactly that: choosing between two given answers should
    not vary run to run on identical input."""
    kwargs = evidence_vlm.generation_kwargs(0.0)
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs, (
        "transformers warns and ignores temperature when do_sample is False; "
        "passing it anyway is noise that looks like a bug")


def test_a_positive_temperature_still_samples():
    """adaptive_confidence_sample derives its entire confidence signal from
    variance across samples -- greedy there would report confidence 1.0 on
    every case, which is not a measurement."""
    kwargs = evidence_vlm.generation_kwargs(0.4)
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.4


def test_a_negative_temperature_is_greedy_not_an_error():
    assert evidence_vlm.generation_kwargs(-1.0)["do_sample"] is False
