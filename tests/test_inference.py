"""The submission entrypoint: /input -> one JSON-encoded answer in /output.

Two properties are graded and everything here is about one of them.

THE ENCODING. Both files hold JSON-encoded STRINGS. The response `"Yes"` is
four bytes including the quotes; bare `Yes` is malformed JSON and fails the
case whatever the answer says.

NEVER FAILING TO ANSWER. Measured: a wrong yes/no still scores 0.7015, a
plausible generic answer on an open question 0.35-0.48, and an empty or
missing response nothing at all -- an empty string crashes the scorer
outright. So every failure mode below is asserted to still leave a valid,
non-empty response file behind, and to exit 0 while doing it. A crash that
writes nothing converts a ~0.7 into a 0.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import inference                                                    # noqa: E402
from surgvu import router                                           # noqa: E402
from surgvu.router import (                                         # noqa: E402
    FALLBACK_OPEN, FALLBACK_POLAR, PROCEDURE_ANSWER,
)
from surgvu.taxonomy import TASK_CLASSES, TOOL_CLASSES              # noqa: E402
from surgvu.train import save_checkpoint                            # noqa: E402

IMAGE_SIZE = 64


# ---------------------------------------------------------------- fixtures

class Tiny(torch.nn.Module):
    """A stand-in backbone: (B, 3, H, W) -> (B, n_outputs) logits.

    NOT a shortcut around the loader. `load_expert` still torch.loads the
    file, sizes the head from `meta["classes"]`, and load_state_dict's strict
    check still has to pass, so the checkpoint contract is exercised in full.
    What it removes is 81 MB of EfficientNetV2-S weights per checkpoint per
    test, which on shared storage dominated the run by an order of magnitude
    and tested nothing this file is about. The real backbone is exercised
    end-to-end against the real checkpoints outside pytest.
    """

    def __init__(self, n_outputs):
        super().__init__()
        self.head = torch.nn.Linear(3, n_outputs)

    def forward(self, batch):
        return self.head(batch.mean(dim=(2, 3)))


@pytest.fixture(autouse=True)
def tiny_backbone(monkeypatch):
    import surgvu.perceive as perceive_module

    monkeypatch.setattr(perceive_module, "build_model",
                        lambda n_outputs, backbone=None, pretrained=True: Tiny(n_outputs))


@pytest.fixture(scope="session")
def checkpoints(tmp_path_factory):
    """Checkpoints in the real format, written from the stand-in backbone."""
    directory = tmp_path_factory.mktemp("models")
    save_checkpoint(
        directory / "tools.pt",
        Tiny(len(TOOL_CLASSES)),
        {"classes": list(TOOL_CLASSES), "backbone": "efficientnet_v2_s",
         "image_size": IMAGE_SIZE, "frames_per_window": 8,
         "thresholds": [0.5] * len(TOOL_CLASSES)})
    save_checkpoint(
        directory / "task.pt",
        Tiny(len(TASK_CLASSES)),
        {"classes": list(TASK_CLASSES), "backbone": "efficientnet_v2_s",
         "image_size": IMAGE_SIZE, "frames_per_window": 8})
    return directory


TOOLS_SHA = "0" * 64


def _serving_block(values, sha256=TOOLS_SHA):
    """The deliberate serving-threshold override, as the config carries it.

    `provenance.checkpoint_sha256` is the tie to the weights these cuts were
    tuned on: thresholds tuned against one checkpoint are meaningless against
    another, and the config's own `sha256` is what says which is bound.
    """
    return {"values": list(values),
            "by_class": dict(zip(TOOL_CLASSES, values)),
            "provenance": {"checkpoint_sha256": sha256}}


def _config(checkpoints, thresholds=None, tool_image_size=IMAGE_SIZE,
            task_image_size=IMAGE_SIZE, frames=4, size=IMAGE_SIZE,
            tools_checkpoint=None, task_checkpoint=None,
            serving_thresholds=None):
    thresholds = list(thresholds or [0.5] * len(TOOL_CLASSES))
    return {
        "schema_version": 2,
        "decode": {"frames": frames, "size": size},
        "experts": {
            "tools": {
                "role": "tools", "activation": "sigmoid",
                "checkpoint": str(tools_checkpoint
                                  or checkpoints / "tools.pt"),
                "checkpoint_name": "tools.pt", "source": "primary",
                "sha256": TOOLS_SHA,
                "backbone": "efficientnet_v2_s", "image_size": tool_image_size,
                "frames_per_window": 8, "classes": list(TOOL_CLASSES),
                "thresholds": thresholds,
                "serving_thresholds": serving_thresholds, "metrics": {},
            },
            "task": {
                "role": "task", "activation": "softmax",
                "checkpoint": str(task_checkpoint or checkpoints / "task.pt"),
                "checkpoint_name": "task.pt", "source": "primary",
                "backbone": "efficientnet_v2_s", "image_size": task_image_size,
                "frames_per_window": 8, "classes": list(TASK_CLASSES),
                "thresholds": None, "metrics": {},
            },
        },
    }


def _write_video(path, n=8, height=64, width=64, fps=1.0):
    frames = [np.full((height, width, 3), 20 + 7 * i, dtype=np.uint8)
              for i in range(n)]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(fps), (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


class Case:
    """One /input + /output pair, plus the config the container would read."""

    def __init__(self, root, config):
        self.root = Path(root)
        self.input = self.root / "input"
        self.output = self.root / "output"
        self.input.mkdir(parents=True, exist_ok=True)
        self.config_path = self.root / "perception.json"
        self.config_path.write_text(json.dumps(config), encoding="utf-8")
        self.video = self.input / inference.VIDEO_NAME
        self.question_path = self.input / inference.QUESTION_NAME
        self.response = self.output / inference.RESPONSE_NAME

    def ask(self, question):
        self.question_path.write_text(json.dumps(question), encoding="utf-8")
        return self

    def run(self, *extra):
        return inference.main(["--input-dir", str(self.input),
                               "--output-dir", str(self.output),
                               "--config", str(self.config_path),
                               "--device", "cpu", *extra])

    def answer(self):
        return json.loads(self.response.read_text(encoding="utf-8"))


@pytest.fixture
def case(tmp_path, checkpoints):
    made = Case(tmp_path, _config(checkpoints))
    _write_video(made.video)
    return made.ask("Is the camera being moved?")


def _assert_answered(case):
    """The invariant every failure mode below must preserve."""
    raw = case.response.read_bytes()
    answer = json.loads(raw.decode("utf-8"))
    assert isinstance(answer, str)
    assert answer.strip(), "an empty answer crashes the scorer outright"
    return answer


# ------------------------------------------------------------ the encoding

def test_the_response_is_a_json_encoded_string_with_its_quotes(case):
    """`"Yes"` is four bytes. Bare `Yes` is malformed JSON and fails."""
    assert case.run() == 0

    assert case.response.read_bytes() == b'"Yes"'
    assert json.loads(case.response.read_text(encoding="utf-8")) == "Yes"


def test_the_question_is_json_decoded_not_read_raw(case, monkeypatch):
    """The file holds `"Are there forceps ...?"` WITH quotes. Read raw, the
    leading quote reaches the router and the question no longer opens with a
    polar auxiliary."""
    seen = []
    # PATCHED ON THE ROUTER, NOT ON `inference`. Commit ae4ef7c ("Wire --vlm
    # through the arbiter") moved the live call: `arbiter.arbitrate` invokes
    # `router.answer_question(question, perception)` itself (arbiter.py), and
    # `inference.answer_question` is no longer on the answering path at all.
    # Patching the old name left this test monkeypatching a seam nothing
    # calls -- it kept "passing" in every run that skipped this file and
    # failed silently as coverage the moment it ran. The invariant it guards
    # (the question is JSON-decoded, so the router sees a string that still
    # opens with a polar auxiliary) is real; the seam had simply moved.
    monkeypatch.setattr(router, "answer_question",
                        lambda question, perception: seen.append(question) or "Yes")
    case.ask("Are there forceps being used here?").run()

    assert seen == ["Are there forceps being used here?"]


def test_the_output_directory_is_created_if_it_does_not_exist(case):
    assert not case.output.exists()

    case.run()

    assert case.response.exists()


def test_the_input_and_output_roots_are_overridable(tmp_path, checkpoints):
    """Nothing may require the literal /input and /output to exist, or none
    of this is testable off the deployment instance."""
    made = Case(tmp_path / "elsewhere", _config(checkpoints))
    _write_video(made.video)
    made.ask("Is the camera being moved?")

    assert made.run() == 0
    assert made.response.parent == tmp_path / "elsewhere" / "output"
    assert _assert_answered(made) == "Yes"


# ------------------------------------------------------------- the wiring

def _stub_predict_window(seen, tool_probs, task_probs):
    """Stands in for inference.predict_window_FRAMES, not predict_window.

    The serving path stopped calling `predict_window` on 2026-08-12, when the
    perception path learned to ensemble: it now calls `predict_window_frames`
    and reduces afterwards, so that two models can be averaged at the FRAME
    level before aggregation. These tests kept patching the old name, and
    monkeypatch.setattr on a missing attribute raises -- so all 23 of them have
    been failing ever since, unnoticed, because the suite cannot run outside
    the container and the container had no pytest.

    The stub therefore returns PER-FRAME probabilities: the same row repeated
    once per frame. Identical rows make every aggregator agree, so the values
    these tests assert on are unchanged by the reduction that now happens
    downstream.
    """
    def fake(model, frames, device, image_size, activation="sigmoid"):
        seen.append({"activation": activation, "image_size": image_size,
                     "device": device, "n_frames": len(frames)})
        probs = tool_probs if activation == "sigmoid" else task_probs
        row = np.asarray(probs, dtype=np.float32)
        return np.tile(row, (max(1, len(frames)), 1))
    return fake


def _probs(classes, **overrides):
    values = dict.fromkeys(classes, 0.01)
    values.update(overrides)
    return [values[name] for name in classes]


def test_tool_probabilities_reach_the_router_through_the_thresholds(
        tmp_path, checkpoints, monkeypatch):
    made = Case(tmp_path, _config(checkpoints, thresholds=[0.9] * 12))
    _write_video(made.video)
    seen = []
    monkeypatch.setattr(inference, "predict_window_frames", _stub_predict_window(
        seen, _probs(TOOL_CLASSES, **{"cadiere forceps": 0.95}),
        _probs(TASK_CLASSES, suturing=0.9)))

    made.ask("Are there forceps being used here?").run()

    assert made.answer() == "Yes"


def test_a_class_over_its_own_tuned_threshold_is_present_below_a_half(
        tmp_path, checkpoints, monkeypatch):
    """The tuned thresholds span 0.05 to 0.95, so a shared 0.5 is not a
    conservative simplification -- it drops the rare classes outright. At 0.45
    against a tuned 0.40 the class is present; under a hardcoded 0.5 it is not,
    and the router's soft-presence rescue does not reach it either."""
    made = Case(tmp_path, _config(checkpoints, thresholds=[0.40] * 12))
    _write_video(made.video)
    monkeypatch.setattr(inference, "predict_window_frames", _stub_predict_window(
        [], _probs(TOOL_CLASSES, **{"stapler": 0.45}),
        _probs(TASK_CLASSES, suturing=0.9)))

    made.ask("Is a stapler being used?").run()

    assert made.answer() == "Yes"


def test_a_class_under_every_threshold_is_not_reported_present(
        tmp_path, checkpoints, monkeypatch):
    made = Case(tmp_path, _config(checkpoints, thresholds=[0.99] * 12))
    _write_video(made.video)
    monkeypatch.setattr(inference, "predict_window_frames", _stub_predict_window(
        [], _probs(TOOL_CLASSES), _probs(TASK_CLASSES, suturing=0.9)))

    made.ask("Is a stapler being used?").run()

    assert made.answer() == "No"


def test_each_expert_is_served_at_its_own_image_size(
        tmp_path, checkpoints, monkeypatch, capsys):
    """Both are 384 today, so a hardcoded constant passes every test until one
    expert is retrained at another resolution and then is silently wrong."""
    made = Case(tmp_path, _config(checkpoints, tool_image_size=48,
                                  task_image_size=96))
    _write_video(made.video)
    seen = []
    monkeypatch.setattr(inference, "predict_window_frames", _stub_predict_window(
        seen, _probs(TOOL_CLASSES), _probs(TASK_CLASSES, suturing=0.9)))

    made.ask("Is the camera being moved?").run()

    by_activation = {entry["activation"]: entry for entry in seen}
    assert by_activation["sigmoid"]["image_size"] == 48
    assert by_activation["softmax"]["image_size"] == 96
    # The checkpoints were trained at 64; the config wins and says so.
    assert "trained at image_size=64" in capsys.readouterr().err


def test_thresholds_that_drifted_from_the_config_are_reported(
        tmp_path, checkpoints, capsys):
    """The config is the frozen binding and the checkpoints are rewritten by
    retraining jobs. When they disagree the config wins -- but silently
    serving thresholds nobody chose is how a stale config goes unnoticed."""
    made = Case(tmp_path, _config(checkpoints, thresholds=[0.4] * 12))
    _write_video(made.video)
    made.ask("Is the camera being moved?").run()

    err = capsys.readouterr().err
    assert "thresholds differ from the config" in err
    assert "rebuild it" in err


# ------------------------------------------ deliberate serving thresholds
#
# `train_tools.py` tunes the checkpoint's thresholds on PER-FRAME validation
# probabilities; the container applies them to the CLIP MEAN of decode.frames
# frames. Those are different distributions, so the config carries a second,
# deliberately different vector tuned on the aggregation serving performs.
#
# The hazard this section exists to pin: the drift guard above was written to
# catch a config that fell behind its weights, and a deliberate divergence
# must not blunt it. `thresholds` therefore stays a MIRROR of the checkpoint
# -- and is what the guard compares -- while `serving_thresholds` is what the
# container applies.

def test_the_serving_thresholds_are_what_the_container_applies(
        tmp_path, checkpoints, monkeypatch):
    """0.45 is under the checkpoint's mirrored 0.5 and over the serving 0.40.
    If the mirror were still being applied the answer would be "No"."""
    made = Case(tmp_path, _config(
        checkpoints, thresholds=[0.5] * 12,
        serving_thresholds=_serving_block([0.40] * 12)))
    _write_video(made.video)
    monkeypatch.setattr(inference, "predict_window_frames", _stub_predict_window(
        [], _probs(TOOL_CLASSES, **{"stapler": 0.45}),
        _probs(TASK_CLASSES, suturing=0.9)))

    made.ask("Is a stapler being used?").run()

    assert made.answer() == "Yes"


def test_a_deliberate_serving_override_is_not_reported_as_drift(
        tmp_path, checkpoints, capsys):
    """The mirror matches the checkpoint, so nothing has drifted. Crying drift
    over the deliberate vector would train the operator to ignore the warning
    that exists to catch a stale config."""
    made = Case(tmp_path, _config(
        checkpoints, thresholds=[0.5] * 12,
        serving_thresholds=_serving_block([0.40] * 12)))
    _write_video(made.video)
    made.ask("Is the camera being moved?").run()

    err = capsys.readouterr().err
    assert "thresholds differ from the config" not in err
    assert "serving thresholds" in err


def test_the_drift_guard_still_fires_underneath_a_serving_override(
        tmp_path, checkpoints, capsys):
    """The whole point of keeping `thresholds` a mirror. The checkpoint says
    0.5, the config's mirror says 0.4: that is an accident, and it is still
    caught even though a deliberate override is also present."""
    made = Case(tmp_path, _config(
        checkpoints, thresholds=[0.4] * 12,
        serving_thresholds=_serving_block([0.40] * 12)))
    _write_video(made.video)
    made.ask("Is the camera being moved?").run()

    err = capsys.readouterr().err
    assert "thresholds differ from the config" in err


def test_serving_thresholds_tuned_on_other_weights_are_refused(
        tmp_path, checkpoints, monkeypatch, capsys):
    """A retrain replaces the weights and the config's serving vector is now
    calibrated for a model that is no longer there. Falling back to the
    checkpoint's own cuts is a known-mediocre channel; applying cuts tuned on
    someone else's probability scale is not bounded at all."""
    made = Case(tmp_path, _config(
        checkpoints, thresholds=[0.5] * 12,
        serving_thresholds=_serving_block([0.40] * 12, sha256="f" * 64)))
    _write_video(made.video)
    monkeypatch.setattr(inference, "predict_window_frames", _stub_predict_window(
        [], _probs(TOOL_CLASSES, **{"stapler": 0.45}),
        _probs(TASK_CLASSES, suturing=0.9)))

    made.ask("Is a stapler being used?").run()

    assert made.answer() == "No"
    assert "tuned on a different checkpoint" in capsys.readouterr().err


def test_serving_thresholds_of_the_wrong_length_are_refused(
        tmp_path, checkpoints, monkeypatch, capsys):
    """They are positional. A short vector must not reach `tools_present`,
    which would leave the tail of the taxonomy on the mirror's cuts and the
    head on the override's."""
    block = _serving_block([0.40] * 12)
    block["values"] = block["values"][:11]
    made = Case(tmp_path, _config(checkpoints, thresholds=[0.5] * 12,
                                  serving_thresholds=block))
    _write_video(made.video)
    monkeypatch.setattr(inference, "predict_window_frames", _stub_predict_window(
        [], _probs(TOOL_CLASSES, **{"stapler": 0.45}),
        _probs(TASK_CLASSES, suturing=0.9)))

    made.ask("Is a stapler being used?").run()

    assert made.answer() == "No"
    assert "11 serving thresholds for 12 classes" in capsys.readouterr().err


def test_a_config_without_a_serving_override_still_serves_the_mirror(
        tmp_path, checkpoints, monkeypatch):
    """The field is optional: a config built before this scheme existed, or
    one whose override was deliberately dropped, must keep working."""
    made = Case(tmp_path, _config(checkpoints, thresholds=[0.40] * 12))
    made.config_path.write_text(json.dumps(
        _config(checkpoints, thresholds=[0.40] * 12)), encoding="utf-8")
    _write_video(made.video)
    monkeypatch.setattr(inference, "predict_window_frames", _stub_predict_window(
        [], _probs(TOOL_CLASSES, **{"stapler": 0.45}),
        _probs(TASK_CLASSES, suturing=0.9)))

    made.ask("Is a stapler being used?").run()

    assert made.answer() == "Yes"


def test_a_checkpoint_trained_on_another_class_order_is_refused(
        tmp_path, checkpoints, monkeypatch):
    """Same head width, different meaning. Loading it succeeds and every
    probability then belongs to the wrong class, which is unreviewable in the
    output -- so it is refused and the case takes the fallback."""
    reversed_classes = list(reversed(TOOL_CLASSES))
    rogue = tmp_path / "rogue_tools.pt"
    save_checkpoint(rogue, Tiny(len(TOOL_CLASSES)),
                    {"classes": reversed_classes, "backbone": "efficientnet_v2_s",
                     "image_size": IMAGE_SIZE, "frames_per_window": 8,
                     "thresholds": [0.5] * len(TOOL_CLASSES)})
    made = Case(tmp_path, _config(checkpoints, tools_checkpoint=rogue))
    _write_video(made.video)
    # Nothing is present, so a model that was allowed to run would answer
    # "No" and could not be mistaken for the "Yes" fallback.
    monkeypatch.setattr(inference, "predict_window_frames", _stub_predict_window(
        [], _probs(TOOL_CLASSES), _probs(TASK_CLASSES, suturing=0.9)))
    made.ask("Are there forceps being used here?")

    assert made.run() == 0
    assert _assert_answered(made) == FALLBACK_POLAR


def test_the_two_heads_use_their_own_activations(
        tmp_path, checkpoints, monkeypatch):
    """Multi-label sigmoid for tools, multi-class softmax for task."""
    made = Case(tmp_path, _config(checkpoints))
    _write_video(made.video)
    seen = []
    monkeypatch.setattr(inference, "predict_window_frames", _stub_predict_window(
        seen, _probs(TOOL_CLASSES), _probs(TASK_CLASSES, suturing=0.9)))

    made.ask("Is the camera being moved?").run()

    assert sorted(entry["activation"] for entry in seen) == ["sigmoid", "softmax"]


def test_the_task_head_decides_an_organ_question(
        tmp_path, checkpoints, monkeypatch):
    made = Case(tmp_path, _config(checkpoints))
    _write_video(made.video)
    monkeypatch.setattr(inference, "predict_window_frames", _stub_predict_window(
        [], _probs(TOOL_CLASSES),
        _probs(TASK_CLASSES, **{"uterine horn": 0.9})))

    made.ask("What organ is being manipulated?").run()

    assert made.answer() == "Uterine horn"


def test_the_configured_frame_count_is_what_gets_decoded(
        tmp_path, checkpoints, monkeypatch):
    made = Case(tmp_path, _config(checkpoints, frames=3))
    _write_video(made.video, n=8)
    seen = []
    monkeypatch.setattr(inference, "predict_window_frames", _stub_predict_window(
        seen, _probs(TOOL_CLASSES), _probs(TASK_CLASSES, suturing=0.9)))

    made.ask("Is the camera being moved?").run()

    assert [entry["n_frames"] for entry in seen] == [3, 3]


def test_frames_are_ui_blurred_before_they_reach_a_model(
        tmp_path, checkpoints, monkeypatch):
    """Blurring the UI band is a challenge RULE, not an optimisation, so the
    serving path may not have a route into a model that bypasses it."""
    import surgvu.perceive as perceive_module

    made = Case(tmp_path, _config(checkpoints, frames=2, size=32))
    _write_video(made.video, n=4)
    calls = []
    real = perceive_module.prepare_frame
    monkeypatch.setattr(perceive_module, "prepare_frame",
                        lambda frame, size=512: calls.append(1) or real(frame, size=size))
    monkeypatch.setattr(inference, "predict_window_frames", _stub_predict_window(
        [], _probs(TOOL_CLASSES), _probs(TASK_CLASSES, suturing=0.9)))

    made.ask("Is the camera being moved?").run()

    assert len(calls) == 2


def test_models_dir_reroots_the_configs_checkpoints_by_basename(
        tmp_path, checkpoints, monkeypatch):
    """The config is built against /staging paths and the container ships the
    weights baked into the image, where /staging is not mounted."""
    made = Case(tmp_path, _config(
        checkpoints,
        tools_checkpoint=tmp_path / "not-mounted" / "tools.pt",
        task_checkpoint=tmp_path / "not-mounted" / "task.pt"))
    _write_video(made.video)
    monkeypatch.setattr(inference, "predict_window_frames", _stub_predict_window(
        [], _probs(TOOL_CLASSES), _probs(TASK_CLASSES, **{"uterine horn": 0.9})))
    made.ask("What organ is being manipulated?")

    made.run("--models-dir", str(checkpoints))

    assert made.answer() == "Uterine horn"


# ------------------------------------------------- it must always answer

def test_a_missing_video_still_answers(case):
    case.video.unlink()

    assert case.run() == 0
    assert _assert_answered(case) == FALLBACK_POLAR


def test_an_unreadable_video_still_answers(case):
    case.video.write_bytes(b"not an mp4, not even close")

    assert case.run() == 0
    assert _assert_answered(case) == FALLBACK_POLAR


def test_an_empty_video_file_still_answers(case):
    case.video.write_bytes(b"")

    assert case.run() == 0
    assert _assert_answered(case)


def test_a_missing_checkpoint_still_answers(tmp_path, checkpoints, capsys):
    made = Case(tmp_path, _config(
        checkpoints, tools_checkpoint=tmp_path / "gone" / "tools_v2.pt"))
    _write_video(made.video)
    made.ask("Are there forceps being used here?")

    assert made.run() == 0
    assert _assert_answered(made) == FALLBACK_POLAR
    # Which expert, bound by what, and why there is nothing to serve. A bare
    # loader error names a path and leaves the reader to guess the rest.
    err = capsys.readouterr().err
    assert "tools checkpoint" in err and "The config binds it" in err


def test_a_missing_perception_config_still_answers(case):
    case.config_path.unlink()

    assert case.run() == 0
    assert _assert_answered(case) == FALLBACK_POLAR


def test_a_crash_inside_our_own_code_still_answers(case, monkeypatch):
    """Stands in for a CUDA error, an OOM, or a bug of ours: whatever it is,
    the case is worth ~0.7 as long as something valid is written."""
    def explode(*args, **kwargs):
        raise RuntimeError("CUDA error: device-side assert triggered")

    monkeypatch.setattr(inference, "predict_window_frames", explode)

    assert case.run() == 0
    assert _assert_answered(case) == FALLBACK_POLAR


def test_a_perception_failure_on_an_open_question_answers_generically(
        case, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(inference, "predict_window_frames", explode)
    case.ask("What instrument is the surgeon using?")

    assert case.run() == 0
    assert _assert_answered(case) == FALLBACK_OPEN


def test_a_perception_failure_does_not_answer_from_an_empty_record(
        case, monkeypatch):
    """The tempting shortcut is to route the question against an empty
    perception dict, which the router tolerates -- and which answers "No" to
    every presence question. Gold polar answers skew Yes and a wrong polar
    costs 0.2985, so a total perception failure takes the calibrated
    fallback instead."""
    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(inference, "predict_window_frames", explode)
    case.ask("Is a needle driver being used in this clip?")

    case.run()

    assert case.answer() == "Yes"


def test_a_perception_failure_on_a_purpose_question_keeps_the_gold_answer(
        case, monkeypatch):
    """A purpose question never consulted perception in the first place.

    `_answer_purpose` reads the tool the QUESTION named and answers from world
    knowledge, so a dead video, a missing checkpoint or a CUDA fault costs it
    nothing -- as long as the fallback routes it instead of reaching for the
    generic sentence. That is the difference between the gold first reference
    (1.0000) and a plausible generic (0.35-0.48).
    """
    def explode(*args, **kwargs):
        raise RuntimeError("CUDA error: device-side assert triggered")

    monkeypatch.setattr(inference, "predict_window_frames", explode)
    case.ask("What is the purpose of using forceps?")

    assert case.run() == 0
    assert _assert_answered(case) == \
        "To grasp and hold tissues or objects during the surgery."


def test_a_perception_failure_on_a_procedure_question_still_names_it(
        case, monkeypatch):
    """`_answer_procedure` is a constant; there is nothing for a failure to
    take away."""
    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(inference, "predict_window_frames", explode)
    case.ask("What type of procedure is being performed?")

    assert case.run() == 0
    assert _assert_answered(case) == PROCEDURE_ANSWER


def test_the_fallback_routes_only_the_intents_that_ignore_perception(
        case, monkeypatch):
    """The perception-dependent intents keep the calibrated fallback. Routed
    against an empty record this question answers "No"; the fallback answers
    "Yes", which is the side the corpus favours."""
    seen = []
    monkeypatch.setattr(inference, "answer_question",
                        lambda question, perception: seen.append(perception) or "No")

    assert inference.fallback_answer("Is a needle driver being used?") == "Yes"
    assert seen == [], "a perception-dependent intent must not be re-routed"


def test_the_fallback_cannot_itself_fail_to_answer(case, monkeypatch):
    """The fallback is the last line of defence and now calls more code than
    a constant lookup. Whatever that code does, something must be written."""
    def explode(*args, **kwargs):
        raise RuntimeError("the classifier itself is broken")

    monkeypatch.setattr(inference, "predict_window_frames", explode)
    monkeypatch.setattr(inference, "classify_question", explode)
    case.ask("What is the purpose of using forceps?")

    assert case.run() == 0
    assert _assert_answered(case) == FALLBACK_OPEN


def test_a_missing_question_file_still_answers(case):
    case.question_path.unlink()

    assert case.run() == 0
    assert _assert_answered(case)


def test_a_question_that_is_not_json_is_still_answered(case):
    """Someone writing the file with `f.write(question)` instead of
    `json.dump` is a contract violation we can absorb, not die on."""
    case.question_path.write_text("Are there forceps being used here?",
                                  encoding="utf-8")

    assert case.run() == 0
    assert _assert_answered(case)


def test_an_empty_answer_can_never_be_written(case, monkeypatch):
    """The last line of defence: an empty string is the one response that
    scores worse than a wrong one, because it crashes the scorer."""
    # Patched on the ROUTER for the reason given in
    # test_the_question_is_json_decoded_not_read_raw: since ae4ef7c the
    # arbiter calls router.answer_question directly, so patching
    # `inference.answer_question` stubbed a function no longer on the path
    # and this test -- the one guarding the single worst outcome available,
    # an empty response that scores 0 -- was exercising nothing.
    monkeypatch.setattr(router, "answer_question",
                        lambda question, perception: "")

    case.run()

    assert _assert_answered(case) == FALLBACK_OPEN


def test_a_failure_is_reported_loudly_on_stderr(case, monkeypatch, capsys):
    def explode(*args, **kwargs):
        raise RuntimeError("device-side assert triggered")

    monkeypatch.setattr(inference, "predict_window_frames", explode)

    case.run()

    err = capsys.readouterr().err
    assert "device-side assert triggered" in err
    assert "FALLBACK" in err


# ------------------------------------------------ motion_v2 best-effort (R18)
#
# --motion-v2's DECODE (decode_clip_multiscale) stays inside the main
# try/except -- if that fails there are no frames and nothing to fall back
# to, so the whole-pipeline fallback is correct there, same as a plain
# decode_clip() failure. But motion_v2's COMPUTATION (motion_record_v2) is
# additive evidence read by nothing yet, computed from frames the appearance
# model already has (or is about to use). A failure inside it must not
# discard that appearance answer for the whole-pipeline fallback -- it must
# be caught, logged, and leave the perception record with no "motion_v2" key.

def test_a_motion_v2_computation_failure_still_yields_a_routed_answer(
        case, monkeypatch, capsys):
    """Before this fix, motion_record_v2 raising fell all the way through
    to the single outer try/except and traded a real, appearance-model-backed
    answer for the generic calibrated fallback -- because an optical-flow
    statistic failed, not because the video or the CNNs did."""
    import surgvu.motion as motion_module

    def explode(*args, **kwargs):
        raise RuntimeError("optical flow blew up")

    monkeypatch.setattr(motion_module, "motion_record_v2", explode)

    captured = {}
    real_route = inference.route

    def spy_route(question, perception, frames, timings, vlm):
        captured["perception"] = perception
        return real_route(question, perception, frames, timings, vlm)

    monkeypatch.setattr(inference, "route", spy_route)

    assert case.run("--motion-v2") == 0

    err = capsys.readouterr().err
    assert "optical flow blew up" in err, "the failure must still be logged loudly"
    assert "FALLBACK" not in err, (
        "a motion_v2 failure must not trip the whole-pipeline fallback")

    assert "perception" in captured, (
        "route() must still run against the appearance model's perception "
        "record -- the whole point of making this best-effort")
    assert "motion_v2" not in captured["perception"], (
        "a failed computation must leave the block absent, not a partial "
        "or stale value")

    assert _assert_answered(case)


def test_a_motion_and_motion_v2_pairing_failure_still_yields_a_routed_answer(
        case, monkeypatch, capsys):
    """When both --motion and --motion-v2 are passed, the v1 block is
    recomputed from a second decode (the uniform burst layout motion_v2's
    multiscale decode does not produce). That second decode+compute pair
    must be exactly as best-effort as motion_v2 itself -- its failure must
    not cost the answer either, and motion_v2 (already computed
    successfully) must survive untouched."""
    import surgvu.motion as motion_module

    def explode(*args, **kwargs):
        raise RuntimeError("burst motion blew up")

    monkeypatch.setattr(motion_module, "motion_record_from_bursts", explode)

    captured = {}
    real_route = inference.route

    def spy_route(question, perception, frames, timings, vlm):
        captured["perception"] = perception
        return real_route(question, perception, frames, timings, vlm)

    monkeypatch.setattr(inference, "route", spy_route)

    assert case.run("--motion-v2", "--motion") == 0

    err = capsys.readouterr().err
    assert "burst motion blew up" in err
    assert "FALLBACK" not in err

    assert "perception" in captured
    assert "motion_v2" in captured["perception"], (
        "motion_v2 succeeded independently and must not be discarded "
        "because the v1 pairing failed")
    assert "motion" not in captured["perception"]

    assert _assert_answered(case)


# --------------------------------------------------------------- timings

def test_every_stage_is_timed_on_stderr(case, capsys):
    """The budget is 10 minutes per case including container start, so where
    the time went has to be readable off a failed run's log."""
    case.run()

    err = capsys.readouterr().err
    for stage in ("decode=", "tools_infer=", "task_infer=", "route=", "total="):
        assert stage in err, "%r missing from:\n%s" % (stage, err)


def test_model_load_time_is_reported_separately_from_inference(case, capsys):
    """Loading is paid once per case -- there is one case per container run --
    so it is the number that decides whether a VLM fits at all."""
    case.run()

    err = capsys.readouterr().err
    assert "load_tools=" in err and "load_task=" in err


# ---------------------------------------------------------------- device

def test_the_device_is_chosen_automatically_when_not_given(
        case, monkeypatch, capsys):
    monkeypatch.setattr(inference.torch.cuda, "is_available", lambda: False)

    case.run("--device", "auto")

    assert "device=cpu" in capsys.readouterr().err


def test_a_device_failure_retries_on_cpu_before_giving_up(
        tmp_path, checkpoints, capsys):
    """A T4 that OOMs or a driver that is not there must not cost the case:
    CPU is slower than the GPU and far faster than a fallback string."""
    made = Case(tmp_path, _config(checkpoints))
    _write_video(made.video)
    made.ask("What organ is being manipulated?")

    assert made.run("--device", "not-a-device") == 0

    assert made.answer() != FALLBACK_OPEN, "the retry should have produced a real answer"
    assert "retrying on cpu" in capsys.readouterr().err


def test_cpu_is_not_retried_against_itself(case, monkeypatch, capsys):
    attempts = []

    def explode(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("boom")

    monkeypatch.setattr(inference, "predict_window_frames", explode)
    case.run("--device", "cpu")

    assert len(attempts) == 1


# ------------------------------------------------------------- VLM seam
#
# Plan 2 (Task 4) replaced the old `vlm_answer`/`try_vlm` pair -- a gate
# restricted to questions the router could not classify -- with
# `build_vlm`/`try_vlm_result`, whose gate is the ARBITER (config/
# arbiter.json's mode), not an intent check here. So "a routed question
# never reaches the VLM seam" is no longer true by design: the shipped
# `challenger` policy drafts a VLM answer for every question. That gate --
# and everything about what the arbiter does once a draft exists -- is
# tested in tests/test_inference_vlm.py (the serving wiring) and
# tests/test_arbiter.py (the policy itself); this file only keeps the two
# properties it is about: the seam is inert without `--vlm` at all, and a
# VLM that crashes never costs the case its answer.

def test_a_vlm_that_crashes_keeps_the_router_answer(case, monkeypatch, capsys):
    """Absorbed at the seam, not escalated: a VLM that OOMs on a T4 must cost
    its own answer and nothing else. The whole-pipeline fallback would produce
    the same string here, so the log is what separates them."""
    class Exploding(object):
        def available(self):
            return True

        def sample(self, video, question, perception, budget_seconds=None):
            raise RuntimeError("out of memory")

    monkeypatch.setattr(inference, "build_vlm", lambda args: Exploding())
    case.ask("Describe what you can see in the upper left corner.")

    assert case.run("--vlm") == 0
    assert _assert_answered(case) == FALLBACK_OPEN
    err = capsys.readouterr().err
    assert "keeping the router's answer" in err
    assert "FALLBACK" not in err


def test_the_shipped_seam_is_inert(case):
    """No `vlm` object, no VLM draft, regardless of what a caller asks --
    `try_vlm_result` is the seam's one entry point now."""
    assert inference.try_vlm_result(case.video, "anything", {}, None, []) is None
