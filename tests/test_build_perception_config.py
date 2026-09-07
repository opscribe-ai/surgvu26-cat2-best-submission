"""The frozen perception binding, and the ways it must refuse to be written.

`config/perception.json` is what the submission container reads instead of
rediscovering image sizes and thresholds from whichever `.pt` files happen to
be on disk. Two properties matter more than the schema:

  * a MISSING checkpoint must abort with nothing written -- a partial config
    routes the graded container to a model that does not exist;
  * the file must say WHICH checkpoint it bound, because the v2 files are
    produced by a retrain that may or may not have landed, and "which weights
    were these numbers measured on" is otherwise unanswerable after the fact.
"""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_perception_config import (                              # noqa: E402
    DEFAULT_FRAMES, DEFAULT_SIZE, build_config, expert_entry, main,
    resolve_checkpoint, serving_block_from_report, sha256,
)
from surgvu.taxonomy import TASK_CLASSES, TOOL_CLASSES              # noqa: E402


# ---------------------------------------------------------------- fixtures

def _tool_meta(**overrides):
    meta = {
        "classes": list(TOOL_CLASSES),
        "backbone": "efficientnet_v2_s",
        "image_size": 384,
        "frames_per_window": 8,
        "thresholds": [0.05 * (i + 1) for i in range(len(TOOL_CLASSES))],
        "macro_f1": 0.618,
        "per_class_f1": {name: 0.5 for name in TOOL_CLASSES},
    }
    meta.update(overrides)
    return meta


def _task_meta(**overrides):
    meta = {
        "classes": list(TASK_CLASSES),
        "backbone": "efficientnet_v2_s",
        "image_size": 384,
        "frames_per_window": 8,
        "accuracy": 0.743,
        "macro_f1": 0.547,
        "per_class_f1": {name: 0.5 for name in TASK_CLASSES},
    }
    meta.update(overrides)
    return meta


def _write_checkpoint(path, meta):
    """A checkpoint the builder can read: it only ever wants `meta`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {}, "meta": meta}, str(path))
    return path


@pytest.fixture
def models(tmp_path):
    directory = tmp_path / "models"
    _write_checkpoint(directory / "tools_v2.pt", _tool_meta())
    _write_checkpoint(directory / "task_v2.pt", _task_meta())
    return directory


# ------------------------------------------------------ checkpoint choice

def test_resolve_checkpoint_prefers_the_first_candidate_that_exists(tmp_path):
    primary = _write_checkpoint(tmp_path / "tools_v2.pt", _tool_meta())
    fallback = _write_checkpoint(tmp_path / "tools_v1.pt", _tool_meta())

    assert resolve_checkpoint("tools", [primary, fallback]) == (primary, "primary")


def test_resolve_checkpoint_falls_back_when_the_retrain_has_not_landed(tmp_path):
    """tools_v2.pt is written by a job that may still be running."""
    missing = tmp_path / "tools_v2.pt"
    fallback = _write_checkpoint(tmp_path / "tools_v1.pt", _tool_meta())

    assert resolve_checkpoint("tools", [missing, fallback]) == (fallback, "fallback")


def test_resolve_checkpoint_names_every_path_it_tried(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        resolve_checkpoint("tools", [tmp_path / "a.pt", tmp_path / "b.pt"])

    message = str(excinfo.value)
    assert "tools" in message and "a.pt" in message and "b.pt" in message


# --------------------------------------------------------- one expert entry

def test_entry_records_the_checkpoint_it_bound_and_how_it_got_there(tmp_path):
    path = _write_checkpoint(tmp_path / "tools_v2.pt", _tool_meta())

    entry = expert_entry("tools", path, "fallback", _tool_meta())

    assert entry["checkpoint"] == str(path)
    assert entry["checkpoint_name"] == "tools_v2.pt"
    assert entry["source"] == "fallback"


def test_entry_carries_backbone_image_size_and_frames_per_window(tmp_path):
    path = _write_checkpoint(tmp_path / "tools_v2.pt", _tool_meta())

    entry = expert_entry("tools", path, "primary",
                         _tool_meta(image_size=320, frames_per_window=4))

    assert entry["backbone"] == "efficientnet_v2_s"
    assert entry["image_size"] == 320
    assert entry["frames_per_window"] == 4


def test_entry_pairs_every_threshold_with_its_own_class(tmp_path):
    """Thresholds are POSITIONAL against the class list -- `tools_present`
    zips the two -- so the config carries the pairing explicitly as well."""
    path = _write_checkpoint(tmp_path / "tools_v2.pt", _tool_meta())
    meta = _tool_meta()

    entry = expert_entry("tools", path, "primary", meta)

    assert entry["classes"] == list(TOOL_CLASSES)
    assert entry["thresholds"] == meta["thresholds"]
    assert entry["thresholds_by_class"] == dict(
        zip(TOOL_CLASSES, meta["thresholds"]))


def test_the_tool_expert_without_thresholds_is_refused(tmp_path):
    path = _write_checkpoint(tmp_path / "tools_v2.pt", _tool_meta())
    meta = _tool_meta()
    del meta["thresholds"]

    with pytest.raises(SystemExit, match="thresholds"):
        expert_entry("tools", path, "primary", meta)


def test_a_threshold_per_class_or_nothing(tmp_path):
    path = _write_checkpoint(tmp_path / "tools_v2.pt", _tool_meta())

    with pytest.raises(SystemExit, match="threshold"):
        expert_entry("tools", path, "primary", _tool_meta(thresholds=[0.5] * 11))


def test_classes_must_be_the_taxonomy_the_router_reads(tmp_path):
    """The record is keyed by class NAME, so a reordered head produces a
    config whose keys are all correct and whose values belong elsewhere."""
    path = _write_checkpoint(tmp_path / "tools_v2.pt", _tool_meta())
    shuffled = list(TOOL_CLASSES)
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]

    with pytest.raises(SystemExit, match="taxonomy"):
        expert_entry("tools", path, "primary", _tool_meta(classes=shuffled))


# ------------------------------------------- the deliberate serving vector
#
# The checkpoint's cuts were tuned on PER-FRAME probabilities; the container
# thresholds the CLIP MEAN. The config therefore carries a second vector,
# tuned on the aggregation serving performs, and the first one stays as the
# mirror `scripts/inference.py` compares the checkpoint against. These tests
# are about the one thing that makes that safe: the deliberate vector may only
# ever travel with the weights it was measured on.

def _report(path, values=None, checkpoint_thresholds=None, classes=None):
    """A scripts/tune_serving_thresholds.py report over `path`'s weights."""
    meta = _tool_meta()
    return {
        "produced_by": "scripts/tune_serving_thresholds.py",
        "role": "tools",
        "classes": list(classes if classes is not None else TOOL_CLASSES),
        "checkpoint_thresholds": list(
            checkpoint_thresholds if checkpoint_thresholds is not None
            else meta["thresholds"]),
        "serving_thresholds": list(values or [0.25] * len(TOOL_CLASSES)),
        "measurements": {
            "clip_mean_checkpoint_thresholds": {"macro_f1": 0.6775},
            "clip_mean_serving_thresholds": {"macro_f1": 0.6971},
        },
        "provenance": {"checkpoint_sha256": sha256(path), "split": "val",
                       "windows": 4635, "date": "2026-08-11"},
    }


def test_a_serving_report_is_embedded_with_its_provenance(tmp_path):
    path = _write_checkpoint(tmp_path / "tools_v2.pt", _tool_meta())
    values = [0.05 * (i + 1) for i in range(len(TOOL_CLASSES))][::-1]

    entry = expert_entry("tools", path, "primary", _tool_meta(),
                         serving=serving_block_from_report(
                             _report(path, values=values)))

    block = entry["serving_thresholds"]
    assert block["values"] == values
    assert block["by_class"] == dict(zip(TOOL_CLASSES, values))
    assert block["provenance"]["checkpoint_sha256"] == entry["sha256"]
    assert block["measurements"]["clip_mean_serving_thresholds"]["macro_f1"]
    # The mirror is untouched: it is what the drift guard compares.
    assert entry["thresholds"] == _tool_meta()["thresholds"]


def test_serving_thresholds_may_not_travel_onto_other_weights(tmp_path):
    """The measured gain is a property of one probability scale. Carrying the
    vector onto a retrain would ship cuts for a model that is not there."""
    path = _write_checkpoint(tmp_path / "tools_v2.pt", _tool_meta())
    other = _write_checkpoint(tmp_path / "tools_v3.pt",
                              _tool_meta(macro_f1=0.7))

    with pytest.raises(SystemExit, match="tuned on checkpoint"):
        expert_entry("tools", other, "primary", _tool_meta(),
                     serving=serving_block_from_report(_report(path)))


def test_serving_thresholds_measured_against_other_cuts_are_refused(tmp_path):
    """The report's before/after describes a comparison against the
    checkpoint's own cuts. If those are not the cuts in this checkpoint, the
    numbers in the config would describe a comparison nobody made."""
    path = _write_checkpoint(tmp_path / "tools_v2.pt", _tool_meta())

    with pytest.raises(SystemExit, match="measured against"):
        expert_entry("tools", path, "primary", _tool_meta(),
                     serving=serving_block_from_report(
                         _report(path, checkpoint_thresholds=[0.5] * 12)))


def test_a_serving_vector_of_the_wrong_length_is_refused(tmp_path):
    path = _write_checkpoint(tmp_path / "tools_v2.pt", _tool_meta())
    block = serving_block_from_report(_report(path))
    block["values"] = block["values"][:11]

    with pytest.raises(SystemExit, match="11 values for 12 classes"):
        expert_entry("tools", path, "primary", _tool_meta(), serving=block)


def test_a_report_tuned_over_another_taxonomy_is_refused(tmp_path):
    """A vector tuned over a reordered class list is not a worse vector, it is
    a vector for other classes -- and it would apply silently."""
    path = _write_checkpoint(tmp_path / "tools_v2.pt", _tool_meta())
    shuffled = list(TOOL_CLASSES)
    shuffled[0], shuffled[3] = shuffled[3], shuffled[0]

    with pytest.raises(SystemExit, match="taxonomy"):
        serving_block_from_report(_report(path, classes=shuffled))


def test_the_task_expert_cannot_take_serving_thresholds(tmp_path):
    path = _write_checkpoint(tmp_path / "task_v2.pt", _task_meta())

    with pytest.raises(SystemExit, match="serving thresholds"):
        expert_entry("task", path, "primary", _task_meta(),
                     serving={"values": [0.5]})


def test_a_rebuild_carries_the_serving_thresholds_forward(models, tmp_path):
    """The reason this exists. Re-running the builder is routine -- a changed
    path, a changed frame count -- and a deliberate calibration that a routine
    rebuild silently reverts is not a calibration."""
    out = tmp_path / "perception.json"
    report = tmp_path / "serving.json"
    values = [0.31] * len(TOOL_CLASSES)
    report.write_text(json.dumps(_report(models / "tools_v2.pt", values=values)),
                      encoding="utf-8")
    main(["--models-dir", str(models), "--out", str(out),
          "--tools-serving-thresholds", str(report)])

    assert main(["--models-dir", str(models), "--out", str(out)]) == 0

    tools = json.loads(out.read_text(encoding="utf-8"))["experts"]["tools"]
    assert tools["serving_thresholds"]["values"] == values


def test_a_rebuild_onto_new_weights_refuses_to_carry_them(models, tmp_path):
    out = tmp_path / "perception.json"
    report = tmp_path / "serving.json"
    report.write_text(json.dumps(_report(models / "tools_v2.pt")),
                      encoding="utf-8")
    main(["--models-dir", str(models), "--out", str(out),
          "--tools-serving-thresholds", str(report)])
    # The retrain lands: same filename, different bytes, same meta.
    _write_checkpoint(models / "tools_v2.pt", _tool_meta(epochs=9))

    with pytest.raises(SystemExit, match="tuned on checkpoint"):
        main(["--models-dir", str(models), "--out", str(out)])


def test_dropping_the_serving_thresholds_is_explicit(models, tmp_path):
    out = tmp_path / "perception.json"
    report = tmp_path / "serving.json"
    report.write_text(json.dumps(_report(models / "tools_v2.pt")),
                      encoding="utf-8")
    main(["--models-dir", str(models), "--out", str(out),
          "--tools-serving-thresholds", str(report)])

    main(["--models-dir", str(models), "--out", str(out),
          "--drop-serving-thresholds"])

    tools = json.loads(out.read_text(encoding="utf-8"))["experts"]["tools"]
    assert "serving_thresholds" not in tools


def test_a_config_built_without_a_report_carries_no_serving_vector(models, tmp_path):
    out = tmp_path / "perception.json"

    main(["--models-dir", str(models), "--out", str(out)])

    tools = json.loads(out.read_text(encoding="utf-8"))["experts"]["tools"]
    assert "serving_thresholds" not in tools


def test_the_task_expert_has_no_thresholds(tmp_path):
    """8-way softmax: argmax needs no cutoff, and inventing one would be a
    lie the serving path could act on."""
    path = _write_checkpoint(tmp_path / "task_v2.pt", _task_meta())

    entry = expert_entry("task", path, "primary", _task_meta())

    assert entry["thresholds"] is None
    assert entry["activation"] == "softmax"


def test_the_tool_expert_is_multi_label(tmp_path):
    path = _write_checkpoint(tmp_path / "tools_v2.pt", _tool_meta())

    entry = expert_entry("tools", path, "primary", _tool_meta())

    assert entry["activation"] == "sigmoid"


def test_a_missing_structural_key_is_refused(tmp_path):
    path = _write_checkpoint(tmp_path / "task_v2.pt", _task_meta())
    meta = _task_meta()
    del meta["image_size"]

    with pytest.raises(SystemExit, match="image_size"):
        expert_entry("task", path, "primary", meta)


def test_validation_metrics_are_carried_verbatim(tmp_path):
    """Everything in `meta` that is not structural is a measurement, and it
    is copied rather than enumerated so a metric added by a later training
    run reaches the config without editing this file."""
    path = _write_checkpoint(tmp_path / "tools_v2.pt", _tool_meta())

    entry = expert_entry("tools", path, "primary",
                         _tool_meta(macro_f1=0.618, epochs=3, novel_metric=7))

    assert entry["metrics"]["macro_f1"] == 0.618
    assert entry["metrics"]["epochs"] == 3
    assert entry["metrics"]["novel_metric"] == 7
    assert entry["metrics"]["per_class_f1"]["stapler"] == 0.5
    assert "classes" not in entry["metrics"]
    assert "thresholds" not in entry["metrics"]


def test_entry_fingerprints_the_weights_file(tmp_path):
    """Which bytes these numbers were measured on. tools_v2.pt is written by
    a retrain, so 'the file called tools_v2.pt' is not an identity."""
    path = _write_checkpoint(tmp_path / "tools_v2.pt", _tool_meta())

    import hashlib

    entry = expert_entry("tools", path, "primary", _tool_meta())

    assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert entry["size_bytes"] == path.stat().st_size


# ------------------------------------------------------------ whole config

def test_config_names_both_experts_and_the_sampling_policy(models):
    config = build_config(models / "tools_v2.pt", "primary", _tool_meta(),
                          models / "task_v2.pt", "fallback", _task_meta(),
                          frames=12, size=256)

    assert set(config["experts"]) == {"tools", "task"}
    assert config["decode"] == {"frames": 12, "size": 256}
    assert config["experts"]["task"]["source"] == "fallback"


# ------------------------------------------------------------- the driver

def test_main_writes_a_config_naming_the_v2_checkpoints(models, tmp_path):
    out = tmp_path / "perception.json"

    assert main(["--models-dir", str(models), "--out", str(out)]) == 0

    config = json.loads(out.read_text(encoding="utf-8"))
    assert config["experts"]["tools"]["checkpoint_name"] == "tools_v2.pt"
    assert config["experts"]["tools"]["source"] == "primary"
    assert config["decode"] == {"frames": DEFAULT_FRAMES, "size": DEFAULT_SIZE}


def test_main_records_that_it_used_the_v1_fallback(models, tmp_path):
    (models / "tools_v2.pt").unlink()
    _write_checkpoint(models / "tools_efficientnet_v2_s.pt", _tool_meta())
    out = tmp_path / "perception.json"

    main(["--models-dir", str(models), "--out", str(out)])

    tools = json.loads(out.read_text(encoding="utf-8"))["experts"]["tools"]
    assert tools["checkpoint_name"] == "tools_efficientnet_v2_s.pt"
    assert tools["source"] == "fallback"


def test_main_takes_explicit_checkpoint_paths(models, tmp_path):
    """Paths are arguments, not constants: a retrain writes new files and the
    config must be buildable against them without editing the script."""
    elsewhere = _write_checkpoint(tmp_path / "elsewhere" / "tools_v3.pt",
                                  _tool_meta())
    out = tmp_path / "perception.json"

    main(["--models-dir", str(models), "--tools-checkpoint", str(elsewhere),
          "--out", str(out)])

    tools = json.loads(out.read_text(encoding="utf-8"))["experts"]["tools"]
    assert tools["checkpoint"] == str(elsewhere)


def test_main_refuses_to_write_a_partial_config(models, tmp_path):
    """The task checkpoint is gone. Writing the tools half would leave the
    container routing task questions at a file that does not exist."""
    (models / "task_v2.pt").unlink()
    out = tmp_path / "perception.json"

    with pytest.raises(SystemExit, match="task"):
        main(["--models-dir", str(models), "--out", str(out)])

    assert not out.exists()


def test_main_does_not_clobber_an_existing_config_when_it_fails(models, tmp_path):
    (models / "task_v2.pt").unlink()
    out = tmp_path / "perception.json"
    out.write_text('{"kept": true}', encoding="utf-8")

    with pytest.raises(SystemExit):
        main(["--models-dir", str(models), "--out", str(out)])

    assert json.loads(out.read_text(encoding="utf-8")) == {"kept": True}
