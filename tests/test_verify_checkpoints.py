"""Tests for the build-time checkpoint gate.

`scripts/verify_checkpoints.py` is the thing standing between a truncated or
stale .pt file and forty graded cases answered with a fallback string. It runs
inside `docker build`, so it may not import torch and it may not pass anything
it has not actually hashed.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import verify_checkpoints as vk  # noqa: E402


def write_models(models_dir, payloads):
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        (models_dir / name).write_bytes(data)


# A miniature taxonomy. The gate is positional-vector logic and provenance
# logic; three classes exercise both and keep a failure legible.
TOOL_CLASSES = ["alpha", "beta", "gamma"]
TASK_CLASSES = ["one", "two"]
MIRROR = [0.10, 0.20, 0.30]          # what the checkpoint tuned, per frame
SERVING = [0.15, 0.20, 0.25]         # what was re-tuned on the clip mean

SENTINEL = object()


def serving_block(digest, values=SENTINEL, by_class=SENTINEL,
                  tuned_against=SENTINEL, provenance=SENTINEL, **overrides):
    """A sound `serving_thresholds` block for weights whose sha256 is `digest`."""
    values = list(SERVING) if values is SENTINEL else values
    block = {
        "values": values,
        "by_class": (dict(zip(TOOL_CLASSES, values))
                     if by_class is SENTINEL else by_class),
        "applies_to": "the clip mean over decode.frames evenly spaced frames",
        "note": "DELIBERATE divergence from `thresholds`.",
        "tuned_against_checkpoint_thresholds":
            list(MIRROR) if tuned_against is SENTINEL else tuned_against,
        "measurements": {"delta_macro_f1": 0.0195},
        "provenance": ({"checkpoint_sha256": digest, "split": "val"}
                       if provenance is SENTINEL else provenance),
    }
    for key, value in overrides.items():
        if value is None:
            block.pop(key, None)
        else:
            block[key] = value
    return block


def make_config(path, models_dir, payloads, schema_version=2, serving=SENTINEL):
    """A perception config whose fingerprints describe `payloads` truthfully.

    Schema 2 by default, with the tool expert carrying a sound serving block --
    i.e. the shape `config/perception.json` actually has today.
    """
    experts = {}
    for role, name in (("tools", "tools_v2.pt"), ("task", "task_v2.pt")):
        data = payloads[name]
        digest = hashlib.sha256(data).hexdigest()
        entry = {"role": role,
                 "checkpoint": "/staging/n/nkalthoff/surgvu26/models/" + name,
                 "checkpoint_name": name,
                 "sha256": digest,
                 "size_bytes": len(data),
                 "classes": list(TOOL_CLASSES if role == "tools" else TASK_CLASSES),
                 "thresholds": list(MIRROR) if role == "tools" else None}
        if role == "tools":
            block = serving_block(digest) if serving is SENTINEL else serving
            if block is not None:
                entry["serving_thresholds"] = block
        experts[role] = entry
    config = {"experts": experts}
    if schema_version is not None:
        config = {"schema_version": schema_version, "experts": experts}
    path.write_text(json.dumps(config))
    return path


def reread(config):
    return json.loads(Path(config).read_text())


def rewrite(config, data):
    Path(config).write_text(json.dumps(data))
    return config


@pytest.fixture
def bound(tmp_path):
    payloads = {"tools_v2.pt": b"tool-weights", "task_v2.pt": b"task-weights"}
    models_dir = tmp_path / "models"
    write_models(models_dir, payloads)
    config = make_config(tmp_path / "perception.json", models_dir, payloads)
    return config, models_dir, payloads


# --------------------------------------------------------------------------

def test_matching_checkpoints_verify(bound):
    config, models_dir, _ = bound
    results = vk.verify(config, models_dir)
    assert [r.ok for r in results] == [True, True]
    assert vk.failures(results) == []


def test_reports_every_expert_the_config_binds(bound):
    config, models_dir, _ = bound
    assert sorted(r.role for r in vk.verify(config, models_dir)) == ["task", "tools"]


def test_wrong_bytes_fail_even_at_the_right_size(bound):
    """One flipped byte is the whole point: a same-size, wrong-content file is
    what a partial copy or a stale retrain leaves behind."""
    config, models_dir, payloads = bound
    (models_dir / "tools_v2.pt").write_bytes(b"tool-weightS")
    results = vk.verify(config, models_dir)
    assert not {r.role: r.ok for r in results}["tools"]
    assert any("sha256" in problem for problem in vk.failures(results))


def test_truncated_checkpoint_fails(bound):
    config, models_dir, _ = bound
    (models_dir / "task_v2.pt").write_bytes(b"task-w")
    assert {r.role: r.ok for r in vk.verify(config, models_dir)}["task"] is False


def test_missing_checkpoint_fails_rather_than_raising(bound):
    """A missing file must be a reported failure, not a traceback: the report
    should still name the OTHER expert's state."""
    config, models_dir, _ = bound
    (models_dir / "tools_v2.pt").unlink()
    results = vk.verify(config, models_dir)
    assert len(results) == 2
    assert {r.role: r.ok for r in results} == {"tools": False, "task": True}
    assert any("missing" in problem for problem in vk.failures(results))


def test_models_dir_reroots_by_basename(bound, tmp_path):
    """The config records /staging paths that do not exist in the image. Only
    the basename may be used."""
    config, models_dir, payloads = bound
    elsewhere = tmp_path / "opt" / "algorithm" / "models"
    write_models(elsewhere, payloads)
    assert [r.ok for r in vk.verify(config, elsewhere)] == [True, True]


def test_the_configs_own_staging_path_is_never_read(bound, monkeypatch):
    """If verify fell back to entry["checkpoint"], a wrong models_dir would
    still pass on a machine where /staging happens to be mounted."""
    config, _, _ = bound
    results = vk.verify(config, Path("/nonexistent-models-dir"))
    assert [r.ok for r in results] == [False, False]


def test_size_mismatch_alone_is_reported(bound):
    """sha256 and size are recorded separately; a config whose size is stale is
    itself a defect worth failing on."""
    config, models_dir, payloads = bound
    data = json.loads(config.read_text())
    data["experts"]["tools"]["size_bytes"] = 999999
    config.write_text(json.dumps(data))
    assert {r.role: r.ok for r in vk.verify(config, models_dir)}["tools"] is False


def test_a_digest_matching_only_at_the_start_still_fails(bound):
    """The comparison must be over the WHOLE digest.

    A prefix comparison passes every ordinary test, because two different
    files differ from their first hex character. This config's sha256 is the
    real digest with only its LAST character changed, which is the one shape
    that separates `digest == expected` from `digest[:8] == expected[:8]`.
    """
    config, models_dir, payloads = bound
    real = hashlib.sha256(payloads["tools_v2.pt"]).hexdigest()
    near_miss = real[:-1] + ("0" if real[-1] != "0" else "1")
    data = json.loads(config.read_text())
    data["experts"]["tools"]["sha256"] = near_miss
    config.write_text(json.dumps(data))
    assert {r.role: r.ok for r in vk.verify(config, models_dir)}["tools"] is False


def test_a_multi_block_checkpoint_verifies(tmp_path):
    """The file is read in 1 MB blocks and every block must reach the hash.

    A checkpoint is 82 MB, so all but the first block of the real thing is
    read by the loop this pins. Reading only the first block would produce the
    wrong digest for any file bigger than a block -- i.e. would reject the
    correct weights -- which is the failure this test, not the tampering test
    below, is what catches.
    """
    payloads = {"tools_v2.pt": b"A" * (vk.BLOCK + 4096), "task_v2.pt": b"task"}
    models_dir = tmp_path / "models"
    write_models(models_dir, payloads)
    config = make_config(tmp_path / "perception.json", models_dir, payloads)
    assert [r.ok for r in vk.verify(config, models_dir)] == [True, True]


def test_corruption_beyond_the_first_block_is_caught(tmp_path):
    """The same-length, same-first-megabyte file: what a partial re-copy of an
    82 MB checkpoint actually leaves behind."""
    payloads = {"tools_v2.pt": b"A" * (vk.BLOCK + 4096), "task_v2.pt": b"task"}
    models_dir = tmp_path / "models"
    write_models(models_dir, payloads)
    config = make_config(tmp_path / "perception.json", models_dir, payloads)
    tampered = payloads["tools_v2.pt"][:vk.BLOCK] + b"B" * 4096
    assert len(tampered) == len(payloads["tools_v2.pt"])
    (models_dir / "tools_v2.pt").write_bytes(tampered)
    assert {r.role: r.ok for r in vk.verify(config, models_dir)}["tools"] is False


def test_main_reports_every_expert_not_just_the_first(bound, capsys):
    """Both lines have to be printed: a build log showing one OK line is not
    evidence that the other checkpoint was ever looked at."""
    config, models_dir, _ = bound
    vk.main([str(config), str(models_dir)])
    out = capsys.readouterr().out
    assert "tools_v2.pt" in out and "task_v2.pt" in out
    weights = [line for line in out.splitlines()
               if "tools_v2.pt" in line or "task_v2.pt" in line]
    assert len(weights) == 2
    assert all(line.rstrip().endswith("OK") for line in weights)


def test_main_exits_zero_when_bound(bound, capsys):
    config, models_dir, _ = bound
    assert vk.main([str(config), str(models_dir)]) == 0
    assert "match the frozen binding" in capsys.readouterr().out


def test_main_exits_nonzero_on_a_mismatch(bound, capsys):
    """A build that keeps going after this has produced a bad image."""
    config, models_dir, _ = bound
    (models_dir / "task_v2.pt").write_bytes(b"nope")
    assert vk.main([str(config), str(models_dir)]) != 0


def test_main_prints_the_digest_it_computed(bound, capsys):
    config, models_dir, payloads = bound
    vk.main([str(config), str(models_dir)])
    out = capsys.readouterr().out
    assert hashlib.sha256(payloads["tools_v2.pt"]).hexdigest() in out


def test_still_verifies_when_torch_cannot_be_imported(tmp_path, bound):
    """It runs inside `docker build` before anything guarantees a working GPU
    stack, and a torch import there costs seconds and can fail outright.

    Asserted by running the script for real with an `import torch` that
    raises, rather than by grepping the source -- a grep would be satisfied by
    a comment and defeated by an indirect import.
    """
    import subprocess
    config, models_dir, _ = bound
    sabotage = tmp_path / "sitecustomize_dir"
    sabotage.mkdir()
    (sabotage / "torch.py").write_text(
        "raise ImportError('torch is deliberately unavailable in this test')")
    env = {"PYTHONPATH": str(sabotage), "PATH": "/usr/bin:/bin"}
    completed = subprocess.run(
        [sys.executable, str(Path(vk.__file__)), str(config), str(models_dir)],
        capture_output=True, env=env)
    assert completed.returncode == 0, completed.stderr.decode()
    assert b"match the frozen binding" in completed.stdout


# ==========================================================================
# THE SERVING-THRESHOLD BINDING
#
# `config/perception.json` is schema 2 and carries a `serving_thresholds`
# block: per-class cuts re-tuned on the CLIP MEAN the container actually
# thresholds, worth +0.0195 macro-F1 over the checkpoint's own per-frame cuts.
#
# `scripts/inference.py` degrades to the mirrored per-frame cuts with a WARNING
# when that block is missing or unfit. That is the right call at serving time
# -- a mediocre threshold beats a case with no answer -- and exactly the wrong
# call at BUILD time, where the only thing the warning does is scroll past. An
# image that quietly ships 0.0195 less is the failure these tests exist to make
# impossible.
# ==========================================================================

def test_a_sound_serving_block_verifies(bound):
    config, models_dir, _ = bound
    results = vk.verify_serving(config)
    assert vk.failures(results) == []
    assert all(result.ok for result in results)


def test_a_sound_serving_block_builds(bound, capsys):
    config, models_dir, _ = bound
    assert vk.main([str(config), str(models_dir)]) == 0


# -- FATAL 1: the block is not there at all --------------------------------

def test_a_missing_serving_block_fails_the_build(bound, capsys):
    """The whole point. Degrading silently is what we are refusing."""
    config, models_dir, payloads = bound
    make_config(config, models_dir, payloads, serving=None)
    assert vk.main([str(config), str(models_dir)]) != 0
    err = capsys.readouterr().err
    assert "serving_thresholds" in err


def test_the_missing_block_failure_names_the_opt_out(bound, capsys):
    """A gate that fails without saying how to proceed deliberately gets
    disabled by whoever hits it at 2am."""
    config, models_dir, payloads = bound
    make_config(config, models_dir, payloads, serving=None)
    vk.main([str(config), str(models_dir)])
    out = capsys.readouterr()
    assert "--serving-thresholds optional" in (out.out + out.err)


def test_a_v1_config_fails_by_default(bound):
    """A schema-1 config cannot carry the block, so building from one is
    building without the improvement. That has to be said out loud."""
    config, models_dir, payloads = bound
    make_config(config, models_dir, payloads, schema_version=1, serving=None)
    assert vk.main([str(config), str(models_dir)]) != 0


# -- THE ESCAPE HATCH ------------------------------------------------------

def test_the_opt_out_lets_a_v1_config_build(bound, capsys):
    """`build_perception_config.py --drop-serving-thresholds` exists, so
    building from its output has to be possible -- by typing something."""
    config, models_dir, payloads = bound
    make_config(config, models_dir, payloads, schema_version=1, serving=None)
    assert vk.main([str(config), str(models_dir),
                    "--serving-thresholds", "optional"]) == 0


def test_the_opt_out_is_loud(bound, capsys):
    """It is waived, not passed. A build log that reads OK for a waived gate
    is the silence this whole change is about."""
    config, models_dir, payloads = bound
    make_config(config, models_dir, payloads, schema_version=1, serving=None)
    vk.main([str(config), str(models_dir), "--serving-thresholds", "optional"])
    out = capsys.readouterr().out
    assert "WAIVED" in out
    assert "WARNING" in out


def test_the_opt_out_does_not_waive_a_malformed_block(bound):
    """`optional` means "I accept an image without the re-tuned cuts", never
    "stop checking". A block that IS there and IS wrong still fails."""
    config, models_dir, payloads = bound
    data = reread(config)
    data["experts"]["tools"]["serving_thresholds"]["values"] = [0.1, 0.2]
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir),
                    "--serving-thresholds", "optional"]) != 0


def test_the_opt_out_does_not_waive_a_misbound_block(bound):
    config, models_dir, payloads = bound
    data = reread(config)
    (data["experts"]["tools"]["serving_thresholds"]["provenance"]
     ["checkpoint_sha256"]) = "f" * 64
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir),
                    "--serving-thresholds", "optional"]) != 0


def test_an_unknown_mode_is_rejected(bound):
    """Not a free-text field: a typo'd `--serving-thresholds optionall` must
    not fall through to a default that happens to be permissive."""
    config, models_dir, _ = bound
    with pytest.raises(SystemExit):
        vk.main([str(config), str(models_dir),
                 "--serving-thresholds", "optionall"])


# -- FATAL 2: the block belongs to different weights -----------------------

def test_a_block_tuned_on_other_weights_fails(bound, capsys):
    """A retrain moves every probability scale the cuts were calibrated
    against. Cuts for a model that is no longer in the image are not a better
    channel than the checkpoint's own -- they are an unbounded one."""
    config, models_dir, _ = bound
    data = reread(config)
    (data["experts"]["tools"]["serving_thresholds"]["provenance"]
     ["checkpoint_sha256"]) = "f" * 64
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir)]) != 0
    assert "tuned on" in capsys.readouterr().err


def test_a_block_with_no_provenance_sha_fails(bound, capsys):
    """The runtime accepts an unnamed provenance -- `tuned_on and bound and
    tuned_on != bound` is False when the field is absent, so the cuts are
    served with nothing tying them to these weights. The BUILD may not.

    The MESSAGE is asserted, not just the exit code: "bound to nothing" and
    "bound to the wrong weights" are different defects with different fixes,
    and a gate that reports the second for the first sends whoever reads it
    looking for a retrain that never happened."""
    config, models_dir, _ = bound
    data = reread(config)
    data["experts"]["tools"]["serving_thresholds"]["provenance"] = {"split": "val"}
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir)]) != 0
    assert "no provenance.checkpoint_sha256" in capsys.readouterr().err


def test_a_block_with_no_provenance_at_all_fails(bound, capsys):
    config, models_dir, _ = bound
    data = reread(config)
    del data["experts"]["tools"]["serving_thresholds"]["provenance"]
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir)]) != 0
    assert "no provenance.checkpoint_sha256" in capsys.readouterr().err


def test_a_provenance_sha_matching_only_at_the_start_fails(bound):
    """The comparison is over the WHOLE digest, for the same reason the
    checkpoint one is."""
    config, models_dir, payloads = bound
    real = hashlib.sha256(payloads["tools_v2.pt"]).hexdigest()
    data = reread(config)
    (data["experts"]["tools"]["serving_thresholds"]["provenance"]
     ["checkpoint_sha256"]) = real[:-1] + ("0" if real[-1] != "0" else "1")
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir)]) != 0


def test_a_block_measured_against_other_cuts_fails(bound, capsys):
    """The block records the checkpoint cuts its before/after was measured
    against. If the config's mirror is not those cuts, the +0.0195 describes a
    comparison this image would not be making."""
    config, models_dir, _ = bound
    data = reread(config)
    (data["experts"]["tools"]["serving_thresholds"]
     ["tuned_against_checkpoint_thresholds"]) = [0.5, 0.5, 0.5]
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir)]) != 0


def test_the_measured_against_cuts_may_not_be_a_short_vector(bound):
    config, models_dir, _ = bound
    data = reread(config)
    (data["experts"]["tools"]["serving_thresholds"]
     ["tuned_against_checkpoint_thresholds"]) = MIRROR[:2]
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir)]) != 0


def test_float_noise_in_the_measured_against_cuts_is_tolerated(bound):
    """The two vectors travel through JSON and through torch floats. A 1e-9
    difference is representation, not a different measurement -- failing on it
    would make the gate the thing people route around."""
    config, models_dir, _ = bound
    data = reread(config)
    (data["experts"]["tools"]["serving_thresholds"]
     ["tuned_against_checkpoint_thresholds"]) = [value + 1e-9 for value in MIRROR]
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir)]) == 0


# -- FATAL 3: the vector itself is unfit -----------------------------------

def test_a_short_serving_vector_fails(bound, capsys):
    """They are positional. A short vector thresholds the head of the taxonomy
    on purpose and the tail by accident."""
    config, models_dir, _ = bound
    data = reread(config)
    block = data["experts"]["tools"]["serving_thresholds"]
    block["values"] = SERVING[:2]
    block["by_class"] = dict(zip(TOOL_CLASSES, SERVING[:2]))
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir)]) != 0
    assert "2 serving thresholds for 3 classes" in capsys.readouterr().err


def test_a_long_serving_vector_fails(bound, capsys):
    """The other end of the same defect: a class the head does not have.

    Asserted on the LENGTH message rather than on the exit code alone -- a
    gate that only checked `len(values) < len(classes)` would still fail this
    config, but for the wrong reason and only because `by_class` happened to
    grow with it."""
    config, models_dir, _ = bound
    data = reread(config)
    block = data["experts"]["tools"]["serving_thresholds"]
    block["values"] = SERVING + [0.4]
    block["by_class"] = dict(zip(TOOL_CLASSES + ["delta"], block["values"]))
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir)]) != 0
    assert "4 serving thresholds for 3 classes" in capsys.readouterr().err


def test_a_boolean_serving_value_fails(bound):
    """JSON `true` is an int in Python and would sail through a bare float()
    as a cut of 1.0."""
    config, models_dir, _ = bound
    data = reread(config)
    block = data["experts"]["tools"]["serving_thresholds"]
    block["values"] = [0.15, True, 0.25]
    block["by_class"] = dict(zip(TOOL_CLASSES, block["values"]))
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir)]) != 0


def test_a_missing_values_vector_fails(bound):
    config, models_dir, _ = bound
    data = reread(config)
    del data["experts"]["tools"]["serving_thresholds"]["values"]
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir)]) != 0


def test_a_non_numeric_serving_value_fails(bound):
    """`"0.15"` compares fine in JSON and is not a threshold."""
    config, models_dir, _ = bound
    data = reread(config)
    block = data["experts"]["tools"]["serving_thresholds"]
    block["values"] = [0.15, "0.20", 0.25]
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir)]) != 0


def test_a_null_serving_value_fails(bound):
    config, models_dir, _ = bound
    data = reread(config)
    data["experts"]["tools"]["serving_thresholds"]["values"] = [0.15, None, 0.25]
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir)]) != 0


def test_a_serving_value_above_one_fails(bound):
    """A sigmoid never exceeds 1, so a cut above it silences that class
    forever -- and reads as a plausible number in a diff."""
    config, models_dir, _ = bound
    data = reread(config)
    block = data["experts"]["tools"]["serving_thresholds"]
    block["values"] = [0.15, 1.4, 0.25]
    block["by_class"] = dict(zip(TOOL_CLASSES, block["values"]))
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir)]) != 0


def test_a_negative_serving_value_fails(bound):
    """The mirror image: a class asserted present in every frame of every
    case."""
    config, models_dir, _ = bound
    data = reread(config)
    block = data["experts"]["tools"]["serving_thresholds"]
    block["values"] = [0.15, -0.2, 0.25]
    block["by_class"] = dict(zip(TOOL_CLASSES, block["values"]))
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir)]) != 0


def test_the_unit_interval_endpoints_are_allowed(bound):
    """0.05 floors and 0.95 ceilings are real tuned values; the bounds
    themselves must not be rejected as if they were malformed."""
    config, models_dir, _ = bound
    data = reread(config)
    block = data["experts"]["tools"]["serving_thresholds"]
    block["values"] = [0.0, 0.5, 1.0]
    block["by_class"] = dict(zip(TOOL_CLASSES, block["values"]))
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir)]) == 0


# -- FATAL 4: the two views of the vector disagree -------------------------

def test_by_class_disagreeing_with_values_fails(bound, capsys):
    """`values` is what is SERVED; `by_class` is what a human reads. A config
    where they differ is one where every review of it was of the wrong
    numbers."""
    config, models_dir, _ = bound
    data = reread(config)
    block = data["experts"]["tools"]["serving_thresholds"]
    block["by_class"]["beta"] = 0.90
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir)]) != 0
    assert "by_class" in capsys.readouterr().err


def test_by_class_naming_a_class_the_head_does_not_have_fails(bound):
    config, models_dir, _ = bound
    data = reread(config)
    block = data["experts"]["tools"]["serving_thresholds"]
    block["by_class"] = {"alpha": 0.15, "beta": 0.20, "delta": 0.25}
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir)]) != 0


def test_a_missing_by_class_fails(bound):
    config, models_dir, _ = bound
    data = reread(config)
    del data["experts"]["tools"]["serving_thresholds"]["by_class"]
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir)]) != 0


# -- FATAL 5: the schema version ------------------------------------------

def test_a_schema_version_the_gate_does_not_understand_fails(bound, capsys):
    """A schema 3 config may carry an invariant this gate has never heard of.
    Passing it would be asserting a check that was never written."""
    config, models_dir, payloads = bound
    make_config(config, models_dir, payloads, schema_version=3)
    assert vk.main([str(config), str(models_dir)]) != 0
    assert "schema" in capsys.readouterr().err


def test_a_future_schema_is_not_waived_by_the_opt_out(bound):
    """`optional` waives the block, not the gate's competence to judge the
    file it is reading."""
    config, models_dir, payloads = bound
    make_config(config, models_dir, payloads, schema_version=3)
    assert vk.main([str(config), str(models_dir),
                    "--serving-thresholds", "optional"]) != 0


def test_a_config_with_no_schema_version_fails(bound):
    """Every config `build_perception_config.py` has ever written declares
    one, v1 included. A file without it was not written by that script."""
    config, models_dir, payloads = bound
    make_config(config, models_dir, payloads, schema_version=None, serving=None)
    assert vk.main([str(config), str(models_dir)]) != 0


def test_a_non_integer_schema_version_fails(bound):
    config, models_dir, payloads = bound
    make_config(config, models_dir, payloads, schema_version="2")
    assert vk.main([str(config), str(models_dir)]) != 0


def test_a_serving_block_under_schema_1_fails(bound):
    """Schema 1 is the version that predates the block. A file claiming both
    is lying about one of them, and the gate cannot tell which."""
    config, models_dir, payloads = bound
    make_config(config, models_dir, payloads, schema_version=1)
    assert vk.main([str(config), str(models_dir)]) != 0


# -- FATAL 6: a block on an expert that has no thresholds ------------------

def test_the_task_expert_may_not_carry_serving_thresholds(bound):
    """The task head is an 8-way softmax with `thresholds: null`. A serving
    vector there is thresholding nothing, and its presence means the config
    was assembled by something that does not understand the two heads."""
    config, models_dir, payloads = bound
    data = reread(config)
    digest = data["experts"]["task"]["sha256"]
    data["experts"]["task"]["serving_thresholds"] = serving_block(digest)
    rewrite(config, data)
    assert vk.main([str(config), str(models_dir)]) != 0


# -- what the build log has to say ----------------------------------------

def test_the_report_names_the_weights_the_block_is_bound_to(bound, capsys):
    """The checkpoint rows print the digest they computed; the serving row
    prints the digest the cuts claim. Same evidence standard."""
    config, models_dir, payloads = bound
    vk.main([str(config), str(models_dir)])
    out = capsys.readouterr().out
    assert hashlib.sha256(payloads["tools_v2.pt"]).hexdigest() in out
    serving = [line for line in out.splitlines()
               if "serving_thresholds" in line]
    assert len(serving) == 1
    assert hashlib.sha256(payloads["tools_v2.pt"]).hexdigest() in serving[0]


def test_the_report_states_the_gain_being_shipped(bound, capsys):
    """The number is the reason the block exists. Printing it is how a build
    log answers "did this image get the re-tuned cuts?" without a re-read of
    the config."""
    config, models_dir, _ = bound
    vk.main([str(config), str(models_dir)])
    assert "0.0195" in capsys.readouterr().out


def test_the_serving_gate_runs_without_torch(bound, tmp_path):
    """Same constraint as the checkpoint gate: this runs inside `docker build`
    before anything has established that the CUDA stack works."""
    import subprocess
    config, models_dir, _ = bound
    sabotage = tmp_path / "no_torch"
    sabotage.mkdir()
    (sabotage / "torch.py").write_text(
        "raise ImportError('torch is deliberately unavailable in this test')")
    completed = subprocess.run(
        [sys.executable, str(Path(vk.__file__)), str(config), str(models_dir)],
        capture_output=True,
        env={"PYTHONPATH": str(sabotage), "PATH": "/usr/bin:/bin"})
    assert completed.returncode == 0, completed.stderr.decode()
    assert b"serving_thresholds" in completed.stdout


# -- the config that actually ships ---------------------------------------

def test_the_repository_config_passes_the_serving_gate():
    """The file in the tree is the one the image is built from. If this fails,
    the gate is wrong or the config is -- either way nothing should ship."""
    repo_config = Path(__file__).resolve().parents[1] / "config" / "perception.json"
    assert vk.failures(vk.verify_serving(repo_config)) == []


def test_the_repository_config_is_a_schema_the_gate_understands():
    repo_config = Path(__file__).resolve().parents[1] / "config" / "perception.json"
    schema = json.loads(repo_config.read_text())["schema_version"]
    assert 1 <= schema <= vk.UNDERSTOOD_SCHEMA


# ==========================================================================
# ONE GATE, THREE CALLERS
#
# The Dockerfile ships; the .def proves the contents on CHTC; the Condor build
# script gates the tarball before it is staged for download. All three must run
# the SAME script with the SAME default, or the thing validated on CHTC is not
# the thing that ships.
# ==========================================================================

CONTAINERS = Path(__file__).resolve().parents[1] / "containers"
CALLERS = {
    "Dockerfile": CONTAINERS / "Dockerfile",
    "surgvu26-submission.def": CONTAINERS / "surgvu26-submission.def",
    "build_submission.sh": CONTAINERS / "build_submission.sh",
}


@pytest.mark.parametrize("name", sorted(CALLERS))
def test_every_build_path_runs_the_gate(name):
    assert "verify_checkpoints.py" in CALLERS[name].read_text()


@pytest.mark.parametrize("name", sorted(CALLERS))
def test_every_build_path_passes_the_serving_mode(name):
    """A caller that omits the flag still gets `required` from the default --
    but then the opt-out cannot be exercised there, and the three drift on the
    one knob that decides whether the improvement ships."""
    assert "--serving-thresholds" in CALLERS[name].read_text()


@pytest.mark.parametrize("name", sorted(CALLERS))
def test_no_build_path_hardcodes_the_opt_out(name):
    """The escape hatch must be typed at build time, not baked into the
    recipe. `required` is the default every caller resolves to."""
    text = CALLERS[name].read_text()
    assert "SERVING_THRESHOLDS=required" in text or "SERVING_THRESHOLDS:-required" in text
    assert "--serving-thresholds optional" not in text
    assert '--serving-thresholds "optional"' not in text


def test_the_def_declares_a_default_for_its_build_arg():
    """`{{ VAR }}` with no `%arguments` default is a hard build failure in
    apptainer -- the ordinary build must not need a --build-arg."""
    text = CALLERS["surgvu26-submission.def"].read_text()
    if "{{" in text:
        assert "%arguments" in text
        assert "SERVING_THRESHOLDS=required" in text


def test_the_two_container_recipes_gate_the_same_paths():
    """Same config, same models dir, same script: a divergence here is an
    image validated on CHTC that is not the image that ships."""
    for name in ("Dockerfile", "surgvu26-submission.def"):
        text = CALLERS[name].read_text()
        assert "/opt/algorithm/scripts/verify_checkpoints.py" in text
        assert "/opt/algorithm/config/perception.json" in text
        assert "/opt/algorithm/models" in text


# --- the temporal scripts must convert what actually ships -------------------
#
# THE 0.048 BUG, AS A TEST. Every v4 conversion arm was built from
# tools_resnet50_x40.pt while being compared against a number that
# tools_resnet50_long.pt produced. Nothing raised: both files exist, both load
# strictly, both are 12-class tool heads. The mismatch was worth 0.0337 on its
# own and survived six hours of measurement because the only place the right
# checkpoint was written down was the reference dump's FILENAME.
#
# A default that silently disagrees with config/perception.json is the whole
# failure mode, so it gets an assertion rather than an audit.

REPO = Path(__file__).resolve().parents[1]

TEMPORAL_SCRIPTS = ("train_temporal.py", "save_temporal_init.py",
                    "verify_temporal.py")


def _served_checkpoints():
    config = json.loads((REPO / "config" / "perception.json").read_text())
    return {name: expert["checkpoint"]
            for name, expert in config["experts"].items()
            if "checkpoint" in expert}


def _module_constants(name):
    """TOOLS_2D / TASK_2D as literals, without importing torch."""
    source = (REPO / "scripts" / name).read_text()
    found = {}
    for line in source.splitlines():
        for const in ("TOOLS_2D", "TASK_2D"):
            prefix = const + " = \""
            if line.startswith(prefix):
                found[const] = line[len(prefix):].rstrip("\"")
    return found


@pytest.mark.parametrize("script", TEMPORAL_SCRIPTS)
def test_the_temporal_scripts_default_to_the_served_checkpoints(script):
    served = _served_checkpoints()
    constants = _module_constants(script)
    assert constants, "%s defines no 2D checkpoint constant to check" % script
    expected = {"TOOLS_2D": served["tools"], "TASK_2D": served["task"]}
    for const, value in constants.items():
        assert value == expected[const], (
            "%s sets %s to %r but config/perception.json serves %r. A "
            "conversion of a checkpoint the pipeline does not serve is a "
            "valid experiment about conversions; it is NOT comparable to a "
            "number the served model produced."
            % (script, const, value, expected[const]))


def test_no_temporal_script_still_points_at_a_non_served_checkpoint():
    """Catches a NEW wrong default, not just the one that has been fixed.

    The parametrised test above only checks constants it recognises by name.
    This one asserts that no .pt path appears anywhere in these scripts unless
    perception.json serves it, so a future TOOLS_2D_V2 cannot reintroduce the
    bug under a name this file has never heard of.
    """
    served = set(_served_checkpoints().values())
    for script in TEMPORAL_SCRIPTS:
        source = (REPO / "scripts" / script).read_text()
        for match in re.findall(r'"(/staging/\S*?\.pt)"', source):
            assert match in served, (
                "%s hardcodes %r, which config/perception.json does not "
                "serve. Either serve it or do not default to it."
                % (script, match))
