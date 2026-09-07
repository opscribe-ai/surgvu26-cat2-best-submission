"""The evidence flags must be inert until they are switched on, and must
never be able to take the pipeline down when they are.

Every one of these components is new and unmeasured against BERTScore. The
container's existing contract is that a failure anywhere still writes an
answer, because a missing response scores zero while a wrong polar answer
still scores 0.7015. New evidence does not get to weaken that.

Separate from tests/test_inference.py only to keep two concurrent
workstreams out of each other's way, same precedent as
tests/test_inference_vlm.py -- the `case`/`checkpoints` fixtures are
imported from it rather than redefined, so there is one definition of what a
container run looks like.

THE `agree` BLOCK (surgvu.agreement, Task 7) has landed and is wired into
`add_evidence` -- see section 3b below. It is computed ONLY when a yolo
record exists (agreement between the CNN heads and a detector that never
ran is not a signal) and lives in its OWN try/except, independent of the
yolo block's, exactly like the variant head's independence from yolo
(section 4).
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import inference                                                    # noqa: E402
from inference import parse_args                                    # noqa: E402

# The fixtures. Imported rather than redefined -- see module docstring.
from test_inference import (                                        # noqa: E402,F401
    Case, IMAGE_SIZE, _assert_answered, _config, _write_video, checkpoints,
    case, tiny_backbone,
)


# --------------------------------------------------------------------------
# 1. the flags themselves: default off, independent of each other and of
#    --motion-v2, and --variant-head legal without --yolo
# --------------------------------------------------------------------------

def test_evidence_flags_default_off():
    args = parse_args(["--input-dir", "/in", "--output-dir", "/out"])
    assert args.yolo is False
    assert args.variant_head is False
    assert args.motion_v2 is False


def test_flags_are_independent():
    args = parse_args(["--input-dir", "/in", "--output-dir", "/out", "--yolo"])
    assert args.yolo is True
    assert args.variant_head is False


def test_variant_head_requires_yolo_or_says_why():
    """--variant-head without --yolo is legal (whole-frame fallback) and must
    not raise; the crop is an improvement, not a precondition."""
    args = parse_args(["--input-dir", "/in", "--output-dir", "/out",
                       "--variant-head"])
    assert args.variant_head is True


def test_yolo_flag_defaults_are_container_paths():
    """The defaults point at where the container's models directory would
    hold them, per the task brief -- not at anything in this repo, since
    nothing here bakes weights into an image."""
    args = parse_args(["--input-dir", "/in", "--output-dir", "/out"])
    assert args.yolo_weights == "/opt/algorithm/models/yolo_best.pt"
    assert args.yolo_repo == "/opt/algorithm/yolov5"
    assert args.variant_weights == "/opt/algorithm/models/variant_head.pt"
    assert args.variant_config == str(inference.DEFAULT_VARIANT_CONFIG)


def test_variant_config_default_is_absolute_not_relative():
    """R32. The other three evidence-flag defaults (--yolo-weights,
    --yolo-repo, --variant-weights) are absolute /opt/algorithm/... strings.
    --variant-config used to be the bare relative string
    "config/variant_head.json" -- the one flag in the set whose default did
    NOT match that pattern. A relative default resolves against whatever the
    process's cwd happens to be; the submission container's runscript does
    not guarantee inference.py runs from /opt/algorithm, so that path could
    fail to resolve there, the R18 best-effort idiom would swallow the
    failure, `perception["variant"]` would never appear, and Task 10b's gate
    would silently return None on every question -- the entire measured
    +0.0543 gain (11-case mean) reduced to zero with nothing louder than one
    WARNING line in a log nobody reads.

    Deliberately asserts absoluteness, not merely the filename: a test that
    only checked `args.variant_config.endswith("config/variant_head.json")`
    would PASS against the bug, since the buggy relative string ends with
    exactly that suffix too.

    BREAKS ON: reverting the --variant-config default back to the bare
    string "config/variant_head.json" (confirmed failing -- see the task-11
    report for the captured output).
    """
    args = parse_args(["--input-dir", "/in", "--output-dir", "/out"])
    path = Path(args.variant_config)
    assert path.is_absolute(), (
        "R32: --variant-config's default must be absolute, exactly like "
        "its three sibling evidence-flag defaults; a relative default "
        "silently disables the variant gate in the container")
    assert path.parts[-2:] == ("config", "variant_head.json")
    # And it must be REPO-derived, not merely some other absolute string
    # someone hardcoded (e.g. a second /opt/algorithm literal) -- derived is
    # what keeps it correct locally AND in the container, per DEFAULT_CONFIG
    # right above it.
    assert args.variant_config == str(inference.DEFAULT_VARIANT_CONFIG)


# --------------------------------------------------------------------------
# 2. _needle_driver_boxes: pure logic, no torch call, but the module itself
#    still requires torch to import (scripts/inference.py imports it at
#    module scope), so this lives here rather than in a torch-free file.
# --------------------------------------------------------------------------

def test_needle_driver_boxes_picks_max_confidence_per_anchor():
    """Two needle-driver detections can survive NMS in one frame; the crop
    fed to the variant head must be the more confident one, matching
    scripts/variant_sample_report.py:_needle_boxes -- NOT whichever happens
    to be last in `by_class`'s list, which is an accident of anchor order
    and NMS's own internal order, not a confidence order.

    BREAKS ON: replacing the `conf > best_conf[idx]` comparison with
    unconditional overwrite (`boxes[idx] = entry["box"]` on every entry,
    which is "last one wins" rather than "most confident one wins").
    """
    yolo_record = {
        "by_class": {
            "needle driver": [
                {"anchor_idx": 0, "conf": 0.4, "box": [0, 0, 10, 10]},
                {"anchor_idx": 0, "conf": 0.9, "box": [5, 5, 15, 15]},
                {"anchor_idx": 2, "conf": 0.6, "box": [1, 1, 2, 2]},
            ],
        },
    }
    boxes = inference._needle_driver_boxes(yolo_record)
    assert boxes == {0: [5, 5, 15, 15], 2: [1, 1, 2, 2]}


def test_needle_driver_boxes_empty_when_no_needle_driver_detections():
    assert inference._needle_driver_boxes({"by_class": {}}) == {}
    assert inference._needle_driver_boxes({"by_class": {"cadiere forceps": [
        {"anchor_idx": 0, "conf": 0.9, "box": [0, 0, 1, 1]}]}}) == {}


def test_needle_driver_boxes_tolerates_a_missing_or_none_record():
    """`add_evidence` only calls this when `yolo_record` is truthy, but the
    function itself must not require that -- a defensive caller elsewhere
    should not be able to crash it."""
    assert inference._needle_driver_boxes(None) == {}
    assert inference._needle_driver_boxes({}) == {}


# --------------------------------------------------------------------------
# fakes for Detector / VariantHead -- no real .pt checkpoint, no yolov5
# checkout, no torch model construction. Patched onto the *source* modules
# (surgvu.detect.Detector, surgvu.variant.VariantHead) rather than onto
# `inference`, because `add_evidence` imports both names locally at call
# time (`from surgvu.detect import Detector, ...`); that import resolves
# against the source module's attribute at the moment it executes, which is
# exactly the idiom tests/test_inference.py already relies on for
# `surgvu.motion.motion_record_v2`.
# --------------------------------------------------------------------------

class FakeDetector:
    """Records construction args; `.detect` returns a scripted per-anchor
    detection list set by the test via `FakeDetector.per_anchor`."""

    per_anchor = {}
    constructed = []

    def __init__(self, weights, repo_dir, conf=0.25, iou=0.45, device="cpu"):
        FakeDetector.constructed.append((weights, repo_dir, device))

    def detect(self, frames, size=640):
        return [FakeDetector.per_anchor.get(i, []) for i in range(len(frames))]


class ExplodingDetector:
    def __init__(self, *args, **kwargs):
        pass

    def detect(self, frames, size=640):
        raise RuntimeError("detector blew up")


class FakeVariantHead:
    """Records the cutoff and the crop boxes it was handed; always returns a
    decided, well-formed variant record."""

    captured_boxes = "UNSET"
    constructed_cutoff = None

    def __init__(self, weights, cutoff, device="cpu", size=224):
        FakeVariantHead.constructed_cutoff = cutoff

    def predict(self, frames, boxes=None):
        FakeVariantHead.captured_boxes = boxes
        return {"version": 1, "family": "large", "p_large": 0.9,
                "p_mega": 0.1, "cutoff": self.constructed_cutoff or 0.6,
                "decided": True}


class ExplodingVariantHead:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        raise RuntimeError("variant head blew up")


@pytest.fixture(autouse=True)
def _reset_fakes():
    FakeDetector.per_anchor = {}
    FakeDetector.constructed = []
    FakeVariantHead.captured_boxes = "UNSET"
    FakeVariantHead.constructed_cutoff = None
    yield


@pytest.fixture
def variant_config_path(tmp_path):
    path = tmp_path / "variant_head.json"
    path.write_text(json.dumps({"cutoff": 0.6, "held_out_cases": []}),
                    encoding="utf-8")
    return path


def _spy_route(monkeypatch, captured):
    """Intercept the perception dict `route()` is actually called with,
    without changing the answer it produces."""
    real_route = inference.route

    def spy(question, perception, frames, timings, vlm):
        captured["perception"] = perception
        return real_route(question, perception, frames, timings, vlm)

    monkeypatch.setattr(inference, "route", spy)


# --------------------------------------------------------------------------
# 3. --yolo: populates perception["yolo"] on success, absent on failure
# --------------------------------------------------------------------------

def test_yolo_success_populates_the_yolo_block(case, monkeypatch):
    import surgvu.detect as detect_module

    FakeDetector.per_anchor = {
        0: [{"cls": "needle driver", "conf": 0.9, "box": [1.0, 2.0, 3.0, 4.0]}],
    }
    monkeypatch.setattr(detect_module, "Detector", FakeDetector)
    captured = {}
    _spy_route(monkeypatch, captured)

    assert case.run("--yolo") == 0

    yolo = captured["perception"]["yolo"]
    assert yolo["max_conf"]["needle driver"] == pytest.approx(0.9)
    assert yolo["by_class"]["needle driver"][0]["anchor_idx"] == 0
    assert "variant" not in captured["perception"]
    assert _assert_answered(case)


def test_yolo_failure_is_swallowed(case, monkeypatch, capsys):
    """A detector that raises must not cost the case its answer, and must
    not trip the whole-pipeline FALLBACK -- the CNN heads already ran.

    BREAKS ON: removing the try/except around the yolo block in
    `add_evidence`, or narrowing it to a specific exception type that does
    not cover a plain RuntimeError.
    """
    import surgvu.detect as detect_module

    monkeypatch.setattr(detect_module, "Detector", ExplodingDetector)
    captured = {}
    _spy_route(monkeypatch, captured)

    assert case.run("--yolo") == 0

    err = capsys.readouterr().err
    assert "detector blew up" in err, "the failure must still be logged loudly"
    assert "FALLBACK" not in err, (
        "a yolo failure must not trip the whole-pipeline fallback")
    assert "yolo" not in captured["perception"], (
        "a failed detector must leave the block absent, not partial or stale")
    assert "agree" not in captured["perception"], (
        "agreement between the CNN heads and a detector that never "
        "produced a record is not a signal; a failed yolo block must never "
        "reach the agreement computation at all")
    assert _assert_answered(case)


def test_yolo_flags_reach_the_detector_constructor(case, monkeypatch):
    """--yolo-weights, --yolo-repo and the resolved device must be exactly
    what `Detector` is constructed with -- not the module's defaults,
    silently ignoring what was passed on the command line."""
    import surgvu.detect as detect_module

    monkeypatch.setattr(detect_module, "Detector", FakeDetector)

    assert case.run("--yolo", "--yolo-weights", "/tmp/example.pt",
                    "--yolo-repo", "/tmp/example-yolov5") == 0

    assert FakeDetector.constructed == [
        ("/tmp/example.pt", "/tmp/example-yolov5", "cpu")]


# --------------------------------------------------------------------------
# 3b. --yolo alone (no --variant-head): populates perception["agree"] too,
#     its own try/except independent of the yolo block's.
# --------------------------------------------------------------------------

def test_yolo_success_also_populates_the_agree_block(case, monkeypatch):
    """`surgvu.agreement` (Task 7) has landed; `add_evidence` now computes
    `perception["agree"]` whenever a yolo record exists, reusing the SAME
    serving thresholds `infer()` already derived for the tools head. This
    replaces the prior `test_yolo_flag_alone_never_populates_agree`, which
    pinned the pre-Task-7 omission -- see this file's module docstring.
    """
    import surgvu.detect as detect_module

    FakeDetector.per_anchor = {
        0: [{"cls": "needle driver", "conf": 0.9, "box": [1.0, 2.0, 3.0, 4.0]}],
    }
    monkeypatch.setattr(detect_module, "Detector", FakeDetector)
    captured = {}
    _spy_route(monkeypatch, captured)

    assert case.run("--yolo") == 0

    assert "yolo" in captured["perception"]
    agree = captured["perception"]["agree"]
    assert agree["version"] == 1
    assert 0.0 <= agree["tool_agreement"] <= 1.0
    assert _assert_answered(case)


def test_agree_not_computed_without_yolo(case, monkeypatch):
    """Requirement 2: agreement between the CNN heads and a detector that
    never ran is meaningless, so `add_evidence` must not even attempt it
    when `--yolo` was not passed (independent of --variant-head)."""
    import surgvu.variant as variant_module

    monkeypatch.setattr(variant_module, "VariantHead", FakeVariantHead)
    captured = {}
    _spy_route(monkeypatch, captured)

    assert case.run() == 0

    assert "yolo" not in captured["perception"]
    assert "agree" not in captured["perception"]
    assert _assert_answered(case)


def test_agree_failure_is_swallowed_and_yolo_block_survives(
        case, monkeypatch, capsys):
    """The agreement computation is its OWN try/except, independent of the
    yolo block's (same reasoning as
    test_variant_head_failure_is_swallowed_and_yolo_block_survives below): a
    raising `agreement_record` must not discard a yolo block that already
    succeeded, and must not trip the whole-pipeline FALLBACK.

    BREAKS ON: folding the agreement computation into the yolo block's own
    try/except, so a raising `agreement_record` also discards `yolo`.
    """
    import surgvu.agreement as agreement_module
    import surgvu.detect as detect_module

    FakeDetector.per_anchor = {
        0: [{"cls": "needle driver", "conf": 0.9, "box": [1.0, 2.0, 3.0, 4.0]}],
    }
    monkeypatch.setattr(detect_module, "Detector", FakeDetector)

    def _exploding_agreement_record(*args, **kwargs):
        raise RuntimeError("agreement blew up")

    monkeypatch.setattr(agreement_module, "agreement_record",
                        _exploding_agreement_record)
    captured = {}
    _spy_route(monkeypatch, captured)

    assert case.run("--yolo") == 0

    err = capsys.readouterr().err
    assert "agreement blew up" in err
    assert "FALLBACK" not in err
    assert "yolo" in captured["perception"], (
        "yolo succeeded independently and must not be discarded because "
        "the agreement computation failed afterward")
    assert "agree" not in captured["perception"]
    assert _assert_answered(case)


# --------------------------------------------------------------------------
# 4. --variant-head: whole-frame fallback without --yolo, crop with it,
#    failure swallowed, and additivity with a successful yolo block
# --------------------------------------------------------------------------

def test_variant_head_without_yolo_uses_whole_frame_fallback(
        case, monkeypatch, variant_config_path):
    """--variant-head alone must not raise, and must hand the head `None`
    for boxes -- VariantHead.predict's own whole-frame path -- because there
    is no detector run to have produced a crop.

    BREAKS ON: `boxes = _needle_driver_boxes(yolo_record) if yolo_record
    else {}` changed so it raises (e.g. dropping the `if yolo_record else`
    guard and calling `_needle_driver_boxes(None)` -- which happens to
    still work, so this test is the one that would catch a REQUIRED-yolo
    regression more directly than a crash would: it asserts `None`
    specifically, not just "did not raise").
    """
    import surgvu.variant as variant_module

    monkeypatch.setattr(variant_module, "VariantHead", FakeVariantHead)

    assert case.run("--variant-head", "--variant-config",
                    str(variant_config_path)) == 0

    assert FakeVariantHead.captured_boxes is None
    assert _assert_answered(case)


def test_variant_head_with_yolo_receives_the_max_confidence_crop(
        case, monkeypatch, variant_config_path):
    """End-to-end version of test_needle_driver_boxes_picks_max_confidence_
    per_anchor: the box `add_evidence` actually hands VariantHead.predict
    through the full --yolo + --variant-head pipeline.
    """
    import surgvu.detect as detect_module
    import surgvu.variant as variant_module

    FakeDetector.per_anchor = {
        0: [
            {"cls": "needle driver", "conf": 0.4, "box": [0.0, 0.0, 10.0, 10.0]},
            {"cls": "needle driver", "conf": 0.9, "box": [5.0, 5.0, 15.0, 15.0]},
        ],
    }
    monkeypatch.setattr(detect_module, "Detector", FakeDetector)
    monkeypatch.setattr(variant_module, "VariantHead", FakeVariantHead)

    assert case.run("--yolo", "--variant-head", "--variant-config",
                    str(variant_config_path)) == 0

    assert FakeVariantHead.captured_boxes == {0: [5.0, 5.0, 15.0, 15.0]}
    assert _assert_answered(case)


def test_variant_head_reads_the_fitted_cutoff_from_its_config(
        case, monkeypatch, variant_config_path):
    """The cutoff served must be the one in --variant-config, not a
    hardcoded value -- it is what keeps the abstention point tied to the
    number it was measured at (see the flag's own help text)."""
    import surgvu.variant as variant_module

    monkeypatch.setattr(variant_module, "VariantHead", FakeVariantHead)

    assert case.run("--variant-head", "--variant-config",
                    str(variant_config_path)) == 0

    assert FakeVariantHead.constructed_cutoff == pytest.approx(0.6)


def test_variant_head_success_populates_the_variant_block(
        case, monkeypatch, variant_config_path):
    import surgvu.variant as variant_module

    monkeypatch.setattr(variant_module, "VariantHead", FakeVariantHead)
    captured = {}
    _spy_route(monkeypatch, captured)

    assert case.run("--variant-head", "--variant-config",
                    str(variant_config_path)) == 0

    variant = captured["perception"]["variant"]
    assert variant["decided"] is True
    assert variant["family"] == "large"
    assert "yolo" not in captured["perception"]
    assert _assert_answered(case)


def test_variant_head_failure_is_swallowed_and_yolo_block_survives(
        case, monkeypatch, capsys, variant_config_path):
    """The variant head and the detector are independently best-effort: a
    failed variant head must not discard a yolo block that already
    succeeded, and must not trip the whole-pipeline FALLBACK.

    BREAKS ON: moving `perception["yolo"] = yolo_record` so it happens after
    (rather than before) the variant block, or wrapping both blocks in one
    shared try/except so a variant failure also discards yolo.
    """
    import surgvu.detect as detect_module
    import surgvu.variant as variant_module

    monkeypatch.setattr(detect_module, "Detector", FakeDetector)
    monkeypatch.setattr(variant_module, "VariantHead", ExplodingVariantHead)
    captured = {}
    _spy_route(monkeypatch, captured)

    assert case.run("--yolo", "--variant-head", "--variant-config",
                    str(variant_config_path)) == 0

    err = capsys.readouterr().err
    assert "variant head blew up" in err
    assert "FALLBACK" not in err
    assert "yolo" in captured["perception"], (
        "yolo succeeded independently and must not be discarded because "
        "the variant head failed afterward")
    assert "variant" not in captured["perception"]
    assert _assert_answered(case)


def test_variant_head_missing_config_is_swallowed(
        case, monkeypatch, capsys, tmp_path):
    """A --variant-config that does not exist is exactly as much a failure
    as a raising head -- json.loads/.read_text raise inside the same
    try/except, so the block is simply absent."""
    import surgvu.variant as variant_module

    monkeypatch.setattr(variant_module, "VariantHead", FakeVariantHead)
    missing = tmp_path / "does-not-exist.json"

    assert case.run("--variant-head", "--variant-config", str(missing)) == 0

    err = capsys.readouterr().err
    assert "WARNING: the variant head failed" in err
    assert "FALLBACK" not in err
    assert _assert_answered(case)


# --------------------------------------------------------------------------
# 5. the safety property, exercised at the inference.py level (see also
#    tests/test_inference_motion_v2.py, which asserts the same property
#    inside clip_record itself)
# --------------------------------------------------------------------------

def test_both_flags_off_perception_carries_neither_block(case, monkeypatch):
    """With --yolo and --variant-head both absent, `add_evidence` must be a
    complete no-op: `perception` reaching `route()` carries neither key,
    which is what makes today's answers byte-identical after this task
    lands. This is the property the whole gate in
    src/surgvu/router.py:_variant_gate_answer depends on staying true until
    someone deliberately turns a flag on.
    """
    captured = {}
    _spy_route(monkeypatch, captured)

    assert case.run() == 0

    assert "yolo" not in captured["perception"]
    assert "variant" not in captured["perception"]
    assert "agree" not in captured["perception"]
    assert _assert_answered(case)
