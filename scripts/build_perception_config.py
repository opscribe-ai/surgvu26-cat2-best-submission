"""Checkpoints -> `config/perception.json`, the frozen perception binding.

WHY THIS FILE EXISTS
--------------------
Everything the serving path needs to run an expert -- which weights, at which
resolution, with which class ordering, behind which per-class threshold --
currently lives in a `meta` block inside an 80 MB `.pt` file. That is fine for
a training script, which has just written it, and wrong for a submission
container, which would have to open both checkpoints to find out what it is
about to route to, and would have no way to state what it bound. This script
reads those `meta` blocks once, validates them against the taxonomy the router
actually reads, and freezes the result into one small JSON file.

FAIL LOUDLY, WRITE NOTHING
--------------------------
A missing checkpoint aborts. It does NOT write the half of the config it could
build: a partial config is worse than no config, because the container starts,
loads the expert that exists, and dies -- or worse, silently answers from one
expert -- when it reaches the one that does not. Plan 2 Task 10 states this
explicitly and the whole file is arranged around it: nothing is written until
both experts have been read and checked.

WHICH CHECKPOINT DID IT BIND?
-----------------------------
`tools_v2.pt` and `task_v2.pt` are produced by retraining jobs that may not
have landed when this runs, so each expert takes an ordered candidate list and
falls back to the `*_efficientnet_v2_s.pt` files. The config records the path,
the basename, whether it was the primary or the fallback, the file size and a
sha256 -- because "the file called tools_v2.pt" is not an identity when a job
is still writing files of that name, and the recorded validation metrics are
only meaningful next to the bytes they were measured on.

TWO THRESHOLD VECTORS, AND WHY
------------------------------
`thresholds` mirrors the checkpoint. `train_tools.py` tuned it on PER-FRAME
validation probabilities and `scripts/inference.py` compares it against the
checkpoint on every run, so an accidental mismatch -- a retrain that landed
under the same filename, a hand edit -- is still caught loudly. It is a drift
detector, not a serving parameter.

`serving_thresholds` is what the container APPLIES. The container thresholds
the 16-frame CLIP MEAN, not a frame, and the per-frame optimum is not the
clip-level optimum. `scripts/tune_serving_thresholds.py` measures that and
emits a report; this script embeds it together with everything needed to
judge it later -- the weights it was tuned on, the split, the date, and the
before/after macro-F1.

That block is CARRIED FORWARD across rebuilds, so re-running this script does
not silently revert a deliberate calibration. It is carried only onto the same
weights: the block names the sha256 it was tuned against, and binding a
checkpoint that does not match it ABORTS rather than shipping cuts calibrated
for a model that is no longer there. Re-tune, or say `--drop-serving-thresholds`.

    python3 scripts/build_perception_config.py            # defaults
    python3 scripts/build_perception_config.py \
        --tools-checkpoint /staging/n/nkalthoff/surgvu26/models/tools_v3.pt
    python3 scripts/build_perception_config.py \
        --tools-serving-thresholds outputs/serving_thresholds.json
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.taxonomy import TASK_CLASSES, TOOL_CLASSES              # noqa: E402

MODELS = "/staging/n/nkalthoff/surgvu26/models"
REPO = Path(__file__).resolve().parents[1]

# The sampling policy, frozen here so the container does not carry it as a
# default argument in two places. 16 frames evenly spaced across the clip is
# what `perceive.DEFAULT_FRAMES` uses; 512 is the size the training shards
# stored, so a serving frame is resampled exactly as a training frame was.
DEFAULT_FRAMES = 16
DEFAULT_SIZE = 512

# 2 adds `serving_thresholds` to an expert entry. A version-1 config is still
# served correctly -- `inference.serving_thresholds` falls back to the mirror
# when the field is absent -- so this marks what a reader may expect to find,
# not a break.
SCHEMA_VERSION = 2

SERVING_APPLIES_TO = ("the clip mean predict.aggregate_window forms over "
                      "decode.frames evenly spaced frames")
SERVING_NOTE = (
    "DELIBERATE divergence from `thresholds` above, which mirrors the "
    "checkpoint so that an ACCIDENTAL mismatch is still caught by "
    "scripts/inference.py. The container applies THESE.")

# role -> (expected class list, activation, does it carry thresholds?)
#
# The activation is part of the binding, not an implementation detail: the
# tool head is 12 independent sigmoids and the task head is an 8-way softmax,
# and serving either through the other's activation produces well-formed
# numbers that mean nothing.
ROLES = {
    "tools": (TOOL_CLASSES, "sigmoid", True),
    "task": (TASK_CLASSES, "softmax", False),
}

# Keys consumed structurally. Everything else in `meta` is a measurement and
# is copied into `metrics` verbatim, so a metric added by a later training run
# reaches the config without an edit here.
STRUCTURAL_KEYS = ("classes", "backbone", "image_size", "frames_per_window",
                   "thresholds")


def _fail(message):
    raise SystemExit(message)


def resolve_checkpoint(role, candidates):
    """(path, "primary"|"fallback") -- the first candidate that exists.

    Missing is fatal, and the message names every path tried: the usual cause
    is a retrain that has not finished writing, and "which files did you look
    for" is the only question worth answering at that moment.
    """
    candidates = [Path(candidate) for candidate in candidates]
    for index, candidate in enumerate(candidates):
        if candidate.exists():
            return candidate, ("primary" if index == 0 else "fallback")
    _fail("%s checkpoint missing: none of %s exists. Refusing to write a "
          "partial perception config -- the container would route to a model "
          "that does not exist."
          % (role, ", ".join(str(candidate) for candidate in candidates)))


def read_meta(path):
    """The `meta` block of a checkpoint. Imported lazily: this script is
    mostly path handling and torch costs seconds to import."""
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "meta" not in payload:
        _fail("%s carries no 'meta' block; it is not one of our checkpoints."
              % (path,))
    return payload["meta"]


def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def serving_block_from_report(report):
    """A `scripts/tune_serving_thresholds.py` report -> the config's block.

    Nothing is trusted here that can be recomputed: `by_class` is rebuilt from
    the values rather than copied, so a report whose two views disagree cannot
    ship the disagreement. What cannot be recomputed -- which weights, which
    split, which aggregation, the before/after -- is carried verbatim, and is
    checked against the bound checkpoint in `expert_entry`.
    """
    for key in ("classes", "serving_thresholds", "checkpoint_thresholds",
                "provenance"):
        if key not in report:
            _fail("the serving-threshold report has no %r. It is not one "
                  "produced by scripts/tune_serving_thresholds.py." % (key,))
    values = [float(value) for value in report["serving_thresholds"]]
    classes = list(report["classes"])
    if classes != list(TOOL_CLASSES):
        _fail("the serving-threshold report was tuned over %r, not the "
              "taxonomy %r. The vector is positional, so embedding it would "
              "threshold every class as another one."
              % (classes, list(TOOL_CLASSES)))
    return {
        "values": values,
        "by_class": dict(zip(classes, values)),
        "applies_to": SERVING_APPLIES_TO,
        "note": SERVING_NOTE,
        "tuned_against_checkpoint_thresholds":
            [float(value) for value in report["checkpoint_thresholds"]],
        "measurements": report.get("measurements", {}),
        "provenance": report["provenance"],
    }


def carried_serving_block(out_path, role):
    """The serving block an existing config already carries, if any.

    This is what stops a routine rebuild from silently reverting a deliberate
    calibration. It is deliberately dumb -- it copies the block and validates
    nothing -- because `expert_entry` is about to check it against the
    checkpoint it is being carried onto, and one validation site is easier to
    trust than two.
    """
    out_path = Path(out_path)
    if not out_path.exists():
        return None
    try:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    except ValueError:
        print("note: %s is not readable JSON; building from the checkpoints "
              "alone" % out_path)
        return None
    block = (existing.get("experts", {}).get(role, {})
             or {}).get("serving_thresholds")
    if block:
        print("%-5s carrying forward the serving thresholds already in %s"
              % (role, out_path))
    return block or None


def _check_serving_block(role, path, block, classes, thresholds, digest):
    """Refuse a serving vector that does not belong to these weights."""
    values = [float(value) for value in (block.get("values") or [])]
    if len(values) != len(classes):
        _fail("%s serving thresholds: %d values for %d classes. They are "
              "positional -- `perceive.tools_present` zips them against the "
              "class list -- so a short vector would leave the tail of the "
              "taxonomy on the checkpoint's cuts."
              % (role, len(values), len(classes)))

    tuned_on = (block.get("provenance") or {}).get("checkpoint_sha256")
    if tuned_on != digest:
        _fail(
            "%s serving thresholds were tuned on checkpoint %s but this "
            "config binds %s (%s). A retrain moves every probability scale "
            "they were calibrated against, so carrying them forward would "
            "ship cuts for a model that is no longer here. Re-run "
            "scripts/tune_serving_thresholds.py against these weights, or "
            "pass --drop-serving-thresholds to serve the checkpoint's own."
            % (role, tuned_on, digest, path))

    against = [float(value) for value in
               block.get("tuned_against_checkpoint_thresholds") or []]
    if len(against) != len(thresholds) or not all(
            math.isclose(a, b, rel_tol=0.0, abs_tol=1e-6)
            for a, b in zip(against, thresholds)):
        _fail(
            "%s serving thresholds were measured against checkpoint cuts %r, "
            "but %s carries %r. The report's before/after numbers describe a "
            "comparison this config would not be making."
            % (role, against, path, thresholds))

    checked = dict(block)
    checked["values"] = values
    checked["by_class"] = dict(zip(classes, values))
    checked.setdefault("applies_to", SERVING_APPLIES_TO)
    checked.setdefault("note", SERVING_NOTE)
    return checked


def expert_entry(role, path, source, meta, serving=None):
    """One expert's binding, validated against the taxonomy the router reads."""
    if role not in ROLES:
        _fail("unknown expert role %r; expected one of %s"
              % (role, sorted(ROLES)))
    expected, activation, needs_thresholds = ROLES[role]
    path = Path(path)

    for key in ("classes", "backbone", "image_size", "frames_per_window"):
        if key not in meta:
            _fail("%s checkpoint %s has no %r in its meta block. The serving "
                  "path reads this config instead of the checkpoint, so a "
                  "value that is not here is a value nothing can supply."
                  % (role, path, key))

    classes = list(meta["classes"])
    if classes != list(expected):
        _fail("%s checkpoint %s does not match the taxonomy: got %r, expected "
              "%r. The perception record is keyed by class NAME, so a head "
              "trained on another ordering reports every class as another one."
              % (role, path, classes, list(expected)))

    thresholds = None
    if needs_thresholds:
        if "thresholds" not in meta:
            _fail("%s checkpoint %s carries no 'thresholds'. Refusing to "
                  "invent one: a default 0.5 produces a full, plausible "
                  "tools_present list from an untuned cutoff." % (role, path))
        thresholds = [float(value) for value in meta["thresholds"]]
        if len(thresholds) != len(classes):
            _fail("%s checkpoint %s has %d thresholds for %d classes. They "
                  "are positional -- `perceive.tools_present` zips them "
                  "against the class list -- so a short list would leave the "
                  "tail of the taxonomy unthresholded."
                  % (role, path, len(thresholds), len(classes)))

    if serving is not None and not needs_thresholds:
        _fail("%s takes no thresholds at all, so it cannot take serving "
              "thresholds either." % (role,))

    digest = sha256(path)
    entry = {
        "role": role,
        "checkpoint": str(path),
        "checkpoint_name": path.name,
        "source": source,
        "sha256": digest,
        "size_bytes": path.stat().st_size,
        "backbone": str(meta["backbone"]),
        "image_size": int(meta["image_size"]),
        "frames_per_window": int(meta["frames_per_window"]),
        "activation": activation,
        "classes": classes,
        "thresholds": thresholds,
        "metrics": {key: value for key, value in meta.items()
                    if key not in STRUCTURAL_KEYS},
    }
    if thresholds is not None:
        entry["thresholds_by_class"] = dict(zip(classes, thresholds))
    if serving is not None:
        entry["serving_thresholds"] = _check_serving_block(
            role, path, serving, classes, thresholds, digest)
    return entry


def build_config(tools_path, tools_source, tools_meta,
                 task_path, task_source, task_meta,
                 frames=DEFAULT_FRAMES, size=DEFAULT_SIZE,
                 tools_serving=None):
    """The whole config, or an exception. Never half of it."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "scripts/build_perception_config.py",
        "decode": {"frames": int(frames), "size": int(size)},
        "experts": {
            "tools": expert_entry("tools", tools_path, tools_source, tools_meta,
                                  serving=tools_serving),
            "task": expert_entry("task", task_path, task_source, task_meta),
        },
    }


def candidates_for(role, explicit, models_dir):
    """Explicit path if given, else v2 then the pre-retrain checkpoint."""
    if explicit:
        return [Path(explicit)]
    models_dir = Path(models_dir)
    return [models_dir / ("%s_v2.pt" % role),
            models_dir / ("%s_efficientnet_v2_s.pt" % role)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", default=MODELS,
                        help="directory holding the checkpoints (default %s)"
                             % MODELS)
    parser.add_argument("--tools-checkpoint",
                        help="explicit path; overrides the v2/v1 search")
    parser.add_argument("--task-checkpoint",
                        help="explicit path; overrides the v2/v1 search")
    parser.add_argument("--out", default=str(REPO / "config" / "perception.json"))
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help="frames sampled evenly across a clip at serving")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE,
                        help="size frames are prepared at before the model's "
                             "own resize to image_size")
    serving = parser.add_mutually_exclusive_group()
    serving.add_argument("--tools-serving-thresholds",
                         help="a scripts/tune_serving_thresholds.py report to "
                              "embed as the tool expert's serving thresholds")
    serving.add_argument("--drop-serving-thresholds", action="store_true",
                         help="do not carry an existing config's serving "
                              "thresholds forward; serve the checkpoint's own "
                              "per-frame cuts instead")
    args = parser.parse_args(argv)

    resolved = {}
    for role, explicit in (("tools", args.tools_checkpoint),
                           ("task", args.task_checkpoint)):
        path, source = resolve_checkpoint(
            role, candidates_for(role, explicit, args.models_dir))
        resolved[role] = (path, source, read_meta(path))
        print("%-5s %-8s %s" % (role, source, path))

    # Carrying forward is the DEFAULT. A rebuild is a routine act -- someone
    # re-runs this after touching a path or a frame count -- and a deliberate
    # calibration that a routine act silently reverts is not a calibration.
    if args.drop_serving_thresholds:
        tools_serving = None
        print("tools dropping any serving thresholds by request")
    elif args.tools_serving_thresholds:
        report = json.loads(Path(args.tools_serving_thresholds)
                            .read_text(encoding="utf-8"))
        tools_serving = serving_block_from_report(report)
        print("tools serving thresholds from %s"
              % args.tools_serving_thresholds)
    else:
        tools_serving = carried_serving_block(args.out, "tools")

    config = build_config(*resolved["tools"], *resolved["task"],
                          frames=args.frames, size=args.size,
                          tools_serving=tools_serving)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("wrote %s" % out)
    for role, entry in config["experts"].items():
        print("  %-5s image_size=%s frames_per_window=%s macro_f1=%s"
              % (role, entry["image_size"], entry["frames_per_window"],
                 entry["metrics"].get("macro_f1")))
    block = config["experts"]["tools"].get("serving_thresholds")
    if block:
        measured = block.get("measurements", {})
        print("  tools SERVING thresholds applied: %s"
              % [round(value, 2) for value in block["values"]])
        print("        clip-mean macro-F1 %s -> %s"
              % (measured.get("clip_mean_checkpoint_thresholds", {})
                 .get("macro_f1"),
                 measured.get("clip_mean_serving_thresholds", {})
                 .get("macro_f1")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
