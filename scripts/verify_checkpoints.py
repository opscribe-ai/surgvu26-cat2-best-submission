"""Build-time gate: the image serves the weights AND the cuts the config binds.

    python scripts/verify_checkpoints.py config/perception.json models/
    python scripts/verify_checkpoints.py config/perception.json models/ \\
        --serving-thresholds optional        # build WITHOUT the re-tuned cuts

`config/perception.json` carries a sha256 and a byte count for each expert.
Checking them while the image is being BUILT, on a machine with a person
watching, is worth a great deal more than discovering a truncated copy on the
grader -- where a checkpoint that fails to load costs every case its answer and
buys back only the router's fallback string.

THE SERVING THRESHOLDS ARE THE SAME KIND OF BINDING

Schema 2 adds a `serving_thresholds` block: per-class cuts re-tuned on the CLIP
MEAN this container thresholds rather than on a frame, worth +0.0195 macro-F1
over the checkpoint's own per-frame cuts, for no retrain. The block names the
sha256 it was measured against so it cannot travel onto other weights.

`scripts/inference.py` treats a missing or unfit block as a reason to fall back
to the mirrored per-frame cuts, with a WARNING. That is right at SERVING time
-- a mediocre threshold is worth far more than a case with no answer -- and
wrong at BUILD time, where the fallback is silent by any measure that matters:
an image ships 0.0195 worse and says so only in a log line nobody reads. So
everything the serving path DEGRADES over, this gate FAILS over:

  * no block at all, on an expert that carries thresholds;
  * a block naming weights other than the ones the config binds -- including a
    block naming NOTHING, which the serving path accepts silently;
  * a vector of the wrong length, or holding something that is not a threshold;
  * `by_class` disagreeing with the `values` that are actually served;
  * a `tuned_against_checkpoint_thresholds` that is not the config's mirror,
    which makes the recorded before/after describe a comparison this image
    would not be making;
  * a `schema_version` this file does not understand, in either direction.

THE ESCAPE HATCH, AND WHY IT IS A FLAG

`build_perception_config.py --drop-serving-thresholds` exists, so building
without the block has to be possible. `--serving-thresholds optional` is that
build's counterpart: it must be typed, it is reported as WAIVED rather than OK,
and it waives ONLY the block's ABSENCE. A block that is present and unfit still
fails the build under it, because "I accept the checkpoint's own cuts" and
"stop checking what I am shipping" are not the same request.

TWO PROPERTIES THIS FILE IS BUILT AROUND

`--models-dir` semantics, exactly as scripts/inference.py uses them: the
checkpoint is located by BASENAME under the given directory. The `checkpoint`
field's `/staging/n/nkalthoff/...` path is never read. It records where the
weights were trained, which is provenance, not a location that exists inside a
container -- and resolving it would let a wrong `models-dir` pass on any
machine that happens to have /staging mounted.

No torch import. This runs inside `docker build`, before anything has
established that the CUDA stack works, and a torch import there costs seconds
and can fail for reasons that have nothing to do with the weights.
"""
import argparse
import hashlib
import json
import math
import sys
from collections import namedtuple
from pathlib import Path

# `waived` is not `ok`: a waived row passes the build but must never print as
# OK. A gate whose waiver reads like a pass is a gate nobody notices was off.
Result = namedtuple("Result", "role name path digest ok problem waived",
                    defaults=(False,))

# Hash in blocks: the checkpoints are ~82 MB each, and reading both whole would
# put 165 MB on the heap of a build step for no reason.
BLOCK = 1024 * 1024

# The highest `schema_version` whose invariants are all checked below. A config
# declaring more than this may carry a binding this file has never heard of,
# and passing it would be asserting a check that was never written.
UNDERSTOOD_SCHEMA = 2

# The version that introduced `serving_thresholds`.
SERVING_SCHEMA = 2

SERVING_MODES = ("required", "optional")

# `tuned_against_checkpoint_thresholds` and the config's mirror travel through
# JSON and through torch floats. 1e-6 is representation noise; anything larger
# is a different vector. Same tolerance build_perception_config.py uses.
THRESHOLD_TOLERANCE = 1e-6


def sha256_of(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(config_path):
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def verify(config_path, models_dir):
    """One Result per expert the config binds, in config order.

    A missing file is a FAILED result, not an exception: the report has to be
    able to say what the other expert's state was too.
    """
    config = load_config(config_path)
    models_dir = Path(models_dir)
    results = []
    for role, entry in config["experts"].items():
        name = entry.get("checkpoint_name") or Path(entry["checkpoint"]).name
        path = models_dir / name
        if not path.exists():
            results.append(Result(role, name, path, None, False,
                                  "missing from the build context"))
            continue
        digest = sha256_of(path)
        size = path.stat().st_size
        problems = []
        if digest != entry["sha256"]:
            problems.append("sha256 %s != %s" % (digest, entry["sha256"]))
        if size != entry["size_bytes"]:
            problems.append("size %d != %d" % (size, entry["size_bytes"]))
        results.append(Result(role, name, path, digest, not problems,
                              "; ".join(problems) or None))

        # ENSEMBLE MEMBERS. Checked because they are SERVED: inference.infer
        # loads every path in `ensemble` and averages it into the record, so a
        # member missing from the build context is not a degraded image, it is
        # an image whose first graded case dies inside the 10-minute budget
        # and falls back to a calibrated string.
        #
        # They are verified by EXISTENCE AND LOADABILITY rather than by digest.
        # The config carries one sha256 -- the primary's -- because that is
        # what the serving-threshold provenance is keyed to, and inventing a
        # second digest field that nothing else reads would be a checksum
        # nobody maintains. What matters here is that the file is present and
        # is a checkpoint for the same taxonomy; a member trained on other
        # classes would report every class as another one.
        for index, member in enumerate(entry.get("ensemble") or [], start=1):
            member_name = Path(member).name
            member_path = models_dir / member_name
            label = "%s[ens%d]" % (role, index)
            if not member_path.exists():
                results.append(Result(label, member_name, member_path, None,
                                      False, "ensemble member missing from "
                                      "the build context; inference.infer "
                                      "loads it and would fail at serving"))
                continue
            member_problems = []
            try:
                import torch

                blob = torch.load(str(member_path), map_location="cpu",
                                  weights_only=False)
                classes = list((blob.get("meta") or {}).get("classes")
                               or blob.get("classes") or [])
                if classes and classes != list(entry["classes"]):
                    member_problems.append(
                        "trained on %r but the config binds %r; the record is "
                        "keyed by class name, so serving this would report "
                        "every class as another one"
                        % (classes[:4], list(entry["classes"])[:4]))
            except Exception as exc:                    # noqa: BLE001
                member_problems.append("does not load: %s" % (exc,))
            results.append(Result(label, member_name, member_path,
                                  sha256_of(member_path),
                                  not member_problems,
                                  "; ".join(member_problems) or None))
    return results


def is_threshold(value):
    """A threshold is a real number a sigmoid probability can actually cross.

    JSON's `true` is an int in Python, and `"0.52"` compares and prints like a
    number without being one; both would sail through a bare `float()`.
    Outside [0, 1] is not a cut at all: above 1 silences a class for every
    frame of every case, below 0 asserts it in all of them, and either reads as
    a plausible number in a diff.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and 0.0 <= value <= 1.0


def schema_problem(config):
    """Why this config's `schema_version` is one the gate cannot reason about.

    Both directions are fatal. Too NEW and there may be a binding here that
    nothing below checks -- passing it would be asserting a check that was
    never written. ABSENT and the file was not written by
    `build_perception_config.py`, which has stamped a version since v1.
    """
    if "schema_version" not in config:
        return ("no schema_version. Every config build_perception_config.py "
                "has written declares one, v1 included, so a file without it "
                "came from somewhere this gate cannot vouch for.")
    version = config["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        return "schema_version %r is not an integer" % (version,)
    if version < 1:
        return "schema_version %r is not a version" % (version,)
    if version > UNDERSTOOD_SCHEMA:
        return ("schema_version %d, but this gate understands up to %d. A "
                "newer config may bind something it does not know to check; "
                "update scripts/verify_checkpoints.py before building."
                % (version, UNDERSTOOD_SCHEMA))
    return None


def serving_problems(role, entry, schema):
    """Every way a PRESENT serving block is unfit to ship, in config order.

    Ordered cheapest-and-most-fundamental first, so the first line of a build
    failure is the thing to fix rather than a consequence of it.
    """
    block = entry["serving_thresholds"]
    classes = entry.get("classes")
    problems = []

    if isinstance(schema, int) and not isinstance(schema, bool) \
            and schema < SERVING_SCHEMA:
        problems.append(
            "the config declares schema_version %d, which predates "
            "serving_thresholds, yet carries the block. One of the two is "
            "wrong and nothing here can tell which." % (schema,))

    if not isinstance(block, dict):
        problems.append("serving_thresholds is %s, not an object"
                        % type(block).__name__)
        return problems
    if not isinstance(classes, list) or not classes:
        problems.append("the %s entry names no classes, so a positional "
                        "threshold vector cannot be checked against anything"
                        % (role,))
        return problems

    values = block.get("values")
    if not isinstance(values, list):
        problems.append("serving_thresholds has no `values` list -- `values` "
                        "is the vector scripts/inference.py actually serves")
        return problems
    if len(values) != len(classes):
        problems.append(
            "%d serving thresholds for %d classes. They are positional -- "
            "perceive.tools_present zips them against the class list -- so a "
            "short vector thresholds the head of the taxonomy on purpose and "
            "the tail by accident." % (len(values), len(classes)))
        return problems
    unfit = [(name, value) for name, value in zip(classes, values)
             if not is_threshold(value)]
    if unfit:
        problems.append(
            "serving thresholds that are not probabilities in [0, 1]: %s"
            % ", ".join("%s=%r" % pair for pair in unfit))

    by_class = block.get("by_class")
    if not isinstance(by_class, dict):
        problems.append("serving_thresholds has no `by_class` map. It is what "
                        "a human reads when reviewing these cuts, and a block "
                        "without it was reviewed as a bare vector.")
    elif sorted(by_class) != sorted(classes):
        problems.append(
            "by_class names %r but the head has %r"
            % (sorted(by_class), sorted(classes)))
    else:
        # A `values` entry that is not a threshold at all is already reported
        # above; comparing it here would only raise on the way to saying so.
        disagree = [name for name, value in zip(classes, values)
                    if is_threshold(value)
                    and not (is_threshold(by_class[name])
                             and math.isclose(float(by_class[name]),
                                              float(value), rel_tol=0.0,
                                              abs_tol=THRESHOLD_TOLERANCE))]
        if disagree:
            problems.append(
                "by_class disagrees with the served `values` for %s. `values` "
                "is what is applied, so every review of this config read the "
                "wrong numbers." % ", ".join(sorted(disagree)))

    provenance = block.get("provenance")
    tuned_on = provenance.get("checkpoint_sha256") \
        if isinstance(provenance, dict) else None
    bound = entry.get("sha256")
    if not tuned_on:
        problems.append(
            "serving thresholds name no provenance.checkpoint_sha256. The "
            "serving path accepts that silently -- it only compares a sha256 "
            "it was GIVEN -- so cuts bound to nothing would ship unnoticed.")
    elif tuned_on != bound:
        problems.append(
            "serving thresholds were tuned on checkpoint %s but the config "
            "binds %s. A retrain moves every probability scale they were "
            "calibrated against, so these are not a better channel than the "
            "checkpoint's own cuts -- they are an unbounded one. Re-run "
            "scripts/tune_serving_thresholds.py against these weights, or "
            "rebuild with --drop-serving-thresholds."
            % (tuned_on, bound))

    mirror = entry.get("thresholds") or []
    against = block.get("tuned_against_checkpoint_thresholds")
    if not isinstance(against, list) or len(against) != len(mirror) or not all(
            is_threshold(a) and is_threshold(b)
            and math.isclose(float(a), float(b), rel_tol=0.0,
                             abs_tol=THRESHOLD_TOLERANCE)
            for a, b in zip(against, mirror)):
        problems.append(
            "serving thresholds were measured against checkpoint cuts %r, but "
            "the config mirrors %r. The recorded before/after describes a "
            "comparison this image would not be making."
            % (against, mirror))

    return problems


def verify_serving(config_path, mode=SERVING_MODES[0]):
    """One Result for the schema, plus one per expert that thresholds.

    `mode="optional"` waives ABSENCE only -- a block that is there and unfit is
    fatal either way. See the module docstring.
    """
    if mode not in SERVING_MODES:
        raise ValueError("unknown serving-threshold mode %r; expected one "
                         "of %s" % (mode, ", ".join(SERVING_MODES)))
    config = load_config(config_path)
    path = Path(config_path)
    schema = config.get("schema_version")
    problem = schema_problem(config)
    results = [Result("config", "schema_version", path,
                      None if problem else str(schema), not problem, problem)]

    for role, entry in config.get("experts", {}).items():
        block = entry.get("serving_thresholds")
        if entry.get("thresholds") is None:
            # The task head is an 8-way softmax with no cuts at all. A serving
            # vector here thresholds nothing, and its presence means whatever
            # assembled this config does not understand the two heads.
            if block is not None:
                results.append(Result(
                    role, "serving_thresholds", path, None, False,
                    "%s carries no thresholds at all, so it cannot carry "
                    "serving thresholds either" % (role,)))
            continue
        if block is None:
            if mode == "optional":
                results.append(Result(
                    role, "serving_thresholds", path, None, True,
                    "no re-tuned cuts in this config; this image will serve "
                    "%s's per-frame checkpoint cuts by request" % (role,),
                    waived=True))
            else:
                results.append(Result(
                    role, "serving_thresholds", path, None, False,
                    "%s binds no serving_thresholds. The container would "
                    "quietly apply the checkpoint's PER-FRAME cuts to the "
                    "CLIP MEAN, which measured 0.0195 macro-F1 worse. Rebuild "
                    "the config with scripts/build_perception_config.py, or "
                    "say --serving-thresholds optional to build without them."
                    % (role,)))
            continue
        problems = serving_problems(role, entry, schema)
        tuned_on = (block.get("provenance") or {}).get("checkpoint_sha256") \
            if isinstance(block, dict) else None
        results.append(Result(
            role, "serving_thresholds", path,
            tuned_on if isinstance(tuned_on, str) else None,
            not problems, "; ".join(problems) or None))
    return results


def failures(results):
    return ["%s (%s): %s" % (result.role, result.name, result.problem)
            for result in results if not result.ok]


def status(result):
    if not result.ok:
        return "FAIL: " + (result.problem or "")
    if result.waived:
        return "WAIVED: " + (result.problem or "")
    return "OK"


def serving_summary(config_path, results):
    """What the build log should say about the cuts this image will apply."""
    lines = []
    config = load_config(config_path)
    for result in results:
        if result.name != "serving_thresholds":
            continue
        if result.waived:
            lines.append(
                "WARNING: built WITHOUT the re-tuned %s serving thresholds. "
                "This image applies the checkpoint's per-frame cuts to the "
                "clip mean, which measured 0.0195 macro-F1 worse. Deliberate: "
                "--serving-thresholds optional was passed." % (result.role,))
            continue
        block = config["experts"][result.role]["serving_thresholds"]
        delta = (block.get("measurements") or {}).get("delta_macro_f1")
        lines.append(
            "%s serving thresholds bound to %s...%s"
            % (result.role, (result.digest or "?")[:12],
               "" if delta is None
               else ": %+.4f macro-F1 on the clip mean over the checkpoint's "
                    "own cuts" % (float(delta),)))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("models_dir")
    # `choices` on purpose: a typo must not fall through to whichever branch
    # happens to be permissive.
    parser.add_argument("--serving-thresholds", choices=SERVING_MODES,
                        default=SERVING_MODES[0],
                        help="`required` (default) fails the build when the "
                             "config binds no re-tuned serving thresholds; "
                             "`optional` builds without them, loudly. Neither "
                             "waives a block that is present and unfit.")
    args = parser.parse_args(argv)

    results = (verify(args.config, args.models_dir)
               + verify_serving(args.config, args.serving_thresholds))
    for result in results:
        print("%-6s %-18s %-64s %s"
              % (result.role, result.name, result.digest or "-",
                 status(result)))
    bad = failures(results)
    if bad:
        print("verification FAILED:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    print("both checkpoints match the frozen binding")
    for line in serving_summary(args.config, results):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
