"""The submission entrypoint: one video + one question -> one answer file.

    read   /input/endoscopic-robotic-surgery-video.mp4
    read   /input/visual-context-question.json      <- a JSON-encoded STRING
    write  /output/visual-context-response.json     <- a JSON-encoded STRING

Both JSON files hold encoded strings, not objects and not raw text. The
question therefore needs `json.load`, and the answer "Yes" is written as the
four bytes `"Yes"` INCLUDING the quotation marks. Emitting bare `Yes` is
malformed JSON and fails the case however right the answer was.

THE ONE RULE THIS FILE IS BUILT AROUND
--------------------------------------
It must never fail to produce an answer. Measured against the official metric
(BERTScore-F1, roberta-large, max over five references):

    a WRONG yes/no answer                      0.7015
    a plausible generic answer, open question   0.35 - 0.48
    an empty or missing response                nothing at all
                                                (an empty string crashes the
                                                 scorer outright)

So an unreadable video, zero decoded frames, a missing checkpoint, a CUDA
error or a bug of our own must still leave a valid, non-empty response behind.
Every one of those is a ~0.7 case that a crash converts into a 0. The pipeline
is wrapped accordingly: failures are logged loudly to stderr, the router's
calibrated fallback is written, and the process exits 0.

The fallback is INTENT-AWARE, and only intent-aware. Most questions take the
calibrated floor -- "Yes" if polar, the generic sentence otherwise -- because
re-routing them against an empty perception record makes them worse, not
better: an empty record answers "No" to every presence question, gold polar
answers in this corpus skew Yes, and a wrong polar answer only costs 0.2985.

But some intents never consulted perception in the first place. "What is the
purpose of using forceps?" is answered from the tool the question names, and
"What kind of procedure is this?" from a constant. For those, listed in the
router's `PERCEPTION_INDEPENDENT_INTENTS`, a dead video costs the answer
nothing -- so the fallback routes them against an empty record and keeps the
answer a healthy run would have given. On a purpose question that is 1.0000
instead of 0.35-0.48. See `fallback_answer`.

BUDGET
------
10 minutes per case INCLUDING container start, one case per container run, so
model load is not amortised. Every stage is timed to stderr for that reason.

DEVICE
------
CUDA when available, CPU otherwise, and a failed device is retried once on CPU
-- CPU is slower than a T4 and enormously faster than a fallback string. The
deployment instance is either No GPU or a single T4 (sm_75): no bf16, no
FlashAttention-2. Nothing here assumes either; the CNNs run in fp32.

THE VLM, AND WHY IT IS OFF
--------------------------
An Evidence VLM (`surgvu.evidence_vlm`) can be attached at `build_vlm`, and it
is OFF unless `--vlm` is passed. Unlike the router -- eleven hardcoded
intents, never abstaining -- the VLM reads the evidence packet (the CNN
probabilities, the YOLO detections WITH their timestamps, the motion vector
as calibrated language, the variant call) alongside a handful of decoded
frames, and drafts its OWN answer via an adaptive-confidence sampler
(`evidence_vlm.adaptive_confidence_sample`). `surgvu.arbiter` is the single
decision point between that draft and the router's answer, under a policy
named in `config/arbiter.json` (`--arbiter-mode` overrides it, defaulting to
whatever the file says -- never a hardcoded string here). The shipped policy,
`challenger`, drafts a VLM answer for EVERY question -- not just the ones the
router cannot classify -- and lets it override whenever the VLM's own
self-consistency confidence clears a floor; see `arbiter.py`'s module
docstring for the measurement this rests on.

The shipped weights are 4-bit NF4 (bitsandbytes), which is CUDA-only. On a
No-GPU deployment draw it is structurally impossible to run, not merely
undesirable, so `EvidenceVlmHandle.available()` is checked BEFORE any
`transformers` import and the router's answer stands untouched -- see
`try_vlm_result`. Every other failure mode (a missing checkpoint directory, an
OOM, a malformed generation) is caught at the same call site: a traceback to
stderr, a WARNING, and the router's answer stands. See the SEAM block below.

THE EVIDENCE PACKET MAY OR MAY NOT REACH THE VLM'S PROMPT -- ONE SWITCH
DECIDES, AND IT MUST MATCH WHATEVER `scripts/train_vlm.py` TRAINED AGAINST.
`EvidenceVlmHandle.sample` renders the VLM's prompt through the same
`evidence_vlm.build_sampling_prompt` that `train_vlm.render_training_prompt`
calls; `config/arbiter.json`'s `vlm_evidence_context` key (overridable
per-run by `--vlm-evidence-context`/`--no-vlm-evidence-context`) is the one
thing that decides whether it is called with the real evidence packet or an
empty one. It defaults to False (bare) because that is what the weights we
can actually train today were trained against -- see
`DEFAULT_VLM_EVIDENCE_CONTEXT` and the SEAM block below for the full
reasoning, and `scripts/train_vlm.py`'s module docstring ("THE EVIDENCE
PACKET IS DELIBERATELY EMPTY AT TRAIN TIME") for the other half of this
coupling.
"""
import argparse
import functools
import json
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                                                   # noqa: E402
import torch                                                        # noqa: E402

from surgvu.aggregate import AGGREGATORS                             # noqa: E402
from surgvu.arbiter import (                                         # noqa: E402
    MODE_CHALLENGER, MODE_FALLBACK, MODE_PRIMARY, arbitrate,
    load_config as load_arbiter_config,
)
from surgvu.evidence_vlm import ConfidenceResult                    # noqa: E402 (a plain dataclass; torch-free)
from surgvu.perceive import (clip_record, decode_clip,              # noqa: E402
                             decode_clip_bursts, load_expert)
from surgvu.predict import predict_window_frames                    # noqa: E402
from surgvu.preprocess import prepare_frame                         # noqa: E402 (used via decode_clip; imported so the blur is visible here)
from surgvu.router import (                                         # noqa: E402
    FALLBACK_OPEN, FALLBACK_POLAR,
    PERCEPTION_INDEPENDENT_INTENTS, answer_question, classify_question,
    finalize_answer, is_polar_question,
)

# The interface slugs. Exact filenames, from the challenge's algorithm
# interface definition (docs/submission_interface.md).
VIDEO_NAME = "endoscopic-robotic-surgery-video.mp4"
QUESTION_NAME = "visual-context-question.json"
RESPONSE_NAME = "visual-context-response.json"

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "config" / "perception.json"
# R32: derived, not a bare relative string, and for the identical reason
# DEFAULT_CONFIG is derived above. A relative "config/variant_head.json"
# resolves against whatever the process's cwd happens to be; the submission
# container copies config/ to /opt/algorithm/config and its runscript does
# not guarantee inference.py runs from /opt/algorithm. A relative default
# there fails to load, the R18 best-effort idiom swallows that failure, the
# `variant` block never appears, and Task 10b's gate silently returns None
# on every question -- the entire measured +0.0543 gain becomes zero with
# nothing louder than one WARNING line in a log nobody reads. REPO-relative
# resolves correctly in both places: locally it's the repo root, and in the
# container inference.py sits at /opt/algorithm/scripts/inference.py so
# parents[1] is /opt/algorithm.
DEFAULT_VARIANT_CONFIG = REPO / "config" / "variant_head.json"

# --yolo-weights/--variant-weights already default to /opt/algorithm/models/*
# so the ENTRYPOINT needs no extra path flag -- see build_submission.sh's
# COPY layout. This mirrors that: an in-container path, not
# evidence_vlm.DEFAULT_MODEL_DIR's bare HF hub id ("Qwen/Qwen2.5-VL-7B-
# Instruct"), which would try a network fetch this offline container's
# HF_HUB_OFFLINE=1 immediately refuses.
#
# REPO-DERIVED, like DEFAULT_VARIANT_CONFIG right above -- and for the
# identical R32 reason, not merely for symmetry. A bare literal
# "/opt/algorithm/models/qwen25vl-7b-nf4" happens to be correct INSIDE the
# container, which is exactly what made the old form of this bug easy to
# miss: it never fails to resolve, it just silently has no local
# counterpart, so a local run (a smoke test on this login node, a future
# --vlm-model-dir pointed at a locally-staged checkpoint) can never reach
# the weights the container ships, and there is nothing to catch that short
# of noticing by hand. REPO-relative resolves correctly in both places, the
# same as DEFAULT_CONFIG/DEFAULT_VARIANT_CONFIG: locally it is
# <repo_root>/models/qwen25vl-7b-nf4, and in the container inference.py sits
# at /opt/algorithm/scripts/inference.py so parents[1] is /opt/algorithm --
# matching containers/build_submission.sh's own basename for the staged
# directory (see that script's VLM_MODEL_SRC block).
#
# THE DIRECTORY DOES NOT EXIST IN THE IMAGE YET, AND -- UNLIKE THE CNN
# CHECKPOINTS -- IS NOT STAGED BY DEFAULT EITHER. See this task's report
# (docs/design/2026-08-24-v5-evidence-pipeline/
# vlm-weight-staging-report.md) and docs/container_build.md for why: the
# only base-model artifact that exists on disk today is the full fp16
# Qwen2.5-VL-7B-Instruct (measured 16 GB), which alone blows the documented
# 10 GB image ceiling by roughly 2x, and `src/surgvu/evidence_vlm.py`'s
# loader (frozen; this project does not modify src/) neither requests
# on-load quantisation nor applies a LoRA adapter separately, so shipping it
# would not even run the fine-tune. A pre-quantised NF4 checkpoint would
# likely fit, but none exists for this model+adapter combination -- see the
# report for the full accounting. Until the controller resolves that,
# every AutoProcessor/AutoModelForImageTextToText.from_pretrained call
# against this path fails fast and offline, and the R18 wrapper around
# try_vlm_result absorbs that exactly like a missing yolo/variant checkpoint
# would -- see add_evidence's docstring for the identical pattern. Passing
# --vlm before those weights are staged is therefore a safe no-op (a CUDA
# device tries, fails fast, and falls back; a No-GPU device never tries at
# all), not a way to crash a case.
DEFAULT_VLM_MODEL_DIR = REPO / "models" / "qwen25vl-7b-nf4"

# ---------------------------------------------------------------------------
# THE JUDGE, AND WHY IT LIVES OUTSIDE THE IMAGE.
#
# Grand Challenge accepts an optional model tarball extracted to
# /opt/ml/model/ at runtime, separate from the container. The judge ships
# there rather than in the image for a hard reason: the image is already
# ~8.2 GiB of a 10 GiB ceiling, and any second model breaks it. The sidecar
# takes those weights out of the image budget entirely.
#
# THE CONSEQUENCE IS THAT ABSENCE IS NORMAL. A submission uploaded without
# the tarball -- or an earlier image redeployed, or a run where the platform
# does not populate the path -- has no judge, and that is a routine
# deployment state, not an error. `arbiter._arbitrate_judge` degrades to
# `challenger`... except that `config/arbiter.json` now ships `fallback`, so
# in practice a missing judge yields exactly the mode we already measured.
# Nothing about this path can make a case fail.
JUDGE_SIDECAR_DIR = Path("/opt/ml/model")

# Checked in order; the first that looks like a real checkpoint wins. The
# in-image path is a development convenience (a judge baked in for local
# testing) and is expected to be absent in a real submission.
DEFAULT_JUDGE_MODEL_DIRS = (
    JUDGE_SIDECAR_DIR / "qwen3vl-4b-judge-nf4",
    JUDGE_SIDECAR_DIR,
    REPO / "models" / "qwen3vl-4b-judge-nf4",
)

# --------------------------------------------------------------------------
# TRAIN/SERVE PROMPT PARITY -- see EvidenceVlmHandle.sample below for where
# this is actually applied, and scripts/train_vlm.py's module docstring
# ("THE EVIDENCE PACKET IS DELIBERATELY EMPTY AT TRAIN TIME" / "THE COUPLING
# THIS CREATES") for the other half of the invariant this constant exists to
# keep.
# --------------------------------------------------------------------------
# `scripts/train_vlm.py`'s `render_training_prompt` calls
# `evidence_vlm.build_sampling_prompt(question, {})` -- an EMPTY context --
# because Task 3 extracted frames only and never ran the CNN/YOLO/motion/
# variant stack over the 15,087 sampled training windows; synthesising an
# evidence packet from ground-truth labels instead was considered and
# REJECTED (it would hand the model the literal answer to the question
# being asked about that same window). `EvidenceVlmHandle.sample` below
# calls that SAME renderer but has a real evidence packet (`perception`)
# available to pass in its place. If it always did, the fine-tuned adapter
# would meet prompt text at inference it never once saw in training -- a
# silent degradation, not a crash, and exactly the failure mode this
# constant exists to prevent.
#
# THE INVARIANT: the context passed at training and the context passed at
# serving must match, and this is the switch that keeps them matched.
# `config/arbiter.json`'s `vlm_evidence_context` key is the single source of
# truth for the shipped value (loaded the same way `--arbiter-mode` already
# overrides that file's `mode` key -- one file, one place to set this, not
# two); `DEFAULT_VLM_EVIDENCE_CONTEXT` below is only the degrade-quietly
# fallback `build_vlm` uses if the config key is absent or the file cannot
# be read, mirroring `arbiter.load_config`'s own "a malformed config must
# degrade the policy, never take down the run" contract. Defaults to False
# (bare) because that is the configuration the weights we can actually
# train today were trained under -- flip this (and the config key) to True
# ONLY together with retraining scripts/train_vlm.py against a real,
# non-label-derived evidence packet. See
# tests/test_train_vlm.py's
# test_serving_and_training_prompts_match_under_the_shipped_default, which
# fails if this drifts out of sync with what that script actually trains
# on.
DEFAULT_VLM_EVIDENCE_CONTEXT = False

# ---------------------------------------------------------------------------
# THE CASE WALL-CLOCK BUDGET. Grand Challenge allows 600 s per case, and a
# MISSING RESPONSE SCORES 0 -- strictly worse than any wrong answer (a wrong
# polar answer still scores ~0.7015). So the VLM's budget cannot be a constant.
#
# `evidence_vlm.adaptive_confidence_sample` anchors its deadline at
# `_clock() + budget_seconds` -- measured from when the VLM STARTS, with no
# knowledge of what the case already spent. That is fine when the router is
# fast and dangerous when it is not, and both happen on the same code:
#
#   validation 9698714, host e2471   17-22 s per case
#   validation 9698732, host e4075   192-317 s per case   (same image, same
#                                    flags, same eleven cases -- node
#                                    contention alone)
#
# A flat 240 s VLM budget on the second of those is 317 + 240 = 557 s plus
# teardown, against 600. That is a coin-flip on a zero.
#
# The same constant is ALSO leaving most of the window unused on the first:
# 22 s of router plus 240 s of VLM is 262 s of a 600 s allowance, and the user
# asked explicitly not to be shy with it. More budget buys real quality here --
# `adaptive_confidence_sample` spends it on additional self-consistency
# samples, which is the mechanism its curtailment logic exists to govern.
#
# So the budget is computed per case as (ceiling - already spent - margin):
# it SHRINKS on slow hardware to protect against the zero, and GROWS on fast
# hardware to use the window. Both directions matter; this is not only a
# safety change.
CASE_WALL_BUDGET_SECONDS = 540.0
#: Held back from the 600 s contract for what `main`'s clock cannot see:
#: apptainer starting an 8.8 GB image, the interpreter and torch importing,
#: and the grader's own teardown after `main` returns. `started` is taken
#: inside `main`, so everything before it is invisible here.

CASE_SAFETY_MARGIN_SECONDS = 45.0
#: Held back from what remains, for work that happens AFTER the VLM returns:
#: arbitration, writing /output, and the tail of a `generate()` call that the
#: per-token StoppingCriteria can only cut at a token boundary.

VLM_MAX_BUDGET_SECONDS = 420.0
#: A ceiling on the grown budget. Not a safety limit -- the arithmetic above
#: already is one -- but a guard against a pathological case spending nine
#: minutes on self-consistency when `adaptive_confidence_sample` would have
#: curtailed on agreement long before.

VLM_MIN_USEFUL_SECONDS = 35.0
#: Below this, do not start the VLM at all. A generation cut off after a few
#: tokens is not a cheap partial answer -- it is a truncated string that the
#: arbiter may then prefer over the router's correct one. Declining is the
#: safe move, and it is logged rather than silent.

#: Set by `main` so `try_vlm_result` can see how much of the case is already
#: spent. A module global rather than a new parameter because `route()`'s
#: signature is pinned by tests that wrap and forward it positionally (see
#: that function's own docstring); threading a new argument through would
#: break them for a value that is genuinely per-process, not per-call.
_CASE_STARTED = None


class _JudgeHolder(object):
    """Carries the judge callable to `route()`, which cannot take a new
    parameter without breaking tests that wrap and forward it positionally
    (see that function's docstring). Same seam as `_CASE_STARTED`."""
    fn = None


_JUDGE = _JudgeHolder()

# The graded unit is a fixed-length clip -- see surgvu.predict's module
# docstring ("The graded unit is a 30-second clip") and surgvu.sampling.
# WINDOW_SECONDS, which is the same constant on the training side. Used only
# to stamp the detector's per-anchor timestamps in `detections_to_record`;
# not imported from surgvu.sampling because that module is a training-only
# concern (window enumeration for shard-building) and this file has never
# depended on it.
CLIP_SECONDS = 30.0


# --------------------------------------------------------------------------
# logging and timing
# --------------------------------------------------------------------------

def log(message):
    """Everything this file says goes to stderr; stdout belongs to nobody."""
    print("[surgvu] %s" % (message,), file=sys.stderr, flush=True)


def log_peak_vram():
    """Peak VRAM this case actually used, or nothing on a No-GPU draw.

    THE NUMBER THAT DECIDES HOW BIG THE MODELS CAN BE. The grader's T4 has
    16 GiB; a 512x512 frame costs ~324 image tokens and the KV cache for even
    32 frames is under 0.6 GB, so frames are cheap in MEMORY and expensive in
    COMPUTE -- but that is arithmetic, and this is measurement. Reported once
    per case, after the answer is written, so it can never delay or break the
    response.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        log("vram peak=%.2f GiB reserved=%.2f GiB of %.1f GiB (%.0f%% used)"
            % (peak, reserved, total, 100.0 * reserved / total))
    except Exception:                            # noqa: BLE001 - never fatal
        pass


@contextmanager
def timed(name, timings):
    """Record a stage's wall time whether or not it succeeded.

    A stage that raised is still worth timing: the interesting log line is the
    one from the run that failed at 9m50s.
    """
    started = time.time()
    try:
        yield
    finally:
        timings.append((name, time.time() - started))


def format_timings(timings):
    return " ".join("%s=%.2fs" % (name, seconds) for name, seconds in timings)


# --------------------------------------------------------------------------
# the two JSON strings
# --------------------------------------------------------------------------

def read_question(path):
    """The question text. `json.load`, because the file holds a JSON string.

    Read raw, the leading quote arrives at the router and "Are there forceps
    ...?" no longer opens with a polar auxiliary, which silently turns a yes/no
    question into an open one. The raw-text branch is the reverse insurance:
    if the file ever holds unquoted text, we take it rather than answering
    from nothing.
    """
    raw = Path(path).read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
    except ValueError:
        log("WARNING: %s is not valid JSON; falling back to its raw text" % (path,))
        return raw.strip().strip('"')
    if isinstance(value, str):
        return value
    log("WARNING: %s holds %s, not a string; coercing"
        % (path, type(value).__name__))
    return raw.strip().strip('"')


def safe_read_question(path, timings):
    """The question, or "" -- this step may not be allowed to raise.

    An unreadable question is not an excuse to write nothing: the open
    fallback still scores 0.35-0.48 and an absent file scores zero.
    """
    with timed("question", timings):
        try:
            return read_question(path)
        except Exception:                       # noqa: BLE001 - see docstring
            log("WARNING: could not read the question at %s" % (path,))
            traceback.print_exc(file=sys.stderr)
            return ""


def write_response(path, answer):
    """The single exit point. Writes a JSON-encoded, non-empty string.

    `json.dump` is the whole point: it is what puts the quotes on.
    """
    text = answer if isinstance(answer, str) else ""
    text = " ".join(text.split())
    if not text:
        log("WARNING: the pipeline produced an empty answer; writing the "
            "generic fallback instead. An empty string crashes the scorer.")
        text = FALLBACK_OPEN
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(text), encoding="utf-8")
    return text


def fallback_answer(question):
    """The best answer still available when perception is unavailable.

    Two branches, and which one a question takes is decided by pure logic --
    `classify_question` reads the question text and nothing else, so it is
    still trustworthy after everything downstream of it has failed.

    ROUTED, against an empty record, for the intents in
    `PERCEPTION_INDEPENDENT_INTENTS`. Their answer forms never read perception
    on a healthy run either: "What is the purpose of using forceps?" is
    answered from the tool the question named, and "What type of procedure is
    this?" from a constant. Losing the video costs those answers nothing, so
    writing a generic sentence instead would be throwing away the gold
    reference (1.0000) for a plausible generic (0.35-0.48).

    THE CALIBRATED FALLBACK for everything else. The router tolerates an empty
    record for those intents too -- it does not raise -- but the answers get
    worse: an empty record says "No" to every presence question while gold
    polar answers in this corpus skew Yes, and it invents the modal instrument
    count out of nothing. "Yes" (polar) or the generic sentence (open) is the
    measured floor; a systematic "No" is not.

    Never empty, and never raises: this is the last line of defence, and it
    now calls more code than a constant lookup did. If the classifier itself
    is what broke, the calibrated string still gets written.
    """
    try:
        if classify_question(question) in PERCEPTION_INDEPENDENT_INTENTS:
            return answer_question(question, {})
    except Exception:                           # noqa: BLE001 - see docstring
        traceback.print_exc(file=sys.stderr)
        log("WARNING: the intent-aware fallback raised; using the calibrated "
            "string")
    return finalize_answer(
        FALLBACK_POLAR if is_polar_question(question) else FALLBACK_OPEN)


# --------------------------------------------------------------------------
# perception, bound by config/perception.json
# --------------------------------------------------------------------------

def load_config(path):
    """The frozen binding. Authoritative over whatever a checkpoint says.

    Reading image sizes, class order and per-class thresholds out of the
    checkpoints at serving time would mean the container's behaviour changes
    whenever a retrain drops a new file into the models directory. The config
    is built once, deliberately, by scripts/build_perception_config.py.
    """
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    experts = config.get("experts")
    if not isinstance(experts, dict) or not {"tools", "task"} <= set(experts):
        raise ValueError("%s names no tools/task experts; it is not a "
                         "perception config" % (path,))
    return config


def checkpoint_path(entry, models_dir=None):
    """Where the weights are. `models_dir` re-roots by basename, which is how
    the same config survives being baked into an image that mounts nothing."""
    path = Path(entry["checkpoint"])
    if models_dir:
        path = Path(models_dir) / path.name
    return path


def serving_thresholds(entry):
    """The per-class cuts this container APPLIES. Not always the checkpoint's.

    Two vectors live in the config and they are not interchangeable:

    `thresholds` MIRRORS the checkpoint. `train_tools.py` tuned it on
    PER-FRAME validation probabilities, and it is here so that
    `load_bound_expert`'s drift guard has something to compare the checkpoint
    against. It exists to catch accidents, not to be served.

    `serving_thresholds` is tuned on the CLIP MEAN that `aggregate_window`
    actually forms over `decode.frames` frames -- the distribution these cuts
    are applied to. Averaging preserves a class's mean and shrinks its
    variance, so the per-frame optimum is not the clip-level optimum, and the
    shipped cuts span 0.05-0.95 where that gap is widest. Measured on all
    4,635 `splits_v2` val windows, re-tuning is worth +0.0196 macro-F1 for no
    retrain. The config records what it was measured on, over which split, on
    which date, and both sides of the before/after.

    So the divergence between the two is DELIBERATE, and this is where that is
    stated. Everything below is the check that it is still the deliberate one:

      * a vector of the wrong length is refused -- they are positional, and a
        short one would threshold the head of the taxonomy on purpose and the
        tail by accident;
      * a vector whose provenance names weights other than the ones the config
        binds is refused. A retrain moves every probability scale it was
        calibrated against, and cuts tuned on a model that is no longer here
        are not a better channel than the checkpoint's own -- they are an
        unbounded one.

    Refused means the checkpoint's mirrored cuts are served instead, loudly.
    Raising would cost the case its answer, which is worth 0.35-0.48 at best;
    a mediocre threshold is worth far more than that.
    """
    mirror = entry.get("thresholds")
    block = entry.get("serving_thresholds")
    if not block:
        return mirror

    values = list(block.get("values") or [])
    classes = list(entry.get("classes") or [])
    if len(values) != len(classes):
        log("WARNING: %s config carries %d serving thresholds for %d classes; "
            "serving the checkpoint's own cuts instead. They are positional, "
            "so a short vector cannot be partially applied."
            % (entry.get("role"), len(values), len(classes)))
        return mirror

    tuned_on = (block.get("provenance") or {}).get("checkpoint_sha256")
    bound = entry.get("sha256")
    if tuned_on and bound and tuned_on != bound:
        log("WARNING: %s serving thresholds were tuned on a different "
            "checkpoint (%s...) than the config binds (%s...); serving the "
            "checkpoint's own cuts instead. Re-run "
            "scripts/tune_serving_thresholds.py against the bound weights."
            % (entry.get("role"), tuned_on[:12], bound[:12]))
        return mirror

    log("%s serving thresholds in effect: a DELIBERATE divergence from the "
        "checkpoint's per-frame cuts, tuned on the clip mean this container "
        "forms. Provenance is in the config." % (entry.get("role"),))
    return values


def expert_meta(entry):
    """The `meta` dict the perception functions expect, taken from the config
    rather than from the checkpoint.

    `meta["thresholds"]` is what `perceive.tools_present` applies, so it is
    the SERVING vector -- see `serving_thresholds` for why that is not always
    the one the checkpoint carries.
    """
    meta = {"classes": list(entry["classes"]),
            "image_size": entry["image_size"],
            "backbone": entry.get("backbone")}
    thresholds = serving_thresholds(entry)
    if thresholds is not None:
        meta["thresholds"] = list(thresholds)
    return meta


def load_bound_expert(entry, device, models_dir=None):
    """One expert, checked against the binding it is supposed to satisfy."""
    path = checkpoint_path(entry, models_dir)
    if not path.exists():
        raise FileNotFoundError(
            "%s checkpoint %s does not exist. The config binds it, so there "
            "is nothing to serve." % (entry.get("role"), path))
    model, meta = load_expert(path, device)
    if list(meta.get("classes", [])) != list(entry["classes"]):
        raise ValueError(
            "%s checkpoint %s was trained on %r but the config binds %r. The "
            "record is keyed by class name, so serving this would report "
            "every class as another one."
            % (entry.get("role"), path, meta.get("classes"), entry["classes"]))
    if meta.get("image_size") != entry["image_size"]:
        log("WARNING: %s checkpoint was trained at image_size=%s but the "
            "config binds %s; using the config."
            % (entry.get("role"), meta.get("image_size"), entry["image_size"]))
    # DRIFT GUARD. Compared against `entry["thresholds"]`, which is the
    # config's MIRROR of the checkpoint -- never against the serving vector.
    # That is the whole reason the mirror is still carried: the deliberate
    # divergence lives in `serving_thresholds`, so anything this catches is an
    # accident (a retrain that landed under the same filename, a hand edit)
    # and is worth shouting about exactly as loudly as it was before.
    if (entry.get("thresholds") is not None
            and list(meta.get("thresholds", [])) != list(entry["thresholds"])):
        log("WARNING: %s checkpoint thresholds differ from the config's; "
            "using the config. The weights on disk are probably newer than "
            "the config -- rebuild it with scripts/build_perception_config.py."
            % (entry.get("role"),))
    return model


def expert_checkpoints(entry):
    """Every checkpoint this expert serves: the bound one, then any ensemble.

    `ensemble` is a list of ADDITIONAL checkpoint paths. Absent, this is a
    one-element list and the behaviour is exactly what shipped in v1.
    """
    return [entry["checkpoint"]] + list(entry.get("ensemble") or [])


def reduce_frames(frame_probs, entry, role):
    """(models, frames, classes) -> (classes,), per the config's aggregation.

    Two reductions, in this order and not the other:

      1. MEAN ACROSS MODELS, per frame. Averaging probabilities rather than
         voting on thresholded predictions is the point of an ensemble -- two
         models that each fall just under their own cut can still average
         above a re-tuned one, and a vote throws exactly that away.
      2. The configured AGGREGATION across frames. `mean` is v1's behaviour
         and remains the default, so a config without the key serves what it
         always did.

    Measured on splits_v2 val, clip-level, thresholds tuned on one case fold
    and scored on the other:

        EfficientNet alone   mean 0.6747   top3 0.6889
        ResNet-50 alone      mean 0.6623   q90  0.6834
        both, ensembled      mean 0.7054   top5 0.7171

    The best aggregator is NOT a constant of the problem -- it differs per
    model, and EndoViT prefers mean where EfficientNet prefers top3. It is
    read from the config per expert for that reason rather than hardcoded.
    """
    stacked = np.asarray(frame_probs, dtype=np.float32)
    if stacked.ndim != 3:
        raise ValueError("%s expected (models, frames, classes); got %r"
                         % (role, stacked.shape))
    name = entry.get("aggregation", "mean")
    if name not in AGGREGATORS:
        log("WARNING: %s config asks for aggregation %r, which does not "
            "exist; serving 'mean'. Known: %s"
            % (role, name, sorted(AGGREGATORS)))
        name = "mean"
    # AGGREGATORS reduce axis 1 of (windows, frames, classes); one clip is a
    # batch of one.
    return AGGREGATORS[name](stacked.mean(axis=0)[None, ...])[0]


def infer(frames, config, device, timings, models_dir=None, motion=None,
         motion_v2=None):
    """Both experts over one clip's frames -> the router's perception record.

    This is `perceive.perceive_clip` unrolled by one level, and only so that
    the two forward passes can be timed separately: against a 10-minute budget
    "inference took 200 s" is not an actionable number and "tools 190 s,
    task 10 s" is.
    """
    tools, task = config["experts"]["tools"], config["experts"]["task"]
    record = {}

    for role, entry in (("tools", tools), ("task", task)):
        paths = expert_checkpoints(entry)
        per_model = []
        for index, checkpoint in enumerate(paths):
            bound = dict(entry, checkpoint=checkpoint)
            with timed("load_%s%s" % (role, index or ""), timings):
                model = load_bound_expert(bound, device, models_dir)
            with timed("%s_infer%s" % (role, index or ""), timings):
                per_model.append(predict_window_frames(
                    model, frames, device, entry["image_size"],
                    activation=entry["activation"]))
            # Freed before the next member loads: two EfficientNets plus a
            # ResNet is still small, but the deployment ceiling is a 16 GiB
            # T4 shared with nothing, and an ensemble is the first thing here
            # that scales with a number in a config file.
            del model
        record[role] = reduce_frames(per_model, entry, role)
        if len(paths) > 1:
            log("%s: ensembled %d checkpoints, aggregation=%s"
                % (role, len(paths), entry.get("aggregation", "mean")))

    return clip_record(record["tools"], expert_meta(tools),
                       record["task"], expert_meta(task), len(frames),
                       motion=motion, motion_v2=motion_v2)


def resolve_devices(requested):
    """The devices to try, in order. A failed accelerator retries on CPU."""
    if requested in (None, "", "auto"):
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return [requested] if requested == "cpu" else [requested, "cpu"]


def infer_with_retry(frames, config, devices, timings, models_dir=None,
                     motion=None, motion_v2=None):
    for index, device in enumerate(devices):
        try:
            log("device=%s" % (device,))
            return infer(frames, config, device, timings, models_dir,
                         motion=motion, motion_v2=motion_v2)
        except Exception:                       # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            if index + 1 >= len(devices):
                raise
            log("WARNING: perception failed on %s; retrying on cpu" % (device,))


# --------------------------------------------------------------------------
# EVIDENCE: the detector, the variant head (Task 11), and CNN-vs-detector
# agreement (Task 7, wired here once surgvu.agreement existed)
# --------------------------------------------------------------------------
# All three are wired so that any failure logs a traceback plus a
# human-readable WARNING and leaves the corresponding block absent -- the
# same best-effort idiom --motion-v2 uses above (ruling R18), not a second
# one. `clip_record`'s optional blocks are purely additive
# (src/surgvu/perceive.py), so with --yolo and --variant-head both off
# nothing below this comment ever runs and `perception` is byte-identical to
# today.
#
# `agree` (surgvu.agreement.agreement_record) has NO flag of its own -- it
# fires automatically whenever a yolo record exists, inside its own
# try/except so an agreement failure can never discard a yolo block that
# already succeeded. Agreement between the CNN heads and a detector that did
# not run (--yolo not passed, or the detector itself failed) is a category
# error, not a signal, so it is never attempted in that case.

def _needle_driver_boxes(yolo_record):
    """{anchor_idx: box} for the MAX-CONFIDENCE needle-driver detection per
    anchor, from a `detections_to_record` block.

    Mirrors `scripts/variant_sample_report.py:_needle_boxes` exactly -- the
    reference implementation the variant head was fed by when its cutoff was
    fitted and measured (val_accuracy 0.8681 @ coverage 0.9969 on the eleven
    graded clips; see the module docstring). A frame can carry more than one
    needle-driver detection after NMS, and `by_class["needle driver"]`'s
    insertion order (anchor order, then NMS's own order within an anchor) is
    not a confidence order by contract -- so the max is taken explicitly
    rather than trusting "last one wins" on a list whose order is an
    implementation detail of a different module.

    A frame with no needle-driver detection at all is simply absent from the
    returned dict, which `VariantHead.predict` treats identically to an
    explicit `None`: falls back to the whole frame. That fallback is what
    makes `--variant-head` legal without `--yolo`, and why an empty or
    missing `yolo_record` here (this function is only called when one
    exists) is not a special case -- an empty `{}` produces an empty dict
    the same way a `None` `yolo_record` does at the call site below.
    """
    boxes, best_conf = {}, {}
    by_class = (yolo_record or {}).get("by_class") or {}
    for entry in by_class.get("needle driver", []):
        idx = entry["anchor_idx"]
        conf = entry["conf"]
        if idx not in best_conf or conf > best_conf[idx]:
            best_conf[idx] = conf
            boxes[idx] = entry["box"]
    return boxes


def add_evidence(perception, frames, args, timings, config):
    """Best-effort `yolo`, `variant` and `agree` blocks, mutating `perception`
    in place. NEVER raises -- every internal failure is caught, logged (a
    traceback plus a human-readable WARNING), and leaves the corresponding
    key simply absent, exactly as it is today when the flag is off.

    Runs AFTER `perception` already holds the CNN heads' answer, so a
    failure here can never cost the case its appearance-model-backed
    answer -- only the evidence this task exists to add.
    """
    yolo_record = None
    if args.yolo:
        try:
            from surgvu.detect import Detector, detections_to_record
            with timed("yolo", timings):
                detector = Detector(args.yolo_weights, args.yolo_repo,
                                    device=resolve_devices(args.device)[0])
                found = detector.detect(frames)
                stamps = [i * (CLIP_SECONDS / max(1, len(frames)))
                         for i in range(len(frames))]
                yolo_record = detections_to_record(found, stamps)
            perception["yolo"] = yolo_record
            log("yolo max_conf=%s" % (yolo_record["max_conf"],))
        except Exception:                            # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            yolo_record = None
            log("WARNING: the detector failed; continuing without it")

        # Its OWN try/except, deliberately not folded into the yolo block
        # above (same reasoning Task 11's review applied to yolo/variant):
        # an agreement failure must never discard a yolo record that already
        # succeeded. Only attempted when a yolo record exists at all --
        # agreement between the CNN heads and a detector that never ran is
        # not a signal, it is a category error.
        if yolo_record is not None:
            try:
                from surgvu.agreement import agreement_record
                # Reuse the SAME serving thresholds `infer()` already
                # derived for `clip_record` (via `expert_meta`), not a
                # second, potentially divergent, derivation of them.
                tool_meta = expert_meta(config["experts"]["tools"])
                tool_thresholds = dict(zip(tool_meta["classes"],
                                           tool_meta["thresholds"]))
                with timed("agree", timings):
                    perception["agree"] = agreement_record(
                        perception["tools"], tool_thresholds, yolo_record)
                log("agree tool_agreement=%.3f top_disagreement=%s"
                    % (perception["agree"]["tool_agreement"],
                       perception["agree"]["top_disagreement"]))
            except Exception:                        # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                log("WARNING: agreement computation failed; continuing "
                    "without it")

    if args.variant_head:
        try:
            from surgvu.variant import VariantHead
            head_config = json.loads(
                Path(args.variant_config).read_text(encoding="utf-8"))
            # The crop is an improvement, not a precondition (see the flag's
            # help text): an absent or empty `yolo_record` -- --yolo not
            # passed, the detector failed above, or it simply never boxed a
            # needle driver -- yields an empty `boxes`, and `boxes or None`
            # hands VariantHead.predict exactly the `None` it treats as
            # "use the whole frame for every one of these frames".
            boxes = _needle_driver_boxes(yolo_record) if yolo_record else {}
            with timed("variant", timings):
                head = VariantHead(args.variant_weights,
                                   head_config["cutoff"],
                                   device=resolve_devices(args.device)[0])
                perception["variant"] = head.predict(frames, boxes or None)
            log("variant family=%s p_large=%.3f decided=%s"
                % (perception["variant"]["family"],
                   perception["variant"]["p_large"],
                   perception["variant"]["decided"]))
        except Exception:                            # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            log("WARNING: the variant head failed; continuing without it")


# --------------------------------------------------------------------------
# SEAM: the Evidence VLM and the arbiter (Plan 2)
# --------------------------------------------------------------------------
# OFF unless `--vlm` is passed, so shipping it is a decision and not an
# accident: the flag has to be added to the container's command line, which is
# a reviewed edit, rather than a config value that can be flipped in passing.
#
# UNLIKE THE OLD (pre-Plan-2) SEAM THIS REPLACES, the VLM is not restricted to
# questions the router cannot classify. `surgvu.arbiter`'s shipped policy
# (`challenger`, config/arbiter.json) drafts a VLM answer for EVERY question
# and may override the router's whenever the VLM's own self-consistency
# confidence clears a floor -- see arbiter.py's module docstring for why that
# is the measured, deliberate choice (fallback's ceiling on the graded sample
# is exactly zero). This function's job is only to produce the VLM's draft
# (a `ConfidenceResult`, or None) and hand it to `arbiter.arbitrate`, which is
# the one place that decides what ships.
#
# THE NO-GPU PATH IS A CORRECTNESS REQUIREMENT, NOT A NICETY. The shipped
# weights are 4-bit NF4 (bitsandbytes), which needs CUDA. On a No-GPU
# deployment draw the VLM is structurally unable to run at all -- `available()`
# is checked FIRST, before any `transformers` import, so a CPU-only instance
# never even attempts the weights and the router's answer stands untouched.
# On a CUDA instance, any other failure (a missing checkpoint directory
# because the weights are not staged yet, an OOM, a malformed generation) is
# still caught at the same call site, R18-style: a traceback to stderr, a
# WARNING, and the router's answer stands. Returning None or raising costs
# nothing -- a VLM that OOMs on a T4 must not be able to take the case with
# it.
class EvidenceVlmHandle(object):
    """A lazily-constructed handle onto `surgvu.evidence_vlm`'s adaptive
    sampler, pinned at THIS container's baked-in weights directory.

    Construction (`build_vlm`) never imports `transformers`, never touches
    the filesystem, and never creates a CUDA context -- it only stores a
    path, two ints, and the logger. Everything expensive happens inside
    `sample()`, called once per case from `try_vlm_result`. There is no
    intent gate upstream of that call any more (see the module note above:
    unlike the pre-Plan-2 seam, every question reaches the VLM when `--vlm`
    is on), so the saving this handle's laziness buys is entirely about
    `--vlm` being off: the No-GPU/no-`transformers` case, and every case in
    a run where the flag was simply never passed.
    """

    def __init__(self, model_dir, n_frames, max_samples, log, arbiter_mode=None,
                 evidence_context=DEFAULT_VLM_EVIDENCE_CONTEXT):
        self.model_dir = str(model_dir)
        self.n_frames = int(n_frames)
        self.max_samples = int(max_samples)
        self.arbiter_mode = arbiter_mode
        self.evidence_context = bool(evidence_context)
        self._log = log

    def available(self):
        """False on a No-GPU draw -- checked BEFORE any `transformers`
        import. `torch` is already imported at this file's module scope for
        the CNN path, so this check costs nothing extra."""
        return bool(torch.cuda.is_available())

    def sample(self, video, question, perception, budget_seconds=None):
        """One `ConfidenceResult` from `surgvu.evidence_vlm`, or a raised
        exception. This method does not itself absorb failures -- see
        `try_vlm_result`, the R18 wrapper at the call site, for that.

        TRAIN/SERVE PARITY SEAM. `self.evidence_context` (see
        `DEFAULT_VLM_EVIDENCE_CONTEXT`'s docstring above and
        `scripts/train_vlm.py`'s module docstring for the other half of this
        coupling) picks what `evidence_vlm.build_sampling_prompt` -- called
        inside `adaptive_confidence_sample` -> `call_vlm`, never
        reimplemented here -- actually renders: the real `perception` packet
        when True, or `{}` (matching `train_vlm.render_training_prompt`'s
        own `build_sampling_prompt(question, {})`) when False, the shipped
        default. Passing `perception` unconditionally here, the way the
        pre-fix code did, is exactly the bug this seam exists to prevent:
        the model would be served evidence text it never saw in training.
        """
        context = perception if self.evidence_context else {}
        # PICK THE FRAME PLAN FROM THE TIME LEFT, not from a constant. See
        # surgvu.frame_plan: prefill cannot be interrupted by the deadline
        # StoppingCriteria (that stops GENERATION, token by token, and prefill
        # happens before the first token), so how much visual evidence to ask
        # for has to be decided BEFORE the call, from the budget that remains.
        # On a fast card this takes the richest tier and genuinely fills the
        # 600s window; on a slow one it steps down rather than gambling.
        plan = None
        try:
            from surgvu import frame_plan as _fp
            # THE SECOND BUDGET. Time alone chose the plan through v6, and
            # a plan that fits 600 s can still be unable to prefill on a
            # 14.56 GiB T4 -- which is what happened on all eleven graded
            # cases. Ask the device how much memory it has and let
            # select_plan honour both. torch is already imported at module
            # scope here; a card that cannot be described leaves vram_gib
            # None, which is exactly the old time-only behaviour.
            vram_gib = None
            try:
                if torch.cuda.is_available():
                    vram_gib = (torch.cuda.get_device_properties(0)
                                .total_memory / (1024 ** 3))
            except Exception:                    # noqa: BLE001 - never fatal
                pass
            plan = _fp.select_plan(budget_seconds if budget_seconds is not None
                                   else _fp.FIXED_OVERHEAD_SECONDS * 30,
                                   vram_gib=vram_gib)
            log("VLM frame %s" % _fp.describe(plan))
        except Exception:                        # noqa: BLE001 - never fatal
            traceback.print_exc(file=sys.stderr)
            log("WARNING: frame planning failed; using the default frame count")
        # budget_seconds=None means "use evidence_vlm's own default" rather
        # than "no budget": passing None through would override the default
        # with nothing and remove the deadline entirely, which is the one
        # outcome the 600 s contract cannot survive.
        extra = ({} if budget_seconds is None
                 else {"budget_seconds": float(budget_seconds)})
        # A plan overrides the configured frame count; None means even the
        # cheapest tier did not fit, and try_vlm_result has already declined
        # by the time we get here in that case.
        frames_per_call = self.n_frames
        if plan is not None:
            frames_per_call = int(plan["n_frames"])
            extra["frame_plan"] = plan
        with _vlm_model_dir_pinned(self.model_dir):
            return self._sample_stepping_down_on_oom(
                video, question, context, plan, frames_per_call, extra)

    def _sample_stepping_down_on_oom(self, video, question, context,
                                     plan, frames_per_call, extra):
        """Sample, and on a CUDA OOM retry with strictly cheaper evidence.

        WHY THIS IS NOT BELT AND BRACES. `MATH_KERNEL_TOKEN_CEILING` is
        measured, not modelled, but it was measured on one model with one
        judge configuration. If a live run exceeds it for a reason the probe
        never saw, `try_vlm_result` absorbs the OOM and the VLM contributes
        NOTHING for the entire case -- indistinguishable in the log from a
        VLM that simply agreed. That silence is the bug being fixed, and on
        Grand Challenge the next chance to correct it is a fresh container
        upload.

        Here an OOM costs one wasted prefill and the answer still arrives.
        `empty_cache` between attempts matters: the failed prefill's
        allocations are freed but stay RESERVED, so without it the retry
        contends with the corpse of the attempt that just died.
        """
        # BOTH IMPORTS ARE LOCAL, and the missing one cost a full build +
        # validation cycle. `adaptive_confidence_sample` is imported inside
        # `sample` (module-scope torch imports are avoided throughout this
        # file), so splitting the retry into its own method took it out of
        # scope: every case raised NameError, `try_vlm_result` absorbed it,
        # and eleven cases reported "absorbed" -- the same silent shape as
        # the OOM this method exists to survive.
        from surgvu import frame_plan as _fp
        from surgvu.evidence_vlm import adaptive_confidence_sample

        attempt = 0
        while True:
            try:
                return adaptive_confidence_sample(
                    {"path": str(video)}, question, context,
                    frames_per_call=frames_per_call,
                    max_samples=self.max_samples, **extra)
            except Exception as exc:            # noqa: BLE001 - see docstring
                if "OutOfMemory" not in type(exc).__name__ or attempt >= 2:
                    raise
                cheaper = _fp.next_cheaper_plan(plan)
                if cheaper is None:
                    raise
                attempt += 1
                try:
                    torch.cuda.empty_cache()
                except Exception:               # noqa: BLE001 - never fatal
                    pass
                log("VLM: OOM at %d tokens; retrying at %d (%s)"
                    % (plan.get("tokens", -1) if plan else -1,
                       cheaper["tokens"], cheaper["label"]))
                plan = cheaper
                frames_per_call = int(cheaper["n_frames"])
                extra["frame_plan"] = cheaper


@contextmanager
def _vlm_model_dir_pinned(model_dir):
    """Point `surgvu.evidence_vlm.call_vlm` at `model_dir` for the duration
    of one sampling run, then restore it.

    `adaptive_confidence_sample` calls `call_vlm(frames, question, context,
    temperature=...)` with no `model_dir` of its own -- it has none -- so
    `call_vlm`'s bare default (`DEFAULT_MODEL_DIR`, the HF hub id
    "Qwen/Qwen2.5-VL-7B-Instruct") is what every sample would otherwise
    request, which is exactly wrong for a no-internet container: this file's
    own `DEFAULT_VLM_MODEL_DIR` is the in-container path Task 4 wires
    instead. This is not a private hack: `evidence_vlm.py`'s own docstring
    for `adaptive_confidence_sample` documents `call_vlm` as looked up from
    THIS module's globals "at call time (not bound at import time), so a
    test can `monkeypatch.setattr` ... and this function will use the
    replacement" -- the exact seam `tests/test_evidence_vlm.py` already
    exercises. Serving code using the same documented seam, scoped to one
    call with a restore in `finally`, is not a second pattern and not a
    permanent mutation a later test (or a later question, if this file ever
    samples twice per process) could be surprised by.
    """
    from surgvu import evidence_vlm

    original = evidence_vlm.call_vlm
    evidence_vlm.call_vlm = functools.partial(original, model_dir=model_dir)
    try:
        yield
    finally:
        evidence_vlm.call_vlm = original


def elapsed_this_case():
    """Seconds since `main` started, or 0.0 if `main` never ran.

    0.0 rather than a raise: a test (or a future caller) that exercises
    `try_vlm_result` directly without going through `main` should get the
    full budget, not a crash. The only cost of being wrong in that direction
    is a longer budget in a context that has no 600 s contract to violate.
    """
    if _CASE_STARTED is None:
        return 0.0
    return max(0.0, time.time() - _CASE_STARTED)


def remaining_vlm_budget():
    """How many seconds the VLM may spend on THIS case.

    (ceiling - already spent - margin), clamped to `VLM_MAX_BUDGET_SECONDS`
    and floored at 0. See the constants' own block for why each number is
    what it is, and why this is computed per case rather than fixed at
    `evidence_vlm.DEFAULT_BUDGET_SECONDS`.

    Breaks if: this returns a NEGATIVE number to a caller that treats it as
    a duration (hence the max(0.0, ...)), or if the clamp is applied before
    the subtraction, which would hand a slow case the full ceiling.
    """
    remaining = (CASE_WALL_BUDGET_SECONDS - elapsed_this_case()
                 - CASE_SAFETY_MARGIN_SECONDS)
    return max(0.0, min(remaining, VLM_MAX_BUDGET_SECONDS))


def try_vlm_result(video, question, perception, vlm, timings):
    """A `ConfidenceResult` from the Evidence VLM, or None -- every failure
    mode is absorbed here, R18-style: a traceback to stderr, a human-readable
    WARNING, and the router's answer stands. Mirrors `add_evidence`'s
    try/except idiom exactly; this is not a second pattern.

    `vlm.available()` is checked BEFORE anything else, and before this
    function (or anything it calls) imports `transformers`: see
    `EvidenceVlmHandle.available` and the module note above this class.
    """
    if vlm is None:
        return None
    if not vlm.available():
        log("VLM: no CUDA device available; the 4-bit NF4 weights cannot "
            "run without one. Skipping -- the router's answer stands.")
        return None
    budget = remaining_vlm_budget()
    if budget < VLM_MIN_USEFUL_SECONDS:
        log("VLM: only %.0fs of the case budget left (below the %.0fs floor); "
            "skipping rather than starting a generation that would be cut off "
            "mid-answer. The router's answer stands."
            % (budget, VLM_MIN_USEFUL_SECONDS))
        return None
    log("VLM: %.0fs budget for this case (%.0fs already spent of a %.0fs "
        "allowance, %.0fs held back)"
        % (budget, elapsed_this_case(), CASE_WALL_BUDGET_SECONDS,
           CASE_SAFETY_MARGIN_SECONDS))
    try:
        with timed("vlm", timings):
            result = vlm.sample(video, question, perception,
                                budget_seconds=budget)
    except Exception:                            # noqa: BLE001 - see the SEAM block
        traceback.print_exc(file=sys.stderr)
        log("WARNING: the VLM seam raised; keeping the router's answer")
        return None
    # Logging is OUTSIDE the try block on purpose: it must never be the thing
    # that raises. `result` here is whatever `vlm.sample()` handed back --
    # `arbiter._is_usable_vlm_result` is the actual validator, and it accepts
    # a non-`ConfidenceResult` without raising, so this branches on the same
    # check rather than assuming the shape and reaching for `.n_calls_used`
    # on something that might not have one.
    if isinstance(result, ConfidenceResult):
        log("VLM: %d call(s) agreed=%s confidence=%.2f answer=%r"
            % (result.n_calls_used, result.agreed, result.confidence,
               result.answer))
    else:
        log("VLM: sample() returned %r, not a ConfidenceResult; the "
            "arbiter will treat it as unusable" % (result,))
    return result


def route(question, perception, video, timings, vlm=None):
    """The router's answer, or the arbiter's blend of it with the Evidence
    VLM's draft.

    `arbiter.arbitrate` runs `router.answer_question` internally and returns
    it UNCHANGED whenever `vlm_result` is absent, of the wrong type, or
    unusable (see its own docstring's fall-through property) -- so with
    `vlm=None` (the shipped default whenever `--vlm` is not passed) this
    function's output is byte-identical to calling `router.answer_question`
    directly, exactly as it was before this task.

    `--arbiter-mode` reaches this call through `vlm.arbiter_mode` rather than
    a parameter of its own, so `route()`'s signature stays exactly what it
    was before Plan 2 -- callers that spy on it by wrapping and forwarding
    positionally (see tests/test_inference.py, tests/test_inference_evidence.py)
    keep working unmodified.
    """
    with timed("route", timings):
        vlm_result = try_vlm_result(video, question, perception, vlm, timings)
        config = load_arbiter_config()
        mode_override = getattr(vlm, "arbiter_mode", None)
        if mode_override:
            config = dict(config, mode=mode_override)
        # Injected, not imported: surgvu.arbiter loads no models and imports
        # no torch, and that separation is what keeps it testable without a
        # GPU. A None here is the normal no-sidecar state.
        judge_fn = getattr(_JUDGE, "fn", None)
        if judge_fn is not None:
            config = dict(config, judge_fn=judge_fn)
        answer = arbitrate(question, perception, vlm_result, config)
    return answer


def find_model_dir(candidates):
    """The first candidate that holds a LOADABLE checkpoint, or None.

    A directory is not a checkpoint: `config.json` plus at least one weight
    shard is the bar, the same one containers/build_submission.sh applies at
    build time. /opt/ml/model/ exists whenever the platform mounts it, empty
    or not, so an existence check alone would report a model that cannot load
    -- and the failure would then surface inside try_vlm_result's
    exception-swallowing wrapper, i.e. silently, on the graded run.
    """
    for candidate in candidates:
        try:
            path = Path(candidate)
            if not (path / "config.json").is_file():
                continue
            if any(path.glob("*.safetensors")) or any(path.glob("*.bin")):
                return path
        except OSError:
            continue
    return None


def find_judge_model_dir(candidates=DEFAULT_JUDGE_MODEL_DIRS):
    return find_model_dir(candidates)


def resolve_vlm_model_dir(explicit=None):
    """Where the answering VLM's weights are, sidecar FIRST.

    THE SIDECAR IS NOW THE PRIMARY LOCATION, not a fallback. int8 weights are
    ~8 GiB and the judge another ~2.95; together with the image they are
    14.18 GiB against a 10 GiB ceiling, so neither can be baked in. Moving
    both out drops the image to ~3.5 GiB -- 35% of the ceiling -- and puts the
    weights in the model tarball, which has no documented size limit.

    The in-image path is kept LAST and still works: it is what v5 shipped
    (NF4, baked in), so an image built the old way keeps running unchanged and
    a submission uploaded without the tarball degrades to the weights it
    carries rather than to nothing.
    """
    if explicit:
        return Path(explicit)
    found = find_model_dir((
        JUDGE_SIDECAR_DIR / "qwen25vl-7b-int8",
        JUDGE_SIDECAR_DIR / "qwen25vl-7b-nf4",
        DEFAULT_VLM_MODEL_DIR,
    ))
    return found or DEFAULT_VLM_MODEL_DIR


def build_judge(args):
    """A `judge_fn(question, perception, candidates) -> str`, or None.

    NEVER RAISES, and never loads at construction time -- same contract as
    `build_vlm`. The model is loaded on FIRST CALL, which for most cases is
    never: `judge.should_consult` skips the whole stage when the router and
    the VLM already agree, and a judge that loaded eagerly would pay several
    seconds of weights for every one of those.

    Returns None when there is no judge to run, which
    `arbiter._arbitrate_judge` treats as "decide the way you would have
    anyway". That is the shipped state for any submission uploaded without
    the model tarball.
    """
    if not getattr(args, "judge", False):
        return None
    model_dir = find_judge_model_dir()
    if model_dir is None:
        log("judge: no checkpoint at %s or its siblings; the arbiter will "
            "decide without one" % (JUDGE_SIDECAR_DIR,))
        return None
    log("judge enabled: %s" % (model_dir,))

    def judge_fn(question, perception, candidates):
        from surgvu import evidence_vlm, judge as judge_mod
        # FREE THE ANSWERING VLM FIRST. Both resident is 11.45 GiB of weights
        # on a 16 GiB T4 before prefill activations; released, the peak is the
        # larger of the two rather than the sum. This costs nothing here:
        # Grand Challenge runs one case per container invocation, the VLM has
        # already produced its candidate by the time a judge is consulted, and
        # the process exits after this case.
        evidence_vlm.release_models(keep=str(model_dir))
        evidence_lines = evidence_vlm.build_sampling_prompt("", perception)
        prompt = judge_mod.build_judge_prompt(question, evidence_lines, candidates)
        frames = evidence_vlm.sample_frames({"path": str(args._judge_video)},
                                            n=JUDGE_FRAMES)
        return evidence_vlm.call_vlm(frames, prompt, {}, temperature=0.0,
                                     model_dir=str(model_dir))

    return judge_fn


#: Frames the judge sees. Fewer than the answering pass: it is comparing two
#: candidate strings against the evidence, not deriving an answer from
#: scratch, and every frame is prefill it cannot interrupt.
JUDGE_FRAMES = 8


def build_vlm(args):
    """The Evidence VLM handle the run will use, or None. Never raises.

    Construction is free by design -- no import of `transformers`, no
    weights touched, no CUDA context -- so a case the router alone handles
    pays nothing for `--vlm` being enabled. A failure here still returns None
    rather than propagating: an unusable VLM is the state this container
    ships in, not a reason to abort the whole case.
    """
    if not args.vlm:
        return None
    try:
        from surgvu.evidence_vlm import DEFAULT_FRAMES_PER_CALL, DEFAULT_MAX_SAMPLES

        model_dir = resolve_vlm_model_dir(args.vlm_model)
        n_frames = args.vlm_frames or DEFAULT_FRAMES_PER_CALL
        max_samples = args.vlm_max_samples or DEFAULT_MAX_SAMPLES
        # `args.vlm_evidence_context` is None unless --vlm-evidence-context/
        # --no-vlm-evidence-context was actually passed -- same "None means
        # defer to config/arbiter.json" contract as `args.arbiter_mode` a few
        # lines below. config/arbiter.json is the single source of truth for
        # the shipped value; DEFAULT_VLM_EVIDENCE_CONTEXT is only the
        # degrade-quietly fallback for a missing/malformed config (see that
        # constant's own docstring for why False/bare).
        evidence_context = args.vlm_evidence_context
        if evidence_context is None:
            evidence_context = bool(load_arbiter_config().get(
                "vlm_evidence_context", DEFAULT_VLM_EVIDENCE_CONTEXT))
        log("VLM enabled: %s (frames_per_call=%d max_samples=%d "
            "arbiter_mode=%s evidence_context=%s)"
            % (model_dir, n_frames, max_samples, args.arbiter_mode or "<config>",
               evidence_context))
        return EvidenceVlmHandle(model_dir, n_frames, max_samples, log,
                                 arbiter_mode=args.arbiter_mode,
                                 evidence_context=evidence_context)
    except Exception:                           # noqa: BLE001 - see docstring
        traceback.print_exc(file=sys.stderr)
        log("WARNING: the VLM could not be constructed; running without it")
        return None


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Overridable so the whole entrypoint is runnable and testable without
    # /input and /output existing, which they do not outside the container.
    parser.add_argument("--input-dir", default="/input")
    parser.add_argument("--output-dir", default="/output")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="the frozen perception binding")
    parser.add_argument("--models-dir",
                        help="re-root the config's checkpoints here by "
                             "basename, for weights baked into the image")
    parser.add_argument("--device", default="auto",
                        help="auto (cuda if present), cuda, or cpu")
    parser.add_argument("--frames", type=int,
                        help="override the config's frame count")
    parser.add_argument("--size", type=int,
                        help="override the config's decode size")
    # OFF BY DEFAULT -- but NOT for the reason first written here, and the
    # correction is the point. The original comment said motion "costs roughly
    # three times the decode", which is true of the decode STEP and wrong
    # about the run: condor/motion_ab.sh measured 6.6 s mean per case without
    # it and 7.0 s with, because decode is about a second of a seven-second
    # case and checkpoint loading dominates. Six percent, against a 600 s
    # budget. Cost is not the argument.
    #
    # The argument is that it buys nothing TODAY. scripts/calibrate_motion.py
    # declined to open the router's gate -- 9.8% of cutting answers would flip
    # Yes->No against a corpus whose gold polar answers skew Yes, with no
    # cutting label to validate a single flip -- so the block would be
    # recorded and ignored. An unused feature in a shipped path is a thing
    # that rots, and the flag makes turning it on a reviewed edit rather than
    # a default nobody revisited.
    parser.add_argument("--judge", action="store_true",
                        help="consult a SECOND model (Qwen3-VL-4B, shipped in "
                             "Grand Challenge's separate model tarball at "
                             "/opt/ml/model/) when the router and the VLM "
                             "disagree. A no-op when no such checkpoint is "
                             "present, which is the state of any submission "
                             "uploaded without the tarball.")
    parser.add_argument("--motion", action="store_true",
                        help="decode each sampled frame's neighbours and "
                             "record motion evidence. Additive: the "
                             "appearance model sees byte-identical input and "
                             "the router ignores the block until a threshold "
                             "is calibrated.")
    # OFF by default, same reasoning as --motion. INDEPENDENT of --motion:
    # the v1 block above feeds a calibrated router gate that is already
    # shipping (scripts/calibrate_motion.py), and removing or altering it
    # here would change answers for a reason unrelated to this flag. The v2
    # block adds multi-scale probes plus optical-flow coherence
    # (surgvu.motion.motion_record_v2) alongside, under its own key, read by
    # nothing yet -- it is additive perception evidence for later tasks, not
    # a replacement for the v1 gate.
    parser.add_argument("--motion-v2", action="store_true",
                        help="multi-scale motion probes plus optical-flow "
                             "coherence. Independent of --motion: the v1 "
                             "block feeds a calibrated router gate that is "
                             "already shipping, and removing it here would "
                             "change answers for a reason unrelated to this "
                             "flag.")
    # OFF by default, and unlike the old (pre-Plan-2) --vlm this replaces, NOT
    # restricted to questions the router cannot classify -- see the SEAM
    # block's module note above `EvidenceVlmHandle` for why: the arbiter, not
    # an intent gate here, is what decides whether the VLM's draft ships.
    # `--vlm-model`/`--vlm-frames`/`--vlm-max-samples` are None here and
    # resolved inside `build_vlm` from `surgvu.evidence_vlm`'s own defaults,
    # so nothing in this file imports that module (or `transformers`) until
    # the flag asks for it.
    parser.add_argument("--vlm", action="store_true",
                        help="draft an answer with the Evidence VLM and let "
                             "the arbiter (config/arbiter.json, or "
                             "--arbiter-mode) decide whether it overrides "
                             "the router's. Off by default; lazily "
                             "constructed, so a case the router alone "
                             "settles pays nothing for this being enabled, "
                             "and it self-disables with no CUDA device -- "
                             "the shipped weights are 4-bit NF4, CUDA-only.")
    parser.add_argument("--vlm-model",
                        help="directory holding the baked-in NF4 weights "
                             "(default: %s)" % (DEFAULT_VLM_MODEL_DIR,))
    parser.add_argument("--vlm-frames", type=int,
                        help="frames shown per VLM sampling call (default: "
                             "surgvu.evidence_vlm.DEFAULT_FRAMES_PER_CALL)")
    parser.add_argument("--vlm-max-samples", type=int,
                        help="cap on adaptive-confidence-sampling calls per "
                             "question (default: "
                             "surgvu.evidence_vlm.DEFAULT_MAX_SAMPLES). Each "
                             "call is a full generation, so this is the "
                             "lever against a runaway wall-clock on a slow "
                             "node -- see evidence_vlm.adaptive_confidence_"
                             "sample's own escalate-to-agreement loop.")
    # None, not a hardcoded mode string: "defaults to whatever config/
    # arbiter.json says" is the requirement this default satisfies. Reaches
    # `arbitrate()` through `EvidenceVlmHandle.arbiter_mode`, read by
    # `route()` -- see that function's docstring for why it is not a
    # parameter of `route()` itself.
    parser.add_argument("--arbiter-mode",
                        choices=(MODE_FALLBACK, MODE_CHALLENGER, MODE_PRIMARY),
                        default=None,
                        help="override config/arbiter.json's mode for this "
                             "run. Only matters when --vlm is also passed -- "
                             "with no VLM draft to arbitrate, every mode "
                             "falls through to the router's answer.")
    # None (not True/False) is the sentinel for "config/arbiter.json decides"
    # -- the same "None means defer to the config file" contract
    # --arbiter-mode already uses above. See DEFAULT_VLM_EVIDENCE_CONTEXT's
    # docstring (and scripts/train_vlm.py's module docstring) for why the
    # shipped default is bare/False: this fine-tune was trained against an
    # EMPTY evidence context, so serving with the real one would be a
    # silent train/serve mismatch, not a crash. Only matters when --vlm is
    # also passed.
    parser.add_argument("--vlm-evidence-context", dest="vlm_evidence_context",
                        action="store_true", default=None,
                        help="render the full evidence packet (CNN "
                             "probabilities, YOLO timestamps, motion "
                             "language, the variant call) into the VLM's "
                             "prompt. DO NOT pass this unless "
                             "scripts/train_vlm.py has been retrained "
                             "against that same evidence packet -- the "
                             "shipped adapter was trained bare. Defaults to "
                             "config/arbiter.json's vlm_evidence_context "
                             "(false).")
    parser.add_argument("--no-vlm-evidence-context", dest="vlm_evidence_context",
                        action="store_false",
                        help="render the VLM's prompt bare -- the shipped "
                             "default, matching what scripts/train_vlm.py "
                             "trains against. Only useful to force this "
                             "explicitly when config/arbiter.json's default "
                             "might otherwise be True.")
    # OFF by default, and INDEPENDENT of every other flag above -- see the
    # "EVIDENCE" section below for the full contract. --yolo runs a second
    # opinion (a 14-class detector) and records its detections; nothing
    # reads them until a later flag does. --variant-head resolves the one
    # Large-vs-Mega distinction the CNN heads cannot make; it does NOT
    # require --yolo (the crop is an improvement, not a precondition -- see
    # its own help text).
    parser.add_argument("--yolo", action="store_true",
                        help="run the 14-class detector as a second opinion. "
                             "Additive: its detections and its disagreement "
                             "with the CNN heads are recorded, and nothing "
                             "reads them unless a later flag does.")
    parser.add_argument("--yolo-weights",
                        default="/opt/algorithm/models/yolo_best.pt")
    parser.add_argument("--yolo-repo", default="/opt/algorithm/yolov5")
    parser.add_argument("--variant-head", action="store_true",
                        help="resolve Large vs Mega needle driver. Uses the "
                             "detector's box when --yolo is on and the whole "
                             "frame otherwise.")
    parser.add_argument("--variant-weights",
                        default="/opt/algorithm/models/variant_head.pt")
    parser.add_argument("--variant-config",
                        default=str(DEFAULT_VARIANT_CONFIG),
                        help="carries the FITTED cutoff and the validation "
                             "accuracy it achieved. Read at serving time so "
                             "the abstention point cannot drift from the "
                             "number it was measured at. Derived from REPO, "
                             "not a bare relative string -- R32: a relative "
                             "default silently disables the variant gate in "
                             "the container (see DEFAULT_VARIANT_CONFIG).")
    return parser.parse_args(argv)


def main(argv=None):
    global _CASE_STARTED
    args = parse_args(argv)
    timings = []
    started = time.time()
    _CASE_STARTED = started
    log("torch=%s cuda_available=%s input=%s output=%s"
        % (torch.__version__, torch.cuda.is_available(),
           args.input_dir, args.output_dir))
    # NAME THE CARD AND ITS VRAM. Every timing this project has was taken on
    # an L40 (CHTC) or a developer's own GPU; the grader runs a T4, which
    # nothing here has ever executed on. Without this line a slow or
    # out-of-memory run on unfamiliar hardware is diagnosed by guesswork --
    # and a 171 s "model load" figure that turned out to be ceph I/O, not the
    # model, already cost this project one wrong latency budget.
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            log("gpu=%s vram=%.1f GiB capability=%d.%d"
                % (props.name, props.total_memory / (1024 ** 3),
                   props.major, props.minor))
        except Exception:                        # noqa: BLE001 - never fatal
            log("gpu: present but could not be described")

    video = Path(args.input_dir) / VIDEO_NAME
    response = Path(args.output_dir) / RESPONSE_NAME
    question = safe_read_question(Path(args.input_dir) / QUESTION_NAME, timings)
    log("question: %r" % (question,))
    # Built OUTSIDE the try: this may not be a way for the VLM to trigger the
    # whole-pipeline fallback. It cannot raise, and it loads nothing.
    vlm = build_vlm(args)
    # The judge needs the clip to look at, and `arbitrate`'s signature is
    # pinned by tests that wrap and forward it positionally -- so the path
    # rides on args rather than becoming a new parameter, the same reasoning
    # as `_CASE_STARTED`.
    args._judge_video = video
    _JUDGE.fn = build_judge(args)

    try:
        config = load_config(args.config)
        frames_wanted = args.frames or config["decode"]["frames"]
        size = args.size or config["decode"]["size"]
        if not video.exists():
            raise FileNotFoundError("no video at %s" % (video,))
        motion = None
        motion_v2 = None
        if args.motion_v2:
            from surgvu.motion import motion_record_v2
            from surgvu.perceive import decode_clip_multiscale
            # The DECODE stays OUTSIDE any best-effort guard (controller
            # ruling R18): these are the frames the appearance model itself
            # needs, and if this fails there is nothing left to answer from
            # -- falling through to the whole-pipeline fallback is correct,
            # exactly as it is for the plain decode_clip() branch below.
            with timed("decode", timings):
                frames, probes = decode_clip_multiscale(
                    video, n_frames=frames_wanted, size=size)
            # The COMPUTATION, by contrast, is best-effort (R18). motion_v2
            # is additive evidence read by nothing yet -- it must never be
            # able to cost the appearance model's answer (the CNNs already
            # ran, or are about to, on `frames`) because an optical-flow
            # statistic raised. Same idiom Task 11 will use for the
            # yolo/variant/agree blocks: log the traceback, log a
            # human-readable WARNING, continue with the record simply
            # absent -- one failure-handling pattern in this file, not two.
            try:
                with timed("motion_v2", timings):
                    motion_v2 = motion_record_v2(frames, probes)
                log("motion_v2 anchors=%d flow_coherence=%s micro_mid=%s"
                    % (motion_v2["anchors"],
                       motion_v2["summary"]["flow_coherence"]["mean"],
                       motion_v2["summary"]["micro_mid"]["mean"]))
            except Exception:                   # noqa: BLE001 - best-effort evidence
                traceback.print_exc(file=sys.stderr)
                motion_v2 = None
                log("WARNING: motion_v2 computation failed; continuing "
                    "without it. motion_v2 is additive evidence read by "
                    "nothing yet, so the appearance model's answer is "
                    "unaffected.")
            if args.motion:
                # Both blocks requested: the v1 gate needs the uniform burst
                # layout, which the multiscale decode does not produce. Pay
                # the second decode rather than approximate one from the
                # other -- the two are computed from independently decoded
                # frame stacks, each byte-identical to what its own flag
                # produces alone.
                #
                # This whole decode+compute pair is best-effort too (R18):
                # it is reached only because --motion-v2 was also passed,
                # and its failure must not discard the appearance answer or
                # the motion_v2 evidence this run may already have.
                try:
                    from surgvu.motion import motion_record_from_bursts
                    with timed("decode_bursts", timings):
                        _, bursts = decode_clip_bursts(
                            video, n_frames=frames_wanted, size=size)
                    with timed("motion", timings):
                        motion = motion_record_from_bursts(bursts, frames)
                    log("motion micro=%s macro=%s"
                        % (motion["micro"]["mean"], motion["macro"]["mean"]))
                except Exception:               # noqa: BLE001 - best-effort evidence
                    traceback.print_exc(file=sys.stderr)
                    motion = None
                    log("WARNING: motion (v1, alongside motion_v2) "
                        "computation failed; continuing without it. The "
                        "router's calibrated gate simply sees no motion "
                        "block, the same as when --motion is not passed.")
        elif args.motion:
            from surgvu.motion import motion_record_from_bursts
            with timed("decode", timings):
                frames, bursts = decode_clip_bursts(
                    video, n_frames=frames_wanted, size=size)
            with timed("motion", timings):
                motion = motion_record_from_bursts(bursts, frames)
            log("decoded %d frames + %d/%d measurable bursts at %dx%d"
                % (len(frames), motion["bursts_measured"], motion["bursts"],
                   size, size))
            log("motion micro=%s macro=%s"
                % (motion["micro"]["mean"], motion["macro"]["mean"]))
        else:
            with timed("decode", timings):
                frames = decode_clip(video, n_frames=frames_wanted, size=size)
            log("decoded %d frames at %dx%d" % (len(frames), size, size))

        perception = infer_with_retry(frames, config,
                                      resolve_devices(args.device), timings,
                                      args.models_dir, motion=motion,
                                      motion_v2=motion_v2)
        log("tools_present=%s task_top=%s"
            % (perception["tools_present"] or "[]", perception["task_top"]))
        # EVERY BLOCK ADD_EVIDENCE MAY ADD IS OPTIONAL AND EVERY FAILURE
        # INSIDE IT IS SWALLOWED -- these are unmeasured components on a
        # pipeline whose one hard guarantee is that it always writes an
        # answer, and evidence that cannot be gathered is evidence the
        # record simply does not carry. See the EVIDENCE section above.
        add_evidence(perception, frames, args, timings, config)
        # `video`, not `frames`: the Evidence VLM re-decodes its own (smaller)
        # frame set from the path via evidence_vlm.sample_frames, rather than
        # reusing the CNN path's decode -- see EvidenceVlmHandle.sample.
        answer = route(question, perception, video, timings, vlm)
    except Exception:                           # noqa: BLE001 - the whole point
        traceback.print_exc(file=sys.stderr)
        answer = fallback_answer(question)
        log("FALLBACK: the pipeline failed; answering %r rather than writing "
            "nothing. A missing response scores zero; a wrong polar answer "
            "still scores 0.7015." % (answer,))

    written = write_response(response, answer)
    timings.append(("total", time.time() - started))
    log("timings %s" % format_timings(timings))
    log_peak_vram()
    log("wrote %s -> %s" % (response, json.dumps(written)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
