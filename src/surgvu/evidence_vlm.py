"""Adaptive confidence sampling for the Evidence VLM: a black-box uncertainty
signal built by resampling, not by reading logits.

THE SHAPE THIS KEEPS
--------------------
A teammate built this first, in `vlm_pass1_adaptive.py` on
`/staging/groups/bhaskar_opscribe/surgvu_vlm_pipeline.tar.gz`, and it had
never shipped. The idea is worth keeping exactly as she had it: sample the
model repeatedly over a fixed piece of evidence, accept the answer when the
samples agree, escalate to more samples when they do not. That needs no
logit access, which matters because nothing in this project's serving stack
assumes one particular inference backend. `ConfidenceResult`, `route()`, and
the sample-until-agree-or-cap loop in `adaptive_confidence_sample()` restate
that idea over this project's own perception and model code rather than
inventing a different design.

WHAT WAS SEVERED, AND WHY
--------------------------
Her file called `opscribe_pipeline.providers.vlm.get_vlm_provider()` for the
model and `opscribe_pipeline.video.{VideoDecoder,FrameStore,SamplingStrategy}`
for frames. SurgVU26 Cat 2 is deliberately independent of OpScribe -- its own
containers, its own venv, and a submission container that is offline and
self-contained -- and must never reach for OpScribe's `.sif` or `pypkgs`
(see the project's own standing note on this). `sample_frames()` below reads
frames through `surgvu.perceive.decode_clip` instead: the same decode, and
the same `preprocess.prepare_frame` UI-band blur, the CNN path already uses.
`call_vlm()` loads Qwen2.5-VL-7B-Instruct directly through `transformers`
instead of a provider factory. Both imports happen lazily, inside the
function bodies, so this module -- including the routing and agreement
logic that decides ACCEPT vs ESCALATE -- stays importable and testable
where torch is not installed, which is this login node and every test in
tests/test_evidence_vlm.py.

`OVERLAY_PROMPT` -- a prompt in her `debug_utils.py` instructing the model to
read the numbered tool list off the bottom UI band -- is DELETED, not ported
under any name, and not renamed into something that reads the same band by a
different route. The challenge rules prohibit using information visible in
the UI to make predictions, and `preprocess.prepare_frame` blurs that band on
every single frame precisely so there is no way around it. A differently
named prompt with the same effect would be the same violation with new
cover.

Her `is_correct()` (`pred == a_clean or pred in a_clean` -- a substring
match) is not ported either: it counts a prediction as "correct" whenever it
merely appears inside an accepted answer string, which inflates every
accuracy number computed with it. This project scores with
`surgvu.scoring.Scorer` (BERTScore-F1), which is what the challenge actually
uses, and that is the only scorer any claim in this module rests on. Nor is
her `parse_question_type()` (a two-branch RECORDS/LOOK_HARDER split) ported
-- this project's router already ships 11 intents and does not need a
twelfth, cruder classifier sitting in front of it.

TEMPERATURE 0.4 IS A HYPOTHESIS, NOT A MEASURED CONSTANT
----------------------------------------------------------
Her docstring reports a sweep over {0.1, 0.3, 0.4, 0.6, 0.8} on 4 real
sample cases, with 0.4 scoring 2/4 "correct" against 1/4 at every other
value. Two things should stop that number from being trusted as-is: n=4 is
not enough to distinguish a real effect from noise, and "correct" there was
`is_correct()`'s substring match, which this project does not use precisely
because it is lenient in a way BERTScore-F1 is not.
`DEFAULT_SAMPLING_TEMPERATURE` below keeps her value anyway -- 0.4 is a
reasonable prior and re-deriving one from scratch under the real metric is
not this task's job -- but nothing in this file should be read as having
validated it, and a future change to it should re-measure under
`surgvu.scoring.Scorer`, not under a sweep like hers.

AGREEMENT IS NOT CALIBRATION
------------------------------
Her one qualitative finding worth keeping regardless of the sweep's
statistical weight: at temperature 0.1 the model was confidently wrong WITH
FULL SELF-AGREEMENT on three of her four cases -- case122, case127 and
case130. Every resample gave the identical answer, and every one of those
answers was wrong. Low temperature makes a model nearly deterministic, so
repeated samples agreeing tells you the model is CONSISTENT with itself,
not that it is RIGHT. A confidence signal built purely from self-consistency
can be saturated at 1.0 by an argmax the model would have produced
regardless of what it was shown. This is the reason this project treats
CNN-vs-YOLO disagreement (`surgvu.agreement`, once it lands) as a BETTER
confidence channel than this one: two independently trained, differently
biased models agreeing is much harder to get "for free" than one model
agreeing with its own low-temperature self. A future reader extending this
module should not read a high `confidence` here as a high probability of
being correct -- only as a low probability of the model having contradicted
itself.
"""

import time
from collections import Counter
from dataclasses import dataclass, field

#: Bump when the shape of `ConfidenceResult.to_dict()` or `route()`'s output
#: changes, so anything that persists a record can tell which shape it wrote.
EVIDENCE_VLM_VERSION = 1

#: `route()`'s two possible decisions. Exported as constants rather than left
#: as string literals so a caller (the Task 3 arbiter) compares against a
#: name, not a typo-able string.
ACCEPT = "ACCEPT"
ESCALATE = "ESCALATE"

#: Qwen2.5-VL-7B-Instruct everywhere in this project -- Evidence VLM,
#: arbitration, and Plan 3's fine-tune base -- per the plan's global
#: constraint. This is the model id, not yet an on-disk path: Task 4 wires
#: the actual baked-in location inside the offline submission image, the way
#: `surgvu.vlm.DEFAULT_MODEL_DIR` already does for the router's fallback VLM.
DEFAULT_MODEL_DIR = "Qwen/Qwen2.5-VL-7B-Instruct"

#: The base the v6 SurgVU LoRA is fine-tuned ON TOP OF -- deliberately a
#: SEPARATE constant from `DEFAULT_MODEL_DIR` above, which stays
#: Qwen2.5-VL-7B-Instruct.
#:
#: WHY NOT JUST REPOINT DEFAULT_MODEL_DIR. That one is also `call_vlm`'s
#: default `model_dir`, i.e. the SERVING fallback. Serving does not normally
#: reach it (scripts/inference.py's `resolve_vlm_model_dir` picks the sidecar
#: or the in-image weights first), but "does not normally" is not "cannot",
#: and a constant that means two things is how a training-side edit silently
#: becomes a serving-side one. Two names, two meanings.
#:
#: WHY THIS MODEL. `nvidia/Qwen2.5-VL-7B-Surg-CholecT50` is Qwen2.5-VL-7B
#: fine-tuned by NVIDIA on CholecT50 for surgical triplet recognition --
#: F1 0.81 instrument, 0.64 verb, 0.60 target. Its config is identical to
#: Qwen2.5-VL-7B-Instruct's on every field that matters
#: (Qwen2_5_VLForConditionalGeneration, qwen2_5_vl, vocab 152064, hidden 3584,
#: 28 layers), which is what makes it a drop-in: the merge/quantise path, the
#: LoRA target modules and the 152064-row embedding check all still hold.
#:
#: AN ABSOLUTE SNAPSHOT PATH, NOT THE HUB ID. condor/train_vlm.sh points
#: HF_HOME at /staging/n/nkalthoff/surgvu26/hf_cache, and this model lives in a
#: DIFFERENT tree (/staging/n/nkalthoff/hf_cache). Under the hub id, an offline
#: execute node would look in the wrong cache, miss, and -- with
#: HF_HUB_OFFLINE set -- fail; without it, silently try to download 16GB.
#:
#: LICENSE: NSCLv1, "for research and development only". Raised with and
#: accepted by the project owner on 2026-08-27 for this research competition.
DEFAULT_FINETUNE_BASE = (
    "/staging/n/nkalthoff/hf_cache/hub/models--nvidia--Qwen2.5-VL-7B-Surg-"
    "CholecT50/snapshots/c1a01db98c72f4fca5ec671405325c4c841dcafd")

#: How many frames back one call to `sample_frames`.
#:
#: RETUNED 5 -> 16 on 2026-08-26, which is the retune the previous comment
#: here ("Matches the ported default; Task 2/4 may retune it") flagged and
#: nobody did. 5 was inherited from the ported sampler and never chosen for
#: this pipeline.
#:
#: THE ASYMMETRY THAT MADE IT WRONG: the CNN path already decodes 16 frames
#: per clip (see a real run's "decoded 16 frames ... at 512x512"), so the VLM
#: was answering questions like "what organ is being manipulated" from FIVE
#: stills of a clip its own pipeline had already looked at sixteen times.
#: Matching 16 removes the asymmetry rather than inventing a new number.
#:
#: AFFORDABLE, measured rather than assumed. At 512x512 a frame costs ~324
#: image tokens after Qwen2.5-VL's 2x2 patch merge, so 16 frames is ~5,184
#: tokens and a KV cache of ~0.30 GB. Against the grader's 16 GiB T4 that
#: takes the NF4 7B's total from ~6.7 GB to ~7.0 GB -- 44% of the card. Frames
#: are cheap in MEMORY and expensive in COMPUTE, and compute is the budget
#: this pipeline has most of: a real case measured 18.34 s of a 600 s limit.
#:
#: Breaks if: this is raised without re-measuring on sm_75. Every timing this
#: project has is from an L40 or a developer GPU; the grader's T4 is roughly
#: 3-5x slower on memory bandwidth, and prefill cost scales with frame count.
DEFAULT_FRAMES_PER_CALL = 16

#: How many times `adaptive_confidence_sample` will call the model before
#: giving up on reaching `agreement_threshold` and falling back to a
#: majority vote.
DEFAULT_MAX_SAMPLES = 3

#: Full agreement required to stop early. A `Counter.most_common(1)` result
#: of 2/2 identical answers is confidence 1.0 at this threshold; nothing
#: below unanimity is treated as "agreed" -- see `adaptive_confidence_sample`
#: for why n=1 also cannot reach `agreed=True` regardless of this value.
DEFAULT_AGREEMENT_THRESHOLD = 1.0

#: See "TEMPERATURE 0.4 IS A HYPOTHESIS" above. Kept as her value, not
#: re-derived, and not to be read as validated.
DEFAULT_SAMPLING_TEMPERATURE = 0.4

#: `route()`'s default confidence floor for ACCEPT. Matches the ported value;
#: Task 3's `config/arbiter.json` is the place a fitted number would replace
#: this, the same way `variant.py`'s cutoff is fitted rather than guessed.
DEFAULT_CONFIDENCE_THRESHOLD = 0.66

#: Generation budget per call. The gold answers this competes with are short
#: (see `surgvu.vlm`'s measured 6-10 word references); a long generation is
#: both slower and worse under BERTScore-F1.
DEFAULT_MAX_NEW_TOKENS = 64

#: Wall-clock seconds `adaptive_confidence_sample` may spend in total --
#: loading the model (once, cached across a run) plus every sample it takes
#: -- before it must stop asking for more and return whatever it already
#: has. THIS PORT'S ORIGINAL SIN: the ported `vlm_pass1_adaptive.py` had no
#: such budget at all, and the retired `surgvu.vlm.QwenVlmFallback` module
#: it sat alongside DID (its own `DEFAULT_BUDGET_SECONDS = 240.0`, via a
#: `StoppingCriteria`) -- that mechanism, not a number invented fresh, is
#: what this constant and `_deadline_stopping_criteria` below restore.
#:
#: The arithmetic behind 240.0, against the numbers this project has
#: actually measured (`plan2-task4-report.md`'s "Budget" section):
#:   - Grand Challenge allows 600s per case, one case per container run.
#:   - The shipped image (CNN path + evidence-prompt construction,
#:     CPU-only, no VLM) measures 94-196s per case -- so in the WORST
#:     observed case, 196s of the 600s is already spent before this
#:     module's sampler is even called.
#:   - A single VLM generation was estimated at ~20-25s/case on a healthy
#:     T4 (processor+load ~8s one-time, ~4-5s per generation,
#:     `DEFAULT_MAX_SAMPLES=3` generations at most) -- i.e. the HEALTHY
#:     case this budget must not get in the way of is roughly an order of
#:     magnitude below the number chosen here.
#:   - 196s (worst-case non-VLM) + 240s (this budget) = 436s, still 164s
#:     (27%) under the 600s ceiling for whatever else the container does
#:     (writing `/output/visual-context-response.json`, logging, process
#:     teardown) -- comfortable margin, not a number shaved to the wire.
#:   - 240s also comfortably covers a pathological node: the retired
#:     module's docstring records "a cold read of 6 GB of weights over
#:     shared storage took 219s once" for a similarly-sized checkpoint: one
#:     such load alone would consume most of this budget, and the
#:     per-sample `_deadline_stopping_criteria` check (see `call_vlm`)
#:     means even a single stalled generation cannot run past what is left
#:     of it.
#: Reusing the retired module's exact value, rather than deriving a new one,
#: is deliberate: it is the same design ("generous by design... exists to
#: bound a pathological node, not to shave a fast one") applied to a
#: pipeline that, per the numbers above, has LESS spare room left in the
#: 600s than the retired module's own 6.2-16.0s-of-600s CNN-only case did --
#: so 240.0 here is, if anything, the more conservative choice of the two
#: contexts it has been used in.
DEFAULT_BUDGET_SECONDS = 240.0

#: Camera-vs-tool split point on `motion_v2`'s `flow_moving_fraction`. NOT a
#: fitted decision boundary -- it sits midway between the two values Task 1
#: measured on clips built to be unambiguous: a global camera pan reads
#: moving_fraction=1.0000, local tool motion reads moving_fraction=0.0822.
#: `flow_coherence` is deliberately NOT used for this call: on those same two
#: clips it separates by only 1.33x (1.0000 vs 0.7513) against
#: moving_fraction's 12x, and on real surgical video it was measured at
#: CHANCE (AUC 0.5075). moving_fraction is carried as the WHOLE signal for
#: this word choice, per `surgvu.motion`'s own docstring.
FLOW_CAMERA_DOMINANT_THRESHOLD = 0.5

#: Magnitude split for "active" vs "still" language, in the mean-absolute-
#: inter-frame-difference units both `surgvu.motion`'s v1 record and v2's
#: `macro_prev`/`macro_next` slots share (0-255, over a 64x64 grayscale
#: reduction -- see `motion._to_work`/`motion._mad`). This is the SAME VALUE
#: as `router.STATIC_ACTIVITY_THRESHOLD` (1.283, the graded clips' measured
#: 10th percentile), copied rather than imported so this module does not pull
#: in the router's regex-heavy import graph for one float; equality with the
#: router's constant is pinned by
#: `test_motion_active_threshold_matches_router_calibration` in
#: tests/test_evidence_prompt.py so the two cannot silently drift. Borrowing
#: it changes what the number is FOR: router.py fitted it to gate a rule that
#: fires or does not, not to choose a word, and v2's micro_short/mid/long
#: slots measure a single frame pair at a shorter offset than the burst-mean
#: v1 measures -- so applying it to those slots is an assumption, not a
#: re-measurement. It is the best available anchor in these units, not a
#: validated language boundary.
MOTION_ACTIVE_THRESHOLD = 1.283

#: How many distinct detected classes the evidence prompt names, and how many
#: timestamps per class, before collapsing the rest into "+N more". Budget
#: awareness, not taste: `surgvu.vlm`'s measured cost for one VLM call on the
#: shipped image is 118-196s CPU-only against a 600s per-case limit -- about
#: 3x expansion room, not 40x -- and a busy clip can produce dozens of
#: detections across 12+ classes. Left uncapped, one busy case could triple
#: the token count (and therefore the generation time) that a quiet case
#: pays; these caps keep the prompt's growth bounded regardless of how many
#: detections a clip produces.
MAX_YOLO_CLASSES_IN_PROMPT = 6
MAX_TIMESTAMPS_PER_CLASS_IN_PROMPT = 4


def normalize_answer(answer):
    """Lightweight normalisation so 'Yes.' and 'yes' count as agreeing.

    Case and trailing punctuation are not information about the model's
    uncertainty; treating them as a disagreement would understate agreement
    for reasons that have nothing to do with the question asked.
    """
    return str(answer).strip().lower().rstrip(".")


@dataclass
class ConfidenceResult:
    """The outcome of one adaptive sampling run.

    `agreed` is True only when the stopping condition inside
    `adaptive_confidence_sample` was reached WHILE there was still room to
    disagree (at least two samples taken). It is not simply
    `confidence >= 1.0`: see that function's docstring for the one-sample
    case, where confidence can compute to 1.0 without `agreed` ever being
    set, because a single answer cannot self-verify.

    `curtailed` and `elapsed_seconds` are additive fields for the wall-clock
    deadline (see `DEFAULT_BUDGET_SECONDS`): both default to values that
    reproduce the pre-deadline behaviour exactly, so a caller built before
    this feature existed (`surgvu.arbiter`, `scripts/inference.py`) reads
    the same result it always did unless it goes looking for these two
    fields specifically.

    `curtailed=True` means the BUDGET, not `max_samples`, is why sampling
    stopped -- `agreed` and `curtailed` answer different questions and must
    never be collapsed into one: `agreed` says whether the samples that
    were taken agreed with each other; `curtailed` says whether the run was
    cut short before it could take as many samples as it might have liked.
    A single sample that survives only because the budget ran out is
    `agreed=False, curtailed=True` -- never presented as if it had agreed
    with anything, because it never had a second sample to agree with (see
    `adaptive_confidence_sample`'s one-sample reasoning, which applies
    unchanged here).
    """
    answer: str
    confidence: float
    n_calls_used: int
    all_answers: list = field(default_factory=list)
    agreed: bool = False
    curtailed: bool = False
    elapsed_seconds: float = 0.0

    def to_dict(self):
        """A strict-JSON-safe, versioned record of this result.

        Every field is coerced to a plain JSON type here rather than trusted
        to already be one -- `confidence` in particular may arrive as a
        numpy scalar from an upstream computation, and `json.dumps` refuses
        those.
        """
        return {
            "version": EVIDENCE_VLM_VERSION,
            "answer": self.answer,
            "confidence": float(self.confidence),
            "n_calls_used": int(self.n_calls_used),
            "all_answers": [str(a) for a in self.all_answers],
            "agreed": bool(self.agreed),
            "curtailed": bool(self.curtailed),
            "elapsed_seconds": float(self.elapsed_seconds),
        }


def route(result, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """High enough confidence -> ACCEPT the sampled answer; otherwise ESCALATE.

    Mirrors the ported design: this is a routing decision, not a scoring
    function, and it does not know whether `result.answer` is actually
    correct -- only whether the samples that produced it agreed often enough
    to trust. See the module docstring's "AGREEMENT IS NOT CALIBRATION"
    section before treating a high threshold as a correctness guarantee.
    """
    if not isinstance(result, ConfidenceResult):
        raise TypeError(
            "route() expects a ConfidenceResult, got %r. Routing on a plain "
            "dict or tuple would silently read the wrong fields."
            % (type(result),))
    if result.confidence >= float(confidence_threshold):
        return {
            "version": EVIDENCE_VLM_VERSION,
            "decision": ACCEPT,
            "answer": result.answer,
            "calls_used": result.n_calls_used,
        }
    return {
        "version": EVIDENCE_VLM_VERSION,
        "decision": ESCALATE,
        "partial_answer": result.answer,
        "calls_used": result.n_calls_used,
    }


def _format_seconds(value):
    return "%.1fs" % (float(value),)


def _render_tools_block(context):
    """`tools_present` (already thresholded by `perceive.tools_present`) with
    each name's own probability from `tools`, or None when nothing cleared
    threshold. Only the classes that already survived their own tuned cutoff
    are named -- this does not re-threshold anything, it renders a decision
    `surgvu.perceive` already made."""
    tools_present = context.get("tools_present")
    if not tools_present:
        return None
    tools = context.get("tools")
    if isinstance(tools, dict):
        parts = ["%s (%.2f)" % (name, float(tools.get(name, 0.0)))
                 for name in tools_present]
    else:
        parts = [str(name) for name in tools_present]
    return ("Instrument classifier, classes above their own threshold: %s."
            % (", ".join(parts),))


def _render_task_block(context):
    """The single top activity class, with its own probability when the
    full distribution is also available. None when `task_top` is absent."""
    task_top = context.get("task_top")
    if not task_top:
        return None
    task = context.get("task")
    if isinstance(task, dict) and task_top in task:
        return ("Activity classifier reports: %s (%.2f)."
                % (task_top, float(task[task_top])))
    return "Activity classifier reports: %s." % (task_top,)


def _render_yolo_block(context):
    """Detections WITH their timestamps -- the statement pooled confidences
    could never make ("needle driver at 3.7s and 9.4s, absent between").
    Distinguishes three states: the `yolo` block absent entirely (returns
    None, no section at all); present but empty (the detector ran and found
    nothing, rendered as an explicit sentence -- that is information, not a
    null); present and populated (timestamps, capped by
    `MAX_YOLO_CLASSES_IN_PROMPT`/`MAX_TIMESTAMPS_PER_CLASS_IN_PROMPT` so a
    busy clip cannot blow the token budget)."""
    yolo = context.get("yolo")
    if not isinstance(yolo, dict):
        return None
    by_class = yolo.get("by_class") or {}
    if not by_class:
        return "Detector: no instruments detected in the sampled frames."
    max_conf = yolo.get("max_conf") or {}
    ordered = sorted(by_class.items(),
                      key=lambda kv: max_conf.get(kv[0], 0.0), reverse=True)
    fragments = []
    for name, entries in ordered[:MAX_YOLO_CLASSES_IN_PROMPT]:
        times = sorted({round(float(e["t_seconds"]), 1) for e in entries})
        shown = times[:MAX_TIMESTAMPS_PER_CLASS_IN_PROMPT]
        time_text = ", ".join(_format_seconds(t) for t in shown)
        if len(times) > len(shown):
            time_text += " (+%d more)" % (len(times) - len(shown),)
        fragments.append("%s at %s" % (name, time_text))
    if len(by_class) > MAX_YOLO_CLASSES_IN_PROMPT:
        fragments.append(
            "+%d more class(es) detected" % (len(by_class) - MAX_YOLO_CLASSES_IN_PROMPT,))
    return "Detector timestamps: %s." % ("; ".join(fragments),)


def _motion_magnitude_word(value):
    if value is None:
        return None
    return "active" if float(value) >= MOTION_ACTIVE_THRESHOLD else "still"


def _motion_dominance_word(value):
    if value is None:
        return None
    return ("camera-dominant" if float(value) >= FLOW_CAMERA_DOMINANT_THRESHOLD
            else "tool-dominant")


def _render_motion_block(context):
    """The motion vector as calibrated language, never as raw floats -- a
    model reading `micro_short=5.0449` learns nothing from it. `motion_v2`
    is preferred when present (it alone carries `flow_moving_fraction`, the
    measured camera-vs-tool discriminator); `motion` (v1) is the fallback,
    which can only speak to magnitude, not to camera-vs-tool, because it has
    no flow. Renders an explicit "no measurable motion" sentence rather than
    omitting the section when the block ran but nothing usable was measured
    (`bursts_measured == 0`, or every `motion_v2` summary slot has
    `measured == 0`) -- that is still a fact the block produced, distinct
    from the block never having run at all, which returns None and omits
    the section entirely."""
    v2 = context.get("motion_v2")
    if isinstance(v2, dict):
        summary = v2.get("summary") or {}

        def _measured_mean(key):
            slot = summary.get(key) or {}
            if not slot.get("measured"):
                return None
            return slot.get("mean")

        magnitude_source = _measured_mean("macro_prev")
        if magnitude_source is None:
            magnitude_source = _measured_mean("macro_next")
        if magnitude_source is None:
            magnitude_source = _measured_mean("micro_short")
        magnitude_word = _motion_magnitude_word(magnitude_source)
        dominance_word = _motion_dominance_word(
            _measured_mean("flow_moving_fraction"))
        words = [word for word in (magnitude_word, dominance_word) if word]
        if not words:
            return "Motion: no measurable motion in the sampled frames."
        return "Motion: %s." % (", ".join(words),)

    v1 = context.get("motion")
    if isinstance(v1, dict):
        if not v1.get("bursts_measured"):
            return "Motion: no measurable motion in the sampled frames."
        macro_mean = (v1.get("macro") or {}).get("mean")
        micro_mean = (v1.get("micro") or {}).get("mean")
        source = macro_mean if macro_mean is not None else micro_mean
        magnitude_word = _motion_magnitude_word(source)
        if magnitude_word is None:
            return "Motion: no measurable motion in the sampled frames."
        return ("Motion: %s (camera vs. tool not distinguishable without "
                "flow)." % (magnitude_word,))

    return None


def _render_variant_block(context):
    """The Large-vs-Mega needle driver call, including `decided`. An
    abstention (`decided=False`, "the head could not tell") is rendered as
    different information from the block being absent (the head did not
    run at all, which returns None here and omits the section)."""
    variant = context.get("variant")
    if not isinstance(variant, dict):
        return None
    confidence = max(float(variant.get("p_large", 0.0)),
                      float(variant.get("p_mega", 0.0)))
    if variant.get("decided"):
        return ("Variant classifier: %s (confidence %.2f)."
                % (variant.get("family"), confidence))
    return ("Variant classifier could not decide between Large and Mega "
            "(confidence %.2f, below its %.2f cutoff)."
            % (confidence, float(variant.get("cutoff", 0.0))))


#: Jaccard tool-presence agreement (`agree["tool_agreement"]`, from
#: `surgvu.agreement.agreement_record`) at or above which the CNN heads and
#: the detector are rendered as CORROBORATING each other rather than
#: DIVERGING. This is a natural majority-overlap split of a 0-1 Jaccard
#: ratio, not a value tuned against graded outcomes -- unlike
#: `MOTION_ACTIVE_THRESHOLD`, nothing downstream gates a decision on this
#: number today, so there was nothing to calibrate it against. Revisit if a
#: router gate is ever built on top of this signal.
AGREE_HIGH_THRESHOLD = 0.5


def _render_agree_block(context):
    """CNN-vs-detector tool-presence agreement (`surgvu.agreement`), as
    calibrated language -- "high"/"low" and which side is the lone caller --
    never the raw Jaccard float a model reading `tool_agreement=0.83` has no
    way to weigh. Same reasoning as `_render_motion_block`.

    WHY THIS BLOCK IS WORTH PROMPT TOKENS. Self-consistency -- sampling one
    model repeatedly and trusting it when the samples agree -- is this
    module's own confidence proxy, and it has a measured failure mode: the
    groupmate's temperature sweep (see this module's docstring, "AGREEMENT
    IS NOT CALIBRATION") found the model confidently wrong WITH FULL
    SELF-AGREEMENT on case122, case127 and case130 at temperature 0.1.
    Agreement WITHIN one model measures determinism, not correctness. Two
    models trained on different objectives from different label formats
    fail differently, so their divergence is a signal self-consistency
    cannot produce at any temperature: HIGH agreement here means the
    instrument classifier and the detector -- independently trained, never
    shown each other's output -- corroborate each other's read of the
    frame, which is stronger evidence than one model being sure of itself.
    LOW agreement means they disagree on which instruments are present,
    which is a reason to look harder at the frame, not a verdict either
    way.

    THE HONEST LIMIT, MEASURED. This is not a correctness oracle: on
    case124 the CNN heads and the detector AGREE on "Bipolar Forceps" (the
    detector reports 0.895 confidence there, and 0.000 for Cadiere
    Forceps) and the gold answer is Cadiere Forceps. Two independently
    trained models agreeing on the wrong answer is still agreement, and
    `agreement_record` reports it honestly. A caller may treat LOW
    agreement as a reason to look harder; it must never treat HIGH
    agreement as proof of correctness.
    """
    agree = context.get("agree")
    if not isinstance(agree, dict):
        return None
    agreement = agree.get("tool_agreement")
    if agreement is None:
        return None
    agreement = float(agreement)
    if agreement >= AGREE_HIGH_THRESHOLD:
        both = agree.get("both_present") or []
        if both:
            return ("Cross-model agreement: high -- the instrument "
                    "classifier and the detector both independently call "
                    "%s." % (", ".join(both),))
        return ("Cross-model agreement: high -- the instrument classifier "
                "and the detector agree no instrument is present.")
    top = agree.get("top_disagreement")
    if top and len(top) == 2:
        name, source = top
        who = "classifier" if source == "cnn_only" else "detector"
        return ("Cross-model agreement: low -- the instrument classifier "
                "and the detector diverge; only the %s calls %s."
                % (who, name))
    return ("Cross-model agreement: low -- the instrument classifier and "
            "the detector diverge on which tools are present.")


#: The renderers `build_sampling_prompt` applies to `context`, one evidence
#: block each, in the order they appear in the prompt. Each returns a
#: rendered line or None (omit). Kept as a tuple, not inlined, so the SEAM
#: below is a one-line addition rather than a restructuring.
_EVIDENCE_RENDERERS = (
    _render_tools_block,
    _render_task_block,
    _render_yolo_block,
    _render_motion_block,
    _render_variant_block,
    # Task 7's `surgvu.agreement` (CNN-vs-YOLO disagreement) has landed; this
    # is the honest confidence channel -- see the module docstring's
    # "AGREEMENT IS NOT CALIBRATION" section for why disagreement between two
    # independently trained, differently biased models is a better signal
    # than this module's own self-consistency sampling. `_render_agree_block`
    # reads only the `agree` dict's own fields (no import of
    # `surgvu.agreement` needed to render it).
    _render_agree_block,
)


def build_sampling_prompt(question, context):
    """Question + the evidence packet -> the text sent to the model for one
    sample.

    This is what makes it an EVIDENCE VLM rather than a stock one:
    `context` may carry the same shape `surgvu.perceive.clip_record` returns
    -- `tools`/`tools_present`, `task`/`task_top`, and the optional `motion`,
    `motion_v2`, `yolo`, `variant`, `agree` blocks -- alongside the ported original's
    `robot_tools`/`task_description` grounding fields, which are kept
    unchanged for backward compatibility with callers that only ever had
    those two. Every evidence block is rendered by its own function in
    `_EVIDENCE_RENDERERS`, each of which returns None to omit a block that is
    not present in `context` at all -- deliberately never "None" or an empty
    header, because a prompt full of nulls teaches the model that nulls are
    normal and wastes context that a 600s-per-case budget cannot spare (see
    `MAX_YOLO_CLASSES_IN_PROMPT`'s docstring for the measured cost this
    keeps in mind). A block that DID run but found nothing (an empty
    detector pass, a motion summary with nothing measured) is rendered as an
    explicit, short sentence rather than omitted -- that is still
    information, and different information from the block never having run.

    Only keys this function and its renderers know about are ever read;
    `context` is never dumped wholesale, so a caller-added key (a stray
    UI-derived string, say) cannot reach the model through this function no
    matter what it is named. Nothing here ever reads or renders the video's
    UI band -- `context`'s only path to a frame's content is
    `surgvu.perceive`, which blurs that band on every frame before this
    module or anything upstream of it ever sees one; see
    tests/test_evidence_prompt.py for the assertion.
    """
    lines = []
    if isinstance(context, dict):
        tools = context.get("robot_tools")
        if tools:
            lines.append(
                "Tools mounted on the robot: %s."
                % ", ".join(str(t) for t in tools))
        task_description = context.get("task_description")
        if task_description:
            lines.append("Procedure context: %s" % (task_description,))
        for renderer in _EVIDENCE_RENDERERS:
            rendered = renderer(context)
            if rendered:
                lines.append(rendered)
    lines.append("Question: %s" % (" ".join(str(question or "").split()),))
    lines.append(
        "Answer as briefly as possible -- prefer a single word or short "
        "phrase over a full sentence.")
    return "\n".join(lines)


def sample_frames(video, n=DEFAULT_FRAMES_PER_CALL):
    """`n` preprocessed frames from `video["path"]`.

    Wired to `surgvu.perceive.decode_clip`: the same evenly-spaced sampling
    and the same `preprocess.prepare_frame` UI-band blur the CNN path already
    uses, so this module cannot see anything the rest of the pipeline is
    forbidden from seeing. The import is inside this function, not at module
    scope, so the module stays importable where torch (which `perceive.py`
    imports at ITS module scope) is not installed.

    The ported original resampled a new, jittered frame set on every one of
    its (up to `max_samples`) retries, using an OpScribe scene-aware sampler
    that could actually vary what it returned call to call. `decode_clip`
    has no such jitter primitive -- it is deterministic evenly-spaced
    sampling -- so calling it again with the same arguments would decode the
    same file for byte-identical output. `adaptive_confidence_sample` below
    therefore calls this exactly once per run and reuses the same frames
    across every sampled answer; the variance the confidence signal measures
    comes from `sampling_temperature`, not from re-rolling which pixels the
    model sees.
    """
    from . import perceive
    return perceive.decode_clip(str(video["path"]), n_frames=int(n))


def sample_frames_planned(video, plan):
    """Frames for one `surgvu.frame_plan` tier -- resolution and multi-scale
    included, not just a count.

    THE THREE LEVERS IN ONE PLACE. `sample_frames` above varies only how MANY
    frames; a plan also carries how BIG (512 or 768) and whether to add a
    dense central burst on top of the evenly-spaced sweep. Those are different
    kinds of evidence, not more of the same:

      size        pixels on target. case124 failed as "Bipolar Forceps"
                  against gold "Cadiere Forceps" -- a fine-grained instrument
                  discrimination, where resolution plausibly matters more
                  than another view of the same scene.
      multiscale  a half-count burst around the clip's midpoint, reusing
                  `perceive.decode_clip_multiscale`'s probe machinery rather
                  than re-deriving temporal sampling here. Evenly-spaced
                  frames can straddle a brief event; a denser central sample
                  is a second look at the part of the clip a question about
                  "this clip" most often refers to.

    Falls back to the flat sweep if the multi-scale decode returns nothing
    usable, because a plan tier that silently produced FEWER frames than the
    tier below it would be worse than not offering the tier at all.
    """
    from . import perceive

    path = str(video["path"])
    n = int(plan["n_frames"])
    size = int(plan["size"])
    centres = perceive.decode_clip(path, n_frames=n, size=size)
    if not plan.get("multiscale"):
        return centres
    try:
        import numpy as np
        extra, _probes = perceive.decode_clip_multiscale(
            path, n_frames=max(1, n // 2), size=size)
        if extra is not None and len(extra):
            return np.concatenate([centres, extra], axis=0)
    except Exception:                            # noqa: BLE001 - see docstring
        pass
    return centres


def generation_kwargs(temperature, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
    """`model.generate` kwargs for `temperature`. Pure, so it is testable
    where torch is not installed -- the same discipline every other decision
    in this module follows.

    TEMPERATURE <= 0 MEANS GREEDY, not "sample at zero". transformers rejects
    `temperature=0.0` alongside `do_sample=True` ("has to be a strictly
    positive float"), which killed the judge probe (cluster 9707948). A caller
    asking for zero is asking for DETERMINISM, and that is `do_sample=False`,
    not a degenerate sampler.

    The judge needs exactly that: choosing between two given answers should
    not vary run to run on identical input. The answering VLM keeps sampling,
    because `adaptive_confidence_sample` derives its entire confidence signal
    from variance across samples -- greedy there would report confidence 1.0
    on every case, which is not a measurement.

    Breaks if: `temperature` is passed through alongside `do_sample=False`
    (transformers warns and ignores it, which is noise that looks like a bug).
    """
    if float(temperature) <= 0.0:
        return {"max_new_tokens": int(max_new_tokens), "do_sample": False}
    return {"max_new_tokens": int(max_new_tokens), "do_sample": True,
            "temperature": float(temperature)}


def call_vlm(frames, question, context, temperature=DEFAULT_SAMPLING_TEMPERATURE,
             model_dir=DEFAULT_MODEL_DIR, device=None, deadline=None):
    """One VLM generation over `frames`, at `temperature`.

    Loads Qwen2.5-VL-7B-Instruct directly through `transformers` -- no
    provider factory, no OpScribe import. `torch`/`transformers`/`PIL` are
    imported here, inside the function, and nowhere at module scope, so this
    file stays importable on a machine without any of them.

    `deadline`, when given, is an absolute `time.time()`-style epoch second
    (as computed by `adaptive_confidence_sample`), not a duration. It bounds
    THIS SINGLE `model.generate()` call via a `StoppingCriteria` -- the same
    mechanism the retired `surgvu.vlm._deadline_criteria` used, restored
    here as `_deadline_stopping_criteria` -- because the between-sample
    checks in `adaptive_confidence_sample` do nothing once a single
    `generate()` call has been entered: nothing else runs again until it
    returns. `deadline=None` (the default) generates with no such bound,
    which matters for a caller (a smoke test, a manual probe) that wants a
    plain, ungoverned call.

    This function may raise (OOM, a missing checkpoint, an absent CUDA
    device for a quantised build, ...). It does not swallow its own
    failures: the ported original did not either, and the fail-safe
    try/except-and-continue idiom this project uses everywhere a VLM call
    happens (see `surgvu.vlm.QwenVlmFallback.answer`) belongs at the call
    site that wires this into serving -- Task 4 -- not duplicated here.
    """
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model, processor = _load_model(model_dir, device)
    prompt = build_sampling_prompt(question, context)
    content = [{"type": "image", "image": Image.fromarray(frame[:, :, ::-1])}
               for frame in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    # TEMPERATURE <= 0 MEANS GREEDY, not "sample at zero". transformers
    # rejects `temperature=0.0` with do_sample=True ("has to be a strictly
    # positive float"), and a caller asking for zero is asking for determinism
    # -- which is `do_sample=False`, not a degenerate sampler.
    #
    # The judge needs exactly that: its job is to choose between two given
    # answers, and a decision that varies run to run on identical input is a
    # coin flip wearing a model's clothes. The answering VLM keeps sampling,
    # because `adaptive_confidence_sample` derives its whole confidence signal
    # from variance across samples (see that function's docstring).
    generate_kwargs = generation_kwargs(temperature)
    if deadline is not None:
        generate_kwargs["stopping_criteria"] = _deadline_stopping_criteria(deadline)
    with torch.inference_mode():
        generated = model.generate(**inputs, **generate_kwargs)
    prompt_length = inputs["input_ids"].shape[1]
    return processor.batch_decode(
        generated[:, prompt_length:], skip_special_tokens=True)[0]


def _deadline_stopping_criteria(deadline):
    """A `transformers.StoppingCriteriaList` that halts generation, token by
    token, once `deadline` (an absolute `time.time()`-style epoch second)
    has passed.

    Restores the mechanism the retired `surgvu.vlm._deadline_criteria` used
    for exactly the same reason: `max_new_tokens` bounds a HEALTHY
    generation, but a pathological one (a stalled node, thermal throttling)
    can spend seconds per token, and `StoppingCriteria.__call__` -- invoked
    once per generated token, the only hook `transformers.generate` offers
    inside a call already in progress -- is the one lever that still works
    once that call has been entered. `torch`/`transformers` are imported
    here, inside the function, not at module scope, matching every other
    torch import in this file (`call_vlm`, `_load_model`); this function is
    itself only ever called from inside `call_vlm`, never at import time.
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return torch.full((input_ids.shape[0],), time.time() >= deadline,
                              dtype=torch.bool, device=input_ids.device)

    return StoppingCriteriaList([_Deadline()])


#: Process-lifetime cache so repeated calls within one adaptive-sampling run
#: (2-3 per case) load the weights once, not once per sample. Keyed by
#: (model_dir, device) so a test or a future multi-model caller cannot get a
#: stale model back for a different directory.
_MODEL_CACHE = {}


def _load_model(model_dir, device):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (str(model_dir), resolved_device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, device_map={"": resolved_device})
    model.eval()
    _MODEL_CACHE[key] = (model, processor)
    return _MODEL_CACHE[key]


def release_models(keep=None):
    """Drop cached models from VRAM, optionally keeping one directory's.

    WHY THIS COSTS NOTHING WHERE IT MATTERS. `_MODEL_CACHE` exists so the 2-3
    samples of one adaptive-sampling run load the weights once. It is a
    PER-PROCESS cache, and Grand Challenge runs ONE CASE PER CONTAINER
    INVOCATION -- so once a case's VLM answer is drafted, those weights are
    never needed again in that process. Holding them is pure occupancy.

    That matters because the judge is a SECOND model. Held together, int8
    (8.76 GiB) plus the judge (2.69 GiB) is 11.45 GiB of weights on a 16 GiB
    T4, before the vision encoder's prefill activations -- the large and least
    predictable term. Released first, the peak is the LARGER of the two rather
    than their sum, which turns a tight fit into a comfortable one.

    Breaks if: this is called between samples of a single
    `adaptive_confidence_sample` run (each sample would then reload ~9 GiB),
    or if `torch.cuda.empty_cache()` is dropped -- freeing the Python
    reference alone leaves the allocator holding the memory, so nvidia-smi and
    the next allocation both still see it as used.
    """
    keep_key = str(keep) if keep is not None else None
    for key in [k for k in _MODEL_CACHE if keep_key is None or k[0] != keep_key]:
        _MODEL_CACHE.pop(key, None)
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                            # noqa: BLE001 - never fatal
        pass


def _clock():
    """The time source `adaptive_confidence_sample`'s deadline logic reads.

    A thin wrapper around `time.time()` rather than a bare reference to it,
    so a test can `monkeypatch.setattr(evidence_vlm, "_clock", fake)` --
    the same seam `sample_frames`/`call_vlm` already use -- and get a fully
    deterministic, non-sleeping clock. `adaptive_confidence_sample` always
    calls `_clock()` (a module-level lookup at call time), never a value
    captured at import time or bound into a default argument, so the
    monkeypatch takes effect for every call made after it is installed.
    """
    return time.time()


def adaptive_confidence_sample(
        video, question, context,
        frames_per_call=DEFAULT_FRAMES_PER_CALL,
        max_samples=DEFAULT_MAX_SAMPLES,
        agreement_threshold=DEFAULT_AGREEMENT_THRESHOLD,
        sampling_temperature=DEFAULT_SAMPLING_TEMPERATURE,
        budget_seconds=DEFAULT_BUDGET_SECONDS,
        frame_plan=None):
    """Sample the model repeatedly; stop when the samples agree, at the cap,
    or when `budget_seconds` runs out -- whichever comes first.

    Calls the module-level `sample_frames` once (see its docstring for why
    this run reuses one frame set rather than re-decoding per attempt), then
    calls the module-level `call_vlm` up to `max_samples` times, checking
    agreement after every sample from the second one onward.

    Stopping rule, unchanged from the ported design:
      - After each sample once at least 2 have been taken, normalise every
        answer so far, take the most common one, and compute
        `agreement = top_count / len(answers)`. If that meets
        `agreement_threshold`, stop: `agreed=True`, `confidence=agreement`.
      - If `max_samples` is exhausted without meeting the threshold, return
        the majority answer over everything sampled, `agreed=False`,
        `confidence` set to that same agreement fraction (which is by
        construction below the threshold).

    `max_samples=1` is a real, deliberately-not-special-cased edge: the loop
    never reaches the `len(answers) >= 2` check, so it falls through to the
    "exhausted" branch with one answer. `confidence` there computes to 1.0
    (`top_count / len(normalized)` = 1/1) but `agreed` is still False,
    because a single answer never had a chance to disagree with anything --
    the same reasoning the ported docstring gave for always needing a second
    data point before trusting agreement at all. A caller reading only
    `confidence` and not `agreed` would be fooled by this case; `route()`
    reads only `confidence`, which is intentional -- see its docstring.

    THE WALL-CLOCK BUDGET (see `DEFAULT_BUDGET_SECONDS` for why 240.0s):
    an absolute deadline (`_clock() + budget_seconds`) is computed once, up
    front. Two independent mechanisms enforce it, because either one alone
    is not enough:
      - BETWEEN samples: the deadline is checked at the top of every loop
        iteration, before requesting another sample. This is the cheap,
        reliable half -- it stops the loop from asking for one more sample
        once the budget is gone -- but it does nothing about a single call
        that is already in flight and hanging.
      - WITHIN one sample: the same deadline is handed to `call_vlm` as
        `deadline=`, which bounds the single `model.generate()` call itself
        via a `StoppingCriteria` (see `_deadline_stopping_criteria`) -- the
        only mechanism available once `generate()` has been entered, and
        the reason the between-sample check by itself would not be enough.

    DEGRADE, NEVER RAISE. When the deadline is hit:
      - with zero samples collected so far, this returns None -- the same
        signal `try_vlm_result` already reads as "no VLM draft", so the
        caller (`surgvu.arbiter`, via `scripts/inference.py`) falls straight
        through to the router's own answer.
      - with at least one sample collected, the majority-vote computation
        above still runs over whatever was collected and is returned as a
        real `ConfidenceResult`, with its genuine `confidence` -- but with
        `curtailed=True` and `agreed` left however it actually came out
        (never forced to True). A one-sample result curtailed by the
        budget is `confidence=1.0, agreed=False, curtailed=True`: the same
        numbers `max_samples=1` produces on its own, plus the flag that
        says WHY sampling stopped there. This distinguishes "the model
        agreed with itself" from "the budget ran out before it had the
        chance to" -- conflating the two is exactly what this fix exists to
        prevent (see the module's "AGREEMENT IS NOT CALIBRATION" section
        for why a caller must not read a curtailed single sample as if it
        were a confident, agreed one).
      - reaching agreement WINS over an already-expired deadline: agreement
        is checked immediately after each sample, before the next
        iteration's deadline check ever runs, so a run that agrees on
        exactly its last affordable sample returns `agreed=True,
        curtailed=False` -- the budget never got a chance to cut off
        anything, because there was nothing left it needed to ask for.
      - `elapsed_seconds` on the result is `_clock() - started`: the actual
        wall-clock time this run took, reported so a future measurement can
        see how close to `budget_seconds` real cases come (see
        `DEFAULT_BUDGET_SECONDS`'s own docstring on why that number is a
        measured decision, not an asserted one, and should be re-measured
        rather than trusted forever).

    `sample_frames`/`call_vlm`/`_clock` are looked up as this module's own
    globals at call time (not bound at import time), so a test can
    `monkeypatch.setattr` any of the three and this function will use the
    replacement without touching torch, perceive, or a real model, and
    without needing real elapsed time to be slow.
    """
    started = _clock()
    deadline = started + float(budget_seconds)
    # `frame_plan`, when given, carries resolution and multi-scale as well as
    # the count -- see sample_frames_planned. Falls back to the flat
    # count-only sampler so every existing caller and test is unaffected.
    frames = (sample_frames_planned(video, frame_plan) if frame_plan
              else sample_frames(video, n=frames_per_call))

    answers = []
    curtailed = False
    for _ in range(int(max_samples)):
        if _clock() >= deadline:
            curtailed = True
            break
        raw_answer = call_vlm(frames, question, context,
                              temperature=sampling_temperature,
                              deadline=deadline)
        answers.append(raw_answer)

        if len(answers) >= 2:
            normalized = [normalize_answer(a) for a in answers]
            counts = Counter(normalized)
            top_answer, top_count = counts.most_common(1)[0]
            agreement = top_count / len(normalized)
            if agreement >= float(agreement_threshold):
                return ConfidenceResult(
                    answer=top_answer,
                    confidence=agreement,
                    n_calls_used=len(answers),
                    all_answers=list(answers),
                    agreed=True,
                    curtailed=False,
                    elapsed_seconds=_clock() - started,
                )

    if not answers:
        return None

    normalized = [normalize_answer(a) for a in answers]
    counts = Counter(normalized)
    top_answer, top_count = counts.most_common(1)[0]
    agreement = top_count / len(normalized)
    return ConfidenceResult(
        answer=top_answer,
        confidence=agreement,
        n_calls_used=len(answers),
        all_answers=list(answers),
        agreed=False,
        curtailed=curtailed,
        elapsed_seconds=_clock() - started,
    )
