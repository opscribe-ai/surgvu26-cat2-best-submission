"""The one decision point between the router's answer and the VLM's draft.

WHAT THIS IS FOR
-----------------
`surgvu.router` always writes an answer: eleven regex intents, each with a
hardcoded form, so the pipeline never abstains and never says anything it
was not pre-programmed to say. `surgvu.evidence_vlm` (Task 1/2) can draft a
different answer by actually reading the evidence packet and the frames.
This module is the single place that decides, for one question, which of
those two strings ships -- under a policy selected by one config key
(`config/arbiter.json`), so switching behaviour is a config edit, not a
code change.

WHAT THIS DELIBERATELY DOES NOT DO
------------------------------------
  * It does not call the VLM. `adaptive_confidence_sample`/`call_vlm` live in
    `evidence_vlm.py`; wiring them into a live run (loading weights, timing
    the call, catching a CUDA failure) is Task 4's job in
    `scripts/inference.py`, not this module's. `arbitrate()` below takes an
    already-computed `vlm_result` (or None) as an argument.
  * It does not modify `router.py` or `evidence_vlm.py`. Both are consumed
    as-is, per the task's constraint.
  * It does not invent a confidence for the router. See
    `get_router_confidence` below -- it always returns None today, because
    nothing in `router.py` computes one, and a fabricated number here would
    silently decide every override in `challenger` mode.

THE FIVE POLICIES
-------------------
    mode         behaviour
    ----         ---------
    fallback     Router answers. VLM invoked only on unknown intent or
                 sub-floor router confidence.   <- SHIPS (see below)
    per_intent   Router answers, EXCEPT on the intents enumerated in
                 `vlm_intents`, where the VLM's draft ships. With that
                 list empty (the default) this is byte-identical to
                 `fallback`, which is what lets it ship inert and be
                 armed one measured intent at a time.
    challenger   VLM always drafts. Router wins ties. VLM overrides only
                 when router confidence is below floor AND VLM confidence
                 above ceiling.
    judge        A second model picks between the router's answer and the
                 VLM's; degrades to `challenger` when absent.
    primary      VLM answers. Router validates and rewrites the FORM so
                 BERTScore-friendly phrasing survives.

All five are implemented in full below (`_arbitrate_fallback`,
`_arbitrate_per_intent`, `_arbitrate_challenger`, `_arbitrate_judge`,
`_arbitrate_primary`); `config/arbiter.json`'s `"mode"` key picks one, and
`arbitrate()` is the only function a caller needs.

THE MODES ARE MUTUALLY EXCLUSIVE. Exactly one handler runs per question;
`fallback` never consults the judge, `per_intent` never applies
`challenger`'s confidence gate. Reading one handler tells you the whole
policy for that mode.

WHY `challenger` SHIPS -- MEASURED, NOT ASSUMED
--------------------------------------------------
`fallback` only ever consults the VLM in two circumstances: the question's
intent is one of the two UNKNOWN_* intents, or the router's confidence for
its intent is measured below a floor. The second condition is structurally
inert (see `get_router_confidence` -- there is no calibrated router
confidence to compare, anywhere, today), so `fallback`'s entire chance of
ever using the VLM reduces to "does this question's intent classify as
`INTENT_UNKNOWN_OPEN` or `INTENT_UNKNOWN_POLAR`".

Measured directly, on this repository's real public sample -- running
`router.classify_question` over all 11 questions in
`baselines/echo_question_candidates.json` (case122 through case132, the
project's own graded sample):

    case122 tool_presence_polar   case127 organ_open
    case123 tool_presence_polar   case128 tool_presence_polar
    case124 tool_identity_open    case129 procedure_open
    case125 suture_polar          case130 purpose_open
    case126 tool_presence_polar   case131 cutting_polar
                                  case132 tool_presence_polar

Zero of eleven classify as `INTENT_UNKNOWN_OPEN`; zero also classify as
`INTENT_UNKNOWN_POLAR`. So on this sample `fallback`'s ceiling is EXACTLY
ZERO: it can never once consult the VLM, let alone let it change an answer,
no matter how good that VLM is. That is precisely the freedom the plan
commissioned this arbiter to provide -- "the VLM as the last chance to get
stuff right" for anything the router has no form for -- and `fallback`
structurally cannot exercise it on the one sample available to measure
against.

`challenger` has no such gate: the VLM is assumed already drafted for every
question (Task 4's job, upstream of this module), and the override
condition depends only on the router's (always-absent) confidence and the
VLM's OWN measured confidence -- never on which of the 11 intents fired.
That is why `challenger`, not `fallback`, is `config/arbiter.json`'s
shipped default. `primary` is implemented and available but not shipped by
default: see its own docstring below for why it is the most conservative of
the three in practice (it only ever changes the polar slice of answers).

TORCH
-----
This module imports `router` and `evidence_vlm`, both of which are
torch-free at module scope (their model-loading and generation code is
imported lazily, inside functions). Nothing here imports torch at module
scope either, so `surgvu.arbiter` stays importable on this login node,
where torch is not installed at all, and in every test in
`tests/test_arbiter.py`.
"""
import json
from pathlib import Path

from . import router
from .evidence_vlm import ACCEPT, ConfidenceResult, DEFAULT_CONFIDENCE_THRESHOLD, normalize_answer, route

#: Bump when the shape of `config/arbiter.json` or `arbitrate()`'s contract
#: changes, mirroring `evidence_vlm.EVIDENCE_VLM_VERSION`.
ARBITER_VERSION = 1

#: v5.1's decision VLM. A SECOND model -- a different generation, Qwen3-VL-4B
#: rather than the answering Qwen2.5-VL-7B -- reads the question, the evidence
#: and BOTH candidates, and returns the final answer.
#:
#: Why a different model and not the warm one already loaded: reusing the
#: answering VLM is nearly free (`evidence_vlm._MODEL_CACHE` would hit) but it
#: would be marking its own homework, and it has been fine-tuned to emit terse
#: answers rather than to follow a judging instruction. A separate model is a
#: genuinely separate opinion, which is the only reason a judging stage exists.
#:
#: DEGRADES TO `challenger` when the judge is unavailable -- absent weights, a
#: failed load, an unparseable reply. The judge ships in Grand Challenge's
#: SEPARATE model tarball (/opt/ml/model/), not in the image, so "no judge
#: present" is a routine deployment state and must never be an error.
MODE_JUDGE = "judge"

MODE_FALLBACK = "fallback"
MODE_CHALLENGER = "challenger"
MODE_PRIMARY = "primary"

#: `per_intent`: the router answers, EXCEPT on an explicitly enumerated set of
#: intents where the VLM has been MEASURED to beat it, in which case the VLM's
#: draft ships unmodified.
#:
#: WHY THIS MODE EXISTS. `fallback` and `challenger` are the two ends of one
#: dial, and the leaderboard has now priced both: `fallback` (router answers
#: everything the router has a form for) scored 0.8015, `challenger` (the VLM
#: may override any intent) scored 0.7737. That 0.0278 gap is the cost of
#: letting the VLM overrule intents it is WORSE at. But "worse on average" is
#: not "worse everywhere" -- a per-intent breakdown of the same eval put the
#: router at 0.9135 and the VLM at 0.8992 overall while a per-intent best-of
#: reached 0.9229, i.e. +0.0094 over the router ALONE, available only to a
#: policy that can pick a different answerer per intent.
#:
#: THE EMPTY SET IS EXACTLY `fallback`. `vlm_intents` defaults to (), and with
#: no intents enumerated this handler's behaviour is byte-identical to
#: `_arbitrate_fallback` -- same UNKNOWN_* escape hatch, same router answer
#: everywhere else. That is deliberate: the mode can ship inert and be armed
#: one intent at a time by a config edit, so a regression is attributable to a
#: single named intent rather than to "the VLM".
MODE_PER_INTENT = "per_intent"

#: `config/arbiter.json`'s shipped value, kept here too so a caller that never
#: reads the file still gets the same choice.
#:
#: CHANGED challenger -> fallback on 2026-08-26, ON LEADERBOARD EVIDENCE.
#:
#: The "WHY `challenger` SHIPS" section above argued from the graded sample
#: that `fallback` consults the VLM on 0 of 11 questions and is therefore
#: inert. That was true and is still true -- and it turned out to be the
#: WRONG THING TO OPTIMISE. Two measurements now agree:
#:
#:   graded 11, official scorer   router-only 0.9309 vs challenger 0.8525
#:   Grand Challenge leaderboard  v2 0.8015     vs v5 (challenger) 0.7737
#:
#: The second is a different test set, produced by a real submission on a T4
#: with the VLM demonstrably running, and it moves in the same direction the
#: eleven cases predicted. Two independent signals agreeing is much stronger
#: than either alone, and it retires the hope that the graded sample was
#: simply unrepresentative on this question.
#:
#: `fallback` does NOT disable the VLM. It ships, loads, and answers questions
#: the router has no template for. What it can no longer do is override the
#: intents the router already covers -- which is where every measured
#: regression came from: two polar flips (1.0000 -> 0.7015 each), a noun that
#: fell 0.2402 -> 0.0036, and a dropped full stop worth 0.0288.
DEFAULT_MODE = MODE_FALLBACK

#: `challenger`/`fallback`'s floor for "the router's confidence is too low
#: to trust". Not independently fitted -- there is nothing to fit it
#: against, since `get_router_confidence` never returns a number today (see
#: its docstring). Kept as a documented placeholder so the config's shape
#: is ready the moment a calibrated router confidence exists.
DEFAULT_ROUTER_CONFIDENCE_FLOOR = 0.5

#: `challenger`'s ceiling for "the VLM's own confidence is high enough to
#: let it win". Reuses `evidence_vlm.DEFAULT_CONFIDENCE_THRESHOLD` (0.66) --
#: the same bar `evidence_vlm.route()` uses for its own ACCEPT decision --
#: rather than picking an independent number: a sampling run the Evidence
#: VLM itself would not ACCEPT is not a run this arbiter should trust to
#: overrule the router either.
DEFAULT_VLM_CONFIDENCE_CEILING = DEFAULT_CONFIDENCE_THRESHOLD

#: `per_intent`'s enumerated set of intents the VLM answers. EMPTY BY DEFAULT,
#: which makes the mode behave exactly like `fallback` (see MODE_PER_INTENT).
#: Populated only from a measured per-intent breakdown, never from a guess:
#: every name added here is a claim that the VLM scored higher than the router
#: on that intent, on an eval large enough for the difference to survive its
#: own standard error.
DEFAULT_VLM_INTENTS = ()

#: `config/arbiter.json`'s contents when the file is missing, unreadable, or
#: not a JSON object -- the same "degrade quietly, never raise" contract
#: `router.load_variant_priors`/`load_commercial_names` use, because a
#: missing config file must never be the reason a case scores 0.
DEFAULT_CONFIG = {
    "version": ARBITER_VERSION,
    "mode": DEFAULT_MODE,
    "router_confidence_floor": DEFAULT_ROUTER_CONFIDENCE_FLOOR,
    "vlm_confidence_ceiling": DEFAULT_VLM_CONFIDENCE_CEILING,
}

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
_ARBITER_CONFIG_PATH = _CONFIG_DIR / "arbiter.json"


def load_config(path=None):
    """Read `config/arbiter.json` -> a dict with at least `DEFAULT_CONFIG`'s
    keys; `DEFAULT_CONFIG` itself if the file is missing, unreadable, not
    valid JSON, or not a JSON object.

    Keys present in the file override the defaults; keys absent from the
    file keep their default value, so a config that only sets `"mode"` is
    still a complete, usable config. This mirrors
    `router.load_variant_priors`'s degrade-quietly contract rather than
    raising, for the same reason: a malformed config file must degrade the
    POLICY, never take down the run.
    """
    resolved = Path(path) if path is not None else _ARBITER_CONFIG_PATH
    try:
        data = json.loads(Path(resolved).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(DEFAULT_CONFIG)
    if not isinstance(data, dict):
        return dict(DEFAULT_CONFIG)
    merged = dict(DEFAULT_CONFIG)
    merged.update(data)
    return merged


def get_router_confidence(intent):
    """The router's own calibrated confidence for `intent`, or None.

    Always None today. `router.py` (consumed as-is by this task, not
    modified) computes an ANSWER for every intent via `ANSWER_FORMS`, but
    never a confidence number alongside it -- its only three uses of the
    word "confidence" are prose in docstrings, none of them a return value.
    Returning a fabricated number here (a flat 0.5, or a per-intent guess)
    would let `_arbitrate_fallback`/`_arbitrate_challenger` silently decide
    every override on an unmeasured constant, which is exactly what
    requirement 5 of this task forbids.

    `_arbitrate_fallback` and `_arbitrate_challenger` both call this and
    both handle the None it returns EXPLICITLY, and DIFFERENTLY: see their
    docstrings for why the same absence is read as "no evidence, do
    nothing" in one mode and "nothing vouches for this, treat as
    challengeable" in the other. Neither ever substitutes a number for the
    None.

    `intent` is accepted (this function does not need it today) so a future
    calibration effort has a natural per-intent slot to populate without
    changing either caller's signature.
    """
    return None


def _is_usable_vlm_result(vlm_result):
    """False for None, a wrong type, an empty/whitespace answer, or a
    non-finite confidence -- every shape "absent, failed, or returned
    nothing usable" (requirement 2) can take.

    `arbitrate()` calls this before doing anything mode-specific, so a
    malformed `vlm_result` can never reach a mode handler. This function
    itself never raises -- it only ever returns True/False -- so a caller
    that hands `arbitrate` a stray dict, a bare string, or a half-built
    `ConfidenceResult` gets the router's answer back, not a traceback. That
    matters because the container's one hard guarantee is that it always
    writes an answer: an exception here would be exactly the class of
    failure the R18 idiom (`scripts/inference.py`'s `try_vlm`) exists to
    keep away from the final write.
    """
    if not isinstance(vlm_result, ConfidenceResult):
        return False
    answer = vlm_result.answer
    if answer is None or not str(answer).strip():
        return False
    try:
        confidence = float(vlm_result.confidence)
    except (TypeError, ValueError):
        return False
    return confidence == confidence  # False only for NaN


def _extract_polarity(text):
    """The VLM answer's leading yes/no token, mapped to the router's own
    two-word vocabulary ("Yes"/"No"), or None if it does not lead with one.

    Deliberately narrow: only the FIRST token, after
    `evidence_vlm.normalize_answer`'s case/punctuation normalisation, is
    inspected. "No sign of bleeding, but yes there is a clip" is not read as
    "Yes" merely because the word occurs in it -- a substring scan would
    repeat exactly the mistake `evidence_vlm.py`'s module docstring
    documents as NOT ported from the original `is_correct()` (`pred ==
    a_clean or pred in a_clean`), which inflated agreement between strings
    that do not actually agree. Anything without a clean leading yes/no
    token (a hedge, a description) returns None, which `_arbitrate_primary`
    treats as "nothing usable for this purpose" and answers from the router
    instead.
    """
    normalized = normalize_answer(text)
    tokens = normalized.split()
    if not tokens:
        return None
    first = tokens[0].strip(",;:.")
    if first == "yes":
        return "Yes"
    if first == "no":
        return "No"
    return None


def _arbitrate_fallback(question, perception, router_answer, vlm_result, cfg):
    """`fallback`: the router answers; the VLM's (already-usable, per
    `_is_usable_vlm_result`) draft is used only when this question's intent
    is one of the two UNKNOWN_* intents, or the router's confidence for
    that intent is measured below `cfg["router_confidence_floor"]`.

    THE SECOND CONDITION IS STRUCTURALLY INERT TODAY. `get_router_confidence`
    always returns None (see its docstring), and None is read here as "no
    evidence of low confidence, so do nothing" -- appropriate for a mode
    that is cautious BY DESIGN and needs an affirmative, measured reason to
    act, not merely the absence of one. That is a real number comparison
    (`confidence < floor`, guarded by `confidence is not None`), not a
    fabricated stand-in: with no confidence available, there is nothing to
    compare, so the branch cannot fire, and this docstring records why
    rather than leaving the reader to wonder if it was forgotten.

    See the module docstring's "WHY `challenger` SHIPS" section for the
    measured consequence: with the confidence branch inert, this mode's
    entire chance of consulting the VLM is "does the intent classify as
    UNKNOWN_OPEN or UNKNOWN_POLAR", which is 0/11 on the graded sample.
    """
    intent = router.classify_question(question)
    if intent in (router.INTENT_UNKNOWN_OPEN, router.INTENT_UNKNOWN_POLAR):
        return vlm_result.answer
    floor = float(cfg.get("router_confidence_floor", DEFAULT_ROUTER_CONFIDENCE_FLOOR))
    confidence = get_router_confidence(intent)
    if confidence is not None and confidence < floor:
        return vlm_result.answer
    return router_answer


def _arbitrate_challenger(question, perception, router_answer, vlm_result, cfg):
    """`challenger`: the VLM is assumed already drafted (by the caller,
    upstream of this module -- Task 4's job); the router wins ties; the VLM
    overrides only when the router's confidence is below
    `cfg["router_confidence_floor"]` AND the VLM's own confidence clears
    `cfg["vlm_confidence_ceiling"]` (checked by delegating to
    `evidence_vlm.route()` rather than re-implementing the same
    threshold comparison a second time).

    THE SAME ABSENCE IS READ OPPOSITE TO `fallback`, ON PURPOSE.
    `get_router_confidence` returns None here exactly as it does for
    `fallback`, but this function treats None as "router confidence is
    below floor" -- the OPPOSITE of `fallback`'s "do nothing". That is not
    an inconsistency; it is what the two modes are FOR. `fallback` is
    cautious by design and needs affirmative, measured evidence of low
    confidence before it will act. `challenger`'s entire reason to exist
    (see the module docstring) is that the router's correctness has never
    been measured per-question at all -- so "no calibrated confidence"
    here means "nothing vouches for this particular router answer", which
    is read as challengeable, not as trustworthy-by-default. Both are
    documented POLICY CHOICES about how to treat one None value; neither
    substitutes a number for it -- the override still requires the VLM's
    OWN measured confidence to independently clear its ceiling, so an
    unproven router answer is only ever replaced by a VLM draft that is
    itself confident, never by default.
    """
    floor = float(cfg.get("router_confidence_floor", DEFAULT_ROUTER_CONFIDENCE_FLOOR))
    ceiling = float(cfg.get("vlm_confidence_ceiling", DEFAULT_VLM_CONFIDENCE_CEILING))
    intent = router.classify_question(question)
    confidence = get_router_confidence(intent)
    router_below_floor = confidence is None or confidence < floor
    vlm_above_ceiling = route(vlm_result, confidence_threshold=ceiling)["decision"] == ACCEPT
    if router_below_floor and vlm_above_ceiling:
        return vlm_result.answer
    return router_answer


def _arbitrate_primary(question, perception, router_answer, vlm_result, cfg):
    """`primary`: the VLM answers; the router validates and rewrites the
    FORM so BERTScore-friendly phrasing survives.

    THE SINGLE MOST IMPORTANT THING ABOUT THIS MODE (requirement 4). Every
    polar intent's form is drawn from a vocabulary of exactly two strings --
    grep `router.ANSWER_FORMS`'s polar entries and every one of them returns
    `FALLBACK_POLAR`, its opposite via `_POLAR_OPPOSITE`, or a literal
    "Yes"/"No", never anything else. A vocabulary that small can absorb the
    VLM's CONTENT (which of the two it means) while keeping the router's
    FORM (which of the two strings is emitted) -- `_extract_polarity` reads
    only a clean leading yes/no token off the VLM's answer and maps it onto
    that same two-word vocabulary; anything else (a hedge, a description
    with no such token) is "nothing usable for this purpose", and this mode
    keeps `router_answer` outright, content and form both.

    OPEN INTENTS ARE DELIBERATELY **NOT** BRIDGED THE SAME WAY. Their forms
    are drawn from open, per-intent vocabularies (a tool name, an organ, a
    task label, a count word...) with no extraction function anywhere in
    this codebase, and improvising one here is exactly the "easy to get
    wrong" move requirement 4 warns about: a wrong guess at "the VLM's
    content, expressed in the router's form" is not distinguishable, from
    reading this function alone, from a bug that leaks raw VLM prose
    straight into a BERTScore-graded answer -- and a wrong noun answer can
    score NEGATIVE (see `router.py`'s own measured -0.086), which is a
    strictly worse failure than this mode simply not touching the answer.
    So for every non-polar intent this returns `router_answer` unchanged:
    the router's form wins outright, and `primary` differs from `fallback`
    and `challenger` only on the polar slice of questions -- a deliberately
    conservative reading of a mode whose name suggests the opposite.
    """
    if router.is_polar_question(question):
        polarity = _extract_polarity(vlm_result.answer)
        if polarity is not None:
            return polarity
    return router_answer


def known_intents():
    """Every intent string `router.classify_question` can return.

    Read off the router's own `INTENT_*` module attributes rather than
    restated as a literal here, so an intent added to `router.py` is
    automatically eligible for `per_intent` without a second edit in this
    file -- and, more importantly, so a TYPO in `config/arbiter.json` cannot
    silently name an intent that does not exist and then never fire.
    """
    return frozenset(
        value
        for name, value in vars(router).items()
        if name.startswith("INTENT_") and isinstance(value, str)
    )


def resolve_vlm_intents(cfg):
    """`cfg["vlm_intents"]` narrowed to intents that actually exist.

    UNKNOWN NAMES ARE DROPPED, NOT RAISED ON. A config naming an intent this
    router has never heard of is a typo or a config from a newer version, and
    the module-wide contract (see `arbitrate`'s docstring, and
    `_is_usable_vlm_result`) is that no config problem may ever be the reason
    a case fails to produce an answer. Dropping the name degrades that intent
    to the router -- the safe direction, and the one `fallback` already takes.

    A non-list value degrades to the empty set for the same reason.
    """
    raw = cfg.get("vlm_intents", DEFAULT_VLM_INTENTS)
    if isinstance(raw, str) or not isinstance(raw, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(raw) & known_intents()


def _arbitrate_per_intent(question, perception, router_answer, vlm_result, cfg):
    """`per_intent`: the router answers, except on the intents named in
    `cfg["vlm_intents"]`, where the VLM's draft ships.

    NO CONFIDENCE GATE, ON PURPOSE. `challenger` requires the VLM's own
    confidence to clear a ceiling before it may override; this mode does not,
    and the difference is not an oversight. The per-intent measurement that
    justifies an entry in `vlm_intents` is taken over ALL eval records for
    that intent, unconditionally -- it is the statement "on this intent, the
    VLM's answers score higher than the router's", including its low
    confidence ones. Adding a confidence gate here would ship a policy nobody
    measured: the high-confidence subset of an intent is a different
    population with a different mean, and the gate would silently hand the
    remainder back to the router at an unknown score. If a confidence-gated
    variant is ever wanted, it should be measured as such first.

    THE UNKNOWN_* ESCAPE HATCH IS KEPT. Both UNKNOWN intents go to the VLM
    regardless of `vlm_intents`, exactly as in `_arbitrate_fallback`: the
    router has no form for them at all, so its "answer" there is a generic
    fallback string, and the VLM cannot do worse than a string written
    without reference to the question.
    """
    intent = router.classify_question(question)
    if intent in (router.INTENT_UNKNOWN_OPEN, router.INTENT_UNKNOWN_POLAR):
        return vlm_result.answer
    if intent in resolve_vlm_intents(cfg):
        return vlm_result.answer
    return router_answer


def _arbitrate_judge(question, perception, router_answer, vlm_result, cfg):
    """`judge`: a second model decides, or `challenger` decides if it cannot.

    THE JUDGE IS OPTIONAL BY CONSTRUCTION. `cfg["judge_fn"]` is injected by
    the caller (scripts/inference.py), which owns model loading; this module
    imports no torch and loads nothing. When it is absent -- no sidecar
    tarball, a failed load, a reply that parsed to nothing -- this falls
    through to `_arbitrate_challenger`, which is the shipped v5 behaviour.
    That is not a defensive nicety: the judge lives in Grand Challenge's
    separate model tarball rather than the image, so running without one is a
    routine deployment state.

    SKIPS ITSELF WHEN THE CANDIDATES AGREE (`judge.should_consult`). A second
    VLM pass is not free, and there is nothing to arbitrate when both answers
    already say the same thing.

    Breaks if: a judge failure propagates instead of falling back (a missing
    response scores 0, strictly worse than either candidate), or if the
    agreement skip is removed (the judge then runs on every case and spends
    budget it cannot change the outcome with).
    """
    from . import judge as judge_mod

    challenger_answer = _arbitrate_challenger(
        question, perception, router_answer, vlm_result, cfg)

    judge_fn = cfg.get("judge_fn")
    if judge_fn is None:
        return challenger_answer

    vlm_answer = getattr(vlm_result, "answer", None)
    if not judge_mod.should_consult(router_answer, vlm_answer):
        return challenger_answer

    try:
        reply = judge_fn(question, perception, (router_answer, vlm_answer))
    except Exception:                            # noqa: BLE001 - see docstring
        return challenger_answer
    answer, _source = judge_mod.parse_judgement(
        reply, (router_answer, vlm_answer))
    if not answer or not str(answer).strip():
        return challenger_answer
    return answer


_MODE_HANDLERS = {
    MODE_FALLBACK: _arbitrate_fallback,
    MODE_CHALLENGER: _arbitrate_challenger,
    MODE_PRIMARY: _arbitrate_primary,
    MODE_PER_INTENT: _arbitrate_per_intent,
    MODE_JUDGE: _arbitrate_judge,
}


def arbitrate(question, perception, vlm_result=None, config=None):
    """The one decision point: the router's answer, the VLM's, or a
    policy-blended result, chosen by `config["mode"]` (`load_config()`'s
    result when `config` is not given).

    THE FALL-THROUGH RUNS BEFORE ANY MODE LOGIC, AND IS THE SAME FOR ALL
    THREE MODES. `router.answer_question(question, perception)` is computed
    first, unconditionally, and is returned immediately -- untouched by any
    mode, any config value, any handler -- whenever `vlm_result` is absent,
    of the wrong type, or usable-shaped but empty/non-finite (see
    `_is_usable_vlm_result`). That return value is therefore BYTE-IDENTICAL
    to calling `router.answer_question` directly, which is the property
    that lets this module land before Task 4 wires a real VLM into
    `scripts/inference.py`, exactly as Task 10b's perception blocks landed
    before Task 11 populated them. `tests/test_arbiter.py` asserts this
    equality directly, for every mode, not just for the shipped default.

    Only once a usable `vlm_result` exists does `config["mode"]` matter at
    all. An unrecognised mode string (a typo in `config/arbiter.json`, a
    config from a future version this code has not seen) degrades to the
    same fall-through -- `router_answer` -- rather than raising, for the
    same "the container always writes an answer" reason
    `_is_usable_vlm_result` exists.

    A mode handler's result is passed through `router.finalize_answer`
    before being returned: the router's own single exit point
    (whitespace-collapse, capitalise-first-character, never-empty), applied
    here too so a VLM-derived answer gets the same treatment a
    router-derived one already had. This is idempotent on an already
    finalized string, so it does not disturb the byte-identical fall-through
    property above (that branch returns `router_answer` directly, before
    this line, and never re-enters it).
    """
    router_answer = router.answer_question(question, perception)
    if not _is_usable_vlm_result(vlm_result):
        return router_answer
    cfg = config if config is not None else load_config()
    mode = cfg.get("mode", DEFAULT_MODE)
    handler = _MODE_HANDLERS.get(mode)
    if handler is None:
        return router_answer
    result = handler(question, perception, router_answer, vlm_result, cfg)
    return router.finalize_answer(result)
