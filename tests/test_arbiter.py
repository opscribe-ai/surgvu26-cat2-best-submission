"""Tests for the arbiter: the one decision point between the router's answer
and the VLM's draft.

Torch-free, mirroring tests/test_evidence_vlm.py and tests/test_router.py:
nothing here needs a real model, a real video, or a real perception record --
`ConfidenceResult` is a plain dataclass and `router.answer_question` runs on
a bare dict, so every test executes in milliseconds on a login node without
torch installed.
"""
import ast
import inspect
import json
import math

import pytest

from surgvu import arbiter, router
from surgvu.evidence_vlm import ConfidenceResult

# --------------------------------------------------------------------------
# The 11 real graded sample questions (case122-case132), used to measure --
# not assume -- that INTENT_UNKNOWN_OPEN/POLAR fire on 0 of them. Loaded
# directly from the file the module docstring cites, so this file breaks
# (loudly) if that measurement's source ever moves or changes shape, rather
# than silently drifting out of sync with the docstring's claim.
# --------------------------------------------------------------------------
_SAMPLE_QUESTIONS_PATH = "baselines/echo_question_candidates.json"


def _load_sample_questions():
    with open(_SAMPLE_QUESTIONS_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    return list(data.values())


SAMPLE_QUESTIONS = _load_sample_questions()


def usable_result(answer="Bleeding from the cystic plate", confidence=0.9):
    """A `ConfidenceResult` that `_is_usable_vlm_result` accepts."""
    return ConfidenceResult(answer=answer, confidence=confidence,
                             n_calls_used=1, all_answers=[answer], agreed=True)


# --------------------------------------------------------------------------
# the 0/11 measurement the module docstring rests its shipped default on
# --------------------------------------------------------------------------

def test_sample_has_eleven_questions():
    assert len(SAMPLE_QUESTIONS) == 11


def test_intent_unknown_open_fires_on_zero_of_eleven_sample_questions():
    """The precise measurement the module docstring cites as the reason
    `fallback` is not shipped by default: this is what makes `fallback`'s
    ceiling on this sample exactly zero, since the confidence branch is
    structurally inert (see test_router_confidence_is_never_fabricated)."""
    intents = [router.classify_question(q) for q in SAMPLE_QUESTIONS]
    assert intents.count(router.INTENT_UNKNOWN_OPEN) == 0


def test_intent_unknown_polar_also_fires_on_zero_of_eleven_sample_questions():
    intents = [router.classify_question(q) for q in SAMPLE_QUESTIONS]
    assert intents.count(router.INTENT_UNKNOWN_POLAR) == 0


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def test_config_file_is_valid_json_with_the_shipped_mode():
    """per_intent since 2026-08-29, arming `tool_identity_open` on v6 stage 2's
    measurement: scripts/per_intent_table.py put the router at 0.2206 and the
    VLM at 0.8777 on that intent (delta +0.6571, se 0.0684 -- about 9.6 sigma,
    n=41 of 300). Every other intent was a tie or inside the 2-sigma bar.

    Was `fallback` from 2026-08-26. The literal is no longer asserted: what
    matters is that the shipped mode is one `arbiter` actually implements, and
    -- the part that can really go wrong -- that anything armed is a REAL
    intent name. `resolve_vlm_intents` silently drops names that are not,
    so a typo does not fail loudly; it just quietly arms nothing and the
    config still looks armed.
    """
    with open("config/arbiter.json", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["mode"] in arbiter._MODE_HANDLERS
    assert data["version"] == 1
    assert isinstance(data["version"], int)

    armed = data.get("vlm_intents", [])
    assert set(armed) <= arbiter.known_intents(), (
        "config/arbiter.json arms an intent name that does not exist; "
        "resolve_vlm_intents would drop it silently: "
        "%s" % sorted(set(armed) - arbiter.known_intents()))
    assert set(arbiter.resolve_vlm_intents(data)) == set(armed), (
        "the armed set does not survive resolve_vlm_intents intact")


def test_config_file_round_trips_through_json_dumps():
    """Strict-JSON-safe: every value in the shipped config must itself be a
    plain JSON type, not (for instance) a numpy scalar or a Decimal."""
    with open("config/arbiter.json", encoding="utf-8") as fh:
        data = json.load(fh)
    assert json.loads(json.dumps(data)) == data


def test_load_config_reads_the_shipped_file_by_default():
    """Reads the FILE, not DEFAULT_CONFIG. Since 2026-08-29 those two disagree
    on `mode` -- the file ships `per_intent`, the in-code default stays
    `fallback` -- which makes this test sharper than it was when both said
    `fallback` and it could not tell the two sources apart.
    """
    with open("config/arbiter.json", encoding="utf-8") as fh:
        on_disk = json.load(fh)

    cfg = arbiter.load_config()
    assert cfg["mode"] == on_disk["mode"]
    assert cfg["version"] == 1


def test_load_config_missing_file_degrades_to_default(tmp_path):
    cfg = arbiter.load_config(tmp_path / "does_not_exist.json")
    assert cfg == arbiter.DEFAULT_CONFIG


def test_load_config_malformed_json_degrades_to_default(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    assert arbiter.load_config(path) == arbiter.DEFAULT_CONFIG


def test_load_config_non_object_json_degrades_to_default(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert arbiter.load_config(path) == arbiter.DEFAULT_CONFIG


def test_load_config_merges_partial_file_over_defaults(tmp_path):
    """A config that only sets `mode` is still a complete, usable config --
    the floor/ceiling keys keep their default values rather than vanishing."""
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"mode": "primary"}), encoding="utf-8")
    cfg = arbiter.load_config(path)
    assert cfg["mode"] == "primary"
    assert cfg["router_confidence_floor"] == arbiter.DEFAULT_ROUTER_CONFIDENCE_FLOOR
    assert cfg["vlm_confidence_ceiling"] == arbiter.DEFAULT_VLM_CONFIDENCE_CEILING


# --------------------------------------------------------------------------
# requirement 5: no fabricated router confidence
# --------------------------------------------------------------------------

@pytest.mark.parametrize("intent", [
    router.INTENT_TOOL_PRESENCE, router.INTENT_TOOL_IDENTITY, router.INTENT_ORGAN,
    router.INTENT_CUTTING, router.INTENT_SUTURE, router.INTENT_PROCEDURE,
    router.INTENT_PURPOSE, router.INTENT_TASK, router.INTENT_COUNT,
    router.INTENT_UNKNOWN_POLAR, router.INTENT_UNKNOWN_OPEN,
])
def test_router_confidence_is_never_fabricated(intent):
    """No intent gets a made-up number; every one of the 11 intents returns
    None. A single non-None return anywhere would mean some intent silently
    started deciding overrides on a fabricated constant."""
    assert arbiter.get_router_confidence(intent) is None


# --------------------------------------------------------------------------
# requirement 2/3: every mode falls through, byte-identically, independently
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", [arbiter.MODE_FALLBACK, arbiter.MODE_CHALLENGER, arbiter.MODE_PRIMARY])
@pytest.mark.parametrize("question", SAMPLE_QUESTIONS)
def test_fallthrough_is_byte_identical_when_vlm_result_is_none(question, mode):
    perception = {}
    assert (arbiter.arbitrate(question, perception, vlm_result=None, config={"mode": mode})
            == router.answer_question(question, perception))


@pytest.mark.parametrize("mode", [arbiter.MODE_FALLBACK, arbiter.MODE_CHALLENGER, arbiter.MODE_PRIMARY])
def test_fallthrough_when_vlm_result_has_empty_answer(mode):
    """Present, but the model generated nothing -- 'returned nothing
    usable', not 'absent'."""
    question = "Is a needle driver being used?"
    perception = {}
    empty = ConfidenceResult(answer="   ", confidence=0.9, n_calls_used=1,
                              all_answers=["   "], agreed=True)
    assert (arbiter.arbitrate(question, perception, vlm_result=empty, config={"mode": mode})
            == router.answer_question(question, perception))


@pytest.mark.parametrize("mode", [arbiter.MODE_FALLBACK, arbiter.MODE_CHALLENGER, arbiter.MODE_PRIMARY])
def test_fallthrough_when_vlm_confidence_is_nan(mode):
    question = "Is a needle driver being used?"
    perception = {}
    broken = ConfidenceResult(answer="Yes", confidence=float("nan"), n_calls_used=1,
                               all_answers=["Yes"], agreed=False)
    assert (arbiter.arbitrate(question, perception, vlm_result=broken, config={"mode": mode})
            == router.answer_question(question, perception))


@pytest.mark.parametrize("mode", [arbiter.MODE_FALLBACK, arbiter.MODE_CHALLENGER, arbiter.MODE_PRIMARY])
def test_fallthrough_when_vlm_result_is_the_wrong_type(mode):
    """A caller that hands over a plain dict or string instead of a
    ConfidenceResult must not crash the arbiter -- it must fall through."""
    question = "Is a needle driver being used?"
    perception = {}
    for bogus in ({"answer": "Yes", "confidence": 0.9}, "Yes", 42, []):
        assert (arbiter.arbitrate(question, perception, vlm_result=bogus, config={"mode": mode})
                == router.answer_question(question, perception))


def test_fallthrough_with_no_config_given_uses_the_shipped_default():
    question = "Is a needle driver being used?"
    perception = {}
    assert arbiter.arbitrate(question, perception) == router.answer_question(question, perception)


def test_unrecognised_mode_string_falls_through_to_router():
    question = "Is a needle driver being used?"
    perception = {}
    result = usable_result(answer="Yes, definitely", confidence=0.99)
    assert (arbiter.arbitrate(question, perception, vlm_result=result,
                              config={"mode": "not_a_real_mode"})
            == router.answer_question(question, perception))


# --------------------------------------------------------------------------
# fallback mode
# --------------------------------------------------------------------------

def test_fallback_uses_vlm_on_unknown_open_intent():
    question = "What complication is developing?"  # -> INTENT_UNKNOWN_OPEN
    assert router.classify_question(question) == router.INTENT_UNKNOWN_OPEN
    perception = {}
    result = usable_result(answer="Bleeding from the cystic artery", confidence=0.4)
    out = arbiter.arbitrate(question, perception, vlm_result=result,
                            config={"mode": "fallback"})
    assert out == "Bleeding from the cystic artery"


def test_fallback_ignores_vlm_on_a_known_intent_even_with_high_vlm_confidence():
    """The whole point of the 0/11 measurement: on a KNOWN intent, fallback
    never listens to the VLM, no matter how confident it is, because the
    router-confidence branch is structurally inert (get_router_confidence is
    always None, which fallback reads as "no evidence, do nothing")."""
    question = "Is a needle driver being used?"
    assert router.classify_question(question) == router.INTENT_TOOL_PRESENCE
    perception = {}
    result = usable_result(answer="Yes, obviously", confidence=0.99)
    out = arbiter.arbitrate(question, perception, vlm_result=result,
                            config={"mode": "fallback"})
    assert out == router.answer_question(question, perception)


# --------------------------------------------------------------------------
# challenger mode (the shipped default)
# --------------------------------------------------------------------------

def test_challenger_overrides_a_known_intent_when_vlm_confidence_clears_ceiling():
    """Unlike fallback, challenger CAN override a known-intent answer: the
    absent router confidence is read as below-floor here, so only the VLM's
    own confidence gates the override."""
    question = "Is a needle driver being used?"  # perception={} -> router says "No"
    perception = {}
    assert router.answer_question(question, perception) == "No"
    result = usable_result(answer="Yes, a needle driver is clearly visible", confidence=0.9)
    out = arbiter.arbitrate(question, perception, vlm_result=result,
                            config={"mode": "challenger", "vlm_confidence_ceiling": 0.66})
    assert out == "Yes, a needle driver is clearly visible"


def test_challenger_router_wins_when_vlm_confidence_at_or_below_ceiling():
    question = "Is a needle driver being used?"
    perception = {}
    result = usable_result(answer="Yes, a needle driver is clearly visible", confidence=0.5)
    out = arbiter.arbitrate(question, perception, vlm_result=result,
                            config={"mode": "challenger", "vlm_confidence_ceiling": 0.66})
    assert out == router.answer_question(question, perception)


def test_challenger_router_wins_exactly_at_the_ceiling():
    """Ties go to the router: evidence_vlm.route() ACCEPTs at >=, so a
    confidence exactly equal to the ceiling DOES accept in evidence_vlm's
    own terms -- documenting this boundary explicitly rather than assuming
    a direction, since evidence_vlm.route() (not this module) owns it."""
    question = "Is a needle driver being used?"
    perception = {}
    ceiling = 0.66
    result = usable_result(answer="Yes, a needle driver is clearly visible", confidence=ceiling)
    out = arbiter.arbitrate(question, perception, vlm_result=result,
                            config={"mode": "challenger", "vlm_confidence_ceiling": ceiling})
    # route() ACCEPTs at == threshold, so this overrides -- pinned so a
    # change to route()'s comparison operator is caught here too.
    assert out == "Yes, a needle driver is clearly visible"


def test_challenger_intent_does_not_matter_only_the_two_confidences_do():
    """The defining difference from fallback: an UNKNOWN intent gets no
    special treatment here -- override fires (or not) purely off the
    confidence comparison, on unknown intents exactly as on known ones."""
    question = "What complication is developing?"  # unknown_open
    perception = {}
    low = usable_result(answer="Bleeding from the cystic artery", confidence=0.3)
    out_low = arbiter.arbitrate(question, perception, vlm_result=low,
                                config={"mode": "challenger", "vlm_confidence_ceiling": 0.66})
    assert out_low == router.answer_question(question, perception)

    high = usable_result(answer="Bleeding from the cystic artery", confidence=0.9)
    out_high = arbiter.arbitrate(question, perception, vlm_result=high,
                                 config={"mode": "challenger", "vlm_confidence_ceiling": 0.66})
    assert out_high == "Bleeding from the cystic artery"


def test_challenger_default_ceiling_matches_evidence_vlm_accept_threshold():
    """The default is not an independent guess: a case the Evidence VLM
    itself would not ACCEPT (evidence_vlm.route()'s own threshold) should
    not be trusted to overrule the router either."""
    from surgvu.evidence_vlm import DEFAULT_CONFIDENCE_THRESHOLD
    assert arbiter.DEFAULT_VLM_CONFIDENCE_CEILING == DEFAULT_CONFIDENCE_THRESHOLD


# --------------------------------------------------------------------------
# primary mode: form always wins; content only flows through the polar slot
# --------------------------------------------------------------------------

def test_primary_rewrites_polar_content_into_the_routers_terse_form():
    """The single most important behaviour of `primary`: the VLM's verbose
    answer is never emitted verbatim -- only its Yes/No content survives,
    in the router's own one-word form."""
    question = "Is a needle driver being used?"
    perception = {}
    assert router.answer_question(question, perception) == "No"
    result = usable_result(answer="Yes, a needle driver is clearly visible in this frame",
                           confidence=0.9)
    out = arbiter.arbitrate(question, perception, vlm_result=result, config={"mode": "primary"})
    assert out == "Yes"
    assert "clearly visible" not in out


def test_primary_keeps_router_answer_when_vlm_polarity_is_unclear():
    """No clean leading yes/no token -- "nothing usable for this purpose" --
    so both form and content come from the router, unchanged."""
    question = "Is a needle driver being used?"
    perception = {}
    result = usable_result(answer="Hard to tell from this camera angle", confidence=0.9)
    out = arbiter.arbitrate(question, perception, vlm_result=result, config={"mode": "primary"})
    assert out == router.answer_question(question, perception)


def test_primary_never_leaks_vlm_content_into_an_open_answer():
    """requirement 4's central risk: an open-intent question must return the
    router's own answer unchanged, even though the VLM drafted (confident,
    fluent, plausible) different content."""
    question = "What instrument is being used?"
    perception = {}
    router_answer = router.answer_question(question, perception)
    result = usable_result(answer="A large clip applier is grasping the vessel", confidence=0.95)
    out = arbiter.arbitrate(question, perception, vlm_result=result, config={"mode": "primary"})
    assert out == router_answer
    assert "clip applier" not in out.lower()


def test_primary_open_question_ignores_config_floor_and_ceiling():
    """primary's decision does not depend on router_confidence_floor or
    vlm_confidence_ceiling at all -- an extreme config must not change its
    open-question behaviour."""
    question = "What instrument is being used?"
    perception = {}
    result = usable_result(answer="A stapler", confidence=0.01)
    out = arbiter.arbitrate(question, perception, vlm_result=result,
                            config={"mode": "primary", "router_confidence_floor": 0.99,
                                    "vlm_confidence_ceiling": 0.0})
    assert out == router.answer_question(question, perception)


# --------------------------------------------------------------------------
# one config key switches behaviour with no code change
# --------------------------------------------------------------------------

def test_switching_the_mode_key_alone_changes_the_outcome():
    """Proof that mode selection is a config edit, not a code change: the
    exact same question, perception, and VLM result produce different
    answers depending solely on config['mode']."""
    question = "Is a needle driver being used?"
    perception = {}
    result = usable_result(answer="Yes, a needle driver is clearly visible", confidence=0.9)

    fallback_out = arbiter.arbitrate(question, perception, vlm_result=result,
                                     config={"mode": "fallback"})
    challenger_out = arbiter.arbitrate(question, perception, vlm_result=result,
                                       config={"mode": "challenger",
                                               "vlm_confidence_ceiling": 0.66})
    primary_out = arbiter.arbitrate(question, perception, vlm_result=result,
                                    config={"mode": "primary"})

    assert fallback_out == "No"                                    # ignored the VLM
    assert challenger_out == "Yes, a needle driver is clearly visible"  # overrode, verbatim
    assert primary_out == "Yes"                                    # overrode, router's form


# --------------------------------------------------------------------------
# torch-free at module scope
# --------------------------------------------------------------------------

def test_no_module_scope_torch_import():
    """`import torch` (or `from torch import ...`) must not appear as a
    top-level statement in arbiter.py. A function-body import is fine and
    untouched by this check -- this walks only `tree.body`, the module's
    direct statements, not nested function definitions."""
    source = inspect.getsource(arbiter)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not any(alias.name.split(".")[0] == "torch" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or node.module.split(".")[0] != "torch"


def test_module_is_already_imported_without_torch():
    """If arbiter.py imported torch at module scope, merely importing
    `surgvu.arbiter` (done at the top of this file) would already have
    failed on this login node, where torch is not installed at all."""
    import sys
    assert "surgvu.arbiter" in sys.modules


# --------------------------------------------------------------------------
# documentation is a contract
# --------------------------------------------------------------------------

def test_module_docstring_records_the_zero_of_eleven_measurement():
    doc = (arbiter.__doc__ or "")
    assert "0/11" in doc or "0 of 11" in doc or "Zero of eleven" in doc


def test_module_docstring_names_challenger_as_shipped():
    doc = (arbiter.__doc__ or "").lower()
    assert "challenger" in doc and "ships" in doc


def test_get_router_confidence_docstring_says_it_is_never_fabricated():
    doc = (arbiter.get_router_confidence.__doc__ or "").lower()
    assert "fabricat" in doc
    assert "none" in doc


def test_exactly_the_five_known_modes_are_registered():
    """Pinned so a mode cannot be added without a deliberate edit here. `judge`
    joined on 2026-08-26 (v5.1's decision VLM); `per_intent` on 2026-08-27,
    to hold the enumerated intent set the per-intent eval populates."""
    assert set(arbiter._MODE_HANDLERS) == {
        "fallback", "challenger", "primary", "judge", "per_intent"}


def test_judge_mode_without_a_judge_is_exactly_challenger():
    """THE DEPLOYMENT-STATE TEST. The judge ships in Grand Challenge's
    SEPARATE model tarball, not in the image, so 'no judge present' is routine
    rather than exceptional -- and must be indistinguishable from the shipped
    v5 behaviour, not a degraded or empty answer."""
    perception = {}
    vlm_result = ConfidenceResult(answer="Yes", confidence=1.0, n_calls_used=2,
                                 all_answers=["Yes", "Yes"], agreed=True,
                                 curtailed=False, elapsed_seconds=1.0)
    with_judge = arbiter.arbitrate("Is a tool visible?", perception, vlm_result,
                                   {"mode": "judge"})
    as_challenger = arbiter.arbitrate("Is a tool visible?", perception,
                                      vlm_result, {"mode": "challenger"})
    assert with_judge == as_challenger


def test_a_judge_that_raises_falls_back_rather_than_propagating():
    """A missing response scores 0 -- strictly worse than either candidate.
    The judge is an improvement attempt, never a new way to fail."""
    def exploding_judge(question, perception, candidates):
        raise RuntimeError("model died")

    vlm_result = ConfidenceResult(answer="No", confidence=1.0, n_calls_used=2,
                                 all_answers=["No", "No"], agreed=True,
                                 curtailed=False, elapsed_seconds=1.0)
    answer = arbitrate_with_judge(exploding_judge, vlm_result)
    assert answer and answer.strip()


def test_the_judge_is_not_consulted_when_the_candidates_agree():
    """A second VLM pass costs real budget and cannot change an answer both
    candidates already give."""
    calls = []

    def counting_judge(question, perception, candidates):
        calls.append(candidates)
        return "Answer 1"

    # router and VLM will both say the same thing for an unknown-intent
    # question, so should_consult must short-circuit.
    vlm_result = ConfidenceResult(answer="Yes", confidence=1.0, n_calls_used=2,
                                 all_answers=["Yes", "Yes"], agreed=True,
                                 curtailed=False, elapsed_seconds=1.0)
    router_answer = arbiter.router.answer_question("Is a tool visible?", {})
    vlm_result = ConfidenceResult(answer=router_answer, confidence=1.0,
                                 n_calls_used=2, all_answers=[router_answer],
                                 agreed=True, curtailed=False,
                                 elapsed_seconds=1.0)
    arbiter.arbitrate("Is a tool visible?", {}, vlm_result,
                      {"mode": "judge", "judge_fn": counting_judge})
    assert calls == [], "judge ran despite the candidates agreeing"


def arbitrate_with_judge(judge_fn, vlm_result, question="Is a tool visible?"):
    return arbiter.arbitrate(question, {}, vlm_result,
                             {"mode": "judge", "judge_fn": judge_fn})


def test_default_mode_constant_is_the_safe_one_when_the_config_is_gone():
    """DEFAULT_MODE is the mode used when config/arbiter.json is MISSING or
    malformed, so it must stay the conservative one -- `fallback`, the mode
    that scored 0.8558 -- regardless of what the file currently ships.

    UNTIL 2026-08-29 this asserted `DEFAULT_MODE == load_config()["mode"] ==
    "fallback"`, i.e. that the in-code default and the shipped file AGREE.
    They deliberately no longer do: the file arms `per_intent` for v6 while
    the fallback-if-the-file-vanishes path stays put. Requiring them to agree
    would force a degraded-mode default to follow every serving experiment,
    which is exactly backwards -- the whole point of the constant is to be the
    thing that does NOT move.

    Leaderboard evidence for keeping `fallback` as the floor: v5 shipped
    `challenger` and scored 0.7737 against v2's 0.8015, the direction the
    graded sample predicted (router-only 0.9309 vs challenger 0.8525).
    """
    assert arbiter.DEFAULT_MODE == "fallback"
    assert arbiter.DEFAULT_CONFIG["mode"] == "fallback"
    # And the degraded path really does yield it, rather than reading the file.
    assert arbiter.load_config("/nonexistent/arbiter.json")["mode"] == "fallback"


# --------------------------------------------------------------------------
# `per_intent`: the router answers, except on an enumerated set of intents
# the VLM has been MEASURED to beat it on.
#
# The tests that matter here are the two SAFETY properties -- an empty set is
# exactly `fallback`, and a typo cannot arm an intent -- because those are
# what let this mode ship before the measurement that populates it lands.
# --------------------------------------------------------------------------

def test_per_intent_with_no_intents_is_byte_identical_to_fallback():
    """THE PROPERTY THAT MAKES THIS MODE SAFE TO SHIP INERT.

    With `vlm_intents` unset, `per_intent` must agree with `fallback` on
    every question, character for character -- not "behave similarly", not
    "agree on the ones we checked". Asserted over all 11 graded questions
    with a VLM draft that is deliberately WRONG for every one of them, so a
    handler that leaked the VLM's answer through would fail loudly rather
    than coincidentally agree.
    """
    perception = {}
    result = usable_result(answer="A wholly unrelated sentence about nothing")
    for question in SAMPLE_QUESTIONS:
        assert (arbiter.arbitrate(question, perception, vlm_result=result,
                                  config={"mode": arbiter.MODE_PER_INTENT})
                == arbiter.arbitrate(question, perception, vlm_result=result,
                                     config={"mode": arbiter.MODE_FALLBACK})), question


def test_per_intent_empty_set_never_ships_the_vlm_answer():
    """The same property stated the other way round, so a bug that broke
    BOTH handlers identically (making the test above vacuously pass) is
    still caught: the VLM's string must not appear in the output at all."""
    draft = "A wholly unrelated sentence about nothing"
    result = usable_result(answer=draft)
    for question in SAMPLE_QUESTIONS:
        out = arbiter.arbitrate(question, {}, vlm_result=result,
                                config={"mode": arbiter.MODE_PER_INTENT})
        assert draft not in out, question


def test_per_intent_ships_the_vlm_answer_on_an_enumerated_intent():
    """The mode's actual job. `count_open` is used because the router's
    count form and the VLM draft below cannot collide."""
    question = next(q for q in SAMPLE_QUESTIONS
                    if router.classify_question(q) == router.INTENT_TOOL_PRESENCE)
    draft = "Absolutely not, nothing of the kind"
    out = arbiter.arbitrate(question, {}, vlm_result=usable_result(answer=draft),
                            config={"mode": arbiter.MODE_PER_INTENT,
                                    "vlm_intents": [router.INTENT_TOOL_PRESENCE]})
    assert out == router.finalize_answer(draft)


def test_per_intent_leaves_intents_outside_the_set_with_the_router():
    """Arming one intent must not arm its neighbours -- the property that
    makes a regression attributable to a single named intent."""
    draft = "Absolutely not, nothing of the kind"
    result = usable_result(answer=draft)
    config = {"mode": arbiter.MODE_PER_INTENT,
              "vlm_intents": [router.INTENT_TOOL_PRESENCE]}
    for question in SAMPLE_QUESTIONS:
        intent = router.classify_question(question)
        out = arbiter.arbitrate(question, {}, vlm_result=result, config=config)
        if intent != router.INTENT_TOOL_PRESENCE:
            assert draft not in out, (question, intent)


def test_per_intent_drops_an_intent_name_that_does_not_exist():
    """A typo in config/arbiter.json must degrade that intent to the router,
    not arm something else and not raise. This is the check that stops a
    misspelled name from looking armed in the config while doing nothing --
    or worse, from matching a future intent by accident."""
    assert arbiter.resolve_vlm_intents(
        {"vlm_intents": ["tool_presence_polar", "tool_presense_polar"]}
    ) == frozenset({router.INTENT_TOOL_PRESENCE})


@pytest.mark.parametrize("bogus", [None, "tool_presence_polar", 7, {"a": 1}])
def test_per_intent_survives_a_malformed_intent_list(bogus):
    """No config value may be the reason a case fails to produce an answer.
    A bare string is included deliberately: it is iterable, so a naive
    `frozenset(raw)` would silently arm 18 single-CHARACTER 'intents'."""
    assert arbiter.resolve_vlm_intents({"vlm_intents": bogus}) == frozenset()


def test_per_intent_still_consults_the_vlm_on_unknown_intents():
    """The UNKNOWN_* escape hatch survives regardless of `vlm_intents`:
    the router has no form for those, so its answer there is a generic
    string written without reference to the question."""
    draft = "The surgeon is irrigating the surgical field"
    question = "Zzz qqq xxx?"
    assert router.classify_question(question) in (router.INTENT_UNKNOWN_OPEN,
                                                  router.INTENT_UNKNOWN_POLAR)
    out = arbiter.arbitrate(question, {}, vlm_result=usable_result(answer=draft),
                            config={"mode": arbiter.MODE_PER_INTENT})
    assert out == router.finalize_answer(draft)


def test_per_intent_names_only_intents_the_router_can_actually_return():
    """`known_intents` is derived from router.INTENT_* attributes; assert it
    matches what `classify_question` is documented to produce, so the
    typo-dropping above cannot quietly start dropping VALID names."""
    known = arbiter.known_intents()
    assert router.INTENT_TOOL_PRESENCE in known
    assert router.INTENT_UNKNOWN_OPEN in known
    for question in SAMPLE_QUESTIONS:
        assert router.classify_question(question) in known
