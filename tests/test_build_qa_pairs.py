"""Tests for scripts/build_qa_pairs.py -- logbook -> QA pairs.

Mirrors tests/test_build_variant_labels.py's style: real column names and
time format (verified against
/staging/groups/bhaskar_opscribe/surgvu/labels_cat2/SURGVU25_train_labels),
never the plan's original made-up header. Every hazard value below is
copied verbatim from docs/design/notes/2026-08-24-label-vocab-
hazards.md, not invented for the test.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_qa_pairs import (  # noqa: E402
    BALANCED_INTENTS, PARAPHRASES, build_records_for_case,
    distinct_windows, extract_frames_for_windows, frame_dir_for_window,
    frame_index_range, generate_examples_for_window, load_heldout,
    load_variant_labels, main, paraphrase_counts_by_intent, part_number,
    resolve_variant_family, resolve_window_video, sample_corpus,
    sample_intent, scan_logbook_hazards, select_cases, stratified_sample,
    video_path_for_record, water_fill_allocate,
    _stable_bool, _stable_choice, _stable_index,
)
from collections import Counter  # noqa: E402

from surgvu.labels import CaseLabels  # noqa: E402
from surgvu.qa_forms import INTENT_TOOL_PRESENCE  # noqa: E402
from surgvu.sampling import enumerate_windows  # noqa: E402
from surgvu.taxonomy import TASK_CLASSES, TOOL_CLASSES  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "case_test"

_TOOLS_HEADER = ("install_case_part,install_case_time,uninstall_case_part,"
                 "uninstall_case_time,arm,commercial_toolname,"
                 "groundtruth_toolname,case\n")
_TASKS_HEADER = ("index,start_part,start_time,stop_part,stop_time,duration,"
                 "taskname,groundtruth_taskname,matched_description,case\n")


def _write_tools(tmp_path, rows, name="tools.csv"):
    path = tmp_path / name
    path.write_text(_TOOLS_HEADER + "".join(rows), encoding="utf-8")
    return path


def _write_tasks(tmp_path, rows, name="tasks.csv"):
    path = tmp_path / name
    path.write_text(_TASKS_HEADER + "".join(rows), encoding="utf-8")
    return path


def _make_case_dir(root, case_id, tool_rows, task_rows):
    case_dir = root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_tools(case_dir, tool_rows)
    _write_tasks(case_dir, task_rows)
    return case_dir


# --------------------------------------------------------------------------
# R30: split discipline -- normalize_case_id, never string equality
# --------------------------------------------------------------------------


def test_heldout_excluded_via_normalization_despite_spelling_mismatch(tmp_path):
    """The CASE DIRECTORY is spelled `case5` (as the public cat2_sample
    directories are: `case122`, no underscore); `load_heldout` has already
    normalised the splits-file spelling to `case_005`. A raw string
    membership test between the two (`"case5" in {"case_005"}`) is False,
    so a naive implementation excludes nothing. select_cases must still
    exclude it because it normalises the DIRECTORY name too before
    comparing.

    Breaks if: select_cases compares the raw directory name (`c in
    heldout_norm`) instead of `normalize_case_id(c) in heldout_norm`.
    """
    root = tmp_path / "labels"
    _make_case_dir(root, "case5", [], [])
    _make_case_dir(root, "case_006", [], [])

    # Prove the trap: naive string equality really does miss it.
    assert "case5" not in {"case_005"}

    heldout_norm = {"case_005"}  # what load_heldout produces from "case5" or "case_005"
    eligible, excluded = select_cases(root, heldout_norm)
    assert excluded == ["case5"]
    assert eligible == ["case_006"]


def test_zero_exclusion_raises(tmp_path):
    """If nothing in the corpus matches the heldout set, that IS the bug
    (ruling R30) -- refuse to proceed rather than silently train on
    everything.

    Breaks if: the "if not excluded: raise" guard in select_cases is
    removed or downgraded to a warning/print.
    """
    root = tmp_path / "labels"
    _make_case_dir(root, "case_001", [], [])
    with pytest.raises(RuntimeError, match="ZERO"):
        select_cases(root, heldout_norm={"case_999"})


def test_load_heldout_normalizes_mixed_spellings(tmp_path):
    splits = tmp_path / "splits_v2.json"
    splits.write_text(json.dumps({"heldout": ["case122", "case_123"]}), encoding="utf-8")
    heldout = load_heldout(splits)
    assert heldout == {"case_122", "case_123"}


def test_load_heldout_missing_key_raises(tmp_path):
    splits = tmp_path / "splits_v2.json"
    splits.write_text(json.dumps({"train": [], "val": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_heldout(splits)


# --------------------------------------------------------------------------
# label-vocab hazards -- tallied, never silently blended together
# --------------------------------------------------------------------------


def test_nan_camera_and_empty_string_are_dropped_and_tallied_separately(tmp_path):
    """`nan(camera in)` (the endoscope) and the empty string are two
    DIFFERENT hazards in the real corpus (1277 and 144 rows respectively)
    and must be counted under distinct reasons, not merged into one bucket.

    Breaks if: scan_logbook_hazards buckets both under one generic
    "dropped" counter instead of "nan_camera" / "empty".
    """
    tools_csv = _write_tools(tmp_path, [
        "1.0,00:00:00.000000,1.0,00:01:00.000000,USM1,30 Endoscope,"
        "nan(camera in),case_x\n",
        "1.0,00:00:00.000000,1.0,00:01:00.000000,USM2,,,"
        "case_x\n",
        "1.0,00:00:00.000000,1.0,00:01:00.000000,USM3,Cadiere Forceps,"
        "cadiere forceps,case_x\n",
    ])
    tasks_csv = _write_tasks(tmp_path, [])
    tool_h, _ = scan_logbook_hazards(tools_csv, tasks_csv)
    assert tool_h["nan_camera"] == 1
    assert tool_h["empty"] == 1
    assert tool_h["kept"] == 1


def test_malformed_quoted_row_is_dropped_as_out_of_scope_or_malformed(tmp_path):
    tools_csv = _write_tools(tmp_path, [
        '1.0,00:00:00.000000,1.0,00:01:00.000000,USM1,Single Site,'
        ' Single Site",case_x\n',
    ])
    tasks_csv = _write_tasks(tmp_path, [])
    tool_h, _ = scan_logbook_hazards(tools_csv, tasks_csv)
    assert tool_h["out_of_scope_or_malformed"] == 1
    assert tool_h["kept"] == 0


def test_clip_applier_trailing_space_is_stripped_and_kept(tmp_path):
    """`clip applier ` (trailing space) is a real, frequent (882-row) value.
    It must be counted as KEPT (normalize_tool strips before whitelisting),
    tallied specifically as normalised-not-verbatim so a reader can see it
    was not simply an exact match.

    Breaks if: the hazard scan (or normalize_tool underneath it) compares
    the raw string without stripping, so "clip applier " fails to match
    "clip applier" and the row is wrongly dropped.
    """
    tools_csv = _write_tools(tmp_path, [
        "1.0,00:00:00.000000,1.0,00:01:00.000000,USM1,Large Clip Applier,"
        "clip applier ,case_x\n",
    ])
    tasks_csv = _write_tasks(tmp_path, [])
    tool_h, _ = scan_logbook_hazards(tools_csv, tasks_csv)
    assert tool_h["kept"] == 1
    assert tool_h["kept_after_normalisation"] == 1


def test_task_names_case_collapse_suturing_and_suturing(tmp_path):
    """`Suturing` (775 rows) and `suturing` (609 rows) are the SAME class.
    Both must be counted as kept, and the differently-cased one tallied
    under case_normalised.

    Breaks if: normalize_task (or this scan) does not lower-case before
    whitelisting, so "Suturing" and "suturing" are treated as two classes.
    """
    tools_csv = _write_tools(tmp_path, [])
    tasks_csv = _write_tasks(tmp_path, [
        "0,1.0,60.0,1.0,120.0,60.0,Skills Drills,Suturing,,case_x\n",
        "1,1.0,200.0,1.0,260.0,60.0,Skills Drills,suturing,,case_x\n",
    ])
    _, task_h = scan_logbook_hazards(tools_csv, tasks_csv)
    assert task_h["kept"] == 2
    assert task_h["case_normalised"] == 1


def test_dissection_task_name_is_dropped_not_in_taxonomy(tmp_path):
    tools_csv = _write_tools(tmp_path, [])
    tasks_csv = _write_tasks(tmp_path, [
        "0,1.0,60.0,1.0,120.0,60.0,Whatever,dissection,,case_x\n",
    ])
    _, task_h = scan_logbook_hazards(tools_csv, tasks_csv)
    assert task_h["unrecognised_or_malformed"] == 1


# --------------------------------------------------------------------------
# every record carries the part (R28) and matches the required schema
# --------------------------------------------------------------------------


def test_every_record_carries_a_part():
    labels = CaseLabels.from_dir(FIXTURE)
    records = build_records_for_case("case_test", labels, [], Counter(), Counter())
    assert records
    for r in records:
        assert r["part"] in ("1.0", "2.0")


def test_record_schema_has_exactly_the_required_keys():
    labels = CaseLabels.from_dir(FIXTURE)
    records = build_records_for_case("case_test", labels, [], Counter(), Counter())
    expected = {"case", "part", "t_start", "t_stop", "question", "answer",
                "intent", "provenance"}
    for r in records:
        assert set(r) == expected


def test_records_are_json_safe_no_nan():
    labels = CaseLabels.from_dir(FIXTURE)
    records = build_records_for_case("case_test", labels, [], Counter(), Counter())
    for r in records:
        json.dumps(r, allow_nan=False)  # raises ValueError on NaN/inf


# --------------------------------------------------------------------------
# answers always come from qa_forms's whitelist-guarded answer_fn
# --------------------------------------------------------------------------


def test_tool_absence_never_names_a_present_tool():
    """The absent_class slot must never name a tool that is actually
    installed in the window -- otherwise the "not in use" answer is a lie
    the model would be trained to repeat.
    """
    labels = CaseLabels.from_dir(FIXTURE)
    windows = enumerate_windows("case_test", labels)
    for w in windows:
        for shape, template, key, slots in generate_examples_for_window(w, [], Counter()):
            if shape == "tool_absence":
                assert slots["absent_class"] not in w.tools


def test_count_answer_matches_number_of_installed_tools():
    labels = CaseLabels.from_dir(FIXTURE)
    windows = enumerate_windows("case_test", labels)
    for w in windows:
        for shape, template, key, slots in generate_examples_for_window(w, [], Counter()):
            if shape == "count":
                assert slots["count"] == len(w.tools)


# --------------------------------------------------------------------------
# paraphrasing -- more than one phrasing per intent, grounded register
# --------------------------------------------------------------------------


def test_every_shape_has_more_than_one_paraphrase():
    for shape, forms in PARAPHRASES.items():
        assert len(forms) > 1, shape


def test_paraphrase_counts_by_intent_are_all_above_one():
    counts = paraphrase_counts_by_intent()
    assert counts
    for intent, n in counts.items():
        assert n > 1, intent


def test_paraphrase_selection_is_deterministic():
    from build_qa_pairs import _render_shape, TPL_TASK
    q1, a1, i1 = _render_shape("task", TPL_TASK, "case_x|1.0|60.000|task", {"task_class": "suturing"})
    q2, a2, i2 = _render_shape("task", TPL_TASK, "case_x|1.0|60.000|task", {"task_class": "suturing"})
    assert (q1, a1, i1) == (q2, a2, i2)


def test_stable_choice_is_key_sensitive_across_many_keys():
    """Not every key should land on the same paraphrase -- otherwise
    "deterministic" degenerated into "constant"."""
    options = list(range(4))
    picks = {_stable_choice(options, "key-%d" % i) for i in range(50)}
    assert len(picks) > 1


# --------------------------------------------------------------------------
# variant_presence: only generated when a family is actually resolvable
# --------------------------------------------------------------------------


def test_variant_presence_generated_when_family_resolvable():
    intervals = [{"part": "1.0", "start": 0.0, "stop": 10000.0, "family": "large", "arm": "USM1"}]
    w = next(w for w in enumerate_windows("case_test", CaseLabels.from_dir(FIXTURE))
             if w.part == "1.0" and "needle driver" in w.tools)
    shapes = generate_examples_for_window(w, intervals, Counter())
    variant_shapes = [s for s in shapes if s[0] == "variant_presence"]
    assert len(variant_shapes) == 1
    assert variant_shapes[0][3]["installed_family"] == "large"


def test_variant_presence_skipped_and_tallied_when_no_variant_data():
    w = next(w for w in enumerate_windows("case_test", CaseLabels.from_dir(FIXTURE))
             if w.part == "1.0" and "needle driver" in w.tools)
    drops = Counter()
    shapes = generate_examples_for_window(w, [], drops)
    assert not [s for s in shapes if s[0] == "variant_presence"]
    assert drops["variant_family_unresolved"] == 1


def test_resolve_variant_family_none_when_two_families_overlap():
    intervals = [
        {"part": "1.0", "start": 0.0, "stop": 100.0, "family": "large", "arm": "USM1"},
        {"part": "1.0", "start": 0.0, "stop": 100.0, "family": "mega", "arm": "USM3"},
    ]
    assert resolve_variant_family(intervals, "1.0", 10.0, 40.0) is None


def test_resolve_variant_family_respects_part():
    intervals = [{"part": "1.0", "start": 0.0, "stop": 100.0, "family": "large", "arm": "USM1"}]
    assert resolve_variant_family(intervals, "2.0", 10.0, 40.0) is None
    assert resolve_variant_family(intervals, "1.0", 10.0, 40.0) == "large"


def test_load_variant_labels_missing_file_returns_empty(tmp_path):
    assert load_variant_labels(tmp_path / "does_not_exist.json") == {}


# --------------------------------------------------------------------------
# end-to-end over the fixture case
# --------------------------------------------------------------------------


def test_generate_examples_cover_the_always_applicable_intents():
    labels = CaseLabels.from_dir(FIXTURE)
    records = build_records_for_case("case_test", labels, [], Counter(), Counter())
    intents = {r["intent"] for r in records}
    from surgvu.qa_forms import (
        INTENT_COUNT, INTENT_CUTTING, INTENT_ORGAN, INTENT_SUTURE,
        INTENT_TASK, INTENT_TASK_CONFIRMATION, INTENT_TOOL_ABSENCE,
        INTENT_TOOL_PRESENCE,
    )
    for expected in (INTENT_TOOL_PRESENCE, INTENT_TASK, INTENT_ORGAN,
                     INTENT_COUNT, INTENT_CUTTING, INTENT_SUTURE,
                     INTENT_TOOL_ABSENCE, INTENT_TASK_CONFIRMATION):
        assert expected in intents, expected


def test_main_writes_jsonl_and_excludes_heldout(tmp_path, capsys):
    root = tmp_path / "labels"
    # Reuse the real fixture's rows for two synthetic cases, one heldout.
    tool_rows = (FIXTURE / "tools.csv").read_text(encoding="utf-8").splitlines(True)[1:]
    task_rows = (FIXTURE / "tasks.csv").read_text(encoding="utf-8").splitlines(True)[1:]
    _make_case_dir(root, "case_050", tool_rows, task_rows)
    _make_case_dir(root, "case_051", tool_rows, task_rows)

    splits = tmp_path / "splits_v2.json"
    splits.write_text(json.dumps({"heldout": ["case51"]}), encoding="utf-8")

    out = tmp_path / "qa_pairs.jsonl"
    rc = main(["--labels-root", str(root), "--splits", str(splits),
              "--variant-labels", str(tmp_path / "no_such_variant.json"),
              "--out", str(out)])
    assert rc == 0

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines
    records = [json.loads(line) for line in lines]
    assert {r["case"] for r in records} == {"case_050"}

    captured = capsys.readouterr()
    assert "excluded (heldout): 1" in captured.out
    assert "case_051" in captured.out


# ============================================================================
# Task 3: sampling policy -- water_fill_allocate, the shared primitive behind
# both "stratify across cases" and "correct the tool_presence_polar skew".
# See docs/design/plans/2026-08-25-v5-plan3-vlm-training.md, Task 3.
# ============================================================================

import random  # noqa: E402


def test_water_fill_splits_evenly_when_capacities_ample():
    """Breaks if: `share, extra = divmod(remaining, n)` is replaced by
    proportional allocation (e.g. `cap / total * target`) -- that is exactly
    the skew-preserving allocation this function exists to avoid.
    """
    alloc = water_fill_allocate({"a": 100, "b": 100, "c": 100}, 90, rng=random.Random(0))
    assert alloc == {"a": 30, "b": 30, "c": 30}


def test_water_fill_saturates_small_capacity_and_redistributes():
    """One case has far fewer records than an equal share would ask for; it
    must be fully used (not truncated further) and the remainder must go to
    the other keys, not disappear.

    Breaks if: a key's own capacity is not honoured (e.g. `alloc[k] = share`
    unconditionally instead of `min(share, capacity)`), which would either
    request more records than a case has or silently drop the shortfall
    instead of giving it to the cases that have room.
    """
    alloc = water_fill_allocate({"small": 2, "big1": 100, "big2": 100}, 30,
                                rng=random.Random(0))
    assert alloc["small"] == 2
    assert alloc["big1"] + alloc["big2"] == 28
    assert abs(alloc["big1"] - alloc["big2"]) <= 1


def test_water_fill_never_exceeds_any_capacity():
    """Breaks if: the saturation branch is removed, so a key with less
    capacity than its share is still handed `share` records -- more than it
    has.
    """
    rng = random.Random(1)
    caps = {"c%d" % i: rng.randint(0, 50) for i in range(20)}
    alloc = water_fill_allocate(caps, 300, rng=random.Random(2))
    for k, v in alloc.items():
        assert v <= caps[k]


def test_water_fill_sums_to_min_of_target_and_total_capacity():
    """Breaks if: the loop's termination condition drops the final
    remainder instead of assigning it (e.g. `remaining = 0` before the
    single-unit-per-key branch runs).
    """
    caps = {"a": 5, "b": 5, "c": 5}
    alloc_under = water_fill_allocate(caps, 9, rng=random.Random(0))
    assert sum(alloc_under.values()) == 9
    alloc_over = water_fill_allocate(caps, 100, rng=random.Random(0))
    assert sum(alloc_over.values()) == 15  # capped by total capacity


def test_water_fill_remainder_spreads_across_keys_not_onto_one():
    """Breaks if: the `share == 0` remainder branch hands every leftover
    unit to the same key (e.g. `keys[0]` instead of a shuffled subset),
    which would silently re-introduce a per-case skew for the very corpus
    sizes where the remainder matters most (many cases, small budget).
    """
    caps = {"c%d" % i: 1000 for i in range(10)}
    alloc = water_fill_allocate(caps, 4, rng=random.Random(0))
    assert sum(alloc.values()) == 4
    assert sum(1 for v in alloc.values() if v > 0) == 4  # four DIFFERENT keys


def test_water_fill_deterministic_given_seeded_rng():
    caps = {"c%d" % i: 7 for i in range(9)}
    a1 = water_fill_allocate(caps, 40, rng=random.Random(42))
    a2 = water_fill_allocate(caps, 40, rng=random.Random(42))
    assert a1 == a2


def test_water_fill_rejects_negative_target():
    with pytest.raises(ValueError):
        water_fill_allocate({"a": 5}, -1, rng=random.Random(0))


# ----------------------------------------------------------------------
# stratified_sample -- water_fill_allocate applied to real record dicts
# ----------------------------------------------------------------------


def _fake_records(case_counts, answer="Yes"):
    """{'case': c, 'answer': answer, ...} records, `n` per case in `case_counts`."""
    out = []
    for case, n in case_counts.items():
        for i in range(n):
            out.append({"case": case, "answer": answer, "i": i})
    return out


def test_stratified_sample_caps_a_dominant_case():
    """One case supplies 1000 of the records; stratified_sample must not let
    it dominate the output.

    Breaks if: `stratified_sample` samples uniformly at random from the
    pooled list instead of allocating per-group first -- pooled random
    sampling from a 1000-vs-5-vs-5 pool would draw almost entirely from the
    dominant case.
    """
    records = _fake_records({"case_a": 1000, "case_b": 5, "case_c": 5})
    rng = random.Random(0)
    sampled = stratified_sample(records, 15, rng, key_fn=lambda r: r["case"])
    per_case = Counter(r["case"] for r in sampled)
    assert per_case["case_b"] == 5
    assert per_case["case_c"] == 5
    assert per_case["case_a"] == 5


def test_stratified_sample_never_exceeds_a_cases_available_records():
    records = _fake_records({"case_a": 3, "case_b": 300})
    rng = random.Random(0)
    sampled = stratified_sample(records, 100, rng, key_fn=lambda r: r["case"])
    per_case = Counter(r["case"] for r in sampled)
    assert per_case["case_a"] == 3
    assert per_case["case_b"] == 97


def test_stratified_sample_deterministic_given_seed():
    records = _fake_records({"case_a": 50, "case_b": 50, "case_c": 50})
    s1 = stratified_sample(records, 30, random.Random(7), key_fn=lambda r: r["case"])
    s2 = stratified_sample(records, 30, random.Random(7), key_fn=lambda r: r["case"])
    assert s1 == s2


def test_stratified_sample_draws_without_replacement():
    """Breaks if: `rng.choices` (with replacement) is used instead of
    `rng.sample`, which could return duplicate records within one case.
    """
    records = _fake_records({"case_a": 10})
    sampled = stratified_sample(records, 10, random.Random(0), key_fn=lambda r: r["case"])
    ids = [r["i"] for r in sampled]
    assert sorted(ids) == list(range(10))


# ----------------------------------------------------------------------
# sample_intent -- the tool_presence_polar Yes/No balance fix, nested
# inside the same per-case stratification.
# ----------------------------------------------------------------------


def test_sample_intent_balances_answer_when_requested():
    """71/29 in, parity out -- the corpus's actual measured skew (66,144
    records, 71.3% Yes) is exactly the shape this test constructs.

    Breaks if: `balance_by_answer=True` is ignored (falls through to the
    plain per-case path), which would reproduce the input's 70/30 skew in
    the output instead of correcting it.
    """
    yes = _fake_records({"case_%02d" % i: 100 for i in range(10)}, answer="Yes")
    no = _fake_records({"case_%02d" % i: 40 for i in range(10)}, answer="No")
    pool = yes + no
    sampled = sample_intent(pool, 200, random.Random(0), balance_by_answer=True)
    per_answer = Counter(r["answer"] for r in sampled)
    assert per_answer["Yes"] == 100
    assert per_answer["No"] == 100


def test_sample_intent_balance_still_stratifies_by_case_within_each_answer():
    """The answer-balance fix must not undo the case-stratification: within
    the Yes bucket alone, one dominant case must not swamp the others.

    Breaks if: the balance branch samples each answer bucket with
    `rng.sample(bucket, want)` directly instead of routing through
    `stratified_sample` again -- that would fix the Yes/No ratio while
    silently re-introducing a per-case skew inside the Yes bucket.
    """
    yes = _fake_records({"dominant": 1000, "rare_a": 5, "rare_b": 5}, answer="Yes")
    no = _fake_records({"dominant": 5, "rare_a": 5, "rare_b": 5}, answer="No")
    sampled = sample_intent(yes + no, 30, random.Random(0), balance_by_answer=True)
    yes_cases = Counter(r["case"] for r in sampled if r["answer"] == "Yes")
    assert yes_cases["rare_a"] == 5
    assert yes_cases["rare_b"] == 5
    assert yes_cases["dominant"] == 5  # not swamped by its 1000 available


def test_sample_intent_without_balance_preserves_natural_answer_ratio():
    """Every other intent (not opted into BALANCED_INTENTS) must sample
    without touching the answer distribution at all.

    Breaks if: `balance_by_answer=False` still routes through the
    answer-splitting branch instead of a single case-stratified pass.
    """
    records = (_fake_records({"c1": 90, "c2": 90}, answer="Yes")
              + _fake_records({"c1": 10, "c2": 10}, answer="No"))
    sampled = sample_intent(records, 40, random.Random(0), balance_by_answer=False)
    per_answer = Counter(r["answer"] for r in sampled)
    # natural ratio is 90:10 -- balancing would force it toward 20:20
    assert per_answer["Yes"] > per_answer["No"]


# ----------------------------------------------------------------------
# sample_corpus -- the top-level entry point over a full qa_pairs.jsonl
# ----------------------------------------------------------------------


def _corpus_fixture():
    records = []
    for case_i in range(6):
        case = "case_%03d" % case_i
        for i in range(50):
            records.append({
                "case": case, "part": "1.0", "t_start": float(i),
                "t_stop": float(i) + 30.0, "question": "q%d" % i,
                "answer": "Yes" if i % 10 < 7 else "No",  # ~70/30, like real
                "intent": INTENT_TOOL_PRESENCE, "provenance": {},
            })
            records.append({
                "case": case, "part": "1.0", "t_start": float(i),
                "t_stop": float(i) + 30.0, "question": "t%d" % i,
                "answer": "suturing", "intent": "task_open", "provenance": {},
            })
    return records


def test_sample_corpus_caps_each_intent_at_max_per_intent():
    records = _corpus_fixture()
    sampled, report = sample_corpus(records, max_per_intent=60, seed=0)
    per_intent = Counter(r["intent"] for r in sampled)
    assert per_intent[INTENT_TOOL_PRESENCE] == 60
    assert per_intent["task_open"] == 60


def test_sample_corpus_balances_only_the_intent_in_BALANCED_INTENTS():
    """Breaks if: BALANCED_INTENTS is empty, or if sample_corpus applies the
    balance fix to every intent rather than only the ones listed there.
    """
    assert INTENT_TOOL_PRESENCE in BALANCED_INTENTS
    records = _corpus_fixture()
    sampled, report = sample_corpus(records, max_per_intent=60, seed=0)
    tool_presence = [r for r in sampled if r["intent"] == INTENT_TOOL_PRESENCE]
    task = [r for r in sampled if r["intent"] == "task_open"]
    tp_answers = Counter(r["answer"] for r in tool_presence)
    assert tp_answers["Yes"] == tp_answers["No"]  # corrected toward parity
    task_answers = Counter(r["answer"] for r in task)
    assert set(task_answers) == {"suturing"}  # untouched, single answer anyway


def test_sample_corpus_report_covers_per_intent_per_case_per_answer():
    """Task 3 requires reporting exactly what was sampled: per-intent,
    per-case, per-answer. Breaks if: `report` omits any of the three, e.g.
    if `per_case` or `per_answer` is dropped from an intent's entry.
    """
    records = _corpus_fixture()
    _, report = sample_corpus(records, max_per_intent=30, seed=0)
    for intent in (INTENT_TOOL_PRESENCE, "task_open"):
        entry = report["intents"][intent]
        assert "per_case" in entry and "per_answer" in entry
        assert sum(entry["per_case"].values()) == entry["sampled"]
        assert sum(entry["per_answer"].values()) == entry["sampled"]


def test_sample_corpus_deterministic_given_seed():
    records = _corpus_fixture()
    s1, _ = sample_corpus(records, max_per_intent=40, seed=3)
    s2, _ = sample_corpus(records, max_per_intent=40, seed=3)
    assert s1 == s2


def test_sample_corpus_does_not_exceed_available_pool():
    records = _corpus_fixture()[:5]  # far fewer than max_per_intent
    sampled, report = sample_corpus(records, max_per_intent=1000, seed=0)
    assert len(sampled) == len(records)


# ============================================================================
# Task 3: frame extraction -- window dedup, path resolution, drop tallying.
# All torch-free: video decode itself is injected via `decode_fn`, exercised
# with a fake rather than surgvu.perceive.decode_clip_multiscale, which
# imports torch at module scope and is not installed on this machine.
# ============================================================================


def test_part_number_reads_the_records_own_part_field():
    """Breaks if: part_number stops going through surgvu.labels.normalize_part
    and instead does e.g. `int(part)` directly, which raises on '1.0'.
    """
    assert part_number("1.0") == 1
    assert part_number("2.0") == 2
    assert part_number("1") == 1
    assert part_number(2) == 2


def test_video_path_for_record_matches_the_real_naming_convention():
    """`<case>_video_part_<NNN>.mp4`, NNN zero-padded to 3 digits -- the same
    convention scripts/dump_motion_v2.py's find_case_videos parses in
    reverse. Breaks if: the padding width changes (e.g. `%d` instead of
    `%03d`), which would build a path that does not exist on staging.
    """
    path = video_path_for_record("/video/root", "case_005", "2.0")
    assert str(path) == "/video/root/case_005/case_005_video_part_002.mp4"


def test_frame_dir_for_window_has_no_float_repr_in_the_path():
    """Millisecond integers, not float reprs -- '1773.300869' rendered
    straight into a path is a real risk of two different-looking strings
    for the same float across platforms/precisions.

    Breaks if: the directory name is built with str(t_start) instead of
    int(round(t_start * 1000)).
    """
    d = frame_dir_for_window("/frames", "case_000", "1.0", 1773.300869, 1803.300869)
    assert "." not in d.name
    assert d.name == "1773301_1803301"


def test_frame_dir_for_window_distinguishes_different_windows():
    d1 = frame_dir_for_window("/frames", "case_000", "1.0", 0.0, 30.0)
    d2 = frame_dir_for_window("/frames", "case_000", "1.0", 30.0, 60.0)
    assert d1 != d2


def test_distinct_windows_groups_records_sharing_one_clip():
    """Multiple QA records from the same window (tool presence, task,
    count, ...) must decode from ONE physical clip, not one each.

    Breaks if: the grouping key omits `part` (or `t_start`/`t_stop`),
    merging windows that are actually different clips, or is per-record
    (no grouping at all), which reintroduces the redundant-decode cost this
    function exists to remove.
    """
    records = [
        {"case": "case_000", "part": "1.0", "t_start": 0.0, "t_stop": 30.0, "intent": "a"},
        {"case": "case_000", "part": "1.0", "t_start": 0.0, "t_stop": 30.0, "intent": "b"},
        {"case": "case_000", "part": "1.0", "t_start": 30.0, "t_stop": 60.0, "intent": "c"},
        {"case": "case_000", "part": "2.0", "t_start": 0.0, "t_stop": 30.0, "intent": "d"},
    ]
    windows = distinct_windows(records)
    assert len(windows) == 3
    key = ("case_000", "1.0", 0.0, 30.0)
    assert sorted(windows[key]) == [0, 1]


def test_frame_index_range_converts_seconds_to_frame_indices():
    rng_span = frame_index_range(1.0, 2.0, fps=60.0, total=1000)
    assert rng_span == (60, 119)


def test_frame_index_range_reports_none_when_out_of_bounds():
    """Breaks if: an out-of-range request is silently clamped into
    `(first, total - 1)` instead of returning None -- a clamped window
    covers different (shorter, shifted) content than the question was
    written against, and that must be a reported drop, not a quiet
    substitution.
    """
    assert frame_index_range(990.0, 1010.0, fps=60.0, total=1000) is None


def test_resolve_window_video_reports_missing_video_file(tmp_path):
    cache = {}
    info, reason = resolve_window_video(str(tmp_path), "case_nope", "1.0", cache)
    assert info is None
    assert reason == "video_missing"


def test_resolve_window_video_caches_by_case_and_part(tmp_path, monkeypatch):
    """Breaks if: the cache key omits `part`, which would make a case's
    second video part reuse the first part's (wrong) total-frame-count and
    fps, or if resolve_window_video reopens the file every call instead of
    reading the cache -- silently expensive across the thousands of QA
    records that share one video part.
    """
    case_dir = tmp_path / "case_000"
    case_dir.mkdir()
    video = case_dir / "case_000_video_part_001.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    for _ in range(20):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()

    calls = []
    real_capture = cv2.VideoCapture

    def counting_capture(path):
        calls.append(path)
        return real_capture(path)

    monkeypatch.setattr(cv2, "VideoCapture", counting_capture)
    cache = {}
    info1, reason1 = resolve_window_video(str(tmp_path), "case_000", "1.0", cache)
    info2, reason2 = resolve_window_video(str(tmp_path), "case_000", "1.0", cache)
    assert reason1 is None and reason2 is None
    assert info1.total == 20
    assert len(calls) == 1  # second call served from cache, not reopened


# ----------------------------------------------------------------------
# extract_frames_for_windows -- the end-to-end driver, decode_fn faked
# ----------------------------------------------------------------------


def _tiny_records():
    return [
        {"case": "case_000", "part": "1.0", "t_start": 0.0, "t_stop": 1.0,
         "intent": "a", "answer": "Yes", "question": "q1"},
        {"case": "case_000", "part": "1.0", "t_start": 0.0, "t_stop": 1.0,
         "intent": "b", "answer": "No", "question": "q2"},
        {"case": "case_000", "part": "1.0", "t_start": 5.0, "t_stop": 6.0,
         "intent": "a", "answer": "Yes", "question": "q3"},
    ]


def _fake_video_setup(tmp_path, total=600, fps=60.0):
    case_dir = tmp_path / "videos" / "case_000"
    case_dir.mkdir(parents=True)
    (case_dir / "case_000_video_part_001.mp4").write_bytes(b"not a real video")
    return str(tmp_path / "videos")


def test_extract_frames_shares_one_decode_across_records_of_one_window(tmp_path, monkeypatch):
    """Two of the three fixture records share window (case_000, 1.0, 0.0,
    1.0); decode_fn must be called ONCE for that window and both records'
    manifest entries must carry the SAME frame_paths.

    Breaks if: extract_frames_for_windows iterates `records` directly
    instead of `distinct_windows(records)`, calling decode_fn once per
    record instead of once per distinct clip.
    """
    video_root = _fake_video_setup(tmp_path)
    decode_calls = []

    def fake_decode(video_path, first, last, n_frames, size):
        decode_calls.append((str(video_path), first, last))
        return [np.zeros((size, size, 3), dtype=np.uint8) for _ in range(n_frames)]

    def fake_resolve(video_root_arg, case, part, cache):
        from build_qa_pairs import VideoInfo
        return VideoInfo(path=Path(video_root_arg) / case / (case + "_video_part_001.mp4"),
                         total=600, fps=60.0), None

    monkeypatch.setattr("build_qa_pairs.resolve_window_video", fake_resolve)

    manifest, drops = extract_frames_for_windows(
        _tiny_records(), video_root, str(tmp_path / "qa_frames"),
        n_frames=2, size=64, decode_fn=fake_decode)

    assert len(decode_calls) == 2  # 2 distinct windows, not 3 records
    assert len(manifest) == 3
    shared = [m["frame_paths"] for m in manifest if m["question"] in ("q1", "q2")]
    assert shared[0] == shared[1]
    assert not drops


def test_extract_frames_tallies_drops_by_record_count_not_window_count():
    """A dropped window with 2 QA records must add 2 to the drop tally, not
    1 -- otherwise "how many records lost their frames" (Task 3's explicit
    reporting requirement) undercounts every window with more than one
    question.

    Breaks if: `drops[reason] += 1` replaces `drops[reason] += len(idxs)`.
    """
    def missing_decode(video_path, first, last, n_frames, size):
        raise AssertionError("should never be called: video is missing")

    manifest, drops = extract_frames_for_windows(
        _tiny_records(), "/no/such/video/root", "/tmp/does-not-matter-qa-frames",
        n_frames=2, size=64, decode_fn=missing_decode)

    assert not manifest
    assert drops["video_missing"] == 3  # all 3 records, across both windows


def test_extract_frames_dry_run_never_calls_decode_fn(tmp_path, monkeypatch):
    """--dry-run must exercise every torch-free step (window dedup, path
    resolution, drop tallying) for real, without ever touching decode_fn --
    that is what makes it runnable on a machine with no torch installed.

    Breaks if: dry_run is ignored and decode_fn is called anyway.
    """
    video_root = _fake_video_setup(tmp_path)

    def fake_resolve(video_root_arg, case, part, cache):
        from build_qa_pairs import VideoInfo
        return VideoInfo(path=Path(video_root_arg) / case / (case + "_video_part_001.mp4"),
                         total=600, fps=60.0), None

    monkeypatch.setattr("build_qa_pairs.resolve_window_video", fake_resolve)

    def exploding_decode(*a, **k):
        raise AssertionError("dry_run must not decode")

    manifest, drops = extract_frames_for_windows(
        _tiny_records(), video_root, str(tmp_path / "qa_frames"),
        n_frames=2, size=64, decode_fn=exploding_decode, dry_run=True)
    assert len(manifest) == 3
    assert not drops
    assert all(len(m["frame_paths"]) == 2 for m in manifest)


def test_resolve_window_video_reports_unreadable_video(tmp_path):
    """A file that exists but that cv2 cannot decode (garbage bytes, not a
    real container) must be its own distinct reason from `video_missing` --
    "the path resolved but the video is broken" is a different failure than
    "the file was never there", and conflating them would hide which one to
    go fix.
    """
    case_dir = tmp_path / "case_000"
    case_dir.mkdir()
    (case_dir / "case_000_video_part_001.mp4").write_bytes(b"not a real video file")
    cache = {}
    info, reason = resolve_window_video(str(tmp_path), "case_000", "1.0", cache)
    assert info is None
    assert reason == "video_unreadable"


def test_extract_frames_tallies_decode_failed_when_decode_fn_raises(tmp_path, monkeypatch):
    """Breaks if: the `except ValueError` around decode_fn is removed (an
    unhandled exception would crash the whole extraction job instead of
    costing just the records sharing that one window) or if the tally key
    stops being distinct from "decode_short" / "video_missing".
    """
    video_root = _fake_video_setup(tmp_path)

    def fake_resolve(video_root_arg, case, part, cache):
        from build_qa_pairs import VideoInfo
        return VideoInfo(path=Path(video_root_arg) / case / (case + "_video_part_001.mp4"),
                         total=600, fps=60.0), None

    monkeypatch.setattr("build_qa_pairs.resolve_window_video", fake_resolve)

    def raising_decode(video_path, first, last, n_frames, size):
        raise ValueError("simulated: all sampled indices failed to read")

    manifest, drops = extract_frames_for_windows(
        _tiny_records(), video_root, str(tmp_path / "qa_frames"),
        n_frames=2, size=64, decode_fn=raising_decode)
    assert not manifest
    assert drops["decode_failed"] == 3


def test_extract_frames_tallies_decode_short_when_fewer_frames_come_back(tmp_path, monkeypatch):
    """Breaks if: a short decode is accepted anyway (e.g. writing whatever
    frames DID come back under fewer filenames than frame_paths promised),
    which would silently hand the manifest a record whose frame_paths do
    not all exist on disk.
    """
    video_root = _fake_video_setup(tmp_path)

    def fake_resolve(video_root_arg, case, part, cache):
        from build_qa_pairs import VideoInfo
        return VideoInfo(path=Path(video_root_arg) / case / (case + "_video_part_001.mp4"),
                         total=600, fps=60.0), None

    monkeypatch.setattr("build_qa_pairs.resolve_window_video", fake_resolve)

    def short_decode(video_path, first, last, n_frames, size):
        return [np.zeros((size, size, 3), dtype=np.uint8) for _ in range(n_frames - 1)]

    manifest, drops = extract_frames_for_windows(
        _tiny_records(), video_root, str(tmp_path / "qa_frames"),
        n_frames=2, size=64, decode_fn=short_decode)
    assert not manifest
    assert drops["decode_short"] == 3


# --------------------------------------------------------------------------
# threaded extraction: `workers > 1` must change ONLY the wall clock
# --------------------------------------------------------------------------

def _threading_setup(tmp_path, monkeypatch):
    video_root = _fake_video_setup(tmp_path)

    def fake_resolve(video_root_arg, case, part, cache):
        from build_qa_pairs import VideoInfo
        return VideoInfo(path=Path(video_root_arg) / case / (case + "_video_part_001.mp4"),
                         total=600, fps=60.0), None

    monkeypatch.setattr("build_qa_pairs.resolve_window_video", fake_resolve)
    return video_root


def _many_records(n_windows=24):
    """Enough distinct windows that a thread pool genuinely interleaves them."""
    records = []
    for w in range(n_windows):
        for q in range(2):
            records.append({
                "case": "case_000", "part": "1.0",
                "t_start": float(w), "t_stop": float(w) + 1.0,
                "question": "q%d_%d" % (w, q), "answer": "a",
                "intent": "tool_presence_polar",
            })
    return records


def test_threaded_and_serial_manifests_are_identical(tmp_path, monkeypatch):
    """THE PROPERTY THE WHOLE CHANGE RESTS ON.

    A manifest whose ROW ORDER depended on thread scheduling would make
    train_vlm.sample_eval_records -- a seeded random.sample over the record
    list -- draw a different eval set on every rebuild. That is invisible
    until two runs disagree and nobody can say why.

    The fake decode sleeps a jittered amount so completion order is
    genuinely NOT submission order; without the order-preserving assembly in
    PASS 3 this fails.
    """
    import time

    def jittery_decode(video_path, first, last, n_frames, size):
        time.sleep(0.002 * ((first % 5) + 1))
        return [np.zeros((size, size, 3), dtype=np.uint8) for _ in range(n_frames)]

    video_root = _threading_setup(tmp_path, monkeypatch)
    serial, drops_serial = extract_frames_for_windows(
        _many_records(), video_root, str(tmp_path / "serial"),
        n_frames=2, size=32, decode_fn=jittery_decode, workers=1)
    threaded, drops_threaded = extract_frames_for_windows(
        _many_records(), video_root, str(tmp_path / "threaded"),
        n_frames=2, size=32, decode_fn=jittery_decode, workers=8)

    assert [m["question"] for m in serial] == [m["question"] for m in threaded]
    # Not a magic number: the fake video is 600 frames at 60fps = 10s, so
    # windows past t=10 are legitimately out of bounds. What matters is that
    # BOTH paths keep the same ones and enough survive for the pool to have
    # genuinely interleaved.
    assert len(serial) == len(threaded) >= 16
    assert dict(drops_serial) == dict(drops_threaded)


def test_threaded_run_writes_every_frame_file(tmp_path, monkeypatch):
    """Order parity is not enough -- the files have to actually be there.
    A pool that swallowed a write error would still return a clean manifest."""
    def fake_decode(video_path, first, last, n_frames, size):
        return [np.zeros((size, size, 3), dtype=np.uint8) for _ in range(n_frames)]

    video_root = _threading_setup(tmp_path, monkeypatch)
    manifest, _ = extract_frames_for_windows(
        _many_records(8), video_root, str(tmp_path / "frames"),
        n_frames=3, size=32, decode_fn=fake_decode, workers=8)

    assert manifest
    for record in manifest:
        assert len(record["frame_paths"]) == 3
        for path in record["frame_paths"]:
            assert Path(path).exists(), path
            assert Path(path).stat().st_size > 0


def test_threaded_decode_failure_drops_the_same_records_as_serial(tmp_path, monkeypatch):
    """A window that fails must cost ALL its records, and must be attributed
    to the same window under threading -- `failed` is keyed by id(job), so a
    bug there would drop somebody ELSE'S window."""
    def flaky_decode(video_path, first, last, n_frames, size):
        if first % 3 == 0:
            raise ValueError("synthetic decode failure at %d" % first)
        return [np.zeros((size, size, 3), dtype=np.uint8) for _ in range(n_frames)]

    video_root = _threading_setup(tmp_path, monkeypatch)
    serial, drops_serial = extract_frames_for_windows(
        _many_records(12), video_root, str(tmp_path / "s"),
        n_frames=2, size=32, decode_fn=flaky_decode, workers=1)
    threaded, drops_threaded = extract_frames_for_windows(
        _many_records(12), video_root, str(tmp_path / "t"),
        n_frames=2, size=32, decode_fn=flaky_decode, workers=8)

    assert drops_serial["decode_failed"] > 0
    assert dict(drops_serial) == dict(drops_threaded)
    assert [m["question"] for m in serial] == [m["question"] for m in threaded]


def test_workers_defaults_to_serial(tmp_path, monkeypatch):
    """The default must stay 1 so nothing that calls this without the new
    argument silently changes behaviour."""
    import inspect

    assert inspect.signature(extract_frames_for_windows).parameters["workers"].default == 1


def test_dry_run_ignores_workers(tmp_path, monkeypatch):
    """--dry-run never decodes, so the pool must not even be constructed;
    the manifest is still the full set of resolved paths."""
    def exploding_decode(*a, **k):
        raise AssertionError("decode_fn must not be called under dry_run")

    video_root = _threading_setup(tmp_path, monkeypatch)
    manifest, drops = extract_frames_for_windows(
        _many_records(6), video_root, str(tmp_path / "d"),
        n_frames=2, size=32, decode_fn=exploding_decode, dry_run=True, workers=8)

    assert len(manifest) == 12
    assert not drops


def test_a_window_whose_frames_all_exist_is_not_decoded_again(tmp_path, monkeypatch):
    """RESUMABILITY. A 16-frame rebuild runs for hours and CHTC preemption on
    a job that long is a real, observed risk. A second run must skip completed
    windows rather than start over."""
    calls = []

    def counting_decode(video_path, first, last, n_frames, size):
        calls.append(first)
        return [np.zeros((size, size, 3), dtype=np.uint8) for _ in range(n_frames)]

    video_root = _threading_setup(tmp_path, monkeypatch)
    root = str(tmp_path / "frames")
    first_run, _ = extract_frames_for_windows(
        _many_records(6), video_root, root, n_frames=2, size=32,
        decode_fn=counting_decode, workers=1)
    decoded_first = len(calls)
    assert decoded_first > 0

    calls.clear()
    second_run, _ = extract_frames_for_windows(
        _many_records(6), video_root, root, n_frames=2, size=32,
        decode_fn=counting_decode, workers=1)

    assert calls == []                                    # nothing re-decoded
    assert [m["question"] for m in second_run] == [m["question"] for m in first_run]


def test_a_partially_written_window_is_decoded_again(tmp_path, monkeypatch):
    """Checked per FILE, not per directory. A job killed mid-window leaves a
    partial directory; treating that as done would put a manifest entry on
    frames that were never written."""
    calls = []

    def counting_decode(video_path, first, last, n_frames, size):
        calls.append(first)
        return [np.zeros((size, size, 3), dtype=np.uint8) for _ in range(n_frames)]

    video_root = _threading_setup(tmp_path, monkeypatch)
    root = str(tmp_path / "frames")
    manifest, _ = extract_frames_for_windows(
        _many_records(4), video_root, root, n_frames=3, size=32,
        decode_fn=counting_decode, workers=1)
    # simulate a kill partway through one window
    Path(manifest[0]["frame_paths"][-1]).unlink()

    calls.clear()
    extract_frames_for_windows(
        _many_records(4), video_root, root, n_frames=3, size=32,
        decode_fn=counting_decode, workers=1)

    assert len(calls) == 1                                # exactly the broken one
    for path in manifest[0]["frame_paths"]:
        assert Path(path).exists()
