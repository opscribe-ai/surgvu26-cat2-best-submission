"""Tests for the flag-combination matrix -- the attribution instrument.

Nothing here imports torch or bert_score: scripts/flag_matrix.py's own
`--mode run` needs neither (it only drives validate_cases.py, which is
built the same way), and `--mode score`'s only heavy import
(`surgvu.scoring.Scorer`) is deferred to inside `_do_score`/`_do_full`, so
importing the module -- and every pure function this file exercises --
never touches it either. That is deliberate: the login node has no torch,
and this suite has to be collectible and runnable there.

The baseline (no flags) MUST be in the matrix. A matrix of only the enabled
combinations measures them against each other and not against what ships,
which is the exact mistake that made v4 unreadable.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import flag_matrix as fm                          # noqa: E402
from flag_matrix import (                          # noqa: E402
    assemble_matrix, bucket_rows, combinations, combo_key, combo_slug,
    run_combo,
)


# --------------------------------------------------------------------------
# combinations() -- the enumerator
# --------------------------------------------------------------------------

def test_baseline_is_present():
    assert () in combinations(["--yolo", "--variant-head"])


def test_every_single_flag_is_present():
    combos = combinations(["--yolo", "--variant-head"])
    assert ("--yolo",) in combos
    assert ("--variant-head",) in combos


def test_the_full_set_is_present():
    combos = combinations(["--yolo", "--variant-head"])
    assert ("--yolo", "--variant-head") in combos


def test_count_is_two_to_the_n():
    assert len(combinations(["--a", "--b", "--c"])) == 8


def test_baseline_is_first():
    """The baseline must sort first, not merely be present -- a table that
    prints it last still reads as though the enabled combinations are the
    reference point rather than what ships today.

    Implementation change that breaks this: reordering the `for size in
    range(...)` loop in `combinations` (e.g. largest subset first).
    """
    combos = combinations(["--yolo", "--variant-head", "--motion-v2"])
    assert combos[0] == ()


def test_ordered_by_subset_size():
    """Every 1-flag combo precedes every 2-flag combo precedes the full set.

    Implementation change that breaks this: replacing the size-major loop
    in `combinations` with a flat `itertools.chain.from_iterable` over an
    unordered set of sizes, or building combos via recursive subset
    generation that interleaves sizes.
    """
    combos = combinations(["--a", "--b", "--c"])
    sizes = [len(combo) for combo in combos]
    assert sizes == sorted(sizes)


def test_exact_sequence_for_two_flags():
    """Pins the ACTUAL order, not just membership, for a small input where
    the whole sequence fits in one assertion.

    Implementation change that breaks this: building `combinations` from a
    `set` of flags instead of the input list (a set scrambles insertion
    order and this assertion would then be flaky/wrong across Python
    processes with different hash seeds).
    """
    assert combinations(["--a", "--b"]) == [
        (), ("--a",), ("--b",), ("--a", "--b"),
    ]


# --------------------------------------------------------------------------
# combo_key / combo_slug
# --------------------------------------------------------------------------

def test_combo_key_baseline_is_literal_string():
    """Implementation change that breaks this: returning "" instead of
    "(baseline)" for the empty combo -- an empty-string key is easy to
    mistake for a missing row when scanning the written JSON."""
    assert combo_key(()) == "(baseline)"


def test_combo_key_joins_with_space():
    assert combo_key(("--yolo", "--variant-head")) == "--yolo --variant-head"


def test_combo_slug_baseline():
    assert combo_slug(()) == "baseline"


def test_combo_slug_strips_leading_dashes_and_joins():
    """Implementation change that breaks this: dropping the `.lstrip("-")`,
    which would put a `--` inside the filename."""
    assert combo_slug(("--yolo", "--variant-head")) == "yolo_variant-head"


def test_combo_slug_is_injective_over_the_swept_flags():
    """Two different combinations must never collide on the same run-record
    filename -- a collision would make `--mode score` silently read one
    combination's candidates under another combination's key."""
    combos = combinations(["--motion-v2", "--yolo", "--variant-head"])
    slugs = [combo_slug(combo) for combo in combos]
    assert len(slugs) == len(set(slugs))


# --------------------------------------------------------------------------
# bucket_rows -- every row lands in exactly one bucket
# --------------------------------------------------------------------------

def test_bucket_rows_splits_success_from_failure():
    rows = [
        {"case_id": "case122", "answer": "Yes", "problem": None, "returncode": 0},
        {"case_id": "case123", "answer": None, "problem": "empty answer",
         "returncode": 0},
    ]
    candidates, errors = bucket_rows(rows)
    assert candidates == {"case122": "Yes"}
    assert "case123" in errors
    assert "case122" not in errors


def test_bucket_rows_treats_nonzero_exit_as_a_failure_even_with_an_answer():
    """inference.py is built to exit 0 always; if it does not, the row is a
    finding, not a detail to average away.

    Implementation change that breaks this: checking only `problem is None`
    and ignoring `returncode`.
    """
    rows = [{"case_id": "case122", "answer": "Yes", "problem": None,
             "returncode": 1}]
    candidates, errors = bucket_rows(rows)
    assert candidates == {}
    assert "case122" in errors


def test_bucket_rows_every_row_lands_in_exactly_one_bucket():
    rows = [{"case_id": "c%d" % i, "answer": ("Yes" if i % 2 else None),
             "problem": (None if i % 2 else "empty"), "returncode": 0}
           for i in range(6)]
    candidates, errors = bucket_rows(rows)
    assert set(candidates) | set(errors) == {row["case_id"] for row in rows}
    assert set(candidates) & set(errors) == set()


# --------------------------------------------------------------------------
# run_combo -- wiring into validate_cases.run_case, no real subprocess
# --------------------------------------------------------------------------

def test_run_combo_passes_the_combo_plus_fixed_args_to_every_case(monkeypatch):
    """Implementation change that breaks this: passing only `combo` and
    dropping `fixed_args`, which would silently starve --yolo/--variant-head
    of their weight paths on the cluster."""
    seen = []

    def fake_run_case(case, work_dir, python, entrypoint, models_dir, device,
                      frames, timeout, extra=()):
        seen.append((case, list(extra)))
        return {"case_id": case, "answer": "Yes", "problem": None,
               "returncode": 0}

    monkeypatch.setattr(fm.validate_cases, "run_case", fake_run_case)
    record = run_combo(["case122", "case123"], ("--yolo", "--variant-head"),
                       "work", "python3", "inference.py", None, "auto",
                       None, 900, fixed_args=["--yolo-weights=/w/best.pt"])

    assert record["candidates"] == {"case122": "Yes", "case123": "Yes"}
    assert seen == [
        ("case122", ["--yolo", "--variant-head", "--yolo-weights=/w/best.pt"]),
        ("case123", ["--yolo", "--variant-head", "--yolo-weights=/w/best.pt"]),
    ]


def test_run_combo_baseline_passes_no_combo_flags(monkeypatch):
    seen = []
    monkeypatch.setattr(fm.validate_cases, "run_case",
                        lambda case, *a, extra=(), **kw: (
                            seen.append(list(extra)) or
                            {"case_id": case, "answer": "Yes", "problem": None,
                             "returncode": 0}))
    run_combo(["case122"], (), "work", "python3", "inference.py", None,
             "auto", None, 900)
    assert seen == [[]]


# --------------------------------------------------------------------------
# run record round trip (pure JSON, no torch)
# --------------------------------------------------------------------------

def test_run_record_round_trips_through_json(tmp_path):
    combo = ("--yolo",)
    fm._write_run_record(tmp_path, combo,
                         {"candidates": {"case122": "Yes"}, "errors": {}})
    record = fm._read_run_record(tmp_path, combo)
    assert record == {"candidates": {"case122": "Yes"}, "errors": {}}


def test_missing_run_record_reads_as_none(tmp_path):
    assert fm._read_run_record(tmp_path, ("--yolo",)) is None


def test_different_combos_write_different_files(tmp_path):
    fm._write_run_record(tmp_path, (), {"candidates": {}, "errors": {}})
    fm._write_run_record(tmp_path, ("--yolo",), {"candidates": {"case122": "Yes"},
                                                 "errors": {}})
    assert fm._read_run_record(tmp_path, ()) != \
        fm._read_run_record(tmp_path, ("--yolo",))


# --------------------------------------------------------------------------
# assemble_matrix -- the result-assembly and error-recording tests
# --------------------------------------------------------------------------

def test_assemble_matrix_records_mean_and_per_case_on_success():
    combos = [(), ("--yolo",)]

    def get_run(combo):
        return {"candidates": {"case122": "Yes"}, "errors": {}}

    def do_score(candidates):
        return 0.75, {"case122": 0.75}

    results = assemble_matrix(combos, get_run, do_score)
    assert results["(baseline)"] == {"mean": 0.75, "per_case": {"case122": 0.75}}
    assert results["--yolo"] == {"mean": 0.75, "per_case": {"case122": 0.75}}


def test_assemble_matrix_case_errors_are_recorded_and_never_reach_the_scorer():
    """A combo with a failed case must not reach the scorer at all: scoring
    a partial candidates dict would silently average fewer than eleven cases
    and inflate the mean -- the exact failure mode this instrument exists to
    prevent (constraint 1: report what was NOT run).

    Implementation change that breaks this: removing the `if errors:
    continue` branch in `assemble_matrix` so a partial `candidates` dict
    reaches `do_score`.
    """
    scored = []

    def get_run(combo):
        return {"candidates": {}, "errors": {"case126": {"problem": "empty answer"}}}

    def do_score(candidates):
        scored.append(candidates)
        return 1.0, {}

    results = assemble_matrix([("--yolo",)], get_run, do_score)
    assert "error" in results["--yolo"]
    assert "case126" in results["--yolo"]["cases"]
    assert scored == []


def test_assemble_matrix_records_missing_run_record_as_an_error():
    """A combination `--mode run` never got to (or crashed hard enough not
    to write a record for) must still appear in the matrix as an error, not
    be silently absent -- 'a matrix with a missing row looks like a
    matrix'.

    Implementation change that breaks this: `if record is None: continue`
    instead of recording an error entry.
    """
    results = assemble_matrix([("--motion-v2",)], lambda combo: None,
                              lambda candidates: (1.0, {}))
    assert results["--motion-v2"]["error"]


def test_assemble_matrix_records_scoring_exceptions_instead_of_crashing():
    """Implementation change that breaks this: letting the exception from
    `do_score` propagate instead of being caught -- one bad combination
    would then abort the whole matrix instead of leaving one labelled
    error row."""
    def get_run(combo):
        return {"candidates": {"case122": "Yes"}, "errors": {}}

    def do_score(candidates):
        raise RuntimeError("bert_score exploded")

    results = assemble_matrix([()], get_run, do_score)
    assert "bert_score exploded" in results["(baseline)"]["error"]


def test_assemble_matrix_every_combo_key_present_even_when_all_fail():
    """Implementation change that breaks this: only adding a results[key]
    entry in the success branch."""
    combos = combinations(["--a", "--b"])
    results = assemble_matrix(combos, lambda combo: None,
                              lambda candidates: (1.0, {}))
    assert set(results) == {combo_key(combo) for combo in combos}


def test_assemble_matrix_preserves_combo_order_in_iteration():
    """Implementation change that breaks this: iterating over a set of
    combos, or a dict keyed by combo, instead of the ordered list -- the
    printed progress and the written JSON's key order would then vary
    between runs even though dict equality wouldn't catch it, which is why
    this checks the insertion order directly."""
    combos = combinations(["--a", "--b"])
    results = assemble_matrix(combos, lambda combo: None,
                              lambda candidates: (1.0, {}))
    assert list(results) == [combo_key(combo) for combo in combos]
