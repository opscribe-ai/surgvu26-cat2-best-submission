"""Score every flag combination on the 11-case sample, baseline included.

WHY THIS EXISTS AND WHY IT RUNS BEFORE THE SUBMISSION, NOT AFTER. v4 carried
both the Aug-13 router batch and the motion gate relative to the last scored
submission, so whatever v4 scores, the cause is unresolved -- a move up
cannot be credited to the motion gate and a move down cannot be blamed on it.
v5 ships six workstreams at once, by explicit decision. The flags are how
attribution is recovered, and they only work if the matrix was recorded.

WHAT THIS IS NOT. Eleven cases is a small sample and the user's instruction
is not to over-weight it. This is a TRIPWIRE, not a gate: it does not decide
whether anything ships. It exists so that when the leaderboard moves, there
is something to read the move against.

THE --variant-head / --yolo INTERACTION IS NOT A BUG. The answer gate's
third condition for using the variant head's Large-vs-Mega call is "a needle
driver was actually detected", which is read off the `yolo` evidence block
-- so `--variant-head` alone is EXPECTED to change nothing on case126/
case132 (the two cases whose gold answer depends on the size distinction):
the gate has no detection to act on without `--yolo` also enabled. Only the
`--yolo --variant-head` combination -- and the full set -- can move them.
This matrix prints case126/case132 next to every combo's mean specifically
so that "only --variant-head, nothing moved" reads as the gate working, not
as the head failing.

WHAT THIS DOES NOT ASSUME. Nothing here hardcodes the diagnostic's predicted
gain. The whole point of running this end to end is that it can refute that
prediction; a script that baked the answer in would not be a measurement.

TWO CONTAINERS, ONE SCRIPT, TWO MODES. `scripts/inference.py` needs
surgvu26-train.sif (torch/torchvision/opencv, plus yolov5's pandas/tqdm/
matplotlib/seaborn for --yolo). `surgvu.scoring.Scorer` needs bert_score/
transformers, which live only in surgvu26-extract.sif's venv -- a
DIFFERENT, non-pytorch base image (see condor/score.sh). Apptainer cannot
nest one exec inside another, so one process cannot hold both environments
at once. Hence three modes:

    --mode run    drives scripts/inference.py once per case per combination
                  through validate_cases.run_case (a subprocess -- see that
                  module's docstring for why one process per case). Writes
                  one JSON record per combination under --run-dir. Needs
                  only what validate_cases.py needs: no torch import of its
                  own, and no bert_score.
    --mode score  reads those per-combination records back and scores each
                  with the real Scorer (in-process, via score_sample.py's
                  load_sample_cases/build_pairs helpers). Needs bert_score/
                  transformers; does not touch inference.py or torch.
    --mode full   does both in one process, for a (hypothetical, or future)
                  environment that has everything. This is also what the
                  unit tests exercise, via injected fakes, since the pure
                  logic (combinations/bucket_rows/assemble_matrix) is
                  identical in every mode.

condor/flag_matrix.sh runs `run` then `score` as two separate, SEQUENTIAL
`apptainer exec` calls from a vanilla (non-container-universe) job -- the
same shape condor/score.sh already uses for the scoring half alone.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_cases                                       # noqa: E402 - subprocess harness, imports no torch
from score_sample import build_pairs, load_sample_cases      # noqa: E402 - imports no torch at module level

WATCH_CASES = ("case126", "case132")
BASELINE_KEY = "(baseline)"


# --------------------------------------------------------------------------
# the enumerator
# --------------------------------------------------------------------------

def combinations(flags):
    """Every subset of `flags`, baseline first, ordered by subset size.

    The empty combination is included deliberately and sorts first. A
    matrix that only compares enabled combinations to each other never
    measures any of them against what actually ships today. The ordering
    (baseline, then every size-1 subset in `flags`' own order, then every
    size-2 subset, ...) is itself load-bearing: it is what the matrix's keys
    depend on to stay comparable across runs. `itertools.combinations` over
    a list, not a set, is what keeps this deterministic.
    """
    ordered = list(flags)
    out = []
    for size in range(len(ordered) + 1):
        out.extend(itertools.combinations(ordered, size))
    return out


def combo_key(combo):
    """The matrix's key for one combination: '--a --b', or the literal
    string '(baseline)' for the empty combination."""
    return " ".join(combo) if combo else BASELINE_KEY


def combo_slug(combo):
    """A filesystem-safe name for one combination's intermediate run file."""
    if not combo:
        return "baseline"
    return "_".join(flag.lstrip("-") for flag in combo)


# --------------------------------------------------------------------------
# mode "run" -- driving validate_cases.py, no torch/bert_score needed
# --------------------------------------------------------------------------

def bucket_rows(rows):
    """Split validate_cases.run_case rows into (candidates, errors).

    `candidates` is {case_id: answer} for every row that exited 0 with a
    problem-free answer (matching validate_cases.candidates' own rule,
    restated here so a case-level error can be captured too, not just
    dropped). `errors` is {case_id: {...}} for every other row -- a
    nonzero exit even with an answer present is still an error
    (inference.py is built to always exit 0; if it does not, that is a
    finding, not a detail to average away).

    Every row lands in EXACTLY ONE of the two dicts. A row silently
    reaching neither is how a missing case would masquerade as a complete
    run.
    """
    candidates, errors = {}, {}
    for row in rows:
        if row.get("problem") is None and row.get("returncode") == 0:
            candidates[row["case_id"]] = row["answer"]
        else:
            errors[row["case_id"]] = {"returncode": row.get("returncode"),
                                      "problem": row.get("problem"),
                                      "stderr_tail": row.get("stderr_tail")}
    return candidates, errors


def run_combo(cases, combo, work_dir, python, entrypoint, models_dir, device,
             frames, timeout, fixed_args=()):
    """Run every sample case once through validate_cases.run_case for one
    flag combination.

    `fixed_args` are appended after the combo's own flags on every call --
    plumbing --yolo/--variant-head need (weight/repo paths) but that is not
    itself part of the sweep, so it must reach EVERY combination, including
    the baseline, identically.

    Returns {"candidates": {...}, "errors": {...}} -- see bucket_rows.
    """
    extra = list(combo) + list(fixed_args)
    rows = [validate_cases.run_case(case, work_dir, python, entrypoint,
                                    models_dir, device, frames, timeout,
                                    extra=extra)
           for case in cases]
    candidates, errors = bucket_rows(rows)
    return {"candidates": candidates, "errors": errors}


def _run_path(run_dir, combo):
    return Path(run_dir) / ("%s.json" % combo_slug(combo))


def _write_run_record(run_dir, combo, record):
    path = _run_path(run_dir, combo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": combo_key(combo), "combo": list(combo),
                                "candidates": record["candidates"],
                                "errors": record["errors"]},
                               indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _read_run_record(run_dir, combo):
    """The record `_write_run_record` wrote for `combo`, or None if there
    is none -- a combination `--mode run` never got to, or that a run
    crashed hard enough not to leave a file behind at all."""
    path = _run_path(run_dir, combo)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"candidates": data["candidates"], "errors": data["errors"]}


# --------------------------------------------------------------------------
# mode "score" -- the real Scorer, in-process
# --------------------------------------------------------------------------

def score_combo(scorer, sample_cases, candidates):
    """(mean, per_case) for one combination's candidates, via the real
    surgvu.scoring.Scorer -- BERTScore-F1, roberta-large, rescaled with
    baseline, MAX over the five references, meaned across cases."""
    pairs = build_pairs(sample_cases, candidates)
    report = scorer.score_many(pairs)
    per_case = {row["case_id"]: row["bertscore_f1"] for row in report["results"]}
    return report["aggregates"]["bertscore_f1"], per_case


# --------------------------------------------------------------------------
# result assembly -- pure, torch-free, the part the tests pin hardest
# --------------------------------------------------------------------------

def assemble_matrix(combos, get_run, do_score):
    """{key: result} for every combo in `combos`, in order.

    `get_run(combo)` -> {"candidates": {...}, "errors": {...}}, or None if
    no run record exists for this combination at all.
    `do_score(candidates)` -> (mean, per_case); may raise.

    A combo's result is exactly one of:
      * {"mean": float, "per_case": {...}}   on a clean run that scored
      * {"error": str, ...}                  otherwise

    and it lands in that second shape for any of three reasons, EACH
    recorded rather than silently dropped:
      1. no run record at all (mode "run" never reached this combination,
         or crashed before writing one),
      2. one or more cases failed (`errors` non-empty) -- checked BEFORE
         scoring is attempted, so a partial candidates dict is never handed
         to the scorer, which would otherwise silently score fewer than the
         full eleven and inflate the mean,
      3. scoring itself raised.

    Every key from `combos` is present in the returned dict either way. A
    matrix with a missing row looks like a matrix; this function's whole
    job is to make sure that never happens.
    """
    results = {}
    for combo in combos:
        key = combo_key(combo)
        record = get_run(combo)
        if record is None:
            results[key] = {"error": "no run record for this combination "
                                     "(mode 'run' did not produce one)"}
            continue
        errors = record.get("errors") or {}
        if errors:
            results[key] = {"error": "case(s) failed: %s"
                                     % ", ".join(sorted(errors)),
                            "cases": errors}
            continue
        try:
            mean, per_case = do_score(record["candidates"])
        except Exception as exc:            # noqa: BLE001 - record, never crash the matrix
            results[key] = {"error": "scoring failed: %r" % (exc,)}
            continue
        results[key] = {"mean": mean, "per_case": per_case}
    return results


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _report(results):
    """Print the mean plus case126/case132 for every combo -- the mean
    alone hides which case, if any, actually moved."""
    for key, result in results.items():
        if "error" in result:
            print("%-40s ERROR: %s" % (key, result["error"]))
            continue
        watch = ["%s=%.4f" % (case_id, result["per_case"][case_id])
                for case_id in WATCH_CASES if case_id in result["per_case"]]
        print("%-40s mean=%.4f  %s" % (key, result["mean"], "  ".join(watch)))


def _write_matrix(out_path, results):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print("wrote %s (%d combinations)" % (out_path, len(results)))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sample_dir", help="the nested cat2_sample directory "
                        "(validate_cases.discover_cases layout)")
    parser.add_argument("--flags", nargs="+",
                        default=["--motion-v2", "--yolo", "--variant-head"],
                        help="the flags to sweep every subset of; the "
                             "baseline (none of them) is always included")
    parser.add_argument("--mode", choices=("full", "run", "score"),
                        default="full",
                        help="see the module docstring for what each needs")
    parser.add_argument("--run-dir", default="./flag_matrix_runs",
                        help="where 'run' writes and 'score' reads the "
                             "per-combination candidate/error records")
    parser.add_argument("--work-dir", default="./flag_matrix_work",
                        help="scratch /input,/output staging area, passed "
                             "through to validate_cases.run_case")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter that runs scripts/inference.py")
    parser.add_argument("--entrypoint",
                        default=str(Path(__file__).resolve().with_name("inference.py")))
    parser.add_argument("--models-dir",
                        help="re-root the config's tools/task checkpoints "
                             "here by basename, same as validate_cases.py")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--fixed-arg", action="append", default=[],
                        dest="fixed_args",
                        help="extra entrypoint argument passed on EVERY "
                             "combination, repeatable -- e.g. "
                             "--yolo-weights=/path, --yolo-repo=/path, "
                             "--variant-weights=/path, which --yolo and "
                             "--variant-head need but which are not "
                             "themselves swept")
    parser.add_argument("--out", default="baselines/flag_matrix.json")
    return parser.parse_args(argv)


def _do_run(args):
    cases = validate_cases.discover_cases(args.sample_dir)
    combos = combinations(args.flags)
    print("== flag_matrix run == %d cases, %d combination(s), flags=%s"
         % (len(cases), len(combos), args.flags), flush=True)
    for combo in combos:
        print("=== %s ===" % combo_key(combo), flush=True)
        record = run_combo(cases, combo, args.work_dir, args.python,
                           args.entrypoint, args.models_dir, args.device,
                           args.frames, args.timeout, args.fixed_args)
        _write_run_record(args.run_dir, combo, record)
        if record["errors"]:
            print("    %d case(s) failed: %s"
                 % (len(record["errors"]), ", ".join(sorted(record["errors"]))),
                 flush=True)
        else:
            print("    %d case(s) answered" % len(record["candidates"]), flush=True)
    print("wrote %d run record(s) under %s" % (len(combos), args.run_dir),
         flush=True)
    return 0


def _do_score(args):
    sample_cases = load_sample_cases(args.sample_dir)
    combos = combinations(args.flags)

    from surgvu.scoring import Scorer
    scorer = Scorer()

    def get_run(combo):
        return _read_run_record(args.run_dir, combo)

    def do_score(candidates):
        return score_combo(scorer, sample_cases, candidates)

    results = assemble_matrix(combos, get_run, do_score)
    _report(results)
    _write_matrix(args.out, results)
    return 0


def _do_full(args):
    cases = validate_cases.discover_cases(args.sample_dir)
    sample_cases = load_sample_cases(args.sample_dir)
    combos = combinations(args.flags)

    from surgvu.scoring import Scorer
    scorer = Scorer()

    records = {}

    def get_run(combo):
        if combo not in records:
            print("=== %s ===" % combo_key(combo), flush=True)
            records[combo] = run_combo(cases, combo, args.work_dir,
                                       args.python, args.entrypoint,
                                       args.models_dir, args.device,
                                       args.frames, args.timeout,
                                       args.fixed_args)
        return records[combo]

    def do_score(candidates):
        return score_combo(scorer, sample_cases, candidates)

    results = assemble_matrix(combos, get_run, do_score)
    _report(results)
    _write_matrix(args.out, results)
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "run":
        return _do_run(args)
    if args.mode == "score":
        return _do_score(args)
    return _do_full(args)


if __name__ == "__main__":
    raise SystemExit(main())
