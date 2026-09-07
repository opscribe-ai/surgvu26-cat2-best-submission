"""Run the submission entrypoint once per sample case, through the real contract.

This is the packaging proof, not a convenience wrapper. It exists to answer
three questions that a direct in-process call cannot:

  1. does `scripts/inference.py` write a valid, non-empty JSON string to
     `/output/visual-context-response.json` for every case, and
  2. how long does ONE case take END TO END -- including interpreter start and
     the torch import -- because the challenge runs one case per container
     invocation, so nothing is amortised across cases, and
  3. does it still work when the checkpoints are NOT at the `/staging/...`
     paths `config/perception.json` records, but re-rooted with `--models-dir`
     the way baked-in weights will be.

Hence one SUBPROCESS per case. Looping in-process would hide the interpreter
and import cost, which is real budget, and would let one case's warm CUDA
context flatter the next one's.

Nothing here imports torch. The harness has to be runnable and testable on a
login node that has no torch at all; the container is the thing under test,
not the thing this file lives in.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from collections import OrderedDict, namedtuple
from pathlib import Path

# The interface slugs, restated rather than imported: importing them from
# scripts/inference.py would drag torch in, and a harness that agreed with the
# code under test by construction could not catch it renaming a file.
VIDEO_NAME = "endoscopic-robotic-surgery-video.mp4"
QUESTION_NAME = "visual-context-question.json"
RESPONSE_NAME = "visual-context-response.json"

QUESTION_SUFFIX = "_question"

# "[surgvu] timings question=0.00s decode=1.20s ... total=21.90s"
TIMINGS_PREFIX = "timings "
TIMING_PAIR = re.compile(r"([A-Za-z_]+)=([0-9]+\.?[0-9]*)s")

Case = namedtuple("Case", "case_id video question")


# --------------------------------------------------------------------------
# discovery and staging
# --------------------------------------------------------------------------

def discover_cases(sample_root):
    """Every case under the nested cat2_sample layout, sorted by case id.

    A case missing either half is an ERROR. Skipping it would turn eleven
    cases into ten and still print a table that looks complete.
    """
    sample_root = Path(sample_root)
    cases = []
    for gt_path in sorted(sample_root.glob("*/*.json")):
        case_id = gt_path.stem
        if case_id.endswith(QUESTION_SUFFIX):
            continue
        video = gt_path.with_suffix(".mp4")
        question = gt_path.with_name("%s%s.json" % (case_id, QUESTION_SUFFIX))
        if not video.exists():
            raise ValueError("case %s has no video at %s" % (case_id, video))
        if not question.exists():
            raise ValueError("case %s has no question at %s" % (case_id, question))
        cases.append(Case(case_id, video, question))
    if not cases:
        raise ValueError("found no cases under %s" % (sample_root,))
    return cases


def stage_case(case, work_root):
    """Build a private `/input` for one case; return (input_dir, output_dir).

    The files are COPIED under the contract's exact names. Symlinking would be
    cheaper and would also mean the container never proves it can read a real
    file at the real path, which is the only thing being tested.
    """
    case_root = Path(work_root) / case.case_id
    input_dir = case_root / "input"
    output_dir = case_root / "output"
    if case_root.exists():
        shutil.rmtree(case_root)
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    shutil.copyfile(str(case.video), str(input_dir / VIDEO_NAME))
    shutil.copyfile(str(case.question), str(input_dir / QUESTION_NAME))
    return input_dir, output_dir


# --------------------------------------------------------------------------
# reading what the run left behind
# --------------------------------------------------------------------------

def parse_timings(text):
    """The stages from the entrypoint's last `timings` line, in order.

    Returns an empty mapping when there is no such line -- a run that died
    before printing one has no timings, and reporting zeroes for it would read
    as an instantaneous success.
    """
    line = None
    for candidate in text.splitlines():
        if TIMINGS_PREFIX in candidate:
            line = candidate
    if line is None:
        return OrderedDict()
    return OrderedDict((name, float(seconds))
                       for name, seconds in TIMING_PAIR.findall(line))


def read_response(path):
    """(answer, problem). `problem` is None only for a valid non-empty string.

    The contract is a JSON-ENCODED STRING. Every other shape is a failure of
    the case even when a human could read the answer out of it:
      * bare `Yes` is malformed JSON,
      * `{"answer": "Yes"}` is not a String interface,
      * `""` crashes the official scorer outright.
    """
    path = Path(path)
    if not path.exists():
        return None, "missing response file at %s" % (path,)
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
    except ValueError:
        return None, "not valid JSON: %r" % (raw[:80],)
    if not isinstance(value, str):
        return None, "not a JSON string but %s" % (type(value).__name__,)
    if not value.strip():
        return None, "empty answer"
    return value, None


# --------------------------------------------------------------------------
# running one case
# --------------------------------------------------------------------------

def build_command(python, entrypoint, input_dir, output_dir,
                  models_dir=None, device="auto", frames=None, extra=()):
    """The argv for one invocation.

    `--frames` is omitted unless overridden so the measurement stays bound to
    `config/perception.json`; passing the config's own value back in would
    make the two look independent when they are not.

    `extra` is appended verbatim and exists for exactly one job: running the
    same 11 cases twice, once with `--vlm` and once without, so the two
    candidate files can be compared byte for byte. It goes LAST so that a flag
    passed through it cannot displace one this harness sets itself.
    """
    command = [str(python), str(entrypoint),
               "--input-dir", str(input_dir),
               "--output-dir", str(output_dir),
               "--device", str(device)]
    if models_dir:
        command += ["--models-dir", str(models_dir)]
    if frames:
        command += ["--frames", str(int(frames))]
    return command + [str(argument) for argument in extra]


def run_case(case, work_root, python, entrypoint, models_dir, device, frames,
             timeout=900, extra=()):
    """One case, one process. Returns the report row."""
    input_dir, output_dir = stage_case(case, work_root)
    command = build_command(python, entrypoint, input_dir, output_dir,
                            models_dir, device, frames, extra)
    started = time.time()
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, timeout=timeout)
        returncode, stderr = completed.returncode, completed.stderr.decode(
            "utf-8", "replace")
    except subprocess.TimeoutExpired:
        returncode, stderr = -1, "TIMEOUT after %ss" % (timeout,)
    wall = time.time() - started

    answer, problem = read_response(output_dir / RESPONSE_NAME)
    return {"case_id": case.case_id,
            "answer": answer,
            "problem": problem,
            "returncode": returncode,
            "wall_seconds": wall,
            "timings": parse_timings(stderr),
            "stderr_tail": "\n".join(stderr.splitlines()[-25:])}


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def candidates(rows):
    """{case_id: answer} for the scorer -- only cases that produced one."""
    return {row["case_id"]: row["answer"]
            for row in rows if row.get("answer")}


def summarise(rows):
    """Aggregate. `ok` is True only if every case answered AND exited 0."""
    valid = sum(1 for row in rows
                if row.get("problem") is None and row.get("returncode") == 0)
    return {"total": len(rows), "valid": valid, "ok": valid == len(rows)}


STAGES = ["question", "decode", "load_tools", "load_task", "tools_infer",
          "task_infer", "route", "total"]


def format_table(rows):
    header = "%-9s %-30s %5s %7s %7s %7s %7s %7s %7s %7s" % (
        "case", "answer", "rc", "wall", "decode", "ld_tool", "ld_task",
        "tools", "task", "total")
    lines = [header, "-" * len(header)]
    for row in rows:
        timings = row["timings"]
        answer = row["answer"] or ("FAIL: %s" % row["problem"])
        if len(answer) > 30:
            answer = answer[:29] + "…"
        lines.append("%-9s %-30s %5s %7.1f %7s %7s %7s %7s %7s %7s" % (
            row["case_id"], answer, row["returncode"], row["wall_seconds"],
            _cell(timings, "decode"), _cell(timings, "load_tools"),
            _cell(timings, "load_task"), _cell(timings, "tools_infer"),
            _cell(timings, "task_infer"), _cell(timings, "total")))
    lines.append("-" * len(header))
    walls = [row["wall_seconds"] for row in rows]
    if walls:
        lines.append("wall: min %.1fs  mean %.1fs  max %.1fs  (budget 600s)"
                     % (min(walls), sum(walls) / len(walls), max(walls)))
    summary = summarise(rows)
    lines.append("valid non-empty JSON string answers: %d/%d  ok=%s"
                 % (summary["valid"], summary["total"], summary["ok"]))
    return "\n".join(lines)


def _cell(timings, name):
    return "%.1f" % timings[name] if name in timings else "-"


# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sample_root")
    parser.add_argument("--work-dir", default="./validate_work")
    parser.add_argument("--out-prefix", default="validate",
                        help="writes <prefix>_candidates.json and "
                             "<prefix>_results.json")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--entrypoint",
                        default=str(Path(__file__).resolve().with_name("inference.py")))
    parser.add_argument("--models-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--label", default="")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--entrypoint-arg", action="append", default=[],
                        dest="entrypoint_args",
                        help="extra argument passed to the entrypoint, "
                             "repeatable. Used to run the same cases with "
                             "--vlm and without it.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cases = discover_cases(args.sample_root)
    print("== %s == %d cases, device=%s frames=%s models_dir=%s extra=%s"
          % (args.label or "validate", len(cases), args.device,
             args.frames or "config", args.models_dir,
             args.entrypoint_args or "[]"), flush=True)

    rows = []
    for case in cases:
        row = run_case(case, args.work_dir, args.python, args.entrypoint,
                       args.models_dir, args.device, args.frames, args.timeout,
                       args.entrypoint_args)
        rows.append(row)
        print("%-9s rc=%-3s wall=%6.1fs  %s"
              % (row["case_id"], row["returncode"], row["wall_seconds"],
                 row["answer"] or ("FAIL: %s" % row["problem"])), flush=True)
        if row["problem"] is not None or row["returncode"] != 0:
            print("---- stderr tail for %s ----\n%s\n----"
                  % (row["case_id"], row["stderr_tail"]), flush=True)

    print()
    print(format_table(rows), flush=True)

    Path("%s_candidates.json" % args.out_prefix).write_text(
        json.dumps(candidates(rows), indent=2, sort_keys=True), encoding="utf-8")
    Path("%s_results.json" % args.out_prefix).write_text(
        json.dumps({"label": args.label, "device": args.device,
                    "frames": args.frames, "rows": rows,
                    "summary": summarise(rows)}, indent=2), encoding="utf-8")
    return 0 if summarise(rows)["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
