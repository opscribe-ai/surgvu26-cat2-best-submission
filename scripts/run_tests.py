"""Run the full test suite INSIDE the container and report machine-readably.

WHY THIS EXISTS. The login node has neither torch nor bert_score, so nine test
modules fail to import there and three more fail on the scorer. Running
`pytest tests/` outside the container reports twelve failures that say nothing
about the code -- which is indistinguishable, at a glance, from twelve real
regressions.

Tonight edited surgvu/{temporal,dataset,extract,router}.py repeatedly while
only the router tests were being run, because those are the ones that work
without torch. This closes that gap: one job, the whole suite, in the image the
code actually runs in.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out")
    parser.add_argument("--path", default="tests")
    args = parser.parse_args(argv)

    proc = subprocess.run([sys.executable, "-m", "pytest", args.path, "-q",
                           "--tb=short"], capture_output=True, text=True)
    # FULL output to staging. Tailing 25 lines shows the FAILED summary and
    # throws away every traceback above it, which is how 21 failures in one
    # module became undiagnosable without another job.
    full = Path("/staging/n/nkalthoff/surgvu26/last_test_run.txt")
    try:
        full.write_text((proc.stdout or "") + "\n--- stderr ---\n"
                        + (proc.stderr or ""), encoding="utf-8")
        print("full output: %s" % full)
    except Exception as error:                        # noqa: BLE001
        print("could not write full output: %s" % error)

    tail = (proc.stdout or "").strip().splitlines()[-25:]
    print("\n".join(tail))
    # STDERR IS PRINTED, NOT SWALLOWED. The first version captured it and
    # showed only stdout, so when pytest failed to start at all the job
    # reported "pytest exit: 1" with no output and nothing to diagnose from.
    # A wrapper that hides the reason for its own failure is worse than no
    # wrapper.
    if proc.stderr and proc.stderr.strip():
        print("\n--- stderr ---")
        print("\n".join(proc.stderr.strip().splitlines()[-15:]))
    print("\npytest exit: %d" % proc.returncode)

    # ENVIRONMENTAL vs REAL. A run that is permanently red teaches everyone to
    # ignore it -- which is how 23 dead serving-path tests went unnoticed for
    # two days -- so failures that say nothing about the code are classified
    # and NAMED rather than left to accumulate.
    #
    # THE CATEGORY IS WHERE A REAL FAILURE HIDES, so what is in it has been
    # checked rather than assumed. It was five; two of those turned out to be
    # a wrong PATH (test_answer_form_eval read the untracked root copy of
    # shipped_candidates.json instead of the tracked outputs/vlm/ one) and are
    # now fixed and passing.
    #
    # The three that remain are genuinely environmental, and the reason is
    # bigger than "bert_score is missing". Probed inside the image on
    # 2026-08-15: it has torch 2.5.1, numpy 2.1.2, requests and tqdm, but NOT
    # transformers and NOT pandas. bert_score needs both. So this is not a
    # one-package gap that could be closed by dropping a wheel into
    # /staging/n/nkalthoff/surgvu26/testpkgs alongside pytest -- it would mean
    # shimming the entire transformers stack into a read-only .sif's
    # PYTHONPATH and hoping it agrees with torch 2.5.1. The scoring venv
    # (/staging/n/nkalthoff/surgvu26/env) exists precisely because this image
    # does not carry that stack; scoring runs there, on purpose.
    #
    # They are classified and NAMED rather than skipped, so a real regression
    # hiding behind the same module name would still show up in the list.
    body = proc.stdout or ""
    environmental = ("No module named 'bert_score'", "No such file or directory")
    failed = [line.split(" ")[1] for line in body.splitlines()
              if line.startswith("FAILED") and len(line.split(" ")) > 1]
    excused = sum(body.count(marker) for marker in environmental)
    real = max(0, len(failed) - excused)
    print("\nfailures: %d total, %d environmental (missing bert_score or an "
          "untransferred file), %d real" % (len(failed), excused, real))
    for name in failed:
        print("   %s" % name)

    summary = [line for line in tail if "passed" in line or "failed" in line]

    # DID PYTEST ACTUALLY FINISH? Everything above counts lines that pytest
    # PRINTS, so a run that dies without printing them counts zero failures
    # and reads as clean. That is not hypothetical: cluster 9660217 took a
    # SIGSEGV at 69% -- returncode -11, no summary line, no FAILED lines --
    # and this script reported "0 total, 0 environmental, 0 real" and exited
    # 0. A green light that a crash can produce is worse than no light, and
    # this whole session had been trusting it.
    #
    # pytest's own exit codes are 0 ok, 1 tests failed, 2 interrupted,
    # 3 internal error, 4 usage error, 5 nothing collected. A NEGATIVE code is
    # death by signal. Anything outside {0, 1}, or a finished-looking run with
    # no summary line, means the count above is not a count of anything.
    crash = None
    if proc.returncode < 0:
        crash = ("pytest was killed by signal %d (segfault, OOM kill, or an "
                 "abort inside a C extension). The failure counts above are "
                 "meaningless: it never printed a summary."
                 % (-proc.returncode))
    elif proc.returncode not in (0, 1):
        crash = ("pytest exited %d, which is not 'ok' (0) or 'tests failed' "
                 "(1) -- 2 is interrupted, 3 internal error, 4 usage error, "
                 "5 nothing collected. No result was produced."
                 % proc.returncode)
    elif not summary:
        crash = ("pytest printed no summary line, so it did not reach the end "
                 "of the run. The failure counts above are not a count of "
                 "anything.")
    if crash:
        print("\nDID NOT COMPLETE: %s" % crash)

    if args.out:
        Path(args.out).write_text(json.dumps({
            "returncode": proc.returncode,
            "failed": failed, "environmental": excused, "real": real,
            "crash": crash,
            "completed": crash is None,
            "summary": summary[-1] if summary else "",
            "tail": tail,
            "stderr_tail": (proc.stderr or "").strip().splitlines()[-10:],
        }, indent=2), encoding="utf-8")
    # Exit 0 when only environmental failures remain, so a green run means
    # something. Any unexplained failure still fails the job -- and so does a
    # run that never finished, whatever it managed to count first.
    if crash:
        return proc.returncode if proc.returncode > 0 else 70
    return 0 if real == 0 else proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
