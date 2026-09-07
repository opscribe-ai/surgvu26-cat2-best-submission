#!/bin/bash
set -uo pipefail
# The 11 public sample cases through the REAL serving path, twice: with motion
# and without. Then diff the answers.
#
#   condor/motion_ab.sh [device]
#
# WHY THIS IS THE TEST THAT MATTERS. Every safety claim about the motion work
# so far is a unit test -- clip_record is byte-identical without the block,
# decode_clip_bursts returns the same centres as decode_clip, the router
# accessors report no-evidence while the gate is closed. All true, all
# checked, and all checked in isolation. This runs the actual entrypoint over
# the actual sample videos and compares the actual answer strings, which is
# the only claim the leaderboard cares about.
#
# THE EXPECTED RESULT IS ZERO DIFFERENCES. STATIC_ACTIVITY_THRESHOLD is None,
# so motion is recorded and ignored. A single changed answer means something
# reads the block that should not, and the whole motion branch stops until it
# is found -- v1 and v2 scored identically to four decimals because their
# answers were byte-identical, so an unexplained diff here is a real risk to
# a number that has never moved.
#
# It also prices the flag: the timings line shows what the extra 32 frames of
# decode cost against the 600 s per-case budget.

echo "host: $(hostname)  start: $(date)"
DEVICE="${1:-auto}"

SAMPLE=/staging/groups/bhaskar_opscribe/surgvu/cat2_sample
MODELS_SRC=/staging/n/nkalthoff/surgvu26/models

for label in motion_off motion_on; do
    echo '{}' > "${label}_candidates.json"
    echo '{}' > "${label}_results.json"
done
echo '{}' > motion_ab.json

[ -d "$SAMPLE" ] || { echo "FATAL: $SAMPLE not visible on this node"; exit 75; }

# Local checkpoints, the same re-rooting the container does. Read the names
# out of the config rather than hardcoding them -- condor/validate.sh still
# names tools_v2.pt, which the shipped config has not bound since v2, and a
# hardcoded name here would rot the same way.
mkdir -p models
python3 - <<'PY' > /tmp/ckpt_names.txt
import json
config = json.load(open("config/perception.json"))
for entry in config["experts"].values():
    print(entry["checkpoint_name"])
PY
while read -r name; do
    [ -z "$name" ] && continue
    cp "$MODELS_SRC/$name" models/ || { echo "FATAL: no $name"; exit 42; }
done < /tmp/ckpt_names.txt
echo "== local checkpoints =="
sha256sum models/*.pt

run_pass () {
    local label="$1"; shift
    echo
    echo "=============================================================="
    echo "PASS $label  $*"
    echo "=============================================================="
    python3 scripts/validate_cases.py "$SAMPLE" \
        --work-dir "./work_$label" \
        --out-prefix "$label" \
        --models-dir "$PWD/models" \
        --device "$DEVICE" \
        --label "$label" \
        "$@"
    echo "PASS $label exit $?"
}

run_pass motion_off
run_pass motion_on --entrypoint-arg=--motion

python3 - <<'PY'
import json
from pathlib import Path

def answers(path):
    if not Path(path).exists():
        return {}
    blob = json.loads(Path(path).read_text() or "{}")
    cases = blob.get("cases", blob)
    out = {}
    for key, value in (cases.items() if isinstance(cases, dict) else []):
        if isinstance(value, dict):
            out[key] = value.get("answer", value.get("response"))
        else:
            out[key] = value
    return out

off = answers("motion_off_candidates.json")
on = answers("motion_on_candidates.json")
keys = sorted(set(off) | set(on))
diffs = [k for k in keys if off.get(k) != on.get(k)]

# DID BOTH PASSES ACTUALLY RUN? A pass that produced nothing answers None to
# everything, and then EVERY case reads as "changed" -- eleven alarming
# regressions that are really one broken command line. That happened on the
# first run of this script: --entrypoint-arg --motion made argparse consume
# --motion as a flag rather than a value, motion_on wrote nothing, and the
# diff printed eleven changes.
#
# Same shape as the segfault that run_tests.py reported as a clean pass: a
# harness that cannot tell "no result" from "a result" will eventually report
# one as the other, and in both directions.
broken = []
for name, side in (("motion_off", off), ("motion_on", on)):
    produced = [k for k, v in side.items() if v not in (None, "")]
    if not produced:
        broken.append("%s produced no answers at all (%d keys, all empty)"
                      % (name, len(side)))
    elif len(produced) < len(keys):
        broken.append("%s answered only %d of %d cases"
                      % (name, len(produced), len(keys)))
if not keys:
    broken.append("neither pass produced a candidates file with cases in it")

if broken:
    print("\n== HARNESS FAILURE, NOT A RESULT ==")
    for line in broken:
        print("  " + line)
    print("  The %d 'changed' answers below are an artefact of a pass that "
          "did not run. Fix the harness before reading them as regressions."
          % len(diffs))

print("\n== ANSWER DIFF: %d case(s), %d changed ==" % (len(keys), len(diffs)))
for key in diffs:
    print("  %s\n    off: %r\n    on : %r" % (key, off.get(key), on.get(key)))
if keys and not diffs and not broken:
    print("  none. Motion is recorded and ignored, as the closed gate intends.")

Path("motion_ab.json").write_text(json.dumps({
    "cases": len(keys), "changed": len(diffs), "changed_cases": diffs,
    "off": off, "on": on,
    "harness_failures": broken,
    # clean requires BOTH that nothing changed AND that both passes ran.
    "clean": bool(keys) and not diffs and not broken,
}, indent=2), encoding="utf-8")
if broken:
    raise SystemExit(70)
PY
DIFF_RC=$?

# THE COMPARISON'S EXIT CODE HAS TO REACH THE JOB. This script ended with a
# bare `exit 0`, which discarded it -- so a harness failure the Python had
# just detected and printed in capitals would still be reported to HTCondor as
# success. That is the third time this session a check has been written whose
# result could not fail anything: run_tests.py counted a segfault as clean,
# this script called a dead pass eleven regressions, and then threw away the
# exit code that said so.
echo "diff exit: $DIFF_RC  end: $(date)"
exit "$DIFF_RC"
