#!/bin/bash
# Sweep (attention backend x frame plan) on a T4-capped sm_75 card.
#
# The point of the sweep is that ONE of these rows is the fix and we do not
# yet know which. Row 1 must FAIL -- it is the grader's exact configuration,
# and if it passes here the harness is not simulating a T4 and nothing below
# it means anything. Read row 1 first.
set -u

# CREATE THE OUTPUT FIRST, BEFORE ANY EXIT PATH. transfer_output_files names
# results.jsonl unconditionally, so a script that dies in preflight leaves
# nothing to transfer and Condor HOLDS the job -- reporting a file-transfer
# error instead of the FATAL line that actually explains it. Job 9716651 was
# held for exactly this, and the real cause ("missing .sif") was only visible
# in the .out log.
: > results.jsonl

# STAGING IS MOUNTED UNDER TWO DIFFERENT NAMES depending on the node: the flat
# /staging/<user> and the hashed /staging/<initial>/<user>. Job 9716130 saw
# the flat form on bhaskargpu4000; job 9716651 did not, on the same machine.
# Resolve it rather than assuming, because assuming costs a full queue cycle
# to find out.
STAGING=""
# THE FLAT /staging/<user> SYMLINKS WERE DELETED BY CHTC ON 2026-08-31.
# Personal staging now lives ONLY under the alphabetised /staging/<initial>/<user>.
# Two jobs died mid-session on the changeover (9716663 build, 9716651 probe),
# each reporting "not visible on this node" for a directory that was simply
# under its new name. Group staging (/staging/groups/...) is UNAFFECTED.
STAGING=/staging/n/nkalthoff/surgvu26
[ -d "$STAGING" ] || { echo "FATAL: $STAGING not visible on this node"; exit 42; }
echo "staging: $STAGING"

IMAGE="${IMAGE:-$STAGING/surgvu26-submission.sif}"
SIDECAR="${SIDECAR:-$STAGING/models_sidecar_v61}"
# case124 on purpose: "What type of forceps is mentioned?" is the ONE intent
# config/arbiter.json currently arms (tool_identity_open), so this is the
# question the VLM is actually allowed to answer today.
VIDEO="${VIDEO:-$STAGING/cat2_sample_graded/case124/case124.mp4}"
PROBE="$(pwd)/condor/probe_t4_memory.py"

echo "host: $(hostname)  start: $(date)"
for p in "$IMAGE" "$SIDECAR" "$VIDEO" "$PROBE"; do
    [ -e "$p" ] || { echo "FATAL: missing $p"; exit 42; }
done
command -v apptainer >/dev/null || { echo "FATAL: no apptainer"; exit 42; }
if [ -z "${_CONDOR_AssignedGPUs:-}" ]; then
    echo "FATAL: no GPU assigned; this probe is meaningless without one"; exit 42
fi
echo "gpu: $_CONDOR_AssignedGPUs"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null

run() {   # label, then flags forwarded to the probe
    local label="$1"; shift
    echo "--- $label"
    apptainer exec --containall --nv \
        -B "$SIDECAR":/opt/ml/model \
        -B "$VIDEO":/data/case.mp4:ro \
        -B "$PROBE":/tmp/probe.py \
        --env PYTHONPATH=/opt/algorithm/src \
        --env "PYTORCH_CUDA_ALLOC_CONF=${ALLOC_CONF:-}" \
        "$IMAGE" python /tmp/probe.py --video /data/case.mp4 "$@" \
        2> >(grep -E "OutOfMemory|Error|error" | tail -3 >&2) \
        | tee -a results.jsonl
}

# ---- CALIBRATION: does the simulator reproduce a measurement we HOLD? -------
# Forced math, no cap, 5184 tokens. Job 9716098 measured 16.5 GiB for exactly
# this on a real sm_75 Quadro RTX 8000. If this row does not land near 16.5,
# the simulator is wrong and NOTHING BELOW IT COUNTS -- stop and read it.
ALLOC_CONF="" run "CALIBRATION forced-math uncapped 16x512 -- EXPECT ~16.5 GiB" \
    --force-math --no-cap

# ---- the control: the grader's exact configuration, on a simulated T4 -------
ALLOC_CONF="" run "T4sim forced-math 16x512 (5184 tok)  EXPECT OOM" --force-math

# ---- Phase 3: the free lunch ------------------------------------------------
ALLOC_CONF="expandable_segments:True" \
    run "T4sim forced-math 16x512 + expandable_segments" --force-math

# ---- Phase 1: can a non-materialising kernel take Qwen2.5-VL's mask? --------
# mem-efficient (cutlass) exists on sm_75; the open question is whether
# transformers hands it a mask it refuses, which is what forces the fallback.
# A pass here says the fix is a load-time flag and costs no evidence at all.
ALLOC_CONF="" run "T4sim mem-efficient-only 16x512" --no-math-sdp
ALLOC_CONF="expandable_segments:True" \
    run "T4sim mem-efficient-only 16x512 + expandable" --no-math-sdp

# ---- Phase 2: buy the fit with tokens (attention cost is quadratic) ---------
ALLOC_CONF="expandable_segments:True" \
    run "T4sim forced-math 16x384 (2912 tok)" --force-math --size 384
ALLOC_CONF="expandable_segments:True" \
    run "T4sim forced-math 12x448 (2352 tok)" --force-math --frames 12 --size 448
ALLOC_CONF="expandable_segments:True" \
    run "T4sim forced-math 8x512  (2592 tok)" --force-math --frames 8

echo
echo "================ SUMMARY ================"
python3 - <<'PY'
import json
rows = []
for line in open("results.jsonl"):
    line = line.strip()
    if line.startswith("{"):
        rows.append(json.loads(line))
hdr = ("status", "tokens", "peak_gib", "cap_gib", "bound", "kernel",
       "alloc_conf", "frames", "size", "wall_s")
print("%-6s %6s %8s %8s %-6s %-6s %-24s %3s %4s %6s"
      % tuple(h[:8] for h in hdr))
for r in rows:
    print("%-6s %6s %8s %8s %-6s %-6s %-24s %3s %4s %6s"
          % tuple(str(r.get(h, "-"))[:24] for h in hdr))
# A row proves a T4 fit when it PASSED under a ceiling no larger than a T4:
# either a big card capped down to one ("simulated"), or a card that is
# already smaller ("card"). An uncapped row proves nothing about a T4.
ok = [r for r in rows
      if r.get("status") == "ok" and r.get("bound") in ("simulated", "card")
      and r.get("kernel") != "free choice"]
print()
if ok:
    best = max(ok, key=lambda r: r.get("tokens", 0))
    print("RICHEST PLAN THAT FITS A T4: %d tokens (%dx%d), peak %.2f GiB "
          "under a %.2f GiB %s ceiling, %s%s"
          % (best["tokens"], best["frames"], best["size"], best["peak_gib"],
             best["cap_gib"], best["bound"],
             "no-math-sdp" if best["no_math_sdp"] else "sdpa",
             " +expandable" if best.get("alloc_conf") else ""))
    print("headroom on a real T4 (14.56 GiB): %.2f GiB"
          % (14.56 - best["peak_gib"],))
else:
    print("NOTHING FIT. Every T4-bounded configuration failed -- read the rows.")
    print("If `bound` is `card` throughout, this ran on an 11 GiB 2080 Ti and")
    print("a failure there does NOT rule out a fit on a 14.56 GiB T4.")
PY
