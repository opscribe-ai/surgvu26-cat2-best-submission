#!/bin/bash
set -uo pipefail
# Validates the SUBMISSION path -- scripts/inference.py, one subprocess per
# case, real /input -> /output layout -- across the 11 public sample cases.
#
#   condor/validate.sh <device>        device is "auto" (GPU node) or "cpu"
#
# Runs inside surgvu26-train.sif; do NOT override PATH (same reason as
# condor/train.sh: the image supplies its own python3).
#
# WHY A LOCAL COPY OF THE CHECKPOINTS. config/perception.json binds them at
# /staging/n/nkalthoff/surgvu26/models/*.pt, which will not exist inside the
# submission image. Copying them into scratch and pointing --models-dir there
# exercises the exact re-rooting the container will do, and proves the
# /staging paths in the config never have to resolve at serving time.
#
# THE NAMES COME FROM THE CONFIG. They used to be hardcoded as tools_v2.pt /
# task_v2.pt, under a comment claiming they were v1's -- so the comment and
# the code already disagreed -- and the shipped config has bound
# tools_resnet50_long.pt since v3. --models-dir re-roots by BASENAME, so this
# script copied two files nobody asked for and inference then looked for a
# third that was not there: FileNotFoundError on every one of the 11 cases.
# It failed loudly, which is the safe direction, but the submission path's own
# validator could not pass at all.
#
# The original intent -- "cannot be pointed at something else by accident" --
# is preserved and strengthened: reading the committed config means this
# validates whatever actually ships, and cannot drift from it again.

echo "host: $(hostname)  start: $(date)"
echo "device argument: ${1:-auto}"
DEVICE="${1:-auto}"

SAMPLE=/staging/groups/bhaskar_opscribe/surgvu/cat2_sample
MODELS_SRC=/staging/n/nkalthoff/surgvu26/models

if [ "$DEVICE" = "cpu" ]; then
    LABELS="cpu_f16_tall cpu_f8_tall cpu_f16_t4"
else
    LABELS="gpu_f16"
fi

# Pre-create every file named in transfer_output_files. HTCondor holds a job
# whose declared outputs are missing at exit, and a job that died early -- an
# unmounted /staging, a missing checkpoint -- must be able to RETRY onto
# another node rather than sit held. An empty `{}` here is unmistakable: the
# scorer refuses a candidates file that is missing cases, so a placeholder can
# never be mistaken for a result.
for label in $LABELS; do
    echo '{}' > "${label}_candidates.json"
    echo '{}' > "${label}_results.json"
done

# ---- fail fast on an execute node that cannot see /staging -----------------
# /staging is not mounted everywhere. Dying here, immediately, lets the retry
# land somewhere it works instead of burning the slot.
if [ ! -d "$SAMPLE" ]; then
    echo "FATAL: $SAMPLE is not visible on this node; nothing to validate"
    exit 42
fi
CKPTS=$(python3 -c "
import json
config = json.load(open('config/perception.json'))
print(' '.join(e['checkpoint_name'] for e in config['experts'].values()))
") || { echo "FATAL: cannot read config/perception.json"; exit 42; }
echo "config binds: $CKPTS"
for name in $CKPTS; do
    [ -r "$MODELS_SRC/$name" ] || {
        echo "FATAL: $MODELS_SRC/$name not readable"; exit 42; }
done

echo "== node =="
nproc
free -g | head -2
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || echo "(no nvidia-smi)"

# ---- local checkpoints, simulating weights baked into the image -----------
mkdir -p models
for name in $CKPTS; do
    cp "$MODELS_SRC/$name" models/ || exit 1
done
echo "== local checkpoints (the config's sha256 must still match) =="
sha256sum models/*.pt

python3 - <<'PY'
import json, torch
config = json.load(open("config/perception.json"))
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| threads", torch.get_num_threads())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0),
          "| capability", torch.cuda.get_device_capability(0))
for role, entry in config["experts"].items():
    print("config binds", role, entry["checkpoint_name"], entry["sha256"])
print("decode", config["decode"])
PY

RC_ALL=0

# run_pass <label> <device> <frames|config> <threads|all>
run_pass () {
    local label="$1" device="$2" frames="$3" threads="$4"
    echo
    echo "================================================================"
    echo "PASS $label  device=$device frames=$frames threads=$threads"
    echo "================================================================"
    local extra=()
    if [ "$frames" != "config" ]; then extra+=(--frames "$frames"); fi
    if [ "$threads" != "all" ]; then
        export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
    else
        unset OMP_NUM_THREADS MKL_NUM_THREADS
    fi
    python3 scripts/validate_cases.py "$SAMPLE" \
        --work-dir ./validate_work \
        --out-prefix "$label" \
        --models-dir "$PWD/models" \
        --device "$device" \
        --label "$label" \
        ${extra[@]+"${extra[@]}"}
    local rc=$?
    echo "PASS $label exit $rc"
    [ "$rc" -ne 0 ] && RC_ALL=1
    return 0
}

if [ "$DEVICE" = "cpu" ]; then
    # The No-GPU deployment instance. Three passes because the lever we may
    # have to pull is `decode.frames`, and because the number of vCPUs the
    # grader gives us is NOT documented -- an 8-thread measurement would be
    # optimistic if the instance has 4.
    run_pass cpu_f16_tall cpu config all
    run_pass cpu_f8_tall  cpu 8      all
    run_pass cpu_f16_t4   cpu config 4
else
    run_pass gpu_f16 auto config all
fi

echo
echo "== outputs =="
ls -la ./*_candidates.json ./*_results.json 2>&1
echo "exit: $RC_ALL  end: $(date)"
exit "$RC_ALL"
