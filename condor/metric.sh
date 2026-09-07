#!/bin/bash
set -uo pipefail
# Runs any CPU-only script that needs the OFFICIAL metric (roberta-large).
#
#   condor/metric.sh <script.py> <out.json> [extra args...]
#
# Generalised out of condor/hedge.sh, which was itself a copy of
# condor/answer_form.sh. Three near-identical wrappers was two too many: the
# symlink bug below was fixed in one of them and would have had to be fixed
# again in the next copy.
#
# universe=vanilla, not container, like its ancestors: this runs
# `apptainer exec` on the extract .sif, which cannot be nested inside a
# container-universe job.
#
# roberta-large loads from the /staging HF cache on CPU and is SLOW to start.
# Several minutes before the first number appears is normal, not a hang.

export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin
echo "host: $(hostname)  start: $(date)"

SCRIPT="${1:?usage: metric.sh <script.py> <out.json> [args...]}"
OUT="${2:?usage: metric.sh <script.py> <out.json> [args...]}"
shift 2

SIF=/staging/n/nkalthoff/surgvu26/surgvu26-extract.sif
PY=/staging/n/nkalthoff/surgvu26/env/bin/python3
export HF_HOME=/staging/n/nkalthoff/surgvu26/hf_cache

# Pre-create the declared output: HTCondor HOLDS a job whose transfer_output
# file is missing at exit, which turns a retryable node fault into a stuck job.
echo '{}' > "$OUT"

# NOT "$PY". The venv's bin/python3 is a SYMLINK to /usr/local/bin/python3,
# which exists only inside the .sif -- `[ -e ]` follows symlinks, so testing it
# from the host resolves to a path that is not there and fails a job that would
# have run perfectly. Cluster 9652325 died exactly this way on e2603. Check the
# venv DIRECTORY, which is real on both sides of the container boundary.
for path in "$SIF" "$(dirname "$(dirname "$PY")")" "$HF_HOME" "$SCRIPT"; do
    [ -e "$path" ] || { echo "FATAL: $path not visible on $(hostname)"; exit 42; }
done

apptainer exec -B /staging --env HF_HOME="$HF_HOME" "$SIF" \
    "$PY" "$SCRIPT" --out "$OUT" "$@"
RC=$?

echo "exit: $RC  end: $(date)"
exit "$RC"
