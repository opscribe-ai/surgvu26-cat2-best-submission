#!/bin/bash
set -uo pipefail
# Runs scripts/hedge_tradeoff.py with the OFFICIAL metric.
#
#   condor/hedge.sh <out.json>
#
# Same shape as condor/answer_form.sh, and for the same reasons: vanilla
# universe running `apptainer exec` on the extract .sif, roberta-large loaded
# from the /staging HF cache on CPU. Several minutes before the first number
# appears is normal, not a hang.
#
# The validation dump is read straight off /staging rather than transferred --
# it is 100 MB+ and every execute node this runs on has the mount.

export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin
echo "host: $(hostname)  start: $(date)"

OUT="${1:-hedge_results.json}"

SIF=/staging/n/nkalthoff/surgvu26/surgvu26-extract.sif
PY=/staging/n/nkalthoff/surgvu26/env/bin/python3
DUMP=/staging/n/nkalthoff/surgvu26/v2/frame_probs_resnetlong_val.npz
export HF_HOME=/staging/n/nkalthoff/surgvu26/hf_cache

# Pre-create the declared output: HTCondor HOLDS a job whose transfer_output
# file is missing at exit, which turns a retryable node fault into a stuck job.
echo '{}' > "$OUT"

# NOT "$PY". The venv's bin/python3 is a SYMLINK to /usr/local/bin/python3,
# which exists only inside the .sif -- `[ -e ]` follows symlinks, so testing it
# from the host resolves to a path that is not there and fails a job that would
# have run perfectly. Cluster 9652325 died exactly this way on e2603. Check the
# venv DIRECTORY, which is real on both sides of the container boundary.
# condor/answer_form.sh omits the same check for the same reason.
for path in "$SIF" "$(dirname "$(dirname "$PY")")" "$DUMP" "$HF_HOME"; do
    [ -e "$path" ] || { echo "FATAL: $path not visible on $(hostname)"; exit 42; }
done

apptainer exec -B /staging --env HF_HOME="$HF_HOME" "$SIF" \
    "$PY" scripts/hedge_tradeoff.py --dump "$DUMP" --out "$OUT"
RC=$?

echo "exit: $RC  end: $(date)"
exit "$RC"
