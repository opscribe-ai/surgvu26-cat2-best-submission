#!/bin/bash
set -uo pipefail
# Runs a CPU-only check inside the TRAINING container, which has torchvision.
#
#   condor/verify.sh <script.py> <out.json> [args...]
#
# container universe, so no `apptainer exec` here -- the image is already the
# job's root. That is the difference from condor/metric.sh, which nests a
# container it cannot nest and therefore runs vanilla.

echo "host: $(hostname)  start: $(date)"
SCRIPT="${1:?usage: verify.sh <script.py> <out.json> [args...]}"
OUT="${2:?usage: verify.sh <script.py> <out.json> [args...]}"
shift 2

# Pre-create the declared output: HTCondor HOLDS a job whose transfer_output
# file is missing at exit, which turns a clean failure into a stuck job -- and
# these checks are EXPECTED to fail sometimes. That is what they are for.
echo '{}' > "$OUT"

export TORCH_HOME=/staging/n/nkalthoff/surgvu26/torch_cache
# pytest is not in the training image and cannot be added to a read-only .sif,
# so it lives in a staging directory on PYTHONPATH. The image has torchvision
# and the scoring venv does not, so this is the only environment where the
# torch-dependent tests can run at all.
export PYTHONPATH="/staging/n/nkalthoff/surgvu26/testpkgs:${PYTHONPATH:-}"
[ -d /staging/n/nkalthoff ] || { echo "FATAL: /staging not mounted"; exit 75; }

python3 "$SCRIPT" --out "$OUT" "$@"
RC=$?
echo "exit: $RC  end: $(date)"
exit $RC
