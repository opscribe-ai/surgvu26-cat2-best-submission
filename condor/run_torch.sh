#!/bin/bash
set -uo pipefail
# Run a torch script on CPU inside the TRAINING container, transferring nothing.
#
#   condor/run_torch.sh <script.py> [args...]
#
# WHY THIS EXISTS SEPARATELY FROM condor/verify.sh. That wrapper forces
# `--out <file>` as the first argument and declares that file in
# transfer_output_files, because the checks it runs are expected to fail
# sometimes and a missing declared output turns a clean failure into a HELD
# job. That contract is right for a check and wrong for a producer:
# save_temporal_init.py writes a CHECKPOINT, to /staging, and there is nothing
# to bring home.
#
# The alternative considered and rejected was passing `--out` twice and
# relying on argparse taking the last one. It works, and a reader would have
# to know that to understand what the job does.

echo "host: $(hostname)  start: $(date)"
SCRIPT="${1:?usage: run_torch.sh <script.py> [args...]}"
shift

export TORCH_HOME=/staging/n/nkalthoff/surgvu26/torch_cache
# Same staging guard as every other wrapper: /staging is advertised by nodes
# that do not deliver it, and exit 75 is the retryable code the submit files
# discriminate on.
[ -d /staging/n/nkalthoff ] || { echo "FATAL: /staging not mounted"; exit 75; }

python3 "$SCRIPT" "$@"
RC=$?
echo "exit: $RC  end: $(date)"
exit $RC
