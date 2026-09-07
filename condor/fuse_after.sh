#!/bin/bash
set -uo pipefail
# Waits for a dump to exist, then runs the fusion test that decides shipping.
#
#   condor/fuse_after.sh <dump.npz> <script.py> <out.json> [extra args...]
#
# WHY WAIT ON THE FILE HERE, when condor/gate_cluster.sh deliberately does not.
# A training checkpoint is rewritten on every improving epoch, so "the file
# stopped changing" cannot be told apart from "the model stopped improving".
# A DUMP is different: it is written once, at the end, by a job that has
# already finished. Its existence is an unambiguous signal.
#
# The point of automating this is that the fusion test -- not the head-to-head
# score -- is what decides whether an arm ships, and it was the one step still
# waiting on a human to notice a file had appeared.

DUMP="${1:?usage: fuse_after.sh <dump.npz> <script.py> <out.json> [args...]}"
SCRIPT="${2:?}"
OUT="${3:?}"
shift 3

echo "host: $(hostname)  start: $(date)"
echo '{}' > "$OUT"          # pre-create: a missing transfer_output file HOLDS

elapsed=0
until [ -f "${DUMP}.json" ]; do
    if [ "$elapsed" -ge 21600 ]; then
        echo "FATAL: ${DUMP}.json never appeared after 6 hours"
        exit 1
    fi
    sleep 120
    elapsed=$((elapsed + 120))
done
echo "dump appeared after ${elapsed}s; fusing"

python3 "$SCRIPT" --out "$OUT" "$@"
RC=$?
echo "exit: $RC  end: $(date)"
exit $RC
