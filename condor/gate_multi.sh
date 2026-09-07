#!/bin/bash
set -uo pipefail
# Blocks until the multi-burst shard pool has stopped growing, then exits 0.
#
#   condor/gate_multi.sh [MIN_SHARDS] [STABLE_CHECKS] [SLEEP_SECONDS]
#
# WHY A JOB AND NOT A SCRIPT ON THE LOGIN NODE. The training arms must start
# when extraction finishes, whether or not anyone is connected -- the SSH
# session dropped at 23:58 last night and took eight hours of supervision with
# it. A DAG node runs under DAGMan on the submit machine and does not care.
#
# WHY "STOPPED GROWING" RATHER THAN A TARGET COUNT. 46 of the 281 (case, part)
# pairs legitimately hold no task segments and produce no shard, so the pool
# tops out at 235 rather than 281 -- and if a handful of jobs fail, a gate that
# waits for an exact count waits forever. Stability plus a floor catches both
# the healthy case and the partial one, and the floor is what stops it from
# declaring victory over an empty directory.

POOL=/staging/n/nkalthoff/surgvu26/shards_multi
MIN_SHARDS="${1:-200}"
STABLE_CHECKS="${2:-6}"
SLEEP_SECONDS="${3:-120}"

echo "host: $(hostname)  start: $(date)"
[ -d /staging/n/nkalthoff ] || { echo "FATAL: /staging not mounted"; exit 75; }

last=-1
stable=0
elapsed=0
while true; do
    count=$(find "$POOL" -name '*.npz' 2>/dev/null | wc -l)
    if [ "$count" -eq "$last" ]; then
        stable=$((stable + 1))
    else
        stable=0
    fi
    echo "$(date +%H:%M:%S) shards=$count stable=$stable/$STABLE_CHECKS"
    if [ "$count" -ge "$MIN_SHARDS" ] && [ "$stable" -ge "$STABLE_CHECKS" ]; then
        echo "pool settled at $count shards after ${elapsed}s"
        exit 0
    fi
    # A hard ceiling so a stalled extraction fails the DAG instead of holding
    # a slot until the walltime limit. 8 hours is well past the 3.5 h the
    # dense pool took.
    if [ "$elapsed" -ge 28800 ]; then
        echo "FATAL: pool still at $count shards after 8 hours; not settling."
        exit 1
    fi
    last="$count"
    sleep "$SLEEP_SECONDS"
    elapsed=$((elapsed + SLEEP_SECONDS))
done
