#!/bin/bash
# Blocks until every listed cluster has left the queue, then exits 0.
#
#   condor/gate_cluster.sh <cluster> [<cluster> ...]
#
# SCHEDULER UNIVERSE, deliberately. This has to run on the submit machine
# because it calls condor_q, which an execute node cannot. DAGMan itself runs
# this way, so the mechanism is not exotic.
#
# WHY NOT WAIT ON THE CHECKPOINT FILE, like condor/gate_multi.sh waits on the
# shard pool. A training checkpoint is REWRITTEN on every improving epoch and
# then not touched again until the next improvement, so "the file stopped
# changing" is indistinguishable from "the model stopped improving" -- and an
# arm that plateaus for two epochs would be declared finished and scored from
# a half-trained checkpoint. Queue membership is the real signal.

set -uo pipefail
echo "gate_cluster: waiting on $* -- start $(date)"
elapsed=0
while true; do
    left=0
    for cluster in "$@"; do
        n=$(condor_q "$cluster" -af ClusterId 2>/dev/null | wc -l)
        left=$((left + n))
    done
    if [ "$left" -eq 0 ]; then
        echo "gate_cluster: all clusters gone after ${elapsed}s"
        # LEAVING THE QUEUE IS NOT THE SAME AS SUCCEEDING. A job that was
        # condor_rm'd -- because its recipe was wrong and it was being
        # restarted -- also leaves the queue, and the gate used to release the
        # dump anyway. That happened at 02:47: three task arms were removed
        # mid-run, their evaluators fired, and one produced a plausible dump of
        # an aborted run's epoch-0 checkpoint under the name the RESTARTED arm
        # would later use. It would have been read as that arm's result.
        #
        # So the gate now demands ExitCode 0 from history. A removed job has no
        # ExitCode and fails this test, which fails the DAG node and skips the
        # dump -- the correct outcome, since there is nothing worth scoring.
        for cluster in "$@"; do
            code=$(condor_history "$cluster" -limit 1 -af ExitCode 2>/dev/null \
                   | head -1)
            if [ "$code" != "0" ]; then
                echo "FATAL: cluster $cluster left the queue with ExitCode" \
                     "'${code:-none}' -- removed or failed, not completed." \
                     "Refusing to release the evaluation."
                exit 1
            fi
        done
        echo "gate_cluster: all clusters completed with ExitCode 0"
        exit 0
    fi
    echo "$(date +%H:%M:%S) still queued: $left"
    sleep 180
    elapsed=$((elapsed + 180))
    # 14 hours. Longer than any arm should take; past that something is stuck
    # and the DAG should fail visibly rather than hold a slot until the pool
    # reclaims it.
    if [ "$elapsed" -ge 50400 ]; then
        echo "FATAL: still $left queued after 14 hours"
        exit 1
    fi
done
