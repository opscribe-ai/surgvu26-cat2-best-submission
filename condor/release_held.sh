#!/bin/bash
# Raise memory on anything HELD for exceeding its cgroup limit, then release.
#
#   condor/release_held.sh [FLOOR_MB]
#
# The multi-burst extraction runs at 4 GB because that matches twenty times as
# many slots as 8 GB does (14,596 against 735, counted live). The trade is that
# the largest jobs -- a full 160-window case -- cross it. Raising the request
# for all 281 to cover the few would cost far more throughput than rescuing the
# few costs.
#
# ESCALATES FROM THE MEASUREMENT, NOT FROM A CONSTANT, and that is a bug fix.
# The first version of this script released everything at a flat 8192 MB. On
# cluster 9659991 five 160-window cases measured 9766 MB, so each one was held
# for JobOutOfResources, released at 8192, killed, held again -- proc 6 went
# round six times. A rescue that always offers the same amount cannot rescue a
# job that needs more than that amount; it just keeps the job alive in a loop
# that looks like progress. Worse, running this script on a timer FEEDS the
# loop, which is exactly what happened here.
#
# So the new request is derived from what the job actually used, with headroom,
# and it can only go UP. The floor argument is a minimum, not the answer.
#
# WHY 1.5x. MemoryUsage is the peak the job reached BEFORE it was killed, so it
# is a lower bound on what the job wanted, not the true peak. A 160-window case
# at 48 frames holds 160 x 48 x 512x512x3 = 6.04 GB of raw frames before
# write_shard encodes any of them; measured peak 9766 MB against a true need
# somewhere above that is consistent with 1.5x being about right and 1.1x not
# being enough.
#
# Idempotent, and safe to run on a queue with nothing held.

FLOOR="${1:-8192}"
CAP=32768          # Beyond this it is a leak, not a big case. Leave it held.

held=$(condor_q -held -af:jr ClusterId ProcId 2>/dev/null | awk '{print $2"."$3}')
[ -z "$held" ] && { echo "nothing held"; exit 0; }

for id in $held; do
    used=$(condor_q "$id" -af MemoryUsage 2>/dev/null | head -1)
    want=$(condor_q "$id" -af RequestMemory 2>/dev/null | head -1)
    case "$used" in ''|*[!0-9]*) used=0 ;; esac
    case "$want" in ''|*[!0-9]*) want=0 ;; esac

    # 1.5x what it was measured using, floored, and never below what it was
    # already given -- a rescue must never hand a job LESS than it just died at.
    grant=$(( used * 3 / 2 ))
    [ "$grant" -lt "$FLOOR" ] && grant="$FLOOR"
    if [ "$grant" -le "$want" ]; then
        grant=$(( want * 3 / 2 ))
    fi

    if [ "$grant" -gt "$CAP" ]; then
        echo "$id: would need ${grant}MB, over the ${CAP}MB cap -- LEFT HELD."
        echo "    A job wanting this much is a leak or a pathological case;"
        echo "    releasing it again would just burn another slot."
        continue
    fi

    if condor_qedit "$id" RequestMemory "$grant" >/dev/null 2>&1; then
        echo "$id: used ${used}MB at ${want}MB -> releasing at ${grant}MB"
        condor_release "$id" >/dev/null 2>&1
    else
        echo "$id: could not edit RequestMemory, left held"
    fi
done
