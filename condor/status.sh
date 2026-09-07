#!/bin/bash
# One picture of every v4 arm: what is running, and its latest epoch.
#
#   condor/status.sh
#
# WHY THIS IS A SCRIPT. Reading logs by hand went wrong twice tonight. Jobs
# submitted before stream_output was enabled have NO .out file until they
# finish, so `grep epoch logs/...` on them returns nothing and looks exactly
# like a job that has not reached its first epoch -- a silent empty result
# where the honest answer is "that file does not exist yet". This falls back
# to condor_tail, which reads a running job's stdout either way.

cd "$(dirname "$0")/.." || exit 1
printf "%-10s %-14s %-4s %s\n" CLUSTER ARM ST "LATEST"
for row in $(condor_q -af:jr ClusterId JobStatus Arguments 2>/dev/null \
             | grep -E "train_temporal|train_tools_3d|dump_temporal" \
             | awk '{print $2":"$3}'); do
    cluster="${row%%:*}"
    state="${row##*:}"
    args=$(condor_q "$cluster" -af Arguments 2>/dev/null)
    arm=$(echo "$args" | grep -oE '\-\-out [^ ]+' | head -1 \
          | sed 's|.*/||; s|\.pt$||; s|\.npz$||; s|^tools_||')
    [ -z "$arm" ] && arm="?"
    case "$state" in 1) st=idle;; 2) st=run;; 5) st=HELD;; *) st="$state";; esac

    line=""
    log=$(ls -t logs/temporal_*_"$cluster".out 2>/dev/null | head -1)
    if [ -n "$log" ] && [ -s "$log" ]; then
        line=$(grep -E "^epoch|^HONEST|^arm " "$log" | tail -1)
    fi
    # No log file is NOT the same as no progress: pre-streaming jobs only
    # write theirs at exit. Ask the running job directly.
    if [ -z "$line" ] && [ "$state" = "2" ]; then
        line=$(timeout 25 condor_tail "$cluster" 2>/dev/null \
               | grep -E "^epoch|^HONEST" | tail -1)
        [ -n "$line" ] && line="$line  (via condor_tail)"
    fi
    [ -z "$line" ] && line="-- no epoch yet --"
    printf "%-10s %-14s %-4s %s\n" "$cluster" "${arm:0:14}" "$st" "$line"
done

held=$(condor_q -held -af ClusterId 2>/dev/null | wc -l)
[ "$held" -gt 0 ] && echo "HELD: $held  -> condor/release_held.sh 8192"
exit 0
