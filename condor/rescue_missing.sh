#!/bin/bash
set -euo pipefail
# Find (case, part) pairs a pool is MISSING, and re-queue exactly those.
#
#   condor/rescue_missing.sh <pool_dir> <cases.txt> <submit.sub> [--submit]
#
# e.g. condor/rescue_missing.sh /staging/n/nkalthoff/surgvu26/shards_multi16 \
#          condor/cases.txt condor/extract_multi16.sub --submit
#
# WHY THIS EXISTS. A shard can go missing for two completely different reasons
# and only one of them is a fault:
#
#   the pair legitimately holds no task segments in that part, so there is
#   nothing to extract and the job says so on stderr and exits 1
#
#   the job never got a working staging mount, exited 75 every time, and
#   EXHAUSTED its retry budget -- so the pair is simply absent
#
# Measured on cluster 9659991: 38 of the first kind, 8 of the second. The
# second kind is SILENT -- nothing fails, no job is held, the queue drains to
# zero and the pool is quietly short eight (case, part) pairs. A pool missing
# windows its twin has turns every downstream comparison into a measurement of
# the dataset rather than of the model, which is the confound
# scripts/verify_multi_pool.py exists to DETECT; this is what FIXES it, so
# detection is not left as an exercise.
#
# HOW THOSE EIGHT ACTUALLY RESOLVED, recorded because the answer is not the
# one the paragraph above implies. Re-run on working nodes, all eight reported
# "no windows" and exited 1: those parts genuinely hold no task segments, and
# the pool was always going to be 235 shards. The mount fault had killed them
# before they could reach the point of SAYING so, which is exactly why the two
# kinds of absence cannot be told apart without re-running -- an exit-75 pair
# is not "lost data", it is "unknown", and this script converts unknown into
# one of the two answers. Final accounting: 235 present + 46 no-windows = 281,
# and 46 is the number the submit file predicted independently.
#
# The two are told apart by the job's own stderr, not by guessing: a pair whose
# log says "no windows" is expected to be absent and is left alone.

POOL="${1:?usage: rescue_missing.sh <pool_dir> <cases.txt> <submit.sub> [--submit]}"
CASES="${2:?}"
SUB="${3:?}"
DO_SUBMIT="${4:-}"

[ -d "$POOL" ]   || { echo "FATAL: pool $POOL does not exist"; exit 1; }
[ -f "$CASES" ]  || { echo "FATAL: case list $CASES does not exist"; exit 1; }
[ -f "$SUB" ]    || { echo "FATAL: submit file $SUB does not exist"; exit 1; }

# THE POOL IS NOT MISSING WHAT IT HAS NOT WRITTEN YET. Run against a live
# extraction this reports every unstarted pair as an unexplained absence --
# measured, 132 "missing" while 150 jobs were still queued -- and submitting
# that would duplicate the entire run. The distinction this script exists to
# draw only means anything once the queue has drained.
EXEC="$(awk -F= '/^executable/ {gsub(/ /,"",$2); print $2}' "$SUB" | tail -1)"
INFLIGHT=0
if [ -n "$EXEC" ]; then
    INFLIGHT="$(condor_q -constraint "regexp(\"$(basename "$EXEC")\", Cmd)" \
        -af ClusterId 2>/dev/null | wc -l)"
fi
if [ "$INFLIGHT" -gt 0 ]; then
    echo "REFUSING: $INFLIGHT job(s) running $(basename "$EXEC") are still in"
    echo "the queue. Everything they have not written yet would be counted as"
    echo "missing and re-queued on top of itself. Wait for the queue to drain:"
    echo "  condor_q -totals"
    exit 2
fi

OUT=condor/generated/cases_rescue.txt
mkdir -p condor/generated

# WRITE TO A PRIVATE TEMP AND MOVE IT INTO PLACE. Two overlapping runs of this
# script -- which happened, one from a background drain-watcher and one by
# hand -- both truncate and both append to the same fixed path, and the result
# interleaves: the list came out with case_112 listed TWICE, so nine jobs were
# submitted for eight pairs and two of them raced to write the same shard file.
# Nothing was corrupted this time because both exited before writing, but a
# rescue tool that can double-submit a writer is a rescue tool that can corrupt
# the pool it is repairing.
#
# mv within one directory is atomic, so a concurrent run either sees the old
# complete file or the new complete file, never a half-written one.
TMP="$(mktemp "${OUT}.XXXXXX")"
trap 'rm -f "$TMP"' EXIT

missing=0
expected_absent=0
present=0

while IFS=, read -r case_id part video; do
    case_id="$(echo "$case_id" | tr -d '[:space:]')"
    part="$(echo "$part" | tr -d '[:space:]')"
    video="$(echo "$video" | tr -d '[:space:]')"
    [ -z "$case_id" ] && continue

    if [ -f "$POOL/${case_id}_part${part}.npz" ]; then
        present=$((present + 1))
        continue
    fi

    # Absent. Which kind? The job's own stderr is the evidence -- a pair with
    # no task segments in this part said so when it ran.
    log="logs/$(basename "$POOL" | sed 's/^shards_//')_${case_id}_part${part}.err"
    if [ -f "$log" ] && grep -q "no windows" "$log" 2>/dev/null; then
        expected_absent=$((expected_absent + 1))
        continue
    fi

    missing=$((missing + 1))
    echo "${case_id}, ${part}, ${video}" >> "$TMP"
done < "$CASES"

# BELT AND BRACES: dedupe regardless of how the file got written. The atomic
# move above prevents the interleave that caused it, but a list that can name
# the same (case, part) twice submits two jobs writing one path, and that is
# worth making impossible rather than merely unlikely.
before=$(wc -l < "$TMP")
sort -u "$TMP" -o "$TMP"
after=$(wc -l < "$TMP")
if [ "$before" -ne "$after" ]; then
    echo "WARNING: dropped $((before - after)) duplicate entrie(s) -- two jobs"
    echo "         for one (case, part) would race to write the same shard."
fi
missing="$after"
mv "$TMP" "$OUT"
trap - EXIT

echo "pool $POOL"
echo "  present:                 $present"
echo "  absent, no task segments: $expected_absent   (expected, left alone)"
echo "  absent, UNEXPLAINED:      $missing   -> $OUT"

if [ "$missing" -eq 0 ]; then
    echo "nothing to rescue."
    rm -f "$OUT"
    exit 0
fi

echo
cat "$OUT"
echo

if [ "$DO_SUBMIT" != "--submit" ]; then
    echo "DRY RUN. Re-run with --submit to queue these, or:"
    echo "  condor_submit $SUB cases=$OUT"
    exit 0
fi

# 8GB and a fresh retry budget. These already failed five times at 4GB, so
# repeating the identical request is the definition of expecting a different
# result from the same input -- and memory is the one variable that is free to
# change here without altering what the shard CONTAINS.
condor_submit "$SUB" cases="$OUT" memory=8192
