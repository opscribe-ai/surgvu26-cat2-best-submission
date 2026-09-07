#!/bin/bash
set -euo pipefail
# One job = one (case, part), written as a MULTI-BURST shard.
#
#   condor/extract_dense.sh CASE_ID PART VIDEO
#
# Same contract as condor/extract.sh -- same image, same labels, same
# frequency table, same window enumeration -- differing only in which frames
# come out: four 0.53 s bursts spread across the window at 15 fps, rather
# than one burst at its centre (dense) or 30 seconds at 1 fps (sparse).
#
# Output goes to /staging/n/nkalthoff, NOT the group directory. The corpus stays
# shared and read-only; anything this project derives is its own.

CASE_ID="$1"
PART="$2"
VIDEO="$3"

STAGING=/staging/groups/bhaskar_opscribe/surgvu
LABELS="$STAGING/labels_cat2/SURGVU25_train_labels/$CASE_ID"
OUTDIR=/staging/n/nkalthoff/surgvu26/shards_multi
OUT="$OUTDIR/${CASE_ID}_part${PART}.npz"

echo "host: $(hostname)  start: $(date)"
echo "case=$CASE_ID part=$PART"
echo "video=$VIDEO"
echo "out=$OUT"

# /staging is not mounted everywhere, and the failure surfaces as a missing
# input rather than as a mount error. Exit 75 so a retry lands elsewhere.
[ -d "$STAGING" ] || { echo "FATAL: $STAGING not visible on this node"; exit 75; }
[ -f "$VIDEO" ]  || { echo "FATAL: video not readable: $VIDEO"; exit 1; }
[ -d "$LABELS" ] || { echo "FATAL: labels not readable: $LABELS"; exit 1; }
[ -f config/tool_frequency.json ] || {
    echo "FATAL: config/tool_frequency.json was not transferred. Without it"
    echo "stratify falls back to case-local rarity, which is wrong and"
    echo "invisible -- and would silently give this shard a different window"
    echo "set from its sparse twin, destroying the comparison."
    exit 1
}

umask 002
mkdir -p "$OUTDIR"

python3 scripts/extract_multi.py \
    "$CASE_ID" "$PART" "$LABELS" "$VIDEO" "$OUT" config/tool_frequency.json
RC=$?

[ -f "$OUT" ] && echo "shard bytes: $(stat -c %s "$OUT")"
echo "exit: $RC  end: $(date)"
exit $RC
