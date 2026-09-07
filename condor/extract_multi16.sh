#!/bin/bash
set -euo pipefail
# One job = one (case, part), written as a MULTI-BURST shard.
#
#   condor/extract_dense.sh CASE_ID PART VIDEO
#
# Same contract as condor/extract.sh -- same image, same labels, same
# frequency table, same window enumeration -- differing only in which frames
# come out: sixteen 0.27 s bursts on the 2D sample points, rather
# than one burst at its centre (dense) or 30 seconds at 1 fps (sparse).
#
# Output goes to /staging/n/nkalthoff, NOT the group directory. The corpus stays
# shared and read-only; anything this project derives is its own.

CASE_ID="$1"
PART="$2"
VIDEO="$3"

STAGING=/staging/groups/bhaskar_opscribe/surgvu
LABELS="$STAGING/labels_cat2/SURGVU25_train_labels/$CASE_ID"
OUTDIR=/staging/n/nkalthoff/surgvu26/shards_multi16
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

# 16 bursts of 3 frames at 15 fps = 48 frames per window.
#
# THREE, not four: the temporal branch needs what happens BEFORE and AFTER each
# sampled moment, which is exactly t-67ms, t, t+67ms. A fourth frame is a
# richer descriptor and costs 20 GB across the pool, and staging is the binding
# constraint here rather than compute.
#
# WHY 16. The 2D model samples 16 moments spread across the 30 s window; the
# 4-burst pool gave the temporal arms FOUR. Measured tonight, that gap is what
# costs them: doubling frames INSIDE the four bursts bought +0.0026, while the
# distance to the 2D model's coverage is ~0.045. Diversity of moments, not
# count of frames.
#
# So the bursts now land on the same 16 moments the 2D path samples, and each
# carries 0.27 s of real motion at 67 ms spacing. The temporal arms get the 2D
# model's breadth AND local motion -- the first configuration in which they are
# not strictly disadvantaged on coverage.
python3 scripts/extract_multi.py \
    "$CASE_ID" "$PART" "$LABELS" "$VIDEO" "$OUT" config/tool_frequency.json \
    16 0.2 15
RC=$?

[ -f "$OUT" ] && echo "shard bytes: $(stat -c %s "$OUT")"
echo "exit: $RC  end: $(date)"
exit $RC
