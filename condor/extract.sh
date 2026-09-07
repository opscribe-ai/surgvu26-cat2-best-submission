#!/bin/bash
set -euo pipefail
# One job = one (case, part). Timestamps reset at the part boundary, so a
# case's two videos are two independent jobs writing two distinct shards.
#
# Runs inside surgvu26-extract.sif (python:3.11-slim + numpy/cv2/PyYAML), so
# `python3` here is the image's interpreter. Do NOT set PATH: the image
# supplies its own environment and overriding it hides that interpreter.

CASE_ID="$1"
PART="$2"
VIDEO="$3"

STAGING=/staging/groups/bhaskar_opscribe/surgvu
LABELS="$STAGING/labels_cat2/SURGVU25_train_labels/$CASE_ID"
OUTDIR="$STAGING/shards"
OUT="$OUTDIR/${CASE_ID}_part${PART}.npz"

echo "host: $(hostname)  start: $(date)"
echo "case=$CASE_ID part=$PART"
echo "video=$VIDEO"
echo "out=$OUT"

# Fail here, loudly, rather than three minutes into a decode.
[ -f "$VIDEO" ]  || { echo "FATAL: video not readable: $VIDEO"; exit 1; }
[ -d "$LABELS" ] || { echo "FATAL: labels not readable: $LABELS"; exit 1; }
[ -f config/tool_frequency.json ] || {
    echo "FATAL: config/tool_frequency.json was not transferred. Without it"
    echo "stratify would fall back to case-local rarity, which is wrong and"
    echo "invisible. Run scripts/build_tool_frequency.py and resubmit."
    exit 1
}

umask 002
mkdir -p "$OUTDIR"

python3 scripts/extract_case.py \
    "$CASE_ID" "$PART" "$LABELS" "$VIDEO" "$OUT" config/tool_frequency.json
RC=$?

[ -f "$OUT" ] && echo "shard bytes: $(stat -c %s "$OUT")"
echo "exit: $RC  end: $(date)"
exit $RC
