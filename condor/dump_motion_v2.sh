#!/bin/bash
set -uo pipefail
# Runs scripts/dump_motion_v2.py inside surgvu26-train.sif, producing the
# --dump JSON scripts/calibrate_motion_v2.py consumes: a list of
# {"case", "t", "vector"} records built from the real source videos and
# tasks.csv, not from scripts/sample_motion.py's different v1 shape.
#
#   condor/dump_motion_v2.sh [CASES [WINDOWS_PER_CASE [SEED]]]
#
# CASES is forwarded to --cases as-is: an int limit (the default, 2, is a
# smoke run) or a comma-separated list of case ids. See
# condor/dump_motion_v2.sub for how to override these from the
# condor_submit command line for the full 155-case sweep.
#
# Runs inside surgvu26-train.sif (the same image condor/pytest.sh and
# condor/motion_ab.sh use), so `python3` here is the image's interpreter.
# Do NOT set PATH: the image supplies its own environment and overriding it
# hides that interpreter.

echo "host: $(hostname)  start: $(date)"

CASES="${1:-2}"
WINDOWS_PER_CASE="${2:-20}"
SEED="${3:-0}"

VIDEO_ROOT=/staging/groups/bhaskar_opscribe/surgvu/videos/surgvu24
LABELS_ROOT=/staging/groups/bhaskar_opscribe/surgvu/labels_cat2/SURGVU25_train_labels
OUT=motion_v2_dump.json

# Fail here, loudly, rather than partway through the first case's decode.
[ -d "$VIDEO_ROOT" ]  || { echo "FATAL: $VIDEO_ROOT not visible on this node"; exit 75; }
[ -d "$LABELS_ROOT" ] || { echo "FATAL: $LABELS_ROOT not visible on this node"; exit 75; }

echo "video_root=$VIDEO_ROOT"
echo "labels_root=$LABELS_ROOT"
echo "cases=$CASES windows_per_case=$WINDOWS_PER_CASE seed=$SEED"
echo "out=$OUT"

python3 scripts/dump_motion_v2.py \
    --video-root "$VIDEO_ROOT" \
    --labels-root "$LABELS_ROOT" \
    --cases "$CASES" \
    --windows-per-case "$WINDOWS_PER_CASE" \
    --seed "$SEED" \
    --out "$OUT"
RC=$?

[ -f "$OUT" ] && echo "dump bytes: $(stat -c %s "$OUT")"
echo "exit: $RC  end: $(date)"
exit "$RC"
