#!/bin/bash
# Streams CholecT45.zip's 90,728 PNGs into 512x512 JPEGs on /staging, then
# builds the stage-1 manifest from SSG-VQA's annotations over them.
#
# Runs inside surgvu26-train.sif for the same reason the other build jobs do:
# cv2 and numpy come from the image, not from a pip install on the node.

# UNBUFFER PYTHON. The submit files set stream_output, but that streams the
# FILE -- Python still block-buffers stdout at 4-8KB when it is not a tty, so
# a multi-hour run's progress arrives in lumps long after the fact. pip's
# output appears promptly and ours does not, which reads as 'the job hung
# after the environment check'.
export PYTHONUNBUFFERED=1

set -u
echo "host: $(hostname)  start: $(date)"

WORKERS="${1:-8}"

SSG_DIR=/staging/groups/bhaskar_opscribe/benchmarking_datasets/cholecystectomy_eval/SSG-VQA
FRAMES_ZIP="$SSG_DIR/CholecT45.zip"
SSG_QA="$SSG_DIR/ssg-qa.zip"
OUT_ROOT=/staging/n/nkalthoff/surgvu26/stage1_frames
MANIFEST=/staging/n/nkalthoff/surgvu26/stage1_manifest.jsonl
SPLITS=/staging/n/nkalthoff/surgvu26/stage1_splits.json

[ -f "$FRAMES_ZIP" ] || { echo "FATAL: $FRAMES_ZIP not visible"; exit 75; }
[ -f "$SSG_QA" ]     || { echo "FATAL: $SSG_QA not visible"; exit 75; }

echo "frames_zip=$FRAMES_ZIP"
echo "out_root=$OUT_ROOT  workers=$WORKERS"

umask 002
python3 scripts/convert_stage1_frames.py \
    --frames-zip "$FRAMES_ZIP" --out-root "$OUT_ROOT" --workers "$WORKERS"
RC=$?
[ $RC -eq 0 ] || { echo "conversion failed rc=$RC"; exit $RC; }

python3 scripts/build_stage1_manifest.py \
    --ssg-qa "$SSG_QA" --frames-zip "$FRAMES_ZIP" \
    --frames-root "$OUT_ROOT" \
    --manifest-out "$MANIFEST" --splits-out "$SPLITS"
RC=$?

echo "converted frames: $(find "$OUT_ROOT" -name '*.jpg' 2>/dev/null | wc -l)"
[ -f "$MANIFEST" ] && echo "manifest lines: $(wc -l < "$MANIFEST")"
echo "end: $(date)  rc=$RC"
exit $RC
