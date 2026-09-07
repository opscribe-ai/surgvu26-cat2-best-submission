#!/bin/bash
# Converts the OpScribe GI corpus's 87k referenced images to 512x512 JPEGs and
# writes a SurgVU-shaped stage-1 manifest. Images are already on /staging; only
# the annotations came from HuggingFace.

# UNBUFFER PYTHON. The submit files set stream_output, but that streams the
# FILE -- Python still block-buffers stdout at 4-8KB when it is not a tty, so
# a multi-hour run's progress arrives in lumps long after the fact. pip's
# output appears promptly and ours does not, which reads as 'the job hung
# after the environment check'.
export PYTHONUNBUFFERED=1

set -u
echo "host: $(hostname)  start: $(date)"
OUT_ROOT=/staging/n/nkalthoff/surgvu26/gi_frames
MANIFEST=/staging/n/nkalthoff/surgvu26/gi_stage1_manifest.jsonl
SPLITS=/staging/n/nkalthoff/surgvu26/gi_stage1_splits.json
umask 002
python3 scripts/build_gi_stage1.py \
    --gi-json /staging/n/nkalthoff/surgvu26/gi_train.json \
              /staging/n/nkalthoff/surgvu26/gi_val.json \
    --out-root "$OUT_ROOT" --manifest-out "$MANIFEST" --splits-out "$SPLITS" --workers 32
RC=$?
echo "converted: $(find "$OUT_ROOT" -name '*.jpg' 2>/dev/null | wc -l)"
[ -f "$MANIFEST" ] && echo "manifest lines: $(wc -l < "$MANIFEST")"
echo "end: $(date)  rc=$RC"
exit $RC
