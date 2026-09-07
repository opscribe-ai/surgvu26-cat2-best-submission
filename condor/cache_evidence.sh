#!/bin/bash

# UNBUFFER PYTHON. The submit files set stream_output, but that streams the
# FILE -- Python still block-buffers stdout at 4-8KB when it is not a tty, so
# a multi-hour run's progress arrives in lumps long after the fact. pip's
# output appears promptly and ours does not, which reads as 'the job hung
# after the environment check'.
export PYTHONUNBUFFERED=1

set -uo pipefail
# Runs scripts/cache_evidence.py inside surgvu26-train.sif on a GPU execute
# node: the SAME perception stack scripts/inference.py runs at serving time
# (surgvu.perceive's CNN tool/task heads, surgvu.detect.Detector,
# surgvu.variant.VariantHead, surgvu.motion.motion_record_v2,
# surgvu.agreement.agreement_record), run once per distinct window of
# /staging/n/nkalthoff/surgvu26/qa_frames_manifest.jsonl and cached to
# /staging/n/nkalthoff/surgvu26/evidence_cache.jsonl -- see
# scripts/cache_evidence.py's own module docstring for the full design and
# why it must never read tools.csv/tasks.csv (the ground truth) to do this.
#
#   condor_submit condor/cache_evidence.sub
#   condor_submit condor/cache_evidence.sub args="--limit 20"   # smoke test
#
# DEPENDENCIES, reusing condor/detect_sample.sh's / condor/variant_sample.sh's
# proven recipe rather than re-deriving one: surgvu26-train.sif ships
# torch/torchvision/opencv-python-headless/PyYAML (containers/
# surgvu26-train.def) -- enough for surgvu.perceive/surgvu.variant -- but NOT
# what yolov5's models/common.py needs at import time: models/common.py
# itself imports pandas and requests directly, and its own import chain
# (utils/dataloaders.py -> utils/plots.py -> utils/general.py) pulls in tqdm,
# matplotlib and seaborn as hard imports before DetectMultiBackend can even
# be defined. Installed into a scratch --target dir, never touching the
# image, exactly like condor/detect_sample.sh/condor/variant_sample.sh
# already do for this identical detector.

echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 -c "import torch; print('cuda', torch.cuda.is_available())"
umask 002

# STAGING MOUNT GUARD, identical to condor/build_qa_frames.sh / condor/
# train_vlm.sh. Every input this job reads (the manifest, the video corpus,
# config/splits_v2.json, the yolo/variant checkpoints) and everything it
# writes (the evidence cache, the drop-count report) lives under /staging,
# so a job without the mount cannot do anything useful -- exit fast and
# non-zero so HTCondor reschedules onto another node rather than failing
# confusingly deeper in.
if ! mkdir -p /staging/n/nkalthoff/surgvu26 2>/dev/null; then
    echo "FATAL: /staging is not mounted on $(hostname). +WantStagingMount" \
         "did not take effect; exiting fast so this reschedules elsewhere." >&2
    exit 75
fi

DEPS="$(pwd)/.cache_evidence_deps"
mkdir -p "$DEPS"
python3 -m pip install --no-cache-dir --target "$DEPS" \
    pandas requests tqdm matplotlib seaborn Pillow
PIP_RC=$?
if [ "$PIP_RC" -ne 0 ]; then
    echo "FATAL: pip install into $DEPS failed with exit $PIP_RC"
    exit "$PIP_RC"
fi
export PYTHONPATH="$DEPS${PYTHONPATH:+:$PYTHONPATH}"

# "$@" forwards whatever condor/cache_evidence.sub's `arguments` line passed
# (the `args` submit-time macro) straight to scripts/cache_evidence.py's own
# argparse -- every default (manifest/splits/video-root/perception-config/
# yolo+variant weights/--out/--report-out) already points at the right
# /staging path, so `args=""` (the default) is itself a complete,
# correct invocation.
python3 scripts/cache_evidence.py "$@"
RC=$?

OUT=/staging/n/nkalthoff/surgvu26/evidence_cache.jsonl
[ -f "$OUT" ] && echo "evidence cache bytes: $(stat -c %s "$OUT")  lines: $(wc -l < "$OUT")"
echo "exit: $RC  end: $(date)"
exit "$RC"
