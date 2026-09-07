#!/bin/bash
set -uo pipefail
# Runs scripts/detect_sample_report.py inside surgvu26-train.sif -- the YOLOv5
# tool detector against the 11 public Cat 2 sample clips, decoded through the
# real serving decoder (surgvu.perceive.decode_clip).
#
# Modelled closely on condor/detect_smoke.sh; see that file for the base
# rationale (why deps are pip-installed into a scratch --target dir rather
# than baked into the image, why PATH is left alone). surgvu26-train.sif (the
# pytorch/pytorch base image + opencv-python-headless + PyYAML) does NOT
# provide what yolov5's models/common.py needs at import time: models/
# common.py itself imports pandas and requests directly, and its own import
# chain (utils/dataloaders.py -> utils/plots.py -> utils/general.py) pulls in
# tqdm, matplotlib and seaborn as hard imports before DetectMultiBackend can
# even be defined.

echo "host: $(hostname)  start: $(date)"

DEPS="$(pwd)/.detect_sample_deps"
mkdir -p "$DEPS"
python3 -m pip install --no-cache-dir --target "$DEPS" \
    pandas requests tqdm matplotlib seaborn Pillow
PIP_RC=$?
if [ "$PIP_RC" -ne 0 ]; then
    echo "FATAL: pip install into $DEPS failed with exit $PIP_RC"
    exit "$PIP_RC"
fi
export PYTHONPATH="$DEPS${PYTHONPATH:+:$PYTHONPATH}"

python3 scripts/detect_sample_report.py \
    --sample-root /staging/groups/bhaskar_opscribe/surgvu/cat2_sample \
    --weights /staging/groups/bhaskar_opscribe/surgvu_yolo_detector/best.pt \
    --yolov5-dir /staging/groups/bhaskar_opscribe/surgvu_yolo_detector/yolov5 \
    --candidates baselines/shipped_candidates.json \
    --conf-floor 0.25 \
    --out-json detect_sample_results.json
RC=$?

echo "exit: $RC  end: $(date)"
exit "$RC"
