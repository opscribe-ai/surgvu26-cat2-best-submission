#!/bin/bash
set -uo pipefail
# Runs scripts/variant_sample_report.py inside surgvu26-train.sif -- the
# trained Large-vs-Mega needle-driver head (surgvu.variant.VariantHead)
# against the 11 public Cat 2 sample clips, cropping to the YOLOv5 detector's
# needle-driver box when it finds one, via the real serving decoder
# (surgvu.perceive.decode_clip).
#
# Modelled closely on condor/detect_sample.sh; see that file for the base
# rationale (why deps are pip-installed into a scratch --target dir rather
# than baked into the image, why PATH is left alone). surgvu26-train.sif
# does NOT provide what yolov5's models/common.py needs at import time:
# models/common.py itself imports pandas and requests directly, and its own
# import chain (utils/dataloaders.py -> utils/plots.py -> utils/general.py)
# pulls in tqdm, matplotlib and seaborn as hard imports before
# DetectMultiBackend can even be defined. The variant head itself only needs
# torch/torchvision/opencv, already in the base image, so this is the exact
# same dependency gap detect_sample.sh already solves -- reused rather than
# re-derived.

echo "host: $(hostname)  start: $(date)"

DEPS="$(pwd)/.variant_sample_deps"
mkdir -p "$DEPS"
python3 -m pip install --no-cache-dir --target "$DEPS" \
    pandas requests tqdm matplotlib seaborn Pillow
PIP_RC=$?
if [ "$PIP_RC" -ne 0 ]; then
    echo "FATAL: pip install into $DEPS failed with exit $PIP_RC"
    exit "$PIP_RC"
fi
export PYTHONPATH="$DEPS${PYTHONPATH:+:$PYTHONPATH}"

python3 scripts/variant_sample_report.py \
    --sample-root /staging/groups/bhaskar_opscribe/surgvu/cat2_sample \
    --weights /staging/groups/bhaskar_opscribe/surgvu_yolo_detector/best.pt \
    --yolov5-dir /staging/groups/bhaskar_opscribe/surgvu_yolo_detector/yolov5 \
    --candidates baselines/shipped_candidates.json \
    --variant-weights /staging/n/nkalthoff/surgvu26/models/variant_head.pt \
    --variant-config /staging/n/nkalthoff/surgvu26/models/variant_head.json \
    --variant-labels config/variant_labels.json \
    --out-json variant_sample_results.json
RC=$?

echo "exit: $RC  end: $(date)"
exit "$RC"
