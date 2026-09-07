#!/bin/bash
set -uo pipefail
# Runs scripts/train_variant.py inside surgvu26-train.sif on a GPU execute
# node. This job needs BOTH halves other training jobs need separately:
#
#   * condor/train.sh's half -- a GPU, the staging-mount guard, and a shared
#     TORCH_HOME so torchvision's ImageNet-pretrained ResNet-18 weights are
#     fetched once and reused, not re-downloaded by every job;
#   * condor/detect_smoke.sh's half -- the scratch pip-install of yolov5's
#     import-time dependencies. surgvu26-train.sif ships only torch,
#     torchvision, opencv-python-headless and PyYAML; yolov5's
#     models/common.py imports pandas and requests directly, and its own
#     import chain (utils/dataloaders.py -> utils/plots.py -> utils/general.py)
#     pulls in tqdm, matplotlib and seaborn as hard imports before
#     DetectMultiBackend can even be defined. scripts/train_variant.py calls
#     surgvu.detect.Detector to crop needle-driver boxes for training
#     examples (unless run with --no-detector), so this job needs the same
#     dependencies detect_smoke.sh installs, for the same reason.
#
# Run from the repo root, after: mkdir -p logs
#   condor_submit condor/train_variant.sub \
#       args="--out-weights /staging/n/nkalthoff/surgvu26/models/variant_head.pt \
#             --out-config /staging/n/nkalthoff/surgvu26/models/variant_head.json"

echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 -c "import torch; print('cuda', torch.cuda.is_available())"
umask 002

# STAGING MOUNT GUARD, same as condor/train.sh. Every input this job reads
# (config/variant_labels.json's video corpus, the yolov5 checkout, best.pt)
# and every output it writes lives under /staging, so a job without the
# mount cannot do anything useful -- exit fast and non-zero so HTCondor
# reschedules onto another node rather than failing 5s later on a confusing
# "file not found".
if ! mkdir -p /staging/n/nkalthoff/surgvu26/models 2>/dev/null; then
    echo "FATAL: /staging is not mounted on $(hostname). +WantStagingMount" \
         "did not take effect; exiting fast so this reschedules elsewhere." >&2
    exit 75
fi

export TORCH_HOME=/staging/n/nkalthoff/surgvu26/torch_cache
mkdir -p "$TORCH_HOME"

DEPS="$(pwd)/.train_variant_deps"
mkdir -p "$DEPS"
python3 -m pip install --no-cache-dir --target "$DEPS" \
    pandas requests tqdm matplotlib seaborn Pillow
PIP_RC=$?
if [ "$PIP_RC" -ne 0 ]; then
    echo "FATAL: pip install into $DEPS failed with exit $PIP_RC"
    exit "$PIP_RC"
fi
export PYTHONPATH="$DEPS${PYTHONPATH:+:$PYTHONPATH}"

python3 "$@"
RC=$?
echo "exit: $RC  end: $(date)"
exit $RC
