#!/bin/bash
set -u
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin
echo "host: $(hostname)  start: $(date)"
command -v apptainer || { echo "FATAL: no apptainer"; exit 1; }

OUT=/staging/n/nkalthoff/surgvu26/surgvu26-train.sif
mkdir -p /staging/n/nkalthoff/surgvu26
export APPTAINER_CACHEDIR="$PWD/.apptainer_cache"
export APPTAINER_TMPDIR="$PWD/.apptainer_tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

if ! apptainer build surgvu26-train.sif surgvu26-train.def; then
  echo "plain build failed, retrying with --fakeroot"
  apptainer build --fakeroot surgvu26-train.sif surgvu26-train.def || exit 1
fi

apptainer exec surgvu26-train.sif python -c "
import torch, torchvision, cv2
print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| cv2', cv2.__version__)
" || exit 1

cp surgvu26-train.sif "$OUT" && chmod 664 "$OUT"
echo "wrote $OUT ($(stat -c %s "$OUT") bytes)"
echo "end: $(date)"
