#!/bin/bash
set -u
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin
echo "host: $(hostname)  start: $(date)"
command -v apptainer || { echo "FATAL: no apptainer"; exit 1; }

# DISTINCT output path. surgvu26-train.sif is in use by running jobs (v6 stage 2
# runs ~23h on it); overwriting it mid-run is not something to find out about
# the hard way. Nothing here writes to that file.
OUT=/staging/n/nkalthoff/surgvu26/surgvu26-train-torch26.sif
mkdir -p /staging/n/nkalthoff/surgvu26

if [ -e "$OUT" ]; then
  echo "FATAL: $OUT already exists; refusing to overwrite. Move it aside first."
  exit 1
fi

export APPTAINER_CACHEDIR="$PWD/.apptainer_cache"
export APPTAINER_TMPDIR="$PWD/.apptainer_tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# The .def's %test block runs here and fails the build on a bad image.
if ! apptainer build surgvu26-train-torch26.sif surgvu26-train-torch26.def; then
  echo "plain build failed, retrying with --fakeroot"
  apptainer build --fakeroot surgvu26-train-torch26.sif surgvu26-train-torch26.def || exit 1
fi

# Re-verify from OUTSIDE the build, against the artifact we are about to ship.
# %test passing during build and the finished .sif being good are two claims.
apptainer exec surgvu26-train-torch26.sif python - <<'EOF' || exit 1
import torch, torchvision, cv2
major, minor = torch.__version__.split('.')[:2]
assert (int(major), int(minor)) >= (2, 6), f"torch {torch.__version__} < 2.6"
# get_arch_list() is [] without a device and build nodes have no GPU; use the
# compiled-in flags, which do not need one.
try:
    arches = torch._C._cuda_getArchFlags()
except Exception:
    arches = None
if arches:
    assert '75' in arches, f"sm_75 missing from {arches}"
    print('arch flags:', arches)
else:
    print('WARNING: sm_75 UNVERIFIED here (no CUDA); check on a GPU node')
from torchvision.ops import nms
print('torch', torch.__version__, '| torchvision', torchvision.__version__,
      '| cv2', cv2.__version__)
EOF

# The actual acceptance test for THIS image: transformers must be willing to
# torch.load. That gate is a version check inside transformers, so it can be
# exercised without a GPU or a real checkpoint -- and if it still refuses, the
# rebuild bought nothing and must not be shipped as though it had.
DEPS=/staging/n/nkalthoff/surgvu26/vlm_pypkgs2
if [ -d "$DEPS" ]; then
  # --bind /staging: apptainer does not bind it by default, so the deps dir is
  # invisible inside the container even though the JOB can see it.
  # --env PYTHONPATH: a bare `PYTHONPATH=... apptainer exec` sets the variable
  # for apptainer itself, not for the process inside. Both were missing in
  # 9713057, which died on ModuleNotFoundError: No module named 'transformers'.
  apptainer exec --bind /staging --env PYTHONPATH="$DEPS" \
      surgvu26-train-torch26.sif python - <<'EOF' || exit 1
import transformers
from transformers.utils.import_utils import check_torch_load_is_safe
check_torch_load_is_safe()          # raises on torch < 2.6
print("OK transformers", transformers.__version__,
      "permits torch.load -- checkpoint resume is available")
EOF
else
  echo "FATAL: $DEPS missing; cannot verify the resume gate, which is the"
  echo "       entire reason for this image. Refusing to ship it unverified."
  exit 1
fi

cp surgvu26-train-torch26.sif "$OUT" && chmod 664 "$OUT"
echo "wrote $OUT ($(stat -c %s "$OUT") bytes)"
echo "end: $(date)"
