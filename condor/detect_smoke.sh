#!/bin/bash
set -uo pipefail
# Runs ONLY tests/test_detect_weights.py inside surgvu26-train.sif.
#
# Modelled on condor/pytest.sh; see that file for the base rationale (why
# deps are pip-installed into a scratch --target dir rather than baked into
# the image, why PATH is left alone). This is a separate, narrower job
# because a dependency failure here (a missing yolov5 import, a stale
# scale_boxes/scale_coords name, a letterbox/scale_coords geometry bug) must
# be unambiguous on its own, not buried in a 1000-test pytest.sub run.
#
# surgvu26-train.def (the pytorch/pytorch base image + opencv-python-headless
# + PyYAML) does NOT provide what yolov5's models/common.py needs at import
# time: models/common.py itself imports pandas and requests directly, and its
# own import chain (utils/dataloaders.py -> utils/plots.py -> utils/general.py)
# pulls in tqdm, matplotlib and seaborn as hard imports before
# DetectMultiBackend can even be defined. Pillow is very likely already
# present as a torchvision dependency, but is listed explicitly below rather
# than assumed, since the container has never actually been exec'd to check.
# scipy appears in yolov5/requirements.txt but is not imported anywhere in
# the DetectMultiBackend/attempt_load/models.yolo import chain used here, so
# it is deliberately NOT installed.

echo "host: $(hostname)  start: $(date)"

DEPS="$(pwd)/.detect_smoke_deps"
mkdir -p "$DEPS"
python3 -m pip install --no-cache-dir --target "$DEPS" \
    pytest pandas requests tqdm matplotlib seaborn Pillow
PIP_RC=$?
if [ "$PIP_RC" -ne 0 ]; then
    echo "FATAL: pip install into $DEPS failed with exit $PIP_RC"
    exit "$PIP_RC"
fi
export PYTHONPATH="$DEPS${PYTHONPATH:+:$PYTHONPATH}"

python3 -m pytest tests/test_detect_weights.py -v -s
RC=$?

echo "exit: $RC  end: $(date)"
exit "$RC"
