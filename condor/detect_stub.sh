#!/bin/bash
set -uo pipefail
# Runs ONLY tests/test_detect_stub.py inside surgvu26-train.sif.
#
# This is the R22/R34/R35/R37 stubbing EXPERIMENT, not a smoke test:
# condor/detect_smoke.sh pip-installs pandas, requests, tqdm, matplotlib,
# seaborn and Pillow into scratch precisely so the real packages back
# yolov5's imports; this script deliberately installs NONE of them -- only
# pytest, exactly what condor/pytest.sh installs.
#
# Ruling R22 scoped this to three names (pandas, requests, PIL). Ruling R34
# corrected that to six (adding tqdm, matplotlib, a second pandas import
# site), and the six-name version's own up-front guard assertion caught a
# real problem: requests, PIL and tqdm turned out to be genuinely
# importable in this environment, because surgvu26-train.sif shares its
# base image (pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime) with
# containers/surgvu26-submission.def and containers/Dockerfile, and that
# base already ships all three. Ruling R35 corrected the scope again: the
# submission image's genuinely MISSING set, as inferred from the shared
# base plus the submission .def's own additions (only
# opencv-python-headless and PyYAML), is pandas/matplotlib/seaborn -- a
# different three. This script still installs nothing but pytest: it does
# NOT install pandas/matplotlib/seaborn (they must stay genuinely absent),
# and it relies on the base image to genuinely provide requests/PIL/tqdm
# (tests/test_detect_stub.py's own guard checks both directions and fails
# loudly if either assumption is wrong).
#
# tests/test_detect_stub.py inserts stub modules into sys.modules for only
# pandas/matplotlib/seaborn, entirely in test code -- nothing here does
# that, and nothing here (or anywhere under src/) installs or fakes those
# packages, or fakes requests/PIL/tqdm (which stay real). If the stub is
# wrong, this job fails; that failure is the result, not something to route
# around by adding a real package back to the list below.
#
# Ruling R37: the R35-scoped job actually ran (cluster 9686702) and failed
# a level deeper than yolov5 -- inside torch._dynamo.trace_rules, which
# enumerates module specs at import time and raises ValueError on a stub's
# __spec__ being None (which a bare types.ModuleType's is, by default, not
# merely absent). tests/test_detect_stub.py's stub class now gives every
# stub a real importlib.machinery.ModuleSpec; nothing here changed -- the
# fix is entirely inside the test file, same as everything else this
# script hands off to.

echo "host: $(hostname)  start: $(date)"

DEPS="$(pwd)/.detect_stub_deps"
mkdir -p "$DEPS"
python3 -m pip install --no-cache-dir --target "$DEPS" pytest
PIP_RC=$?
if [ "$PIP_RC" -ne 0 ]; then
    echo "FATAL: pip install pytest into $DEPS failed with exit $PIP_RC"
    exit "$PIP_RC"
fi
export PYTHONPATH="$DEPS${PYTHONPATH:+:$PYTHONPATH}"

python3 -m pytest tests/test_detect_stub.py -v -s
RC=$?

echo "exit: $RC  end: $(date)"
exit "$RC"
