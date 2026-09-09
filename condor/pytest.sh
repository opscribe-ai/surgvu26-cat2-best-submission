#!/bin/bash
set -uo pipefail
# Runs the test suite inside surgvu26-train.sif.
#
# The pytorch/pytorch base image (surgvu26-train.def) ships torch,
# torchvision, opencv-python-headless and PyYAML, but not pytest. Rather than
# rebuild the image for a test-only dependency, this installs pytest into a
# scratch target directory and puts that on PYTHONPATH before running. Do NOT
# set PATH: the image supplies its own python3/pip and overriding it hides
# them, same as condor/extract.sh.
#
# pyproject.toml is deliberately not transferred (see condor/pytest.sub), so
# pytest's `pythonpath`/`testpaths` ini options never take effect here;
# conftest.py at the repo root puts `src` on sys.path instead, and `tests` is
# passed explicitly below rather than relied on from ini config.

echo "host: $(hostname)  start: $(date)"

DEPS="$(pwd)/.pytest_deps"
mkdir -p "$DEPS"
python3 -m pip install --no-cache-dir --target "$DEPS" pytest
PIP_RC=$?
if [ "$PIP_RC" -ne 0 ]; then
    echo "FATAL: pip install pytest into $DEPS failed with exit $PIP_RC"
    exit "$PIP_RC"
fi
export PYTHONPATH="$DEPS${PYTHONPATH:+:$PYTHONPATH}"

# -m "not slow" excludes tests marked @pytest.mark.slow (pyproject.toml:
# "require bert-score/torch and model download"). surgvu26-train.def
# deliberately installs only opencv-python-headless and PyYAML alongside the
# base image's torch/torchvision (see its "Independent by construction"
# docstring) -- bert-score is a scoring-time dependency, not a training one,
# and its tests would also need to download a roberta-large checkpoint that
# an execute node has no guaranteed path to. Filtering here, rather than
# registering the marker via an ini file, works even though pyproject.toml
# is not transferred: -m matches on the marker name applied by the decorator
# and does not require it to be registered.
python3 -m pytest tests -v -m "not slow"
RC=$?

echo "exit: $RC  end: $(date)"
exit "$RC"
