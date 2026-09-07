#!/bin/bash

# UNBUFFER PYTHON. The submit files set stream_output, but that streams the
# FILE -- Python still block-buffers stdout at 4-8KB when it is not a tty, so
# a multi-hour run's progress arrives in lumps long after the fact. pip's
# output appears promptly and ours does not, which reads as 'the job hung
# after the environment check'.
export PYTHONUNBUFFERED=1

set -uo pipefail
# Runs scripts/train_vlm.py inside surgvu26-train.sif on a GPU execute node:
# LoRA fine-tune of Qwen2.5-VL-7B-Instruct (Task 4 of v5 plan3; see
# docs/design/plans/2026-08-25-v5-plan3-vlm-training.md and
# scripts/train_vlm.py's own module docstring for the full design).
#
#   condor_submit condor/train_vlm.sub \
#       args="--output-dir /staging/n/nkalthoff/surgvu26/models/vlm_lora"
#
# THE ENVIRONMENT, reusing condor/vlm_eval.sh's proven recipe rather than
# re-deriving one from scratch:
#
#   * surgvu26-train.sif ships only torch/torchvision/opencv-python-headless/
#     PyYAML (containers/surgvu26-train.def) -- nothing that can load or
#     fine-tune a VLM.
#   * transformers 4.57.6, plus accelerate/tokenizers/safetensors/
#     huggingface_hub/psutil/packaging/tqdm/regex/filelock (its own
#     transitive deps), already live at
#     /staging/n/nkalthoff/surgvu26/vlm_pypkgs2, PRUNED of torch/torchvision/
#     numpy -- built for the Qwen3-VL serving fallback (condor/vlm_eval.sh)
#     and reused here UNMODIFIED rather than duplicated: it already supports
#     Qwen2.5-VL (integrated in transformers well before 4.57.6), and this
#     job's own inputs above (huggingface_hub, psutil, packaging, tqdm,
#     regex, filelock, PyYAML) already satisfy every non-torch dependency
#     `peft` and `bitsandbytes` need.
#   * `peft` (LoRA) and `bitsandbytes` (NF4) are genuinely new here and
#     installed fresh with --no-deps, mirroring EXACTLY why vlm_eval.sh
#     installs bitsandbytes that way: `pip install peft` (or bitsandbytes)
#     with unpinned deps resolves and downloads its OWN torch, which then
#     shadows the container's 2.5.1+cu121 through PYTHONPATH and breaks
#     torchvision's C++ ops -- this cost job 9629572 once already, for
#     bitsandbytes alone. --no-deps plus an explicit rm -rf of anything that
#     slips through anyway (belt to that braces, same as vlm_eval.sh) closes
#     it for both packages here.
#
# HF_HOME is redirected to /staging/n/nkalthoff/surgvu26/hf_cache (already
# used by earlier work in this project, per TORCH_HOME's identical
# convention in condor/train.sh/condor/train_variant.sh): the base model's
# ~16-17GB of full-precision weights download ONCE and every retry/resume
# reuses the cache instead of re-pulling it.
#
# Run from the repo root, after: mkdir -p logs

echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 -c "import torch; print('cuda', torch.cuda.is_available())"
umask 002

# STAGING MOUNT GUARD, identical to condor/train.sh / condor/train_variant.sh.
# Every input this job reads (the manifest, config/splits_v2.json, the
# vlm_pypkgs2 deps, the HF cache) and everything it writes (checkpoints, the
# adapter, the eval report) lives under /staging, so a job without the mount
# cannot do anything useful -- exit fast and non-zero so HTCondor reschedules
# onto another node rather than failing confusingly deeper in.
if ! mkdir -p /staging/n/nkalthoff/surgvu26/models 2>/dev/null; then
    echo "FATAL: /staging is not mounted on $(hostname). +WantStagingMount" \
         "did not take effect; exiting fast so this reschedules elsewhere." >&2
    exit 75
fi

export HF_HOME=/staging/n/nkalthoff/surgvu26/hf_cache
mkdir -p "$HF_HOME"

DEPS=/staging/n/nkalthoff/surgvu26/vlm_pypkgs2
[ -d "$DEPS" ] || { echo "FATAL: $DEPS not visible on $(hostname)"; exit 75; }

# ---- peft + bitsandbytes, and nothing else --------------------------------
# --no-deps IS LOAD-BEARING -- see the header comment above. Both packages
# need torch/transformers/accelerate/numpy, all already satisfied by the
# container plus $DEPS; --no-deps is what stops pip from re-resolving them
# anyway and pulling its own torch.
NEWDEPS="$(pwd)/.train_vlm_deps"
mkdir -p "$NEWDEPS"
# TWO pip calls, deliberately, and the split matters.
#
# --no-deps for peft/bitsandbytes/bert_score: pip must not re-resolve torch and
# pull its own build over the container's 2.5.1+cu121.
#
# WITH deps for pandas/matplotlib: bert_score imports BOTH at module scope
# (bert_score/score.py:7 does `import matplotlib.pyplot as plt`), neither ships
# in the train image, and matplotlib has its own transitive chain (contourpy,
# cycler, fonttools, kiwisolver, pyparsing, python-dateutil, packaging). Listing
# those by hand is how the R22/R34/R35 stub experiment burned four job cycles
# discovering one module at a time; let pip resolve them. Neither pulls torch.
# PINNED to what containers/Dockerfile installs. Unpinned, this resolved
# bitsandbytes 0.50.2 on 2026-08-26 while the image pins 0.50.1 -- and an
# adapter trained under one bitsandbytes then quantised and served under
# another is a mismatch that surfaces inside try_vlm_result's
# exception-swallowing wrapper, silently, on the graded run.
#
# condor/merge_quantise.sh was pinned first; this file is its sibling and was
# missed, so the drift stayed live in the script that trains the adapter. Both
# now match the image.
python3 -m pip install --no-cache-dir --no-deps --target "$NEWDEPS" \
    "peft==0.20.0" "bitsandbytes==0.50.1" bert_score \
 && python3 -m pip install --no-cache-dir --target "$NEWDEPS" \
    pandas matplotlib
PIP_RC=$?
if [ "$PIP_RC" -ne 0 ]; then
    echo "FATAL: pip install peft/bitsandbytes into $NEWDEPS failed with exit $PIP_RC"
    exit "$PIP_RC"
fi
# Belt to that braces: if a future pip ignores --no-deps, the shadowing
# copies go anyway rather than silently deciding which torch/numpy the run
# uses (identical reasoning to condor/vlm_eval.sh's bitsandbytes install).
rm -rf "$NEWDEPS"/torch "$NEWDEPS"/torchvision "$NEWDEPS"/nvidia \
       "$NEWDEPS"/triton "$NEWDEPS"/numpy "$NEWDEPS"/numpy.libs
export PYTHONPATH="$DEPS:$NEWDEPS${PYTHONPATH:+:$PYTHONPATH}"

echo "== effective environment, AFTER PYTHONPATH is set =="
python3 - <<'PY'
import torch, transformers, peft, bitsandbytes
print("torch", torch.__version__, "from", torch.__file__)
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("bitsandbytes", bitsandbytes.__version__)
print("cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
      torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "")
PY
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "FATAL: effective-environment import check failed (exit $RC)"
    exit "$RC"
fi

# TEE TO STAGING SO A RETRY CANNOT DESTROY THE DIAGNOSIS.
#
# The submit file sets max_retries=5, and Condor TRUNCATES output/error on
# each retry. So a job that fails, requeues, and fails again leaves only the
# last attempt's log -- and if the retry gets further than the original (warm
# caches make that likely), the traceback that explains the FIRST failure is
# simply gone. That happened to the v6 smoke train (job 9710300, exit 1 at
# 16:20): the error was destroyed by its own retry before it could be read.
#
# On a 6-18 hour training run that is the difference between diagnosing a
# failure and rerunning blind.
#
# pipefail (set at the top) makes the pipeline exit with python's status, not
# tee's, so $RC stays the code Condor sees.
KEEP_DIR=/staging/n/nkalthoff/surgvu26/joblogs
mkdir -p "$KEEP_DIR" 2>/dev/null
KEEP="$KEEP_DIR/train_vlm_${CONDOR_CLUSTER:-nocluster}_$(date +%Y%m%d_%H%M%S).log"
echo "preserving a copy of this attempt at $KEEP"
python3 "$@" 2>&1 | tee -a "$KEEP"
RC=${PIPESTATUS[0]}
echo "exit: $RC  end: $(date)" | tee -a "$KEEP"
exit $RC
