#!/bin/bash

# UNBUFFER PYTHON. The submit files set stream_output, but that streams the
# FILE -- Python still block-buffers stdout at 4-8KB when it is not a tty, so
# a multi-hour run's progress arrives in lumps long after the fact. pip's
# output appears promptly and ours does not, which reads as 'the job hung
# after the environment check'.
export PYTHONUNBUFFERED=1

set -uo pipefail
# Runs scripts/merge_and_quantise_vlm.py inside surgvu26-train.sif on a GPU
# execute node: merges the trained LoRA adapter into the fp16 Qwen2.5-VL-7B-
# Instruct base and quantises the result to 4-bit NF4 (see that script's own
# module docstring for the full design and why merging reverses
# scripts/train_vlm.py's own "never merge" note deliberately).
#
#   condor_submit condor/merge_quantise.sub
#
# THE ENVIRONMENT is condor/train_vlm.sh's PROVEN recipe, reused unchanged
# except for which extra packages get installed:
#
#   * surgvu26-train.sif ships only torch/torchvision/opencv-python-headless/
#     PyYAML (containers/surgvu26-train.def).
#   * transformers 4.57.6 + accelerate/tokenizers/safetensors/
#     huggingface_hub/psutil/packaging/tqdm/regex/filelock already live at
#     /staging/n/nkalthoff/surgvu26/vlm_pypkgs2 (built for the Qwen3-VL
#     serving fallback, reused here unmodified -- it already supports
#     Qwen2.5-VL).
#   * `peft` (LoRA merge) and `bitsandbytes` (NF4 quantisation) are
#     installed fresh with --no-deps -- load-bearing, per condor/
#     train_vlm.sh's own header: an unpinned `pip install peft` or
#     `bitsandbytes` resolves and downloads its OWN torch, which then
#     shadows the container's 2.5.1+cu121 through PYTHONPATH and breaks
#     torchvision's C++ ops (this cost job 9629572 once already, for
#     bitsandbytes alone). No bert_score/pandas/matplotlib here -- unlike
#     condor/train_vlm.sh, this job never scores against
#     surgvu.scoring.Scorer (verify_checkpoint prints one generation for a
#     human to read, it does not compute BERTScore), so that whole second
#     pip call and its matplotlib chain is not needed.
#
# HF_HOME is redirected to /staging/n/nkalthoff/surgvu26/hf_cache -- the SAME
# cache condor/train_vlm.sh populated, so this job's `find_local_snapshot_
# dir` finds the already-downloaded fp16 base with NO network access at
# all (this script also sets HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE itself,
# belt to this suspenders).
#
# Run from the repo root, after: mkdir -p logs

echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 -c "import torch; print('cuda', torch.cuda.is_available())"
umask 002

# STAGING MOUNT GUARD, identical to condor/train_vlm.sh. Every input this
# job reads (the HF cache, the trained adapter, the qa_frames_manifest.jsonl
# verification frame, the vlm_pypkgs2 deps) and everything it writes (the
# merged fp16 checkpoint, the final quantised checkpoint) lives under
# /staging, so a job without the mount cannot do anything useful.
if ! mkdir -p /staging/n/nkalthoff/surgvu26/models 2>/dev/null; then
    echo "FATAL: /staging is not mounted on $(hostname). +WantStagingMount" \
         "did not take effect; exiting fast so this reschedules elsewhere." >&2
    exit 75
fi

export HF_HOME=/staging/n/nkalthoff/surgvu26/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$HF_HOME"

DEPS=/staging/n/nkalthoff/surgvu26/vlm_pypkgs2
[ -d "$DEPS" ] || { echo "FATAL: $DEPS not visible on $(hostname)"; exit 75; }

# ---- peft + bitsandbytes, and nothing else --------------------------------
# --no-deps IS LOAD-BEARING -- see the header comment above.
NEWDEPS="$(pwd)/.merge_quantise_deps"
mkdir -p "$NEWDEPS"
# PINNED TO WHAT THE IMAGE INSTALLS. This was `peft bitsandbytes`, unpinned,
# and on 2026-08-26 it resolved bitsandbytes 0.50.2 while containers/Dockerfile
# pins 0.50.1. A checkpoint is QUANTISED here and LOADED there: the
# quantization_config baked into config.json is read back by a different
# version than the one that wrote it, and any incompatibility surfaces inside
# try_vlm_result's exception-swallowing wrapper -- silently, on the graded run,
# as "the VLM declined" rather than as a version error.
#
# The versions are the image's, not the newest: the image is what actually
# serves, so it is the authority. Bumping either means bumping both.
python3 -m pip install --no-cache-dir --no-deps --target "$NEWDEPS" \
    "peft==0.20.0" "bitsandbytes==0.50.1"
PIP_RC=$?
if [ "$PIP_RC" -ne 0 ]; then
    echo "FATAL: pip install peft/bitsandbytes into $NEWDEPS failed with exit $PIP_RC"
    exit "$PIP_RC"
fi
# Belt to that braces: if a future pip ignores --no-deps, the shadowing
# copies go anyway rather than silently deciding which torch/numpy the run
# uses (identical reasoning to condor/train_vlm.sh's own install).
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

python3 "$@"
RC=$?
echo "exit: $RC  end: $(date)"
exit $RC
