#!/bin/bash
set -uo pipefail
# Runs the gated VLM fallback over the questions the router cannot route, and
# proves the 11 shipped answers are unchanged with it enabled.
#
#   condor/vlm_eval.sh
#
# Runs inside surgvu26-train.sif; do NOT override PATH (the image supplies its
# own python3/pip, same as condor/train.sh).
#
# THE ENVIRONMENT, and the two traps in it:
#
#   * transformers 4.57.6 comes from /staging/n/nkalthoff/surgvu26/vlm_pypkgs2,
#     a PRUNE-BASED install. `pip install transformers` pulls its own torch
#     (2.13.0+cu130, 4.8 GB with bundled CUDA) which shadows the container's
#     2.5.1+cu121 through PYTHONPATH and breaks torchvision's C++ ops. The
#     deps directory has torch/torchvision/nvidia/triton/numpy pruned out of
#     it for exactly that reason. Do not pip install transformers here.
#   * bitsandbytes is NOT in that directory and NOT in the image. It is
#     installed fresh below because it is the one package whose dependencies
#     the container already satisfies, so pip adds it alone.
#
# The effective environment is printed AFTER PYTHONPATH is set, never before:
# a header describing the pre-change environment reads as confirmation that
# things are fine while the real code runs somewhere else.

echo "host: $(hostname)  start: $(date)"

SAMPLE=/staging/groups/bhaskar_opscribe/surgvu/cat2_sample
MODELS_SRC=/staging/n/nkalthoff/surgvu26/models
VLM_SRC=$MODELS_SRC/qwen3vl-8b-nf4
DEPS=/staging/n/nkalthoff/surgvu26/vlm_pypkgs2

# Declared outputs must exist even on an early exit, or HTCondor holds the job
# instead of retrying it onto a node where /staging is mounted.
for name in shipped fallbackopen vlmcontext vlmnocontext vlmoff vlmon; do
    echo '{}' > "${name}_candidates.json"
done
echo '[]' > vlm_fallback_pairs.json
echo '{}' > vlm_offon_diff.json

for path in "$SAMPLE" "$VLM_SRC" "$DEPS" "$MODELS_SRC/tools_v2.pt"; do
    [ -e "$path" ] || { echo "FATAL: $path not visible on this node"; exit 42; }
done

# Informational only. The image has no nvidia-smi on its PATH, so its absence
# says nothing about the card -- torch below is what decides, and an early
# `|| exit` on this line once threw away a perfectly good GPU node.
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader \
    || echo "(no nvidia-smi in this image)"

# ---- bitsandbytes, and nothing else ---------------------------------------
# --no-deps IS LOAD-BEARING. `pip install --target` resolves dependencies into
# the target directory whether or not the environment already satisfies them,
# so without it bitsandbytes drags in torch 2.13.0+cu130 (plus 1.5 GB of
# nvidia-* wheels), that torch lands first on PYTHONPATH, and the run reports
# `cuda False` on a node holding an H200. That is the same shadowing trap the
# vlm_pypkgs2 prune exists for, wearing a different package's name -- it cost
# job 9629572. bitsandbytes needs torch and numpy, both already in the image.
BNB="$(pwd)/.bnb"
mkdir -p "$BNB"
python3 -m pip install --no-cache-dir --no-deps --target "$BNB" bitsandbytes \
    || exit 43
# Belt to that braces: if a future pip ignores --no-deps, the shadowing copies
# go anyway rather than silently deciding which torch the run uses.
rm -rf "$BNB"/torch "$BNB"/torchvision "$BNB"/nvidia "$BNB"/triton \
       "$BNB"/numpy "$BNB"/numpy.libs
export PYTHONPATH="$DEPS:$BNB${PYTHONPATH:+:$PYTHONPATH}"

echo "== effective environment, AFTER PYTHONPATH is set =="
python3 - <<'PY'
import hashlib, sys
import torch, transformers, bitsandbytes
print("torch", torch.__version__, "from", torch.__file__)
print("transformers", transformers.__version__)
print("bitsandbytes", bitsandbytes.__version__)
print("cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
      torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "")
# The measurement is bound to one perception config; another session is
# regenerating that file, so the run records which bytes it saw.
print("config/perception.json sha256",
      hashlib.sha256(open("config/perception.json", "rb").read()).hexdigest())
# The NF4 weights are a bitsandbytes artifact. Without CUDA this job measures
# nothing, so it exits for a retry rather than reporting a fail-safe decline
# as if it were a result about the model.
sys.exit(0 if torch.cuda.is_available() else 42)
PY
[ $? -eq 42 ] && { echo "FATAL: no usable CUDA device"; exit 42; }

# ---- weights onto local disk ----------------------------------------------
# Not part of the deployment budget: in the submission image they are already
# local. Copying makes the measured load time the one the container will see.
echo "== staging weights to local disk =="
start=$(date +%s)
mkdir -p models vlm
cp "$MODELS_SRC/tools_v2.pt" "$MODELS_SRC/task_v2.pt" models/ || exit 1
cp -r "$VLM_SRC"/. vlm/ || exit 1
echo "copy took $(( $(date +%s) - start ))s"
du -sh vlm

# ---- 1. the measurement ----------------------------------------------------
echo
echo "================================================================"
echo "1. VLM answers for the questions the router cannot route"
echo "================================================================"
python3 scripts/vlm_fallback_eval.py "$SAMPLE" \
    --models-dir "$PWD/models" --vlm-model "$PWD/vlm" --device auto
RC_EVAL=$?
echo "eval exit $RC_EVAL"

# ---- 2. byte-identity of the 11 shipped answers ---------------------------
# The claim is not "the VLM behaved"; it is that it CANNOT touch a routed
# question. Two full passes through the real /input -> /output contract, one
# subprocess per case, differing only in the flag.
echo
echo "================================================================"
echo "2. all 11 sample answers, VLM disabled vs enabled"
echo "================================================================"
python3 scripts/validate_cases.py "$SAMPLE" --work-dir ./work_off \
    --out-prefix vlmoff --models-dir "$PWD/models" --device auto --label vlm-off
RC_OFF=$?
python3 scripts/validate_cases.py "$SAMPLE" --work-dir ./work_on \
    --out-prefix vlmon --models-dir "$PWD/models" --device auto --label vlm-on \
    --entrypoint-arg=--vlm \
    --entrypoint-arg=--vlm-model --entrypoint-arg="$PWD/vlm"
RC_ON=$?

python3 - <<'PY'
import json
off = json.load(open("vlmoff_candidates.json"))
on = json.load(open("vlmon_candidates.json"))
same = off == on
differ = {k: (off.get(k), on.get(k)) for k in set(off) | set(on)
          if off.get(k) != on.get(k)}
print("VLM off/on identical over %d cases: %s" % (len(off), same))
for case_id, pair in sorted(differ.items()):
    print("  DIFFERS %s: %r -> %r" % (case_id, pair[0], pair[1]))
json.dump({"identical": same, "n": len(off), "differences": differ},
          open("vlm_offon_diff.json", "w"), indent=2)
PY

# ---- 3. the fail-safe, end to end -----------------------------------------
echo
echo "================================================================"
echo "3. fail-safe: the VLM enabled with weights that are not there"
echo "================================================================"
mkdir -p failsafe/input failsafe/output
cp "$SAMPLE/case129/case129.mp4" \
   failsafe/input/endoscopic-robotic-surgery-video.mp4
printf '%s' '"Describe what you can see in the upper left corner."' \
   > failsafe/input/visual-context-question.json
python3 scripts/inference.py --input-dir failsafe/input \
    --output-dir failsafe/output --models-dir "$PWD/models" --device auto \
    --vlm --vlm-model /definitely/not/here
echo "failsafe exit $?  response: $(cat failsafe/output/visual-context-response.json)"

echo
echo "eval=$RC_EVAL off=$RC_OFF on=$RC_ON  end: $(date)"
exit "$RC_EVAL"
