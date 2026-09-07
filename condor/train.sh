#!/bin/bash
set -u
# Runs inside surgvu26-train.sif; do not override PATH.
echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 -c "import torch; print('cuda', torch.cuda.is_available())"
umask 002

# STAGING MOUNT GUARD. `+WantStagingMount = true` is set in the submit file
# and is USUALLY honoured, but cluster 9638187 landed on a node where it
# silently was not: /staging was absent, mkdir failed against a read-only /,
# and the job then died 5 s later on a "checkpoint does not exist" that
# describes a mount problem as a missing file. Every input this repo trains
# on lives under /staging, so a job without it cannot do anything useful.
#
# Exit fast and non-zero so HTCondor reschedules onto another node, rather
# than burning a slot or -- worse -- reporting a confusing downstream error.
# Paired with a raised `max_retries` in train.sub, since the retry is the
# whole point.
if ! mkdir -p /staging/n/nkalthoff/surgvu26/models 2>/dev/null; then
    echo "FATAL: /staging is not mounted on $(hostname). +WantStagingMount" \
         "did not take effect; exiting fast so this reschedules elsewhere." >&2
    exit 75
fi

# torchvision downloads ImageNet weights to $TORCH_HOME/hub on first use. Left
# unset it defaults to the job's ephemeral scratch, so every execute node
# would re-download the backbone weights instead of reusing a cached copy.
# Pointing it at staging means the first job populates the cache and every
# later job (this task's smoke test, the real run, any future ablation)
# reuses it instead of re-fetching.
export TORCH_HOME=/staging/n/nkalthoff/surgvu26/torch_cache
mkdir -p "$TORCH_HOME"

# v2 experiment dependencies (timm, for EndoViT). Installed --no-deps against
# the container's own python, so it adds three pure packages and cannot
# shadow the container's torch -- which is the failure that once pulled a
# 4.8 GB cu130 torch over the top of 2.5.1+cu121 and broke torchvision's C++
# ops. Prepended only when present, so every pre-v2 job is unaffected.
V2_PKGS=/staging/n/nkalthoff/surgvu26/v2/pypkgs
if [ -d "$V2_PKGS" ]; then
    export PYTHONPATH="$V2_PKGS${PYTHONPATH:+:$PYTHONPATH}"
fi

python3 "$@"
RC=$?
echo "exit: $RC  end: $(date)"
exit $RC
