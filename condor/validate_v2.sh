#!/bin/bash
set -uo pipefail
# Build a v2 perception config from a named pair of checkpoints, then run the
# SUBMISSION path over the 11 public sample cases with it.
#
#   condor/validate_v2.sh <tools.pt> <task.pt> <probs.npz> <label>
#
# WHY THIS EXISTS SEPARATELY FROM condor/validate.sh. That script validates
# `config/perception.json` -- the SHIPPED config -- and hardcodes v1's
# checkpoint names on purpose, so it cannot be pointed at something else by
# accident. This one is the experiment arm: it builds a config that is not in
# the repo, proves it end to end, and touches nothing v1 depends on.
#
# THE QUESTION IT ANSWERS. Every v2 perception gain so far changed ZERO answers
# on these 11 cases -- +0.0424 macro-F1 bought nothing the grader can see,
# because the router asks coarse questions of a fine-grained record. The
# sensitivity analysis then found that 3 of the 11 are wrong and ALL THREE are
# repairable by a single perception change, two of them by exactly the
# confusions this model is better at (case124 bipolar-vs-cadiere, case126 a
# needle driver never rising above 0.107). So this run is not another proxy
# measurement. It is the first direct test of whether a better tool model
# reaches the graded output at all.

TOOLS_CKPT="${1:?tools checkpoint basename}"
TASK_CKPT="${2:?task checkpoint basename}"
PROBS="${3:?dump_frame_probs .npz for the tools checkpoint}"
LABEL="${4:-v2}"

echo "host: $(hostname)  start: $(date)"
echo "tools=$TOOLS_CKPT task=$TASK_CKPT probs=$PROBS label=$LABEL"

SAMPLE=/staging/groups/bhaskar_opscribe/surgvu/cat2_sample
MODELS_SRC=/staging/n/nkalthoff/surgvu26/models
CONFIG=config/perception_${LABEL}.json

# Pre-create the declared outputs. HTCondor holds a job whose transfer_output
# files are missing at exit, and a job that died on an unmounted /staging must
# be able to RETRY onto another node rather than sit held. `{}` is
# unmistakable: the scorer refuses a candidates file missing cases.
echo '{}' > "${LABEL}_candidates.json"
echo '{}' > "${LABEL}_results.json"
echo '{}' > "$CONFIG"

# PRE-CREATE EVERY DECLARED OUTPUT, before the staging guard below can exit.
# HTCondor HOLDS a job whose transfer_output_files are missing at exit, and a
# hold PREEMPTS max_retries -- so the retryable exit 75 below never got its
# retry. Cluster 9655757 landed on zliu-chtcgpu5000, which does not carry the
# staging mount, exited 75 exactly as designed, and was then held for a missing
# serving_resnetlong.json instead of being rescheduled elsewhere.
#
# The same trap was fixed in condor/metric.sh and condor/verify.sh tonight;
# this is the third place it lives.
echo '{}' > "${LABEL}_candidates.json"
echo '{}' > "${LABEL}_results.json"
mkdir -p config && echo '{}' > "config/perception_${LABEL}.json"
echo '{}' > "serving_${LABEL}.json"

# ---- fail fast where /staging is not mounted ------------------------------
if [ ! -d "$SAMPLE" ]; then
    echo "FATAL: $SAMPLE not visible on this node"
    exit 75
fi
for need in "$MODELS_SRC/$TOOLS_CKPT" "$MODELS_SRC/$TASK_CKPT" "$PROBS"; do
    if [ ! -r "$need" ]; then
        echo "FATAL: $need not readable on this node"
        exit 75
    fi
done

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "(no nvidia-smi)"

# ---- the checkpoint's own identity, which the config builder will check ----
# Its sha256 and its per-frame cuts both have to be carried into the serving
# report, because build_perception_config.py refuses a serving vector whose
# provenance does not name the weights being bound. That refusal is the guard
# that stops a retrain from inheriting stale thresholds, so it is satisfied
# with the real values rather than disabled.
python3 - "$MODELS_SRC/$TOOLS_CKPT" <<'PY' > ckpt_facts.json
import hashlib, json, sys, torch
path = sys.argv[1]
digest = hashlib.sha256()
with open(path, "rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
meta = torch.load(path, map_location="cpu", weights_only=False)["meta"]
json.dump({"sha256": digest.hexdigest(),
           "thresholds": meta.get("thresholds")}, sys.stdout)
PY
if [ ! -s ckpt_facts.json ]; then
    echo "FATAL: could not read checkpoint identity"
    exit 1
fi
SHA=$(python3 -c "import json;print(json.load(open('ckpt_facts.json'))['sha256'])")
CUTS=$(python3 -c "import json;print(json.dumps(json.load(open('ckpt_facts.json'))['thresholds']))")
echo "checkpoint sha256 $SHA"
echo "checkpoint cuts   $CUTS"

# ---- serving thresholds from the dump we already paid a GPU pass for ------
python3 scripts/serving_thresholds_from_dump.py \
    --probs "$PROBS" \
    --frames 16 \
    --aggregation mean \
    --checkpoint-sha256 "$SHA" \
    --checkpoint-thresholds "$CUTS" \
    --members "$TOOLS_CKPT" \
    --out serving_${LABEL}.json || exit 1

# ---- the config ------------------------------------------------------------
python3 scripts/build_perception_config.py \
    --tools-checkpoint "$MODELS_SRC/$TOOLS_CKPT" \
    --task-checkpoint "$MODELS_SRC/$TASK_CKPT" \
    --tools-serving-thresholds serving_${LABEL}.json \
    --out "$CONFIG" || exit 1

echo "== built $CONFIG =="
python3 -c "
import json
c = json.load(open('$CONFIG'))
for role, e in c['experts'].items():
    print(role, e['checkpoint_name'], e['backbone'], e['sha256'][:12])
print('decode', c['decode'])
"

# ---- local checkpoints, simulating weights baked into the image ------------
mkdir -p models
cp "$MODELS_SRC/$TOOLS_CKPT" "$MODELS_SRC/$TASK_CKPT" models/ || exit 1
sha256sum models/*.pt

# ---- the 11 cases, one subprocess each, through the real contract ----------
python3 scripts/validate_cases.py "$SAMPLE" \
    --work-dir ./validate_work \
    --out-prefix "$LABEL" \
    --models-dir "$PWD/models" \
    --device auto \
    --label "$LABEL" \
    --entrypoint-arg="--config=$CONFIG"
RC=$?

echo "validate exit $RC"
ls -la ./*_candidates.json ./*_results.json 2>&1
echo "exit: $RC  end: $(date)"
exit "$RC"
