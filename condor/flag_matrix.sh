#!/bin/bash
set -uo pipefail
# Runs scripts/flag_matrix.py end to end: every subset of --motion-v2/--yolo/
# --variant-head (baseline included) through the REAL submission entrypoint
# on the 11 public sample cases, then scored with the OFFICIAL metric.
#
# THE QUESTION THIS ANSWERS. A Large-vs-Mega needle-driver head was trained
# and wired behind a four-condition answer gate; a diagnostic that ran the
# head directly against case126/case132 predicted +0.0543 on the 11-case
# mean. That number has never been measured end to end through the real
# pipeline. This job is that measurement -- for every flag combination, not
# just the one that is about to ship, because v4 already shipped two changes
# at once and its leaderboard movement could not be attributed to either.
# This is a TRIPWIRE, not a gate: it does not decide what ships.
#
# TWO CONTAINERS, SEQUENTIALLY, NOT NESTED. scripts/inference.py needs
# surgvu26-train.sif (torch/torchvision/opencv, plus yolov5's pandas/tqdm/
# matplotlib/seaborn for --yolo, the same gap condor/detect_sample.sh and
# condor/variant_sample.sh already solve). surgvu.scoring.Scorer needs
# bert_score/transformers, which live only in surgvu26-extract.sif's venv --
# a DIFFERENT, non-pytorch base image (python:3.11-slim + numpy/cv2/PyYAML;
# see condor/score.sh). Apptainer cannot nest one `exec` inside another, so
# this runs as universe=vanilla (like condor/score.sh, condor/metric.sh) and
# calls `apptainer exec` TWICE, back to back, never from inside a
# container-universe job:
#
#   phase 1 (train.sif)   scripts/flag_matrix.py --mode run
#                          -- drives scripts/inference.py once per case per
#                          combination via validate_cases.run_case, writes
#                          one JSON record per combination
#   phase 2 (extract.sif) scripts/flag_matrix.py --mode score
#                          -- reads those records, scores each with the real
#                          Scorer, writes baselines/flag_matrix.json
#
# THIS IS A MEASUREMENT TOOL, NOT A PIPELINE CHANGE: it does not touch src/,
# does not change any answer, and decides nothing.
#
# Modelled closely on condor/detect_sample.sh (the yolov5 scratch-deps
# install), condor/validate.sh / condor/validate_v2.sh (copying the config's
# tools/task checkpoints locally and re-rooting with --models-dir, exactly
# the re-rooting the submission image will do), and condor/score.sh /
# condor/metric.sh (the extract-venv invocation, including the symlink trap
# noted below).
#
# Run from the repo root, after:
#   mkdir -p logs
#   condor_submit condor/flag_matrix.sub

export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin
echo "host: $(hostname)  start: $(date)"

TRAIN_SIF=/staging/n/nkalthoff/surgvu26/surgvu26-train.sif
EXTRACT_SIF=/staging/n/nkalthoff/surgvu26/surgvu26-extract.sif
EXTRACT_PY=/staging/n/nkalthoff/surgvu26/env/bin/python3
SAMPLE=/staging/groups/bhaskar_opscribe/surgvu/cat2_sample
MODELS_SRC=/staging/n/nkalthoff/surgvu26/models
YOLO_WEIGHTS=/staging/groups/bhaskar_opscribe/surgvu_yolo_detector/best.pt
YOLO_REPO=/staging/groups/bhaskar_opscribe/surgvu_yolo_detector/yolov5
VARIANT_WEIGHTS=/staging/n/nkalthoff/surgvu26/models/variant_head.pt
export HF_HOME=/staging/n/nkalthoff/surgvu26/hf_cache

# Pre-create the declared output. HTCondor HOLDS a job whose
# transfer_output_files is missing at exit, and a hold PREEMPTS max_retries
# -- so a job that dies on an unmounted /staging must be able to RETRY onto
# another node rather than sit stuck. Same trap fixed in condor/validate_v2
# .sh, condor/metric.sh, condor/verify.sh.
mkdir -p baselines
echo '{}' > baselines/flag_matrix.json

# ---- fail fast where /staging (or the extract venv) is not visible --------
# NOT "$EXTRACT_PY" directly: the venv's bin/python3 is a SYMLINK to
# /usr/local/bin/python3, which exists only INSIDE the .sif -- `[ -e ]`
# follows symlinks, so testing it from the host resolves to a path that is
# not there and fails a job that would have run perfectly (cluster 9652325,
# see condor/metric.sh). Check the venv directory instead, which is real on
# both sides of the container boundary.
for path in "$TRAIN_SIF" "$EXTRACT_SIF" \
            "$(dirname "$(dirname "$EXTRACT_PY")")" \
            "$SAMPLE" "$HF_HOME" "$YOLO_WEIGHTS" "$YOLO_REPO" "$VARIANT_WEIGHTS"; do
    if [ ! -e "$path" ]; then
        echo "FATAL: $path not visible on $(hostname)"
        exit 75
    fi
done

echo "== node =="
nproc
free -g | head -2
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || echo "(no nvidia-smi)"

# ---- local checkpoints, simulating weights baked into the image -----------
# Same reasoning as condor/validate.sh: config/perception.json binds the
# tools/task checkpoints at /staging/n/nkalthoff/... paths that will not exist
# inside the submission image. Copying locally and re-rooting with
# --models-dir exercises the exact re-rooting the container does, rather
# than reading /staging directly just because this job happens to have it
# mounted. --yolo-weights/--variant-weights are read straight from
# /staging instead (same as condor/detect_sample.sh, condor/variant_sample
# .sh): they are independent CLI flags, not part of the config's --models-
# -dir re-rooting, and this job is a diagnostic, not the packaging proof
# condor/validate.sh already is.
CKPTS=$(apptainer exec -B /staging "$TRAIN_SIF" python3 -c "
import json
config = json.load(open('config/perception.json'))
print(' '.join(e['checkpoint_name'] for e in config['experts'].values()))
") || { echo "FATAL: cannot read config/perception.json"; exit 1; }
echo "config binds: $CKPTS"
mkdir -p models
for name in $CKPTS; do
    cp "$MODELS_SRC/$name" models/ || exit 1
done
sha256sum models/*.pt

# ---- phase 0: yolov5's missing deps, scratch-installed once ---------------
# Run INSIDE the container (not the bare host) so the installed wheels match
# train.sif's own python/ABI -- same gap condor/detect_sample.sh and condor/
# variant_sample.sh solve, for the identical reason.
DEPS="$(pwd)/.flag_matrix_deps"
mkdir -p "$DEPS"
apptainer exec -B /staging "$TRAIN_SIF" python3 -m pip install --no-cache-dir \
    --target "$DEPS" pandas requests tqdm matplotlib seaborn Pillow
PIP_RC=$?
if [ "$PIP_RC" -ne 0 ]; then
    echo "FATAL: pip install into $DEPS failed with exit $PIP_RC"
    exit "$PIP_RC"
fi

# ---- phase 1: every flag combination through the real entrypoint ----------
# scripts/inference.py, ONE SUBPROCESS PER CASE, inside surgvu26-train.sif.
# --yolo-weights/--yolo-repo/--variant-weights reach EVERY combination
# (including the baseline, harmlessly -- inference.py only reads them when
# --yolo/--variant-head are also passed) via flag_matrix.py's --fixed-arg,
# which is independent of the --flags sweep itself.
apptainer exec -B /staging --env PYTHONPATH="$DEPS" "$TRAIN_SIF" python3 \
    scripts/flag_matrix.py "$SAMPLE" \
    --mode run \
    --run-dir ./flag_matrix_runs \
    --work-dir ./flag_matrix_work \
    --models-dir "$PWD/models" \
    --device auto \
    --python python3 \
    --fixed-arg="--yolo-weights=$YOLO_WEIGHTS" \
    --fixed-arg="--yolo-repo=$YOLO_REPO" \
    --fixed-arg="--variant-weights=$VARIANT_WEIGHTS"
RUN_RC=$?
echo "phase 1 (run) exit $RUN_RC"

# ---- phase 2: score every combination with the OFFICIAL metric ------------
# BERTScore-F1/roberta-large, in the extract venv (bert_score/transformers
# are NOT in surgvu26-train.sif -- see condor/score.sh). Runs even if phase 1
# had per-case failures: assemble_matrix records those as error rows rather
# than needing every combination to have succeeded, and a matrix missing a
# row it could have reported is worse than one reporting the failure.
apptainer exec -B /staging --env HF_HOME="$HF_HOME" "$EXTRACT_SIF" "$EXTRACT_PY" \
    scripts/flag_matrix.py "$SAMPLE" \
    --mode score \
    --run-dir ./flag_matrix_runs \
    --out baselines/flag_matrix.json
SCORE_RC=$?
echo "phase 2 (score) exit $SCORE_RC"

echo "== baselines/flag_matrix.json =="
cat baselines/flag_matrix.json

RC=0
[ "$RUN_RC" -ne 0 ] && RC=1
[ "$SCORE_RC" -ne 0 ] && RC=1
echo "exit: $RC  end: $(date)"
exit "$RC"
