#!/bin/bash
set -uo pipefail
# Builds the detector v2 training corpus and runs the groupmate's YOLOv5
# recipe on it (v5 plan4, Task 2): 886 hand-labeled images (oversampled) +
# a capped, class-protected slice of the 601,261 logbook-constrained
# pseudo-labels from Task 1, trained v5m for 300 epochs per the MONAI/
# SurgToolLoc reference recipe she staged
# (surgvu_yolo_detector/yolov5/detection_files/run_5fold.sh) -- adapted for
# a SINGLE GPU (this job's actual, proven-working shape, matching
# condor/train_vlm.sub's precedent) rather than the recipe's aspirational
# 4-GPU/batch-128 DDP invocation. See scripts/build_detector_v2_dataset.py's
# own module docstring for the oversample/cap/protect reasoning, and this
# script's own comments below for why batch/LR are NOT re-derived from
# scratch: her real, shipped v1 run (batch 16, same hyp.surg14cls.yaml,
# 77%/74% P/R) and the reference recipe (batch 128, same hyp file) already
# bracket an 8x range with the SAME lr0=0.0005 -- this run's batch 32 sits
# inside that already-proven range, not outside it.
#
#   mkdir -p logs
#   condor_submit condor/train_detector_v2.sub
#
# THIS DOES NOT SHIP ANYTHING. The shipped best.pt (v1) stays byte-identical
# until v2 is measured against it through scripts/flag_matrix.py -- see the
# plan's Global Constraints and Task 2's own "do not swap on mAP alone".
# This job only trains and reports v1-comparable metrics on the untouched
# 240-image hand-labeled val split; wiring the flag-matrix re-measurement is
# a separate, later step the controller runs by hand (see this repo's
# detector-v2-report.md for the exact command).
#
# DEPENDENCIES: surgvu26-train.sif ships torch/torchvision/opencv-python-
# headless/PyYAML but not what yolov5/models/common.py needs at import time
# (pandas, requests, tqdm, matplotlib, seaborn) -- identical install to
# condor/pseudo_label.sh/condor/detect_sample.sh, reused verbatim.
#
# WEIGHTS: PRE-STAGED, not downloaded. yolov5s.pt (her v1's base) sits at
# surgvu_yolo_detector/yolov5s.pt; yolov5m.pt (this run's base) did not exist
# anywhere in this project, so it was fetched once from the ultralytics v7.0
# release and staged at
#
#   /staging/n/nkalthoff/surgvu26/models/yolov5m.pt
#   42,806,829 bytes
#   sha256 61d933360ba5a7733a36764996c800287d973889d875227f5beedd2473a97a56
#
# The earlier draft passed the BARE NAME `yolov5m.pt` and relied on
# yolov5/utils/downloads.py's attempt_download() to fetch it at train time,
# reasoning that the same mechanism must have produced yolov5s.pt. That
# reasoning is inference, not verification: nothing in this project has ever
# confirmed an execute node can reach github.com, and the cost of being wrong
# is a job that matches a scarce GPU, installs its deps, builds the dataset,
# and only THEN dies at the download -- after burning the slot. Staging the
# file costs 40 MB and removes the question. The check below is FATAL rather
# than a fallback to auto-download: a silent fall back to the network is
# exactly how this failure would return.
#
# RESUME AFTER EVICTION: re-submit with no arguments. If
# $PROJECT/$NAME/weights/last.pt already exists this script resumes from it
# (train.py's own --resume, which restores the rest of that run's argument
# set from its own opt.yaml) instead of restarting at epoch 0 -- a 300-epoch
# run over a corpus this size makes restarting from scratch after an
# eviction extremely costly.

echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 -c "import torch; print('cuda', torch.cuda.is_available())"
umask 002

DETECTOR_V2_ROOT=/staging/n/nkalthoff/surgvu26/detector_v2
HAND_LABELED_ROOT="$DETECTOR_V2_ROOT/hand_labeled"
DATASET_DIR="$DETECTOR_V2_ROOT/dataset"
RUNS_DIR="$DETECTOR_V2_ROOT/runs"
YOLO_DATASET_TAR=/staging/groups/bhaskar_opscribe/surgvu_yolo_detector/yolo_dataset.tar.gz
YOLOV5_DIR=/staging/groups/bhaskar_opscribe/surgvu_yolo_detector/yolov5
PSEUDO_IMAGES_DIR=/staging/n/nkalthoff/surgvu26/pseudo_labels/images
PSEUDO_LABELS_DIR=/staging/n/nkalthoff/surgvu26/pseudo_labels/labels
PSEUDO_REPORT=/staging/n/nkalthoff/surgvu26/pseudo_labels/pseudo_label_report.json
SPLITS=config/splits_v2.json

MODEL="${DETECTOR_V2_MODEL:-v5m}"
IMG_SIZE="${DETECTOR_V2_IMG:-640}"
BATCH="${DETECTOR_V2_BATCH:-32}"
EPOCHS="${DETECTOR_V2_EPOCHS:-300}"

# The pre-staged base weights -- see the WEIGHTS note above for why this is
# fatal rather than a fallback to yolov5's own downloader.
#
# CHECKED HERE, NOT AT THE TRAIN CALL, and the position is the whole point: by
# the time train.py runs, this job has already matched a scarce GPU, pip-
# installed its deps, and built a ~40-50K-image dataset. Discovering a missing
# 40 MB file at that point wastes all of it. This is the earliest place the
# value is known.
BASE_WEIGHTS="${BASE_WEIGHTS:-/staging/n/nkalthoff/surgvu26/models/yolov5${MODEL#v5}.pt}"
if [ ! -f "$BASE_WEIGHTS" ]; then
    echo "FATAL: base weights not staged at $BASE_WEIGHTS."
    echo "  Stage them on the submit node, then resubmit:"
    echo "    curl -sSL -o $BASE_WEIGHTS \\"
    echo "      https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5${MODEL#v5}.pt"
    exit 42
fi
echo "base weights: $BASE_WEIGHTS ($(stat -c %s "$BASE_WEIGHTS") bytes)"
RUN_NAME="${DETECTOR_V2_NAME:-v2_${MODEL}_${IMG_SIZE}_${EPOCHS}ep}"

# STAGING MOUNT GUARD, identical to condor/pseudo_label.sh / condor/
# train_vlm.sh. Everything this job reads (the hand-labeled tarball, the
# pseudo-label pool, the yolov5 checkout) and everything it writes (the
# combined dataset, training outputs) lives under /staging.
if ! mkdir -p "$DATASET_DIR" "$RUNS_DIR" 2>/dev/null; then
    echo "FATAL: /staging is not mounted on $(hostname). +WantStagingMount" \
         "did not take effect; exiting fast so this reschedules elsewhere." >&2
    exit 75
fi

DEPS="$(pwd)/.train_detector_v2_deps"
mkdir -p "$DEPS"
python3 -m pip install --no-cache-dir --target "$DEPS" \
    pandas requests tqdm matplotlib seaborn Pillow
PIP_RC=$?
if [ "$PIP_RC" -ne 0 ]; then
    echo "FATAL: pip install into $DEPS failed with exit $PIP_RC"
    exit "$PIP_RC"
fi
export PYTHONPATH="$DEPS${PYTHONPATH:+:$PYTHONPATH}"

# EXTRACT THE HAND-LABELED SET, idempotently -- only if not already done by
# an earlier attempt of this same job (train.txt existing is the marker).
if [ ! -f "$HAND_LABELED_ROOT/yolo_dataset/train.txt" ]; then
    echo "extracting yolo_dataset.tar.gz to $HAND_LABELED_ROOT"
    mkdir -p "$HAND_LABELED_ROOT"
    tar xzf "$YOLO_DATASET_TAR" -C "$HAND_LABELED_ROOT"
    TAR_RC=$?
    if [ "$TAR_RC" -ne 0 ]; then
        echo "FATAL: extracting $YOLO_DATASET_TAR failed with exit $TAR_RC"
        exit "$TAR_RC"
    fi
else
    echo "hand-labeled set already extracted at $HAND_LABELED_ROOT, skipping"
fi

# BUILD THE COMBINED CORPUS (deterministic, seeded -- see
# scripts/build_detector_v2_dataset.py's own module docstring for the
# oversample/cap/protect design and its two independent R30 layers). Always
# re-run: it is fast relative to training and reruns to the SAME output
# given the same seed/inputs, so re-running after a retry is a correctness
# safeguard, not wasted work.
python3 scripts/build_detector_v2_dataset.py \
    --hand-root "$HAND_LABELED_ROOT" \
    --pseudo-images-dir "$PSEUDO_IMAGES_DIR" \
    --pseudo-labels-dir "$PSEUDO_LABELS_DIR" \
    --pseudo-report "$PSEUDO_REPORT" \
    --splits "$SPLITS" \
    --out-dir "$DATASET_DIR" \
    "$@"
BUILD_RC=$?
if [ "$BUILD_RC" -ne 0 ]; then
    echo "FATAL: build_detector_v2_dataset.py failed with exit $BUILD_RC" >&2
    exit "$BUILD_RC"
fi

cd "$YOLOV5_DIR" || { echo "FATAL: cannot cd into $YOLOV5_DIR"; exit 1; }

LAST_CKPT="$RUNS_DIR/$RUN_NAME/weights/last.pt"
if [ -f "$LAST_CKPT" ]; then
    echo "resuming from $LAST_CKPT"
    python3 train.py --resume "$LAST_CKPT"
else
    echo "starting a fresh run: model=$MODEL img=$IMG_SIZE batch=$BATCH epochs=$EPOCHS"
    python3 train.py \
        --img "$IMG_SIZE" \
        --batch "$BATCH" \
        --epochs "$EPOCHS" \
        --data "$DATASET_DIR/surg_14cls_v2.yaml" \
        --cfg "models/yolov5${MODEL#v5}.yaml" \
        --weights "$BASE_WEIGHTS" \
        --hyp data/hyps/hyp.surg14cls.yaml \
        --project "$RUNS_DIR" \
        --name "$RUN_NAME" \
        --device 0 \
        --workers 8 \
        --optimizer Adam \
        --exist-ok
fi
RC=$?

BEST="$RUNS_DIR/$RUN_NAME/weights/best.pt"
[ -f "$BEST" ] && echo "best checkpoint bytes: $(stat -c %s "$BEST")"
RESULTS_CSV="$RUNS_DIR/$RUN_NAME/results.csv"
[ -f "$RESULTS_CSV" ] && echo "--- last 5 lines of results.csv ---" && tail -n 5 "$RESULTS_CSV"
echo "exit: $RC  end: $(date)"
exit "$RC"
