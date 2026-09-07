#!/bin/bash
set -uo pipefail
# Runs scripts/pseudo_label_shards.py inside surgvu26-train.sif on a GPU
# execute node (v5 plan4, Task 1): the groupmate's YOLOv5 `best.pt` over the
# 235 unlabeled `.npz` shards under /staging/groups/bhaskar_opscribe/surgvu/
# shards, with every detection checked against tools.csv/tasks.csv via
# surgvu.labels.CaseLabels -- a detection of a tool the logbook says was not
# mounted at that instant is dropped, not reinforced. Accepted detections
# are written as YOLO-format images+labels under /staging/n/nkalthoff/surgvu26/
# pseudo_labels/{images,labels}, in the SAME layout yolo_dataset.tar.gz uses,
# so Task 2 can concatenate the two without a conversion step. See
# scripts/pseudo_label_shards.py's own module docstring for the full design
# (in particular why the logbook constraint is what makes this safe at all,
# and the two out-of-taxonomy classes that always yield ~0).
#
#   condor_submit condor/pseudo_label.sub
#   condor_submit condor/pseudo_label.sub args="--limit 5"   # smoke test
#
# A SMOKE TEST FIRST is strongly recommended, same reasoning as condor/
# cache_evidence.sub: --limit caps how many NEW shards this run processes
# (a shard already recorded in the resume manifest never counts against it),
# so this pays for a handful of real decode+detect passes and reports real
# per-shard wall-clock before committing to the full 235-shard corpus.
#
# RESUME AFTER EVICTION: re-submit with no arguments (or the same --out-root/
# --manifest). scripts/pseudo_label_shards.py's own load_processed_shards
# skips every shard already recorded in pseudo_label_manifest.jsonl, mirroring
# condor/cache_evidence.sub's identical resume-by-unit convention -- this job
# can run for hours over 235 shards and HTCondor eviction is a real, observed
# risk on this cluster (see condor/cache_evidence.sub's own note).
#
# DEPENDENCIES: surgvu26-train.sif ships torch/torchvision/opencv-python-
# headless/PyYAML, but NOT what yolov5's models/common.py needs at import
# time (pandas, requests, tqdm, matplotlib, seaborn -- see condor/
# detect_sample.sh/condor/cache_evidence.sh for the same install, reused
# verbatim here rather than re-derived).

echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 -c "import torch; print('cuda', torch.cuda.is_available())"
umask 002

# STAGING MOUNT GUARD, identical to condor/cache_evidence.sh / condor/
# build_qa_frames.sh. Every input this job reads (the shards, the label
# CSVs, best.pt, the yolov5 checkout, config/splits_v2.json) and everything
# it writes (pseudo-labelled images/labels, the resume manifest, the report)
# lives under /staging, so a job without the mount cannot do anything
# useful -- exit fast and non-zero so HTCondor reschedules elsewhere rather
# than failing confusingly deeper in.
if ! mkdir -p /staging/n/nkalthoff/surgvu26/pseudo_labels 2>/dev/null; then
    echo "FATAL: /staging is not mounted on $(hostname). +WantStagingMount" \
         "did not take effect; exiting fast so this reschedules elsewhere." >&2
    exit 75
fi

DEPS="$(pwd)/.pseudo_label_deps"
mkdir -p "$DEPS"
python3 -m pip install --no-cache-dir --target "$DEPS" \
    pandas requests tqdm matplotlib seaborn Pillow
PIP_RC=$?
if [ "$PIP_RC" -ne 0 ]; then
    echo "FATAL: pip install into $DEPS failed with exit $PIP_RC"
    exit "$PIP_RC"
fi
export PYTHONPATH="$DEPS${PYTHONPATH:+:$PYTHONPATH}"

# "$@" forwards whatever condor/pseudo_label.sub's `arguments` line passed
# (the `args` submit-time macro) straight to scripts/pseudo_label_shards.py's
# own argparse -- every default (shards-dir/labels-root/weights/yolov5-dir/
# splits/out-root) already points at the right /staging path, so `args=""`
# (the default) is itself a complete, correct invocation.
python3 scripts/pseudo_label_shards.py "$@"
RC=$?

REPORT=/staging/n/nkalthoff/surgvu26/pseudo_labels/pseudo_label_report.json
[ -f "$REPORT" ] && echo "report bytes: $(stat -c %s "$REPORT")" && cat "$REPORT"
MANIFEST=/staging/n/nkalthoff/surgvu26/pseudo_labels/pseudo_label_manifest.jsonl
[ -f "$MANIFEST" ] && echo "manifest lines: $(wc -l < "$MANIFEST")"
echo "exit: $RC  end: $(date)"
exit "$RC"
