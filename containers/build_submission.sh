#!/bin/bash
set -uo pipefail
# Stages the Docker build context AND builds the equivalent Apptainer image.
#
# Runs on a COMPUTE node (never the login node -- an apptainer build wedges
# ap2001). Produces two artifacts in /staging/n/nkalthoff/surgvu26/:
#
#   submission_context.tar.gz   the complete `docker build` context: src/,
#                               scripts/, config/, a STRIPPED yolov5/ checkout,
#                               models/ (both perception checkpoints plus
#                               yolo_best.pt and variant_head.pt, plus the
#                               --vlm weights IF VLM_MODEL_SRC resolves to a
#                               real directory -- see that variable's own
#                               comment; absent by default today) and
#                               Dockerfile + .dockerignore. Download this to a
#                               machine with Docker and run one command.
#   surgvu26-submission.sif     the same recipe built with Apptainer, so the
#                               image CONTENTS can be validated here. This is
#                               NOT the shipping artifact -- Grand Challenge
#                               takes only `docker save` tarballs.

export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin
echo "host: $(hostname)  start: $(date)"
command -v apptainer || { echo "FATAL: no apptainer"; exit 1; }

# STAGING IS MOUNTED UNDER TWO DIFFERENT NAMES depending on which execute
# node the build lands on: the flat /staging/<user> and the hashed
# /staging/<initial>/<user>. Job 9716663 died in 7 s on e2478 with "FATAL:
# /staging/n/nkalthoff/surgvu26 not visible on this node" -- not because
# staging was missing, but because it was mounted under the other name. The
# early probe below is correct and did its job; the path it was probing was
# the assumption that was wrong. Resolve it instead of hardcoding one form.
DEST=""
# THE FLAT /staging/<user> SYMLINKS WERE DELETED BY CHTC ON 2026-08-31.
# Personal staging now lives ONLY under the alphabetised /staging/<initial>/<user>.
# Two jobs died mid-session on the changeover (9716663 build, 9716651 probe),
# each reporting "not visible on this node" for a directory that was simply
# under its new name. Group staging (/staging/groups/...) is UNAFFECTED.
DEST=/staging/n/nkalthoff/surgvu26
[ -d "$DEST" ] || { echo "FATAL: $DEST not visible on this node"; exit 42; }
echo "staging: $DEST"
MODELS_SRC="$DEST/models"

# --yolo/--variant-head's dependencies. Named to match the Dockerfile's own
# defaults exactly: --yolo-repo defaults to /opt/algorithm/yolov5 (the COPY
# destination below), --yolo-weights to /opt/algorithm/models/yolo_best.pt
# (hence the rename on copy -- the source file is named best.pt) and
# --variant-weights to /opt/algorithm/models/variant_head.pt (already the
# source's own name, no rename needed).
YOLO_REPO_SRC=/staging/groups/bhaskar_opscribe/surgvu_yolo_detector/yolov5
YOLO_WEIGHTS_SRC=/staging/groups/bhaskar_opscribe/surgvu_yolo_detector/best.pt
VARIANT_WEIGHTS_SRC="$DEST/models/variant_head.pt"

# --vlm's baked-in weights -- OPTIONAL, unlike every path above. Named to
# match scripts/inference.py's DEFAULT_VLM_MODEL_DIR exactly (REPO / "models"
# / "qwen25vl-7b-nf4", which is /opt/algorithm/models/qwen25vl-7b-nf4 once
# COPYed by the Dockerfile's existing `COPY models/ /opt/algorithm/models/`
# layer -- no Dockerfile/.def change needed for this path to reach the image).
# Overridable, per this task's requirement that a different checkpoint build
# without editing this script -- e.g. once a later fine-tune step produces a
# new self-contained directory:
#
#   VLM_MODEL_SRC=/staging/n/nkalthoff/surgvu26/models/qwen25vl-7b-nf4-v2 \
#       containers/build_submission.sh
#
# THE DEFAULT BELOW DOES NOT EXIST TODAY, ON PURPOSE. See this task's report
# (docs/design/2026-08-24-v5-evidence-pipeline/
# vlm-weight-staging-report.md) for the full accounting: the only base-model
# artifact actually on disk is the full fp16 Qwen2.5-VL-7B-Instruct (measured
# 16 GB at /staging/n/nkalthoff/surgvu26/hf_cache), which alone blows the
# documented 10 GB image ceiling by roughly 2x once added to this image's
# other layers, and `src/surgvu/evidence_vlm.py`'s loader (frozen -- this
# project does not modify src/) neither requests on-load quantisation nor
# applies a LoRA adapter separately, so shipping the fp16 base would not even
# run the fine-tune sitting at models/vlm_lora/checkpoint-1200, on top of not
# fitting. A pre-quantised, adapter-merged NF4 checkpoint would likely fit
# (~5-6 GB, by the ratio the retired models/qwen3vl-8b-nf4 artifact -- a
# DIFFERENT model -- actually measures at) but does not exist for THIS model
# and THIS adapter, and producing one is a decision for the controller, not
# this script (see the report). So the block below STAGES IF PRESENT and
# SKIPS -- loudly, but not fatally -- IF ABSENT, unlike the FATAL probes
# above: an ordinary CNN-only build must keep working while that decision is
# pending, exactly the "verified-safe no-op" state --vlm already ships in.
VLM_MODEL_SRC="${VLM_MODEL_SRC:-$DEST/models/qwen25vl-7b-nf4}"

# The build gate's serving-threshold mode, defaulted here so BOTH the tarball
# gate below and the Apptainer build take the same knob the Dockerfile's
# `ARG SERVING_THRESHOLDS=required` takes. To stage a context built without the
# re-tuned clip-mean cuts (the counterpart of `build_perception_config.py
# --drop-serving-thresholds`), type it -- in the environment of a local run,
# or by adding this line to containers/build_submission.sub:
#
#   environment = "SERVING_THRESHOLDS=optional"
#
# It waives only the block's ABSENCE; a block that is present and unfit still
# fails.
SERVING_THRESHOLDS="${SERVING_THRESHOLDS:-required}"
echo "serving-threshold gate: $SERVING_THRESHOLDS"

# ---- probe /staging EARLY --------------------------------------------------
# /staging is read-only on some execute nodes and unmounted on others. A build
# that discovers this after 20 minutes of pip and mksquashfs has burned the
# slot; exit 42 immediately so the retry lands somewhere it works.
if [ ! -d "$DEST" ]; then
    echo "FATAL: $DEST not visible on this node"; exit 42
fi
if ! touch "$DEST/.writeprobe.$$" 2>/dev/null; then
    echo "FATAL: $DEST is not writable on this node"; exit 42
fi
rm -f "$DEST/.writeprobe.$$"
# The checkpoint list is DERIVED from the config rather than hardcoded. A
# config that ensembles names more than one checkpoint per expert, and a
# hardcoded pair would stage the primaries, pass every gate that only knows
# about primaries, and produce an image whose first graded case dies looking
# for a file that was never copied in.
CONFIG="${CONFIG:-config/perception.json}"
CKPTS=$(python3 - "$CONFIG" <<'EOF'
import json, sys
from pathlib import Path
cfg = json.load(open(sys.argv[1]))
names = []
for entry in cfg["experts"].values():
    for path in [entry["checkpoint"]] + list(entry.get("ensemble") or []):
        name = Path(path).name
        if name not in names:
            names.append(name)
print(" ".join(names))
EOF
) || { echo "FATAL: could not read checkpoints from $CONFIG"; exit 1; }
echo "config=$CONFIG checkpoints:$CKPTS"
for ckpt in $CKPTS; do
    [ -r "$MODELS_SRC/$ckpt" ] || { echo "FATAL: cannot read $MODELS_SRC/$ckpt"; exit 42; }
done
# Same probe, same reasoning, for the evidence-pipeline dependencies: fail in
# seconds rather than after the yolov5 strip and the tar/mksquashfs work below.
for path in "$YOLO_REPO_SRC" "$YOLO_WEIGHTS_SRC" "$VARIANT_WEIGHTS_SRC"; do
    [ -r "$path" ] || { echo "FATAL: cannot read $path"; exit 42; }
done
[ -d "$YOLO_REPO_SRC" ] || { echo "FATAL: $YOLO_REPO_SRC is not a directory"; exit 42; }
echo "staging probe OK"

# ---- assemble the build context -------------------------------------------
rm -rf context
mkdir -p context/models
cp -r src scripts config context/ || exit 1
for ckpt in $CKPTS; do
    cp "$MODELS_SRC/$ckpt" context/models/ || exit 1
done
cp "$YOLO_WEIGHTS_SRC" context/models/yolo_best.pt || exit 1
cp "$VARIANT_WEIGHTS_SRC" context/models/variant_head.pt || exit 1

# --vlm's weights -- see VLM_MODEL_SRC's own comment above for why this is
# the one artifact in this script that is staged OPTIONALLY. A directory,
# not a single file (like yolo_best.pt/variant_head.pt): `from_pretrained`
# needs the whole self-contained checkpoint -- config.json, tokenizer,
# safetensors shards and, if quantised, the quantization_config that tells
# transformers/bitsandbytes how to reconstruct it -- not one weights file.
# REQUIRE_VLM turns the skip below from a notice into a FATAL. Default 0 keeps
# the CNN-only build working while no NF4 checkpoint exists. Set REQUIRE_VLM=1
# for every build that is meant to SHIP the VLM: without it a mistyped
# VLM_MODEL_SRC produces a fully green build in which --vlm is inert, which is
# the exact silent-nothing failure this project has now hit seven times (wrong
# import name, never-executed detect path, wrong import order, no answer path,
# unresolvable config path, stripped export.py, train/serve prompt mismatch).
# A build is not evidence that a flag does anything.
REQUIRE_VLM="${REQUIRE_VLM:-0}"
echo "VLM staging gate: REQUIRE_VLM=$REQUIRE_VLM src=$VLM_MODEL_SRC"

# `VLM_MODEL_SRC=none` IS THE ONLY WAY TO SAY "NO VLM IN THIS IMAGE", and it
# had to be added (2026-08-29) because the absence-based idiom was actively
# dangerous.
#
# VLM_MODEL_SRC defaults to $DEST/models/qwen25vl-7b-nf4 -- and $DEST is
# /staging/n/nkalthoff/surgvu26, NOT the context directory, so that path EXISTS:
# it is v5.2's merge of the bare-trained vlm_lora. "Just don't set
# REQUIRE_VLM" therefore does not build a CNN-only image. It silently bakes V5
# WEIGHTS into the container. Job 9714497 did exactly that while trying to
# build v6's sidecar image, and the only reason it was caught is that the
# context tarball came out 5.49 GB instead of ~200 MB.
#
# That is worse than it sounds, because the weights it picks up are the ones
# `check_evidence_parity` exists to reject: bare-trained, against a config that
# now ships `vlm_evidence_context: true`.
STAGE_VLM=1
if [ -z "$VLM_MODEL_SRC" ] || [ "$VLM_MODEL_SRC" = "none" ]; then
    if [ "$REQUIRE_VLM" = "1" ]; then
        echo "FATAL: REQUIRE_VLM=1 contradicts VLM_MODEL_SRC=none. Choose one:" \
             "bake a checkpoint into the image, or ship it in the model" \
             "tarball. Doing neither is what REQUIRE_VLM exists to prevent."
        exit 42
    fi
    STAGE_VLM=0
    echo "VLM weights DELIBERATELY NOT staged (VLM_MODEL_SRC=none)."
    echo "  This is a SIDECAR build. The answering VLM ships in the model"
    echo "  tarball extracted to /opt/ml/model/, which scripts/inference.py's"
    echo "  resolve_vlm_model_dir checks BEFORE the in-image path -- so an"
    echo "  in-image copy would be outranked by the upload anyway, and having"
    echo "  both is how a stale tarball silently decides which model runs."
fi

if [ "$STAGE_VLM" = "1" ] && [ -d "$VLM_MODEL_SRC" ]; then
    echo "staging VLM weights from $VLM_MODEL_SRC"
    # A directory is not a checkpoint. `evidence_vlm._load_model` is a bare
    # `from_pretrained` on this path, so it needs config.json plus real weight
    # shards; a half-copied or wrong directory would pass `-d` and then fail at
    # inference time inside the container's failure-swallowing wrapper, i.e.
    # silently. Check the two files that must exist before we commit to it.
    if [ ! -f "$VLM_MODEL_SRC/config.json" ]; then
        echo "FATAL: $VLM_MODEL_SRC has no config.json -- not a loadable checkpoint"
        exit 1
    fi
    if ! ls "$VLM_MODEL_SRC"/*.safetensors >/dev/null 2>&1; then
        echo "FATAL: $VLM_MODEL_SRC has no *.safetensors -- not a loadable checkpoint"
        exit 1
    fi
    cp -r "$VLM_MODEL_SRC" context/models/qwen25vl-7b-nf4 || exit 1
    du -sh context/models/qwen25vl-7b-nf4
elif [ "$STAGE_VLM" = "1" ] && [ "$REQUIRE_VLM" = "1" ]; then
    echo "FATAL: REQUIRE_VLM=1 but $VLM_MODEL_SRC does not exist on this node." \
         "Refusing to build an image whose --vlm flag would be a silent no-op." \
         "Either point VLM_MODEL_SRC at the merged+quantised NF4 checkpoint, or" \
         "build without REQUIRE_VLM to ship the CNN-only pipeline deliberately."
    exit 42
elif [ "$STAGE_VLM" = "1" ]; then
    echo "VLM weights NOT staged: $VLM_MODEL_SRC does not exist on this" \
         "node. --vlm ships as a verified-safe no-op in this build -- see" \
         "docs/container_build.md and" \
         "docs/design/2026-08-24-v5-evidence-pipeline/vlm-weight-staging-report.md" \
         "for why, and set VLM_MODEL_SRC to build with real weights once" \
         "that decision is made. Set REQUIRE_VLM=1 to make this fatal."
fi
cp Dockerfile .dockerignore context/ 2>/dev/null || cp Dockerfile context/ || exit 1

# ---- stage a STRIPPED yolov5 checkout --------------------------------------
# The full checkout carries its own `.git` history (~19 MB, more than the rest
# of the checkout combined) plus training/demo code (`train.py`,
# `val.py`, `detect.py`, `classify/`, `data/`, `utils/aws`,
# `utils/docker`, `utils/flask_rest_api`, `utils/google_app_engine`,
# `utils/loggers`, ...) that `surgvu.detect.Detector` never imports.
#
# What stays: `models/{__init__,common,experimental,yolo}.py` and every
# top-level `utils/*.py` module. `models.common` is what `--yolo-repo` is
# imported through (see src/surgvu/detect.py); `models.experimental.
# attempt_load` -- which `DetectMultiBackend.__init__` calls for a `.pt`
# checkpoint, i.e. exactly our path, not a defensive branch -- itself does
# `from models.yolo import Detect, Model` at call time (needed to unpickle a
# checkpoint saved as a `models.yolo.Model`, and to tag those classes for a
# `export.py` (30KB) MUST BE KEPT despite being export tooling:
# `DetectMultiBackend._model_type` at models/common.py:538 does a RUNTIME
# `from export import export_formats` inside a function body, so no AST walk
# over top-of-file `from utils.X`/`from models.X` imports can see it.
# Excluding it built an image where the detector raised ModuleNotFoundError,
# inference.py's best-effort wrapper swallowed it with one WARNING, --yolo
# was silently inert, and eleven cases validated green at the BASELINE score
# (cluster 9693855). Do not re-exclude it.
# torch-version compatibility fix), which is why `models/yolo.py` is kept
# despite never being imported directly by `common.py`. `models/yolo.py` in
# turn imports `utils.autoanchor`. The whole `utils/` package is kept flat
# (every top-level `.py`, not just the modules on this exact call path) as
# insurance against exactly the kind of under-scoping this project already
# hit three times stubbing pandas/matplotlib/seaborn (rulings R22/R34/R35 in
# tests/test_detect_stub.py) -- verified complete by an AST walk over the
# stripped tree for every remaining `from utils.X` / `from models.X`, not by
# guessing from a single call trace. `LICENSE` is kept (yolov5 is GPL-3.0);
# `requirements.txt` is dropped -- nothing in this image installs from it,
# the Dockerfile's own pinned pip line is the one that runs.
# `du -s --block-size=1`, NOT `du -sb` (`-b` implies `--apparent-size`, which
# was measured to report a wildly inflated number over this /staging mount --
# 101104779 bytes for a checkout whose files sum to 20691490 bytes by
# `find -type f -exec stat --format=%s`, which `--block-size=1` without
# `--apparent-size` matches to within normal block-rounding).
YOLOV5_BEFORE=$(du -s --block-size=1 "$YOLO_REPO_SRC" | cut -f1)
rsync -a \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.github/' \
    --exclude='.dockerignore' \
    --exclude='.gitattributes' \
    --exclude='.gitignore' \
    --exclude='.pre-commit-config.yaml' \
    --exclude='setup.cfg' \
    --exclude='CONTRIBUTING.md' \
    --exclude='README.md' \
    --exclude='tutorial.ipynb' \
    --exclude='requirements.txt' \
    --exclude='classify/' \
    --exclude='detection_files/' \
    --exclude='data/' \
    --exclude='train.py' \
    --exclude='val.py' \
    --exclude='detect.py' \
    --exclude='hubconf.py' \
    --exclude='models/tf.py' \
    --exclude='models/hub/' \
    --exclude='models/*.yaml' \
    --exclude='utils/aws/' \
    --exclude='utils/docker/' \
    --exclude='utils/flask_rest_api/' \
    --exclude='utils/google_app_engine/' \
    --exclude='utils/loggers/' \
    "$YOLO_REPO_SRC/" context/yolov5/ || exit 1
YOLOV5_AFTER=$(du -s --block-size=1 context/yolov5 | cut -f1)
echo "== yolov5 checkout stripped: $YOLOV5_BEFORE bytes -> $YOLOV5_AFTER bytes =="
du -sh "$YOLO_REPO_SRC" context/yolov5

# __pycache__ trees ride along in transferred source dirs and would otherwise
# be baked into the image and into the tarball.
find context -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
find context -name '*.pyc' -delete 2>/dev/null

echo "== build context =="
du -sh context
find context -maxdepth 2 -type d | sort
sha256sum context/models/*.pt

echo "== the context's checkpoints and serving thresholds against the binding =="
# The same gate the image build runs, with the same flag -- failing here means
# a bad tarball is never staged for download, rather than failing on the user's
# laptop, or worse, building an image that silently serves the checkpoint's
# per-frame cuts.
python3 context/scripts/verify_checkpoints.py \
    "context/$CONFIG" context/models \
    --serving-thresholds "$SERVING_THRESHOLDS" || exit 1

tar -czf submission_context.tar.gz -C context . || exit 1
ls -l submission_context.tar.gz
sha256sum submission_context.tar.gz
cp submission_context.tar.gz "$DEST/" && chmod 664 "$DEST/submission_context.tar.gz"
echo "staged $DEST/submission_context.tar.gz"

# ---- build the Apptainer equivalent ---------------------------------------
export APPTAINER_CACHEDIR="$PWD/.apptainer_cache"
export APPTAINER_TMPDIR="$PWD/.apptainer_tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# --build-arg carries the same knob into the .def's %arguments default, so the
# SIF this validates and the image that ships gate identically.
BUILD_ARG="SERVING_THRESHOLDS=$SERVING_THRESHOLDS"
if ! apptainer build --build-arg "$BUILD_ARG" \
        surgvu26-submission.sif surgvu26-submission.def; then
    echo "plain build failed, retrying with --fakeroot"
    apptainer build --fakeroot --build-arg "$BUILD_ARG" \
        surgvu26-submission.sif surgvu26-submission.def || exit 1
fi
ls -l surgvu26-submission.sif

# ---- run the built image against a REAL sample case ------------------------
# The %post smoke test used a synthetic clip. This one is case127, whose gold
# first reference is "Uterine horn", through the real /input -> /output layout
# with /input bound read-only exactly as the grader mounts it.
SAMPLE=/staging/groups/bhaskar_opscribe/surgvu/cat2_sample/case127
if [ -d "$SAMPLE" ]; then
    rm -rf smoke_in smoke_out && mkdir -p smoke_in smoke_out
    cp "$SAMPLE/case127.mp4" smoke_in/endoscopic-robotic-surgery-video.mp4
    cp "$SAMPLE/case127_question.json" smoke_in/visual-context-question.json
    echo "== case127 through the built image, /input mounted READ-ONLY =="
    cat smoke_in/visual-context-question.json; echo
    apptainer run --containall \
        -B "$PWD/smoke_in":/input:ro -B "$PWD/smoke_out":/output \
        surgvu26-submission.sif
    echo "run exit: $?"
    echo "response bytes: $(cat smoke_out/visual-context-response.json 2>&1)"
    echo "gold references: $(cat "$SAMPLE/case127.json")"
else
    echo "WARNING: $SAMPLE not visible; skipped the real-case smoke test"
fi

cp surgvu26-submission.sif "$DEST/" && chmod 664 "$DEST/surgvu26-submission.sif"
echo "staged $DEST/surgvu26-submission.sif ($(stat -c %s surgvu26-submission.sif) bytes)"
echo "end: $(date)"
