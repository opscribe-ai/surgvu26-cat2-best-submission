# SurgVU 2026 Category 2 -- Grand Challenge algorithm container.
#
#   read   /input/endoscopic-robotic-surgery-video.mp4
#   read   /input/visual-context-question.json      (a JSON-encoded STRING)
#   write  /output/visual-context-response.json     (a JSON-encoded STRING)
#
# Build context: the repository root PLUS a `models/` directory holding
# tools_v2.pt, task_v2.pt, yolo_best.pt and variant_head.pt, PLUS a stripped
# yolov5/ checkout (models.common + utils.* only -- no .git, no training/
# export scripts). `containers/build_submission.sh` stages exactly that tree;
# see docs/container_build.md.
#
#   docker build --platform linux/amd64 -f containers/Dockerfile -t surgvu26-cat2:v1 .
#   docker save surgvu26-cat2:v1 | gzip -c > surgvu26-cat2-v1.tar.gz
#
# CONSTRAINTS THIS FILE IS BUILT AGAINST
#   * 10 minutes per case, ONE case per invocation -- model load is not
#     amortised, so nothing may be deferred to a warm second call.
#   * 32 GB DRAM.
#   * The instance is either No GPU or a single T4 (sm_75): no bf16, no
#     FlashAttention-2. The CNNs run fp32 and are indifferent to both.
#   * NO INTERNET at inference. Every weight is in the image and nothing is
#     installed at runtime. The build-time smoke test below is what proves it.

# cu121 ships sm_75 kernels, which is the deployment card. Pinned by digest-
# free tag on purpose: this is the same base as containers/surgvu26-train.def,
# so the image we validate on CHTC and the image we ship share a torch build.
FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# PYTHONUNBUFFERED: the stage timings go to stderr and a buffered crash would
# lose the log line that says where the 10 minutes went.
# The *_OFFLINE and HOME-ish vars are belt and braces: nothing in the CNN path
# calls out to a hub (surgvu.perceive.load_expert builds its backbone with
# pretrained=False precisely so serving never needs a torchvision download),
# but a future VLM added at the SEAM must fail loudly rather than hang on a
# DNS timeout inside the budget.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TORCH_HOME=/opt/algorithm/.torch \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.mpl \
    CUDA_MODULE_LOADING=LAZY

# opencv-python-headless, not opencv-python: the GUI build pulls libGL and an
# X stack that a headless grader does not have, and the failure is an import
# error at run time -- inside the budget, after the video is already staged.
# Versions pinned to the ones validated on CHTC in surgvu26-train.sif.
#
# pandas/matplotlib/seaborn ADDED for --yolo. The staged yolov5 checkout
# (models/common.py and its own utils/dataloaders.py, utils/general.py,
# utils/plots.py) hard-imports all three AT MODULE SCOPE -- `import
# models.common` fails before `DetectMultiBackend` is even defined without
# them, regardless of whether anything on the .pt-checkpoint detection path
# ever calls into them. A four-round experiment (tests/test_detect_stub.py,
# rulings R22/R34/R35/R37) tried substituting `sys.modules` stubs instead of
# installing the real packages, to avoid this rebuild; it was abandoned
# because the missing-module surface kept moving under it (three names, then
# six, then a different three, then a torch._dynamo internal needing a real
# `__spec__`) -- an unbounded amount of remaining risk for three packages
# that together add well under 100 MB. `requests`, `PIL` and `tqdm` -- the
# yolov5 chain's other three hard imports -- are NOT pinned here: the
# `pytorch/pytorch` base already ships all three (confirmed by a container
# run), and pinning packages the base already provides risks a version
# fight with whatever torchvision/torch actually need.
# Pinned to versions compatible with this base's numpy (2.1.2, confirmed via
# docs/compliance_audit.md) and Python 3.11: pandas 2.2.3 and matplotlib
# 3.9.2 are the first releases of each to declare numpy-2.x support; seaborn
# 0.13.2 is its first release built against pandas 2.2/matplotlib 3.9.
RUN pip install --no-cache-dir \
        "opencv-python-headless==4.11.0.86" \
        "PyYAML==6.0.2" \
        "pandas==2.2.3" \
        "matplotlib==3.9.2" \
        "seaborn==0.13.2" \
 && python -c "import torch, torchvision, cv2, yaml, pandas, matplotlib, seaborn; \
print('torch', torch.__version__, '| torchvision', torchvision.__version__, \
'| cv2', cv2.__version__, '| pandas', pandas.__version__, \
'| matplotlib', matplotlib.__version__, '| seaborn', seaborn.__version__)"

# ---- the Evidence VLM's serving dependencies -------------------------------
# WITHOUT THIS LAYER `--vlm` CANNOT WORK AT ALL, weights or no weights.
# `evidence_vlm._load_model` imports `transformers` inside the function, and
# `try_vlm_result` swallows every exception and keeps the router's answer --
# so a missing `transformers` does not crash, it just makes the flag
# decorative, with a WARNING in a stderr stream nobody transfers back. That
# is the ninth instance of the silent-nothing family in this project, and it
# would have been the most expensive: shipping a merged, quantised, verified
# 5 GB checkpoint into an image that physically cannot import the library
# that loads it.
#
# THE VERSIONS ARE NOT A CHOICE. They are read off the environment that
# actually produced the checkpoint -- condor/train_vlm.sh printed, in job
# 9697941's own log:
#
#     torch 2.5.1+cu121   transformers 4.57.6
#     peft 0.20.0         bitsandbytes 0.50.1
#
# `torch` matches this base exactly (2.5.1+cu121), which is why these pins
# transplant safely. `peft` is deliberately ABSENT: the adapter is merged
# into the base weights before packaging (scripts/merge_and_quantise_vlm.py),
# so serving loads one plain checkpoint and never needs peft. `accelerate` IS
# required -- `_load_model` passes `device_map={"": ...}`, which transformers
# routes through accelerate.
#
# NO --target AND NO --no-deps HERE, unlike condor/train_vlm.sh and
# condor/merge_quantise.sh, and the difference is deliberate. Those scripts
# install into a PYTHONPATH directory that pip cannot see the container's own
# packages from, so an unpinned resolve pulls its OWN torch plus a full set of
# nvidia_* CUDA 13 wheels which then shadow the container's cu121 and break
# torchvision's C++ ops (this cost job 9629572 once already). A plain install
# into this image's site-packages sees torch 2.5.1 already satisfying
# `transformers>=2.1` and leaves it alone. The verification line below is what
# PROVES that rather than assuming it: it fails the build if torch's version
# or CUDA build moved.
#
# bitsandbytes is checked by `importlib.metadata.version`, NOT by importing
# it, and that asymmetry is deliberate. bitsandbytes probes for a CUDA driver
# at import time; this image is built on whatever node the build job matched,
# which is routinely a CPU-only machine. Importing it there can fail or emit a
# fatal-looking diagnostic and take the whole build down for a reason that has
# nothing to do with the image being correct. The metadata check proves the
# package is INSTALLED, which is all a build node can honestly establish --
# proving it LOADS is the job of condor/validate_image.sub with gpus=1, on
# hardware that actually has a card.
RUN pip install --no-cache-dir \
        "transformers==4.57.6" \
        "accelerate==1.14.0" \
        "bitsandbytes==0.50.1" \
 && python -c "import torch, transformers, accelerate; \
from importlib.metadata import version; \
assert torch.__version__ == '2.5.1+cu121', \
    'the VLM deps moved torch to %s -- see the note on this layer' % torch.__version__; \
print('torch', torch.__version__, '| transformers', transformers.__version__, \
'| accelerate', accelerate.__version__, '| bitsandbytes', version('bitsandbytes'))"

# Grand Challenge runs algorithm containers unprivileged. Creating the user
# explicitly (rather than relying on the platform's --user) means the paths
# below are owned by whoever actually runs, so a write to /output cannot fail
# on a permission the image chose.
RUN groupadd -r algorithm -g 1000 \
 && useradd -m --no-log-init -r -g algorithm -u 1000 algorithm \
 && mkdir -p /input /output /opt/algorithm \
 && chown algorithm:algorithm /output /opt/algorithm

WORKDIR /opt/algorithm

# src, scripts, config, the yolov5 checkout, and BOTH+TWO checkpoints. Copied
# as separate layers rather than one so that editing the router does not
# invalidate the 165 MB weight layer (or the yolov5 layer) on a rebuild.
#
# scripts/ IS NARROWED TO THE TWO THE IMAGE ACTUALLY EXECUTES. Nothing else in
# scripts/ is reachable from inside the container: inference.py is the
# ENTRYPOINT and gate 2, verify_checkpoints.py is gate 1, inference.py imports
# only the standard library, torch and `surgvu.*` from src/ (which it puts on
# sys.path itself), verify_checkpoints.py imports only the standard library,
# and neither they nor any module under src/surgvu/ ever shells out or imports
# by name -- there is no subprocess, os.system, runpy or import_module
# anywhere on the serving path. The other 19 scripts are training, splitting,
# tuning and evaluation tools that only ever run on CHTC; shipping them put
# this project's /staging layout in an image handed to the organizers for no
# benefit (docs/compliance_audit.md §7). The `scripts/...py` names that remain
# in comments and error strings are advice to a maintainer, not calls.
#
# yolov5/ is what `--yolo-repo`'s default (`/opt/algorithm/yolov5`) points at.
# `surgvu.detect.Detector._load` puts this directory on `sys.path` and imports
# `models.common.DetectMultiBackend` through it -- see src/surgvu/detect.py.
# `containers/build_submission.sh` stages a STRIPPED checkout here (no `.git`,
# no training/export scripts, no demo/deployment subdirs) -- see that file's
# own comment for exactly what survives the strip and why.
# `models/` now also carries `yolo_best.pt` (--yolo-weights' default) and
# `variant_head.pt` (--variant-weights' default), staged by the same script;
# neither is named in config/perception.json, so scripts/verify_checkpoints.py
# -- which only ever iterates config["experts"] -- does not check them, and
# needs no change for their presence (see that script's docstring: nothing
# it does scans models_dir for extra files).
COPY --chown=algorithm:algorithm src/ /opt/algorithm/src/
COPY --chown=algorithm:algorithm scripts/inference.py \
                                 scripts/verify_checkpoints.py \
                                 /opt/algorithm/scripts/
COPY --chown=algorithm:algorithm config/  /opt/algorithm/config/
COPY --chown=algorithm:algorithm models/  /opt/algorithm/models/
COPY --chown=algorithm:algorithm yolov5/  /opt/algorithm/yolov5/

# ---------------------------------------------------------------------------
# BUILD-TIME GATE 1: the weights AND the cuts are the ones the config binds.
#
# The config carries a sha256 and a byte count per expert. Checking them HERE
# means a stale or truncated checkpoint fails the build, on a machine with a
# person watching, instead of failing every graded case with a fallback string.
#
# The same gate checks the schema-2 `serving_thresholds` block -- the per-class
# cuts re-tuned on the clip mean, worth +0.0195 macro-F1. scripts/inference.py
# falls back to the checkpoint's per-frame cuts with a WARNING when that block
# is missing or bound to other weights, which at BUILD time is an image that
# ships 0.0195 worse and says so only in a log line nobody reads. So the gate
# fails instead.
#
# SERVING_THRESHOLDS is the escape hatch, and it has to be typed:
#
#   docker build --build-arg SERVING_THRESHOLDS=optional ...
#
# is the counterpart of `build_perception_config.py --drop-serving-thresholds`.
# It waives only the block's ABSENCE -- a block that is present and unfit still
# fails the build -- and the build log says WAIVED rather than OK.
#
# A script rather than an inline heredoc: `RUN cmd <<EOF` needs BuildKit's
# Dockerfile 1.4+ frontend, which would make this file depend on a `# syntax=`
# directive and a frontend image pulled at build time. A plain RUN of a file
# that already had to be in the image works on every builder.
# ---------------------------------------------------------------------------
ARG SERVING_THRESHOLDS=required
RUN python /opt/algorithm/scripts/verify_checkpoints.py \
        /opt/algorithm/config/perception.json /opt/algorithm/models \
        --serving-thresholds "${SERVING_THRESHOLDS}"

# ---------------------------------------------------------------------------
# BUILD-TIME GATE 2: the whole entrypoint runs, offline, and writes a valid
# answer.
#
# A synthetic 40-frame video is generated here rather than shipped, so no test
# fixture has to live in the image. This exercises decode -> both experts ->
# router -> write on CPU, which is the No-GPU deployment path exactly. If a
# dependency were missing, a checkpoint unloadable, or a class list out of
# sync with the config, this fails the BUILD.
#
# --yolo --variant-head ADDED here alongside --motion, matching the ENTRYPOINT
# below exactly. This is the whole point of baking pandas/matplotlib/seaborn
# in rather than only proving they `import` cleanly above: it exercises the
# actual `models.common` -> `DetectMultiBackend` -> `attempt_load` chain
# against the real staged `yolo_best.pt` and `yolov5/` checkout, on a random-
# noise clip where the detector is expected to find nothing -- finding
# nothing is a valid, successful run; failing to import or crashing is not.
# If GATE 2 fails here, the ENTRYPOINT below would fail identically at
# grading time with no chance to fix it.
#
# --vlm ADDED 2026-08-25 (Plan 2, Task 4), matching the ENTRYPOINT below.
# `docker build` has no GPU, so this is precisely the No-GPU deployment draw:
# EvidenceVlmHandle.available() sees torch.cuda.is_available() False BEFORE
# any `transformers` import and the seam skips, exactly as
# tests/test_inference_vlm.py's no-CUDA tests assert on the login node. This
# is not a weaker proof than a real generation would be -- it is the SAME
# guarantee the No-GPU deployment instance needs, exercised for real inside
# the actual built image rather than only in a unit test. `transformers` is
# deliberately NOT pip-installed above (see the ENTRYPOINT's own note): with
# no weights staged yet, installing it now would prove nothing this gate
# doesn't already prove for free, at the cost of image bloat with no
# offsetting benefit until Plan 3 lands a checkpoint.
# ---------------------------------------------------------------------------
RUN set -eu; \
    mkdir -p /tmp/smoke/input /tmp/smoke/output; \
    python -c "\
import cv2, numpy as np; \
w = cv2.VideoWriter('/tmp/smoke/input/endoscopic-robotic-surgery-video.mp4', \
                    cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 512)); \
[w.write(np.random.randint(0, 255, (512, 640, 3), dtype=np.uint8)) for _ in range(40)]; \
w.release()"; \
    printf '%s' '"Are there forceps being used here?"' \
        > /tmp/smoke/input/visual-context-question.json; \
    python /opt/algorithm/scripts/inference.py \
        --input-dir /tmp/smoke/input --output-dir /tmp/smoke/output \
        --models-dir /opt/algorithm/models --device cpu \
        --yolo --variant-head --vlm; \
    python -c "\
import json, sys; \
v = json.load(open('/tmp/smoke/output/visual-context-response.json')); \
assert isinstance(v, str) and v.strip(), 'entrypoint wrote %r' % (v,); \
print('smoke answer:', json.dumps(v))"; \
    rm -rf /tmp/smoke

USER algorithm

# --models-dir is the whole reason config/perception.json can keep recording
# the /staging paths the checkpoints were trained at: the image re-roots them
# by basename, so nothing under /staging ever has to resolve here.
# --input-dir/--output-dir are given explicitly rather than left to the
# defaults so that this line, alone, states the contract.
# --motion ADDED 2026-08-16. It decodes each sampled frame's neighbours and
# records motion evidence, which router.STATIC_ACTIVITY_THRESHOLD (now 1.283)
# uses to downgrade "is tissue being cut?" from Yes to No on a still scene.
# Measured cost 6.6 -> 7.0 s per case against a 600 s budget. Without this
# flag the threshold has nothing to read and the rule silently falls back to
# the presence proxy -- the two changes only work as a pair.
# --yolo --variant-head ADDED 2026-08-25. A flag-matrix measurement over 88
# real entrypoint invocations found +0.0543 mean macro-F1 on the 11 graded
# sample cases (case126 and case132 both 0.7015 -> 1.0000) from this pair
# alone, cleanly attributed against the --motion-only baseline. --yolo-weights,
# --yolo-repo, --variant-weights and --variant-config all default to the
# paths this image COPYs its own dependencies to (see above), so no further
# flags are needed here. This is the reason pandas/matplotlib/seaborn were
# added to the pip install above and the yolov5 checkout was COPYed in: the
# submission image could not run this pair at all without them.
#
# --vlm ADDED 2026-08-25 (Plan 2, Task 4) -- the arbiter, not an intent gate,
# now decides whether the Evidence VLM's draft overrides the router's answer
# (config/arbiter.json's shipped "challenger" mode; --arbiter-mode overrides
# it, and is deliberately NOT added here, so the config file stays the one
# place the shipped mode is named). Lazily constructed and never able to
# raise (build_vlm/try_vlm_result's R18 wrapping) and structurally unable to
# run at all on a No-GPU deployment draw (4-bit NF4 is CUDA-only) -- see
# scripts/inference.py's SEAM block and tests/test_inference_vlm.py's
# explicit no-CUDA tests.
#
# THIS FLAG IS STILL A SAFE NO-OP IN THIS IMAGE, AND DELIBERATELY SO --
# revisited 2026-08-25 now that a LoRA fine-tune checkpoint actually exists
# (models/vlm_lora/checkpoint-1200 on staging). `transformers` is still NOT
# pip-installed above, and no VLM weights are COPYed above, because the
# staging DECISION -- not merely the staging MECHANISM -- says not to yet:
# containers/build_submission.sh now HAS a source-path variable
# (VLM_MODEL_SRC) it can stage a self-contained checkpoint directory from,
# but its default does not resolve to anything on disk, and that is on
# purpose. The only base-model artifact that exists is the full fp16
# Qwen2.5-VL-7B-Instruct, measured at 16 GB -- added to this image's other
# layers plus transformers/bitsandbytes, that is roughly 2x the documented
# 10 GB ceiling, and `src/surgvu/evidence_vlm.py`'s loader (frozen; this
# project does not modify src/) neither quantises on load nor applies a LoRA
# adapter separately, so shipping the fp16 base would not even run the
# fine-tune on top of not fitting. A pre-quantised, adapter-merged NF4
# checkpoint would likely fit (comparable to the ~5-6 GB the retired,
# differently-sized models/qwen3vl-8b-nf4 artifact measures at) but does not
# exist for this model+adapter pair -- see this task's report,
# docs/design/2026-08-24-v5-evidence-pipeline/
# vlm-weight-staging-report.md, and docs/container_build.md, for the full
# accounting and the options once the controller decides how to proceed. On
# a No-GPU draw the seam skips before ever needing `transformers`; on a CUDA
# draw it tries, fails fast and offline (HF_HUB_OFFLINE=1), and falls back
# to the router's answer -- same outcome, extra latency only, exactly as
# before this revisit. The flag itself, together with the .def's
# `%runscript`, is unchanged so the two recipes cannot drift on it.
ENTRYPOINT ["python", "/opt/algorithm/scripts/inference.py", \
            "--input-dir", "/input", \
            "--output-dir", "/output", \
            "--models-dir", "/opt/algorithm/models", \
            "--motion", "--yolo", "--variant-head", "--vlm", "--judge"]
