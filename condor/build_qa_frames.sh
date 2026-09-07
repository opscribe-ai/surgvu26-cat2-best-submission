#!/bin/bash

# UNBUFFER PYTHON. The submit files set stream_output, but that streams the
# FILE -- Python still block-buffers stdout at 4-8KB when it is not a tty, so
# a multi-hour run's progress arrives in lumps long after the fact. pip's
# output appears promptly and ours does not, which reads as 'the job hung
# after the environment check'.
export PYTHONUNBUFFERED=1

set -uo pipefail
# Runs `scripts/build_qa_pairs.py --extract-frames` inside surgvu26-train.sif
# (v5 Plan 3, Task 3): samples Task 2's qa_pairs.jsonl (case-stratified, and
# tool_presence_polar answer-balanced toward parity -- see
# scripts/build_qa_pairs.py's sample_corpus/water_fill_allocate) and decodes
# frames for the sampled records via surgvu.perceive.decode_clip_multiscale,
# which applies preprocess.prepare_frame (crop + UI-band blur, exactly as
# serving) and seeks directly in the source video rather than cutting or
# re-encoding a temporary clip (ruling R15).
#
#   condor/build_qa_frames.sh [MAX_PER_INTENT [SEED [FRAMES_PER_WINDOW [DRY_RUN]]]]
#
# MAX_PER_INTENT/SEED/FRAMES_PER_WINDOW forward to the script's own
# --max-per-intent/--seed/--frames-per-window (defaults 2000/0/4 -- see
# scripts/build_qa_pairs.py's DEFAULT_MAX_PER_INTENT and
# DEFAULT_FRAMES_PER_WINDOW docstrings for why). DRY_RUN, if "1", adds
# --dry-run: every video is resolved and every frame-index range is
# computed and reported as a real drop count, but the actual pixel decode
# and JPEG write are skipped -- the one mode of this job that needs no
# GPU-class work and is cheap to smoke-test before paying for a full run.
#
# Runs inside surgvu26-train.sif (the same image condor/dump_motion_v2.sh
# and condor/pytest.sh use), so `python3` here is the image's interpreter.
# Do NOT set PATH: the image supplies its own environment and overriding it
# hides that interpreter.

echo "host: $(hostname)  start: $(date)"

MAX_PER_INTENT="${1:-2000}"
SEED="${2:-0}"
FRAMES_PER_WINDOW="${3:-4}"
DRY_RUN="${4:-0}"
# Decode threads. The job already requests 4 CPUs and used exactly one of
# them; at 247 frames/min a 16-frame rebuild projects to ~24h.
WORKERS="${5:-1}"

# Overridable like the outputs below: the v2 corpus regenerates qa_pairs to a
# parallel file (deterministic absence golds) and must not read the v1 one.
QA_PAIRS="${QA_PAIRS:-/staging/n/nkalthoff/surgvu26/qa_pairs.jsonl}"
VIDEO_ROOT=/staging/groups/bhaskar_opscribe/surgvu/videos/surgvu24

# THE OUTPUT PATHS ARE OVERRIDABLE, AND THE DEFAULTS ARE THE v1 (4-frame) TREE.
#
# A rebuild at a different --frames-per-window writes a DIFFERENT corpus to the
# same three paths, silently replacing the manifest every trained adapter was
# fitted on. The v1 tree took 4h55m to produce and is the only fallback if a
# rebuild is wrong, so it must not be clobbered by a job that is, from the
# submit file's point of view, just the same job with one number changed.
#
# Pass SUFFIX (e.g. SUFFIX=_v2) via the submit file's environment to write a
# parallel tree instead. Empty (the default) reproduces the historical paths
# exactly, so an unsuffixed rerun still behaves as it always did.
SUFFIX="${SUFFIX:-}"
FRAMES_ROOT="${FRAMES_ROOT:-/staging/n/nkalthoff/surgvu26/qa_frames${SUFFIX}}"
MANIFEST_OUT="${MANIFEST_OUT:-/staging/n/nkalthoff/surgvu26/qa_frames_manifest${SUFFIX}.jsonl}"
REPORT_OUT="${REPORT_OUT:-/staging/n/nkalthoff/surgvu26/qa_frames_report${SUFFIX}.json}"

# Refuse to overwrite a manifest that already exists. Rebuilding on top of a
# populated frames tree is the specific accident this guards: the frame files
# are written per window, so a 16-frame run over a 4-frame tree leaves a
# MIXTURE -- windows carrying 16 frames where it got that far and 4 where it
# did not -- and the manifest would list whichever count the run recorded. No
# error, no obvious symptom, and a corpus nobody can characterise afterwards.
if [ -e "$MANIFEST_OUT" ] && [ "${ALLOW_OVERWRITE:-0}" != "1" ]; then
    echo "FATAL: $MANIFEST_OUT already exists."
    echo "  Set SUFFIX to write a parallel tree, or ALLOW_OVERWRITE=1 to mean it."
    exit 76
fi

# Fail here, loudly, rather than partway through the sample/decode.
[ -f "$QA_PAIRS" ]  || { echo "FATAL: $QA_PAIRS not visible on this node"; exit 75; }
[ -d "$VIDEO_ROOT" ] || { echo "FATAL: $VIDEO_ROOT not visible on this node"; exit 75; }

echo "qa_pairs=$QA_PAIRS"
echo "video_root=$VIDEO_ROOT"
echo "frames_root=$FRAMES_ROOT"
echo "manifest_out=$MANIFEST_OUT"
echo "report_out=$REPORT_OUT"
echo "max_per_intent=$MAX_PER_INTENT seed=$SEED frames_per_window=$FRAMES_PER_WINDOW dry_run=$DRY_RUN workers=$WORKERS"

DRY_FLAG=""
if [ "$DRY_RUN" = "1" ]; then
    DRY_FLAG="--dry-run"
fi

umask 002
python3 scripts/build_qa_pairs.py --extract-frames \
    --qa-pairs "$QA_PAIRS" \
    --video-root "$VIDEO_ROOT" \
    --frames-root "$FRAMES_ROOT" \
    --manifest-out "$MANIFEST_OUT" \
    --report-out "$REPORT_OUT" \
    --max-per-intent "$MAX_PER_INTENT" \
    --seed "$SEED" \
    --frames-per-window "$FRAMES_PER_WINDOW" \
    --workers "$WORKERS" \
    $DRY_FLAG
RC=$?

[ -f "$MANIFEST_OUT" ] && echo "manifest bytes: $(stat -c %s "$MANIFEST_OUT")"
echo "exit: $RC  end: $(date)"
exit "$RC"
