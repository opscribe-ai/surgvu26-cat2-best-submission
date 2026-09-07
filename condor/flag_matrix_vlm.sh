#!/bin/bash
set -uo pipefail
# The VLM's end-to-end measurement on the 11 public sample cases: baseline vs
# --vlm, at BOTH arbiter modes, scored with the OFFICIAL metric.
#
# WHY THIS EXISTS RATHER THAN A FLAG ADDED TO condor/flag_matrix.sh.
# That job runs scripts/inference.py inside surgvu26-train.sif, which has NO
# transformers -- the VLM jobs inject it through PYTHONPATH from
# /staging/n/nkalthoff/surgvu26/vlm_pypkgs2. Adding "--vlm" to its swept flags
# would therefore have measured a VLM that never loaded: try_vlm_result
# swallows the ImportError, logs a WARNING into a stderr stream this job does
# not surface, and the router answers all eleven cases. The result would look
# exactly like "the VLM changes nothing", which is the single most expensive
# wrong conclusion available here.
#
# So this drives the SUBMISSION image instead. That is not a workaround, it is
# strictly better evidence: surgvu26-submission.sif already contains
# transformers 4.57.6 / accelerate 1.14.0 / bitsandbytes 0.50.1 and the NF4
# checkpoint at /opt/algorithm/models/qwen25vl-7b-nf4, so nothing has to be
# reconstructed, and what gets measured is the artifact that ships rather than
# a rehearsal of it.
#
# GPU IS MANDATORY, NOT PREFERRED. 4-bit NF4 is bitsandbytes/CUDA-only and
# EvidenceVlmHandle.available() returns torch.cuda.is_available(). On a No-GPU
# slot every --vlm combination silently degrades to the router and this job
# reports a confident, meaningless zero. The .sub pins Capability >= 7.5 and
# this script REFUSES to run without a visible device.
#
# TWO MODES, NOT A SWEEP. docs/design/notes/2026-08-26-vlm-error-
# structure-and-arbiter-mode.md measured the VLM's entire loss to be
# open-ended nouns with >=98% polar accuracy, which argues for `primary` (VLM
# on polar, router untouched elsewhere) over the shipped `challenger`. That
# argument came from fp16 on held-out CORPUS cases; this is NF4 on the graded
# distribution, so it gets measured rather than assumed. Two targeted
# configurations, because 8 combinations x 11 cases x 240 s of VLM budget is
# ~3 h of GPU spent mostly on combinations nobody would ship.
#
# THIS IS A MEASUREMENT TOOL: it does not touch src/, changes no answer, and
# decides nothing.

SUBMISSION_SIF=/staging/n/nkalthoff/surgvu26/surgvu26-submission.sif
EXTRACT_SIF=/staging/n/nkalthoff/surgvu26/surgvu26-extract.sif
EXTRACT_PY=/staging/n/nkalthoff/surgvu26/env/bin/python3
SAMPLE=/staging/groups/bhaskar_opscribe/surgvu/cat2_sample
HF_HOME=/staging/n/nkalthoff/surgvu26/hf_cache

echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

for path in "$SUBMISSION_SIF" "$EXTRACT_SIF" "$SAMPLE" "$HF_HOME"; do
    [ -e "$path" ] || { echo "FATAL: missing $path"; exit 42; }
done

# The refusal described above. Exit 42 (node-level fatal, matching this repo's
# /staging-probe convention) so a retry lands on a slot that actually has a card
# rather than producing a green run proving nothing.
apptainer exec --nv -B /staging "$SUBMISSION_SIF" python3 -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('FATAL: no CUDA device visible inside the image; every --vlm '
             'combination would silently fall back to the router')
print('cuda', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
" || exit 42

# Prove the flag is live BEFORE spending an hour measuring it. A load failure
# here is loud; the same failure inside the sweep is swallowed.
apptainer exec --nv -B /staging "$SUBMISSION_SIF" python3 -c "
import transformers, accelerate, bitsandbytes, json, pathlib
d = pathlib.Path('/opt/algorithm/models/qwen25vl-7b-nf4')
assert d.is_dir(), 'NF4 checkpoint absent from the image -- --vlm is inert'
q = json.loads((d / 'config.json').read_text()).get('quantization_config')
assert q and q.get('quant_method') == 'bitsandbytes', 'no quantization_config'
print('vlm deps ok:', transformers.__version__, '| quant:', q.get('bnb_4bit_quant_type'))
" || { echo "FATAL: the image cannot load the VLM; refusing to measure a no-op"; exit 1; }

RC=0
for MODE in challenger primary; do
    echo "== arbiter mode: $MODE =="
    RUN_DIR="./flag_matrix_runs_vlm_$MODE"
    OUT="baselines/flag_matrix_vlm_$MODE.json"
    STAGED_RUNS="/staging/n/nkalthoff/surgvu26/flag_matrix_runs_vlm_$MODE"
    mkdir -p baselines

    # PHASE 1 IS THE EXPENSIVE HALF AND MUST SURVIVE A PHASE 2 FAILURE.
    # Phase 1 is 11 GPU-bound VLM cases; phase 2 is CPU-bound BERTScore. When
    # phase 2 hung in 9698787 the whole job had to be killed, and phase 1's
    # completed run records -- which live in the job's scratch and are not in
    # transfer_output_files -- went with it. Fifteen minutes of a scarce GPU
    # allocation, thrown away by a font cache.
    #
    # So phase 1's records are copied to /staging the moment they exist, and a
    # resubmission reuses them instead of re-running the GPU work. This is the
    # same resume discipline scripts/merge_and_quantise_vlm.py already applies
    # to its own stages.
    if [ -d "$STAGED_RUNS" ] && [ -n "$(ls -A "$STAGED_RUNS" 2>/dev/null)" ]; then
        echo "  reusing staged phase-1 records from $STAGED_RUNS (skipping the GPU work)"
        mkdir -p "$RUN_DIR"
        cp -r "$STAGED_RUNS"/. "$RUN_DIR"/ || exit 1
        P1=0
    else

    # THE DRIVER RUNS FROM THE HOST, THE ENTRYPOINT FROM THE IMAGE.
    #
    # The first attempt (9698756) ran /opt/algorithm/scripts/flag_matrix.py,
    # which DOES NOT EXIST: the Dockerfile deliberately narrows the image's
    # scripts/ to inference.py and verify_checkpoints.py. Phase 1 died
    # instantly with "can't open file", phase 2 scored eight combinations that
    # had no run records, and the job exited 0.
    #
    # Running the driver from the transferred working directory (apptainer
    # binds $PWD) costs nothing in fidelity: flag_matrix.py is a harness that
    # spawns one subprocess per case. What is actually UNDER TEST stays
    # entirely in-image -- --entrypoint is the image's inference.py,
    # --models-dir is the image's models, and the transformers/bitsandbytes it
    # imports are the image's. The harness only decides which cases to run.
    apptainer exec --nv -B /staging "$SUBMISSION_SIF" python3 \
        scripts/flag_matrix.py "$SAMPLE" \
        --mode run \
        `# --flags=--vlm, NOT "--flags --vlm": --flags is nargs="+", so a` \
        `# following token that itself starts with "-" is parsed as the next` \
        `# OPTION, leaving --flags with zero arguments and argparse exiting 2.` \
        `# Cost a whole GPU allocation (9698776) to learn.` \
        --run-dir "$RUN_DIR" \
        --work-dir "./flag_matrix_work_vlm_$MODE" \
        --entrypoint /opt/algorithm/scripts/inference.py \
        --models-dir /opt/algorithm/models \
        --device auto \
        --python python3 \
        --flags=--vlm \
        --fixed-arg=--yolo \
        --fixed-arg=--variant-head \
        --fixed-arg="--arbiter-mode=$MODE"
    P1=$?
    fi
    echo "  phase 1 ($MODE) exit $P1"
    if [ "$P1" -eq 0 ] && [ ! -d "$STAGED_RUNS" ]; then
        mkdir -p "$STAGED_RUNS"
        cp -r "$RUN_DIR"/. "$STAGED_RUNS"/ 2>/dev/null \
            && echo "  staged phase-1 records to $STAGED_RUNS" \
            || echo "  WARNING: could not stage phase-1 records; a retry will redo the GPU work"
    fi
    # FATAL, not a note. The first run let a phase-1 exit of 2 through, scored
    # eight empty combinations, printed "no run record for this combination"
    # eight times, and reported exit 0 -- a green job that measured nothing.
    # Scoring records that do not exist is never worth doing.
    if [ "$P1" -ne 0 ]; then
        echo "FATAL: phase 1 ($MODE) produced no run records; refusing to"
        echo "  'score' them. Check the lines above for why inference.py did"
        echo "  not run."
        exit 1
    fi

    # MPLCONFIGDIR AND HOME ARE LOAD-BEARING, NOT TIDINESS.
    #
    # Run 9698787 hung here for 32 minutes with its last words "Matplotlib is
    # building the font cache; this may take a moment." CPU frozen at 256 s,
    # no file growth, phase 1 already exit 0 -- fifteen minutes of GPU work
    # stranded behind a font cache. matplotlib writes that cache under
    # $HOME/.cache/matplotlib, and inside `apptainer exec` HOME is the
    # submitting user's home, which is not writable from the execute node.
    # It does not fail; it stalls.
    #
    # `timeout` is the belt to that braces: scoring 22 records against
    # roberta-large is minutes, not an hour, so anything past 45 minutes is a
    # hang and should be reported as one rather than silently consuming the
    # allocation. Exit 124 from timeout is caught by the SC check below.
    mkdir -p "$PWD/.mplcache" "$PWD/.fakehome"
    timeout 2700 apptainer exec -B /staging \
        --env HF_HOME="$HF_HOME" \
        --env MPLCONFIGDIR="$PWD/.mplcache" \
        --env HOME="$PWD/.fakehome" \
        --env XDG_CACHE_HOME="$PWD/.mplcache" \
        "$EXTRACT_SIF" "$EXTRACT_PY" \
        scripts/flag_matrix.py "$SAMPLE" \
        --mode score \
        --run-dir "$RUN_DIR" \
        --out "$OUT"
    SC=$?
    [ "$SC" -eq 124 ] && echo "  FATAL: phase 2 ($MODE) TIMED OUT after 45min -- see the MPLCONFIGDIR note"
    echo "  phase 2 ($MODE) exit $SC"
    [ "$SC" -ne 0 ] && RC=1
    [ -f "$OUT" ] && { echo "  -- $OUT --"; cat "$OUT"; }
done

echo "exit: $RC  end: $(date)"
exit "$RC"
