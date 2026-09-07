#!/bin/bash
set -uo pipefail
# Runs the answer-FORM experiment with the OFFICIAL metric.
#
#   condor/answer_form.sh <forms.json> <perception.json> <out.json>
#
# Everything the job needs is transferred into the scratch ROOT regardless of
# the directory it was submitted from, so the paths above are bare filenames,
# not repo paths.
#
# universe=vanilla, not container, for the same reason as condor/score.sub:
# this runs `apptainer exec` on the extract .sif, which cannot be nested inside
# a container-universe job.
#
# roberta-large loads from the /staging HF cache on CPU and is SLOW to start.
# Several minutes before the first number appears is normal, not a hang.

export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin
echo "host: $(hostname)  start: $(date)"

FORMS="${1:-answer_forms.json}"
PERCEPTION="${2:-perception_serving.json}"
OUT="${3:-answer_form_results.json}"

SIF=/staging/n/nkalthoff/surgvu26/surgvu26-extract.sif
PY=/staging/n/nkalthoff/surgvu26/env/bin/python3
SAMPLE=/staging/groups/bhaskar_opscribe/surgvu/cat2_sample
export HF_HOME=/staging/n/nkalthoff/surgvu26/hf_cache

# Pre-create the declared output. HTCondor holds a job whose transfer_output
# file is missing at exit, and a job that died on an unmounted /staging must be
# able to RETRY onto another node rather than sit held.
echo '{}' > "$OUT"

for path in "$SIF" "$SAMPLE" "$HF_HOME" "$FORMS" "$PERCEPTION"; do
    [ -e "$path" ] || { echo "FATAL: $path not visible on this node"; exit 42; }
done

echo "== inputs =="
sha256sum "$FORMS" "$PERCEPTION"

apptainer exec -B /staging --env HF_HOME="$HF_HOME" "$SIF" \
    "$PY" scripts/answer_form_eval.py "$SAMPLE" \
    --forms "$FORMS" --perception "$PERCEPTION" --out "$OUT"
RC=$?

echo "exit: $RC  end: $(date)"
exit "$RC"
