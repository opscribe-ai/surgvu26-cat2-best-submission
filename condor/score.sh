#!/bin/bash
set -uo pipefail
# Scores one or more candidate-answer files with the OFFICIAL metric
# (BERTScore-F1, roberta-large, max over the five references).
#
#   condor/score.sh <candidates.json> [<candidates.json> ...]
#
# Runs under universe=vanilla, not container: the scoring environment is the
# extract .sif plus a venv in /staging, and `apptainer exec` cannot be nested
# inside a container-universe job.
#
# roberta-large is loaded from the /staging HF cache on CPU and is SLOW to
# start -- several minutes before the first number appears. That is normal.

export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin
echo "host: $(hostname)  start: $(date)"

SIF=/staging/n/nkalthoff/surgvu26/surgvu26-extract.sif
PY=/staging/n/nkalthoff/surgvu26/env/bin/python3
SAMPLE=/staging/groups/bhaskar_opscribe/surgvu/cat2_sample
export HF_HOME=/staging/n/nkalthoff/surgvu26/hf_cache

# Fail fast where /staging is unmounted or the scoring env is missing, so the
# retry moves to a node where it is not.
for path in "$SIF" "$SAMPLE" "$HF_HOME"; do
    [ -e "$path" ] || { echo "FATAL: $path not visible on this node"; exit 42; }
done

RC_ALL=0
for candidates in "$@"; do
    label="$(basename "$candidates" _candidates.json)"
    echo
    echo "================================================================"
    echo "SCORING $candidates  as '$label'"
    echo "================================================================"
    cat "$candidates"
    apptainer exec -B /staging --env HF_HOME="$HF_HOME" "$SIF" \
        "$PY" scripts/score_sample.py "$SAMPLE" "$candidates" --label "$label"
    rc=$?
    echo "score exit $rc for $label"
    [ "$rc" -ne 0 ] && RC_ALL=1
done

echo "exit: $RC_ALL  end: $(date)"
exit "$RC_ALL"
