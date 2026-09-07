#!/bin/bash
set -uo pipefail
# Scores the VLM-fallback measurement with the OFFICIAL metric.
#
#   condor/vlm_score.sh <dir holding the outputs of condor/vlm_eval.sh>
#
# Two things get scored and they answer different questions:
#
#   the candidate files   what the SET scores when the two constant-answered
#                         cases are replaced -- shipped vs generic fallback vs
#                         VLM. Run through scripts/score_sample.py, so the
#                         per-case rows are directly comparable to every other
#                         number in this repo.
#   the pairs file        every row including the paraphrases, each candidate
#                         against the SOURCE case's real references. Scored
#                         with surgvu.scoring.Scorer, which is the same metric
#                         score_sample.py uses; it is a separate call only
#                         because these rows are not one-per-case.
#
# roberta-large loads from the /staging HF cache on CPU and is SLOW to start.
# Several minutes before the first number appears is normal, not a hang.

export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin
echo "host: $(hostname)  start: $(date)"

RESULTS="${1:-.}"
SIF=/staging/n/nkalthoff/surgvu26/surgvu26-extract.sif
PY=/staging/n/nkalthoff/surgvu26/env/bin/python3
SAMPLE=/staging/groups/bhaskar_opscribe/surgvu/cat2_sample
export HF_HOME=/staging/n/nkalthoff/surgvu26/hf_cache

for path in "$SIF" "$SAMPLE" "$HF_HOME"; do
    [ -e "$path" ] || { echo "FATAL: $path not visible on this node"; exit 42; }
done

RC_ALL=0
for label in shipped fallbackopen vlmcontext vlmnocontext vlmoff vlmon; do
    file="$RESULTS/${label}_candidates.json"
    [ -s "$file" ] || { echo "skipping $label: no $file"; continue; }
    echo
    echo "================================================================"
    echo "SCORING $label"
    echo "================================================================"
    cat "$file"
    apptainer exec -B /staging --env HF_HOME="$HF_HOME" "$SIF" \
        "$PY" scripts/score_sample.py "$SAMPLE" "$file" --label "$label"
    rc=$?
    [ "$rc" -ne 0 ] && RC_ALL=1
done

echo
echo "================================================================"
echo "SCORING the paraphrase rows against their source case's gold"
echo "================================================================"
apptainer exec -B /staging --env HF_HOME="$HF_HOME" \
    --env PAIRS="$RESULTS/vlm_fallback_pairs.json" "$SIF" "$PY" - <<'PY'
import json, os, sys
from collections import defaultdict
from statistics import mean

sys.path.insert(0, "src")
from surgvu.scoring import Scorer

rows = json.load(open(os.environ["PAIRS"]))
scorer = Scorer()
arms = ["fallback", "vlmcontext", "vlmnocontext", "shipped"]

print("%-14s %-9s %-40s %s" % ("id", "group", "question",
                               "  ".join("%12s" % a for a in arms)))
print("-" * 130)
by_group = defaultdict(lambda: defaultdict(list))
for row in rows:
    scores = {}
    for arm in arms:
        candidate = row["candidates"].get(arm)
        # A None candidate is the VLM DECLINING, and at serving time a decline
        # is the generic sentence. Scoring it as the fallback is what the
        # container would actually emit; scoring it as 0 would be a fiction.
        if not candidate:
            candidate = row["candidates"]["fallback"]
        scores[arm] = scorer.score_one(candidate, row["references"])["bertscore_f1"]
        by_group[row["group"]][arm].append(scores[arm])
    print("%-14s %-9s %-40s %s"
          % (row["id"], row["group"], row["question"][:38],
             "  ".join("%12.4f" % scores[a] for a in arms)))
    for arm in arms:
        print("      %-12s %r" % (arm, row["candidates"].get(arm)))

print()
print("%-10s %5s %s" % ("group", "n", "  ".join("%12s" % a for a in arms)))
print("-" * 80)
for group in sorted(by_group):
    values = by_group[group]
    print("%-10s %5d %s"
          % (group, len(values[arms[0]]),
             "  ".join("%12.4f" % mean(values[a]) for a in arms)))
PY
rc=$?
[ "$rc" -ne 0 ] && RC_ALL=1

echo "exit: $RC_ALL  end: $(date)"
exit "$RC_ALL"
