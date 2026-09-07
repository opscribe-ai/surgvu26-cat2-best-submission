#!/bin/bash
# Runs scripts/per_intent_table.py -- the ONLY sanctioned way to populate
# `vlm_intents` in config/arbiter.json.
#
# Modelled on condor/merge_quantise.sh, which works, and deliberately kept
# CPU-ONLY: this job never loads the VLM. It loads the router (regex) and a
# BERTScorer over roberta-large, and scores ~300 router answers. Requesting a
# GPU would only queue it behind real GPU work for no gain.
#
# Run from the repo root, after: mkdir -p logs
#   condor_submit condor/per_intent.sub args="..."

export PYTHONUNBUFFERED=1
set -uo pipefail

echo "host: $(hostname)  start: $(date)"
umask 002

# Same staging guard as condor/merge_quantise.sh: exit 75 fast on a node where
# +WantStagingMount did not take effect, so it reschedules elsewhere instead of
# failing confusingly deep inside the run.
if ! ls /staging/n/nkalthoff/surgvu26 >/dev/null 2>&1; then
    echo "FATAL: /staging is not mounted on $(hostname). +WantStagingMount" \
         "did not take effect; exiting fast so this reschedules elsewhere." >&2
    exit 75
fi

# roberta-large (what BERTScorer loads) is already in this cache -- verified
# 2026-08-29 -- so the run stays offline.
export HF_HOME=/staging/n/nkalthoff/surgvu26/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DEPS=/staging/n/nkalthoff/surgvu26/vlm_pypkgs2
[ -d "$DEPS" ] || { echo "FATAL: $DEPS not visible on $(hostname)"; exit 75; }

# bert_score is NOT in $DEPS (checked 2026-08-29) and src/surgvu/scoring.py
# imports it lazily inside Scorer._bert_scorer, so the failure would otherwise
# surface only after the router had already been scored. peft comes along
# because per_intent_table.py imports train_vlm, which imports it at module
# level even though no adapter is loaded here.
NEWDEPS="$(pwd)/.per_intent_deps"
mkdir -p "$NEWDEPS"
#
# THE SECOND INSTALL IS NOT OPTIONAL, and leaving it out is what killed job
# 9714366 and then 9714371: `bert_score/score.py` imports BOTH
# `matplotlib.pyplot` (line 7) and `pandas` (line 9) at MODULE scope, so
# --no-deps yields a bert_score that cannot even be imported, and the failure
# lands in the environment check before any work starts.
#
# `pandas matplotlib` -- BOTH, exactly as written. condor/train_vlm.sh already
# had this right and proved it on job 9713019; 9714366 failed because this line
# was written from scratch instead of copied, and 9714371 failed again because
# the fix added only the package the FIRST traceback happened to name. The
# second name was one line further down the same file.
#
# The two-call shape is the point: --no-deps for the packages that would
# otherwise resolve and download their OWN torch, which then shadows the
# container's through PYTHONPATH; then a normal dep-resolving install for the
# pure-python stack bert_score needs.
python3 -m pip install --no-cache-dir --no-deps --target "$NEWDEPS" \
    "bert_score" "peft==0.20.0" \
 && python3 -m pip install --no-cache-dir --target "$NEWDEPS" \
    pandas matplotlib
PIP_RC=$?
if [ "$PIP_RC" -ne 0 ]; then
    echo "FATAL: pip install bert_score/peft/matplotlib into $NEWDEPS failed with exit $PIP_RC"
    exit "$PIP_RC"
fi
# Never let the pip target shadow the container's own torch stack.
rm -rf "$NEWDEPS"/torch "$NEWDEPS"/torchvision "$NEWDEPS"/nvidia \
       "$NEWDEPS"/triton "$NEWDEPS"/numpy "$NEWDEPS"/numpy.libs \
       "$NEWDEPS"/transformers

export PYTHONPATH="$DEPS:$NEWDEPS${PYTHONPATH:+:$PYTHONPATH}"

echo "== effective environment, AFTER PYTHONPATH is set =="
python3 - <<'PY'
import torch, transformers, bert_score, matplotlib
print("torch", torch.__version__, "from", torch.__file__)
print("transformers", transformers.__version__)
print("matplotlib", matplotlib.__version__)
# getattr, not bert_score.__version__: the attribute is not part of the
# package's documented surface, and an AttributeError here would fail the run
# for a reason that has nothing to do with whether the import works.
print("bert_score", getattr(bert_score, "__version__", "(no __version__)"))
from bert_score import BERTScorer  # the symbol surgvu.scoring actually uses
print("BERTScorer importable:", BERTScorer is not None)
PY
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "FATAL: effective-environment import check failed (exit $RC)"
    exit "$RC"
fi

python3 "$@"
RC=$?
echo "exit: $RC  end: $(date)"
exit $RC
