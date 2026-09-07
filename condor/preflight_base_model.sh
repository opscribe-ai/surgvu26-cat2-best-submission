#!/bin/bash
export B=/staging/n/nkalthoff/hf_cache/hub/models--nvidia--Qwen2.5-VL-7B-Surg-CholecT50/snapshots/c1a01db98c72f4fca5ec671405325c4c841dcafd
set -u
echo "host: $(hostname)"
B=/staging/n/nkalthoff/hf_cache/hub/models--nvidia--Qwen2.5-VL-7B-Surg-CholecT50/snapshots/c1a01db98c72f4fca5ec671405325c4c841dcafd
echo "== base model visibility =="
[ -d "$B" ] && echo "DIR OK" || { echo "FATAL: base dir not visible from this node"; exit 75; }
for f in config.json model.safetensors.index.json preprocessor_config.json tokenizer_config.json; do
  [ -r "$B/$f" ] && echo "  $f readable ($(stat -Lc %s "$B/$f") bytes)" || echo "  $f MISSING/UNREADABLE"
done
n=$(ls "$B"/model-*.safetensors 2>/dev/null | wc -l)
echo "  shards: $n"
tot=0; for s in "$B"/model-*.safetensors; do tot=$((tot+$(stat -Lc %s "$s"))); done
echo "  total shard bytes: $tot"
echo "== python side =="
# transformers is NOT baked into surgvu26-train.sif -- condor/train_vlm.sh
# puts it on PYTHONPATH from a pre-staged deps dir. The first run of this
# preflight omitted that and failed on ModuleNotFoundError, proving only that
# a bare container lacks transformers. Same DEPS as the training job, so this
# checks the environment the training job will actually have.
export PYTHONPATH=/staging/n/nkalthoff/surgvu26/vlm_pypkgs2${PYTHONPATH:+:$PYTHONPATH}
python3 - <<'PY'
import json, os, sys
sys.path.insert(0,"src")
B=os.environ["B"]
c=json.load(open(os.path.join(B,"config.json")))
print("  arch:", c["architectures"], "| vocab:", c["vocab_size"], "| layers:", c["num_hidden_layers"])
from surgvu.evidence_vlm import DEFAULT_FINETUNE_BASE
print("  DEFAULT_FINETUNE_BASE matches:", DEFAULT_FINETUNE_BASE == B)
import transformers
print("  transformers", transformers.__version__)
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained(B, local_files_only=True)
print("  AutoProcessor loaded OK:", type(p).__name__)
tok = getattr(p, "tokenizer", None)
print("  tokenizer vocab:", len(tok) if tok else "n/a")
PY
rc=$?
echo "rc=$rc"
exit $rc
