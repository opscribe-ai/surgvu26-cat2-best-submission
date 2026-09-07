# v6 runbook

The order below is not a suggestion. Three of these steps depend on an
artefact the previous one produces, and two of the dependencies are invisible
until something fails hours in.

**v5.2 scored 0.8558 and is the locked fallback.** Nothing here can regress it:
v6 ships with `config/arbiter.json`'s `vlm_intents` empty, which is
byte-identical to the configuration that scored 0.8558.

## The curriculum

| stage | base | data | why |
|---|---|---|---|
| 0 | `nvidia/Qwen2.5-VL-7B-Surg-CholecT50` | — | already surgical: triplet F1 0.81 instrument. Free. |
| 1 | stage 0 | SSG-VQA, ~55k pairs | question variety over real surgical frames |
| 2 | **stage 1's adapter** | SurgVU v2, 16-frame | the task, the taxonomy, **the answer form** |

Stage 2 is LAST because BERTScore grades against SurgVU's answer forms. Stage 1
is ~2x the SurgVU corpus, so mixing them lets it dominate the output
distribution.

## Order of operations

### 1. Corpus regeneration (DONE)
`qa_pairs_v2.jsonl`, 377,557 records. Only `tool_absence` answers changed
(29,618) -- they were a random frames->answer mapping on 8.6% of the corpus.

### 2. Frame extraction at 16 (job 9710248, ~6h)
```
condor_submit condor/build_qa_frames.sub suffix=_v2 \
  qa_pairs=/staging/n/nkalthoff/surgvu26/qa_pairs_v2.jsonl \
  frames_per_window=16 max_per_intent=2000 workers=8 cpus=8 disk=60GB
```
16 frames is not optional: serving samples 16
(`evidence_vlm.DEFAULT_FRAMES_PER_CALL`) and **both previous adapters were
trained on 4**. That mismatch is silent -- nothing errors, the model is just
answering from a prompt shape it never saw.

### 3. Stage-1 frames (job 9710238, ~1h) -- PARALLEL with step 2
```
condor_submit condor/convert_stage1_frames.sub workers=8 cpus=8 disk=40GB
```
Produces `stage1_frames/` + `stage1_manifest.jsonl` + `stage1_splits.json`.

### 3b. Base-model preflight (VERIFIED 2026-08-27, job 9710283, rc=0)
```
condor_submit condor/preflight_base_model.sub          # 2 CPUs, no GPU, ~3 min
```
Proven on execute node e2018: the NVIDIA base resolves through its symlinked
blobs, four shards sum to 16,584,414,544 bytes, config reports
Qwen2_5_VLForConditionalGeneration / vocab 152064 / 28 layers,
DEFAULT_FINETUNE_BASE names that exact path, and
`AutoProcessor.from_pretrained(local_files_only=True)` loads as
Qwen2_5_VLProcessor.

**Tokenizer vocab is 151,665 against a model vocab_size of 152,064 -- a gap of
399, and that is CORRECT.** It is Qwen's embedding padding (152064 =
128 x 1188). The same gap killed merge job 9698669 when
`check_tokenizer_matches_embeddings` was written as an equality; it has been
directional since. Do not "fix" it.

Note the preflight needs `PYTHONPATH=/staging/n/nkalthoff/surgvu26/vlm_pypkgs2`
-- transformers is NOT baked into surgvu26-train.sif. Its first run failed on
exactly that and proved only that a bare container lacks transformers.

### 4. Stage-1 training -- can start as soon as step 3 lands
Needs no evidence cache, so it runs while step 5 is still going.
```
condor_submit condor/train_vlm.sub args="\
  --manifest /staging/n/nkalthoff/surgvu26/stage1_manifest.jsonl \
  --splits   /staging/n/nkalthoff/surgvu26/stage1_splits.json \
  --output-dir /staging/n/nkalthoff/surgvu26/models/v6_stage1 \
  --lora-r 32 --lora-alpha 64 --epochs 1 --skip-frame-check"
```
`--epochs 1`, not 2: 87k diverse examples is representation learning, not
fitting, and halving it buys ~6h on a serial chain.
`--skip-frame-check` only AFTER the build job exited 0 with its drop tally.
```
```

### 5. Evidence cache -- NOT NEEDED, and that was a deliberate choice
`train_vlm.attach_evidence` is ALL-OR-NOTHING: one sampled window missing from
the cache and the run raises, hours in.

The sample is fully determined by `qa_pairs_v2` + seed + `max_per_intent`, so
the overlap was computed BEFORE the extraction finished rather than discovered
afterwards:

| max_per_intent | windows | cached | missing | GPU cost |
|---|---|---|---|---|
| **2000** | 15,087 | 15,087 | **0** | **none** |
| 3000 | 19,023 | 10,631 | 8,392 (44%) | ~10h |

The job was originally submitted at 3000 and restarted at 2000. The 11,000
extra SurgVU records are not worth ten GPU-hours plus a longer extraction --
particularly since stage 1 contributes ~55k records either way. The existing
`evidence_cache.jsonl` covers the v2 sample exactly.

If a future rebuild DOES need new windows: `cache_evidence.py` resumes -- point
it at the same `--out` and it skips everything already cached. From cold it
took 18h for 15,087 windows.

### 6. Stage-2 training -- needs steps 2, 4 AND 5

**`args=` must NOT begin with the script path.** Both submit files already
supply it (`arguments = "scripts/train_vlm.py $(args)"`,
`arguments = "scripts/merge_and_quantise_vlm.py $(args)"`). Repeating it makes
argparse see a stray positional and exit 2 -- and because `OnExitRemove` only
clears on `ExitCode =?= 0`, the job REQUEUES and burns all 5 retries failing
the same way in ~2 min each. Cost me job 9712985. Start `args=` at `--`.
```
condor_submit condor/train_vlm.sub args="\
  --manifest /staging/n/nkalthoff/surgvu26/qa_frames_manifest_v2.jsonl \
  --splits config/splits_v2.json \
  --evidence-cache /staging/n/nkalthoff/surgvu26/evidence_cache.jsonl \
  --init-adapter /staging/n/nkalthoff/surgvu26/models/v6_stage1 \
  --output-dir /staging/n/nkalthoff/surgvu26/models/v6_stage2 \
  --lora-r 32 --lora-alpha 64 --epochs 2 --lr 1e-4 \
  --max-frames 16 --skip-frame-check"
```
`--max-frames 16`: MATCH SERVING. `evidence_vlm.DEFAULT_FRAMES_PER_CALL` is
16, and the manifest holds 16. An earlier draft of this runbook said 8 to save
wall clock -- that is WRONG, and I launched 9712986 on it before catching it.
Training at 8 does not fix the train/serve mismatch, it just moves it: the
merge guard enforces parity, so an 8-frame adapter forces PRODUCTION down to
8 and throws away half the visual evidence at inference. Cost is real -- 16
frames measured ~2x the step time of 8 -- but pay it here, not at serving.
```
```
`--init-adapter` is what makes this a curriculum rather than two unrelated
runs. It loads with `is_trainable=True`; without that peft loads every weight
frozen and the run trains NOTHING while looking completely healthy.
`run_training` aborts at 0 trainable parameters.

Lower LR than stage 1 (1e-4 vs 2e-4): enough to establish the answer form
without erasing stage 1's perception.

**`--evidence-cache` COMMITS YOU TO A SERVING FLAG.** An adapter trained with
evidence in its prompt MUST be served with `config/arbiter.json`'s
`vlm_evidence_context: true`, or it gets an empty context it never saw. It is
`false` today, which is correct for the CURRENTLY SHIPPED adapter (`vlm_lora`,
trained bare). Flipping it is part of shipping v6, not a bug fix.

### 6b. If stage 2 is evicted -- RESUME, do not restart

PROVEN 2026-08-28 (jobs 9713236 + 9713241): the torch 2.6 image resumes a
checkpoint, printing `resuming from checkpoint: ...`. The torch 2.5.1 image
WRITES checkpoints perfectly well; it only cannot READ them. So a run started
on the old image is still recoverable -- through the NEW one.

    condor_rm <cluster>            # stop the auto-retry FIRST
    condor_submit condor/train_vlm.sub sif=surgvu26-train-torch26.sif args="<same args>"

`max_retries = 5` means Condor requeues an evicted job on the DEFAULT image,
which then warns "cannot resume" and restarts from step 0 while burning a
retry. Recovery has to be a deliberate resubmit, not the automatic one.

### 7. Merge + quantise
```
condor_submit condor/merge_quantise.sub args="\
  --adapter-dir /staging/n/nkalthoff/surgvu26/models/v6_stage2 \
  --merged-dir /staging/n/nkalthoff/surgvu26/models/v6_merged_fp16 \
  --output-dir /staging/n/nkalthoff/surgvu26/models/qwen25vl-7b-v6-nf4"
```
The flag is `--output-dir`. It is NOT `--out-dir`: this file said `--out-dir`
until 2026-08-29, and `scripts/merge_and_quantise_vlm.py` has no such option,
so the command as written would have died on `unrecognized arguments` and then
requeued under `max_retries = 5` -- the Attempt-1 failure of 2026-08-28,
repeated verbatim.

**`condor_submit -dry-run` cannot catch that.** It expands submit macros and
stops; it never runs argparse, so it proves only that the script path appears
exactly once. Check flag names against the script's own `add_argument` calls
BEFORE spending a GPU job:

    grep -n "add_argument" scripts/merge_and_quantise_vlm.py

`--merged-dir` is passed explicitly and deliberately. Its default
(`models/vlm_merged_fp16`) is shared with the v5 merge, and stage 1 SKIPS
itself when `_looks_like_a_saved_model_dir(args.merged_dir)` is true -- so a
leftover v5 intermediate would be quantised as if it were v6, exit 0, and be
silently wrong. A v6-specific path makes that collision impossible.
`--output-dir` likewise does not overwrite the shipping `qwen25vl-7b-nf4`, so
v5.2 stays rollback-able.

Verified on 2026-08-29 before submitting cluster 9714365: dry-run resolved to
the script path exactly once, and all three flags exist in the script. Frame parity also passes with NO code change
now that stage 2 trains at 16: `training_config.json` says `max_frames: 16`
and `DEFAULT_FRAMES_PER_CALL` is 16.

The ONE thing still to change at merge is `config/arbiter.json`:
`vlm_evidence_context` must go `false` -> `true`, because stage 2 trained with
`--evidence-cache` (`training_config.json` says `evidence_context: true`).
Do NOT flip it early -- the currently shipped `vlm_lora` was trained bare, and
`false` is correct for it. It flips WITH the new adapter.
```
```
Base defaults to `DEFAULT_FINETUNE_BASE` and train/merge are pinned equal by a
test -- if they disagree `merge_and_unload()` STILL SUCCEEDS and emits one
model's base plus another's deltas, with no symptom but a quality drop.

### 8. Arm the VLM per-intent -- the "earn its way in" gate

This does NOT run on `ap2001`: `per_intent_table.py` imports `train_vlm` (torch)
and scores the router's answers with a BERTScorer over roberta-large. Submit it.
`condor/per_intent.sub` + `condor/per_intent.sh` exist for exactly this and are
CPU-only, so the job does not queue behind GPU work.

```
condor_submit condor/per_intent.sub args="\
  --eval-report /staging/n/nkalthoff/surgvu26/models/v6_stage2/eval_report.json \
  --evidence-cache /staging/n/nkalthoff/surgvu26/evidence_cache.jsonl \
  --manifest /staging/n/nkalthoff/surgvu26/qa_frames_manifest_v2.jsonl \
  --splits config/splits_v2.json \
  --max-eval-examples 300 --seed 0 \
  --label v6_stage2 \
  --output /staging/n/nkalthoff/surgvu26/models/v6_stage2/per_intent_v6.json"
```

**Four arguments in there are correctness-critical, and this file got all four
wrong before 2026-08-29** (it said `--max-eval-examples 2400` and passed neither
`--manifest` nor `--splits`). The script rebuilds the eval sample by re-running
train_vlm's own pipeline and joins it to the report BY INDEX, so the rebuild has
to reproduce the sample exactly:

- `--manifest` -- the script's own default is the **v1** manifest. v6 stage 2
  trained on `qa_frames_manifest_v2.jsonl`.
- `--splits` -- defaults to `None`. v6 stage 2 used `config/splits_v2.json`
  (train 115 / val 29 / heldout 11, mutually disjoint; the 300 eval items all
  come from `val`, verified 2026-08-29).
- `--max-eval-examples 300` and `--seed 0` -- job 9713019 passed NEITHER, so
  both took train_vlm's defaults, `DEFAULT_MAX_EVAL_EXAMPLES = 300` and
  `DEFAULT_SEED = 0`. `2400` rebuilds a different sample.

Getting any of them wrong does not corrupt the table: `case_id` is recomputed at
each index and a single mismatch aborts. It just wastes the job.

KNOWN RESIDUAL RISK: stage 2 ran with `--skip-frame-check`, so
`filter_records_with_frames` never ran over its val split, while
`build_eval_records` here always applies it. If a val frame is missing on disk
the two orderings diverge and the case_id guard aborts. That guard firing means
the frames are incomplete -- not that the arguments above are wrong.

Copy the printed `vlm_intents` into `config/arbiter.json` and set
`"mode": "per_intent"`. Arm intents ONE AT A TIME so a regression is
attributable to a named intent rather than to "the VLM". Empty list = v5.2.

### 9. Container rebuild + validate
Non-negotiable even under "no more testing": a container that crashes writes no
response, and **a missing response scores 0** on every question -- strictly
worse than any wrong answer.

**Pass `VLM_MODEL_SRC` EXPLICITLY. This is now a correctness requirement, not a
convenience.**

```
VLM_MODEL_SRC=/staging/n/nkalthoff/surgvu26/models/qwen25vl-7b-v6-nf4 \
    containers/build_submission.sh
```

`config/arbiter.json` was flipped to `vlm_evidence_context: true` on
2026-08-29, because v6 stage 2 trained WITH the evidence packet and
`check_evidence_parity` (rightly) refuses to merge an evidence-trained adapter
against a bare-serving config. That flip is correct FOR V6 AND ONLY FOR V6.

`build_submission.sh`'s `VLM_MODEL_SRC` default is
`$DEST/models/qwen25vl-7b-nf4` -- deliberately a path that normally does not
exist, so `--vlm` stages-if-present and is otherwise a loud no-op (see that
variable's own comment; do not "fix" it). The hazard is the OTHER v5 artifact:
`/staging/n/nkalthoff/surgvu26/models/qwen25vl-7b-nf4` DOES exist and holds the
v5.2 merge of the bare-trained `vlm_lora`. Build with that path now and the
image pairs a bare-trained model with `vlm_evidence_context: true` -- served an
evidence block it never saw once in training. Nothing raises. The only symptom
is the score. That is precisely the failure `check_evidence_parity` exists to
prevent, reintroduced one layer further down where no guard is watching.

The v5.2 artifacts are deliberately left in place (`qwen25vl-7b-nf4`,
`models/vlm_lora`) so v5.2 stays reproducible, so the rule is simply: name the
v6 path every time.

## The per-intent table UNDERSTATES the VLM. Read this before arming.

Measured answer lengths:

| corpus | p50 | p90 | max |
|---|---|---|---|
| SurgVU (our val) | **1 word** | 8 | 13 |
| GI (stage 1) | **21 words** | 36 | 624 |

SurgVU golds are bare nouns ("Cadiere Forceps", "Three", "Yes"). GI answers are
full sentences. Stage 2 runs last and pulls the model back toward brevity, but
some sentence habit will survive -- that is what stage 1 is FOR.

**The graded set rewards what our val corpus punishes.** Its golds are FIVE
human references and BERTScore takes the MAX:

    case127: "Uterine horn"
             "The organ being manipulated is the uterine horn."   <- GI style
             "Uterine horn is being manipulated."
             "The manipulation involves the uterine horn."
             "The organ in focus is the uterine horn."

A full-sentence answer matches references 2-5 as well as a bare noun matches
reference 1. Our val corpus has ONE reference, usually the bare noun, so
`scripts/per_intent_table.py` scores sentence-shaped answers BELOW what the
real grader would give them.

Consequence: the table is a LOWER BOUND on the VLM per intent, and the 2-sigma
bar makes it conservative twice over. An intent the table shows as roughly
tied is one the graded set probably favours the VLM on. Do not read a
near-miss as "the VLM lost".

Serving caps at 64 new tokens (`evidence_vlm.DEFAULT_MAX_NEW_TOKENS`).
Comfortable for SurgVU (max 13 words); it truncates GI's top ~5%, which is
another reason stage 2 goes last.

## Startup cost, MEASURED -- budget for it

`filter_records_with_frames` stats every frame of every record before the
`--max-train-examples` subsample. Measured on the v6 smoke train (job 9710300),
serial, on cold cephfs metadata:

    started 15:40:37 -> left stat phase 16:09:32   = 28m55s for 93,420 stats
    => 54 stats/second

| manifest | frames | stats | serial | threaded (32) |
|---|---|---|---|---|
| v1 | 4 | 93,420 | **29 min** (measured) | **~16+ min** (measured) |
| v2 | 16 | 373,680 | ~116 min | ~65 min |

**THREADING BARELY HELPS -- about 1.8x for 32 threads.** cephfs metadata is
effectively serialized server-side; more threads do not move it. This was
measured twice today (here and in the GI image converter, where 32 threads
were no faster than 8) and both times the thread count was a bad predictor of
throughput.

**So use `--skip-frame-check` on the real runs.** The check exists only to
tolerate a partially-extracted corpus. Once the extraction job has reported
its drop tally and exited 0, every `frame_paths` entry is present by
construction and the check is an hour of pure startup cost on the v2 manifest.
It defaults OFF and prints a warning when used; with it, a missing frame
crashes a training step instead of dropping a record, which is the trade.

Stage 2 would have spent nearly two hours on this before its first gradient
step, and it presents as "the model is loading slowly" -- 20 CPU-seconds
across 25 minutes of wall clock, memory flat, nothing in the log. It is fixed
(threaded, order-preserving), but the numbers are here because the same shape
will recur: **cephfs does ~54 cold stats/second, serial.**

Note the v5 training run did NOT show this, which briefly looked like evidence
against the diagnosis. It ran minutes after its frames were written, so the
metadata cache was warm. A run days later on the same manifest pays full price.

## Changing the serving frame count (stage 2 trains at 8)

Two frame counts exist and they are INDEPENDENT:

| constant | governs |
|---|---|
| `evidence_vlm.DEFAULT_FRAMES_PER_CALL` | the VLM |
| `config/perception.json` -> `decode.frames` | the CNNs |

Training stage 2 at `--max-frames 8` requires editing
**`DEFAULT_FRAMES_PER_CALL` to 8** and rebuilding the container. Do NOT
instead add `--vlm-frames 8` to the entrypoint: `check_frame_parity` reads the
CONSTANT, so a flag-based override would let the guard pass while serving used
a different number -- reintroducing the exact mismatch the guard exists to
catch. The container entrypoint does not pass `--vlm-frames` today, so the
constant governs the shipped image; keep it that way.

The CNN decode count stays 16 regardless -- the CNNs were trained on it and it
has nothing to do with the VLM's prompt.

## Traps, all paid for once already

- **A dry run wrote the production manifest.** Right size, right shape, every
  frame missing. Dry runs write `.dryrun` now.
- **`pool.map` drains its iterable eagerly.** The stage-1 converter's
  "streaming" generator pulled 59GB into RAM; job held at 9,766MB of 8GB.
- **Frame geometry.** `prepare_frame` SQUASHES to 512x512. Letterboxing stage 1
  would have given the curriculum two geometries.
- **`JPEG_QUALITY` is 90, not 92.** Import it, never restate it.
- **Two labels roots.** `labels_cat2` is correct (377,557 records);
  `labels_v2` yields exactly 2/3 of that and breaks the evidence-cache join.
- **The graded golds use GENERIC tool names.** case124 is `Cadiere Forceps`.
  "Large needle driver" appears only in QUESTIONS, all of them Yes/No. Do not
  "fix" the corpus toward commercial names.
