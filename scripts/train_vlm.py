"""LoRA fine-tune of Qwen2.5-VL-7B-Instruct on the SurgVU QA corpus (Task 4 of
the v5 plan3 VLM training pipeline; see
docs/design/plans/2026-08-25-v5-plan3-vlm-training.md).

WHY THIS EXISTS is stated at length in that plan's header and is not
repeated here beyond one sentence: every stock VLM measured on this task
scores BELOW a baseline that never looks at the video (0.4923-0.5743
against a 0.6959 zero-perception floor and 0.8766 shipped), because gold
answers reconstruct OUR label taxonomy -- 12 tools, 8 tasks -- which a
general VLM has never seen, and this fine-tune is the mechanism that closes
that gap.

WHAT THIS SCRIPT CONSUMES. `/staging/n/nkalthoff/surgvu26/qa_frames_manifest.
jsonl` -- Task 3's output: one JSON record per sampled QA example, each
carrying `case`/`part`/`t_start`/`t_stop`/`question`/`answer`/`intent`/
`provenance` (Task 2's shape) plus `frame_paths` (4 JPEGs already decoded
through `preprocess.prepare_frame`'s UI-band blur -- Task 3's job, not this
one's) and `frame_dir`. This script never opens a source video and never
re-derives an answer; it only reads the manifest, splits it, prompts over
it, and trains.

SPLIT DISCIPLINE -- TWO DIFFERENT GUARANTEES, NOT ONE CHECK REUSED TWICE.

  1. GRADED-CASE EXCLUSION (ruling R30). The manifest was already built by
     scripts/build_qa_pairs.py from labels_root cases with
     config/splits_v2.json's `heldout` list excluded -- so by construction
     this manifest should contain ZERO of the 11 graded cases. "Should" is
     exactly the word ruling R30 says not to trust: the variant head was
     trained on the graded cases (0.9011 contaminated vs 0.8681 clean)
     because an earlier exclusion step silently matched nothing. So this
     script re-derives the check from `config/splits_v2.json` INDEPENDENTLY
     of whatever build_qa_pairs.py did -- `load_case_universe` reads the
     file's train/val/heldout lists itself, normalises every id with
     `surgvu.sampling.normalize_case_id` (never string equality: the public
     sample dirs spell a case `case122`, this repo's labels spell it
     `case_122`), and refuses to proceed if the heldout list is empty or
     overlaps train/val -- a corrupted splits file would otherwise make
     every guard below vacuously pass. `verify_manifest_clean` is the actual
     leakage check against the real data file: it raises loudly if any
     manifest case normalises into the heldout set, or into a case
     `config/splits_v2.json` does not know at all. An earlier draft of this
     module instead built an intermediate `train|val|heldout` union and
     asserted that subtracting `heldout` from it removed a non-empty set --
     that assertion is a TAUTOLOGY (heldout is unioned in and then
     subtracted back out in the same breath, so it can never fail) and was
     deleted once a test proved it could never fire; see
     `verify_manifest_clean`'s own docstring for the replacement reasoning.

  2. TRAIN/EVAL SPLIT FOR THIS SCRIPT'S OWN HELD-OUT EVALUATION (ruling
     R28). "Split by CASE, never by example" -- examples from one 30s
     window are near-duplicates of each other (many QA records share one
     window: tool presence, task, organ, count, ... are all asked about the
     SAME clip), so an example-level split reports memorisation as
     accuracy, exactly the failure the variant head paid for once already.
     Rather than inventing a second random case split, `assign_case_split`
     reuses `config/splits_v2.json`'s own `train`/`val` partition directly:
     it is already case-disjoint by construction (a fact `load_case_universe`
     itself verifies), already fitted for reasonable per-class val coverage
     (see that file's own `meta` block), and reusing it means this script's
     eval set can never accidentally overlap its train set through a coding
     mistake in a from-scratch splitter.

THE EVIDENCE PACKET IS DELIBERATELY EMPTY AT TRAIN TIME -- A KNOWN GAP, NOT
AN OVERSIGHT. The plan asks for prompts that include "the CNN tool/task
probabilities, the YOLO detections, the variant block, the motion vector"
alongside the question, rendered by `evidence_vlm._EVIDENCE_RENDERERS` --
that is what makes serving an "Evidence VLM" rather than a stock one.
`render_training_prompt` below calls the exact same renderer,
`evidence_vlm.build_sampling_prompt`, so the shared Question:/"answer as
briefly as possible" skeleton is byte-identical between training and
serving. But it is called with an EMPTY context (`{}`) at train time, for a
reason that is not laziness: Task 3 extracted FRAMES only, never ran the
CNN/YOLO/motion/variant models over the 15,087 sampled windows to cache
their outputs, and computing that now is its own GPU pipeline outside this
task's scope. The alternative -- synthesising an evidence packet from the
GROUND-TRUTH tool/task labels this corpus was generated from -- was
considered and rejected: `evidence_vlm._render_tools_block` would then be
handed the literal answer to the tool_presence/tool_identity/count
questions asked about that SAME window, and the model would learn to read
the evidence line instead of the pixels, which is the opposite of what a
perception fine-tune is for. So this is an honest, bounded gap: this run
teaches the model the TAXONOMY and the QUESTION FORMS against real frames
with no evidence text; wiring a genuine (noisy, model-produced) evidence
packet into both training and serving is follow-up work, flagged here
rather than silently glossed over.

THE COUPLING THIS CREATES WITH `scripts/inference.py`, AND THE SWITCH THAT
KEEPS IT HONEST. Because this script trains against a BARE prompt, serving
must draft with the same bare prompt or the fine-tuned adapter meets prompt
text at inference it never saw at training -- a silent degradation, not a
crash, since nothing about a bad answer says "this is because the prompt
changed shape since training." `scripts/inference.py`'s `EvidenceVlmHandle.
sample` calls the identical renderer, `evidence_vlm.build_sampling_prompt`,
but has a full evidence packet (`perception`: CNN tool/task probabilities,
YOLO detections with timestamps, motion as calibrated language, the variant
call) available to pass instead of `{}`. Whether it does is gated by ONE
switch -- `config/arbiter.json`'s `vlm_evidence_context` key, overridable
per-run by `--vlm-evidence-context`/`--no-vlm-evidence-context` -- defaulted
to False (bare) in `scripts/inference.py`'s `DEFAULT_VLM_EVIDENCE_CONTEXT`
precisely because bare is what this script actually trains against today.
THE INVARIANT: the context passed here at training and the context passed
there at serving must match, and that switch is what keeps them matched. Do
not flip it to True without first retraining this script against a real
(non-label-derived) evidence packet; `tests/test_train_vlm.py`'s
`test_serving_and_training_prompts_match_under_the_shipped_default` fails
loudly if the shipped config drifts out of sync with what this script
trains on.

T4 / sm_75 COMPATIBILITY. The serving target is a T4 (16 GiB, sm_75 -- no
native bf16, no FlashAttention-2) or no GPU at all. Training may run on a
larger, newer card (an L40, an A100, ...) that DOES support bf16 and FA2,
and using them there would produce an adapter whose numerics were never
validated on the hardware it ships on. Two choices remove that risk instead
of hoping it does not matter:

  * `bnb_config_kwargs()` fixes `bnb_4bit_compute_dtype="float16"`, never
    "bfloat16" -- the dtype the 4-bit dequantised matmuls actually run in,
    regardless of what the training GPU could also support.
  * `ATTN_IMPLEMENTATION = "sdpa"` -- never "flash_attention_2", which does
    not run on sm_75 at all. `sdpa` runs identically on the training GPU and
    on a T4, so training and serving take the SAME attention code path
    rather than "fast on the training card, hope sdpa agrees at serving
    time".

The LoRA adapter itself is never merged into the base model: 4-bit NF4 is
not losslessly mergeable back to a dense checkpoint, and shipping the
adapter separately (loaded on top of a freshly-quantised base at serving
time, exactly as `evidence_vlm.DEFAULT_MODEL_DIR` already names the base
model id this fine-tune starts from) is both the smaller artefact and the
one whose numerics are re-derived on whatever hardware loads it, not frozen
into a merge computed elsewhere.

CHECKPOINTING FOR PREEMPTION. CHTC jobs get evicted. `--output-dir` is
expected to be an absolute path under `/staging` (matching every other
training job in this repo -- `variant_head.pt`, `tools_v2.pt`, ... all live
there, never in job scratch), so `transformers.Trainer`'s own
`save_steps`/`save_total_limit` checkpoints survive eviction, and
`_latest_checkpoint` auto-resumes from whatever is already on disk when the
controller resubmits the same command. This script does not implement its
own resume logic beyond calling `Trainer.train(resume_from_checkpoint=...)`
-- Trainer's own checkpointing (optimizer state, scheduler state, RNG state,
step count) is well-tested elsewhere and reimplementing it here would be
this script inventing a worse copy.

HYPERPARAMETER DEFAULTS, AND WHY. Every one below is a CLI argument, not a
hardcoded constant, so a follow-up sweep can override it without editing
this file; the values below are this run's starting point, not a claim they
are optimal.

  * `--epochs` (2): the sampled corpus caps at ~2000 examples per intent
    across ~14 intents (see scripts/build_qa_pairs.py's
    `DEFAULT_MAX_PER_INTENT`), and each intent's PHRASING space is small on
    purpose (a handful of paraphrases per shape). A LoRA adapter's low added
    capacity makes wholesale memorisation of 7B parameters unlikely, but a
    corpus this templated can still be memorised at the PHRASING level in
    more than a couple of passes -- two is the conservative middle of the
    2-3 range this kind of instruction-style fine-tune typically uses,
    leaving `--epochs` free for a follow-up sweep once one real run's eval
    curve exists to compare against.
  * `--lora-r` (16) / `--lora-alpha` (32): r=16 is the common middle of
    PEFT's own documented 8-64 range for instruction-tuning a 7B model;
    alpha=2r is the standard scaling convention (effective LoRA update
    magnitude is alpha/r times the low-rank product, and 2x is what most
    published QLoRA recipes use rather than 1x or 4x). Lower risks
    under-fitting the 12-tool/8-task taxonomy's combinatorial breadth;
    higher risks overfitting a corpus whose intents are individually
    templated.
  * `--lora-dropout` (0.05): PEFT's own LoRA default; a corpus this large
    (tens of thousands of examples) does not need aggressive dropout, but a
    small nonzero value costs nothing and guards against the phrasing-level
    memorisation `--epochs`'s reasoning already worries about.
  * `--batch-size` (1) / `--grad-accum` (16): each example carries 4 images
    at 512x512 -- expensive per-example activation memory on top of a
    4-bit 7B base plus its (unquantised) vision tower. Batch size 1 is the
    safe default given this script cannot know in advance which GPU
    HTCondor hands it beyond the `>=24000MB` floor `condor/train_vlm.sub`
    requires; gradient accumulation restores an effective batch of 16 for a
    stable gradient estimate without the peak-memory cost of a real batch
    of 16.
  * `--lr` (2e-4): the QLoRA paper's own default learning rate for
    LoRA-only updates against a frozen quantised base -- one to two orders
    of magnitude higher than full fine-tuning uses, because only the small
    adapter matrices are being updated.

WHAT COULD NOT BE RUN OR TESTED HERE. This login node has no `torch`,
`transformers`, `peft`, or `bitsandbytes` installed at all (confirmed:
`import torch` fails), and running heavy work on it is forbidden regardless.
Every function that touches any of those four is written so the import
happens INSIDE the function body, not at module scope -- so this whole file
stays IMPORTABLE and its data-plane logic stays TESTABLE here (see
tests/test_train_vlm.py: manifest loading, the split/exclusion guards, the
prompt/message assembly, the label-masking arithmetic, the dataset's
`__getitem__` against real tiny JPEGs -- PIL, unlike torch, IS installed on
this login node). What is NOT, and cannot be, exercised outside the training
container: model loading and quantisation, the collator's actual tensor
construction, the training loop itself, and the generation-based evaluation
against `surgvu.scoring.Scorer`. Those paths are written to the same
conventions already proven elsewhere in this repo (`evidence_vlm.call_vlm`'s
message shape, `evidence_vlm.DEFAULT_MODEL_DIR`'s exact base model id,
`condor/vlm_eval.sh`'s proven transformers/bitsandbytes install recipe) but
have not themselves been run. The controller submits `condor/train_vlm.sub`
to actually exercise them; a short `--max-train-examples`/`--epochs 1`
smoke invocation before committing to the full run is strongly recommended.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.evidence_vlm import (  # noqa: E402
    DEFAULT_FINETUNE_BASE as DEFAULT_BASE_MODEL, build_sampling_prompt,
)
from surgvu.sampling import normalize_case_id  # noqa: E402

# --------------------------------------------------------------------------
# defaults -- see the module docstring's "HYPERPARAMETER DEFAULTS" section
# for why each of these, not just what.
# --------------------------------------------------------------------------

DEFAULT_MANIFEST = "/staging/n/nkalthoff/surgvu26/qa_frames_manifest.jsonl"
DEFAULT_SPLITS = "config/splits_v2.json"
DEFAULT_OUTPUT_DIR = "/staging/n/nkalthoff/surgvu26/models/vlm_lora"

DEFAULT_EPOCHS = 2
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 16
DEFAULT_LR = 2e-4
DEFAULT_SEED = 0
DEFAULT_MAX_EVAL_EXAMPLES = 300
DEFAULT_SAVE_STEPS = 200
DEFAULT_LOGGING_STEPS = 10

IGNORE_INDEX = -100

#: Language-model decoder projections only -- Qwen2.5-VL's LM backbone is a
#: Qwen2-family decoder using these module names. The vision tower
#: (`visual.*`) is deliberately left frozen: the two CNNs already measure
#: real skill on this corpus's PERCEPTION, so this fine-tune's job is
#: teaching the LANGUAGE side to answer in this project's taxonomy and
#: question forms, not re-training vision features. Also keeps the adapter
#: small and its target-module names independent of the vision tower's own
#: (differently-shaped) attention blocks.
LORA_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

#: NEVER "flash_attention_2" -- see the module docstring's "T4 / sm_75
#: COMPATIBILITY" section. `sdpa` is supported on every CUDA card this
#: project touches, including the sm_75 T4 the model must serve on, so
#: training and serving take the identical attention code path.
ATTN_IMPLEMENTATION = "sdpa"


# ============================================================================
# torch-free: manifest, split discipline, prompt/message assembly. Every
# function in this section is exercised directly by tests/test_train_vlm.py
# on this login node.
# ============================================================================


def load_manifest(path):
    """One dict per non-blank line of `path` -- the manifest
    scripts/build_qa_pairs.py's --extract-frames mode writes."""
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_case_universe(splits_path):
    """(train_norm, val_norm, heldout_norm) -- every id in
    `config/splits_v2.json`'s three lists, normalised via
    `surgvu.sampling.normalize_case_id` and checked pairwise disjoint.

    Every list must be non-empty: a missing or empty `heldout` list is
    exactly the config corruption ruling R30 warns about (it would make
    every exclusion guard below vacuously pass instead of genuinely
    checking anything). Pairwise-disjoint because `assign_case_split`
    assumes the three sets partition the known case universe; a case
    appearing in two lists should fail HERE, with a clear cause, rather than
    surface later as an ambiguous "both train and val" error somewhere else.

    Breaks if: this reads the raw id strings instead of normalising each one
    with `normalize_case_id` before building the three sets.
    """
    data = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    sets = {}
    for key in ("train", "val", "heldout"):
        ids = data.get(key)
        if not ids:
            raise ValueError(
                "%s has no non-empty %r list -- cannot verify split "
                "discipline without it" % (splits_path, key))
        sets[key] = {normalize_case_id(c) for c in ids}
    train_norm, val_norm, heldout_norm = sets["train"], sets["val"], sets["heldout"]
    overlaps = {
        "train/val": sorted(train_norm & val_norm),
        "train/heldout": sorted(train_norm & heldout_norm),
        "val/heldout": sorted(val_norm & heldout_norm),
    }
    bad = {k: v for k, v in overlaps.items() if v}
    if bad:
        raise RuntimeError(
            "%s's train/val/heldout lists are not disjoint: %s"
            % (splits_path, bad))
    return train_norm, val_norm, heldout_norm


def verify_manifest_clean(records, train_norm, val_norm, heldout_norm):
    """Normalised case ids actually present in `records`, having raised
    loudly if any of them leaked from the graded/heldout set or came from a
    case `config/splits_v2.json` does not know about at all.

    THIS is the actual R30 guard against the REAL manifest file --
    independent of whatever scripts/build_qa_pairs.py already did to build
    it (ruling R30: verify independently, do not trust upstream). An
    earlier version of this module computed an intermediate
    `universe = train_norm | val_norm | heldout_norm` and asserted that
    subtracting `heldout_norm` from it removed a non-empty set -- that
    assertion is a TAUTOLOGY (heldout_norm is unioned into `universe` in the
    same breath it is subtracted back out, so it can never fail as long as
    `heldout_norm` is non-empty, which `load_case_universe` already
    guarantees on its own) and was deleted rather than kept as a
    reassuring-looking dead check. The check that can actually fail, and
    therefore the one worth having, is the one below: does THIS manifest's
    OWN case list, independently normalised, intersect the heldout set at
    all. On a correctly-built manifest that intersection is empty -- a
    PASSING verification, not a suspicious one; this function is not
    expected to remove anything, unlike scripts/build_qa_pairs.py's own
    `select_cases`, which starts from the full un-filtered label directory
    and must remove exactly 11.

    Breaks if: the `norm in heldout_norm` check is replaced by a raw
    membership test against the manifest's own un-normalised `case` strings
    (which would miss `case122` when `heldout_norm` holds `case_122`), or
    either `raise` below is downgraded to a print/log.
    """
    eligible = train_norm | val_norm
    present, leaked, unknown = set(), set(), set()
    for record in records:
        norm = normalize_case_id(record["case"])
        present.add(norm)
        if norm in heldout_norm:
            leaked.add(norm)
        elif norm not in eligible:
            unknown.add(norm)
    if leaked:
        raise RuntimeError(
            "HELDOUT LEAKAGE: %d graded case(s) found in the training "
            "manifest: %s. This is ruling R30's contaminated-run bug "
            "(0.9011 contaminated vs 0.8681 clean) -- refusing to train."
            % (len(leaked), sorted(leaked)))
    if unknown:
        raise RuntimeError(
            "%d case(s) in the manifest are not in config/splits_v2.json's "
            "train, val, OR heldout lists at all: %s -- this manifest does "
            "not match the splits file passed to this script."
            % (len(unknown), sorted(unknown)))
    return present


def assign_case_split(records, train_norm, val_norm):
    """(train_records, val_records) -- split BY CASE (ruling R28), reusing
    `config/splits_v2.json`'s own train/val partition rather than inventing
    a second, from-scratch random split. Every record's case must resolve to
    EXACTLY one of `train_norm`/`val_norm` (already proven disjoint by
    `load_case_universe`); a record whose case resolves to neither, or
    (defensively) to both, raises rather than being silently dropped or
    silently placed on both sides.

    Breaks if: cases are assigned by hashing the RECORD (e.g. `hash(question)
    % 2`) instead of by the case id, which would put near-duplicate examples
    from one window on both sides of the split.
    """
    train_records, val_records, unassigned = [], [], []
    for record in records:
        norm = normalize_case_id(record["case"])
        in_train, in_val = norm in train_norm, norm in val_norm
        if in_train and in_val:
            raise RuntimeError(
                "case %s (normalised %s) is in BOTH splits_v2.json's train "
                "and val lists -- the splits file is internally "
                "contaminated" % (record["case"], norm))
        if in_train:
            train_records.append(record)
        elif in_val:
            val_records.append(record)
        else:
            unassigned.append(record)
    if unassigned:
        raise RuntimeError(
            "%d manifest record(s) belong to case(s) absent from BOTH "
            "splits_v2.json train and val lists: %s -- refusing to silently "
            "drop or silently include them"
            % (len(unassigned), sorted({r["case"] for r in unassigned})))
    return train_records, val_records


def filter_records_with_frames(records, workers=32):
    """(kept, dropped_count) -- `records` whose every `frame_paths` entry
    exists on disk right now.

    Task 3's frame extraction may still be catching up to the manifest at
    the moment this script runs (JPEGs are written after the manifest JSONL,
    not atomically with it); this is the honest way to run against a
    partially-extracted corpus without crashing on a missing file deep
    inside a training step, and it reports the loss rather than silently
    training on fewer examples than it looks like.

    THREADED, BECAUSE THIS IS 373,680 STAT CALLS ON CEPHFS.

    Serially this dominated startup: the v2 manifest is 23,355 records x 16
    frames, and a cold cephfs stat runs about 65/second, so the check cost ~96
    MINUTES before a single gradient step. Measured on the v6 smoke train (job
    9710300): 20 CPU-seconds across 17 minutes of wall clock -- pure I/O wait,
    with the model not yet even loading. The v1 manifest at 4 frames cost ~24
    minutes the same way.

    It also runs BEFORE `--max-train-examples`, so a 48-record smoke run pays
    the full corpus's stat bill. That ordering is deliberate and stays: the
    subsample must be drawn from records that actually have frames, or it
    silently trains on fewer examples than it asked for.

    ORDER IS PRESERVED, AND THAT IS NOT COSMETIC. `sample_eval_records` draws
    a SEEDED `random.sample` over this function's output, so a result whose
    order depended on thread scheduling would select a different eval set on
    every run -- the same irreproducibility the threaded frame extraction had
    to avoid. `ThreadPoolExecutor.map` yields results in INPUT order, and the
    kept list is rebuilt by zipping those flags back against the original
    records, never by appending from workers.

    Breaks if: the existence check is dropped, or checks only the first path
    in `frame_paths` instead of every one.
    """
    records = list(records)
    if not records:
        return [], 0

    def _has_all_frames(record):
        return all(Path(p).exists() for p in record.get("frame_paths", ()))

    if workers and workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        # `records` is already materialised, so map's eager consumption of the
        # iterable costs nothing here -- unlike the zip-reading converters,
        # where it pulled payloads into memory.
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            flags = list(pool.map(_has_all_frames, records))
    else:
        flags = [_has_all_frames(record) for record in records]

    kept = [record for record, ok in zip(records, flags) if ok]
    return kept, len(records) - len(kept)


def sample_eval_records(val_records, n, seed):
    """Up to `n` records from `val_records`, deterministically for a fixed
    `seed`.

    Case-disjoint from training BY CONSTRUCTION -- `val_records` already
    came from `assign_case_split`'s val bucket, which never shares a case
    with the train bucket, so this function only bounds how many
    generation-based BERTScore calls the evaluation actually pays for; it
    adds no split logic of its own.

    Breaks if: `random.Random(seed)` is replaced by the unseeded global
    `random` module, which would make eval non-reproducible run to run.
    """
    if n is None or n <= 0 or len(val_records) <= n:
        return list(val_records)
    rng = random.Random(seed)
    return rng.sample(val_records, n)


def evidence_key(record):
    """The join key between `qa_frames_manifest.jsonl` and
    `evidence_cache.jsonl`: `(case, part, t_start, t_stop)`.

    Both files are produced from the SAME manifest, so the floats are
    bit-identical in practice -- but they are rounded to 6 decimals here
    anyway, because a join that silently misses on a last-bit difference
    would not raise, it would just leave records unmatched, and
    `attach_evidence` would then reject the whole run for a reason that has
    nothing to do with the data. Six decimals is microsecond precision on a
    seconds-valued timestamp: far finer than any real window boundary, far
    coarser than float noise.
    """
    return (str(record["case"]), str(record["part"]),
            round(float(record["t_start"]), 6), round(float(record["t_stop"]), 6))


def load_evidence_cache(path):
    """`{evidence_key: evidence_dict}` from `scripts/cache_evidence.py`'s
    output -- REAL, model-produced perception packets (CNN tool/task
    probabilities, motion_v2, yolo, variant, agree), not label-derived ones.

    That distinction is the whole point of this file's "THE EVIDENCE PACKET
    IS DELIBERATELY EMPTY AT TRAIN TIME" section: synthesising evidence from
    the labels would teach the model to read the evidence line instead of
    the pixels, because label-derived evidence is never wrong. Cached
    evidence is exactly as noisy at train time as at serve time.
    """
    cache = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cache[evidence_key(record)] = record.get("evidence") or {}
    if not cache:
        raise ValueError("%s held no evidence records" % (path,))
    return cache


def attach_evidence(records, cache):
    """Add `record["evidence"]` to every record, or raise.

    ALL-OR-NOTHING, AND THAT IS THE POINT. A partial join does not fail --
    it trains on a MIXTURE of evidence-bearing and empty prompts, which is
    strictly worse than either choice made deliberately: the model learns
    that the evidence block is sometimes absent and can be ignored, and
    nothing in the loss curve or the eval score says so. This project has
    now hit ten variants of that failure shape; this one is closed by
    refusing to start.

    Breaks if: the raise is softened to a warning-and-continue, or to
    filling unmatched records with `{}`.
    """
    missing = []
    for record in records:
        key = evidence_key(record)
        if key not in cache:
            missing.append(key)
        else:
            record["evidence"] = cache[key]
    if missing:
        raise KeyError(
            "%d of %d records have no cached evidence (first few: %s). "
            "Refusing to train on a mixture of evidence-bearing and empty "
            "prompts -- see attach_evidence's docstring. Re-run "
            "scripts/cache_evidence.py against THIS manifest first."
            % (len(missing), len(records), missing[:3]))
    return records


def render_training_prompt(question, evidence=None):
    """The exact text shown to the model for `question`.

    This is `evidence_vlm.build_sampling_prompt(question, {})`, called
    directly -- not a reimplementation kept in sync by hand. The empty `{}`
    context means every evidence-block renderer inside `build_sampling_prompt`
    finds nothing to render and the result is just the shared
    Question:/"answer as briefly as possible" skeleton every serving-time
    prompt also carries as its base. See the module docstring's "THE
    EVIDENCE PACKET IS DELIBERATELY EMPTY AT TRAIN TIME" section for why the
    evidence blocks themselves are not populated here.

    THIS `{}` IS HALF OF A COUPLING WITH `scripts/inference.py`. That file's
    `EvidenceVlmHandle.sample` calls this same `build_sampling_prompt` but
    with `perception` (the real evidence packet) available; whether it
    passes that or an empty dict is `config/arbiter.json`'s
    `vlm_evidence_context` switch (see the module docstring's "THE COUPLING
    THIS CREATES" section). The two calls must agree on which context they
    pass, or a model trained against this bare skeleton is served evidence
    text it never saw. Changing this call's `{}` to something non-empty
    without flipping that switch (and vice versa) breaks the invariant
    `tests/test_train_vlm.py`'s
    `test_serving_and_training_prompts_match_under_the_shipped_default`
    checks.

    Breaks if: this stops calling `build_sampling_prompt` and instead builds
    its own question string, which is exactly the two-copies-drift failure
    Task 1's module docstring exists to prevent.
    """
    return build_sampling_prompt(question, evidence or {})


def build_messages(question, images, answer=None, evidence=None):
    """The chat-format `messages` list for one QA example.

    Mirrors `evidence_vlm.call_vlm`'s content shape exactly: every entry in
    `images` becomes one `{"type": "image", ...}` content block, in order,
    followed by exactly one `{"type": "text", ...}` block carrying
    `render_training_prompt(question)` -- all inside a single user turn.
    `images` may be real `PIL.Image` objects (training/eval) or any
    placeholder value (a plain string, in a torch-free structural test) --
    this function never opens, decodes, or even looks inside an image, it
    only counts and positions them, so it is exercised directly by
    tests/test_train_vlm.py with no PIL objects at all.

    `answer=None` is the EVAL/inference shape: a user turn only, ready for
    `add_generation_prompt=True`. `answer` given appends the assistant turn
    a supervised training example needs -- `mask_prompt_tokens` is what
    makes sure training loss is computed on only that turn's tokens, never
    on the prompt.

    Breaks if: the text block is inserted BEFORE the image blocks instead of
    after, which would silently change what every downstream prompt looks
    like without changing this function's return type.
    """
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text",
                    "text": render_training_prompt(question, evidence)})
    messages = [{"role": "user", "content": content}]
    if answer is not None:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": str(answer)}],
        })
    return messages


def mask_prompt_tokens(token_ids, prompt_len, ignore_index=IGNORE_INDEX):
    """Supervised-fine-tuning labels for one example: `ignore_index` for the
    first `prompt_len` positions (the question + evidence-packet + chat
    scaffolding the model is never trained to reproduce), and the real token
    id for every position from `prompt_len` onward (the assistant's answer,
    the only span this fine-tune computes loss on).

    Pure list arithmetic -- no torch. This is the whole label-masking
    contract `Collator` needs; the real collator (torch-only, inside
    `Collator.__call__`) is a two-line wrapper that converts this function's
    list into a `torch.tensor`, so the invariant that actually matters (loss
    only on the answer span, never on the prompt) is proven here, without a
    GPU.

    Breaks if: the returned list uses `ignore_index` for the WRONG span
    (e.g. `token_ids[:prompt_len] + [ignore_index] * (len - prompt_len)`,
    which would train on the prompt and mask the answer -- the exact
    opposite of a supervised fine-tune), or `prompt_len` is silently clamped
    instead of raising when it is out of range.
    """
    if not (0 <= prompt_len <= len(token_ids)):
        raise ValueError(
            "prompt_len=%d out of range for %d token id(s)"
            % (prompt_len, len(token_ids)))
    return [ignore_index] * prompt_len + list(token_ids[prompt_len:])


def subsample_frames(frame_paths, max_frames):
    """`max_frames` evenly-spaced entries of `frame_paths`, endpoints kept.

    WHY THIS EXISTS. Extraction is the expensive, irreversible step: decoding
    16 frames per window took hours and cannot be narrowed afterwards without
    redoing it. Training cost, by contrast, scales with the frames actually
    fed -- v5's run took 11.2h at 4 frames, and 16 is 4x the vision tokens.
    Extracting 16 and CHOOSING at train time decouples the two, so the frame
    count can be set from the remaining clock rather than from what a decode
    job happened to write.

    EVENLY SPACED, NOT THE FIRST N. The frames span a 30-second window; taking
    a prefix would train the model on the first third of every clip and show
    it none of what the question is often about.

    THE SERVING SIDE MUST MATCH WHATEVER THIS RETURNS. `evidence_vlm.
    DEFAULT_FRAMES_PER_CALL` is the number sampled at inference, and a model
    trained on k frames served n != k is the exact silent mismatch v6 exists
    to fix (both earlier adapters were fitted on 4 and served 16). See
    tests/test_train_vlm.py's parity test.

    `max_frames` of 0/None, or more frames than exist, returns the list
    unchanged.
    """
    if not max_frames or max_frames <= 0 or max_frames >= len(frame_paths):
        return list(frame_paths)
    n = len(frame_paths)
    step = (n - 1) / float(max_frames - 1) if max_frames > 1 else 0.0
    picks = sorted({int(round(i * step)) for i in range(max_frames)})
    # Rounding can collide on short lists; top up in order so the count is
    # exactly max_frames rather than "however many survived deduplication".
    for i in range(n):
        if len(picks) >= max_frames:
            break
        if i not in picks:
            picks.append(i)
    return [frame_paths[i] for i in sorted(picks)[:max_frames]]


def load_frames(frame_paths):
    """`PIL.Image` (RGB) for every path in `frame_paths`, in order.

    The JPEGs on disk are what `cv2.imencode` wrote in
    scripts/build_qa_pairs.py's --extract-frames mode from BGR arrays;
    `cv2.imencode` writes a standard JPEG regardless of the input array's
    channel order (it performs the BGR->YCbCr conversion itself), so any
    standard JPEG reader -- including PIL, unlike raw OpenCV arrays -- reads
    back the correct true-colour image with no channel swap needed here.

    Imported lazily so this module stays importable where PIL is absent;
    PIL happens to BE installed on this project's login node (unlike torch),
    so this function, unusually for this file, is directly exercised by
    tests/test_train_vlm.py against real tiny JPEGs.
    """
    from PIL import Image
    return [Image.open(str(p)).convert("RGB") for p in frame_paths]


def bnb_config_kwargs():
    """The `BitsAndBytesConfig` kwargs this script always uses, as a plain
    dict of JSON-safe values -- no `torch`/`bitsandbytes` import, so this is
    directly testable on the login node where neither is installed.

    `bnb_4bit_compute_dtype` is the STRING `"float16"` here; the caller
    (`load_model_and_processor`, torch-only) converts it to `torch.float16`
    right before constructing the real config object. See the module
    docstring's "T4 / sm_75 COMPATIBILITY" section for why this is
    `"float16"` and never `"bfloat16"`, regardless of what the training
    GPU could also support.

    Breaks if: `"bnb_4bit_compute_dtype"` is changed to `"bfloat16"`.
    """
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": True,
    }


# ============================================================================
# torch-dependent. Every import of torch/transformers/peft/bitsandbytes is
# INSIDE a function body, per this file's own module-scope import discipline
# -- see the docstring's "WHAT COULD NOT BE RUN OR TESTED HERE" section.
# `QADataset` is the one exception worth calling out: it is a plain class,
# NOT a `torch.utils.data.Dataset` subclass, specifically so defining it does
# not require importing torch at module scope either.
# ============================================================================


class QADataset:
    """Map-style dataset over manifest records that have survived
    `filter_records_with_frames`.

    Deliberately NOT a `torch.utils.data.Dataset` subclass: PyTorch's
    `DataLoader` only requires an object supporting `__len__`/`__getitem__`
    (the documented "map-style dataset" protocol), and importing torch just
    to subclass its ABC would make this whole module uncollectable anywhere
    torch is absent -- which is everywhere except the training container.

    `__getitem__` loads the record's frames via `load_frames` (PIL only, no
    torch) and returns the raw question/answer/images; tokenisation and
    tensor construction happen in `Collator`, not here, so a `DataLoader`
    with `num_workers > 0` parallelises image decode across worker
    PROCESSES instead of serialising it on the main training step.
    """

    def __init__(self, records, max_frames=0):
        self.records = list(records)
        # 0 means "every frame the manifest lists". See subsample_frames: the
        # manifest is built at 16 and the training frame count is chosen from
        # the remaining clock, so this is a knob rather than a constant.
        self.max_frames = int(max_frames or 0)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return {
            "question": record["question"],
            "answer": record["answer"],
            "images": load_frames(
                subsample_frames(record["frame_paths"], self.max_frames)),
            # `.get`, not `[...]`: absent when --evidence-cache was not passed,
            # which is the shipped default and must keep rendering the empty
            # context this file's "DELIBERATELY EMPTY AT TRAIN TIME" section
            # describes. attach_evidence guarantees it is present for ALL
            # records or for none, never for some.
            "evidence": record.get("evidence"),
        }


class Collator:
    """`transformers.Trainer`'s `data_collator`: a list of `QADataset` items
    -> one batch dict of stacked tensors.

    For each example: tokenise the FULL conversation (user turn with images
    + assistant turn with the answer) and, separately, the PROMPT-ONLY
    conversation (the same user turn, `add_generation_prompt=True`, no
    assistant turn) with the SAME images -- the two token sequences share an
    identical prefix token-for-token (a chat template only ever APPENDS the
    assistant turn after the user turn and its generation marker; it never
    reorders anything before it), so the prompt-only encoding's length is
    exactly `prompt_len` for `mask_prompt_tokens`. Right-padded to the
    batch's own max length; `pixel_values`/`image_grid_thw` are concatenated
    (not stacked) along their leading dimension, matching how Qwen2-VL-family
    processors themselves pack a variable number of image patches per
    example.
    """

    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = (processor.tokenizer.pad_token_id
                             if processor.tokenizer.pad_token_id is not None
                             else processor.tokenizer.eos_token_id)

    def __call__(self, batch):
        import torch

        all_input_ids, all_labels = [], []
        pixel_values_list, grid_thw_list = [], []

        for example in batch:
            images = example["images"]
            evidence = example.get("evidence")
            full_messages = build_messages(
                example["question"], images, answer=example["answer"],
                evidence=evidence)
            prompt_messages = build_messages(
                example["question"], images, answer=None, evidence=evidence)

            full_encoded = self.processor.apply_chat_template(
                full_messages, tokenize=True, add_generation_prompt=False,
                return_dict=True, return_tensors="pt")
            prompt_encoded = self.processor.apply_chat_template(
                prompt_messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt")

            input_ids = full_encoded["input_ids"][0]
            prompt_len = int(prompt_encoded["input_ids"].shape[1])
            labels = torch.tensor(
                mask_prompt_tokens(input_ids.tolist(), prompt_len),
                dtype=torch.long)

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            pixel_values_list.append(full_encoded["pixel_values"])
            grid_thw_list.append(full_encoded["image_grid_thw"])

        lengths = [seq.shape[0] for seq in all_input_ids]
        max_len = max(lengths)

        def right_pad(seq, value):
            pad_amount = max_len - seq.shape[0]
            if pad_amount <= 0:
                return seq
            pad = torch.full((pad_amount,), value, dtype=seq.dtype)
            return torch.cat([seq, pad], dim=0)

        input_ids = torch.stack(
            [right_pad(seq, self.pad_token_id) for seq in all_input_ids])
        labels = torch.stack(
            [right_pad(seq, IGNORE_INDEX) for seq in all_labels])
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for row, length in enumerate(lengths):
            attention_mask[row, :length] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": torch.cat(pixel_values_list, dim=0),
            "image_grid_thw": torch.cat(grid_thw_list, dim=0),
        }


def trainable_parameter_report(model):
    """{"trainable": N, "total": M, "trainable_pct": ...} over
    `model.named_parameters()`. Called once after `get_peft_model` so the
    run's own log states the LoRA adapter's exact size rather than an
    estimate."""
    trainable, total = 0, 0
    for _, param in model.named_parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return {
        "trainable": trainable,
        "total": total,
        "trainable_pct": (100.0 * trainable / total) if total else 0.0,
    }


#: The Qwen2.5-VL chat template, vendored from Qwen/Qwen2.5-VL-7B-Instruct's
#: `chat_template.json`.
#:
#: WHY THIS FILE EXISTS, AND WHY IT IS NOT READ FROM THE CHECKPOINT.
#: `nvidia/Qwen2.5-VL-7B-Surg-CholecT50` ships NO `chat_template.json` at all,
#: so `AutoProcessor.from_pretrained` yields a processor whose
#: `apply_chat_template` raises "this processor does not have a chat template"
#: -- which is how the v6 smoke train died (job 9710359).
#:
#: THE OBVIOUS FIX IS THE DANGEROUS ONE. That checkpoint's
#: `tokenizer_config.json` DOES carry a `chat_template`, so falling back to it
#: looks correct and costs one line. It is the TEXT-ONLY Qwen2.5 template:
#: 2,427 characters, tools-aware, and containing no `vision_start`, no
#: `image` and no `video` handling whatsoever. Training through it would
#: render every prompt WITHOUT IMAGE TOKENS -- a vision model fine-tuned on
#: text alone, converging to a plausible loss curve, with nothing anywhere
#: reporting that the frames were never seen.
#:
#: The architecture is what decides this. The checkpoint's own config says
#: `Qwen2_5_VLForConditionalGeneration`, so the Qwen2.5-VL template is the
#: correct one and NVIDIA's packaging simply carries a stale text template.
#: Vendored into the repo rather than read from another snapshot so it is
#: version-controlled and travels with the code.
CHAT_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "config" / "qwen25vl_chat_template.jinja"


def load_vl_chat_template(path=None):
    """The vendored Qwen2.5-VL chat template, verified to handle vision.

    The `vision_start` assertion is the point: a template without it produces
    prompts with no image tokens, and every downstream symptom of that is
    indistinguishable from a model that simply learned poorly.
    """
    path = Path(path) if path is not None else CHAT_TEMPLATE_PATH
    template = path.read_text(encoding="utf-8")
    if "vision_start" not in template or "image" not in template:
        raise ValueError(
            "%s is not a vision chat template (no vision_start/image "
            "handling). Training through it would render prompts with no "
            "image tokens at all." % (path,))
    return template


def load_model_and_processor(base_model_id, lora_r=DEFAULT_LORA_R,
                             lora_alpha=DEFAULT_LORA_ALPHA,
                             lora_dropout=DEFAULT_LORA_DROPOUT,
                             adapter_dir=None, device_map="auto",
                             trainable=False):
    """(model, processor). `adapter_dir=None` builds a FRESH LoRA adapter
    over a newly 4-bit-quantised base (training entry point);
    `adapter_dir` given loads an already-trained adapter on top of the same
    quantised base instead (evaluation / --eval-only entry point) via
    `peft.PeftModel.from_pretrained`, never a merge (see the module
    docstring's "T4 / sm_75 COMPATIBILITY" section for why NF4 bases are not
    merged).

    `trainable=True` WITH `adapter_dir` IS THE CURRICULUM ENTRY POINT -- it
    CONTINUES training an existing adapter instead of starting a new one, which
    is how v6's stage 2 (SurgVU) builds on stage 1 (SSG-VQA) rather than
    discarding it.

    IT IS A SEPARATE FLAG BECAUSE PEFT'S DEFAULT IS THE DANGEROUS ONE.
    `PeftModel.from_pretrained` defaults to `is_trainable=False`: it loads the
    weights with `requires_grad=False` everywhere, for inference. Hand that to
    a Trainer and the run does not fail -- it trains zero parameters, the loss
    barely moves, checkpoints are written on schedule, and the "fine-tuned"
    output is a byte-for-byte copy of the adapter you started from. Nothing in
    the log says so. `run_training` therefore refuses to start when the
    trainable-parameter count is zero.
    """
    import torch
    from transformers import (
        AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig,
    )

    kwargs = dict(bnb_config_kwargs())
    compute_dtype = getattr(torch, kwargs.pop("bnb_4bit_compute_dtype"))
    bnb_config = BitsAndBytesConfig(bnb_4bit_compute_dtype=compute_dtype, **kwargs)

    processor = AutoProcessor.from_pretrained(base_model_id)
    if getattr(processor, "chat_template", None) is None:
        # See CHAT_TEMPLATE_PATH: the surgical base ships none, and the one in
        # its tokenizer_config is text-only. Attach the VL template its own
        # architecture requires.
        processor.chat_template = load_vl_chat_template()
        print("processor had no chat template; attached the vendored "
              "Qwen2.5-VL one (%s)" % CHAT_TEMPLATE_PATH.name)
    if "vision_start" not in (processor.chat_template or ""):
        raise ValueError(
            "the processor's chat template has no vision_start handling -- "
            "prompts would carry no image tokens. Refusing to train.")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_id, quantization_config=bnb_config, device_map=device_map,
        attn_implementation=ATTN_IMPLEMENTATION)

    if adapter_dir is None:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=int(lora_r), lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout), bias="none",
            task_type="CAUSAL_LM", target_modules=list(LORA_TARGET_MODULES))
        model = get_peft_model(model, lora_config)
    else:
        from peft import PeftModel

        if trainable:
            # Same preparation the fresh-adapter branch does: cast norms/head
            # to fp32 and make the quantised base's inputs require grad. A
            # continued run needs it as much as a fresh one -- without it the
            # LoRA weights sit on a base that never propagates gradients back
            # into them.
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model)
        model = PeftModel.from_pretrained(model, str(adapter_dir),
                                          is_trainable=bool(trainable))

    return model, processor


def _latest_checkpoint(output_dir):
    """The most recent `transformers.Trainer` checkpoint under `output_dir`,
    or None -- what makes a resubmitted job resume instead of restarting.
    """
    from transformers.trainer_utils import get_last_checkpoint

    if not Path(output_dir).is_dir():
        return None
    return get_last_checkpoint(str(output_dir))


def run_training(args, train_records):
    """Fine-tune the LoRA adapter over `train_records`, checkpointing to
    `args.output_dir` throughout so a preempted job resumes rather than
    restarts."""
    from transformers import Trainer, TrainingArguments

    init_adapter = getattr(args, "init_adapter", None)
    if init_adapter:
        print("CONTINUING from adapter %s (curriculum stage >= 2)" % init_adapter)
    model, processor = load_model_and_processor(
        args.base_model, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        adapter_dir=init_adapter, trainable=bool(init_adapter))

    report = trainable_parameter_report(model)
    print("trainable parameters: %d / %d (%.4f%%)"
          % (report["trainable"], report["total"], report["trainable_pct"]))
    # REFUSE TO TRAIN NOTHING. See load_model_and_processor's docstring: a
    # frozen adapter trains happily to completion and emits a copy of its
    # input. This is the one check that turns that into a failure.
    if report["trainable"] == 0:
        raise SystemExit(
            "0 trainable parameters -- the model would train nothing and "
            "write back a copy of what it started from. With --init-adapter "
            "this means is_trainable did not take; without it, that "
            "get_peft_model did not run.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trainable_parameters.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    # THE FRAME COUNT TRAVELS WITH THE ADAPTER.
    #
    # A model trained on k frames and served n != k is this project's single
    # most expensive failure shape: both pre-v6 adapters were fitted on 4 and
    # served 16, and nothing anywhere errored -- the model simply answered
    # from a prompt shape it had never seen. Documenting the requirement is
    # what was tried and what failed.
    #
    # Writing it INTO the adapter directory means the number cannot be
    # separated from the weights it describes, so the merge step can check it
    # (see merge_and_quantise_vlm.check_frame_parity) instead of trusting
    # whoever launches the job to remember.
    #
    # `resolved_frames` is 0 when every manifest frame is used -- meaningful
    # only alongside the manifest, which is why that is recorded too.
    (output_dir / "training_config.json").write_text(json.dumps({
        "max_frames": int(getattr(args, "max_frames", 0) or 0),
        "manifest": str(args.manifest),
        "base_model": str(args.base_model),
        "init_adapter": str(getattr(args, "init_adapter", "") or ""),
        "epochs": float(args.epochs),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lr": float(args.lr),
        "evidence_context": bool(getattr(args, "evidence_cache", None)),
    }, indent=2), encoding="utf-8")

    dataset = QADataset(train_records, max_frames=getattr(args, "max_frames", 0))
    if getattr(args, "max_frames", 0):
        print("feeding %d evenly-spaced frame(s) per record -- serving's "
              "evidence_vlm.DEFAULT_FRAMES_PER_CALL MUST match this"
              % args.max_frames)
    collator = Collator(processor)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=3,
        # NEVER bf16 here -- see bnb_config_kwargs's docstring. The compute
        # dtype the 4-bit base runs in and the dtype the trainer casts
        # activations/gradients to must agree, and both must be a dtype the
        # T4 this model serves on actually supports.
        bf16=False,
        fp16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=dataset, data_collator=collator)

    resume = None if args.no_resume else _latest_checkpoint(output_dir)

    # RESUME IS IMPOSSIBLE ON THIS TORCH, AND FAILING FAST BEATS BURNING
    # RETRIES ON IT.
    #
    # transformers >= 4.56 refuses to torch.load an optimizer state unless
    # torch >= 2.6 (CVE-2025-32434), and this project PINS torch 2.5.1+cu121
    # because the T4 it serves on is sm_75. So `trainer.train(
    # resume_from_checkpoint=...)` raises before the first step:
    #
    #   ValueError: Due to a serious vulnerability issue in `torch.load` ...
    #   we now require users to upgrade torch to at least v2.6
    #
    # That is worse than merely not resuming. Once ANY checkpoint exists, every
    # subsequent attempt dies on it in about four minutes -- so condor's
    # max_retries=5 spends all five attempts failing identically. Observed on
    # v6 stage 1 (job 9712725): a real failure at 3h49m, then two instant
    # retries that never reached a training step.
    #
    # Starting fresh loses the elapsed hours, which is bad; looping five times
    # without training loses them AND the retries. So: warn loudly, drop the
    # resume, and train.
    if resume is not None:
        import torch as _torch

        major, minor = (int(x) for x in _torch.__version__.split(".")[:2])
        if (major, minor) < (2, 6):
            print("WARNING: found checkpoint %s but torch %s cannot resume it "
                  "(transformers requires torch>=2.6 to torch.load optimizer "
                  "state, CVE-2025-32434; torch is pinned at 2.5.1 for sm_75). "
                  "STARTING FROM SCRATCH rather than failing every retry."
                  % (resume, _torch.__version__))
            resume = None
    if resume:
        print("resuming from checkpoint: %s" % resume)
    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print("wrote LoRA adapter + processor to %s" % output_dir)
    return model, processor


def run_eval(args, eval_records):
    """Generation-based evaluation over `eval_records` against
    `surgvu.scoring.Scorer` (BERTScore-F1) -- never token accuracy, which
    would flatter the model and measure the wrong thing.

    ONE reference per example, not five: `qa_forms`/`build_qa_pairs.py`
    generate a single canonical answer per record, unlike the official
    11-sample grading harness's five human references. `Scorer.score_one`
    still runs correctly against a one-element reference list (it is simply
    the F1 against that one string, with no MAX-over-many-references to
    take), but the resulting aggregate number is a RELATIVE signal for
    whether this fine-tune helped -- not a leaderboard-comparable score, and
    it should not be read as one.
    """
    import torch
    from surgvu.scoring import Scorer

    if not eval_records:
        print("no eval records; skipping evaluation")
        return None

    adapter_dir = args.adapter_dir or args.output_dir
    model, processor = load_model_and_processor(
        args.base_model, adapter_dir=adapter_dir)
    model.eval()

    scorer = Scorer()
    pairs = []
    for record in eval_records:
        images = load_frames(
            subsample_frames(record["frame_paths"], args.max_frames))
        # TRAIN/EVAL PARITY: the eval must render the same prompt shape the
        # collator does, or the reported bertscore_f1 measures a model on
        # inputs it never trained on.
        messages = build_messages(record["question"], images, answer=None,
                                  evidence=record.get("evidence"))
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        candidate = processor.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True)[0].strip()
        case_key = "%s|%s|%.3f" % (record["case"], record["part"], record["t_start"])
        pairs.append((case_key, candidate, [record["answer"]]))

    result = scorer.score_many(pairs)
    print("held-out CASE eval: n=%d  bertscore_f1=%.4f"
          % (len(pairs), result["aggregates"]["bertscore_f1"]))

    out_path = Path(args.output_dir) / "eval_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote %s" % out_path)
    return result


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _print_data_report(all_present, heldout_norm, train_records, val_records,
                       train_dropped_frames, val_dropped_frames, eval_records):
    print("distinct case(s) in manifest: %d" % len(all_present))
    print("heldout verification: 0 of %d known graded case(s) found in the "
         "manifest (checked against %s)" % (len(heldout_norm), sorted(heldout_norm)))
    print()
    print("train: %d record(s) [%d dropped for missing frame files]"
         % (len(train_records), train_dropped_frames))
    print("val:   %d record(s) [%d dropped for missing frame files]"
         % (len(val_records), val_dropped_frames))
    print("eval sample drawn from val: %d record(s)" % len(eval_records))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help="qa_frames_manifest.jsonl (Task 3's output).")
    parser.add_argument("--splits", default=DEFAULT_SPLITS,
                        help="config/splits_v2.json -- train/val/heldout.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help="HF model id or local path of the base VLM.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Where the LoRA adapter, processor, and "
                             "training checkpoints are written. Should be "
                             "an absolute /staging path so it survives "
                             "Condor eviction.")
    parser.add_argument("--adapter-dir", default=None,
                        help="[--eval-only] adapter to evaluate; defaults "
                             "to --output-dir.")
    parser.add_argument("--skip-frame-check", action="store_true",
                        help="Skip filter_records_with_frames entirely. That "
                             "check stats EVERY frame of EVERY record -- "
                             "measured at 29 minutes serial and still ~12 "
                             "minutes threaded on the v1 manifest, because "
                             "cephfs metadata scales poorly with threads. It "
                             "exists only to tolerate a PARTIALLY EXTRACTED "
                             "corpus; when the extraction job finished cleanly "
                             "every frame_paths entry is present by "
                             "construction and this is pure startup cost. Use "
                             "it ONLY after confirming the extraction reported "
                             "no drops -- with it, a missing frame surfaces as "
                             "a crash inside a training step instead of a "
                             "dropped record.")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="feed only this many EVENLY SPACED frames per "
                             "record (0 = all of them). The manifest is built "
                             "at 16; v5's run took 11.2h at 4 frames, so this "
                             "is how training cost is matched to the clock "
                             "without redoing extraction. WHATEVER IS USED "
                             "HERE, evidence_vlm.DEFAULT_FRAMES_PER_CALL must "
                             "match it at serving.")
    parser.add_argument("--init-adapter", default=None,
                        help="CONTINUE training from this adapter instead of "
                             "starting a fresh LoRA -- v6's curriculum: stage "
                             "2 (SurgVU) initialised from stage 1 (SSG-VQA). "
                             "Loaded with is_trainable=True; the run aborts if "
                             "that leaves 0 trainable parameters.")

    parser.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Per-device train batch size.")
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-eval-examples", type=int,
                        default=DEFAULT_MAX_EVAL_EXAMPLES)
    parser.add_argument("--max-train-examples", type=int, default=None,
                        help="Cap the training set (e.g. for a smoke run); "
                             "unset trains on every eligible record.")
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS)
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)

    parser.add_argument("--evidence-cache", default=None,
                        help="scripts/cache_evidence.py output "
                             "(evidence_cache.jsonl). When given, every train "
                             "and eval prompt carries the REAL, model-produced "
                             "perception packet for its window, and serving "
                             "must set config/arbiter.json's "
                             "vlm_evidence_context to true to match. Omitted "
                             "(the default) renders the empty context.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; evaluate --adapter-dir (or "
                             "--output-dir) against the val-case sample.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not auto-resume from the latest checkpoint "
                             "under --output-dir; start training fresh.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run only the torch-free data plane (manifest "
                             "load, split, exclusion guard, frame-presence "
                             "check) and report, then exit before any "
                             "torch/model code runs. Works on this login "
                             "node; nothing else in this script does.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    records = load_manifest(args.manifest)
    if not records:
        raise SystemExit("no records read from %s" % args.manifest)

    train_norm, val_norm, heldout_norm = load_case_universe(args.splits)
    present = verify_manifest_clean(records, train_norm, val_norm, heldout_norm)

    train_records, val_records = assign_case_split(records, train_norm, val_norm)
    if args.skip_frame_check:
        print("--skip-frame-check: NOT verifying that frame files exist. A "
              "missing frame will now crash a training step rather than drop "
              "its record. Only correct when the extraction finished cleanly.")
        train_dropped = val_dropped = 0
    else:
        train_records, train_dropped = filter_records_with_frames(train_records)
        val_records, val_dropped = filter_records_with_frames(val_records)

    if args.max_train_examples and len(train_records) > args.max_train_examples:
        rng = random.Random(args.seed)
        train_records = rng.sample(train_records, args.max_train_examples)

    eval_records = sample_eval_records(val_records, args.max_eval_examples, args.seed)

    # OPTION (b): real, model-produced evidence in the training prompt.
    #
    # Attached AFTER the split and the frame filter, so the all-or-nothing
    # check in attach_evidence is applied to exactly the records that will be
    # trained and evaluated on -- not to records that are about to be dropped,
    # which would fail a run for windows it was never going to use.
    #
    # BOTH train and eval get it, or the reported bertscore_f1 measures the
    # model on a prompt shape it never saw.
    if args.evidence_cache:
        cache = load_evidence_cache(args.evidence_cache)
        print("evidence cache: %d window(s) from %s"
              % (len(cache), args.evidence_cache))
        attach_evidence(train_records, cache)
        attach_evidence(eval_records, cache)
        print("evidence attached to %d train / %d eval record(s)"
              % (len(train_records), len(eval_records)))
        print("REMEMBER: serving must set config/arbiter.json's "
              "vlm_evidence_context to true for this adapter, or the model is "
              "served an empty context it never trained on.")
    else:
        print("no --evidence-cache: training with the EMPTY evidence context "
              "(the shipped default; vlm_evidence_context stays false)")

    _print_data_report(present, heldout_norm, train_records, val_records,
                       train_dropped, val_dropped, eval_records)

    if args.dry_run:
        print()
        print("--dry-run: stopping before any torch/model code runs")
        return 0

    if not args.eval_only:
        run_training(args, train_records)

    run_eval(args, eval_records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
