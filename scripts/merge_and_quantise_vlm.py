"""Merge the SurgVU VLM LoRA adapter into its fp16 base and quantise the
merged checkpoint to 4-bit NF4, producing the ONE self-contained directory
`src/surgvu/evidence_vlm.py::_load_model` can actually load.

WHY THIS SCRIPT EXISTS. A LoRA fine-tune of Qwen2.5-VL-7B-Instruct completed
(cluster 9697941, exit 0): held-out CASE eval bertscore_f1 = 0.9092, against
0.4923-0.5743 for every stock VLM measured on this task (see
scripts/train_vlm.py's module docstring for the full accounting). The
adapter lives at `/staging/n/nkalthoff/surgvu26/models/vlm_lora` (the final
adapter, at the top level -- `checkpoint-2200` etc. under it are
intermediate Trainer checkpoints, not what this script reads). Two
independent, already-measured facts block shipping it as-is (see this
project's `docs/design/2026-08-24-v5-evidence-pipeline/
vlm-weight-staging-report.md`):

  1. The only base-model artefact on disk is fp16, measured 16 GB
     (`/staging/n/nkalthoff/surgvu26/hf_cache`, 5 safetensors shards).
     3.46 GB (today's image) + 16 GB + the adapter lands near 19.7 GB
     against a documented 10 GB ceiling.
  2. `evidence_vlm.py::_load_model` (frozen -- this script does not touch
     `src/`) calls a bare `AutoModelForImageTextToText.from_pretrained(
     model_dir, device_map=...)`: no `peft.PeftModel` application, no
     `quantization_config`. It can only ever load a checkpoint that is,
     BY ITSELF, already both fine-tuned and quantised.

This script produces exactly that artefact.

A DELIBERATE, USER-APPROVED REVERSAL OF `scripts/train_vlm.py`'S OWN NOTE.
That file's docstring says: "The LoRA adapter itself is never merged into
the base model: 4-bit NF4 is not losslessly mergeable back to a dense
checkpoint, and shipping the adapter separately ... is both the smaller
artefact and the one whose numerics are re-derived on whatever hardware
loads it." That was a real design tradeoff, not a correctness claim -- and
it assumed a SERVING loader that could apply an adapter on top of a
freshly-quantised base. `evidence_vlm._load_model` is not that loader, and
is frozen: it takes one directory and calls bare `from_pretrained` on it.
Given that constraint plus the 10 GB image ceiling (which rules out shipping
the fp16 base unquantised), merging is the only shape that satisfies both
requirements at once. The user has weighed this tradeoff explicitly and
chosen to merge; this note exists so a future reader does not "fix" this
back to the adapter-only design without re-deriving why that design assumed
a different loader than the one this project actually ships.

THE TWO-STAGE MEMORY STRATEGY, AND WHY IT IS TWO STAGES NOT ONE.

  STAGE 1 -- MERGE (CPU, fp16). Load the base at full fp16 precision (NOT
  through `train_vlm.py`'s `BitsAndBytesConfig` path -- merging a LoRA delta
  into an already-4-bit-packed base is not how `peft.merge_and_unload()` is
  designed to work; the standard QLoRA-style merge reloads the base at full
  precision, merges there, and quantises the RESULT separately, which is
  exactly what stage 2 does). Apply the trained adapter via
  `peft.PeftModel.from_pretrained`, call `merge_and_unload()`, save the
  merged fp16 checkpoint to an intermediate directory. Runs on CPU
  deliberately: `merge_and_unload()` is elementwise weight arithmetic
  (`base_weight += (lora_B @ lora_A) * scaling` per targeted Linear layer),
  not a forward pass, so it needs no CUDA kernel -- keeping this stage on
  CPU means the ~16 GB fp16 model never has to fit on a GPU at all, leaving
  the GPU this job requests entirely free for stage 2. THE PROCESSOR SAVED
  ALONGSIDE THE MERGED WEIGHTS COMES FROM `adapter_dir` (the trained
  adapter's OWN tokenizer/processor files), never from the raw base
  snapshot -- confirmed, not assumed, to matter: the trained adapter
  directory's `tokenizer.json`/`added_tokens.json` may not be byte-identical
  to the raw base's, and a vocab/embedding mismatch does not raise on its
  own (see "THE TOKENIZER/EMBEDDING GUARD" below). Pulling the tokenizer
  from the wrong place would be a silent, undetectable-by-inspection bug.

  STAGE 2 -- QUANTISE (CUDA, NF4). Reload the merged fp16 checkpoint FRESH
  (the stage-1 process's Python objects are deleted and garbage-collected
  first, so this stage's peak memory is not stacked on top of stage 1's)
  through a `BitsAndBytesConfig` -- the SAME recipe `scripts/train_vlm.py`'s
  `bnb_config_kwargs()` used for training (imported from there, not
  retyped, so this checkpoint's quantisation cannot silently drift from what
  the 0.9092 eval was actually measured against), and save the quantised,
  self-contained result -- config.json (carrying its own
  `quantization_config`), tokenizer/processor files, and the packed NF4
  safetensors shards -- to the FINAL output directory. Requires CUDA:
  bitsandbytes' 4-bit packing has no CPU implementation in the version this
  project installs.

  STAGE 3 -- SIZE CHECK. Walk the final directory, sum real bytes, compare
  against the documented 10 GB image ceiling minus the current 3.46 GB
  image. STOPS HERE, loudly, non-zero exit, BEFORE verification, if it does
  not fit -- there is no point generating a demo answer from an artefact
  that cannot be uploaded regardless.

  STAGE 4 -- VERIFY. Reload the FINAL directory with the EXACT bare call
  `evidence_vlm._load_model` uses (no `quantization_config`, no
  `torch_dtype` -- this script's own quantisation step is responsible for
  everything `_load_model` needs to already be baked into
  `output_dir/config.json`), and run one real generation over a real
  preprocessed surgical frame from Task 3's training manifest -- not a
  synthetic tensor. Prints the question, the gold answer, and the model's
  output, and RAISES if the output is empty. This is the check that
  actually matters: a checkpoint can save cleanly and report a plausible
  size while still generating garbage (or nothing) if the merge silently
  dropped the adapter's effect, or the quantised save/reload round-trip lost
  the weights -- this project has produced seven variants of exactly that
  "looks like it worked and did nothing" failure shape already.

THE TOKENIZER/EMBEDDING GUARD (`check_tokenizer_matches_embeddings`). A
second, more specific failure mode than an empty generation: the trained
adapter directory carries its own `added_tokens.json`, so its tokenizer is
not guaranteed to match the raw base model's -- and if the SAVED merged
model's embedding matrix ever disagreed with the tokenizer saved alongside
it, a token id encoded by the tokenizer could index a ROW OF THE EMBEDDING
MATRIX THAT MEANS SOMETHING ELSE ENTIRELY. That does not raise; it produces
plausible-looking text from the wrong embedding, which is worse than a
crash given that this project's serving path swallows VLM errors and falls
back to the router's own answer -- a garbage-but-non-crashing VLM output
would ship completely silently. This check compares `len(processor.
tokenizer)` against `model.get_input_embeddings().weight.shape[0]` and
raises if they disagree; it runs twice -- once right after the merge
(`merge_adapter_into_base`, before spending GPU time quantising something
already broken) and again in `verify_checkpoint` against the FINAL,
RELOADED-FROM-DISK checkpoint (proving the saved artefact itself is
consistent, not just the in-memory objects mid-merge).

OFFLINE, ALWAYS. `find_local_snapshot_dir` below never falls back to a
network fetch -- it raises `FileNotFoundError` if the base model is not
already present in the local HF cache. Every `from_pretrained` call in this
script also passes `local_files_only=True`, and `main()` sets
`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` before any heavy import, as defence
in depth: this script must never attempt a network fetch, and neither may
the artefact it produces, since serving time reuses the identical
`from_pretrained(model_dir)` call this script verifies against.

WHAT COULD NOT BE RUN OR TESTED HERE. This login node has no torch,
transformers, peft, or bitsandbytes installed (confirmed elsewhere in this
project; heavy compute is also forbidden on this node regardless). Every
function that touches any of those four imports it INSIDE the function
body, not at module scope, so this whole file stays IMPORTABLE and its
path-resolution / size-accounting / fits-under-ceiling logic stays TESTABLE
here (see tests/test_merge_and_quantise.py). What is NOT, and cannot be,
exercised on this login node: the actual merge, the actual quantisation, and
the actual verification generation. Those run only inside
condor/merge_quantise.sub on a GPU compute node.
"""
import argparse
import os
import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from surgvu.evidence_vlm import (                                    # noqa: E402
    DEFAULT_FINETUNE_BASE as DEFAULT_BASE_MODEL_ID,
    DEFAULT_FRAMES_PER_CALL as SERVING_FRAMES_PER_CALL, build_sampling_prompt,
)
from train_vlm import (                                              # noqa: E402
    DEFAULT_MANIFEST, DEFAULT_OUTPUT_DIR as DEFAULT_ADAPTER_DIR,
    bnb_config_kwargs, filter_records_with_frames, load_frames, load_manifest,
)

# --------------------------------------------------------------------------
# defaults
# --------------------------------------------------------------------------

DEFAULT_HF_CACHE_DIR = "/staging/n/nkalthoff/surgvu26/hf_cache"

#: Intermediate, DELETED BY DEFAULT once quantisation succeeds (see
#: `main`'s `--keep-merged`) -- this directory exists only to break the
#: merge/quantise pipeline's memory footprint into two stages; it is not
#: itself a deliverable and left on /staging it is ~16 GB of dead weight.
DEFAULT_MERGED_DIR = "/staging/n/nkalthoff/surgvu26/models/vlm_merged_fp16"

#: THE final artefact's path. Chosen to be BYTE-IDENTICAL to
#: `containers/build_submission.sh`'s own `VLM_MODEL_SRC` default
#: (`"${VLM_MODEL_SRC:-$DEST/models/qwen25vl-7b-nf4}"`, where
#: `DEST=/staging/n/nkalthoff/surgvu26`) and to `scripts/inference.py`'s
#: `DEFAULT_VLM_MODEL_DIR` basename (`models/qwen25vl-7b-nf4`) -- the
#: staging mechanism for `--vlm`'s weights already exists (see
#: `docs/design/2026-08-24-v5-evidence-pipeline/
#: vlm-weight-staging-report.md`) and was built to pick this exact path up
#: with ZERO further wiring once a real checkpoint exists here. Overridable
#: via `--output-dir` for a differently-named build (e.g. a `-v2` sweep)
#: without editing this file, matching every other path default in this
#: repo's scripts.
DEFAULT_OUTPUT_DIR = "/staging/n/nkalthoff/surgvu26/models/qwen25vl-7b-nf4"

#: Binary units (GiB, 1024**3), matching how `du -sh` -- the tool every
#: prior size figure in this project's docs was measured with (16 GB base,
#: 3.46 GB image) -- reports. Using the SAME convention as those existing
#: numbers, not decimal GB, is what makes this script's own size report
#: directly comparable to them rather than silently 7% optimistic.
BYTES_PER_GIB = 1024 ** 3

#: This build's current image size. Kept at 3.46 after re-deriving it from
#: measurements, and the derivation is worth recording because two errors
#: cancel here:
#:
#:   * The shipped surgvu26-submission.sif is 3,464,957,952 bytes, which is
#:     3.227 GiB -- not 3.46. The original 3.46 was the DECIMAL GB figure
#:     (3.465 GB) carried over as though it were GiB, making this constant
#:     ~0.23 GiB conservative.
#:   * That conservatism is now spent, deliberately. The VLM's serving
#:     dependencies were added to both container recipes on 2026-08-26
#:     (transformers 4.57.6 / accelerate 1.14.0 / bitsandbytes 0.50.1),
#:     measured at 563 MB uncompressed against the staged vlm_pypkgs2 tree,
#:     which lands at roughly 0.25 GiB once squashfs compresses it.
#:
#: 3.227 + ~0.25 = ~3.48, so 3.46 remains an honest figure rather than a
#: stale one. Recorded rather than left as a coincidence, because the next
#: person to add a layer will otherwise re-derive it from the wrong base.
#:
#: Still not measured here: this script never builds the image (that is the
#: controller's job, and this project's standing rule is that builds run on
#: a compute node, never the login node), so this is arithmetic feeding a
#: projection. `check_fits_under_ceiling` is a GO/NO-GO signal on that
#: projection, not a substitute for the built image's own size.
CURRENT_IMAGE_SIZE_GIB = 3.46

#: The documented Grand Challenge submission ceiling.
IMAGE_SIZE_CEILING_GIB = 10.0

#: The grader's card. Named in the challenge configuration: "NVIDIA T4 Tensor
#: Core GPU (16 GiB VRAM)". This is what binds a SIDECAR model, since sidecar
#: weights never enter the image.
T4_VRAM_GIB = 16.0

#: Left for activations, the KV cache and image tokens. A 512x512 frame is
#: ~324 vision tokens after Qwen2.5-VL's 2x2 merge, so 16 frames is ~5,184
#: tokens and ~0.30 GB of KV -- but the vision encoder's own activations
#: during prefill are the larger and less predictable term, and running a card
#: out of memory mid-case writes no answer at all, which scores 0.
VRAM_HEADROOM_GIB = 4.0


# ============================================================================
# torch-free: path resolution, preflight checks, size accounting, the
# fits-under-ceiling check, and picking a real verification example. Every
# function in this section is exercised directly by
# tests/test_merge_and_quantise.py on this login node.
# ============================================================================


def hf_cache_folder_name(model_id):
    """`"Qwen/Qwen2.5-VL-7B-Instruct"` -> `"models--Qwen--Qwen2.5-VL-7B-
    Instruct"` -- the Hugging Face Hub cache's own folder-naming convention
    (`models--<owner>--<name>`), reproduced here as pure string logic so
    resolving a local snapshot never needs `huggingface_hub` imported (it
    is not installed on this login node, and this function must be testable
    without it).

    Breaks if: `"/"` is replaced by something other than `"--"`, which would
    look for a folder name the real HF cache never actually creates.
    """
    return "models--" + str(model_id).replace("/", "--")


def find_local_snapshot_dir(model_id, hf_cache_dir):
    """The resolved local snapshot directory for `model_id` inside
    `hf_cache_dir`'s `hub/` tree, or raise `FileNotFoundError` -- NEVER
    falls back to a network fetch, which is the whole point of this
    function existing rather than just calling
    `AutoModel.from_pretrained(model_id, cache_dir=hf_cache_dir)` and
    trusting `local_files_only` alone to stop a fetch.

    A "valid" snapshot is a directory under `snapshots/` that contains a
    `config.json`; the most recently modified one wins (there is normally
    exactly one -- multiple would mean two revisions were pulled, and the
    newest is the reasonable default).

    Breaks if: the `config.json` existence check is dropped, which would
    return an empty or partially-downloaded snapshot directory as if it
    were complete.
    """
    # AN ABSOLUTE PATH IS ALREADY A SNAPSHOT DIRECTORY -- TAKE IT AS ONE.
    #
    # `hf_cache_folder_name` is pure string logic on a HUB ID ("owner/name" ->
    # "models--owner--name"). Handed a filesystem path it produces nonsense:
    # "/staging/n/nkalthoff/hf_cache/hub/models--nvidia--..." becomes
    # "models----staging--nkalthoff--hf_cache--hub--models--nvidia--...", a
    # folder no cache ever creates, and the lookup fails with a message
    # blaming a missing model rather than a mis-typed one.
    #
    # v6's base (`evidence_vlm.DEFAULT_FINETUNE_BASE`) is exactly that: an
    # absolute snapshot path, because the surgical checkpoint lives in a
    # different cache tree than the one HF_HOME points at. The `config.json`
    # check below is applied to it too -- an absolute path still has to be a
    # real, complete snapshot, it just does not get name-mangled first.
    candidate = Path(model_id)
    if candidate.is_absolute():
        if not (candidate / "config.json").exists():
            raise FileNotFoundError(
                "%r is an absolute path but has no config.json -- an "
                "incomplete snapshot, or a directory that is not one."
                % (model_id,))
        return candidate

    hub_dir = Path(hf_cache_dir) / "hub" / hf_cache_folder_name(model_id)
    snapshots_dir = hub_dir / "snapshots"
    if not snapshots_dir.is_dir():
        raise FileNotFoundError(
            "no local snapshot for %r under %s (looked for %s) -- this "
            "script never falls back to a network download; stage the "
            "base model into the HF cache first."
            % (model_id, hf_cache_dir, snapshots_dir))
    candidates = [p for p in snapshots_dir.iterdir()
                  if p.is_dir() and (p / "config.json").exists()]
    if not candidates:
        raise FileNotFoundError(
            "%s has snapshot directories but none contain a config.json "
            "(an incomplete or corrupted local cache?)" % (snapshots_dir,))
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def verify_adapter_dir(adapter_dir):
    """`adapter_dir` if it looks like a real, complete LoRA adapter
    directory (an `adapter_config.json` plus an `adapter_model.safetensors`
    or `.bin`); raises `FileNotFoundError` otherwise. A preflight check, so
    a missing or half-written adapter fails in one clear line here rather
    than deep inside `peft.PeftModel.from_pretrained`.

    Breaks if: the weight-file check is dropped, which would accept a
    directory holding only `adapter_config.json` (e.g. one still being
    written by a training job) as if the adapter itself were present.
    """
    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(
            "%s has no adapter_config.json -- not a LoRA adapter directory"
            % (adapter_dir,))
    has_weights = (any(adapter_dir.glob("adapter_model.safetensors"))
                   or any(adapter_dir.glob("adapter_model.bin")))
    if not has_weights:
        raise FileNotFoundError(
            "%s has adapter_config.json but no adapter_model.safetensors/"
            ".bin -- incomplete adapter directory" % (adapter_dir,))
    return adapter_dir


def directory_size_bytes(path):
    """Total real bytes on disk under `path`, summed file by file.

    Broken symlinks are skipped rather than raising (a stale intermediate
    artefact should not crash a size report); every other file's size is
    counted via `os.path.getsize`, which follows a live symlink to its
    target's real size -- this project's own `save_pretrained` output is
    plain files, but this function is written to not assume that.

    Breaks if: this switches to summing `st_blocks`/allocated-block size
    instead of `st_size` (apparent size) -- either is a defensible choice in
    general, but ONLY apparent size is comparable to the `du -sh`-derived
    3.46 GB/16 GB figures this script's own ceiling check is measured
    against; mixing the two units would make the fits-under-ceiling
    comparison silently wrong.
    """
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            file_path = os.path.join(dirpath, name)
            if os.path.islink(file_path) and not os.path.exists(file_path):
                continue
            try:
                total += os.path.getsize(file_path)
            except OSError:
                continue
    return total


def check_fits_under_ceiling(model_dir_bytes,
                              current_image_gib=CURRENT_IMAGE_SIZE_GIB,
                              ceiling_gib=IMAGE_SIZE_CEILING_GIB,
                              sidecar=False):
    """{"model_gib", "current_image_gib", "ceiling_gib", "budget_gib",
    "projected_total_gib", "margin_gib", "fits"} -- pure arithmetic over
    already-measured sizes, so this is exercised directly on this login
    node without needing the real 16 GB base or a real quantised checkpoint
    on disk.

    `fits` uses `<=`, not `<`: a checkpoint landing EXACTLY on the ceiling
    is not a size failure. `budget_gib` (the 10 GB ceiling minus the
    current 3.46 GB image) and `margin_gib` (budget minus this checkpoint's
    own size) are reported separately so a caller can see not just PASS/FAIL
    but how much room was left, or by how much it missed.

    Breaks if: `fits` is computed as `model_gib <= ceiling_gib` instead of
    `projected_total_gib <= ceiling_gib` -- that would silently forget to
    add the current image's own 3.46 GB, passing a checkpoint that actually
    blows the ceiling once the image it ships inside is accounted for.
    """
    model_gib = model_dir_bytes / BYTES_PER_GIB

    # SIDECAR WEIGHTS ARE NOT IN THE IMAGE, so the image ceiling does not
    # apply to them. Grand Challenge extracts an optional model tarball to
    # /opt/ml/model/ at runtime, separate from the container, with no
    # documented size limit of its own.
    #
    # This gate refused the int8 checkpoint (8.76 GiB, projected 12.22 against
    # 10.00) and was RIGHT to under the assumption it was written with -- that
    # weights ship inside the image. That assumption is what changed: int8 plus
    # the judge is 14.18 GiB baked in, so both moved out, and the image drops
    # to ~3.5 GiB. Refusing a sidecar artefact for exceeding a ceiling it is
    # not subject to would block the architecture that exists to escape it.
    #
    # The check is not simply skipped: VRAM still binds. The grader's T4 has
    # 16 GiB, so a sidecar model is measured against that instead, which is
    # the constraint that actually applies at runtime.
    if sidecar:
        budget_gib = T4_VRAM_GIB - VRAM_HEADROOM_GIB
        projected_total_gib = model_gib
        return {
            "model_gib": model_gib,
            "current_image_gib": 0.0,
            "ceiling_gib": budget_gib,
            "budget_gib": budget_gib,
            "projected_total_gib": projected_total_gib,
            "margin_gib": budget_gib - model_gib,
            "fits": model_gib <= budget_gib,
            "gate": "sidecar/VRAM",
        }

    budget_gib = ceiling_gib - current_image_gib
    projected_total_gib = current_image_gib + model_gib
    return {
        "model_gib": model_gib,
        "current_image_gib": current_image_gib,
        "ceiling_gib": ceiling_gib,
        "budget_gib": budget_gib,
        "projected_total_gib": projected_total_gib,
        "margin_gib": budget_gib - model_gib,
        "fits": projected_total_gib <= ceiling_gib,
    }


def _looks_like_a_saved_model_dir(path):
    """True if `path` already holds a complete `from_pretrained`-loadable
    checkpoint (a `config.json` plus at least one weights shard) -- what
    `main` uses to decide whether a re-submitted, previously-evicted job can
    skip straight past a stage it already finished, the same resume
    discipline `scripts/train_vlm.py`'s `_latest_checkpoint` gives the
    training job.

    Breaks if: only `config.json` is checked and the weights-shard check is
    dropped, which would treat a directory `save_pretrained` was killed
    partway through (config written, shards not yet flushed) as already
    done.
    """
    path = Path(path)
    if not (path / "config.json").exists():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


def pick_verification_example(manifest_path=DEFAULT_MANIFEST):
    """The first record in `manifest_path` whose frame JPEGs all exist on
    disk right now -- a REAL preprocessed surgical frame (Task 3's
    UI-band-blurred JPEGs from `qa_frames_manifest.jsonl`), not a synthetic
    image, for `verify_checkpoint` to generate against.

    Reuses `train_vlm.filter_records_with_frames`'s own existence predicate
    rather than re-deriving it a second time -- but deliberately PER RECORD
    (`filter_records_with_frames([record])`, one at a time, short-circuiting
    on the first hit) rather than calling it once over the WHOLE manifest.
    The manifest this project ships has 23,355 records at 4 frame paths
    each -- filtering all of it up front would cost ~93k `Path.exists()`
    calls against `/staging` just to then take element `[0]`, which is
    minutes of wall clock spent for a function that only ever needs ONE
    record. Reusing the same predicate function keeps the "what counts as
    present" logic from drifting between this script and
    `scripts/train_vlm.py`; only the iteration strategy differs.

    Breaks if: this reverts to `filter_records_with_frames(records)` called
    once over the entire list (correct, but reintroduces the O(N) stat
    sweep this docstring exists to explain avoiding), or the loop returns
    `record` directly instead of `kept[0]` (which would skip the existence
    check entirely and just return the first record unchecked).
    """
    records = load_manifest(manifest_path)
    for record in records:
        kept, _dropped = filter_records_with_frames([record])
        if kept:
            return kept[0]
    raise RuntimeError(
        "no record in %s has all its frame files present on disk -- "
        "cannot pick a real verification example" % (manifest_path,))


# ============================================================================
# torch-dependent. Every import of torch/transformers/peft/bitsandbytes is
# INSIDE a function body, matching this project's own module-scope import
# discipline (see `evidence_vlm.py`/`train_vlm.py`) -- so this file stays
# importable, and everything above stays testable, where none of those four
# packages exist.
# ============================================================================


def ensure_vl_chat_template(processor, where):
    """Attach the Qwen2.5-VL chat template if `processor` has none, and refuse
    a template that cannot render images.

    THE SERVING CONSEQUENCE IS WHY THIS IS HERE AND NOT ONLY IN TRAINING.
    Both merge stages `save_pretrained` this processor into the directory the
    container ships, and `evidence_vlm.call_vlm` calls
    `processor.apply_chat_template` on it at inference. A processor saved
    without a template makes EVERY case raise, every response missing, and a
    missing response scores 0 -- strictly worse than any wrong answer.

    `nvidia/Qwen2.5-VL-7B-Surg-CholecT50` ships no chat_template.json, and the
    one in its tokenizer_config is the text-only Qwen2.5 template with no
    vision handling at all, so the adapter directories derived from it inherit
    the same gap. See train_vlm.CHAT_TEMPLATE_PATH for the full measurement.
    """
    from train_vlm import load_vl_chat_template

    if getattr(processor, "chat_template", None) is None:
        processor.chat_template = load_vl_chat_template()
        print("%s had no chat template; attached the vendored Qwen2.5-VL one"
              % where)
    if "vision_start" not in (processor.chat_template or ""):
        raise ValueError(
            "%s's chat template has no vision_start handling -- a model saved "
            "with it would render prompts carrying no image tokens, and every "
            "served case would answer from text alone." % where)
    return processor


def check_frame_parity(adapter_dir, serving_frames, strict=True):
    """Raise if the adapter was trained on a different frame count than
    serving will feed it. Returns the trained count (or None if unrecorded).

    THIS PROJECT'S MOST EXPENSIVE FAILURE SHAPE, MADE LOUD.

    Both pre-v6 adapters were fine-tuned on 4 frames and served 16. Nothing
    errored anywhere: `transformers` accepts any number of images, the model
    generates fluent text, the container passes validation, and the only
    symptom is a score. Documenting the requirement is exactly what was tried,
    and it is what failed.

    `scripts/train_vlm.py` writes `training_config.json` INTO the adapter
    directory so the number cannot be separated from the weights it describes.
    This runs at merge time because that is the one step every adapter passes
    through on its way to being served.

    A MISSING FILE IS NOT AN ERROR. Adapters trained before this existed have
    no such file, and refusing to merge them would break the fallback path for
    a check that is advisory for them. It warns loudly and returns None.

    `max_frames: 0` means "every frame the manifest lists", which is only
    meaningful alongside that manifest -- so it is reported, not compared.
    """
    config_path = Path(adapter_dir) / "training_config.json"
    if not config_path.exists():
        print("WARNING: %s has no training_config.json -- cannot verify that "
              "its training frame count matches serving's %d. Adapters "
              "predating v6 are expected to look like this."
              % (adapter_dir, serving_frames))
        return None
    try:
        trained = int(json.loads(config_path.read_text(encoding="utf-8"))
                      .get("max_frames", 0) or 0)
    except (OSError, ValueError, TypeError) as exc:
        print("WARNING: could not read %s (%s) -- frame parity unverified"
              % (config_path, exc))
        return None
    if trained == 0:
        print("frame parity: adapter trained on EVERY manifest frame "
              "(max_frames=0); serving samples %d. Verify against the "
              "manifest's own frames-per-window." % serving_frames)
        return 0
    if trained != serving_frames:
        message = (
            "FRAME MISMATCH: adapter %s was trained on %d frame(s) per record "
            "but serving samples %d (evidence_vlm.DEFAULT_FRAMES_PER_CALL). "
            "The model would answer from a prompt shape it never saw, and "
            "nothing downstream would report it. Retrain, or change "
            "DEFAULT_FRAMES_PER_CALL to %d."
            % (adapter_dir, trained, serving_frames, trained))
        if strict:
            raise ValueError(message)
        print("WARNING: " + message)
    else:
        print("frame parity OK: trained and served on %d frame(s)" % trained)
    return trained


def check_evidence_parity(adapter_dir, strict=True):
    """Raise if the adapter was trained with an evidence block in its prompt
    but `config/arbiter.json` will serve it without one (or vice versa).

    THE SAME TRAP AS `check_frame_parity`, ON THE OTHER AXIS.

    `train_vlm.attach_evidence` puts a rendered perception block into every
    training prompt when `--evidence-cache` is passed. At serving,
    `config/arbiter.json`'s `vlm_evidence_context` decides whether that block
    is rendered at all. Train with it and serve without and the model receives
    an EMPTY context where it always saw tool confidences, task posteriors and
    YOLO detections -- a prompt shape it never encountered. As ever: nothing
    raises, the model still answers, and the only symptom is a score.

    `scripts/train_vlm.py` already prints "REMEMBER: serving must set
    vlm_evidence_context to true for this adapter" at the end of an
    evidence-bearing run. That is a log line in a job nobody re-reads. This is
    the same statement made at the one step every adapter passes through.

    The two adapters that exist today make the point: `vlm_lora` was trained
    bare and `vlm_evidence` with evidence, and the shipped config's `false` is
    correct for the FORMER. Getting that backwards is a silent regression.
    """
    config_path = Path(adapter_dir) / "training_config.json"
    if not config_path.exists():
        return None
    try:
        trained_with = bool(json.loads(config_path.read_text(encoding="utf-8"))
                            .get("evidence_context", False))
    except (OSError, ValueError, TypeError):
        return None

    from surgvu import arbiter

    serving_with = bool(arbiter.load_config().get("vlm_evidence_context", False))
    if trained_with != serving_with:
        message = (
            "EVIDENCE CONTEXT MISMATCH: adapter %s was trained %s an evidence "
            "block, but config/arbiter.json's vlm_evidence_context is %s. The "
            "model would be served a prompt shape it never saw, and nothing "
            "downstream would report it. Set vlm_evidence_context to %s."
            % (adapter_dir,
               "WITH" if trained_with else "WITHOUT",
               str(serving_with).lower(), str(trained_with).lower()))
        if strict:
            raise ValueError(message)
        print("WARNING: " + message)
    else:
        print("evidence-context parity OK: trained and served %s evidence"
              % ("with" if trained_with else "without"))
    return trained_with


def check_tokenizer_matches_embeddings(processor, model):
    """Raise `RuntimeError` if `processor`'s tokenizer has MORE tokens than
    `model`'s input-embedding matrix has rows; returns the vocab length
    otherwise.

    THE CHECK IS DIRECTIONAL, AND THAT IS THE CORRECTION. It was first
    written as `!=`, which failed this exact model on its first real run:
    the tokenizer has 151,665 tokens (151,643 base + 22 added) while the
    embedding matrix has 152,064 rows. That is not a defect and has nothing
    to do with the adapter -- `config.json` in the RAW BASE snapshot already
    declares `vocab_size = 152064`, because Qwen pads the embedding matrix
    up to a tensor-core-friendly multiple of 128 (152064 = 128 * 1188),
    leaving 399 unused rows.

    Only one direction is dangerous. If the tokenizer can emit an id the
    embedding matrix has no row for (vocab > rows), that id indexes out of
    range or into unrelated memory and produces the silent garbage described
    below. Surplus rows (rows > vocab) are simply never addressed: every id
    in 0..vocab-1 still lands on its own correct row. Failing on padding
    would block every correctly-merged Qwen checkpoint there is.

    THIS EXISTS BECAUSE OF A MEASURED, NOT HYPOTHETICAL, RISK: the trained
    adapter directory (`/staging/n/nkalthoff/surgvu26/models/vlm_lora`)
    carries its own `added_tokens.json`, so training's tokenizer is not
    guaranteed to match the raw base model's -- and a vocab/embedding size
    mismatch does NOT raise on its own. It produces silently
    plausible-looking garbage (an out-of-range token id indexing into an
    unrelated embedding row), which is worse than a crash here: this
    project's serving path swallows VLM errors and falls back to the
    router's own answer, so a garbage-but-non-crashing VLM output would
    ship completely silently. Called twice: once right after the merge in
    `merge_adapter_into_base` (before spending GPU time quantising
    something already broken), and again in `verify_checkpoint` against the
    FINAL, RELOADED-FROM-DISK checkpoint -- proving the saved,
    self-contained artefact is internally consistent, not just the
    in-memory objects mid-merge.

    Uses `len(processor.tokenizer)` (the FULL vocabulary, including any
    added tokens) rather than `tokenizer.vocab_size` (which reports only the
    base vocabulary and would miss exactly the discrepancy this check
    exists to catch).

    Breaks if: this compares `tokenizer.vocab_size` instead of
    `len(tokenizer)`, or the `>` check is downgraded to a printed warning
    instead of a raise, or it is "tightened" back to `!=` (see above -- that
    rejects the padding every Qwen checkpoint ships with).
    """
    vocab_len = len(processor.tokenizer)
    embedding_rows = model.get_input_embeddings().weight.shape[0]
    if vocab_len > embedding_rows:
        raise RuntimeError(
            "tokenizer/embedding size mismatch: the processor's tokenizer "
            "has %d tokens but the model's input-embedding matrix has only "
            "%d rows, so the tokenizer can emit an id the model has no row "
            "for. This is the exact silent-garbage failure mode this check "
            "exists to catch -- an out-of-range token id would index into "
            "unrelated memory without ever raising. Refusing to proceed. "
            "Check that the processor/tokenizer files came from the TRAINED "
            "adapter directory's own tokenizer (which carries added tokens "
            "the raw base checkpoint does not have), not from the raw base "
            "model's HF cache."
            % (vocab_len, embedding_rows))
    if embedding_rows > vocab_len:
        # Expected: Qwen pads to a multiple of 128. Reported, not fatal --
        # the surplus rows are simply never addressed. Printed rather than
        # silent so that a LARGE or changing gap is still visible to a human.
        print("  tokenizer %d tokens, embedding matrix %d rows "
              "(%d padding rows -- expected for this base, not an error)"
              % (vocab_len, embedding_rows, embedding_rows - vocab_len))
    return vocab_len


def merge_adapter_into_base(base_snapshot_dir, adapter_dir, merged_dir):
    """Fold the trained LoRA adapter into the fp16 base's own weights, on
    CPU, and save the merged fp16 checkpoint to `merged_dir`.

    See this module's docstring, "STAGE 1", for why CPU and why fp16 (not
    the 4-bit base `scripts/train_vlm.py` trained against) is the correct
    base to merge onto.

    THE PROCESSOR COMES FROM `adapter_dir`, NOT `base_snapshot_dir` --
    load-bearing, not a stylistic choice. `adapter_dir`'s tokenizer/
    processor files are the ones training actually used (and they may
    differ from the raw base's own -- see `check_tokenizer_matches_
    embeddings`'s docstring); pulling the processor from the raw base cache
    instead would risk a silent vocab/embedding mismatch that produces
    plausible-looking garbage rather than an error. This is also what makes
    `merged_dir` (and, downstream, the final quantised `output_dir`)
    SELF-CONTAINED: `evidence_vlm._load_model` calls
    `AutoProcessor.from_pretrained(model_dir)` on the SAME directory as the
    model, so the correct tokenizer must already be saved alongside it, not
    left for a caller to source from somewhere else.

    `local_files_only=True` on every `from_pretrained` call -- defence in
    depth alongside `main`'s `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`
    exports: this must never reach the network.
    """
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(adapter_dir), local_files_only=True)
    ensure_vl_chat_template(processor, "adapter %s" % adapter_dir)
    base_model = AutoModelForImageTextToText.from_pretrained(
        str(base_snapshot_dir), torch_dtype=torch.float16,
        device_map={"": "cpu"}, local_files_only=True)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged = peft_model.merge_and_unload()

    check_tokenizer_matches_embeddings(processor, merged)

    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    processor.save_pretrained(str(merged_dir))

    del peft_model, merged, base_model, processor
    gc.collect()
    return Path(merged_dir)


#: int8 as an alternative to NF4. NOT a change of recipe for its own sake --
#: NF4 was chosen under a constraint that no longer binds.
#:
#: 4-bit was forced by the 10 GB IMAGE ceiling: fp16 is 16 GB, and the only
#: way to fit the VLM inside the container at all was to quantise hard. Grand
#: Challenge also accepts a SEPARATE model tarball extracted to /opt/ml/model/,
#: which takes the weights out of the image budget entirely, and the grader's
#: T4 has 16 GiB of VRAM against the ~7 GB the NF4 build actually uses. So the
#: reason for 4-bit is gone on both axes.
#:
#: int8 is ~8 GB, still comfortable on a 16 GiB card, and recovers fidelity
#: that 4-bit rounding threw away -- from the SAME fine-tune, with no
#: retraining and no new data. `llm_int8_threshold` 6.0 is bitsandbytes'
#: default outlier cutoff; lowering it keeps more values in fp16 at the cost
#: of speed, and there is no measurement here that would justify moving it.
#:
#: Breaks if: this is used on a card below sm_75 (bitsandbytes int8 has the
#: same compute-capability floor as its 4-bit path), or if the caller forgets
#: that a checkpoint saved this way carries `load_in_8bit` in its own
#: config.json -- `evidence_vlm._load_model`'s bare `from_pretrained` reads
#: the recipe back out of there, so the serving side needs no change at all.
INT8_CONFIG_KWARGS = {
    "load_in_8bit": True,
    "llm_int8_threshold": 6.0,
}


def quantisation_kwargs(precision):
    """The BitsAndBytesConfig kwargs for `precision` ('nf4' or 'int8').

    'nf4' delegates to `train_vlm.bnb_config_kwargs()` rather than restating
    it, so the 4-bit path cannot drift from the recipe the 0.9092 held-out
    score was measured under.
    """
    if precision == "int8":
        return dict(INT8_CONFIG_KWARGS)
    if precision == "nf4":
        return dict(bnb_config_kwargs())
    raise ValueError("unknown precision %r (expected 'nf4' or 'int8')"
                     % (precision,))


def quantise_merged_checkpoint(merged_dir, output_dir, precision="nf4"):
    """Reload the merged fp16 checkpoint fresh and quantise it to 4-bit
    NF4, saving a SELF-CONTAINED checkpoint to `output_dir` -- config.json
    (carrying its own `quantization_config`), tokenizer/processor files, and
    the packed NF4 safetensors -- so `evidence_vlm._load_model`'s bare
    `from_pretrained(model_dir, device_map=...)` (no `quantization_config`
    argument of its own) can load it: `transformers` reads the quantisation
    recipe back OUT OF `output_dir/config.json`'s own `quantization_config`
    block, which `save_pretrained` writes there automatically for an
    already-quantised model. `verify_checkpoint` is what actually proves
    that claim rather than trusting it.

    THE RECIPE IS THE SAME ONE TRAINING USED: `bnb_config_kwargs()` is
    imported from `scripts/train_vlm.py`, not retyped here, so this cannot
    silently drift from the recipe the 0.9092 held-out CASE bertscore_f1 was
    actually measured under (NF4, double-quant, float16 compute dtype --
    never bfloat16, per that function's own T4/sm_75 reasoning).

    Requires CUDA -- bitsandbytes' 4-bit packing has no CPU implementation
    in the version this project installs; raises a clear `RuntimeError`
    instead of a deep bitsandbytes stack trace if none is visible.
    """
    import gc

    import torch
    from transformers import (
        AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "quantise_merged_checkpoint requires a CUDA device (bitsandbytes "
            "4-bit packing is CUDA-only); torch.cuda.is_available() is False")

    kwargs = quantisation_kwargs(precision)
    dtype_name = kwargs.pop("bnb_4bit_compute_dtype", None)
    if dtype_name is not None:
        kwargs["bnb_4bit_compute_dtype"] = getattr(torch, dtype_name)
    bnb_config = BitsAndBytesConfig(**kwargs)
    print("  quantising at %s: %s" % (precision, sorted(kwargs)))

    model = AutoModelForImageTextToText.from_pretrained(
        str(merged_dir), quantization_config=bnb_config,
        device_map={"": "cuda"}, local_files_only=True)
    processor = AutoProcessor.from_pretrained(
        str(merged_dir), local_files_only=True)
    ensure_vl_chat_template(processor, "merged model %s" % merged_dir)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    processor.save_pretrained(str(output_dir))

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return Path(output_dir)


def verify_checkpoint(output_dir, manifest_path=DEFAULT_MANIFEST,
                       max_new_tokens=32):
    """Load `output_dir` back with the EXACT bare call
    `evidence_vlm._load_model` uses -- no `quantization_config`, no
    `torch_dtype`, nothing this script's own quantisation step is not
    already responsible for baking into `output_dir/config.json` -- and run
    one real generation over a real preprocessed surgical frame (Task 3's
    manifest, never a synthetic tensor). Prints the question, the gold
    answer, and the model's output; raises `RuntimeError` if the output is
    empty.

    THIS IS THE CHECK THAT ACTUALLY MATTERS. A checkpoint can save cleanly
    and report a plausible size while still generating garbage (or nothing)
    if the merge silently dropped the adapter's effect, or the quantised
    save/reload round-trip lost the weights -- see this module's own
    docstring for why that failure shape gets named explicitly rather than
    assumed away. The output is printed, not scored against
    `surgvu.scoring.Scorer` here: one example is not a CASE-level eval (that
    measurement already exists, in `scripts/train_vlm.py`'s `run_eval`) --
    this function's job is to prove the artefact GENERATES SOMETHING REAL,
    and leave judging its quality to the human reading this job's log.
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    record = pick_verification_example(manifest_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The EXACT calls `evidence_vlm._load_model` makes -- proving THAT
    # loader, unmodified, works against this directory, not a more
    # permissive call this script could get away with instead.
    processor = AutoProcessor.from_pretrained(str(output_dir))
    model = AutoModelForImageTextToText.from_pretrained(
        str(output_dir), device_map={"": device})
    model.eval()

    # Proves the SAVED, SELF-CONTAINED checkpoint is internally consistent
    # -- not just the in-memory objects mid-merge, which
    # `merge_adapter_into_base` already checked once before quantising.
    # STOPS here, loudly, rather than papering over a mismatch: see
    # `check_tokenizer_matches_embeddings`'s docstring for why this failure
    # mode is silent (garbage output, no exception) if left unchecked.
    check_tokenizer_matches_embeddings(processor, model)

    images = load_frames(record["frame_paths"])
    prompt = build_sampling_prompt(record["question"], {})
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
    prompt_length = inputs["input_ids"].shape[1]
    output_text = processor.batch_decode(
        generated[:, prompt_length:], skip_special_tokens=True)[0].strip()

    print("VERIFY: case=%s question=%r" % (record["case"], record["question"]))
    print("VERIFY: gold answer=%r" % (record["answer"],))
    print("VERIFY: model output=%r" % (output_text,))

    if not output_text:
        raise RuntimeError(
            "verification FAILED: %s produced an EMPTY generation from a "
            "real frame -- this is exactly the 'looks like it worked and "
            "did nothing' failure mode; refusing to declare success"
            % (output_dir,))
    return output_text


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--allow-frame-mismatch", action="store_true",
                        help="downgrade the training/serving frame-count check "
                             "to a warning. Only for a deliberate experiment: "
                             "the mismatch it guards is silent everywhere else "
                             "and cost this project two submissions.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL_ID,
                        help="HF model id of the fp16 base (must already be "
                             "cached locally under --hf-cache-dir).")
    parser.add_argument("--hf-cache-dir", default=DEFAULT_HF_CACHE_DIR)
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR,
                        help="The trained LoRA adapter (top level of "
                             "scripts/train_vlm.py's --output-dir, not a "
                             "checkpoint-NNNN subdirectory).")
    parser.add_argument("--merged-dir", default=DEFAULT_MERGED_DIR,
                        help="Intermediate fp16 merged checkpoint; deleted "
                             "after quantisation unless --keep-merged.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Final self-contained 4-bit NF4 checkpoint. "
                             "Defaults to the exact path "
                             "containers/build_submission.sh's VLM_MODEL_SRC "
                             "already looks for.")
    parser.add_argument("--sidecar", action="store_true",
                        help="this checkpoint ships in Grand Challenge's "
                             "SEPARATE model tarball (/opt/ml/model/), not in "
                             "the image, so the 10 GiB image ceiling does not "
                             "apply. Checked against the T4's 16 GiB of VRAM "
                             "instead, which is what actually binds at "
                             "runtime.")
    parser.add_argument("--no-adapter", action="store_true",
                        help="quantise the BASE model with no fine-tune "
                             "merged in. For the v5.1 judge, which is a "
                             "different model from the answering VLM and has "
                             "no adapter of its own.")
    parser.add_argument("--precision", default="nf4", choices=("nf4", "int8"),
                        help="nf4 (4-bit, ~5.5 GiB -- what shipped in v5) or "
                             "int8 (~8 GiB, better fidelity from the same "
                             "fine-tune). int8 exceeds the 10 GB image "
                             "ceiling, so it must ship via Grand Challenge's "
                             "separate model tarball.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help="qa_frames_manifest.jsonl, for picking a real "
                             "frame to verify against.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--force-remerge", action="store_true",
                        help="Redo stage 1 even if --merged-dir already "
                             "looks complete.")
    parser.add_argument("--force-requantise", action="store_true",
                        help="Redo stage 2 even if --output-dir already "
                             "looks complete.")
    parser.add_argument("--keep-merged", action="store_true",
                        help="Do not delete --merged-dir after "
                             "quantisation succeeds. Default deletes it: it "
                             "exists only to split this pipeline's memory "
                             "footprint into two stages and is otherwise "
                             "~16GB of dead weight on /staging.")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip the load-back-and-generate check. For "
                             "debugging this script only -- never use this "
                             "to ship an unverified checkpoint.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve and validate every path (base "
                             "snapshot, adapter dir) and print the plan, "
                             "then exit before any torch/model code runs. "
                             "Works on this login node; nothing else in "
                             "this script does.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    snapshot_dir = find_local_snapshot_dir(args.base_model, args.hf_cache_dir)

    # --no-adapter QUANTISES A BASE MODEL, no fine-tune involved. Added for
    # the v5.1 judge: the decision VLM is deliberately a DIFFERENT model from
    # the answering one (a different generation, not just different weights),
    # so there is no adapter to merge -- only a base checkpoint to compress.
    #
    # It sets merged_dir to the base snapshot rather than special-casing the
    # pipeline, so stage 1's existing resume check (`_looks_like_a_saved_
    # model_dir`) sees a complete checkpoint and skips the merge on its own.
    # One code path, two uses.
    if args.no_adapter:
        adapter_dir = None
        args.merged_dir = str(snapshot_dir)
        print("--no-adapter: quantising the BASE model, no merge")
    else:
        adapter_dir = verify_adapter_dir(args.adapter_dir)

    print("base model snapshot:            %s" % snapshot_dir)
    print("adapter dir:                    %s" % (adapter_dir or "<none>"))
    print("merged (intermediate fp16) dir: %s" % args.merged_dir)
    print("output (final quantised) dir:   %s" % args.output_dir)

    # BEFORE the dry-run exit, deliberately: this is pure filesystem work and
    # a mismatch should be catchable without spending a GPU job to find it.
    if adapter_dir is not None:
        check_frame_parity(adapter_dir, SERVING_FRAMES_PER_CALL,
                           strict=not args.allow_frame_mismatch)
        check_evidence_parity(adapter_dir,
                              strict=not args.allow_frame_mismatch)

    if args.dry_run:
        print()
        print("--dry-run: stopping before any torch/model code runs")
        return 0

    # Defence in depth alongside every from_pretrained(..., local_files_only
    # =True) call above and below: this process must never reach the
    # network, at build time or (since this produces the exact directory
    # evidence_vlm._load_model reads) at serving time either.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if args.force_remerge or not _looks_like_a_saved_model_dir(args.merged_dir):
        print("\n== stage 1: merging adapter into base (CPU, fp16) ==")
        merge_adapter_into_base(snapshot_dir, adapter_dir, args.merged_dir)
    else:
        print("\nmerged fp16 checkpoint already present at %s; skipping "
              "stage 1 (pass --force-remerge to redo)" % (args.merged_dir,))

    if args.force_requantise or not _looks_like_a_saved_model_dir(args.output_dir):
        # The precision is INTERPOLATED, not spelled out. This banner read
        # "to 4-bit NF4" unconditionally until 2026-08-29, while the line
        # below correctly passed `precision=args.precision` -- so the v6 int8
        # sidecar build (job 9714496) announced itself as NF4 while actually
        # producing int8. The `quantising at <precision>` line inside
        # quantise_merged_checkpoint was the only honest one, and a reader who
        # trusted the banner would have mis-labelled the artifact.
        print("\n== stage 2: quantising merged checkpoint to %s =="
              % ("4-bit NF4" if args.precision == "nf4" else "8-bit int8"))
        quantise_merged_checkpoint(args.merged_dir, args.output_dir,
                                    precision=args.precision)
    else:
        print("\nquantised checkpoint already present at %s; skipping "
              "stage 2 (pass --force-requantise to redo)" % (args.output_dir,))

    if not args.keep_merged and _looks_like_a_saved_model_dir(args.output_dir):
        import shutil
        print("\nremoving intermediate merged fp16 checkpoint at %s"
              % (args.merged_dir,))
        shutil.rmtree(args.merged_dir, ignore_errors=True)

    print("\n== stage 3: size check ==")
    size_bytes = directory_size_bytes(args.output_dir)
    fit = check_fits_under_ceiling(size_bytes, sidecar=args.sidecar)
    print("output directory size: %.2f GiB (%d bytes)"
          % (fit["model_gib"], size_bytes))
    print("current image size (documented, unaffected by this script): "
          "%.2f GiB" % fit["current_image_gib"])
    print("projected total: %.2f GiB against a %.2f GiB ceiling"
          % (fit["projected_total_gib"], fit["ceiling_gib"]))
    print("margin: %.2f GiB" % fit["margin_gib"])

    if not fit["fits"]:
        print("\nSTOP: this checkpoint does NOT fit under the documented "
              "%.2f GiB image ceiling. Refusing to proceed to verification "
              "or to recommend staging this artefact -- fix the "
              "quantisation (or the image) before retrying, do not upload "
              "this." % fit["ceiling_gib"])
        return 2

    print("fits under the ceiling with %.2f GiB to spare." % fit["margin_gib"])

    if args.skip_verify:
        print("\n--skip-verify: NOT loading the checkpoint back. Do not "
              "ship this without verifying it separately.")
        return 0

    print("\n== stage 4: verify (load back with the frozen serving "
          "loader's exact call; generate over a real frame) ==")
    try:
        verify_checkpoint(args.output_dir, manifest_path=args.manifest,
                          max_new_tokens=args.max_new_tokens)
    except Exception:
        print("VERIFICATION FAILED -- see traceback below. Do not stage or "
              "upload this checkpoint.")
        raise

    # NAME THE ACTUAL PRECISION. This said "4-bit NF4" unconditionally, and
    # printed it over an int8 build (cluster 9707732) -- a success line that
    # misnames the artefact it just verified. Cosmetic until someone reads two
    # logs side by side to decide which checkpoint to ship, which is exactly
    # when it stops being cosmetic.
    precision_label = {"nf4": "4-bit NF4", "int8": "8-bit int8"}.get(
        args.precision, args.precision)
    ceiling_label = ("T4 VRAM budget" if args.sidecar
                     else "image ceiling")
    kind = precision_label if args.no_adapter else "%s, fine-tuned" % precision_label
    print("\nSUCCESS: %s is a self-contained, %s checkpoint that fits "
          "under the %s and generates a real answer when loaded back with "
          "the frozen serving loader."
          % (args.output_dir, kind, ceiling_label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
