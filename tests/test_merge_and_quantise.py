"""Tests for the torch-free parts of scripts/merge_and_quantise_vlm.py.

Deliberately torch-free, mirroring tests/test_train_vlm.py and
tests/test_evidence_vlm.py: this login node has no torch/transformers/peft/
bitsandbytes installed at all, so any module-scope import of one of those
would fail EVERY test below at collection, not just the ones that exercise
it. That is itself checked directly below (import succeeds at all).

What IS exercised here, with real files on this login node (PIL is
installed, unlike torch): path resolution against a fake local HF cache
layout, adapter-directory validation, directory-size accounting, the
fits-under-ceiling arithmetic, the resume-skip check, and picking a real
verification example out of a tiny manifest with real JPEGs.

What is NOT, and cannot be, exercised here: the actual merge
(`merge_adapter_into_base`), the actual quantisation
(`quantise_merged_checkpoint`), and the actual verification generation
(`verify_checkpoint`). Those run only inside condor/merge_quantise.sub on a
GPU compute node -- see this repo's report at
docs/design/2026-08-24-v5-evidence-pipeline/merge-quantise-report.md
for what was, and was not, run.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import merge_and_quantise_vlm as mq  # noqa: E402


# ============================================================================
# the module imports at all, with no torch/transformers/peft/bitsandbytes
# ============================================================================


def test_module_imports_without_torch():
    """The strongest available proof the module-scope import discipline
    holds: every test in this file would fail at collection, not just the
    ones that exercise torch, if this module imported it at module scope."""
    assert hasattr(mq, "main")
    assert hasattr(mq, "merge_adapter_into_base")
    assert hasattr(mq, "quantise_merged_checkpoint")
    assert hasattr(mq, "verify_checkpoint")


def test_no_torch_transformers_peft_bitsandbytes_import_at_module_scope():
    """Breaks if: `import torch` (or transformers/peft/bitsandbytes) is
    moved out of a function body and up to module scope anywhere in this
    file -- checked directly against the source, the same way
    tests/test_evidence_vlm.py checks for a forbidden import."""
    import inspect
    source = inspect.getsource(mq)
    lines = source.splitlines()
    # Only the module's OWN top-level import block (before the first `def`)
    # may not reference these packages; a lazy import inside a function body
    # is fine and expected.
    first_def = next(i for i, line in enumerate(lines) if line.startswith("def "))
    header = "\n".join(lines[:first_def])
    for forbidden in ("import torch", "import transformers", "import peft",
                     "import bitsandbytes"):
        assert forbidden not in header, (
            "%r found before the first function definition" % (forbidden,))


# ============================================================================
# hf_cache_folder_name / find_local_snapshot_dir -- NEVER a network fallback
# ============================================================================


def test_hf_cache_folder_name_matches_hub_convention():
    assert (mq.hf_cache_folder_name("Qwen/Qwen2.5-VL-7B-Instruct")
            == "models--Qwen--Qwen2.5-VL-7B-Instruct")


def _write_fake_snapshot(hf_cache_dir, model_id, revision="abc123",
                         mtime=None):
    folder = mq.hf_cache_folder_name(model_id)
    snap = Path(hf_cache_dir) / "hub" / folder / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")
    return snap


def test_find_local_snapshot_dir_finds_a_real_snapshot(tmp_path):
    snap = _write_fake_snapshot(tmp_path, "Qwen/Qwen2.5-VL-7B-Instruct")
    found = mq.find_local_snapshot_dir("Qwen/Qwen2.5-VL-7B-Instruct", tmp_path)
    assert found == snap


def test_find_local_snapshot_dir_raises_instead_of_falling_back_to_network(tmp_path):
    """Breaks if: this catches the missing-directory case and returns the
    bare model id string instead of raising -- that string would then be
    handed to `from_pretrained`, which WOULD attempt a network fetch."""
    with pytest.raises(FileNotFoundError):
        mq.find_local_snapshot_dir("Qwen/Qwen2.5-VL-7B-Instruct", tmp_path)


def test_find_local_snapshot_dir_raises_on_snapshot_without_config(tmp_path):
    """A snapshot directory that exists but never got a config.json (a
    partial/corrupted download) must not be treated as usable.

    Breaks if: the `(p / "config.json").exists()` filter is dropped."""
    folder = mq.hf_cache_folder_name("Qwen/Qwen2.5-VL-7B-Instruct")
    snap = tmp_path / "hub" / folder / "snapshots" / "incomplete"
    snap.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        mq.find_local_snapshot_dir("Qwen/Qwen2.5-VL-7B-Instruct", tmp_path)


def test_find_local_snapshot_dir_picks_the_newest_of_multiple(tmp_path):
    """Breaks if: the newest-mtime tie-break is removed and this instead
    picks an arbitrary (e.g. alphabetically-first) snapshot."""
    import os
    import time

    old = _write_fake_snapshot(tmp_path, "Qwen/Qwen2.5-VL-7B-Instruct",
                               revision="old")
    time.sleep(0.05)
    new = _write_fake_snapshot(tmp_path, "Qwen/Qwen2.5-VL-7B-Instruct",
                               revision="new")
    # Force distinguishable mtimes regardless of filesystem timestamp
    # resolution.
    now = time.time()
    os.utime(str(old / "config.json"), (now - 100, now - 100))
    os.utime(str(new / "config.json"), (now, now))
    found = mq.find_local_snapshot_dir("Qwen/Qwen2.5-VL-7B-Instruct", tmp_path)
    assert found == new


# ============================================================================
# verify_adapter_dir
# ============================================================================


def test_verify_adapter_dir_accepts_safetensors_adapter(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"fake")
    assert mq.verify_adapter_dir(tmp_path) == tmp_path


def test_verify_adapter_dir_accepts_bin_adapter(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.bin").write_bytes(b"fake")
    assert mq.verify_adapter_dir(tmp_path) == tmp_path


def test_verify_adapter_dir_raises_without_config(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"fake")
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        mq.verify_adapter_dir(tmp_path)


def test_verify_adapter_dir_raises_without_weights(tmp_path):
    """Breaks if: the weight-file existence check is dropped, which would
    accept a directory a training job is still writing (config flushed,
    weights not yet) as if the adapter were already complete."""
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="adapter_model"):
        mq.verify_adapter_dir(tmp_path)


# ============================================================================
# directory_size_bytes
# ============================================================================


def test_directory_size_bytes_sums_real_file_sizes(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 1000)
    (tmp_path / "b.bin").write_bytes(b"y" * 2000)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.bin").write_bytes(b"z" * 500)
    assert mq.directory_size_bytes(tmp_path) == 3500


def test_directory_size_bytes_skips_broken_symlinks(tmp_path):
    """Breaks if: a broken symlink's failed `os.path.getsize` is allowed to
    propagate instead of being skipped, which would crash a size report on
    a leftover dangling link rather than reporting real bytes."""
    (tmp_path / "real.bin").write_bytes(b"x" * 100)
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "does_not_exist.bin")
    assert mq.directory_size_bytes(tmp_path) == 100


def test_directory_size_bytes_empty_dir_is_zero(tmp_path):
    assert mq.directory_size_bytes(tmp_path) == 0


# ============================================================================
# check_fits_under_ceiling
# ============================================================================


def test_fits_under_ceiling_when_comfortably_under_budget():
    result = mq.check_fits_under_ceiling(4 * mq.BYTES_PER_GIB)
    assert result["fits"] is True
    assert result["model_gib"] == pytest.approx(4.0)
    assert result["margin_gib"] > 0


def test_does_not_fit_when_over_budget():
    """The scenario the task explicitly warns about: an unquantised fp16
    base landing near 16GB against a ~6.54GB budget."""
    result = mq.check_fits_under_ceiling(16 * mq.BYTES_PER_GIB)
    assert result["fits"] is False
    assert result["margin_gib"] < 0


def test_fits_boundary_is_inclusive():
    """Breaks if: `fits` is computed with strict `<` instead of `<=`,
    which would flip a checkpoint landing EXACTLY on the ceiling from a
    pass to a fail."""
    exact_budget_bytes = int(
        (mq.IMAGE_SIZE_CEILING_GIB - mq.CURRENT_IMAGE_SIZE_GIB)
        * mq.BYTES_PER_GIB)
    result = mq.check_fits_under_ceiling(exact_budget_bytes)
    assert result["fits"] is True


def test_fits_under_ceiling_accounts_for_the_current_image_size():
    """Breaks if: `fits` is computed as `model_gib <= ceiling_gib` instead
    of `projected_total_gib <= ceiling_gib` -- forgetting to add the
    current 3.46GB image would pass a checkpoint that actually blows the
    ceiling once the image it ships inside is accounted for."""
    # 8 GiB alone is under the bare 10 GiB ceiling, but 3.46 + 8 = 11.46 is
    # over it.
    result = mq.check_fits_under_ceiling(8 * mq.BYTES_PER_GIB)
    assert result["fits"] is False


# ============================================================================
# _looks_like_a_saved_model_dir -- the resume/skip check
# ============================================================================


def test_looks_like_a_saved_model_dir_true_for_complete_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"fake")
    assert mq._looks_like_a_saved_model_dir(tmp_path) is True


def test_looks_like_a_saved_model_dir_false_for_missing_directory(tmp_path):
    assert mq._looks_like_a_saved_model_dir(tmp_path / "nope") is False


def test_looks_like_a_saved_model_dir_false_when_config_only(tmp_path):
    """Breaks if: only `config.json`'s existence is checked, which would
    treat a directory a killed `save_pretrained` call left half-written
    (config flushed, no shards yet) as already complete."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert mq._looks_like_a_saved_model_dir(tmp_path) is False


def test_looks_like_a_saved_model_dir_false_when_weights_only(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"fake")
    assert mq._looks_like_a_saved_model_dir(tmp_path) is False


# ============================================================================
# pick_verification_example -- a REAL frame, short-circuiting search
# ============================================================================


def _write_tiny_jpeg(path):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(str(path), "JPEG")


def _write_manifest(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_pick_verification_example_returns_first_record_with_real_frames(tmp_path):
    present = tmp_path / "present.jpg"
    _write_tiny_jpeg(present)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [
        {"case": "case_000", "question": "q0", "answer": "a0",
         "frame_paths": [str(tmp_path / "missing.jpg")]},
        {"case": "case_001", "question": "q1", "answer": "a1",
         "frame_paths": [str(present)]},
    ])
    record = mq.pick_verification_example(manifest)
    assert record["case"] == "case_001"


def test_pick_verification_example_raises_when_no_record_has_real_frames(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [
        {"case": "case_000", "question": "q0", "answer": "a0",
         "frame_paths": [str(tmp_path / "missing.jpg")]},
    ])
    with pytest.raises(RuntimeError, match="no record"):
        mq.pick_verification_example(manifest)


def test_pick_verification_example_short_circuits_before_scanning_every_record(tmp_path, monkeypatch):
    """The manifest this project ships has 23,355 records; scanning all of
    them just to return the first hit would be minutes of wasted
    `Path.exists()` calls against /staging. This proves the search stops at
    the first match rather than filtering the whole list first.

    Breaks if: `pick_verification_example` reverts to calling
    `filter_records_with_frames(records)` once over the ENTIRE manifest
    instead of per-record with an early return."""
    present = tmp_path / "present.jpg"
    _write_tiny_jpeg(present)

    calls = []
    real_filter = mq.filter_records_with_frames

    def counting_filter(records):
        calls.append(len(records))
        return real_filter(records)

    monkeypatch.setattr(mq, "filter_records_with_frames", counting_filter)

    manifest = tmp_path / "manifest.jsonl"
    records = [{"case": "case_%03d" % i, "question": "q", "answer": "a",
               "frame_paths": [str(tmp_path / "missing.jpg")]}
              for i in range(50)]
    records.append({"case": "case_hit", "question": "q", "answer": "a",
                    "frame_paths": [str(present)]})
    records.extend({"case": "case_after_%03d" % i, "question": "q",
                    "answer": "a", "frame_paths": [str(tmp_path / "missing.jpg")]}
                   for i in range(50))
    _write_manifest(manifest, records)

    record = mq.pick_verification_example(manifest)
    assert record["case"] == "case_hit"
    # Every call filters exactly ONE record, never the whole list.
    assert all(n == 1 for n in calls)
    # And the search stopped once it found the hit (51 records in: 0..49
    # missing plus the hit at index 50), not after scanning all 101.
    assert len(calls) == 51


# ============================================================================
# check_tokenizer_matches_embeddings -- the silent-garbage guard the
# coordinator's correction required. Torch-free: plain stand-in objects
# that only need to support `len()` and `.get_input_embeddings().weight.
# shape[0]`, exactly what the real function reads.
# ============================================================================


class _FakeWeight:
    def __init__(self, rows):
        self.shape = (rows, 4096)


class _FakeEmbedding:
    def __init__(self, rows):
        self.weight = _FakeWeight(rows)


class _FakeModel:
    def __init__(self, rows):
        self._rows = rows

    def get_input_embeddings(self):
        return _FakeEmbedding(self._rows)


class _FakeTokenizer:
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n


class _FakeProcessor:
    def __init__(self, n):
        self.tokenizer = _FakeTokenizer(n)


def test_check_tokenizer_matches_embeddings_passes_when_equal():
    result = mq.check_tokenizer_matches_embeddings(
        _FakeProcessor(152064), _FakeModel(152064))
    assert result == 152064


def test_check_tokenizer_matches_embeddings_allows_qwens_padding_rows(capsys):
    """THE REAL NUMBERS FROM THE RUN THAT CAUGHT THIS. Job 9698669's first
    attempt died here after a 60-second five-shard load: the tokenizer has
    151,665 tokens (151,643 base + 22 added) and the embedding matrix has
    152,064 rows, and the check was written as `!=`.

    That is not a defect and is nothing to do with the adapter -- the RAW
    BASE snapshot's own config.json declares vocab_size = 152064, because
    Qwen pads the embedding matrix to a multiple of 128 (152064 = 128 *
    1188). The 399 surplus rows are never addressed; every id in
    0..151664 lands on its own correct row.

    Must not raise, and must SAY the gap out loud so a large or changed one
    is still visible to a human reading the log.
    """
    result = mq.check_tokenizer_matches_embeddings(
        _FakeProcessor(151665), _FakeModel(152064))
    assert result == 151665
    out = capsys.readouterr().out
    assert "151665" in out and "152064" in out and "399" in out


def test_check_tokenizer_matches_embeddings_raises_when_mismatched():
    """The scenario the coordinator's correction warned about: training's
    tokenizer (carrying its own added_tokens.json) disagreeing with the
    model's embedding matrix. Must raise, not warn -- a silent mismatch
    produces plausible-looking garbage, not a crash, which is worse."""
    with pytest.raises(RuntimeError, match="mismatch"):
        mq.check_tokenizer_matches_embeddings(
            _FakeProcessor(152064), _FakeModel(151936))


def test_check_tokenizer_matches_embeddings_uses_len_not_vocab_size():
    """Breaks if: this reads `tokenizer.vocab_size` instead of
    `len(tokenizer)` -- vocab_size reports only the base vocabulary and
    would miss a mismatch caused by ADDED tokens, exactly the discrepancy
    this check exists to catch."""
    class _TokenizerWithMisleadingVocabSize:
        vocab_size = 100  # deliberately wrong/unused

        def __len__(self):
            return 152064

    class _ProcessorWithThatTokenizer:
        tokenizer = _TokenizerWithMisleadingVocabSize()

    # Should PASS: len() (152064) matches the embedding rows, even though
    # .vocab_size (100) would not have.
    result = mq.check_tokenizer_matches_embeddings(
        _ProcessorWithThatTokenizer(), _FakeModel(152064))
    assert result == 152064


def test_merge_loads_processor_from_adapter_dir_not_base_snapshot():
    """Breaks if: `AutoProcessor.from_pretrained` inside
    `merge_adapter_into_base` is sourced from `base_snapshot_dir` instead of
    `adapter_dir` -- which would silently pull the RAW base model's
    tokenizer instead of the one training actually used and saved,
    reintroducing the exact vocab/embedding mismatch risk
    `check_tokenizer_matches_embeddings` exists to catch (the trained
    adapter directory carries its own added_tokens.json; the raw base
    directory does not)."""
    import inspect
    lines = inspect.getsource(mq.merge_adapter_into_base).splitlines()
    # The real call, not the docstring's mention of evidence_vlm._load_model's
    # own `AutoProcessor.from_pretrained(model_dir)` call -- distinguished by
    # actually assigning to `processor =`.
    idx = next(i for i, line in enumerate(lines)
              if "processor = AutoProcessor.from_pretrained(" in line)
    window = "\n".join(lines[idx:idx + 2])
    assert "adapter_dir" in window
    assert "base_snapshot_dir" not in window


def test_merge_calls_the_tokenizer_embedding_check_before_saving():
    """Breaks if: `check_tokenizer_matches_embeddings` is called AFTER
    `merged.save_pretrained(...)` instead of before -- which would still
    write a broken checkpoint to disk before catching the problem, wasting
    the disk write and risking a caller reading the checkpoint before the
    check ever runs."""
    import inspect
    source = inspect.getsource(mq.merge_adapter_into_base)
    check_pos = source.index("check_tokenizer_matches_embeddings(")
    save_pos = source.index("merged.save_pretrained(")
    assert check_pos < save_pos


# ============================================================================
# CLI defaults
# ============================================================================


def test_default_output_dir_matches_build_submission_shs_vlm_model_src():
    """Breaks if: DEFAULT_OUTPUT_DIR is changed to a path other than the
    exact one containers/build_submission.sh's VLM_MODEL_SRC default
    already looks for -- the whole point of this constant is that the
    existing staging mechanism picks this artefact up with ZERO further
    wiring once it exists."""
    assert (mq.DEFAULT_OUTPUT_DIR
            == "/staging/n/nkalthoff/surgvu26/models/qwen25vl-7b-nf4")


def test_default_adapter_dir_is_imported_from_train_vlm_not_retyped():
    """Breaks if: DEFAULT_ADAPTER_DIR is hardcoded here instead of imported
    from train_vlm.DEFAULT_OUTPUT_DIR, which would let the two silently
    drift if the training script's own default ever changes."""
    import train_vlm
    assert mq.DEFAULT_ADAPTER_DIR == train_vlm.DEFAULT_OUTPUT_DIR


def test_default_base_model_id_is_imported_from_evidence_vlm_not_retyped():
    from surgvu.evidence_vlm import DEFAULT_FINETUNE_BASE, DEFAULT_MODEL_DIR
    # CHANGED 2026-08-27 for v6: the merge base is now
    # evidence_vlm.DEFAULT_FINETUNE_BASE (the NVIDIA surgical checkpoint), not
    # DEFAULT_MODEL_DIR (which stays the SERVING fallback). Still imported
    # rather than retyped -- that is what this test is really protecting, and
    # tests/test_train_vlm.py pins that train and merge agree on it.
    assert mq.DEFAULT_BASE_MODEL_ID == DEFAULT_FINETUNE_BASE
    assert mq.DEFAULT_BASE_MODEL_ID != DEFAULT_MODEL_DIR


def test_quantisation_recipe_is_imported_from_train_vlm_not_retyped():
    """Breaks if: this script re-types its own BitsAndBytesConfig kwargs
    instead of importing train_vlm.bnb_config_kwargs, which could silently
    drift from the recipe the 0.9092 held-out eval was actually measured
    under."""
    import train_vlm
    assert mq.bnb_config_kwargs is train_vlm.bnb_config_kwargs


def test_dry_run_reaches_the_stop_point_without_torch(tmp_path):
    """The one true end-to-end smoke test this login node CAN run: a real
    invocation of main() against the real base-model cache and adapter
    directory on /staging, stopping before any torch import. If this ever
    imports torch, it fails here (torch is not installed) rather than in
    the condor job."""
    exit_code = mq.main(["--dry-run"])
    assert exit_code == 0


def test_dry_run_raises_clearly_when_base_model_not_cached(tmp_path):
    """A HUB-ID base that is not in the cache must raise, never fetch.

    `--base-model` is passed EXPLICITLY as a hub id. It used to be omitted,
    relying on the default -- but v6's default is an absolute snapshot path
    (evidence_vlm.DEFAULT_FINETUNE_BASE), which resolves on its own and
    ignores --hf-cache-dir entirely, so the old form stopped exercising the
    cache-miss path it was written for and passed for the wrong reason.
    """
    exit_code_raises = False
    try:
        mq.main(["--dry-run", "--hf-cache-dir", str(tmp_path),
                 "--base-model", "Qwen/Qwen2.5-VL-7B-Instruct"])
    except FileNotFoundError:
        exit_code_raises = True
    assert exit_code_raises


def test_dry_run_raises_when_an_absolute_base_has_no_config(tmp_path):
    """The absolute-path branch still has to be a COMPLETE snapshot. A bare
    directory must raise rather than sail through to a torch load that fails
    much later and much less legibly."""
    bare = tmp_path / "not-a-snapshot"
    bare.mkdir()
    with pytest.raises(FileNotFoundError, match="no config.json"):
        mq.find_local_snapshot_dir(str(bare), str(tmp_path))


def test_absolute_base_resolves_to_itself(tmp_path):
    """The v6 path: an absolute snapshot dir with a config.json is returned
    unchanged, NOT name-mangled through hf_cache_folder_name."""
    snap = tmp_path / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")
    assert mq.find_local_snapshot_dir(str(snap), "/nonexistent/cache") == snap


def test_dry_run_raises_clearly_when_adapter_dir_incomplete(tmp_path):
    with pytest.raises(FileNotFoundError):
        mq.main(["--dry-run", "--adapter-dir", str(tmp_path)])


def test_build_arg_parser_defaults_match_module_constants():
    args = mq.build_arg_parser().parse_args([])
    assert args.base_model == mq.DEFAULT_BASE_MODEL_ID
    assert args.hf_cache_dir == mq.DEFAULT_HF_CACHE_DIR
    assert args.adapter_dir == mq.DEFAULT_ADAPTER_DIR
    assert args.merged_dir == mq.DEFAULT_MERGED_DIR
    assert args.output_dir == mq.DEFAULT_OUTPUT_DIR
    assert args.keep_merged is False
    assert args.skip_verify is False
    assert args.force_remerge is False
    assert args.force_requantise is False


# ---------------------------------------------------------------------------
# int8 vs NF4. NF4 was forced by the 10 GB IMAGE ceiling; the model-tarball
# sidecar removes that constraint, and the grader's T4 has 16 GiB against the
# ~7 GB the NF4 build uses.
# ---------------------------------------------------------------------------

def test_nf4_delegates_to_the_training_recipe_rather_than_restating_it():
    """The 4-bit path must not drift from the recipe the 0.9092 held-out score
    was measured under, so it is imported, not retyped."""
    from train_vlm import bnb_config_kwargs
    assert mq.quantisation_kwargs("nf4") == bnb_config_kwargs()


def test_nf4_keeps_float16_compute_dtype():
    """The grader's T4 is sm_75 and has no bfloat16. A bf16 recipe would load
    fine on the H200 that builds it and fail on the card that grades it."""
    assert mq.quantisation_kwargs("nf4")["bnb_4bit_compute_dtype"] == "float16"


def test_int8_is_a_distinct_recipe_with_no_4bit_keys():
    """Passing bnb_4bit_* alongside load_in_8bit is a config that says two
    contradictory things; bitsandbytes would honour one and ignore the other."""
    kwargs = mq.quantisation_kwargs("int8")
    assert kwargs["load_in_8bit"] is True
    assert not any(k.startswith("bnb_4bit") for k in kwargs)
    assert "load_in_4bit" not in kwargs


def test_an_unknown_precision_raises_rather_than_defaulting():
    """A typo must not silently produce a 4-bit build labelled int8 -- the two
    differ by ~2.5 GB on disk, so the mistake would survive a size check."""
    with pytest.raises(ValueError, match="unknown precision"):
        mq.quantisation_kwargs("int4")


def test_quantisation_kwargs_returns_a_fresh_dict_each_call():
    """The caller pops keys out of it; a shared dict would be emptied for
    every subsequent caller in the same process."""
    first = mq.quantisation_kwargs("int8")
    first.pop("load_in_8bit")
    assert "load_in_8bit" in mq.quantisation_kwargs("int8")


# ---------------------------------------------------------------------------
# The sidecar gate. Weights shipped in Grand Challenge's model tarball never
# enter the image, so the image ceiling does not bind them -- but VRAM does.
# ---------------------------------------------------------------------------

INT8_CHECKPOINT_BYTES = 9407962626   # the real 8.76 GiB int8 checkpoint


def test_the_image_gate_correctly_refuses_int8():
    """Not a bug -- this is the gate working. int8 in the IMAGE is 12.22 GiB
    against a 10 GiB ceiling, and it refused to recommend uploading it."""
    fit = mq.check_fits_under_ceiling(INT8_CHECKPOINT_BYTES)
    assert fit["fits"] is False
    assert fit["margin_gib"] < 0


def test_the_sidecar_gate_admits_the_same_checkpoint():
    """The assumption changed, not the arithmetic: int8 plus the judge is
    14.18 GiB baked in, so both moved to /opt/ml/model/ and the image dropped
    to ~3.5 GiB. Refusing a sidecar artefact for exceeding a ceiling it is not
    subject to would block the architecture that exists to escape it."""
    fit = mq.check_fits_under_ceiling(INT8_CHECKPOINT_BYTES, sidecar=True)
    assert fit["fits"] is True
    assert fit["gate"] == "sidecar/VRAM"


def test_the_sidecar_gate_still_binds_on_VRAM():
    """NOT simply skipped. The grader's T4 has 16 GiB and running it out of
    memory mid-case writes no answer, which scores 0 -- worse than any wrong
    answer. A model too big for the card must still be refused."""
    too_big = int(15.0 * mq.BYTES_PER_GIB)
    fit = mq.check_fits_under_ceiling(too_big, sidecar=True)
    assert fit["fits"] is False


def test_the_sidecar_budget_reserves_headroom_for_activations():
    """The budget is the T4's VRAM MINUS headroom, not all of it. The vision
    encoder's prefill activations are the large, less predictable term."""
    fit = mq.check_fits_under_ceiling(INT8_CHECKPOINT_BYTES, sidecar=True)
    assert fit["budget_gib"] == mq.T4_VRAM_GIB - mq.VRAM_HEADROOM_GIB
    assert mq.VRAM_HEADROOM_GIB > 0


# --------------------------------------------------------------------------
# frame parity: the training/serving frame count, enforced rather than
# documented
# --------------------------------------------------------------------------

def _adapter_with(tmp_path, **config):
    import json as _json

    d = tmp_path / "adapter"
    d.mkdir(exist_ok=True)
    (d / "training_config.json").write_text(_json.dumps(config), encoding="utf-8")
    return d


def test_frame_mismatch_raises(tmp_path):
    """THIS PROJECT'S MOST EXPENSIVE FAILURE SHAPE.

    Both pre-v6 adapters were fine-tuned on 4 frames and served 16. Nothing
    errored: transformers accepts any number of images, the model generates
    fluent text, the container passes validation, and the only symptom is a
    score. Documenting the requirement is what was tried, and what failed.
    """
    adapter = _adapter_with(tmp_path, max_frames=8)
    with pytest.raises(ValueError, match="FRAME MISMATCH"):
        mq.check_frame_parity(adapter, 16)


def test_frame_parity_passes_when_they_agree(tmp_path):
    adapter = _adapter_with(tmp_path, max_frames=8)
    assert mq.check_frame_parity(adapter, 8) == 8


def test_missing_training_config_warns_but_does_not_raise(tmp_path):
    """Adapters predating v6 have no such file. Refusing to merge them would
    break the fallback path for a check that is advisory for them."""
    bare = tmp_path / "old_adapter"
    bare.mkdir()
    assert mq.check_frame_parity(bare, 16) is None


def test_malformed_training_config_warns_but_does_not_raise(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "training_config.json").write_text("{not json", encoding="utf-8")
    assert mq.check_frame_parity(d, 16) is None


def test_max_frames_zero_is_reported_not_compared(tmp_path):
    """0 means 'every frame the manifest lists' -- meaningful only alongside
    that manifest, so comparing it to a serving count would be nonsense."""
    adapter = _adapter_with(tmp_path, max_frames=0)
    assert mq.check_frame_parity(adapter, 16) == 0


def test_strict_false_downgrades_to_a_warning(tmp_path):
    """--allow-frame-mismatch, for a deliberate experiment only."""
    adapter = _adapter_with(tmp_path, max_frames=4)
    assert mq.check_frame_parity(adapter, 16, strict=False) == 4


def test_serving_frame_constant_is_imported_from_evidence_vlm():
    """Imported, not restated -- the whole point is that one number governs
    both sides."""
    from surgvu.evidence_vlm import DEFAULT_FRAMES_PER_CALL

    assert mq.SERVING_FRAMES_PER_CALL == DEFAULT_FRAMES_PER_CALL


def _shipped_evidence_context():
    """config/arbiter.json's `vlm_evidence_context`, read through the same
    loader the guard itself uses.

    WHY THESE TESTS ASK RATHER THAN ASSERT A LITERAL
    --------------------------------------------------
    Until 2026-08-29 the two tests below hardcoded `false`, because that was
    what `vlm_lora` needed. v6 stage 2 trains WITH evidence, so shipping it
    required flipping the config -- and the tests failed, not because the
    invariant broke, but because they had encoded which adapter happened to
    ship rather than the invariant itself.

    The invariant is train/serve PARITY: an adapter must be served the prompt
    shape it was trained on. That statement is true in both directions and
    survives the next flip. Pinning the literal only buys a second failure the
    next time someone legitimately changes it.
    """
    from surgvu import arbiter

    return bool(arbiter.load_config().get("vlm_evidence_context", False))


def test_evidence_context_mismatch_raises(tmp_path):
    """THE SAME TRAP AS FRAME PARITY, ON THE OTHER AXIS.

    train_vlm.attach_evidence puts a rendered perception block into every
    training prompt when --evidence-cache is passed; at serving,
    config/arbiter.json's vlm_evidence_context decides whether that block is
    rendered at all. Train with it and serve without and the model receives an
    EMPTY context where it always saw tool confidences, task posteriors and
    YOLO detections. Nothing raises, the model still answers, and the only
    symptom is a score.

    An adapter trained the OPPOSITE way to whatever the config currently
    serves must trip this, whichever way round that happens to be.
    """
    adapter = _adapter_with(tmp_path,
                            evidence_context=not _shipped_evidence_context())
    with pytest.raises(ValueError, match="EVIDENCE CONTEXT MISMATCH"):
        mq.check_evidence_parity(adapter)


def test_evidence_parity_passes_for_the_adapter_the_config_matches(tmp_path):
    """The pairing the shipped config describes must not warn.

    Today that is v6 stage 2, trained WITH evidence against
    `vlm_evidence_context: true`. Before 2026-08-29 it was `vlm_lora`, trained
    bare against `false`. Both are correct pairings; the test is about the
    agreement, not about which one is in force.
    """
    shipped = _shipped_evidence_context()
    adapter = _adapter_with(tmp_path, evidence_context=shipped)

    # check_evidence_parity returns `trained_with` on success -- NOT a
    # pass/fail flag. The old assertion here read `is False`, which looked
    # like "did not warn" but was really just the bare adapter's own
    # evidence_context echoed back; it would have failed for a correctly
    # paired evidence-trained adapter. Assert agreement instead.
    assert mq.check_evidence_parity(adapter) is shipped


def test_evidence_parity_is_silent_without_a_training_config(tmp_path):
    """Adapters predating v6 carry no record of how they were trained;
    refusing to merge them would break the fallback path."""
    bare = tmp_path / "old"
    bare.mkdir()
    assert mq.check_evidence_parity(bare) is None


def test_evidence_parity_strict_false_downgrades_to_warning(tmp_path):
    adapter = _adapter_with(tmp_path, evidence_context=True)
    assert mq.check_evidence_parity(adapter, strict=False) is True


# --------------------------------------------------------------------------
# chat template: a processor saved without one makes EVERY served case score 0
# --------------------------------------------------------------------------

class _Proc:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template


def test_missing_template_is_replaced_with_the_vl_one():
    """Both merge stages save_pretrained this processor into the directory the
    container ships, and evidence_vlm calls apply_chat_template on it at
    inference. Saved without a template, every case raises, every response is
    missing, and a missing response scores 0."""
    proc = mq.ensure_vl_chat_template(_Proc(None), "test")
    assert "vision_start" in proc.chat_template


def test_a_text_only_template_is_refused():
    """The dangerous case: nvidia's checkpoint carries a text-only Qwen2.5
    template in tokenizer_config, so a processor can HAVE a template and still
    render prompts with no image tokens."""
    text_only = "{%- if tools %}{{- 'no vision' }}{%- endif %}"
    with pytest.raises(ValueError, match="no vision_start handling"):
        mq.ensure_vl_chat_template(_Proc(text_only), "test")


def test_an_existing_vision_template_is_left_alone():
    """A processor that already has a proper VL template must not be
    overwritten -- it may legitimately differ from the vendored copy."""
    import train_vlm

    good = train_vlm.load_vl_chat_template()
    proc = mq.ensure_vl_chat_template(_Proc(good), "test")
    assert proc.chat_template == good
