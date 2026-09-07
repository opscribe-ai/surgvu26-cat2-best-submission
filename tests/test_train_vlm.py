"""Tests for the torch-free parts of scripts/train_vlm.py (Task 4 of the v5
plan3 VLM training pipeline).

Deliberately torch-free, mirroring tests/test_evidence_vlm.py: this login
node has no torch/transformers/peft/bitsandbytes installed at all (confirmed
in scripts/train_vlm.py's own docstring), so any module-scope import of one
of those would fail EVERY test below at collection, not just the ones that
exercise it. `QADataset` and `load_frames` are the two exceptions worth
using for real rather than stubbing: PIL IS installed here, so tests below
write real tiny JPEGs and read them back.

What is NOT exercised here, and cannot be on this login node: model
loading/quantisation, `Collator.__call__`'s actual tensor construction, the
training loop, and the generation-based evaluation. See
scripts/train_vlm.py's own docstring, "WHAT COULD NOT BE RUN OR TESTED
HERE."
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import train_vlm as tv                                              # noqa: E402
from train_vlm import (  # noqa: E402
    ATTN_IMPLEMENTATION, IGNORE_INDEX, QADataset, assign_case_split,
    bnb_config_kwargs, build_arg_parser, build_messages,
    filter_records_with_frames, load_case_universe, load_frames,
    load_manifest, mask_prompt_tokens, render_training_prompt,
    sample_eval_records, verify_manifest_clean,
)
from surgvu.evidence_vlm import build_sampling_prompt  # noqa: E402


# ============================================================================
# manifest loading
# ============================================================================


def test_load_manifest_reads_jsonl_and_skips_blank_lines(tmp_path):
    """Breaks if: blank-line skipping is removed, which would raise on
    json.loads('') instead of silently skipping a trailing newline."""
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        '{"case": "case_000", "question": "q1", "answer": "a1"}\n'
        "\n"
        '{"case": "case_001", "question": "q2", "answer": "a2"}\n',
        encoding="utf-8")
    records = load_manifest(path)
    assert [r["case"] for r in records] == ["case_000", "case_001"]


def test_load_manifest_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "does_not_exist.jsonl")


# ============================================================================
# R30: independent heldout verification, split from R28's case split
# ============================================================================


def _write_splits(tmp_path, train, val, heldout):
    path = tmp_path / "splits_v2.json"
    path.write_text(json.dumps({"train": train, "val": val, "heldout": heldout}),
                    encoding="utf-8")
    return path


def test_load_case_universe_normalises_mixed_spellings(tmp_path):
    """The public sample dirs spell a case `case122`; splits_v2.json spells
    it `case_122`. Breaks if: load_case_universe stores the raw strings
    instead of normalize_case_id(c) for every id."""
    path = _write_splits(tmp_path, train=["case_001"], val=["case2"],
                         heldout=["case122", "case_123"])
    train_norm, val_norm, heldout_norm = load_case_universe(path)
    assert train_norm == {"case_001"}
    assert val_norm == {"case_002"}
    assert heldout_norm == {"case_122", "case_123"}


def test_load_case_universe_raises_on_empty_heldout(tmp_path):
    """Breaks if: the `if not ids: raise` guard is removed, which would let
    a config file that lost its heldout key exclude nothing while
    downstream code proceeds as if it were clean."""
    path = _write_splits(tmp_path, train=["case_001"], val=["case_002"], heldout=[])
    with pytest.raises(ValueError, match="heldout"):
        load_case_universe(path)


def test_load_case_universe_raises_on_missing_key(tmp_path):
    path = tmp_path / "splits_v2.json"
    path.write_text(json.dumps({"train": ["case_001"], "val": ["case_002"]}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="heldout"):
        load_case_universe(path)


def test_load_case_universe_raises_when_lists_overlap(tmp_path):
    """Breaks if: the pairwise-disjoint check after normalisation is
    removed -- a case spelled two ways across two lists (case5 in train,
    case_005 in val) would otherwise silently pass through as two distinct
    ids instead of the collision it actually is."""
    path = _write_splits(tmp_path, train=["case5"], val=["case_005"],
                         heldout=["case_999"])
    with pytest.raises(RuntimeError, match="not disjoint"):
        load_case_universe(path)


def test_verify_manifest_clean_raises_on_leaked_case():
    """Breaks if: the `norm in heldout_norm` check is replaced by a raw
    membership test against the record's UN-normalised case string (which
    would miss `case122` when `heldout_norm` holds `case_122`)."""
    train_norm = {"case_001"}
    val_norm = set()
    heldout_norm = {"case_122"}
    records = [{"case": "case_001"}, {"case": "case122"}]  # unnormalised spelling
    with pytest.raises(RuntimeError, match="HELDOUT LEAKAGE"):
        verify_manifest_clean(records, train_norm, val_norm, heldout_norm)


def test_verify_manifest_clean_raises_on_unknown_case():
    train_norm = {"case_001"}
    val_norm = set()
    heldout_norm = {"case_999"}
    records = [{"case": "case_777"}]  # in neither list at all
    with pytest.raises(RuntimeError, match="not in config/splits_v2.json"):
        verify_manifest_clean(records, train_norm, val_norm, heldout_norm)


def test_verify_manifest_clean_passes_and_reports_present_cases_when_clean():
    """Breaks if: `present` stops including val-case ids (e.g. only unions
    `train_norm` into `eligible`), which would misreport every val case's
    manifest rows as 'unknown'."""
    train_norm = {"case_001"}
    val_norm = {"case_002"}
    heldout_norm = {"case_999"}
    records = [{"case": "case_001"}, {"case": "case_001"}, {"case": "case2"}]
    present = verify_manifest_clean(records, train_norm, val_norm, heldout_norm)
    assert present == {"case_001", "case_002"}


# ============================================================================
# R28: split by CASE, never by example
# ============================================================================


def test_assign_case_split_keeps_every_record_of_one_case_on_one_side():
    """Breaks if: records are assigned by index/hash instead of by their
    case id, which could put two records sharing one 30s window (near-
    duplicates) on opposite sides of the split."""
    train_norm = {"case_001"}
    val_norm = {"case_002"}
    records = [
        {"case": "case_001", "question": "q1"},
        {"case": "case_001", "question": "q2"},
        {"case": "case_002", "question": "q3"},
    ]
    train_records, val_records = assign_case_split(records, train_norm, val_norm)
    assert {r["question"] for r in train_records} == {"q1", "q2"}
    assert {r["question"] for r in val_records} == {"q3"}


def test_assign_case_split_raises_on_case_in_both_sets():
    train_norm = {"case_001"}
    val_norm = {"case_001"}
    with pytest.raises(RuntimeError, match="BOTH"):
        assign_case_split([{"case": "case_001"}], train_norm, val_norm)


def test_assign_case_split_raises_on_case_in_neither_set():
    """Breaks if: an unassigned record is silently dropped instead of
    raising -- a smaller-than-expected training set must never look like a
    clean one."""
    with pytest.raises(RuntimeError, match="absent from BOTH"):
        assign_case_split([{"case": "case_999"}], {"case_001"}, {"case_002"})


def test_assign_case_split_normalises_spelling():
    train_norm = {"case_001"}
    records = [{"case": "case1"}]  # public-sample spelling, no underscore
    train_records, val_records = assign_case_split(records, train_norm, set())
    assert len(train_records) == 1
    assert val_records == []


# ============================================================================
# frame-presence filtering (partial-extraction safety)
# ============================================================================


def test_filter_records_with_frames_drops_records_missing_any_frame(tmp_path):
    """Breaks if: only frame_paths[0] is checked instead of every path."""
    present = tmp_path / "present.jpg"
    present.write_bytes(b"x")
    missing = str(tmp_path / "missing.jpg")
    records = [
        {"frame_paths": [str(present), str(present)]},
        {"frame_paths": [str(present), missing]},
    ]
    kept, dropped = filter_records_with_frames(records)
    assert len(kept) == 1
    assert dropped == 1


def test_filter_records_with_frames_keeps_all_when_all_present(tmp_path):
    present = tmp_path / "present.jpg"
    present.write_bytes(b"x")
    records = [{"frame_paths": [str(present)]} for _ in range(3)]
    kept, dropped = filter_records_with_frames(records)
    assert len(kept) == 3
    assert dropped == 0


# ============================================================================
# eval sampling
# ============================================================================


def test_sample_eval_records_is_deterministic_for_a_fixed_seed():
    records = [{"case": "case_%03d" % i} for i in range(50)]
    a = sample_eval_records(records, n=10, seed=7)
    b = sample_eval_records(records, n=10, seed=7)
    assert a == b
    assert len(a) == 10


def test_sample_eval_records_returns_everything_when_n_exceeds_pool():
    records = [{"case": "case_000"}, {"case": "case_001"}]
    result = sample_eval_records(records, n=100, seed=0)
    assert result == records


# ============================================================================
# prompt / message assembly -- must match evidence_vlm exactly
# ============================================================================


def test_render_training_prompt_matches_evidence_vlm_build_sampling_prompt():
    """The training prompt must match the serving prompt (plan's own
    wording). Breaks if: train_vlm.py stops calling build_sampling_prompt and
    reimplements its own question string, silently drifting from serving."""
    question = "What task is being performed in this clip?"
    assert render_training_prompt(question) == build_sampling_prompt(question, {})


def test_render_training_prompt_carries_no_evidence_blocks():
    """Empty context at train time (see module docstring) means the prompt
    is exactly the shared skeleton -- no tools/task/yolo/motion/variant
    lines. Breaks if: render_training_prompt starts passing a non-empty
    context, silently introducing ground-truth-derived evidence text."""
    rendered = render_training_prompt("Is a needle driver being used?")
    assert rendered == (
        "Question: Is a needle driver being used?\n"
        "Answer as briefly as possible -- prefer a single word or short "
        "phrase over a full sentence.")


_ARBITER_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "arbiter.json"


def _shipped_vlm_evidence_context():
    """The single source of truth for whether `scripts/inference.py` renders
    the real evidence packet into the VLM's prompt: `config/arbiter.json`'s
    own `vlm_evidence_context` key, read directly from disk here.

    Not read by importing `scripts/inference.py` -- that module `import
    torch`s at module scope (see its own docstring), which is not installed
    on this login node and would fail collection of this whole (otherwise
    torch-free) file. Reading the same JSON file `inference.load_arbiter_config`
    reads is not a second copy of the switch; it is the same one file, read
    a second way, exactly the same relationship `arbiter.load_config` has to
    the file it reads. Defaults to False (bare) if the key is ever absent,
    mirroring `scripts/inference.py`'s own `DEFAULT_VLM_EVIDENCE_CONTEXT`
    degrade-quietly fallback for a missing/malformed config.
    """
    data = json.loads(_ARBITER_CONFIG_PATH.read_text(encoding="utf-8"))
    return bool(data.get("vlm_evidence_context", False))


def test_serving_and_training_prompts_match_under_the_shipped_default():
    """THE TRAIN/SERVE PROMPT PARITY INVARIANT.

    `scripts/inference.py`'s `EvidenceVlmHandle.sample` renders the VLM's
    prompt with the real evidence packet (`perception`) when
    `config/arbiter.json`'s `vlm_evidence_context` is true, and with `{}`
    (bare) when it is false -- see that method's own docstring.
    `scripts/train_vlm.py` renders the SAME prompt through the same function,
    passing `record["evidence"]` when `--evidence-cache` was given and nothing
    when it was not, so the two must agree on which context is used, or the
    served prompt is text the model never once saw in training.

    UPDATED 2026-08-29. This test previously asserted that training is bare
    FULL STOP, and so failed the moment v6 shipped. That was the test being
    wrong, not the invariant: `render_training_prompt` has always taken an
    `evidence` argument (train_vlm.py:615 passes it), and v6 stage 2 trained
    with `--evidence-cache`, i.e. WITH the packet. What must hold is that the
    switch and the training run agree -- which is what is checked below, in
    both directions.

    A realistic, NON-EMPTY `perception` dict is used below -- deliberately
    populating every renderer in `evidence_vlm._EVIDENCE_RENDERERS` (tools,
    task, yolo, motion, variant) -- rather than comparing `{}` against `{}`.
    Comparing two empty dicts would pass regardless of what the shipped
    config actually selects, which is exactly the passes-either-way defect
    this test must not have: it has to be sensitive to
    `config/arbiter.json`'s real, on-disk value, not to a value hardcoded on
    both sides of the assertion.

    Breaks if: `config/arbiter.json`'s `vlm_evidence_context` is flipped to
    `true` without `scripts/train_vlm.py` being retrained against a real
    evidence packet -- see this file's own module docstring and
    `scripts/train_vlm.py`'s "THE COUPLING THIS CREATES" section for why
    that flip must never happen alone.
    """
    question = "Is a needle driver being used in this clip?"
    perception = {
        "tools_present": ["needle_driver"],
        "tools": {"needle_driver": 0.87},
        "task_top": "suturing",
        "task": {"suturing": 0.91},
        "yolo": {
            "by_class": {
                "needle_driver": [{"t_seconds": 3.7}, {"t_seconds": 9.4}],
            },
            "max_conf": {"needle_driver": 0.9},
        },
        "motion_v2": {
            "summary": {
                "macro_prev": {"measured": 1, "mean": 5.0},
                "flow_moving_fraction": {"measured": 1, "mean": 0.1},
            },
        },
        "variant": {
            "decided": True, "family": "Large", "p_large": 0.8, "p_mega": 0.1,
        },
    }

    shipped = _shipped_vlm_evidence_context()

    serving_context = perception if shipped else {}
    serving_prompt = build_sampling_prompt(question, serving_context)
    training_prompt = render_training_prompt(
        question, perception if shipped else None)

    assert serving_prompt == training_prompt, (
        "Serving would render a VLM prompt the trained adapter never saw. "
        "config/arbiter.json's vlm_evidence_context and the way the shipped "
        "adapter was trained must agree: flip the switch only together with "
        "an adapter whose training_config.json records the matching "
        "evidence_context (scripts/merge_and_quantise_vlm.py's "
        "check_evidence_parity enforces exactly this at merge time).")

    # SENSITIVITY, so this cannot pass vacuously. The docstring above warns
    # that comparing two empty dicts would pass whatever the config selects;
    # the same hazard exists now in reverse, since both sides are derived from
    # `shipped`. Rendering the OPPOSITE choice must NOT match, which is only
    # true if `perception` really does change the prompt text.
    mismatched = render_training_prompt(
        question, None if shipped else perception)
    assert serving_prompt != mismatched, (
        "The evidence packet made no difference to the rendered prompt, so "
        "this test could not detect a train/serve mismatch at all.")


def test_build_messages_omits_assistant_turn_when_answer_is_none():
    """Breaks if: an assistant turn is appended even when answer=None,
    which would break the eval/generation code path (add_generation_prompt
    expects the LAST turn to be the user turn)."""
    messages = build_messages("q?", images=["img0"], answer=None)
    assert [m["role"] for m in messages] == ["user"]


def test_build_messages_appends_assistant_turn_with_the_answer():
    messages = build_messages("q?", images=["img0"], answer="Yes")
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["content"] == [{"type": "text", "text": "Yes"}]


def test_build_messages_puts_every_image_before_the_text_block():
    """Breaks if: the text block is inserted before the images, or an image
    is dropped -- both silently change what the model is shown without
    changing this function's return shape."""
    messages = build_messages("q?", images=["img0", "img1", "img2"], answer=None)
    content = messages[0]["content"]
    assert [c["type"] for c in content] == ["image", "image", "image", "text"]
    assert [c["image"] for c in content[:3]] == ["img0", "img1", "img2"]


def test_build_messages_answer_is_stringified():
    """count_open answers are plain strings already, but this guards
    against a caller passing a non-string (e.g. an int count) silently
    reaching the chat template as the wrong type."""
    messages = build_messages("q?", images=[], answer=3)
    assert messages[1]["content"][0]["text"] == "3"


# ============================================================================
# collator's shape contract: the label-masking arithmetic, pure python
# ============================================================================


def test_mask_prompt_tokens_masks_exactly_the_prompt_span():
    """Breaks if: the mask is applied to the wrong span (e.g. masking the
    ANSWER instead of the prompt), which would silently train the model to
    reproduce the question and never learn from the answer."""
    token_ids = [10, 11, 12, 13, 14]
    labels = mask_prompt_tokens(token_ids, prompt_len=3)
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 13, 14]


def test_mask_prompt_tokens_prompt_len_zero_masks_nothing():
    token_ids = [1, 2, 3]
    assert mask_prompt_tokens(token_ids, prompt_len=0) == token_ids


def test_mask_prompt_tokens_prompt_len_equal_to_length_masks_everything():
    token_ids = [1, 2, 3]
    assert mask_prompt_tokens(token_ids, prompt_len=3) == [IGNORE_INDEX] * 3


def test_mask_prompt_tokens_raises_when_prompt_len_out_of_range():
    """Breaks if: an out-of-range prompt_len is silently clamped (e.g. via
    slicing) instead of raising -- a clamp here would silently mask the
    wrong span rather than surfacing the bug that produced it."""
    with pytest.raises(ValueError):
        mask_prompt_tokens([1, 2, 3], prompt_len=4)
    with pytest.raises(ValueError):
        mask_prompt_tokens([1, 2, 3], prompt_len=-1)


# ============================================================================
# T4 / sm_75 compatibility knobs
# ============================================================================


def test_bnb_config_uses_fp16_compute_dtype_never_bf16():
    """Breaks if: bnb_4bit_compute_dtype is changed to 'bfloat16' -- the T4
    the model serves on has no native bf16 support."""
    kwargs = bnb_config_kwargs()
    assert kwargs["bnb_4bit_compute_dtype"] == "float16"
    assert kwargs["load_in_4bit"] is True
    assert kwargs["bnb_4bit_quant_type"] == "nf4"


def test_attn_implementation_is_not_flash_attention_2():
    """Breaks if: ATTN_IMPLEMENTATION is set to 'flash_attention_2', which
    does not run on the sm_75 T4 this model must serve on."""
    assert ATTN_IMPLEMENTATION != "flash_attention_2"
    assert ATTN_IMPLEMENTATION == "sdpa"


# ============================================================================
# QADataset -- real PIL, no torch (PIL IS installed on this login node)
# ============================================================================


def _write_tiny_jpeg(path):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(str(path), "JPEG")


def test_qadataset_len_and_getitem_load_real_frames(tmp_path):
    frame_path = tmp_path / "frame_00.jpg"
    _write_tiny_jpeg(frame_path)
    records = [{
        "question": "Is a needle driver being used?",
        "answer": "Yes",
        "frame_paths": [str(frame_path), str(frame_path)],
    }]
    dataset = QADataset(records)
    assert len(dataset) == 1
    item = dataset[0]
    assert item["question"] == "Is a needle driver being used?"
    assert item["answer"] == "Yes"
    assert len(item["images"]) == 2
    assert item["images"][0].size == (8, 8)


def test_load_frames_preserves_order(tmp_path):
    """Breaks if: load_frames sorts or otherwise reorders paths instead of
    reading them in the given order."""
    paths = []
    for i in range(3):
        p = tmp_path / ("f%d.jpg" % i)
        from PIL import Image
        Image.new("RGB", (4, 4), color=(i * 10, 0, 0)).save(str(p), "JPEG")
        paths.append(str(p))
    images = load_frames(paths)
    assert len(images) == 3
    for image, expected_r in zip(images, (0, 10, 20)):
        # JPEG is lossy; allow a small tolerance around the encoded value.
        assert abs(image.getpixel((0, 0))[0] - expected_r) <= 5


# ============================================================================
# CLI defaults -- match the documented hyperparameter reasoning
# ============================================================================


def test_cli_defaults_match_documented_hyperparameters():
    args = build_arg_parser().parse_args([])
    assert args.epochs == 2
    assert args.lora_r == 16
    assert args.lora_alpha == 32
    assert args.batch_size == 1
    assert args.grad_accum == 16
    assert args.lr == pytest.approx(2e-4)
    assert args.max_eval_examples == 300


def test_cli_output_dir_defaults_to_a_staging_path():
    """Checkpoints must survive Condor eviction -- see the module docstring.
    Breaks if: --output-dir's default is changed to a relative/scratch
    path."""
    args = build_arg_parser().parse_args([])
    assert args.output_dir.startswith("/staging/")


def test_cli_dry_run_and_eval_only_are_off_by_default():
    args = build_arg_parser().parse_args([])
    assert args.dry_run is False
    assert args.eval_only is False
    assert args.no_resume is False


# ---------------------------------------------------------------------------
# Option (b): real evidence in the training prompt. The invariant that matters
# is ALL-OR-NOTHING -- a partial join trains a mixture and nothing reports it.
# ---------------------------------------------------------------------------

def _rec(case, part, t0, t1, **extra):
    r = {"case": case, "part": part, "t_start": t0, "t_stop": t1,
         "question": "q?", "answer": "a", "frame_paths": []}
    r.update(extra)
    return r


def test_evidence_key_joins_manifest_and_cache_records():
    """Both files come from the same manifest, so the key must match across
    them by value, not identity."""
    m = _rec("case_000", "1.0", 1773.300869, 1803.300869)
    c = {"case": "case_000", "part": "1.0",
         "t_start": 1773.300869, "t_stop": 1803.300869, "evidence": {"x": 1}}
    assert tv.evidence_key(m) == tv.evidence_key(c)


def test_evidence_key_tolerates_last_bit_float_noise():
    """Rounded to 6 decimals -- microsecond precision on a seconds-valued
    timestamp. Far finer than any real window boundary, far coarser than
    float noise, so a join cannot miss for a reason unrelated to the data."""
    a = _rec("case_000", "1.0", 1773.3008690000001, 1803.300869)
    b = _rec("case_000", "1.0", 1773.3008689999997, 1803.300869)
    assert tv.evidence_key(a) == tv.evidence_key(b)


def test_attach_evidence_raises_on_a_partial_join(tmp_path):
    """THE LOAD-BEARING TEST. A partial join does not fail on its own -- it
    trains on a MIXTURE of evidence-bearing and empty prompts, teaching the
    model that the evidence block is optional, with nothing in the loss curve
    or the eval score to say so. Refusing to start is the only signal."""
    records = [_rec("case_000", "1.0", 1.0, 2.0), _rec("case_001", "1.0", 3.0, 4.0)]
    cache = {tv.evidence_key(records[0]): {"tools": {}}}
    with pytest.raises(KeyError, match="no cached evidence"):
        tv.attach_evidence(records, cache)


def test_attach_evidence_populates_every_record_on_a_full_join():
    records = [_rec("case_000", "1.0", 1.0, 2.0), _rec("case_001", "1.0", 3.0, 4.0)]
    cache = {tv.evidence_key(r): {"marker": r["case"]} for r in records}
    tv.attach_evidence(records, cache)
    assert [r["evidence"]["marker"] for r in records] == ["case_000", "case_001"]


def test_render_training_prompt_without_evidence_is_unchanged():
    """The shipped default must keep rendering the empty context -- this is
    half of the coupling with scripts/inference.py's vlm_evidence_context."""
    assert tv.render_training_prompt("q?") == tv.render_training_prompt("q?", None)
    assert tv.render_training_prompt("q?", {}) == tv.render_training_prompt("q?")


def test_render_training_prompt_with_evidence_actually_differs():
    """Breaks if the evidence argument is accepted and then dropped on the
    floor -- which would look exactly like a successful option-(b) run."""
    plain = tv.render_training_prompt("q?")
    withev = tv.render_training_prompt(
        "q?", {"tools_present": ["needle driver"], "variant": {"family": "large"}})
    assert withev != plain
    assert len(withev) > len(plain)


# --------------------------------------------------------------------------
# v6 base model: the surgical checkpoint, and the two invariants around it
# --------------------------------------------------------------------------

def test_train_and_merge_agree_on_the_base_model():
    """THE INVARIANT THAT BREAKS SILENTLY IF IT BREAKS AT ALL.

    scripts/merge_and_quantise_vlm.py merges the trained LoRA back into a base
    it loads independently. If it loads a DIFFERENT base than the adapter was
    fitted on, `merge_and_unload()` still succeeds -- the shapes match, both
    are Qwen2.5-VL-7B -- and produces a model whose weights are one model's
    base plus another model's deltas. No exception, no warning, and the only
    symptom is a quality drop indistinguishable from a bad fine-tune.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "merge_and_quantise_vlm",
        Path(__file__).resolve().parents[1] / "scripts" / "merge_and_quantise_vlm.py")
    merge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merge)

    import train_vlm

    assert train_vlm.DEFAULT_BASE_MODEL == merge.DEFAULT_BASE_MODEL_ID


def test_finetune_base_is_distinct_from_the_serving_fallback():
    """`DEFAULT_MODEL_DIR` is `call_vlm`'s default `model_dir` -- the SERVING
    fallback -- and must not follow the fine-tune base around. Two names, two
    meanings; collapsing them is how a training-side edit silently becomes a
    serving-side one."""
    from surgvu.evidence_vlm import DEFAULT_FINETUNE_BASE, DEFAULT_MODEL_DIR

    assert DEFAULT_FINETUNE_BASE != DEFAULT_MODEL_DIR
    assert DEFAULT_MODEL_DIR == "Qwen/Qwen2.5-VL-7B-Instruct"


def test_finetune_base_is_an_absolute_path_not_a_hub_id():
    """condor/train_vlm.sh points HF_HOME at
    /staging/n/nkalthoff/surgvu26/hf_cache, and the surgical checkpoint lives in
    a DIFFERENT tree (/staging/n/nkalthoff/hf_cache). Under a bare hub id an
    offline execute node looks in the wrong cache and misses -- failing with
    HF_HUB_OFFLINE set, or silently pulling 16GB without it."""
    from surgvu.evidence_vlm import DEFAULT_FINETUNE_BASE

    assert DEFAULT_FINETUNE_BASE.startswith("/")
    assert "Surg-CholecT50" in DEFAULT_FINETUNE_BASE


# --------------------------------------------------------------------------
# v6 curriculum: continuing training from a previous stage's adapter
# --------------------------------------------------------------------------

def test_load_model_and_processor_defaults_to_not_trainable():
    """`trainable` must default False so the eval path (--eval-only, which
    passes adapter_dir) keeps loading a frozen adapter for inference."""
    import inspect

    import train_vlm

    params = inspect.signature(train_vlm.load_model_and_processor).parameters
    assert params["trainable"].default is False
    assert params["adapter_dir"].default is None


def test_continuing_a_stage_passes_is_trainable_to_peft():
    """THE SILENT FAILURE THIS GUARDS.

    `PeftModel.from_pretrained` defaults to is_trainable=False -- it loads
    every weight with requires_grad=False, for inference. Handed to a Trainer
    that run does not fail: it trains zero parameters, the loss barely moves,
    checkpoints appear on schedule, and the 'fine-tuned' output is a
    byte-for-byte copy of its input, with nothing in the log saying so.
    """
    import inspect

    import train_vlm

    source = inspect.getsource(train_vlm.load_model_and_processor)
    assert "is_trainable=bool(trainable)" in source
    # ...and the k-bit preparation must happen for a continued run too, or the
    # quantised base never propagates gradients back into the LoRA weights.
    assert "prepare_model_for_kbit_training" in source


def test_run_training_refuses_to_train_zero_parameters():
    """The check that converts the silent no-op above into a loud failure."""
    import inspect

    import train_vlm

    source = inspect.getsource(train_vlm.run_training)
    assert 'report["trainable"] == 0' in source
    assert "raise SystemExit" in source


def test_trainable_parameter_report_counts_zero_for_a_frozen_model():
    """The input to that guard: a model with everything frozen reports 0, so
    the guard actually fires rather than dividing by a nonzero count."""
    import train_vlm

    class _Param:
        def __init__(self, n, requires_grad):
            self._n, self.requires_grad = n, requires_grad

        def numel(self):
            return self._n

    class _Model:
        def named_parameters(self):
            return [("a", _Param(100, False)), ("b", _Param(50, False))]

    report = train_vlm.trainable_parameter_report(_Model())
    assert report["trainable"] == 0
    assert report["total"] == 150


def test_trainable_parameter_report_counts_unfrozen_parameters():
    import train_vlm

    class _Param:
        def __init__(self, n, requires_grad):
            self._n, self.requires_grad = n, requires_grad

        def numel(self):
            return self._n

    class _Model:
        def named_parameters(self):
            return [("a", _Param(100, False)), ("b", _Param(50, True))]

    report = train_vlm.trainable_parameter_report(_Model())
    assert report["trainable"] == 50 and report["total"] == 150


# --------------------------------------------------------------------------
# frame subsampling: decoupling training cost from what extraction wrote
# --------------------------------------------------------------------------

def _frames(n=16):
    return ["f%02d" % i for i in range(n)]


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 8, 12, 16])
def test_subsample_returns_exactly_k_frames(k):
    """Rounding can collide on short lists. The count must be exactly k, not
    'however many survived deduplication' -- a batch whose records carry
    different image counts is a different prompt shape per example."""
    import train_vlm

    assert len(train_vlm.subsample_frames(_frames(), k)) == k


def test_subsample_keeps_both_endpoints():
    """The frames span a 30-second window. Taking a PREFIX would train the
    model on the first third of every clip and never show it what the question
    is often about."""
    import train_vlm

    got = train_vlm.subsample_frames(_frames(), 4)
    assert got[0] == "f00" and got[-1] == "f15"


def test_subsample_is_evenly_spaced():
    import train_vlm

    got = train_vlm.subsample_frames(_frames(), 4)
    assert got == ["f00", "f05", "f10", "f15"]


def test_subsample_preserves_order():
    import train_vlm

    got = train_vlm.subsample_frames(_frames(), 8)
    assert got == sorted(got)


@pytest.mark.parametrize("k", [0, None, 16, 32])
def test_subsample_passthrough_cases(k):
    """0/None means 'all of them'; asking for more than exist returns what
    exists rather than padding or raising."""
    import train_vlm

    assert train_vlm.subsample_frames(_frames(), k) == _frames()


def test_subsample_does_not_mutate_its_input():
    import train_vlm

    original = _frames()
    train_vlm.subsample_frames(original, 4)
    assert original == _frames()


def test_qadataset_defaults_to_every_frame():
    """The default must stay 'all', so a run that does not pass --max-frames
    behaves exactly as before this knob existed."""
    import inspect

    import train_vlm

    assert inspect.signature(train_vlm.QADataset).parameters["max_frames"].default == 0


# --------------------------------------------------------------------------
# filter_records_with_frames: threaded, and ORDER-PRESERVING
# --------------------------------------------------------------------------

def _frame_rec(name, paths):
    return {"question": name, "frame_paths": [str(p) for p in paths]}


def _corpus(tmp_path, n=60, frames=4):
    """n records, every frame present, in a known order."""
    out = []
    for i in range(n):
        paths = []
        for f in range(frames):
            p = tmp_path / ("r%03d_f%d.jpg" % (i, f))
            p.write_bytes(b"x")
            paths.append(p)
        out.append(_frame_rec("q%03d" % i, paths))
    return out


def test_filter_preserves_input_order(tmp_path):
    """NOT COSMETIC. sample_eval_records draws a SEEDED random.sample over
    this function's output, so an order that depended on thread scheduling
    would select a different eval set on every run -- the same
    irreproducibility the threaded frame extraction had to avoid."""
    import train_vlm

    records = _corpus(tmp_path, n=60)
    kept, dropped = train_vlm.filter_records_with_frames(records, workers=16)
    assert dropped == 0
    assert [r["question"] for r in kept] == ["q%03d" % i for i in range(60)]


def test_threaded_and_serial_agree_exactly(tmp_path):
    import train_vlm

    records = _corpus(tmp_path, n=40)
    # drop a frame from a few records, scattered
    for i in (3, 17, 39):
        Path(records[i]["frame_paths"][-1]).unlink()

    serial, d_serial = train_vlm.filter_records_with_frames(records, workers=1)
    threaded, d_threaded = train_vlm.filter_records_with_frames(records, workers=16)

    assert d_serial == d_threaded == 3
    assert [r["question"] for r in serial] == [r["question"] for r in threaded]


def test_a_record_missing_any_frame_is_dropped(tmp_path):
    """Checks EVERY path, not just the first -- a partially written window
    must not train."""
    import train_vlm

    records = _corpus(tmp_path, n=4, frames=5)
    Path(records[2]["frame_paths"][3]).unlink()          # middle frame, middle record
    kept, dropped = train_vlm.filter_records_with_frames(records, workers=8)
    assert dropped == 1
    assert [r["question"] for r in kept] == ["q000", "q001", "q003"]


def test_filter_handles_an_empty_corpus(tmp_path):
    import train_vlm

    assert train_vlm.filter_records_with_frames([], workers=8) == ([], 0)


def test_filter_does_not_mutate_its_input(tmp_path):
    import train_vlm

    records = _corpus(tmp_path, n=6)
    before = [r["question"] for r in records]
    train_vlm.filter_records_with_frames(records, workers=8)
    assert [r["question"] for r in records] == before


def test_skip_frame_check_defaults_off():
    """The check must stay on by default. Skipping it turns a dropped record
    into a crash inside a training step, which is only an acceptable trade
    when the extraction is known to have finished cleanly.

    Tests the PARSER, not the source text: an earlier version of this grepped
    inspect.getsource(main) for the argparse line, which lives in
    build_arg_parser and so never matched."""
    import train_vlm

    parser = train_vlm.build_arg_parser()
    assert parser.parse_args([]).skip_frame_check is False
    assert parser.parse_args(["--skip-frame-check"]).skip_frame_check is True


def test_skip_frame_check_announces_what_it_disabled():
    """A silent skip would make a crash mid-training inexplicable. The run
    must say it is not checking."""
    import inspect

    import train_vlm

    source = inspect.getsource(train_vlm.main)
    assert "if args.skip_frame_check:" in source
    assert "NOT verifying that frame files exist" in source


# --------------------------------------------------------------------------
# the chat template: the surgical base ships none, and its tokenizer's is
# text-only
# --------------------------------------------------------------------------

def test_vendored_template_handles_vision():
    """THE FAILURE THIS PREVENTS IS SILENT.

    nvidia/Qwen2.5-VL-7B-Surg-CholecT50 ships no chat_template.json, and the
    `chat_template` in its tokenizer_config is the TEXT-ONLY Qwen2.5 one --
    2,427 chars, tools-aware, with no vision_start, image or video handling.
    Falling back to it costs one line and looks correct; training through it
    would render every prompt WITHOUT IMAGE TOKENS, and a vision model
    fine-tuned on text alone still converges to a plausible loss curve.
    """
    import train_vlm

    template = train_vlm.load_vl_chat_template()
    assert "vision_start" in template
    assert "image" in template
    assert "video" in template


def test_load_vl_chat_template_rejects_a_text_only_template(tmp_path):
    """The assertion is the whole value of the loader -- without it this is
    just read_text()."""
    import train_vlm

    bad = tmp_path / "text_only.jinja"
    bad.write_text("{%- if tools %}\n{{- 'no vision here' }}\n{%- endif %}",
                   encoding="utf-8")
    with pytest.raises(ValueError, match="not a vision chat template"):
        train_vlm.load_vl_chat_template(bad)


def test_the_vendored_template_file_is_in_the_repo():
    """Vendored rather than read from another checkpoint's snapshot, so it is
    version-controlled and travels with the code."""
    import train_vlm

    assert train_vlm.CHAT_TEMPLATE_PATH.exists()
    assert train_vlm.CHAT_TEMPLATE_PATH.name.endswith(".jinja")


def test_loader_refuses_a_processor_whose_template_lacks_vision():
    """The second guard: even a processor that HAS a template must have a
    vision one, or the run refuses rather than training on text."""
    import inspect

    import train_vlm

    source = inspect.getsource(train_vlm.load_model_and_processor)
    assert 'if "vision_start" not in (processor.chat_template or "")' in source
    assert "Refusing to train" in source


def test_resume_is_abandoned_when_torch_cannot_load_optimizer_state():
    """transformers >= 4.56 refuses to torch.load an optimizer state unless
    torch >= 2.6 (CVE-2025-32434), and this project PINS torch 2.5.1 because
    the T4 it serves on is sm_75.

    The failure mode this guards is not 'cannot resume' -- it is that once ANY
    checkpoint exists, every retry dies on it in minutes, so condor's
    max_retries=5 spends all five attempts without reaching a training step.
    Observed on v6 stage 1 (job 9712725).
    """
    import inspect

    import train_vlm

    source = inspect.getsource(train_vlm.run_training)
    assert "(major, minor) < (2, 6)" in source
    assert "STARTING FROM SCRATCH" in source
    assert "resume = None" in source
