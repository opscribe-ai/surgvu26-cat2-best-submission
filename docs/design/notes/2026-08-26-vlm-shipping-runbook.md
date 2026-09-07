# Runbook: getting the VLM from a merged checkpoint into a scored submission

Written while the merge job (9698669) was queued, so the sequence is decided
before the artifacts exist rather than improvised once they do. Every step
names what would make it a FALSE PASS, because most of this project's costly
failures validated green.

## 0. Preconditions already satisfied

- `transformers==4.57.6`, `accelerate==1.14.0`, `bitsandbytes==0.50.1` are in
  BOTH `containers/Dockerfile` and `containers/surgvu26-submission.def`.
  Without these `--vlm` cannot work at all; a build that lacks them is green
  and inert.
- `containers/build_submission.sh` takes `REQUIRE_VLM=1` and refuses to build
  when the checkpoint is missing or is not a loadable directory.
- `condor/validate_image.sh`'s EXPECTED is repointed to generation 2, and its
  GAIN block names case126="Yes" / case132="No" explicitly.
- `condor/validate_image.sub` takes `gpus=1`.

## 1. Confirm the merge actually produced something loadable

    condor_q -l 9698669 | grep -i requiregpus     # -af EVALUATES and prints
                                                  # "undefined"; use -l
    du -sh /staging/n/nkalthoff/surgvu26/models/qwen25vl-7b-nf4
    ls /staging/n/nkalthoff/surgvu26/models/qwen25vl-7b-nf4

FALSE PASS TO AVOID: exit 0 is not evidence. `scripts/merge_and_quantise_vlm.py`
is designed to STOP without verifying if the size gate fails, which is also a
clean exit path. Read the log for the stage-3 line and the reload-and-generate
output, and confirm `config.json` carries a `quantization_config` block --
that block is what lets `evidence_vlm._load_model`'s bare `from_pretrained`
reconstruct the NF4 weights with no quantisation argument of its own.

## 2. Build, with the guard armed

    condor_submit containers/build_submission.sub

Add `environment = "REQUIRE_VLM=1"` to that .sub first (the line is already
there, commented, with the reason). Without it a mistyped VLM_MODEL_SRC
produces a green CNN-only image and the whole exercise is silently wasted.

Watch for the second pip layer's own print: `torch 2.5.1+cu121 | transformers
4.57.6 | ...`. The build asserts torch did not move; if the VLM deps dragged
in a different torch the build fails there rather than at inference.

## 3. Validate BOTH deployment draws -- this is not optional

The grader may allocate either No GPU or one T4. They exercise different code.

    condor_submit condor/validate_image.sub             # No-GPU draw
    condor_submit condor/validate_image.sub gpus=1      # T4-like draw

No-GPU expectation: `try_vlm_result` logs "VLM: no CUDA device available"
and the router's answer stands. All eleven cases answered, GAIN intact.
NF4 is bitsandbytes/CUDA-only, so this is correct behaviour, not a failure.

GPU expectation: the IMAGE STDERR section shows the VLM actually LOADING and
generating. If it shows a swallowed WARNING instead, the flag is inert --
that is the tenth silent-nothing, and the answers can still all be right
because the router alone produces them.

BOTH runs must end with `case126="Yes" and case132="No" both hold`.

## 4. Measure before believing

`scripts/flag_matrix.py` needs no changes for this. `--flags` is generic and
`--fixed-arg` carries an argument onto every combination:

    python3 scripts/flag_matrix.py <sample_dir> \
        --flags --vlm \
        --fixed-arg=--yolo --fixed-arg=--variant-head \
        --fixed-arg=--arbiter-mode=challenger \
        --out baselines/flag_matrix_vlm_challenger.json

    # same again with --arbiter-mode=primary

Two configurations, not a full sweep: 8 combinations x 11 cases at up to 240 s
of VLM budget each is ~3 h of GPU for combinations nobody will ship. Two
targeted runs are 22 invocations each.

Why both modes: `docs/design/notes/2026-08-26-vlm-error-structure-and-
arbiter-mode.md` shows the VLM's entire measured loss is open-ended nouns and
its polar accuracy is >=98%, which argues for `primary` (VLM on polar, router
untouched elsewhere). That argument is from held-out CORPUS cases at fp16.
This is the graded distribution at NF4. Decide on these numbers.

ALSO re-run the held-out 300 through the QUANTISED model. The 0.9092 was
fp16+LoRA. Quantisation error is a measurement, not an assumption, and if NF4
costs polar accuracy then `primary` is the wrong mode no matter what the fp16
numbers said.

## 5. Budget

The CNN-only path measured 118-196 s of 600 s with `GPUs = 0`. The VLM adds up
to `DEFAULT_BUDGET_SECONDS = 240` on top, guarded by a real deadline
(`_deadline_stopping_criteria`, checked per token). Confirm the observed worst
case from step 4, then consider raising the budget toward the window -- the
user has asked explicitly not to leave it unused. A missing response scores 0,
so the deadline stays non-negotiable.

## 6. Submission

**The user submits to Grand Challenge manually. Never submit for them.**
Hand over: the image path, the two validation results, and the flag-matrix
numbers for both arbiter modes.
