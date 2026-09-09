# v5 Plan 2 -- W4 + W5: The Evidence VLM and the Arbiter


**Goal:** Put a VLM in the shipping path so the pipeline can answer questions the 11-intent router has no form for -- the "last chance to get stuff right" of the original v5 vision -- and give it enough authority to actually change an answer.

**Architecture:** The VLM reads the evidence packet (CNN probabilities, timestamped YOLO detections, the variant block, the motion vector) alongside the 16 anchor frames. It is not a blind captioner; it is a reader of our own perception output. An arbiter then decides between the router's form and the VLM's draft, under a policy set by one config key.

**Spec:** `docs/design/specs/2026-08-24-v5-parallel-build-design.md` (W4, W5)
**Depends on:** Plan 1 (the evidence packet, complete). **Depends on for quality:** Plan 3 / W6a -- an untuned model measurably makes things worse.

## Why this plan exists -- the router's ceiling is structural

Demonstrated directly against the shipped router:

    "How much blood loss has occurred?"   -> count_open    -> "One"
    "Is the patient stable?"              -> unknown_polar -> "Yes"
    "What complication is developing?"    -> unknown_open  -> "The procedure involves surgical instruments."
    "Describe what the assistant is doing." -> task_open   -> "Suturing"

Eleven regex intents always match something, and every match has a hardcoded form. The pipeline never
abstains and never says anything it was not pre-programmed to say. No amount of perception work changes
that; only a generative answerer does.

**And the VLM has never shipped.** There is no `--vlm` in `containers/Dockerfile` or
`containers/surgvu26-submission.def`. The flag exists in `scripts/inference.py` and has never been in an
ENTRYPOINT.

## Global Constraints

- **Grand Challenge runtime:** 10 min/case, one case per invocation, 32 GB DRAM, and either a single T4
  (16 GiB, sm_75 -- **no bf16, no FlashAttention-2**) or **NO GPU AT ALL**. No internet.
- **7B at fp16 is ~15 GB against 16 GiB.** The shipped path is **4-bit NF4**, which is CUDA-only -- so the
  VLM **cannot run on a No-GPU draw** and the pipeline must still answer without it. That is a
  correctness requirement, not a nicety.
- 4-bit output is not bit-identical across GPU architectures. Anything tuned on Ampere/L40 is
  **re-validated on T4-class hardware** before it ships.
- **The container always writes an answer.** A missing response scores 0; a wrong polar answer still
  scores 0.7015. Every VLM path is wrapped so any failure logs and continues -- the R18 idiom already
  established in `scripts/inference.py`. Follow it; do not invent a second pattern.
- Every frame passes through `preprocess.prepare_frame`. The bottom UI band stays blurred.
- **`OVERLAY_PROMPT` from the groupmate's code is deleted, not ported.** It instructs the model to read
  the numbered tool list in the UI band, which the rules prohibit.
- **No `opscribe_pipeline` imports.** SurgVU26 Cat 2 is independent, and the container is offline and
  self-contained.
- Model: `Qwen/Qwen2.5-VL-7B-Instruct` everywhere -- Evidence VLM, arbitration, and Plan 3's fine-tune base.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/surgvu/evidence_vlm.py` *(new)* | The Evidence VLM: packet -> prompt -> answer + confidence. |
| `src/surgvu/arbiter.py` *(new)* | The three policies. One decision point, one config key. |
| `config/arbiter.json` *(new)* | `{"mode": "challenger", ...}` plus the confidence floors. |
| `scripts/inference.py` *(modify)* | `--vlm` wired through the arbiter; `--arbiter-mode`. |
| `containers/Dockerfile`, `containers/surgvu26-submission.def` *(modify)* | `--vlm` in the ENTRYPOINT and the parallel `%runscript`; the weights COPYed in. |

---

### Task 1: Port the adaptive confidence sampler

The groupmate's `vlm_pass1_adaptive.py` already implements adaptive self-consistency sampling with an
ACCEPT/ESCALATE route. Keep the shape; sever the dependencies.

**Files:** Create `src/surgvu/evidence_vlm.py`; Test `tests/test_evidence_vlm.py`

- [ ] Port `ConfidenceResult(answer, confidence, n_calls_used, all_answers, agreed)` and
  `route(result, confidence_threshold)` -> ACCEPT/ESCALATE.
- [ ] **Sever `from opscribe_pipeline...`** -- both the VLM provider and the video decoder. Use
  `surgvu.perceive` for frames.
- [ ] **Delete `OVERLAY_PROMPT`.** Do not port it under any name.
- [ ] **Treat `temperature=0.4` as a hypothesis, not a constant.** Its supporting sweep covers 4 cases and
  is scored by `is_correct()` using `pred == a_clean or pred in a_clean` -- substring matching that
  inflates accuracy. Re-measure before adopting. Keep the qualitative finding as a caution: at
  temperature 0.1 the model was confidently wrong with full self-agreement on case122, case127 and
  case130 -- **agreement is not calibration**, which is why Task 7's cross-model disagreement signal is
  the better confidence channel.
- [ ] Torch imports inside methods; the module must import without torch so the prompt-assembly and
  routing logic is testable on the login node.
- [ ] Commit.

---

### Task 2: The evidence prompt

This is what makes it an *Evidence* VLM rather than a stock one, and it is the difference between the
0.49-0.57 measurements and something useful.

**Files:** Modify `src/surgvu/evidence_vlm.py`; Test `tests/test_evidence_prompt.py`

- [ ] Render the packet into the prompt: `tools` probabilities above threshold, YOLO detections **with
  their timestamps** ("needle driver at 3.7s and 9.4s, absent between" -- a statement the pooled
  confidences could never make), the variant block when present, and the motion vector as calibrated
  language ("active, tool-dominant") rather than raw floats.
- [ ] Include the CNN/YOLO **disagreement** when Task 7 lands -- that is the honest uncertainty channel.
- [ ] Assert the prompt NEVER contains anything derived from the UI band.
- [ ] Test that an absent evidence block simply omits its section rather than emitting "None" or an empty
  header -- a prompt full of nulls teaches the model that nulls are normal.
- [ ] Commit.

---

### Task 3: The arbiter, with three policies

**Files:** Create `src/surgvu/arbiter.py`, `config/arbiter.json`; Test `tests/test_arbiter.py`

| mode | behaviour |
|---|---|
| `fallback` | Router answers; VLM only on unknown intent or sub-floor confidence. |
| **`challenger`** *(SHIP THIS)* | VLM always drafts; router wins ties; VLM overrides only when router confidence is below floor AND VLM confidence above ceiling. |
| `primary` | VLM answers; router validates and rewrites the *form* so BERTScore-friendly phrasing survives. |

- [ ] **Ship `challenger`, not `fallback`.** `INTENT_UNKNOWN_OPEN` fires on 0 of 11 sample questions, so
  `fallback` has a measured ceiling of exactly zero -- it is the freedom this design was commissioned for,
  gated out by default. Record that measurement in the module docstring.
- [ ] All three implemented; one config key selects. Switching modes must not require a code change.
- [ ] Every mode must fall through to the router's answer if the VLM is absent, failed, or returned
  nothing usable.
- [ ] Commit.

---

### Task 4: Wire `--vlm` through serving AND into the container

**Files:** Modify `scripts/inference.py`, `containers/Dockerfile`, `containers/surgvu26-submission.def`

- [ ] `--vlm` and `--arbiter-mode` flags; the VLM constructed lazily so a router-handled case pays nothing.
- [ ] Wrap every VLM call in the R18 idiom -- traceback + WARNING + continue with the router's answer.
- [ ] **Add `--vlm` to the Dockerfile ENTRYPOINT and the .def `%runscript`, together.** The .def's own
  header warns that "a flag present in one and not the other would validate a container nobody uploads."
- [ ] COPY the model weights into the image. Check the resulting image size against Grand Challenge's
  limit before building.
- [ ] **Test the No-GPU path explicitly**: with no CUDA available the VLM must be skipped and the routed
  answer written, not an exception.
- [ ] Commit; the controller builds on a compute node.

---

### Task 5: Measure it honestly

- [ ] Extend `scripts/flag_matrix.py` with `--vlm` and the arbiter modes. The baseline row stays.
- [ ] Run the matrix. **Report the result whichever way it goes.** If the VLM lowers the sample score,
  that is a finding, not a failure to hide -- and per the user's standing instruction the eleven cases are
  not assumed representative of the leaderboard set, so a local regression is not by itself grounds to
  abandon it.
- [ ] Measure wall-clock per case against the 10-minute budget, on both draws.

---

## Self-Review

**Ordering:** Plan 3 (W6a) should land before this plan's Task 5 is trusted -- an untuned Qwen measured
0.4923-0.5743, all below a baseline that ignores the video. Tasks 1-4 can be built in parallel with
Plan 3; only the *evaluation* depends on the tuned weights.

**The biggest risk** is the T4. 7B at 4-bit is the only configuration that fits, it cannot run at all on a
No-GPU draw, and 4-bit numerics differ across architectures. Task 4's No-GPU test and the T4-class
re-validation are not optional.

**What this plan does not do:** it does not pretrain on public endoscopic corpora (cholect50, DSAD,
endovis18_vqa). That is W6a's deferred second phase, recorded in Plan 3.
