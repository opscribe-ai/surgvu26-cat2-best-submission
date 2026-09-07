# SurgVU 2026 Category 2 — v5: Perception Ensemble + Evidence VLM

**Date:** 2026-08-24
**Status:** Design approved by user, pending spec review
**Supersedes:** nothing — extends `2026-08-04-surgvu26-cat2-design.md` (the Evidence-Arbitration Router), which remains the architectural baseline.
**Approach:** "C" — full parallel build. Chosen by the user over a measured, staged alternative, explicitly and after hearing the attribution objection. See *Risks → Attribution*.

---

## Why v5 exists

v4 is a hard-routed mixture of experts: 12-intent regex router, 12 answer forms, CNN tool/task perception, motion gating. It works. On the 11-case local sample it scores **0.8766** against a **0.6959** zero-perception baseline, and against **0.8294** for a blind (no-video) router. Perception is therefore worth roughly **+0.047** on the sample, net of the cases it breaks.

But the remaining error is small, concentrated, and entirely one kind:

| Case | Score | Failure |
|---|---|---|
| case124 | 0.2402 | Wrong tool noun — predicted Bipolar, gold Cadiere |
| case126 | 0.7015 | Polar wrong — needle driver not detected |
| case132 | 0.7015 | Polar wrong — Large vs Mega needle driver variant |
| ×8 others | 1.0000 | — |

Total recoverable on the sample: **0.1234**, of which case124 alone is **62%**. Not one of these three is a routing, organ, purpose, or procedure error. **All three are tool perception.** That is the entire thesis of v5: the router is done, the perception is not.

Two further measurements bound the design:

- **`INTENT_UNKNOWN_OPEN` fires on 0 of 11 sample questions.** A VLM scoped strictly to unknown intents therefore has a ceiling of exactly zero improvement. If the VLM is to matter, the arbiter must be able to reach further than fallback — hence W5 implements three policies even though fallback ships by default.
- **Stock VLMs score below the blind baseline.** Measured: 4B-fp16 open 0.5743, 4B-fp16 closed 0.5216, 8B-NF4 closed 0.5501, 8B-NF4 open 0.4923 — all beneath 0.6959. The cause is structural, not a prompt defect: gold answers reconstruct *our label taxonomy* (12 tools, 8 tasks, commercial name families), which the CNNs are trained on and a general VLM has never seen. **A VLM that is not fine-tuned on that taxonomy will lose to a lookup table.** This is why W6 is not optional garnish.

---

## Inputs this design consumes

### The groupmate's YOLO detector — verified

At `/staging/groups/bhaskar_opscribe/surgvu_yolo_detector/best.pt` (14.5 MB, 2026-08-21), with run artifacts in `surg_14cls_run_results.tar.gz`.

Training configuration, from `opt.yaml`: **yolov5s**, 100 epochs, batch 16, imgsz 640, Adam, lr0 5e-4, single GPU. A sound single-GPU adaptation of the 4-GPU MONAI/SurgToolLoc `run_5fold.sh` reference recipe (which assumes v5m, 300 epochs, 4×GPU, batch 128).

Final metrics, from `results.csv` epoch 99: P 0.773 / R 0.740 / mAP@0.5 0.773 / mAP@0.5:0.95 0.404. Best epoch 97 at mAP@0.5 0.792.

**The split is clean.** 886 train / 240 val images. Clip-ID analysis: 624 distinct train clips, 156 distinct val clips, **zero shared**. No near-duplicate frame leakage; the metrics are not inflated by that mechanism. (Case-level disjointness is unverified — clip IDs do not encode case — and is a known residual.)

Per-class recall from `confusion_matrix.png` — the number that actually matters here, far more than mAP:

| Class | Recall | | Class | Recall |
|---|---|---|---|---|
| suction irrigator | 1.00 | | tip-up fenestrated grasper | 0.62 |
| monopolar curved scissors | 0.94 | | vessel sealer | 0.62 |
| bipolar dissector | 0.94 | | grasping retractor | 0.56 |
| stapler | 0.91 | | force bipolar | 0.50 |
| bipolar forceps | 0.89 | | clip applier | 0.48 |
| needle driver | 0.89 | | prograsp forceps | 0.43 |
| permanent cautery hook/spatula | 0.88 | | | |
| cadiere forceps | 0.81 | | | |

Two findings drive W2:

1. **Bipolar↔Cadiere confusion is 0.01 in each direction.** That is precisely the error costing us case124 — the largest single recoverable item in the whole system.
2. **Needle driver recall is 0.89**, and case126 is a needle-driver detection miss.

The weak classes fail by *omission*, not confusion: their columns show background-FN of 0.22–0.40 with near-empty off-diagonals. Quiet when unsure, right when it fires. That is the correct failure shape for a guardrail signal.

What it cannot do: the vocabulary has exactly one `needle driver` class, so it cannot resolve **Large vs Mega** (case132). That requires W2's variant head.

**Note for the detector's author:** mAP@0.5:0.95 = 0.404 is the least relevant metric for Category 2. We never consume box coordinates to form an answer — we consume tool *presence* and *timing*. Box tightness at IoU 0.9 does not affect BERTScore at all. mAP@0.5 and the confusion diagonal are the numbers that matter, and both are good. Remaining headroom with **no new labels**: v5s→v5m and 100→300 epochs, per the reference recipe.

### The groupmate's VLM pipeline

At `/staging/groups/bhaskar_opscribe/surgvu_vlm_pipeline.tar.gz` (187 MB, of which ~188 MB is `sample_data/case122..case132`; the code is 686 lines).

Adopted:
- `vlm_pass1_adaptive.py` — `adaptive_confidence_sample(video, question, context, frames_per_call=5, max_samples=3, agreement_threshold=1.0, sampling_temperature=0.4)` returning `ConfidenceResult(answer, confidence, n_calls_used, all_answers, agreed)`, and `route(result, confidence_threshold=0.66)` → ACCEPT / ESCALATE. The adaptive-sampling shape is good and is kept.

Rejected or amended, with reasons:
- **`OVERLAY_PROMPT` in `debug_utils.py` is deleted, not ported.** It instructs the model to read the numbered tool list in the bottom UI band. The challenge rules state that using information available in the UI to make predictions is not allowed. It is currently debug-only and unreachable from `run_pipeline`, and it must never enter inference *or label generation*. There is deliberately no code path bypassing `prepare_frame`.
- **The `opscribe_pipeline` imports are severed.** `from opscribe_pipeline.providers.vlm import get_vlm_provider` and `from opscribe_pipeline.video import VideoDecoder, FrameStore, SamplingStrategy` violate both the SurgVU-independence rule and the offline self-contained container requirement.
- **`temperature=0.4` is treated as a hypothesis, not a constant.** The supporting sweep covers 4 cases and is scored by `is_correct()`, which uses `pred == a_clean or pred in a_clean` — substring matching that inflates accuracy. Re-measure before adopting. The *qualitative* finding is retained as a design caution: at temperature 0.1 the model was confidently wrong with full agreement on case122, case127 and case130, i.e. **agreement is not calibration**, which is exactly why W2's cross-model disagreement signal exists.
- `surgvu_pipeline_v1.py`'s `parse_question_type()` has two branches (RECORDS / LOOK_HARDER) against our 12-intent router. Superseded.

### Data assets

- `/staging/groups/bhaskar_opscribe/surgvu/labels_cat2/SURGVU25_train_labels/` — 155 case directories, each with `tools.csv` (install/uninstall part + time, arm, `commercial_toolname`, `groundtruth_toolname`) and `tasks.csv` (start/stop, `taskname`, `groundtruth_taskname`, `matched_description`).
- `/staging/groups/bhaskar_opscribe/surgvu/shards` — 38 GB, 235 `.npz` files. The pseudo-labeling and fine-tuning corpus.
- `/staging/n/nkalthoff/` — public endoscopic corpora: `cholect50` 22 G, `DSAD.zip` 20 G, `endovis18_vqa` 2.9 G, `cataracts` 62 G.

---

## Architecture

```
Step 0   Ingest video                          unchanged
Step 1   Decode → 16 anchor frames             unchanged
Step 1b  Burst geometry (multi-scale)          W1   --motion-v2
Step 1c  Motion feature vector + optical flow  W1   --motion-v2
Step 2   Perception ensemble                   W2   --yolo, --variant-head
         CNNs + YOLO + variant head,
         timestamped confidences
Step 2b  Evidence packet assembly              W3   (always on)
Step 3/4 Evidence VLM                          W4   --vlm
Step 5   Arbiter                               W5   --arbiter-mode={fallback,challenger,primary}
Step 6   Answer form + emit                    unchanged
```

Every workstream is independently flaggable. This is not a hedge against the design; it is how attribution is recovered after a multi-change submission.

---

## W1 — Motion geometry and features (steps 1b, 1c)

**Current state.** 16 anchors, each with a 3-frame burst at **±67 ms** (`perceive.py:104`, `BURST_FPS = 15.0`). Micro = within-burst mean absolute difference of 64×64 Rec.601 luma; macro = between burst centres at 1.875 s. Both collapse into a single threshold.

*(Corrected 2026-08-24 while writing Plan 1: an earlier draft of this section said ±0.67 s, which is wrong by a factor of ten. The corrected figure strengthens the argument rather than weakening it — see defect 1.)*

**Three defects.**
1. The ±67 ms offset is unjustified — never swept against any objective. Worse, **nothing at all is sampled between 67 ms and 1875 ms**, a 28× span, and that is exactly the range a single tool stroke occupies.
2. A single threshold discards nearly all of the information the scores carry.
3. Mean absolute difference cannot distinguish **camera motion** from **tool motion**. A slow endoscope pan and an active dissection produce similar scalars.

**Changes.**

*Burst geometry.* **Add** log-spaced probes at **±{133, 400, 1200} ms** → 7 frames per anchor, 112 decoded total, filling the empty 67 ms–1875 ms span. This answers "how many neighbors and at what times" with a spread across timescales rather than a point guess. The existing burst is **not modified**: its uniform layout is the `shards_multi16` format that `surgvu/dataset.py` and `surgvu/temporal.py` read, so a non-uniform spacing would silently change what "micro activity" means in both. The two samplers coexist.

*Motion becomes a vector.* Per anchor:
```
MotionVector = [micro_short, micro_mid, micro_long,
                macro_prev, macro_next,
                flow_mag_mean, flow_mag_p90, flow_coherence]
```

*Optical flow.* Farnebäck dense flow at 128×128, OpenCV, CPU-only — no GPU dependency, so it survives a No-GPU draw. `flow_coherence` = fraction of flow vectors within 30° of the global median direction. High coherence ⇒ camera-dominant; low coherence ⇒ tool-dominant. **This is the discrimination the current scalar structurally cannot make**, and it is the reason to add flow at all.

*Leverage rather than threshold.* The vector enters the evidence packet, and reaches the VLM as calibrated natural language ("active, tool-dominant"; "still, camera-dominant"). The binary threshold survives only where a specific router form requires a yes/no.

*Tuning objective.* Offsets and thresholds are swept against a **proxy label derived from `tasks.csv`**: timestamps inside an annotated task interval are "active", timestamps between intervals are "idle". This is a real objective function over 155 cases, not intuition. It is not the answer metric, and its limitations are recorded rather than hidden.

**Flag:** `--motion-v2`. **Cost:** 7× frame decode, CPU flow on 112 frames. Must be measured against the 10-min budget.

---

## W2 — Perception ensemble (step 2)

**CNNs stay.** They are the guardrail and they are what earns +0.047 over blind. Nothing here removes them.

**YOLO joins as a second opinion.** `best.pt` runs on the 16 anchors.

*Class handling.* Her 14 classes = our 12 + `bipolar dissector` + `suction irrigator`. The two extras are **kept as evidence, not suppressed**. A confident suction-irrigator detection is useful *negative* evidence — it constrains what else the frame can contain — even though it can never be the answer. Mapping 14→12 happens only at answer-formatting time, in Step 6.

*Timestamped confidences.* Every detection carries `(anchor_idx, t_seconds, class, conf, box)`. Today confidences are pooled across frames and lose time entirely. Time-resolved detections unlock ordering questions ("what was used first / next") and let the VLM observe structure like "needle driver present at t=3.7 s and t=9.4 s, absent between" — which is a different claim from "needle driver present with confidence 0.6".

*Variant head — new.* Large vs Mega needle driver. This is the only component that can address case132.

- **Labels are free.** `tools.csv` records `commercial_toolname` per install interval, so every frame inside an interval is automatically labeled Large-family or Mega-family. No annotation, no boxes required.
- **Whole-frame binary classifier**, cropped to YOLO's needle-driver box when one is available. That crop is the concrete payoff of having a detector at all.
- **Case-level priors are useless and must not be used as a shortcut.** 137 of 154 cases contain *both* families; only 16 (10.4%) are single-family. The variant must be resolved visually, per clip.
- Corpus prior for calibration only: needle driver n=1629 — Large family 1022 (62.7%), Mega family 605 (37.1%).

*Disagreement as signal.* CNN-vs-YOLO agreement is computed explicitly and exported (`tool_agreement`, `top_disagreement`). Disagreement is the honest uncertainty channel — unlike self-consistency agreement, which the groupmate's own temperature sweep showed can be confidently wrong.

*Router fix, in scope here.* `src/surgvu/router.py:651` lists `"large"`, `"mega"`, `"medium"`, `"long"`, `"micro"`, `"wristed"` among words that "identify nothing on their own", so **"large needle driver" collapses to "needle driver"**. With a variant head those words become resolvable, so the strip list must be narrowed to preserve family terms. This bug affects 3 of 11 sample questions (27%), of which we currently score 1/3.

**Flags:** `--yolo`, `--variant-head`.

---

## W3 — Evidence packet (step 2b, always on)

One typed structure that every downstream stage reads. Without this contract, six concurrent workstreams become unmaintainable.

**It already exists in embryo and must be extended, not replaced.** `perceive.clip_record()` is this structure today, and it carries a hard-won safety property: omit an optional block and the returned dict is byte-identical to the pre-block version, asserted by `tests/test_perceive.py`. Every new evidence source becomes another optional block on that record, inheriting the guarantee that enabling a source cannot by itself move a shipped answer. *(Recorded 2026-08-24 while writing Plan 1; an earlier draft of this section proposed a new parallel type, which would have thrown that property away.)*

```
EvidencePacket:
  frames   : 16 anchors (+ burst references)
  motion   : MotionVector per anchor
  cnn      : per anchor {tool_probs[12], task_probs[8]}
  yolo     : per anchor [Detection{cls, conf, box, t_seconds}]
  variant  : per anchor {large, mega} | None
  agree    : {tool_agreement: float, top_disagreement: (a, b)}
  router   : {intent, slots, form, deterministic_answer | None}
```

Serializable, loggable, and diffable across flag combinations — which is what makes the ablation in *Validation* possible at all.

---

## W4 — Evidence VLM (steps 3, 4)

Consolidates the old steps 3 and 4 into a single VLM that receives **the 16 frames plus the evidence packet rendered as text**, and drafts an answer. It is not a blind captioner; it is a reader of our own perception output. That is the difference between the 0.49–0.57 measured for stock VLMs and what this is trying to be.

*Model — decided.* **`Qwen/Qwen2.5-VL-7B-Instruct` is the single VLM for every VLM stage in v5**: the Evidence VLM here, any decision/arbitration call in W5, and the fine-tuning base in W6a. One model, one checkpoint, one quantization path — no second family to validate, tune, or fit into the image. This also matches the groupmate's existing `VLM_CONFIG` default, so her pipeline ports without a model swap.

Serving consequence, and it is tight: 7B at fp16 is ~15 GB of weights against a **16 GiB T4**, which leaves no room for the vision tower and activations. **The shipped path is 4-bit NF4** (~5 GB), which bitsandbytes supports on sm_75. That in turn means (a) the VLM cannot run at all on a No-GPU draw, and (b) 4-bit output is not bit-identical across architectures, so anything tuned elsewhere is re-validated on T4-class hardware before it ships. Note the measured 8B-NF4 scores (0.4923 open / 0.5501 closed) were **stock, pre-fine-tune** — they bound what an untuned 7B would do here, not what a taxonomy-tuned one will.

*Provenance.* Built from the groupmate's `adaptive_confidence_sample`, ported off `opscribe_pipeline`, retaining `ConfidenceResult` and the ACCEPT/ESCALATE `route()` shape. Sampling parameters re-measured, not inherited.

*Hard runtime constraints — these are not negotiable and they shape the whole stage.*
- Grand Challenge grants **10 min per case, one case per invocation, 32 GB DRAM**, and **either No GPU or a single T4** (16 GiB, sm_75). No internet.
- sm_75 means **no bf16 and no FlashAttention-2**.
- 4-bit NF4 is bitsandbytes/CUDA-only, so **on a No-GPU draw the VLM cannot run at all**.
- 4-bit output is not bit-identical across GPU architectures; anything tuned on Ampere must be validated on T4-class hardware before it ships.

**Therefore the VLM is a flag, and the pipeline must emit a valid answer without it.** The no-VLM path is v4 plus W1–W3. This is a correctness requirement, not a fallback nicety.

**Flag:** `--vlm`.

---

## W5 — Arbiter (step 5)

A configuration policy, not a rewrite. The user's decision was "try fallback first, but don't close out the others" — so all three are implemented and one is selected by a single key.

| Mode | Behaviour |
|---|---|
| `fallback` *(default, ships)* | Router answers. VLM invoked only on unknown intent or sub-floor router confidence. |
| `challenger` | VLM always drafts. Router wins ties. VLM overrides only when router confidence is below floor **and** VLM confidence is above ceiling. |
| `primary` | VLM answers. Router validates and rewrites the *answer form*, so BERTScore-friendly phrasing survives regardless of who produced the content. |

The measured `INTENT_UNKNOWN_OPEN` = 0/11 means `fallback` cannot change the sample score. That is a known property of the default, accepted deliberately: fallback is the safe ship, and `challenger`/`primary` exist so the decision can be revisited on evidence rather than re-implemented.

Anything the router does not have a hard-coded, high-confidence form for is handed to the VLM — the VLM is the last chance to get it right, per the user's framing.

**Flag:** `--arbiter-mode={fallback,challenger,primary}`.

---

## W6 — Training

### W6a — VLM pretrain then fine-tune

*Pretrain* on staged public endoscopic corpora for surgical-scene grounding: `cholect50` (22 G), `DSAD` (20 G), `endovis18_vqa` (2.9 G). **`cataracts` (62 G) is excluded or heavily down-weighted** — wrong domain (ophthalmic, not abdominal/robotic), and it is the largest disk cost in the set.

*Fine-tune on SurgVU.* Base model `Qwen/Qwen2.5-VL-7B-Instruct` (see W4). QA pairs generated automatically from `tools.csv` + `tasks.csv` across 155 cases × 235 shards, in the exact answer forms the router emits.

**This is the highest-leverage item in the design.** The measured collapse of stock VLMs (0.4923–0.5743, all below the 0.6959 blind baseline) is a taxonomy-mismatch problem. Fine-tuning on generated QA over our own label vocabulary is the only mechanism that closes it. Every other workstream is worth hundredths; this one is worth the difference between a VLM that helps and a VLM that must be kept switched off.

Label generation reads only `tools.csv`/`tasks.csv` and decoded frames. It never reads the UI band.

### W6b — YOLO v2

*Logbook-constrained pseudo-labeling.* Run `best.pt` over the 235 shards and **accept a detection only if the logbook says that tool is mounted at that timestamp**. The logbook turns pseudo-labeling from error-amplifying into error-suppressing: the constraint removes exactly the false positives an unconstrained self-training loop would learn to reinforce.

*Retrain* at the reference recipe — v5m, 300 epochs — on 886 hand-labeled plus tens of thousands of constrained pseudo-labeled images. Expected to lift the weak classes (prograsp 0.43, clip applier 0.48, force bipolar 0.50, grasping retractor 0.56), which fail from data scarcity: their instance counts in the hand-labeled set are 38–77 against 222 for the largest class.

---

## Validation

The user's instruction is explicit: do not over-weight the local evaluation, take the leap, accept a different score in either direction, and iterate. This section is therefore a **tripwire and an attribution instrument, not a gate**. Nothing here blocks a ship.

1. **Flag-combination matrix on the 11-case sample.** Record the score for each combination that ships or nearly ships. Cheap, and it is the only thing that will make the eventual leaderboard delta interpretable.
2. **Timing budget, measured not assumed.** 112 decoded frames + CPU optical flow + CNNs + YOLO + up to 3 VLM samples, against 10 min/case, on both draws (T4 and No GPU).
3. **T4-class validation of any quantized path** before it ships.
4. **YOLO re-validated on our own held-out cases**, not only her clip-disjoint val split — case-level disjointness is currently unverified.
5. Carried forward and still approved but de-prioritized: the synthetic evaluation corpus, and the non-circular question-space coverage audit. The existing `scripts/router_coverage.py` audit reports 98.1% intent accuracy against `tests/fixtures/question_variants.json` — 159 hand-written variants, i.e. it is circular and that number should not be quoted as evidence of anything.

---

## Risks

**Attribution.** v5 lands six workstreams at once. The leaderboard move will not be attributable to any single one. This risk was raised, understood, and the user chose the parallel build regardless; it is recorded here as a known cost, not a reopened argument. The per-workstream flags plus the combination matrix are the mitigation, and they only work if the matrix is actually recorded before submitting.

This is the second time in a row. The already-shipped v4 carried **both** the Aug-13 router batch and the motion gate relative to the last scored submission (v2 = 0.8015), so any v4 movement is likewise unattributable — in particular, **a v4 move is not attributable to the motion gate.**

**Submission budget and clock.** Deadlines are **Sep 6 / 13 / 27, 2026**; today is Aug 24. The submission budget is finite (per the Aug-04 design: 10 preliminary attempts, 2 final), several preliminary attempts are already spent, and v4 is uploaded and awaiting a score. The exact remaining count is not verified in this document and should be confirmed before planning around it. Either way v5 is on a short clock against a small number of scoring opportunities. This does not change the decision; it does mean **W6a — the highest-leverage item — should not be the last thing started.**

**Compute budget.** 112 frames + flow + two detectors + adaptive VLM sampling against 10 min/case is not obviously affordable. If it is not, the fallback ordering is: reduce burst offsets from 3 to 2, then cap VLM `max_samples` at 2, then drop flow to 64×64.

**No-GPU draw.** If GC allocates No GPU, the VLM contributes nothing and v5 degrades to v4 + W1 + W2. The design must be correct in that state, not merely survive it.

**Container size.** VLM weights + YOLO + CNNs + variant head in one offline image. Needs a size check before build.

**Small-sample over-reading.** Several YOLO per-class recalls rest on few val instances (the rarest class has 34 total instances across the whole set, so on the order of 7 in val). The 0.94 and 1.00 entries should not be read as precise.

**Pseudo-label drift.** Even logbook-constrained, self-training can entrench systematic errors within the permitted class set. Hold out the 886 hand-labeled images from the pseudo-label loop and report v2 against them.

---

## Key decisions

1. **Perception, not routing, is the remaining error.** All three sample failures are tool identification. v5 spends its effort accordingly.
2. **YOLO is additive evidence, never a replacement.** The CNNs earn their place; the detector is a second opinion whose disagreement is itself a signal.
3. **Motion becomes features, not a threshold.** Optical-flow coherence is added specifically to separate camera from tool motion, which the current scalar cannot do.
4. **The VLM must be fine-tuned or it must stay off.** Stock VLMs measurably lose to a lookup table on this metric.
5. **One VLM everywhere: `Qwen/Qwen2.5-VL-7B-Instruct`,** served 4-bit NF4 on the T4. Evidence VLM, arbitration, and fine-tuning base are all the same checkpoint.
6. **The arbiter is a config key.** Three policies implemented, one shipped, revisitable on evidence.
7. **Every workstream is flagged.** The user gets the big swing; the flags are what let us learn from it afterwards.
8. **The UI band stays out.** `OVERLAY_PROMPT` is deleted rather than ported, and label generation reads only the logbook and decoded frames.

---

## Out of scope

- Submitting to Grand Challenge. The user submits manually.
- Opening a PR, merging to main, or making the repository public.
- Touching `cat1_test_set_public.zip` or rewriting `config/splits.json`.
- Any use of the OpScribe container, venv, HF cache, or `pypkgs`. SurgVU26 Cat 2 is independent.
- Heavy work on the CHTC login node `ap2001`. Training and container builds run in compute jobs.
