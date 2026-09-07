# v5 Plan 4 — W6b: Detector v2, and W6a Phase 2: Endoscopic Pretrain


**Goal:** Finish the two workstreams the other plans deliberately deferred — extend the groupmate's detector with logbook-constrained pseudo-labels and the reference training recipe she staged, and pretrain the VLM on public endoscopic corpora before the SurgVU fine-tune.

**Spec:** `docs/design/specs/2026-08-24-v5-parallel-build-design.md` (W6b; W6a phase 2)

## What the groupmate already built, and what is still unused

She staged three things. Only one is in the shipping path.

| asset | location | status |
|---|---|---|
| `best.pt`, 14-class YOLOv5s | `/staging/groups/bhaskar_opscribe/surgvu_yolo_detector/` | **SHIPPING** — Task 6/11, and worth +0.0543 in combination |
| the yolov5 checkout | same | **SHIPPING** — `Detector` loads from it, and it goes into the image |
| 886 train / 240 val labeled images | `yolo_dataset.tar.gz` | trained `best.pt`; **never extended** |
| `run_5fold.sh` — the MONAI/SurgToolLoc reference recipe | `yolov5/detection_files/` | **UNUSED**: it specifies **v5m, 300 epochs, 4 GPUs, batch 128**; she ran **v5s, 100 epochs, 1 GPU, batch 16** |
| `surgvu_vlm_pipeline.tar.gz` — adaptive confidence sampling | `/staging/groups/bhaskar_opscribe/` | **UNUSED** — ported by Plan 2 Task 1 |

Her detector's measured per-class recall (clip-disjoint split, verified): needle driver 0.89, bipolar
forceps 0.89, cadiere forceps 0.81, monopolar curved scissors 0.94 — but prograsp 0.43, clip applier
0.48, force bipolar 0.50, grasping retractor 0.56. **The weak classes are the rare ones**: their
instance counts in the hand-labeled set are 38-77 against 222 for the largest. That is a data problem,
and there are 38 GB of unlabeled shards sitting next to it.

---

## Global Constraints

- **Split discipline (R30).** Exclude `config/splits_v2.json`'s `heldout` — the 11 graded cases — from
  every training and pseudo-labeling corpus. Compare with `surgvu.sampling.normalize_case_id`, never
  string equality. **Fail loudly if the exclusion removes zero cases.**
- Every frame passes through `preprocess.prepare_frame`; the UI band stays blurred, in label generation
  as much as at serving.
- Report drop counts everywhere. A silently smaller corpus is indistinguishable from a full one.
- Never build containers or run heavy work on the login node.
- The shipped `best.pt` must remain byte-identical until a v2 is *measured* better — a detector swap
  changes the +0.0543 result and must be re-measured through the flag matrix, not assumed.

---

### Task 1: Logbook-constrained pseudo-labels

The constraint is what makes this safe. Unconstrained self-training amplifies a detector's own errors;
the logbook says which tools were *mounted* at each timestamp, so a detection of anything else is
provably wrong and can be dropped rather than reinforced.

**Files:** Create `scripts/pseudo_label_shards.py`, `condor/pseudo_label.sub`/`.sh`; Test `tests/test_pseudo_label.py`

- [ ] Run `best.pt` over the 235 `.npz` shards (38 GB, `/staging/groups/bhaskar_opscribe/surgvu/shards`).
- [ ] **Accept a detection only if `tools.csv` says that tool is installed at that timestamp.** Reuse
  `surgvu.labels`; times are `HH:MM:SS.ffffff` strings (R14) and carry a part (R28).
- [ ] Drop and TALLY: detections of un-mounted tools, detections below a confidence floor, frames whose
  logbook interval cannot be resolved.
- [ ] Report the per-class yield. The interesting number is how many new instances the rare classes gain
  — if prograsp goes from 38 to 400, that is the case for a retrain; if it gains 12, it is not.
- [ ] Commit; controller submits.

---

### Task 2: Detector v2 on the reference recipe

**Files:** Create `scripts/train_detector_v2.py` (or adapt her `run_5fold.sh`), `condor/train_detector_v2.sub`/`.sh`

- [ ] Train **v5m for 300 epochs**, per the MONAI recipe she staged, on the 886 hand-labeled images PLUS
  the constrained pseudo-labels. Her run was v5s/100 on a single GPU — the recipe assumes 4 GPUs and
  batch 128, so adapt the batch and learning rate rather than copying the command.
- [ ] **Hold out the 886 hand-labeled images from the pseudo-label loop** and report v2's metrics against
  them, so improvement is measured against human labels and not against v1's own opinions.
- [ ] Report per-class recall against v1's table above. The rare classes are the whole point.
- [ ] **Do not swap the shipped detector on the strength of mAP.** Re-run `condor/flag_matrix.sub` with v2
  and compare the graded sample end to end. mAP@0.5:0.95 is nearly irrelevant here — we consume presence
  and timing, not box tightness.
- [ ] Commit; controller submits.

---

### Task 3: Endoscopic pretrain (W6a phase 2)

**Files:** Create `scripts/pretrain_vlm.py`, `condor/pretrain_vlm.sub`/`.sh`

- [ ] Pretrain `Qwen/Qwen2.5-VL-7B-Instruct` on staged public endoscopic corpora for surgical-scene
  grounding: `cholect50` (22 G), `DSAD` (20 G), `endovis18_vqa` (2.9 G), all under `/staging/n/nkalthoff/`.
- [ ] **Exclude or heavily down-weight `cataracts` (62 G)** — ophthalmic, not abdominal/robotic, and the
  largest disk cost in the set. If included, say why and report its effect separately.
- [ ] This runs BEFORE Plan 3's SurgVU fine-tune, and the fine-tune runs on top of it. Report both
  checkpoints so the contribution of each stage is separable — otherwise this repeats v4's attribution
  failure at the model level.
- [ ] Commit; controller submits.

---

### Task 4: Re-measure everything through the flag matrix

- [ ] Extend `scripts/flag_matrix.py` to sweep detector v1 vs v2 and the pretrain/fine-tune checkpoints.
- [ ] The baseline row stays. Every claim about an improvement must come from this table, not from a
  component's own validation metric.

---

## Self-Review

**The honest risk on Task 1-2:** pseudo-labeling may add little for the rare classes precisely because
the detector rarely finds them — a class with 0.43 recall generates few confident detections to harvest.
The logbook constraint mitigates false positives but cannot create true positives. Task 1's per-class
yield report exists to answer that before Task 2 spends a 300-epoch run.

**On Task 3:** pretraining is a refinement on top of the fine-tune's taxonomy alignment, not a substitute
for it. If compute is scarce, Plan 3's fine-tune comes first — it is what closes the measured gap.

---

## Task 2 status (2026-08-26): dataset builder + training job wired, not yet run

**Delivered:** `scripts/build_detector_v2_dataset.py`, `condor/train_detector_v2.sub`/`.sh`,
`tests/test_build_detector_v2_dataset.py` (32 tests, all pass on the login node — pure Python, no
torch). Nothing was submitted, built, or swapped; the shipped `best.pt` is untouched.

**The ratio decision.** 886 hand-labeled images against 601,261 pseudo-labeled ones is a 679:1
imbalance — concatenated verbatim, an epoch would be >99.8% teacher opinion. The builder instead:

- Oversamples every hand-labeled train image `--hand-oversample` (default 20) times, plus an EXTRA
  `--zero-gain-oversample-boost` (default 20, i.e. 40x total) for any hand-labeled image containing
  `bipolar dissector` or `suction irrigator` — the two classes Task 1 measured at literally zero
  pseudo-label gain (both are `OUT_OF_TAXONOMY`; the logbook can structurally never confirm either, so
  no amount of pseudo-labeling can help them — oversampling is the only lever available against their
  *relative* dilution as the rest of the corpus grows around them).
- Splits the pseudo pool into PROTECTED (contains a class whose yield multiplier ≤
  `--protect-below-multiplier`, default 100 — `tip-up fenestrated grasper` ~6.8x, `stapler` ~25x,
  `grasping retractor` ~66x) and COMMON (everything else — `needle driver` ~1884x down to `permanent
  cautery hook/spatula` ~344x). Protected images are kept in full, uncapped; common images are subject
  to a seeded uniform subsample capped at `--pseudo-common-cap` (default 20,000). A single subsample over
  the whole 601K pool would apply the same shrinkage to a class with 530 new instances as to one with
  484,021 — exactly how Task 1's rare-class wins could be erased by Task 2 instead of amplified.
- Net effect at the defaults: ~17,720 hand-labeled slots (886 × 20, plus the zero-gain boost) against an
  estimated ~26,000–35,000 pseudo slots (protected pool size unmeasured — see below — plus the 20,000
  common cap) — roughly 1:1.5 to 1:2 by volume, not 1:679. The two zero-gain classes retain their
  ORIGINAL 886-set proportion within the hand-labeled portion (which itself now makes up a much smaller
  share of the total corpus than in v1) — this is a partial mitigation, not a fix; Task 2's own per-class
  recall report against the untouched 240-image val split is what actually answers whether it worked.
- The 240-image hand-labeled val split is copied through verbatim and never touched by a pseudo-labeled
  path, so v2 is measured against human labels, not against the teacher's own opinions.

**R30, two independent layers** (both unit-tested): `verify_pseudo_harvest_excluded_heldout` trusts-but-
verifies the harvest's own `pseudo_label_report.json.shards_heldout_excluded` field is nonzero (it is —
14 of 235 shards, matching the 11-case heldout list exactly); `assert_no_heldout_pseudo_images`
independently re-derives the case id from every candidate pseudo image's own filename via
`normalize_case_id` and raises if any resolve into `config/splits_v2.json`'s heldout list. Unlike
`pseudo_label_shards.split_heldout_shards`, finding zero in the second layer is the *expected* outcome
(the pool is already clean) — finding any is treated as a hard failure, not a silent drop-and-continue,
because contamination on a pool whose own report claims to be clean means something upstream broke.

**Not executed.** This session did not run the builder against the real 601,261-image pool (only
tmp_path fixtures), did not extract the real `yolo_dataset.tar.gz`, and did not submit
`condor/train_detector_v2.sub` — no torch/GPU on the login node, and heavy work must run on a compute
node per the project's guardrail. The exact protected/common pseudo-image split sizes, the real
train.txt/val.txt line counts, and any wall-clock estimate for 300 v5m epochs over a corpus this size are
therefore unmeasured, not merely unreported.

**Before shipping anything:** re-run `scripts/flag_matrix.py` with v2's weights via `--fixed-arg
--yolo-weights=<v2 best.pt> --fixed-arg --yolo-repo=<yolov5 checkout>` and compare against the existing
baseline row — mAP is not the gate; whether `_variant_gate_answer`'s needle-driver detection condition
flips on the graded cases is.
