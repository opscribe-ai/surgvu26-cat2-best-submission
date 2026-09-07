# v5 Plan 3 — W6a: QA Generation and VLM Fine-Tuning


**Goal:** Fine-tune `Qwen/Qwen2.5-VL-7B-Instruct` on QA pairs generated from the SurgVU logbook so the VLM can answer questions the 11-intent router has no form for — which is the failure the whole v5 design was commissioned to fix.

**Architecture:** The logbook (`tools.csv`, `tasks.csv`) already encodes, per case and per time interval, which tools are installed and which task is underway. That is a free, large, exactly-on-taxonomy supervision signal. Generate QA pairs from it in the answer forms the router emits, pair them with frames decoded through the real serving preprocessing, and fine-tune. No human annotation anywhere.

**Tech Stack:** Python 3, PyTorch, `Qwen/Qwen2.5-VL-7B-Instruct`, PEFT/LoRA, bitsandbytes 4-bit NF4, CHTC GPU jobs.

**Spec:** `docs/design/specs/2026-08-24-v5-parallel-build-design.md` (W6a)

## Why this is the highest-leverage item, stated plainly

Every stock VLM measured on this task scores BELOW a baseline that never looks at the video:

    CNN + router (shipped)     0.8766
    blind router, no video     0.8294
    zero-perception baseline   0.6959
    4B-fp16 open               0.5743
    8B-NF4 closed              0.5501
    4B-fp16 closed             0.5216
    8B-NF4 open                0.4923

Those numbers are NOT an argument against using a VLM. They are the argument FOR this plan. The cause is
structural: gold answers reconstruct OUR label taxonomy — 12 tools, 8 tasks, commercial-name families —
which the CNNs are trained on and a general VLM has never seen. Fine-tuning on that taxonomy is the only
mechanism that closes it. Citing the untuned score as a reason not to tune is circular.

And the router's ceiling is structural too, demonstrated directly:

    "How much blood loss has occurred?"   -> count_open    -> "One"
    "Is the patient stable?"              -> unknown_polar -> "Yes"
    "What complication is developing?"    -> unknown_open  -> "The procedure involves surgical instruments."

Eleven regex intents always match something. Every match has a hardcoded form. No amount of perception
work fixes a question the router has no form for.

## Global Constraints

- **Split discipline is absolute.** Every generated example must exclude `config/splits_v2.json`'s
  `heldout` list — the 11 graded cases. Compare with `surgvu.sampling.normalize_case_id`, never string
  equality: sample `caseNNN` and corpus `case_NNN` are the same case but never equal as strings. **Fail
  loudly if the exclusion removes zero cases** — a silent zero-exclusion IS the bug (ruling R30, which
  cost a full contaminated training run).
- **Read `docs/design/notes/2026-08-25-label-vocab-hazards.md` before writing the generator.** The
  raw logbook is not clean: `groundtruth_toolname` contains `nan(camera in)` 1277 times, 144 empty
  strings, and a malformed ` Single Site"` row — roughly 19% of that column is not a tool. `clip applier `
  carries a trailing space. Every task name appears in BOTH cases (`Suturing` 775 / `suturing` 609).
  WHITELIST against `surgvu.taxonomy`, never blacklist.
- **Every generator must report its drop counts.** A generator silently yielding 80% of rows is
  indistinguishable from one yielding 100% until the model is wrong and nobody knows which stage lost the
  data.
- **Every frame passes through `preprocess.prepare_frame`.** The bottom UI band stays blurred — a
  challenge rule, and it applies to training-data generation exactly as it does to serving. There must be
  no path around it.
- **Serving reality:** Grand Challenge gives 10 min/case, 32 GB DRAM, and either a single T4 (16 GiB,
  sm_75 — no bf16, no FlashAttention-2) or NO GPU. 7B at fp16 is ~15 GB against 16 GiB, so the shipped
  path is 4-bit NF4. 4-bit output is not bit-identical across architectures; anything tuned elsewhere is
  re-validated on T4-class hardware before it ships.
- Never build containers or run heavy work on the login node.
- Do not modify `src/surgvu/{detect,variant,perceive,motion,flow,router}.py` — settled.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/build_qa_pairs.py` *(new)* | Logbook -> QA pairs. Torch-free, testable on the login node. |
| `src/surgvu/qa_forms.py` *(new)* | The question templates and answer forms. One place, so generation and evaluation cannot drift. |
| `scripts/train_vlm.py` *(new)* | LoRA fine-tune of Qwen2.5-VL-7B on the generated pairs. |
| `condor/build_qa_pairs.sub` / `.sh` *(new)* | Frame extraction job (needs the container for decode). |
| `condor/train_vlm.sub` / `.sh` *(new)* | The training job. |

---

### Task 1: QA forms — the shared vocabulary

Generation and evaluation must use ONE definition of what a question looks like and what a correct answer
looks like. Two copies drift, and the drift is invisible until the model is scored.

**Files:** Create `src/surgvu/qa_forms.py`; Test `tests/test_qa_forms.py`

**Interfaces:**
- Produces `QA_TEMPLATES`: a tuple of `{intent, question_template, answer_fn}` covering at minimum the
  router's existing intents (tool presence, tool identity, task, organ, purpose, procedure, count,
  cutting, suture) PLUS open-ended forms the router has no intent for.
- Produces `render(template, **slots) -> (question, answer)`.

- [ ] **Step 1: Write the failing test** — assert every template renders both a question and a non-empty
  answer; assert answers for taxonomy-valued slots are drawn from `surgvu.taxonomy` and not free text;
  assert no template can render an answer containing a raw logbook artefact (`nan(`, a trailing space, an
  unpaired quote).
- [ ] **Step 2: Run it, confirm it fails** (`ModuleNotFoundError`).
- [ ] **Step 3: Implement.** Templates phrased the way the graded questions are phrased — look at the 11
  sample questions for register, not for content. Include the "large/mega" family forms, since the variant
  head already resolves those and the VLM should learn the same distinction.
- [ ] **Step 4: `python3 -m pytest tests/test_qa_forms.py -q`** — must pass on the login node.
- [ ] **Step 5: Commit.**

---

### Task 2: QA-pair generation from the logbook

**Files:** Create `scripts/build_qa_pairs.py`; Test `tests/test_build_qa_pairs.py`

**Interfaces:**
- Consumes `tools.csv`, `tasks.csv`, `config/splits_v2.json`, `src/surgvu/qa_forms.py`.
- Produces `qa_pairs.jsonl`: one record per example, `{case, part, t_start, t_stop, question, answer, intent, provenance}`.

- [ ] **Step 1: Write the failing test.** Cover: heldout cases excluded (via `normalize_case_id`, and a
  test that a STRING-equality implementation fails); zero-exclusion raises; `nan(camera in)` and the empty
  string dropped and TALLIED; `clip applier ` stripped before whitelisting; task names case-normalised so
  `Suturing` and `suturing` collapse; every emitted answer drawn from the taxonomy.
- [ ] **Step 2: Run it, confirm it fails.**
- [ ] **Step 3: Implement.** Reuse `surgvu.labels.parse_hms` and `normalize_part` — `install_case_time` is
  `HH:MM:SS.ffffff`, NOT a float (ruling R14). Emit the video PART on every record (ruling R28) —
  timestamps reset at part boundaries and 126 of 155 cases have more than one part.
- [ ] **Step 4: Run it for real** over all 155 cases and paste the full output including drop counts and
  the excluded-case count. Sanity-check the totals against the label-vocab note before trusting them.
- [ ] **Step 5: Commit** the script, its tests, and a committed sample of the output (not the full file).

---

### Task 3: Frame extraction for the QA corpus

**Files:** Create `condor/build_qa_frames.sub` / `.sh`; extend `scripts/build_qa_pairs.py` with a
`--extract-frames` mode.

- [ ] Decode frames for each QA record via `surgvu.perceive.decode_clip` (which applies `prepare_frame`),
  writing to `/staging/n/nkalthoff/surgvu26/qa_frames/`. Model the job on `condor/dump_motion_v2.sub`, which
  already handles source video, the staging mount, and `request_disk` sized for the 2.97GB image.
- [ ] Report how many records lost their frames and why. A silent drop here is a silent change to the
  training distribution.
- [ ] Commit; the controller submits.

---

### Task 4: LoRA fine-tune of Qwen2.5-VL-7B

**Files:** Create `scripts/train_vlm.py`, `condor/train_vlm.sub` / `.sh`; Test `tests/test_train_vlm.py`
(torch-free parts: the split filter, the prompt assembly, the collator's shape contract).

- [ ] **Split by CASE, never by example.** Examples from one interval are near-duplicates; an
  example-level split reports memorisation as accuracy. This is ruling R28's lesson, and the variant head
  already paid for it once.
- [ ] Prompt assembly must include the **evidence packet** — the CNN tool/task probabilities, the YOLO
  detections with their timestamps, the variant block, the motion vector — rendered as text alongside the
  frames. That is the "Evidence VLM" of the spec: it reads our own perception output rather than guessing
  from pixels alone, and it is what distinguishes this from the stock-model measurements above.
- [ ] 4-bit NF4 base + LoRA adapters. Report trainable parameter count.
- [ ] Evaluate on held-out CASES with the real metric (`surgvu.scoring.Scorer`, BERTScore-F1), not on
  token accuracy. Token accuracy is not the objective and will flatter the model.
- [ ] Commit; the controller submits the training job.

---

## Self-Review

**Spec coverage:** W6a's "fine-tune on SurgVU" is Tasks 1-4. W6a's "pretrain on public endoscopic corpora"
(cholect50, DSAD, endovis18_vqa) is NOT in this plan — deliberately deferred as a second phase, because
the fine-tune is what closes the taxonomy gap and the pretrain is a refinement on top of it. Recorded here
so the omission is visible rather than forgotten.

**The thing most likely to go wrong:** the generator silently producing a corpus that does not match the
graded question distribution. Task 1 exists to make the forms explicit and shared; Task 2's drop-count
reporting exists to make losses visible. Neither guarantees the distribution is right — only a scored
run does.

**What this plan does NOT do:** it does not wire the VLM into serving. That is Plan 2 (W4 + W5), which
must also get `--vlm` into the container ENTRYPOINT, where it has never been.
