# SurgVU 2026 Category 2 -- Evidence-Arbitration Router

**Date:** 2026-08-04 (revised 2026-08-05 -- metric corrected to BERTScore-F1)
**Status:** Design approved, pending spec review
**Deadlines:** Final test phase Aug 21 – Sep 6 (2 submissions only). Report + public repo + presentation due Sep 13. Challenge day Sep 27, Strasbourg.

---

## Goal

Win Category 2 of the SurgVU 2026 challenge (surgical video question answering). Secondary goal: the routing and confidence machinery should be reusable outside the challenge, but no design decision here is made for that reason -- this system is deliberately independent of the OpScribe architecture.

---

## What winning actually requires

Prizes are **two-tiered**, and the tier is gated on performance:

| Placement | "Overall" tier | "2026" tier |
|---|---|---|
| 1st | $3,000 | $1,000 |
| 2nd | $2,000 | $500 |
| 3rd | $1,000 | $250 |

The gate, quoted from the prizes page: *"teams performing better than last year's methods will be eligible for 'overall' case prizes, otherwise the '2026' prizes."* A team cannot win both tiers. Separately, **$500 for best methodology report**, and *"all submissions will be eligible for the best methodology report prize, regardless of how their algorithms performed."*

So the objective is not "place well" -- it is **beat the 2025 methods re-scored under the 2026 metric**. That is a 3× difference in prize money at first place.

### The bar is computable in advance

The metric changed from BLEU to BERTScore this year, so last year's published 0.4215 tells us nothing. But the organizers require *"editor access to submission containers for the username `aneeqzia_isi`... to enable re-evaluation on future datasets"* -- they re-run prior winners' containers under the new scoring. And every prior Cat 2 winner's container is public:

| Year | Team | Repo |
|---|---|---|
| 2025 | Capybara (1st) | `huuquan1994/surgvu25-cat2-submission` |
| 2025 | UoM-SurgicalAI (2nd) | `mobarakol/UoM_SurgicalAI_SurgVU_Challenge2025` |
| 2025 | Medibot (3rd) | `bravefox12138/surgvu2025vqa` |
| 2024 | PDMYR (1st) | `SalenGit/surgvu2024-category2-rank1` |

**We can run Capybara's container against the official 2026 evaluation container ourselves and obtain the exact number we need to beat.** This turns an unknown into a measured target, and it doubles as the strongest possible baseline for ablations. It is the single highest-value early task after the metric experiments.

### Submission checklist

All four are required; incomplete submissions are ineligible for cash prizes.

1. Successful algorithm-container run on the final testing phase.
2. Editor access on the container granted to `aneeqzia_isi`.
3. Methodology report using the provided **LaTeX template**, emailed to `isi.challenges@intusurg.com`, listing official team members, with GitHub links, reproduction instructions, and any external datasets used.
4. Pre-recorded presentation, **maximum three minutes**, using the provided PowerPoint template.

### Submission budget

- **Preliminary phase: up to 10 attempts** on a small dataset, for debugging. This is the iteration surface and it is finite -- spend attempts deliberately.
- **Final phase: 2 attempts**, best score counts.

---

## The metric

The official evaluation container is public at
[`isi-challenges/surgvu26-category-2-eval-public`](https://github.com/isi-challenges/surgvu26-category-2-eval-public). We do not have to infer the metric or reimplement it -- **we can run the exact scorer locally**, which eliminates an entire class of risk.

**Primary: BERTScore-F1**, `roberta-large`, `rescale_with_baseline=True`, **max over the five references**, then meaned across cases.

```python
bert_scorer = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True)
cands_expanded = [cand] * len(refs)
_P, _R, F1 = bert_scorer.score(cands_expanded, refs)
bertscore_f1s.append(F1.max().item())
```

**Secondary, all reported in `metrics.json`:** NLI entailment (`cross-encoder/nli-deberta-v3-base`), NLI×BERT, BLEU-4, ROUGE-1/2/L. BLEU is smoothed (`SmoothingFunction().method1`) and is no longer the ranking metric.

### Three consequences of the metric that drive the design

**1. Terse answers are optimal where a terse reference exists.** Reference lists lead with the bare form -- `"Yes"`, `"No"`, `"Cadiere Forceps"`, `"Uterine horn"` -- and scoring takes the max over references. An identical string scores F1 = 1.0, and rescaling maps 1.0 to 1.0. So emitting exactly `"Yes"` on a yes/no question hits the ceiling. **No elaborate answer can beat a one-word exact match**, and a long answer risks scoring lower.

**2. Correctness now dominates phrasing.** Under n-gram overlap, a wrong-polarity answer that echoes the question's noun phrase still collects most of the credit. Under BERTScore against a terse reference, `"No"` scored against `"Yes"` has almost no lexical or semantic overlap to fall back on. The exact cost is unmeasured and is experiment #1, but the direction is not in doubt: **the perception stack and the router now carry the score.** This is good news for the design -- the routing work is worth more under this metric, not less.

**3. BERTScore and NLI see raw text; BLEU and ROUGE see normalized text.** In `evaluate.py`, BLEU and ROUGE are computed on lowercased, punctuation-stripped strings, but the candidate and references are passed to BERTScore and NLI **un-normalized**. Casing and terminal punctuation therefore affect the primary metric. The organizers' own test fixture probes exactly this -- `answer001c.json` is `"yes"` against a ground truth of `"Yes"`. We match reference casing exactly.

The NLI secondary metric is worth noting even though it does not rank: NLI genuinely captures contradiction, so a wrong-polarity answer is punished there in a way BERTScore may not punish it. Any hedging strategy that games BERTScore would show up as a collapsed NLI score in the challenge paper. We optimize for being right.

---

## What the task actually is

Analysis of the public sample set and the training label tables shows Category 2 is **metadata recovery and single-slot filling**, not open-ended reasoning over video.

**The container is invoked once per case** -- one video, one question, one answer. The time budget is therefore *per case*, not per test set, which is what makes a multi-pass routing system affordable.

```
/input/endoscopic-robotic-surgery-video.mp4
/input/visual-context-question.json    →  "Was a large needle driver used in this clip?"
/output/visual-context-response.json   →  "Yes"
```

### Provenance of the question evidence -- read this before trusting any of it

There is **no 2026 sample set**. The 2026 data-description page links `SURGVU25_cat_2_sample_set_public.zip`, described as *"10 video samples with questions and answers in the same format used for evaluation to be used as reference"* -- it is last year's set, last uploaded 2025-07-22, and the GCS bucket contains no 2026 object at all. It actually holds 11 cases, not 10.

So everything below about question *distribution* is extrapolated from **prior-year reference material**, in a year where the metric changed and the tool-label scope was restated. Where the 2026 site's own text conflicts with these samples, the site wins.

The only 2026-native question evidence that exists publicly is in the evaluation container. Its ground-truth fixtures are deliberately scrubbed (`case002` reads *"No, the specific tool was not used"*, with the tool name redacted), but `test_metrics.py` names three real questions:

- *"In this procedure, what is cauterized by the surgeon?"*
- *"Which surgical specialty is this procedure associated with?"*
- *"Are forceps involved in the procedure?"*

and its tool-question ground truth is `"Forceps"` -- class-level, no commercial name.

**Questions are single-slot.** Of the 11 prior-year sample questions, 7 are yes/no and the remaining 4 fill exactly one slot: tool type, organ, procedure type, and tool purpose. Three of the seven yes/no questions ask at commercial-name granularity. The 2026-native examples add two slot types those 11 do not cover: **surgical specialty**, and **anatomy-plus-action** (*what* is being cauterized, not merely whether cautery occurs).

**Ground truth was written from the label tables, not from pixels.** The tells are in the reference wording: *"No, forceps are not **mentioned**"*, *"Is a large needle driver among the **listed** tools?"*, *"What procedure is this **summary** describing?"*. The corresponding tables are:

- `tasks.csv` -- `start_time, stop_time, groundtruth_taskname, matched_description`
- `tools.csv` -- `install_case_time, uninstall_case_time, arm, commercial_toolname, groundtruth_toolname`

For a clip at time *t*: the "listed tools" are every `tools.csv` row whose install interval covers *t*; the "summary" is the `matched_description` of the covering task segment.

### Three exploitable consequences

**1. `matched_description` is a closed corpus of 21 strings.** Across all 155 cases there are exactly 21 unique values, and most task classes map to one or two:

| Task | Unique descriptions |
|---|---|
| Uterine horn | 1 |
| Suspensory ligaments | 1 |
| Range of motion | 1 |
| Rectal artery/vein | 2 |
| Retraction and collision avoidance | 2 |
| Suturing | 12 |

The 2025 winner read this as a defect -- their report calls 21 captions "insufficient for training" and abandoned fine-tuning on that basis. It is not a training set; it is a **retrieval corpus**. Classify the task, retrieve the string, and you hold verbatim the text the ground-truth answers were generated from. Twenty-one strings fit trivially in a YAML file inside the container, which also satisfies the no-network constraint.

**2. Exactly three instruments are installed at any time, out of 12 test-relevant classes.** The data description states that for the extent of each clip *"there are three robotic surgical tools installed and within the surgical field"* and that each clip *"can contain up to three of 12 possible tools."* This reconciles exactly with the arm structure in `tools.csv`: four USM arms, one of which carries the endoscope (`nan(camera in)`), leaving three instrument arms.

This is a hard structural constraint and it should be built into the model, not learned. The tool recogniser is therefore a **constrained top-3 selection over 12 classes**, not 12 or 20 independent binary decisions. It collapses the hypothesis space from 2¹² to C(12,3) = 220 and makes the "how many did we predict" failure mode impossible by construction.

**Class scope is 12, not 20.** The label tables contain rarer classes -- `suction irrigator`, `synchroseal`, `curved scissors`, `potts scissors`, `tenaculum forceps`, `bipolar dissector`, `crocodile grasper` -- but the data description is explicit: *"these tools will not be part of the testing set and the submissions will not need to recognize those."* They are usable as auxiliary training signal and nothing more.

**3. Installed ≠ visible.** Confirmed directly by the data description: tools *"may be obscured or otherwise temporarily not visible"* even though installed, and ground truth keys off install intervals. The estimation target is **installation state** -- temporally smooth, strongly predicted by task context, and materially easier than detection. Getting better at *seeing* tools can move you away from ground truth.

---

## Architecture

Seven components. Each is independently testable and communicates through plain data structures.

Two streams converge: the video stream recovers the metadata, the question stream determines which slot needs filling and at what granularity. They meet at the router, which decides whether the first VLM pass can be trusted or whether more evidence is worth buying.

### 1. Knowledge Bundle (YAML)

Declarative, MONAI-bundle style. Shipped inside the image; no network at runtime.

- `taxonomy.yaml` -- the 12 test-relevant `groundtruth_toolname` classes, the 7 out-of-scope rare classes marked as train-only, and the 43 commercial names mapped to their class with empirical priors (used only as a hedge, see below)
- `descriptions.yaml` -- the 21 `matched_description` strings, keyed by task class
- `answer_forms.yaml` -- per slot type, the answer surface form (see component 7)
- `routing.yaml` -- confidence thresholds, per-tool budgets, frame counts

Everything tunable lives here, so a sweep is a config sweep rather than a code change. This is also the artifact that makes the method legible in the report.

### 2. Question Parser

Text only, deterministic, no model. Produces:

- **slot type** -- `tool_presence` | `tool_type` | `organ` | `action` | `anatomy_action` | `procedure` | `specialty` | `world_knowledge`

  The last four are each attested by a 2026-native example: `procedure` and `specialty` from *"Which surgical specialty is this procedure associated with?"*, `anatomy_action` from *"In this procedure, what is cauterized by the surgeon?"*, `world_knowledge` from the prior-year *"What is the purpose of using forceps?"*. `specialty` is answerable from the task class alone via YAML and needs no perception beyond it.
- **queried entity**, normalized against `taxonomy.yaml`
- **granularity** -- class-level or commercial-level

`world_knowledge` questions (e.g. *"What is the purpose of using forceps?"*) need no perception and route straight to the renderer.

### 2b. The UI overlay -- prohibited, and why that matters

Every *training* frame carries a burned-in UI bar listing the instrument on each of the four arms by commercial name, plus the endoscope angle. It is legible in all 11 Cat 2 sample clips and it matches ground truth exactly.

**It is out of bounds.** The challenge states plainly: *"using the information available in the UI to make predictions is not allowed. To enforce this, the UI will be blurred from the test set eliminating this information."* The organizers publish an example blurred test frame alongside that statement.

So this is a rule, not merely a robustness concern, and three consequences follow:

1. **Blurring during training is mandatory.** A model trained on unblurred frames uses UI information to make predictions, whether or not we intended it to. The 51-pixel Gaussian over the bottom 45 px is compliance, not a hyperparameter -- which is also why the 2025 winner blurred it before inference.
2. **Match the organizers' blur exactly.** Their example test frame defines the region and strength. Our preprocessing should reproduce it rather than approximate it, or the CNN sees a different distribution at test than it trained on.
3. **Commercial-name granularity really is unanswerable from pixels.** The one place that information appeared is now removed by construction. That closes the question raised in Experiment 2 in the negative and confirms the class-fallback branch: `P(commercial | class, task)` priors are the only available recourse, and probably the question is class-level anyway.

There is no overlay-reading component in this design. Recorded here so it is not rediscovered and mistaken for an opportunity.

### 3. Tool Install-State Recogniser

A fine-tuned image CNN predicting *installed* rather than *visible*, with temporal smoothing across the clip. It is the **only** source of tool identity available at test time, since the UI is blurred out -- so its accuracy sets the ceiling on every tool-presence question, which is roughly two-thirds of the sample set. It trains and infers on blurred frames throughout. Trained on SurgVU24 video against install intervals derived from `tools.csv`; optionally pre-trained on SurgToolLoc-2022, which the rules explicitly permit ("free to use any public dataset (including previous challenges)"). The seven out-of-scope rare classes are retained as auxiliary training signal but never emitted.

**Output formulation -- revised twice by measurement. Read the second revision; the first was measuring the wrong thing.**

*First measurement (2026-08-08).* Sampling timestamps inside task segments showed 78% of moments with exactly three instruments -- which looked like a strong prior but not a hard constraint.

*Second measurement (2026-08-08, corrective).* That 78% counts **occupied arm slots**, not tool classes. The model predicts a 12-way class vector, and **two arms frequently carry the same class** -- a Large Needle Driver on USM1 and a Large SutureCut Needle Driver on USM3 are both `needle driver`, and collapse to one label. That collapse is not rare: `needle driver` occupies more than one arm in **10,583 of 21,522** sampled moments, nearly half.

| Basis | Peak | Share at 3 | Share at 2 |
|---|---|---|---|
| Occupied arm slots | 3 | 78.2% | 6.9% |
| Distinct classes, sampled per task segment | 2 | 33.4% | 51.9% |
| **Distinct classes, per 30-second window** *(the training and test unit)* | **3** | **44.6%** | **38.6%** |

The third row is the operative one: windows are what the model consumes and what the evaluation serves, so the distribution must be duration-weighted over windows rather than sampled per segment.

**So the challenge's "three instruments are installed" is a fact about arms, and says almost nothing about the label the model emits.** Hard-constraining the output to three classes would be wrong not one time in five, but closer to one time in two.

**Decision (unchanged in direction, strengthened in justification): 12 independent sigmoid outputs, threshold tuned on validation, with the *class-based, window-weighted* distribution above as a soft prior.** Top-3 is the tie-break only when the thresholded set is ambiguous.

There is exploitable structure here for later: given the arm count is almost always three, an inferred class count of two implies a duplicated class, and `needle driver` accounts for the overwhelming majority of duplicates. That is a usable prior for disambiguating borderline predictions, not something to hard-code.

A per-arm formulation -- three heads, each a 12-way softmax, using the `arm` column in `tools.csv` -- remains attractive on paper but is now doubly disfavoured: the heads are not identifiable (nothing in the image says which arm it is), *and* the exactly-three premise it depends on does not hold. Ablation only.

**Backbone.** EfficientNetV2-S by default -- the 2025 winner reached 97% macro-F1 with it on this exact data, which is a strong prior. ResNet or VGG are equally reasonable starting points; this is an empirical choice, not a principled one, and the ablation should include at least one alternative.

### 4b. Action Recogniser (temporal CNN)

A small temporal CNN answering action questions directly -- *"Is tissue being cut during this clip?"* -- from densely sampled frames.

This exists because action recognition is a purpose-built ConvNet task, not a VLM task. Routing *"is tissue being cut"* through a 7B model is slower, costs a pass we may not be able to afford, and is probably less accurate than a model trained on exactly that question. Training labels come from the task segments in `tasks.csv` combined with tool presence, since cutting co-occurs with monopolar curved scissors and specific task classes.

Unlike the detector, this expert **produces the answer rather than evidence**, so its branch emits directly with no second VLM pass.

### 4. Task Classifier and Description Retrieval

Classifies the clip into the 8 task classes (after case-normalizing the 15 raw label spellings), then retrieves the corresponding `matched_description`. **Organ is derived from the retrieved description via YAML, not from a separate classifier** -- the description already names the anatomy.

This replaces the 2025 winner's `max_new_tokens=2048` description-generation pass with a lookup, which is both faster and closer to ground truth.

### 5. VLM Pass 1 with self-consistency

The VLM answers given the retrieved description, the tool list, and frames -- **sampled N times over independent frame subsets**. Confidence is the **agreement rate across those N samples**, not a logprob and not a verbalized score.

The budget comes from an inefficiency in the 2025 winner: they allocate `max_new_tokens=2048` to an answer that is one short sentence. Under this metric the target answer is often a single word, so capping generation at ~16 tokens funds N=3–5 samples at no net cost.

**Precision: 4-bit NF4, fp16 compute dtype. Decided, not swept.**

The choice is close to forced by the 16 GB T4:

| Precision | 7B weights | Verdict |
|---|---|---|
| fp16 | ~14 GB | Leaves ~1 GB for KV cache, video visual tokens, and the CNNs. Near-certain OOM on multi-frame input. |
| int8 | ~7 GB | Fits, but `bitsandbytes` int8 is markedly slower than both fp16 and NF4 because of its mixed-precision decomposition -- and we need N sequential passes. |
| **4-bit NF4** | **~4 GB** | ~11 GB of headroom for visual tokens and the co-resident CNNs, and fast. It is also the exact configuration the 2025 winner ran on this platform. |

Self-consistency sampling is what makes this decision load-bearing: N passes are only affordable in the cheapest regime, so precision and the confidence budget are the same decision.

The known hazard is that **4-bit output is not identical across GPU architectures** -- the 2025 winner reports Turing, Ampere, and Hopper diverging on identical inputs. That is not an argument for a different precision; it is an argument for validating on T4-class hardware, which the integration-test requirement already covers. We develop on CHTC's Ampere cards and must not trust a 4-bit number that has not been reproduced on Turing.

### 6. Router / Arbiter

Plain Python and a threshold file -- no model. It fires when the VLM's slot value disagrees with the perception bank, or when self-consistency is low.

The governing principle is narrow: **escalate only to something that can actually reduce the uncertainty you have.** If no evidence you could buy would change the answer, buy none.

| Situation | Branch | Second pass? | Why |
|---|---|---|---|
| Agreement, high consistency | Emit | no | Nothing worth buying |
| Tool question, **class-level**, disagreement or flip-flopping | Detector arbiter | **yes** | Occlusion is the classifier's known blind spot, and vision can fix it |
| Tool question, **commercial-level** | Class fallback | no | No pixel separates a *Large* from a *Mega* needle driver |
| Organ, anatomy-plus-action | Dense re-sample | **yes** | Anatomy needs more frames, not boxes |
| Action | Action recogniser | no | A purpose-built temporal CNN answers this directly and better than a 7B model would |
| Procedure, specialty | Task class wins | no | These derive from the exercise label, not from pixels. If the VLM disagrees here, the VLM is wrong |
| World knowledge | Never escalates | no | The video was never involved |
| Parser recognizes nothing | Generic VLM path | -- | Degrade, don't fail |

**The second pass runs only when an expert produced *evidence* rather than an *answer*.** The detector and dense re-sampling hand the VLM something new to look at, so re-asking is worth the latency. The class fallback, the task-class-wins branch, and the action recogniser each produce the slot value themselves -- re-asking there would add cost and give the model a chance to talk itself out of an answer that is already correct. Those three emit directly.

### Framing: this is a hard-routed mixture of experts

The router is a gate and the branches are experts, which makes the mixture-of-experts framing a natural way to describe the system. Two caveats worth stating precisely, because a reviewer will raise both:

- Classic MoE means a **learned** gating network over expert *subnetworks* trained jointly, usually with soft or top-k routing over token representations. Ours is a hand-written rule gate over heterogeneous, separately-trained experts -- CNNs, a detector, a VLM, and lookup tables. The honest description is a **hard-routed heterogeneous expert ensemble**, or in the tool-use literature, learned tool routing.
- Taking the framing seriously raises the obvious question: **if it is a gate, why is it hand-written?** It need not be. A small classifier over (question type, perception outputs, agreement rate) → branch, supervised by the synthetic validation set, is a strictly more general router than an `if/else` chain.

Plan: ship the rule-based gate first, since it is transparent, debuggable, and needs no training data. Then train a learned gate and compare on synthetic validation. If the learned gate wins, it becomes the headline contribution; if it does not, the comparison is still a result worth reporting.

On the commercial-level branch specifically: the organizers state test labels use `groundtruth_toolname` only, so the correct answer to *"is a **large** needle driver installed"* is most likely determined by whether a `needle driver` is installed at all. `P(commercial | class, task)` priors are computed and available, but the router consults them only if Experiment 2 shows they help.

The detector never contributes location to the answer text. It contributes *evidence* that a specific tool is present.

### 7. Answer-Form Selector

Chooses the surface form from `answer_forms.yaml` by slot type, then emits. Under BERTScore the rule is short and empirical rather than template-heavy:

| Slot type | Form | Rationale |
|---|---|---|
| `tool_presence` | bare `Yes` / `No` | exact match with the terse reference → F1 = 1.0 |
| `tool_type`, `organ` | bare entity, reference casing | same |
| `procedure` | short noun phrase | terse reference usually present |
| `world_knowledge` | one sentence | no terse reference exists for these |

Casing and terminal punctuation are matched to the observed reference style, because BERTScore sees raw text. The form choice is deterministic and sweepable -- we measure each option against the official scorer rather than arguing about it.

---

## Validation

Two pillars.

**The official evaluation container, run locally.** It is public, so our local number is the real number rather than an approximation. This replaces what would otherwise have been a metric reimplementation plus a calibration exercise.

**A synthetic validation set.** The 2025 winner tuned on 11 sample videos and had to hand-correct wrong answers in them; 11 cases cannot support fitting a confidence threshold. Because ground truth is generated from metadata we already hold, we **replicate the generator**: for a sampled clip, read the covering `tools.csv` rows and `tasks.csv` segment, and emit a question plus five paraphrased references in the observed style -- terse first, then paraphrases. This produces thousands of scored cases from the same inputs the organizers used.

The 11 public samples are held out from all tuning as a golden set, to detect drift between the synthetic distribution and the real one.

### Experiment 0 -- establish the bar

Run Capybara's public 2025 container against the official 2026 evaluation container on the 11-case sample set. This produces the BERTScore number we must beat for the "overall" prize tier, and gives us the strongest available baseline for every ablation. Repeat for the other three public prior-winner containers if time allows.

### Experiment 2 -- does commercial-name granularity matter at all?

Three of the eleven sample questions name a *"large needle driver"*, which is a `commercial_toolname`. But the data description states plainly that *"the test labels will only be based on groundtruth_toolname column"*, and that statement is not scoped to Category 1. These cannot both be fully true of the 2026 test set.

Resolve empirically on the 11 samples: for each commercial-name question, score (a) answering from the class alone against (b) answering from `P(commercial | class, task)` priors. If (a) matches ground truth as often as (b), the priors are dead weight and the design drops them -- which makes the system simpler and removes a whole component. This is a genuine question, and it should be settled with a measurement rather than an argument.

### Experiment 1, before anything else is built

Measure the metric's actual shape against the official scorer:

- What does a wrong-polarity answer cost? (`"No"` against a `"Yes"` reference set)
- What does `"yes"` cost versus `"Yes"`? (the organizers' own fixture asks this)
- Does a full sentence ever beat the bare form when both are correct?
- How much does a *correct* answer in the wrong register lose?

Every downstream decision about answer form depends on these four numbers, and all four are cheap to obtain.

---

## Error handling

Every failure degrades to a plausible answer rather than an exception. An empty output scores at the floor; a well-formed guess does not.

| Failure | Response |
|---|---|
| Time budget exceeded | Emit the pass-1 answer; skip arbitration |
| OOM | Reduce frame count, retry once, then perception + form selector with no VLM |
| Question parser finds no known slot type | Generic VLM path with retrieved description |
| Perception bank and VLM both low-confidence | Answer from priors alone via the form selector |
| Any unhandled exception | Highest-prior answer for the parsed slot type |

---

## Testing

Test-first, per the project's standing practice.

- **Unit:** question parser (every sample question plus adversarial phrasings), form selector, prior computation from `tools.csv`.
- **Golden:** the 11 public samples, end to end, scored by the official container.
- **Integration:** full container on a T4-class GPU. Not optional -- the 2025 winner reports that **4-bit quantized VLMs produce different outputs on Turing vs Ampere vs Hopper**, so validating on an A100 and submitting to a T4 is untested code.
- **Ablation, on synthetic validation:** each component removed in turn, to establish that the router earns its runtime.

---

## Key decisions

- **4-bit NF4, picked rather than swept.** fp16 does not fit alongside video tokens and the CNNs; int8 fits but is slow in `bitsandbytes`; NF4 leaves headroom and is what won on this platform last year. Validated on T4-class hardware rather than assumed.
- **Purpose-built ConvNets wherever the output space is fixed.** Tools, task, and actions are all closed-vocabulary problems where we control the output and have labels -- so they are CNNs, not VLM prompts. The VLM is reserved for composing an answer to an arbitrary question, which is the only thing it is uniquely good at.
- **Rule-based gate first, learned gate second.** The routing policy ships as transparent `if/else`, then a trained gate is evaluated against it on synthetic validation. Either outcome is reportable.
- **The second VLM pass is conditional on new visual evidence, not on escalation.** Branches that resolve from a table emit directly. Re-asking the model about information it already had costs latency and risks it reasoning itself away from a correct answer.
- **Approach A (evidence arbitration) over a literal confidence cascade or a full Bayesian slot filler.** The cascade has no principled confidence signal; the Bayesian version needs calibration data for every evidence source and degrades badly if the reliabilities are wrong. A is structured so the cascade is its degenerate fallback and the Bayesian criterion is a one-function substitution in the router.
- **Detector as evidence, not localization.** No sample question asks *where*.
- **Retrieve the description; do not generate it.** 21 strings, closed set.
- **Answer tersely by default.** An exact match with the terse reference is the ceiling; elaboration can only cost.
- **Optimize for correctness, not for gaming the metric.** NLI entailment is reported alongside BERTScore and would expose any hedging strategy.
- **Target installation state, not visibility.** That is what the ground truth encodes.
- **No separate organ classifier.** Redundant with description retrieval.
- **Independent of OpScribe.** No shared code, no 72B model, no fine-tuned adapters. The T4's 16 GB settles this regardless.

---

## Risks

**The question-distribution evidence is a year stale, and it is the largest risk in the design by a wide margin.** There is no 2026 sample set; the 11 cases we tune against are 2025 reference material, in a year where the metric changed from BLEU to BERTScore and the tool-label scope was restated as class-level. The three 2026-native questions visible in the evaluation container already reveal two slot types the 11 do not contain. It is entirely possible the 2026 question generator was rewritten.

Mitigations: hold the 11 out of all tuning; keep the question parser's slot taxonomy open rather than closed, with a generic fallback path that degrades gracefully on an unrecognized slot type; treat any large gap between synthetic and golden scores as stop-and-reassess rather than something to tune away; and spend one preliminary submission early, deliberately, to sample the real distribution before committing the architecture's tunables.

**Runtime limit per phase is unpublished.** It bounds how many passes the router can afford. Requested from the organizers; until known, `routing.yaml` carries a conservative budget and the design degrades gracefully under a tighter one.

~~**2026 labels may differ from the SurgVU25 labels in the bucket.**~~ **Closed 2026-08-08.** The challenge's participant download page lists exactly the bucket objects we already hold -- `surgvu24_videos_only.zip`, `surgvu24_labels_updated_v2.zip`, `SURGVU25_cat2_train_labels.zip`, and the SurgVU25 Cat 2 QA samples -- and states the corpus is the published SurgVU dataset ([arXiv 2501.09209](https://arxiv.org/pdf/2501.09209)) plus the Cat 2 labels. Our labels are current. It also confirms there is no 2026 sample set, so the question-distribution risk above stands unchanged.

Worth noting from the same page: `cat1_test_set_public.zip` is listed as the **Cat 1 *validation* set**, not a test set. It carries clevis bounding boxes and is therefore legitimate in-domain box data should the detector ever need fine-tuning.

**Quantization nondeterminism across GPU architectures.** Mitigated by the integration-test requirement above.

**Blur mismatch between our preprocessing and the organizers'.** The UI is blurred out of the test set by the organizers, and we blur it ourselves during training. If the two blurs differ in region or strength, the CNN trains on one distribution and infers on another -- a silent accuracy loss on the component that carries two-thirds of the questions. Mitigation: reproduce the organizers' published example blurred frame rather than approximating it, and verify on the Cat 1 validation set, which is drawn from blurred test material.

**Registration closes Aug 15.** Team must be created on the platform and the signed agreement form sent to `isi.challenges@intusurg.com`. Acceptance was confirmed 2026-06-11, but the platform-side team creation should be verified before the deadline rather than assumed.

**Ten preliminary attempts, then two final ones.** Both budgets are finite and the final leaderboard is not an iteration surface. Every submission must be justified by a local result first.

---

## Data provenance -- a hard boundary

The rule, verbatim: *"Your team is free to use any public dataset (including previous challenges) for pre-training models. If your team has private data that you are pre-training models on, then that data will need to be disclosed and publicly released upon submission to the challenge."*

**Permitted, no disclosure burden:** SurgToolLoc 2022 (prior challenge), GraSP, Cholec80/CholecT50, EndoVis instrument sets, any off-the-shelf pretrained backbone (EfficientNetV2, ResNet, GroundingDINO, OWLv2), and the base VLM.

**Categorically excluded:** anything touched by UW Health or Platform R data. The disclosure clause is not a paperwork step -- it requires *public release*, and PHI can never be publicly released. Any weight that has seen that data is therefore permanently ineligible, with no remediation path.

This is why the separation from OpScribe is structural rather than stylistic. It is not enough to avoid copying code: **no checkpoint, adapter, or derived artifact whose training history includes private clinical data may enter this repository.** The 72B endoscopy model, its adapters, and anything fine-tuned on UW cases are all out, regardless of how well they might perform.

Every model in this system is either trained from scratch on SurgVU/SurgToolLoc, or an unmodified public checkpoint. That property must hold at submission and be stateable in one sentence in the methodology report.

---

## Out of scope

- Category 1 (tool localization) -- a detector is trained here only as an internal arbiter, and beating the Cat 1 prize bar is not a goal.
- Any use of challenge test data for training or tuning.
- Fine-tuning the VLM. The 2025 winner tried it, it failed for a structural reason (21 captions), and our design routes around the need.
