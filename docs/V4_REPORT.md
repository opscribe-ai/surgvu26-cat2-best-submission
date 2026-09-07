# v4 — the temporal programme, 2026-08-13 into 08-14

## READ THIS FIRST — current understanding as of 06:45

This file is append-only and contains FOUR reversals of my own claims. Sections
below are in the order things were learned, not the order they should be read.

### What ships, and what does not

**Ships: four router changes**, all measured, all with the 11 public sample
answers verified byte-identical before and after. The unanswerable-question
guard, generic instrument presence, scene-level questions, and plural answers
(+0.4964 expected on plural questions, with the break-even on its one
assumption computed both ways). Router closes at 98.1% intent, 45/45 held out,
289 unit tests.

**Does not ship: anything temporal.** On the TASK head, +0.0030 against its own
base -- the +0.0199 it shows against the shipped head is +0.0169 base
difference and +0.0030 training. On TOOLS, three arms from two different bases
converge on the same 0.75-0.77 from opposite directions: the strong base falls,
the weak one climbs. That is an attractor, not a gain, and it retires the
"+0.0229 improvement" recorded at 05:45 as an artefact of starting below it.
No blend weight helps at any arm, on either head.

**Does not ship: the checkpoint swap.** `task_resnet50_x40` scores +0.0180
description on validation and FAILS the 11-case sample -- six of eleven
task_top predictions change and one answer changes for the worse
("Uterine horn" -> "Gallbladder"). Almost certainly overfitting the split that
selected it. This was written down at 04:45 as evidence to confirm rather than
a change to make, and 45 minutes later the confirmation came back negative.

### The five action items

All delivered. Multi-burst extraction (235 shards, verified); both conversions
(TSM and I3D, 7/7 invariants); r2plus1d trained properly (burst-aware sampling
-- the old trainer would have spliced 7.5-second jump cuts into clips it called
continuous); multiple clips per window (closing a 30x gradient deficit that was
a loader bug, not a property of 3D convolution); and the Kinetics ablation --
**0.7053 against 0.4533 from scratch, +0.2520 for Kinetics**. YouTube action
pretraining transfers substantially to surgery. It just does not transfer
enough to overcome 18 layers at 112px against a fine-tuned ResNet-50 at 384.

### The two findings worth carrying forward

1. **The evaluation pipeline is validated** to +0.0000 against both shipped
   references. It was not, for the first six hours: every number before 04:10
   carried a wrong base checkpoint (0.0337) and a logits-vs-probabilities
   aggregation mismatch (0.0145) -- together larger than any effect being
   measured.

2. **23 serving-path tests had been dead since 2026-08-12** and nobody could
   have known, because the suite could not run anywhere: no torch on the login
   node, no pytest in the container. Fixed; whole suite now 835 passed, 5
   environmental, 0 real.

### What I would do differently

Reproduce the reference number through the new pipeline BEFORE running a single
arm. Seven bugs tonight produced plausible numbers for the wrong thing. Five
were caught by internal checks; the two that mattered most were caught only by
a control that had to match a number computed elsewhere.

## The question

v3 concluded that temporal modelling adds nothing, from a comparison that
varied four things at once: 18 layers against 50, Kinetics against
ImageNet-plus-surgical fine-tuning, 112px against 384, and 1.07 s of a 30 s
window against all of it. Only the last is the hypothesis. v4 removes them one
at a time.

## SETTLED

### 1. The confounds, decomposed

|                                   | no shift | + shift |
|-----------------------------------|----------|---------|
| 8 frames of a 2 s burst           | 0.7283   | 0.7108  |
| 8 frames spread across the 30 s   | 0.7348   | 0.7238  |
| 16 frames spread (the shipped 2D) | 0.7802   | —       |

- **Span is nearly free**: widening 2 s to 30 s is +0.0065 without the shift,
  +0.0130 with it.
- **Frame count is the real handicap**: ~0.045 of the gap between a temporal
  arm and the shipped model is that temporal arms read 8 frames and the 2D
  model reads 16. Nothing to do with temporal modelling.
- **The shift itself costs 0.0175 on the narrow input, 0.0110 on the wide
  one** — small, and smaller where there is more to see, which is the
  direction a temporal mechanism should move.

I originally attributed 0.0519 to "the input" and called it three times the
mechanism's cost. That comparison moved span and frame count together; the
split above is the corrected version.

### 2. Resolution is not the problem

The v3 Kinetics arm at 224px scores 0.7427 honest against 112px's 0.7494 —
*worse*. More pixels is not the missing ingredient.

### 3. BatchNorm was suppressing every conversion

A conversion starts from weights already fine-tuned 20 epochs on this corpus.
Training it updates BatchNorm's running statistics — overwriting what the
classifier was fitted against — and that happens regardless of learning rate:

    untrained conversion, self-tuned   0.7609
    1 epoch, BN frozen                 0.7530
    1 epoch, BN free, lr 1e-5          0.6288
    1 epoch, BN free, lr 1e-4          0.6277

A tenfold change in step size moved the first epoch by 0.0011. Freezing moved
it by 0.124.

**Corrected once**: unfrozen BN at 1e-5 is a three-epoch *detour*, not a
demolition — it recovers to 0.7100 by epoch 3. At 1e-4 the two effects compound
into genuine collapse (0.6277, 0.4981, 0.4280, val_loss 0.23 → 1.15). Freezing
is the right default because on an 8–10 epoch budget a three-epoch detour is a
third of the run, not because the alternative is fatal.

Frozen-BN training is also the first thing tonight that *improved* a
conversion: 0.7283 untrained → **0.7582 honest** trained.

### 4. Fusion: no tool arm adds anything to the 2D model

Every dumped tool arm, best blend weight **0.00**, folds disagreeing on all:

    temporal_tsm_frozenbn   0.7802 -> 0.7640   w=0.00   err-r 0.647
    temporal_i3d_dense      0.7802 -> 0.7768   w=0.00   err-r 0.574
    temporal_tsm_control2   0.7802 -> 0.7528   w=0.00   err-r 0.696

Error correlation with the 2D model runs 0.574–0.705. That is substantial
overlap but **not** redundancy — at r≈0.65 an arm still holds independent
error, so correlation alone does not explain the null. What does is correlation
*together with* weakness: every arm is 0.02–0.14 below the 2D model, and a
model that is both worse and largely agreeing has no weight at which it helps.

### 5. The pool is sound

235 shards, 24,578 windows, identical window counts, task labels and tool sets
to the dense pool, 32 frames per window everywhere. Verified rather than
assumed.

## PENDING

- **The task head.** Four consecutive epochs above the shipped model
  (acc 0.9569 vs 0.9456, desc 0.9687 vs 0.9581 — margins ~+0.011), but these
  are self-tuned per-epoch numbers. The honest dump decides, and then fusion
  decides whether it ships. This is the axis v3 predicted would pay.
- **16-frame TSM**, testing whether closing the frame-count gap closes the
  distance to 0.7802.
- **The Kinetics ablation.** Reported at +0.19 after epoch 2 and −0.05 after
  epoch 3; both arms then diverged and were restarted at a corrected rate. The
  question is open.

## Mistakes worth keeping

Each of these produced a plausible number for the wrong thing, and each was
caught by a check rather than by intuition.

- `meta.get("fold_div") or 8` rebuilt the **no-shift control with the shift on**,
  because 0 is falsy. Caught only because both dumps returned 0.7108 identically.
- The multi-clip loader fixed a 30× gradient deficit and thereby ran the
  learning rate 4× faster than it was measured at. Both Kinetics arms diverged.
- BatchNorm freezing applied before the epoch loop is undone by
  `run_clip_epoch`'s own `model.train()` call. Binding it to `train()` is what
  makes it hold.
- The dump JSON did not record which shard pool it scored, so the report was
  guessing from filenames and mislabelling multi-pool rows as dense.
- The task head's evaluator was tools-only and would have crashed at ~05:00 on
  the one arm currently beating the shipped model.

## Correction, 02:15 — the task arm was mis-selected and cut short

I decided at 01:43 not to restart the running task arm for the selection-metric
fix, on the grounds that the two metrics disagreed by 0.0022. The finished
trajectory says that was wrong:

    epoch 0  acc 0.9536  macroF1 0.8234*  desc 0.9648   <- checkpoint saved here
    epoch 1  acc 0.9545  macroF1 0.8234   desc 0.9648
    epoch 2  acc 0.9545  macroF1 0.8012   desc 0.9670
    epoch 3  acc 0.9569  macroF1 0.8018   desc 0.9687
    epoch 4  acc 0.9597  macroF1 0.8082   desc 0.9713
    epoch 5  acc 0.9629  macroF1 0.8153   desc 0.9741   <- best on acc AND desc

Selecting on macro-F1 kept the epoch-1 checkpoint and threw away 0.0093 of
description accuracy, four times my estimate. Worse, accuracy and description
accuracy rose MONOTONICALLY from epoch 2 to epoch 5, so the six-epoch budget
also cut the arm off while it was still improving.

Relaunched as cluster 9655269: 14 epochs, selection on description accuracy,
everything else identical. The original arm's dump still runs -- it is a real
measurement of a real checkpoint, just not of this arm's best one.

## 02:22 — the task head beats the shipped model on the honest protocol

The first temporal arm in this project to do so, on the axis v3 predicted.

    task TSM (multi-burst, frozen BN, 1e-5)   accuracy 0.9545   desc 0.9648
    2D task head, same windows and folds      accuracy 0.9456   desc 0.9581
                                                       +0.0089        +0.0067

And that is the MIS-SELECTED epoch-1 checkpoint. The same run's epoch 5 reached
desc 0.9741 self-tuned, so the relaunched arm should land higher.

### Fusion: a null, but a different KIND of null from the tools arms

    2D alone (w=0)         0.9581
    3D alone (w=1)         0.9648
    fusion at chosen w     0.9607   (+0.0026)
    fold A picks w=0.25, fold B picks w=0.50  ->  DISAGREE  ->  null

Every TOOLS arm had best w = 0.00: no temporal weight helped at all. Here both
folds independently chose a NONZERO weight and disagree only about its size.
That is still a null by the standing rule and is treated as one, but it is not
the same finding.

### The decision rule changes when the arm is stronger

Fusion is the right test for an arm that is WEAKER than the shipped model --
it asks whether the arm carries something independent. This arm is stronger
standalone, so the question is replacement, not blending.

**What is not yet established**: the per-fold accuracies are 0.9387 and 0.9702,
a spread of 0.031 -- larger than the +0.0089 margin. So the improvement is not
yet separable from which cases landed in which fold. The relaunched arm
(cluster 9655269, 14 epochs, correct selection) is the evidence that would
settle it, and a wider margin is exactly what it should produce if the effect
is real.

## 02:35 — replicating the task result, on two axes

The task result rests on one arm whose margin (+0.0089) is smaller than the
spread between its own case folds (0.031). One arm cannot settle that, so two
replicates are running:

    cluster 9655273   I3D task head    a DIFFERENT MECHANISM
    cluster 9655274   TSM task, seed 13   a DIFFERENT SEED

They attack different explanations. A different seed tests whether the margin
survives run-to-run variation. A different mechanism tests whether the effect
belongs to temporal modelling at all, or only to the particular way TSM mixes
channels -- the same logic that made r3d_18 worth running beside r2plus1d_18 in
v3, where landing within 0.001 of each other is what made that null credible
rather than a single bad run.

If both replicates clear the 2D task head, the finding is real and the action
is to replace the task head. If they scatter, the margin was fold noise and
tonight's headline is that TOOLS are settled and tasks are still open.

## 03:15 — frame count was the wrong diagnosis; temporal DIVERSITY is the right one

The clean test, same pool and everything else equal:

    tsm_multi    (8 frames, multi pool)   0.7620
    tsm_multi16  (16 frames, multi pool)  0.7646    +0.0026

I predicted ~0.045 from doubling the frames. It bought 0.0026.

The reason is the caveat recorded when that arm was launched: the multi pool
holds only FOUR distinct temporal locations. Taking 16 frames instead of 8
samples the same four 0.53 s bursts more densely -- it does not add moments.
The sparse pool the 2D model reads holds THIRTY distinct moments, and samples
16 of them.

So the ~0.045 attributed earlier to "frame count" is really TEMPORAL DIVERSITY:
how many different instants the model sees, not how many frames. That is a
different instruction for v5 -- more bursts, fewer frames each -- and it is the
opposite of what "frame count" would have implied.

Cluster 9655304 tests it directly: TSM over the SPARSE pool at 16 frames, which
is the 2D model's exact input, 30 distinct moments. If it reaches 0.7802 the
shift costs nothing on that input and the entire tools gap was what we fed it.
If it lands below, the shift itself is what the gap measures.

That experiment was only valid after fixing a bug it would have hit: the spread
layout computed `i * step` with step = depth // frames, so on a 30-frame window
16 frames meant frames 0-15 -- the first sixteen seconds, called "spread".

## 03:33 — a caveat on the mechanism replicate, recorded BEFORE it reports

The two task replicates are not equally clean.

    task_tsm_seed13   TSM @384, seed 13     differs ONLY in seed
    task_i3d_multi    I3D @224              differs in mechanism AND resolution

I3D runs at 224px because a 3D-inflated ResNet-50 at 384 does not fit the
cards; TSM runs at the 2D model's own 384. So if the I3D arm fails to clear the
2D task head, that is not clean evidence against the finding -- it is the same
resolution confound that made the v3 comparison unreadable, reappearing inside
the replication.

Currently it is not clearing: 0.9217, 0.9478, 0.9402 description across three
epochs, against the 2D head's 0.9581. The seed replicate meanwhile is at
0.9657, 0.9666, 0.9732 -- above the 2D head on every epoch and above the
original arm at the same epoch.

Read that as: the SEED replicate is the load-bearing one. The mechanism
replicate can confirm the finding if it clears, but cannot refute it if it does
not, because it varies two things at once. Saying so now, before its final
number lands, so the interpretation is not chosen after seeing the result.

## 04:10 — THE PIPELINE IS VALIDATED, AND THE TOOLS RESULT WAS PART ARTEFACT

The no-shift control, built from the SHIPPED checkpoint and scored the way the
2D path scores, reproduces the canonical number exactly:

    shipped checkpoint, probability averaging   0.7802   <- canonical, to 4 dp
    shipped checkpoint, logit averaging         0.7657
    x40 checkpoint,     logit averaging         0.7320   <- what every arm was
                                                            measured against

That is the first end-to-end validation of the temporal evaluation path
tonight, and it decomposes the 0.048 discrepancy exactly:

    0.0145   aggregation. TemporalWrapper averages LOGITS; the 2D path
             averages PROBABILITIES. Predicted offline at 0.7657 by inverting
             the reference dump's own sigmoid, and measured at 0.7657.
    0.0337   the base model. Every conversion inherited tools_resnet50_x40.pt
             while the reference came from tools_resnet50_long.pt.

### What this does to tonight's tools conclusion

Every tools arm carried BOTH errors. The best of them, tsm_multi16 at 0.7609,
was built on a base worth 0.0337 less than the reference and scored a way that
costs a further 0.0145 -- against an apparent gap of 0.0193.

So "temporal modelling does not help on tools" is not established by tonight's
numbers. It may still be true; it is not what was measured. The corrected
experiment is running:

    cluster 9655520   TSM, shipped checkpoint, multi pool, 16 frames
    cluster 9655521   TSM, shipped checkpoint, SPARSE pool, 16 frames
                      -- the 2D model's exact input

### What it does to the task conclusion

Strengthens it. The task arm beat the shipped task head's 0.9456 / 0.9581 while
starting from task_resnet50_x40.pt -- a base that, if the tools gap carries
over, is materially weaker than the one it was being compared against.

### Why this survived a night of checks

The correct checkpoint is written down in exactly one place: the reference
dump's FILENAME, frame_probs_resnetlong_val.npz. Not in a sidecar, not in a
doc, not in a submit file. The 7/7 invariant suite could not catch it either --
it verifies that a conversion reproduces ITS OWN base model, which was true.
Nothing checked that the base model was the right one.

## 04:15 — the Kinetics arms stopped early, on purpose

Both overfit from epoch 2 onward: training loss falling, validation loss and
macro-F1 both deteriorating.

    r2plus1d_multi     0.5440  0.4654  0.6516  0.5798  0.3894  0.3688
    r2plus1d_scratch   0.2408  0.1311  0.3463  0.3299  0.1025

Twelve epochs was simply too many for this configuration; the useful part of
both runs is over by epoch 2. Their best checkpoints were already saved, so
they were stopped at epoch 5 and those checkpoints dumped DIRECTLY rather than
through their DAG gates -- the gate now refuses to release an evaluation for a
job that did not exit 0, which is correct in general and would have discarded
these on a technicality.

The GPUs went to the corrected TSM arms, which are the experiment that matters
now that the base-checkpoint error is known.

**The ablation, with that caveat**: Kinetics 0.6516 against scratch 0.3463 at
their respective bests. That is a real gap in the expected direction, from two
runs that were both stopped for overfitting rather than converged -- so it
answers "does Kinetics transfer to surgery at all" (yes, substantially) and
not "how much is it worth in a well-tuned run".

## 04:40 — THE TASK HEADLINE REVERSES. x40 is BETTER than the shipped checkpoint.

Both task controls, no shift, 16 frames, sparse pool, probability averaging:

    task_resnet50_long  (shipped, the reference)   acc 0.9456   desc 0.9581
    task_resnet50_x40   (what the arms inherited)  acc 0.9650   desc 0.9761

The shipped reference reproduces to +0.0000 on both metrics, so the pipeline is
validated. And x40 -- the checkpoint every task arm was built from -- is BETTER
than the shipped one by +0.0194 accuracy and +0.0180 description.

I assumed the opposite. At 03:55 I wrote that the task result "gets stronger"
because the arms started from a weaker base, generalising from the tools head
where x40 is 0.0337 WEAKER. I labelled it an assumption and then leaned on it
anyway. It is wrong: the direction reverses between the two heads.

### What that does to the claim

The task arm scored desc 0.9648. Its own base scores 0.9761. So the arm did not
beat anything -- it LOST about 0.011 relative to where it started, and cleared
the shipped model's 0.9581 only because it began above it.

"A temporal conversion beat the shipped task head" is not supported. The
supported statement is much narrower: a checkpoint that was already better than
the shipped one remained better after being converted and trained, while losing
ground in the process.

### The one thing that could still change this

The base is measured on the SPARSE pool at 16 frames; the arm runs on the MULTI
pool at 8. Different inputs, so base-versus-arm is not yet like for like.
Cluster 9655533 scores the same untrained x40 conversion on the arm's exact
input -- multi pool, 8 frames. That is the number the arm has to beat, and it
is the comparison I should have built before running any task arm at all.

## 04:45 — a free improvement to the SHIPPED model, unrelated to temporal work

The two task checkpoints, scored identically through the validated pipeline
(no shift, 16 frames, sparse pool, probability averaging):

    task_resnet50_long   acc 0.9456   desc 0.9581   <- what config/perception.json serves
    task_resnet50_x40    acc 0.9650   desc 0.9761

The checkpoint the submission does NOT use is better on the task head by
+0.0194 accuracy and +0.0180 description, on the same windows and folds that
every other decision in this project has been made on.

On tools the ordering is the opposite -- long is 0.0337 BETTER than x40 -- so
this is not "x40 is the better model", it is "the two heads disagree, and the
config picked the same suffix for both".

### Why this was invisible

`config/perception.json` names its checkpoints, but nothing in the repo records
what `_x40` versus `_long` MEANS or which scored better per head. x40 is dated
2026-08-12 19:14, after the v2 report was written, so the config plausibly
predates it and was never revisited.

### What I am NOT doing

Not changing the config. This touches the submission, the gain is measured on
the validation split that also SELECTED both checkpoints, and swapping a
served model deserves an explicit decision rather than a 04:45 edit. The
evidence is here; the call is the user's.

### What would confirm it

Re-scoring both task checkpoints through the SERVING path rather than the
research one, and checking the 11-case sample end to end. Both are cheap and
neither has been done.

## 04:55 — the task question, settled against its own base

The untrained x40 conversion scored on the ARM'S EXACT INPUT (multi pool, 8
frames) -- the comparison that should have existed before any task arm ran:

    base: x40, no shift, multi pool, 8 frames    acc 0.9642   desc 0.9750

Against it:

    arm v1  macro-F1 selected, 6 epochs   desc 0.9648   -0.0102
    arm v2  desc selected, epoch 6 of 14  desc 0.9780   +0.0030
    seed 13 desc selected, 10 epochs      desc 0.9795   +0.0045  (self-tuned)

So the first arm LOST ground to its own starting point, which is the retraction
already recorded. The properly-selected arms are marginally above it -- by
0.0030 and 0.0045, against a between-fold spread of 0.016 to 0.031.

**The honest reading: the temporal conversion is NEUTRAL on the task head.**
Not the win claimed at 02:22, not the loss implied at 04:40. It neither helps
nor hurts once it is compared against the right baseline and selected on the
right metric.

The MECHANISM replicate (I3D) peaked at 0.9560 and finished at 0.9303, below
both its base and the shipped head -- consistent with the 224px confound
recorded before it reported, so it neither confirms nor refutes.

### What actually moves the number

    swapping the served task checkpoint     +0.0180 description
    the best temporal conversion            +0.0030 description, inside noise

The checkpoint choice is six times the size of the temporal effect and costs
nothing but a config edit and a validation run.

## 05:30 — the checkpoint swap FAILS its confirmation. Do not do it.

The +0.0180 description accuracy that `task_resnet50_x40` shows over the
shipped `task_resnet50_long` does not survive the 11-case sample. Running both
end to end through the real router:

    6 of 11 cases change task_top
    1 of 11 changes its ANSWER, and changes it for the worse:

        case127  task_top  uterine horn -> skills application
                 answer    "Uterine horn" -> "Gallbladder"

x40's predictions also look degenerate on the sample: `rectal artery/vein` on
four of eleven cases and `skills application` on two, where the shipped model
produces varied and plausible labels.

**Recommendation reversed: do not swap the task checkpoint.** The validation
gain is most likely x40 overfitting the split that also selected it -- which is
exactly the failure the sample check exists to catch, and exactly why the
finding was written down as "confirm first" rather than acted on at 04:45.

It also re-confirms the standing result about the router: six perception
records changed and only one answer did. The channel is lossy, which cuts both
ways -- it protects the submission from a bad perception change as effectively
as it hides a good one.

## 05:35 — the five action items, and what each returned

The user's list, verbatim: "multi-burst extraction, convert 2d neural net
trying BOTH methods, train r2plus1d_18 the right way, multiple clips per window
per epoch, then ablation between the new kinetics and 2d --> 3d cnn
transformations."

**1. Multi-burst extraction — DONE.** 235 shards, 24,578 windows, verified
against both twins: identical window counts, task labels and tool sets, 32
frames per window. It bought less than expected: span is worth +0.007 to
+0.013, and the pool's four distinct temporal locations turned out to be the
binding constraint rather than frame count.

**2. Both conversions — DONE.** TSM and I3D, 7/7 wiring invariants, and a
control that reproduces the shipped reference to +0.0000. Best tools conversion
so far 0.7817 self-tuned at epoch 0 from the shipped base; honest numbers
pending.

**3. r2plus1d trained properly — DONE, with a caveat.** Burst-aware sampling
(the old trainer would have spliced 7.5-second jump cuts into its clips), four
clips per window, learning rate scaled to the step count. It overfit from epoch
2 and was stopped at epoch 5. Honest: **0.7053**.

**4. Multiple clips per window — DONE.** Shipped in `ShardTemporal`, closing a
30x gradient-sample deficit that had nothing to do with 3D convolution and
everything to do with how the loader was written.

**5. The Kinetics ablation — DONE.** Honest clip-level, identical protocol:

    r2plus1d, Kinetics-400 init   0.7053
    r2plus1d, from scratch        0.4533

**+0.2520 for Kinetics.** The hypothesis that "Kinetics isn't good at surgery"
is wrong in its strong form: pretraining on YouTube human actions is worth a
quarter of a macro-F1 point on surgical instrument identification. It is simply
not worth *enough* to overcome 18 layers at 112px against a fine-tuned
ResNet-50 at 384.

Both numbers come from runs stopped for overfitting rather than converged, so
they bound the transfer's value from below.

## 05:45 — the first clean positive on tools: +0.0229 over its own base

    tsm_sparse16, trained (x40 base, sparse pool, 16 frames)   0.7549
    the same conversion UNTRAINED, same input, same averaging  0.7320
                                                               +0.0229

Same base model, same frames, same aggregation, same folds. This is the
comparison the whole night has been trying to construct without a confound in
it, and it is positive: temporal training improves the model it starts from by
0.023 on tools.

It is still below the shipped references -- -0.011 against long's
logit-averaged 0.7657 and -0.025 against the probability-averaged 0.7802 -- but
it began 0.034 BELOW that base, because it inherited x40.

If the same +0.023 carries from the shipped base, a corrected arm lands near
0.789 logit-averaged. That is what clusters 9655520 (multi pool) and 9655521
(sparse pool) are measuring, each with both a logit and a probability
evaluator attached so it is compared against the reference computed the same
way.

**What this does NOT yet say.** +0.0229 over one's own base is not the same as
beating the shipped model, and the earlier fusion result stands: no tools arm
has added anything to the 2D model at any blend weight. An arm that improves
itself and still loses to the shipped model, while carrying largely the same
errors, is interesting for v5 and not shippable today.

## 06:30 — a finding outside the action items: 23 serving-path tests were dead

Not part of the night's plan, and probably the most valuable thing in this
file.

Every test in `tests/test_inference.py` that patches the forward pass has been
failing since **2026-08-12**. That day the serving path stopped calling
`predict_window` and started calling `predict_window_frames`, so an ensemble
could average two models at the FRAME level before aggregating. The tests kept
patching the old name, `monkeypatch.setattr` on a missing attribute raises, and
23 tests died at once.

**Nobody noticed because the suite could not run anywhere.** The login node has
no torch -- nine modules fail to import -- and the training image had no
pytest. "The tests pass" has meant "the router tests pass" for two days, while
the code that produces the SUBMISSION went unexercised: serving thresholds,
per-head activations, UI blurring, and the failure fallbacks that keep a
crashed case worth ~0.7 instead of 0.

Fixed, with the stub returning per-frame probabilities because that is what the
new function returns. Identical rows make every aggregator agree, so the
asserted values are unchanged rather than weakened. **46 passed, 0 failed.**

Infrastructure so it cannot hide again:

  * pytest installed to `/staging/n/nkalthoff/surgvu26/testpkgs` and put on
    PYTHONPATH by `condor/verify.sh` -- a read-only .sif cannot be modified
  * `tests/` and `containers/` added to the transfer list
  * `scripts/run_tests.py` prints stderr, writes full output to staging, and
    classifies the five remaining failures as environmental (missing
    bert_score, untransferred fixtures) while NAMING them, so a real regression
    behind the same module still shows

**Whole suite now: 835 passed, 5 environmental, 0 real.**

### Router, final state

    full battery      156/159   98.1%   intent
    tool targets       85/88    96.6%
    end-to-end         21/21   100.0%
    held-out battery   45/45   100.0%
    unit tests        289 passed

Four changes shipped tonight -- the unanswerable-question guard, generic
instrument presence, scene-level questions, and plural answers -- with all 11
public sample answers verified byte-identical before and after.

## 07:15 — the router verified through the REAL serving path

`scripts/inference.py` run end to end over all 11 public sample cases, decoding
video and loading the shipped checkpoints, on cluster 9655759:

    valid non-empty JSON answers   11/11   ok=True
    wall per case                  3.9s min, 4.2s mean, 5.6s max (budget 600s)
    answers changed vs the last shipped validation run:   0

Every check before this went through `answer_question()` with cached perception
records, which is the right unit-level test and is not the path the graders
run. Four router changes shipped tonight and the only end-to-end evidence
predated the plural-answer change being wired. Now it does not.

## 07:15 — the corrected task arm, honest

    task TSM v2 (desc-selected, 14 epochs)   acc 0.9655   desc 0.9780
    2D task head                             acc 0.9456   desc 0.9581
                                                  +0.0199        +0.0199
    its own base (x40 conversion, untrained) acc 0.9642   desc 0.9750
                                                  +0.0013        +0.0030

Two things changed from the earlier arm: description-accuracy selection kept
the right epoch, and the per-fold accuracies tightened from a 0.031 spread to
0.0035 -- so the +0.0199 against the shipped model now clears its own noise
comfortably, which the earlier +0.0089 did not.

The attribution is unchanged and is the part that matters: **+0.0169 of that
comes from the x40 base being better than the shipped one, and +0.0030 from the
temporal training.** And the x40 base is the checkpoint the sample check
rejected at 05:30. So the temporal contribution remains inside noise, and the
apparent headline remains a checkpoint effect wearing a temporal costume.

## 07:25 — tools: training converges to the same level regardless of the base

Three arms, same recipe, different starting points (self-tuned per epoch):

    base long  0.7928 ->  0.7817  0.7616  0.7523        (tsm_shipped16, multi)
    base long  0.7928 ->  0.7771  0.7610  0.7446        (tsm_shipped_sparse)
    base x40   0.7694 ->  0.7513  0.7575  0.7543  0.7655  0.7723   (tsm_sparse16)

The strong base falls toward ~0.75. The weak base climbs toward ~0.77. Both
head for the same place from opposite directions.

**That is an attractor, not a gain.** The temporal training is not adding
information to the model it inherits -- it is re-fitting the network to
whatever level this input and recipe support, which lands around 0.75-0.77
regardless of what it started from. It reads as +0.023 when you start below the
attractor and as -0.04 when you start above it, and the +0.023 measured at
05:45 is that artefact rather than a benefit.

Consistent with the fusion result, which found no blend weight helping at any
arm: a model re-fitted to the same data through a lossier input carries the
same information, worse.

**Caveat: the two shipped-base arms are at epoch 2 of 8.** The x40 arm dipped
at epoch 2 before rising, so monotonic decline over three epochs is suggestive
and not final. What would overturn this is either shipped-base arm climbing
back above 0.7928.

## 08:00 — tools, apples to apples on the shipped model's own input

Same base checkpoint, same pool, same frames, same aggregation, same folds --
the comparison that took all night to construct without a confound:

    sparse arm, trained     logits 0.7616   probs 0.7593
    the same conversion,
    untrained (its base)    logits 0.7657   probs 0.7802
                                  -0.0041         -0.0209

The arm trains with logit averaging, so -0.0041 is the fairer number and it is
still negative. **Converting and training the shipped model makes it worse on
its own input.**

That completes the tools picture and it is consistent across every arm:

  * from a WEAK base the conversion rises (+0.0229) toward ~0.76
  * from the SHIPPED base it falls (-0.0041 to -0.0209) toward the same place
  * no blend weight helps at any arm
  * error correlation with the 2D model runs 0.57-0.70

Temporal modelling on this corpus does not add information. It re-fits the
network to whatever the input supports, and a burst-sampled input supports
slightly less than the 16 spread frames the 2D model already reads.

Still pending: the multi-pool arm scored 0.7793 logits / 0.7728 probs, which is
the closest any temporal arm has come to the canonical 0.7802 -- but its
matching baseline (shipped checkpoint, multi pool, 16 frames) has not been
measured yet, and comparing it to the SPARSE baseline would repeat exactly the
cross-pool mistake this section exists to avoid. Clusters 9656220 and 9656221.

## 08:05 — the mechanism DOES pay, but only on input that contains motion

The multi-pool baseline completes the pair, and it splits by input:

    MULTI pool (4 bursts, frames 67 ms apart -- real motion)
        trained    0.7793 logits
        its base   0.7706 logits
                   +0.0087

    SPARSE pool (16 moments, frames 1 s apart -- no motion to see)
        trained    0.7616 logits
        its base   0.7657 logits
                   -0.0041

Same mechanism, same base checkpoint, same frame count, same aggregation. The
only difference is whether consecutive frames are 67 ms or 1 s apart, and the
sign of the effect flips with it.

**That is the temporal mechanism doing exactly what it should**: it helps when
there is motion in the input and costs a little when there is not, because the
25% of channels it spends on the time axis are wasted on frames a second apart.

It also revises the attractor reading from 07:25. Convergence toward ~0.76 was
real, but it is not the whole story -- on motion-bearing input the arm lands
ABOVE its own base rather than below it.

### What it does not change

0.7793 against the canonical 0.7802 is -0.0009: parity, not a win, from a model
reading four distinct moments where the 2D model reads sixteen. And fusion
still finds no helpful blend weight at any arm, with error correlation
0.57-0.70 -- the arm is not seeing anything the 2D model misses, it is seeing
the same things about as well.

So the honest statement is narrower than "3D works" and wider than "3D does
nothing": **the temporal mechanism contributes +0.0087 when fed real motion,
which is enough to reach parity with the shipped model on a much narrower view
of the window, and not enough to beat it or to add to it.**

## 2026-08-15, 10:40 — the residual arm: temporal capacity that is ADDED, not traded

Everything above tests conversions. A conversion rebuilds the 2D network with
time inside it, and the measurements are consistent about what that costs:

    untrained TSM conversion, scored honestly    0.011-0.018 BELOW the 2D model
    best trained arm of the night                0.7793 against 0.7802
    fusion with the 2D model                     no helpful weight, at any arm

The reading that fits all three is that TSM and I3D do not ADD temporal
capacity — they TRADE appearance capacity for it. `fold_div=8` replaces a
quarter of every residual block's channels with time-shifted copies of its
neighbours' channels; an inflated kernel is repeated along time and divided by
its extent. Neither adds a parameter. So the mechanism starts in debt and has
to earn back what installing it spent, and across a night of arms it earned
back roughly what it spent.

`ResidualTemporal` (`src/surgvu/temporal.py`) does not touch the 2D network:

    logits_i = twod(centre frame of burst i) + alpha * motion(burst i)

with `alpha` a learned scalar **initialised to zero** and the whole 2D trunk
frozen and pinned in eval mode. At initialisation the second term vanishes and
the model computes exactly what the shipped model computes over the same
sixteen moments — asserted to floating-point equality against the real shipped
checkpoint in `scripts/verify_temporal.py`, both per frame and after the
probability averaging the score is actually computed on. The trainable surface
is `alpha` plus a ~1.6M-parameter motion branch; the 47M-parameter ResNet-50
cannot move.

**Why the branch reads differences.** At 67 ms spacing two frames are nearly
identical in appearance, so `feature(t+1) - feature(t)` is almost purely
motion. Handing the branch raw features instead would let it re-learn
appearance that the frozen trunk already encodes better, and then "temporal
helped" would mean "a second appearance model helped".

**Why alpha=0 beside a RANDOM branch, rather than alpha=1 beside a zeroed
head.** Both are the 2D model at init. The first is ReZero: `dL/d(alpha)` is
the branch output dotted with the upstream gradient, which is non-zero, so
alpha escapes on the first step and the branch trains behind it. The second
lets a randomly-scaled correction reach the logits the moment the head moves.
There is a test asserting alpha receives gradient, because a gate that can
never open would reproduce the 2D number forever and look exactly like a
completed experiment that found nothing.

**What this changes about the comparison.** The arm's base is no longer 0.7802.
It is the alpha=0 dump on `shards_multi16`, which is the same MODEL on
different FRAMES: the sparse pool samples 16 bin centres of a 30-frame 1 fps
window, and the 16-burst pool puts a burst centre at the same fraction of the
window but at continuous time. Same sampling design, different frames. So
`scripts/save_temporal_init.py --mechanism residual` runs FIRST, its number is
the base, and the gain is measured against that — the rule that came out of
Attempt 1, where two contaminations worth 0.048 survived six hours precisely
because the reference was never reproduced through the new pipeline.

**Supporting pieces.** A `bursts` loader layout that emits whole bursts in time
order and never jitters across a boundary (the arm splits the clip at burst
boundaries, so a one-frame jitter would put the tail of one burst and the head
of the next into a single "motion" — a 7.5-second jump cut called a 67 ms
step); the same picks mirrored in `clip_indices` so the evaluator scores the
frames the arm trained on; chunked trunk forwards, because 16 bursts x 3 frames
at 384px is 192 images in a batch of four and conv1 alone would hold 1.8 GB.

**The honest prior.** This construction removes the reason temporal arms have
started behind. It does not create motion information where there is none, and
the tool head remains the wrong place to look for it: tool INSTALLATION state
does not change within a window, so motion cannot predict something constant
over the interval being modelled. The task head is the better bet, and the
measured mechanism effect — **+0.0087 on motion-bearing input, -0.0041 on
sparse input** — is the size of effect to expect, not a transformation.

## 2026-08-15, 11:55 — the motion calibration, and what it forbids

Frame-difference activity over 1,131 windows of 40 training shards, computed
by `scripts/calibrate_motion.py`. The script was written to be able to return
a negative and it returned a mixed one, which is more useful.

### 1. The statistic is real, not a constant with noise on it

    micro (within a burst, 67 ms)    mean 4.689  p10 2.512  p90 7.445  CoV 0.44
    macro (between centres, 1.875 s) mean 19.270 p10 11.598 p90 28.166 CoV 0.35

Macro is 4.11x micro, which is the sanity check passing: frames 1.875 s apart
differ more than frames 67 ms apart, and by roughly the right order.

### 2. It carries task information — a 1.90x spread, in the predicted direction

Restricted to classes with n >= 40, because the others cannot be read:

    skills application                   n=204   6.240
    rectal artery/vein                   n=244   5.136
    suspensory ligaments                 n=91    4.731
    suturing                             n=360   4.112
    uterine horn                         n=179   3.725
    retraction and collision avoidance   n=40    3.292

Skills application is dexterity drill — gross, continuous movement. Retraction
and collision avoidance is holding tissue still. They sit at opposite ends,
1.90x apart, which is what physics predicts and therefore weak evidence that
the statistic measures what it claims to.

The reported AUCs are NOT evidence and should not be quoted: `other` has n=1,
so `suturing_vs_other` AUC 0.958 and `range of motion_vs_other` AUC 1.000 are
one-sample artefacts. The only interpretable one is suturing vs range of
motion at 0.316 (n 360 vs 12) — the right direction, far too few on one side.

### 3. THE CUTTING RULE IS NOT JUSTIFIED. The gate stays closed.

471 windows hold a credible cutting tool; 46 of them (9.8%) fall in the bottom
activity decile. So the proposed rule — answer "cut?" with Yes only if a
cutting tool is present AND the scene is moving — would flip about one cutting
answer in ten from Yes to No.

That is the interesting middle: not inert, not reckless. And it is exactly
where the rule must NOT ship, for a reason already written down in the router:

    a wrong polar answer costs 0.2985 of one case, and the gold polar answers
    in this corpus skew Yes; a threshold tuned to avoid false positives is
    systematically too strict for a bet this cheap

`credible_tools` is deliberately PERMISSIVE because of that measured skew. The
motion rule pushes the opposite way, and there is **no cutting label in this
corpus** to establish that any one of those 46 flips is correct. Shipping it
would mean betting against a measured prior on unvalidated evidence, to change
9.8% of one intent. `STATIC_ACTIVITY_THRESHOLD` stays None.

This is the second time a plausible perception win has failed its confirmation
— the first was the x40 checkpoint swap, +0.0180 on validation and six changed
predictions on the sample. Both were caught by asking what the change would
DO rather than whether the signal was real.

### 4. Where it redirects the work

The task head, which is where the handoff independently pointed and where
these numbers point too. Task classification is open-ended rather than polar,
so the Yes-skew argument does not apply, and the 1.90x spread says motion
carries information the appearance model may not already have. That is a case
for the LEARNED branches — `ResidualTemporal` with the local gate alpha and
the sequence gate beta — supervised on the task head and scored against the
alpha=0 base, not for a hand-written rule.

Caveats on this calibration, stated so they are not rediscovered later: the 40
shards are the first 40 in path order rather than a random sample of cases; the
class balance is whatever those cases happened to hold; and the statistic does
not separate camera motion from instrument motion, so "skills application
moves more" may partly be "the scope moves more during drills".

## 2026-08-15, 12:40 — the vision system, and three classes of bug it surfaced

### What was built

A three-expert perception system, layered so each piece is separately
testable and the shipped answers cannot move until something is deliberately
turned on.

    L1  E1  2D appearance      the shipped ResNet-50 pair, FROZEN, untouched
        E2  micro-motion       within a burst, 67 ms       (analytic)
        E3  macro-motion       across burst centres, 30 s  (analytic)
        E2/E3 learned          ResidualTemporal, alpha and beta, both zero
    L2  PerceptionRecord       the appearance record plus an additive
                               "motion" block
    L3  router                 tri-state accessors, gate CLOSED
    L4  harness                closed-gate tests + an end-to-end A/B

### The evidence, in the order it was produced

    residual arm at alpha=0 vs the 2D model, same batch     0.0 exactly
    branch effect at alpha=1, for scale                     0.47
    motion micro/macro independence (synthetic)             56.75/0.00, 0.00/56.46
    task-class activity spread (n>=40 classes)              1.90x
    cutting answers the proposed rule would flip            9.8%
    serving A/B, 11 sample cases, motion on vs off          0 changed
    validate.sh vs motion_ab off vs motion_ab on            all identical
    --motion cost, measured                                 6.6s -> 7.0s per case
    test suite                                              932 passed, 0 real

### Why the gate is closed

The motion statistic is real and the router rule it was built for is not
justified. Details in the 11:55 entry: 9.8% of cutting answers would flip
Yes->No, against a corpus whose gold polar answers skew Yes, with no cutting
label to validate a single flip. `STATIC_ACTIVITY_THRESHOLD` stays None.

### THREE CLASSES OF BUG, each found more than once

Worth naming, because each was found by looking rather than by being told,
and each recurred:

**1. A check that cannot fail.** `run_tests.py` counted a SIGSEGV as "0
failures, 0 real" and exited 0, because every number it produced was parsed
from lines a crashed pytest never printed. `motion_ab.sh` reported a dead pass
as eleven changed answers, then discarded the exit code that said so. A
verification that cannot distinguish "no result" from "a good result" will
eventually report one as the other, and the direction is not predictable --
today it produced one false alarm and one false all-clear.

**2. A constant whose source of truth moved.** `verify_temporal.py` defaulted
to the x40 checkpoint the shipped config has not bound since v3 (the 0.048
contamination, now an assertion). The extraction submit file sized memory for
32 frames per window while the pool holds 48 (6.04 GB of raw frames on a
160-window case). `validate.sh` copied `tools_v2.pt` while the config bound
`tools_resnet50_long.pt`, so the SUBMISSION path's own validator could not
pass at all. Standing check: does this name come from the config, or from
someone's memory?

**3. A rescue that sustains what it rescues.** `release_held.sh` released
every held job at a flat 8192 MB; five 160-window cases measured 9,766 MB and
cycled -- held, released at 8192, killed -- six times for one of them. Calling
it on a timer FED the loop. It now escalates from the measured peak and can
never hand a job less than it just died at.

### What is not done

The pool finished at 234 shards and the last case is still extracting. The
alpha=0 baseline dumps are built and waiting on a complete, verified pool; the
learned task arm is queued behind that. The calibration pointed at the task
head and the learned branches are the way to test it -- not a hand-written
rule.

## 2026-08-15, 13:40 — the alpha=0 baselines, and a prediction that missed

The rule from Attempt 1 is to reproduce the reference through the new pipeline
before trusting any arm on it. Both heads, on shards_multi16, at alpha=0 --
where the residual model IS the shipped 2D model by construction:

    head    alpha=0 base                    reference              delta
    task    0.9454 acc / 0.9577 desc        0.9456 / 0.9581        -0.0002 / -0.0004
    tools   0.7673 macro-F1                 0.7802                 -0.0129

### The task head reproduces. The tools head missed the prediction.

The prediction recorded at 11:13, BEFORE either ran, was "within about 0.01,
the cuDNN run-to-run noise floor". Task landed at -0.0004. Tools landed at
-0.0129, which is outside that band, and the same entry said "materially below
that is a bug to find, not a pool to accept". So it was investigated rather
than accepted.

### It is not a bug, and the first explanation was wrong

The obvious hypothesis was that macro-F1 with tuned thresholds is hypersensitive
to small probability shifts in a way argmax accuracy is not. MEASURED, AND
FALSE:

    perturb every threshold by +-0.010     macro-F1 swings 0.0026
    add sigma=0.02 noise to the probs      macro-F1 moves +0.0004

So the metric is robust and -0.0129 is a real difference in predictions.

The actual mechanism is class support. Macro-F1 is the unweighted mean over
twelve classes whose support on this split runs from 29 to 2308:

    stapler                        29    F1 0.8214
    tip-up fenestrated grasper     68    F1 0.0000   (structurally unmeasurable)
    clip applier                  197    F1 0.8084
    ...
    cadiere forceps              2308    F1 0.8870

One flipped window in a 29-support class moves that class's F1 by 0.0345 and
macro-F1 by 0.0029. **Four or five flipped windows are the entire -0.0129** --
which is exactly what moving each sampled moment by up to 0.438 s does to
marginal detections of instruments entering or leaving the field.

Three independent lines agree the pipeline is sound: the task head reproduces
to -0.0004 through the IDENTICAL code path, so a wrong sampler, wrong
aggregation or wrong checkpoint would have to be tools-only; the metric is
robust to perturbation; and the rare-class arithmetic accounts for the size.

### What this changes

**The base for a tools arm on shards_multi16 is 0.7673, not 0.7802.** An arm
scoring 0.7750 on this pool has beaten its base by +0.0077 and would look like
a 0.0052 regression against the wrong comparator. `dump_temporal_probs` used
to print "2D ResNet-50, same windows and folds: 0.7802", which was true about
windows and folds and quietly false about FRAMES; it now says which pool the
literal came from and that the arm's base is the dump's own alpha=0 number.

The task head needs no such correction: -0.0004 is within noise of the shipped
reference, and it is the head the calibration pointed at.

## 2026-08-15, 13:55 — the feature cache, verified, and a bug I reintroduced

### The cache

`ResidualTemporal` freezes the trunk completely -- that freeze is what makes
alpha=beta=0 exactly the shipped model -- and `train_temporal.py` then
recomputed that frozen trunk from JPEG on every epoch. Measured: 1.83
windows/s, 1.6 h per epoch at 24 frames, 3.1 h at 48, **thirty hours for eight
epochs** of pushing pixels through weights that cannot change.

What `SequenceBranch` actually reads is 16 pooled 2048-d vectors per window
plus the 16 per-centre 2D logits. That is **1.61 GB in fp16** for all 24,578
windows, and the caching pass decodes only the 16 CENTRE frames rather than all
48 -- three times cheaper than one epoch, replacing all of them.

    epoch on the GPU path      1.6 - 3.1 hours
    epoch on the cache         20.4 s for THREE epochs
    GPU slots willing to run   0     (measured, twice, at cpus=8 and cpus=4)
    CPU slots at >=4GB         14,604

The second row is the speedup; the last two are the bigger win. The cache moves
the experiment off the scarcest resource in the pool, which is why the GPU arm
sat idle for twenty minutes while a CPU job did three epochs in twenty seconds.

**Verified against the full forward**, on all 4,635 val windows:

    max_abs_diff       1.78e-07
    mean_abs_diff      4.76e-09
    argmax_agreement   1.0
    cases_aligned      true

What it cannot cache is the LOCAL branch: spatial maps for every frame of every
burst, 2048x12x12 each, about 696 GB. A local arm keeps paying full decode.
That is a third independent reason to test the 30 s timescale first, alongside
the calibration pointing at the task head and the local branch being
order-invariant after its mean.

### I reintroduced the v4 aggregation bug

`train_sequence_cached.py` evaluated by squashing the MEAN LOGITS and reported
the alpha=0 base as accuracy 0.9415 / description 0.9538. The dump of the same
checkpoint on the same windows said 0.9454 / 0.9577. The cached logits had
already been verified identical to the dump's to 1.78e-07, so the 0.0039 gap
could only be aggregation -- `softmax(mean(logits))` is not
`mean(softmax(logits))`.

That is the bug that cost 0.0145 and most of a night in v4, whose fix was
`--aggregate probs`. I wrote it again, in a new file, three weeks later. Both
forms are one line, both produce plausible probabilities over the right
classes, and nothing distinguishes them except a side-by-side against a number
computed the other way.

Two things made it catchable, and both were deliberate:

  the trainer prints its alpha=0 base BEFORE training, so a wrong number sits
  next to a right one

  the alpha=0 base had already been established through the dump, so there WAS
  a right one to sit next to

After the fix the trainer's base reads 0.9454 / 0.9577 -- identical to the
dump. The loss still averages logits, which matches what train_temporal.py
optimises; only evaluation changed, because evaluation has to measure what the
scorer measures.

### The verification chain, end to end

Every link now checked against an independent computation of the same thing:

    pool            vs its twin              0 missing / 0 extra / 0 problems
    residual model  vs the 2D model at 0     exactly 0.0, real weights
    cache           vs the full forward      1.78e-07, argmax 1.0
    cached trainer  vs the dump              base identical, 0.9454 / 0.9577
    serving+motion  vs serving               0 of 11 answers changed

## 2026-08-15, 14:45 — the 30-second branch: a null, and a false positive caught in the act

Six configurations of `SequenceBranch` on the task head, trained on cached
trunk features, scored against the alpha=0 base of **0.9577** established
through the same pipeline.

    config                          params    honest    h.gain  naive gain     bias
    ref  h=256 lr1e-3 wd.01      1,118,729    0.9540   -0.0037     +0.0024  +0.0060
    cfg1 h=64  lr1e-4 wd.01        169,097    0.9551   -0.0026     +0.0015  +0.0041
    cfg2 h=64  lr1e-4 wd.1 d.3     169,097    0.9553   -0.0024     +0.0019  +0.0043
    cfg3 h=256 lr1e-4 wd.1 d.5   1,118,729    0.9566   -0.0011     +0.0022  +0.0032
    cfg4 h=32  lr3e-4 wd.1 d.3      75,337    0.9579   +0.0002     +0.0017  +0.0015
    cfg5 h=256 lr1e-5 wd.01      1,118,729    0.9575   -0.0002     +0.0019  +0.0022

`honest` = the epoch chosen on one case fold and scored on the other, both
directions. `naive` = the best epoch on all of val, which is what almost
everyone reports.

### THE RESULT: the 30 s branch adds nothing to the task head

Best honest gain across six configurations is **+0.0002** -- one window in
4,635. Across a 15x capacity range (75k to 1.1M parameters), three learning
rates spanning 100x, dropout 0 to 0.5 and weight decay 0.01 to 0.1, **no
configuration clears its own base.**

The shape matters and rules out a tuning failure. A mis-tuned branch looks
like "bad at high capacity, good at the right capacity" -- a peak somewhere in
the sweep. This is MONOTONE toward zero: the branch HURTS when it has enough
capacity to overfit (-0.0037) and does NOTHING when it does not (+0.0002). It
converges on the behaviour beta=0 already gives for free, which is exactly
what the residual construction guarantees as a floor.

beta did open -- it reached +0.054 on the reference config, so the gate was
used and the branch reached the logits. Validation simply did not follow.

### THE METHODOLOGY FINDING, which may be the more useful half

**Every naive number is positive. Every honest number is at or below zero.**

    naive gains   +0.0015 to +0.0024, six for six
    honest gains  -0.0037 to +0.0002, six for six

Reported the ordinary way -- best epoch on validation -- this sweep produces
six consecutive "wins" of exactly the size this project treats as a result.
All six are artefacts of picking the best of N epochs on the set being
reported. Without the fold protocol added at 14:04, TODAY WOULD HAVE ENDED
WITH "+0.0024, the 30 s branch helps the task head" written down.

And the bias is not a constant to be subtracted; it scales with capacity,
which is what theory predicts because capacity is what lets a model exploit
validation noise:

    1,118,729 params  ->  +0.0060 bias
      169,097 params  ->  +0.0042
       75,337 params  ->  +0.0015

That is a clean empirical demonstration on real data, worth carrying into the
methodology report: a temporal-modelling literature reporting +0.002 gains
from best-epoch-on-val selection is reporting its own selection procedure.

### What this does and does not settle

SETTLED: the across-burst (30 s) timescale, on the TASK head, through a
learned branch, adds nothing measurable. That was the strongest remaining
temporal hypothesis -- the calibration pointed at the task head, the pool was
built to remove the coverage handicap, and the residual construction removed
the reason conversions started behind.

NOT SETTLED: the LOCAL (0.2 s) branch on the task head is untrained. It cannot
use the cache -- it needs spatial feature maps for every frame, about 696 GB
-- so each arm costs ~1.6 h per epoch instead of two minutes. Given six
configurations of the cheap branch found nothing, and given the local branch's
per-burst corrections are order-invariant after their mean, spending GPU days
on it is hard to justify before something else changes.

ALSO NOT SETTLED: whether ANY perception gain reaches the answer. That remains
the binding question -- v1 and v2 scored identically to four decimals because
their answers were byte-identical, and the router's cutting rule still answers
an event question with a presence proxy.

### Why no confirmation dump was run

`dump_temporal_probs` would rebuild the arm from JPEG and score it
independently, at ~40 GPU-minutes. It was not run because the arm does not
clear its base, so nothing downstream depends on the number, and the
trainer-versus-dump agreement was already established AT beta=0 to four
decimals (0.9454 / 0.9577 both ways) with the cached logits verified to
1.78e-07. Had any config cleared its base, the dump would have been mandatory
before believing it.

## 2026-08-15, 15:20 — two "environmental" failures were a wrong path

`scripts/run_tests.py` classifies failures as environmental when they are
missing `bert_score` or a missing file, so that a permanently-red suite does
not teach everyone to ignore it. Its own comment says why that matters: "how 23
dead serving-path tests went unnoticed for two days."

Five results sat in that excused category. Two did not belong there.
`test_answer_form_eval` read `REPO/shipped_candidates.json` -- the UNTRACKED
copy that a job dropped in the submit directory -- while the tracked copy is
`outputs/vlm/shipped_candidates.json`. Byte-identical, but only the tracked one
is transferred to an execute node, so those two tests had never run anywhere
and had been excused since they were written.

They were a wrong path, not an environment. **An excused category is where a
real failure hides**, and the excusing was working exactly as designed --
"missing file" is a true description of what happened.

Made permanent earlier the same afternoon, too: the `/*.json` hygiene rule
added at 14:00 ignores the root copy, so it could never be committed and the
test could never have been fixed by transferring it. A cleanup that cements a
bug is worth catching.

Fixed by pointing at the tracked copy and transferring `outputs/` in
verify.sub. **934 passed, 3 environmental, 0 real** -- and the three that
remain genuinely need `bert_score`, which is in the scoring venv rather than
the read-only training image.

## 2026-08-15, 16:12 — pricing the motion rule on the graded set, and a retraction

`calibrate_motion.py` measured the rule's flip rate on TRAINING windows nobody
grades. The eleven public sample cases are the only place we hold a question
and its gold references together, so they are the only place the rule can be
priced. Measured through the real serving decoder:

    case122  micro 3.698   Are there forceps being used here?
    case123  micro 2.970   Is a large needle driver among the listed tools?
    case124  micro 2.722   What type of forceps is mentioned?
    case125  micro 2.883   Is a suture required in this surgical step?
    case126  micro 1.700   Was a large needle driver used in this clip?
    case127  micro 3.011   What organ is being manipulated?
    case128  micro 5.083   Is a needle driver involved in the procedure?
    case129  micro 3.129   What procedure is this summary describing?
    case130  micro 1.283   What is the purpose of using forceps...?
    case131  micro 4.013   Is tissue being cut during this clip?
    case132  micro 1.010   Was a large needle driver used during the surgery?

### A RETRACTION

Before running this I wrote that the rule "would have turned the one perfect
answer in the set into a wrong one", reasoning that case131's gold is
unanimously Yes, we answer Yes, and that scores 1.0000. The first two are true.
The inference was not: **case131 measures 4.013, well ABOVE the 2.512 boundary
and the second-highest activity in the sample.** The rule would not have fired
on it. I asserted a consequence before measuring the input, which is the thing
this report keeps catching other people's numbers for.

### What is actually true, and it is a better reason

**The rule fires on ZERO of the eleven graded cases.** The three clips below
the boundary -- case126, case130, case132 -- are two tool-presence questions
and a purpose question; `_answer_cutting` never runs on any of them.

So the 11-case check can neither validate NOR invalidate this rule. A change
that moves ~10% of cutting answers in production while being provably inert on
every case we can grade is a change we cannot measure. That is a stronger
argument for `STATIC_ACTIVITY_THRESHOLD = None` than "it might hurt".

### AND THE THRESHOLD DOES NOT TRANSFER

    training windows   median 4.326   p10 2.512
    sample clips       median 2.970   min 1.010   max 5.083

The graded clips are systematically LESS active than training windows, so a
boundary calibrated to catch the bottom 10% of training catches **3 of 11 =
27%** of the sample. Any motion threshold fitted on `shards_multi16` and
applied at serving time fires nearly three times more often than its
calibration implies.

That generalises past this rule to any future one. It is also exactly the
class of error that only surfaces after a submission moves the wrong way,
because nothing about a threshold announces which distribution it was fitted
on. If a motion threshold is ever wanted, calibrate it on clips that resemble
what is graded -- or at minimum report both quantiles side by side.

## 2026-08-15, 16:30 — where the sample's headroom actually is, and how much is reachable

The eleven graded cases score **0.8767** mean (container_gpu, and today's
`validate.sh` run reproduces the same eleven answers byte for byte). Eight are
perfect. All the headroom is in three:

    case124  0.2402   gap 0.7598   worth 0.0691 of the 11-case mean
    case126  0.7015   gap 0.2985   worth 0.0271
    case132  0.7015   gap 0.2985   worth 0.0271
                                   -------
                            total  0.1233

The heldout split in `config/splits_v2.json` is EXACTLY these eleven cases, so
none of this leaked into training and the tool labels for them can be read
directly. Doing that changes what the three numbers mean.

### case124 — a genuine, fixable perception error. The biggest single item.

    Q     "What type of forceps is mentioned?"
    gold  Cadiere Forceps
    ours  Bipolar Forceps
    perception: bipolar 0.992, cadiere 0.059

`case_124` has Cadiere Forceps (3 intervals), Maryland Bipolar Forceps (2) and
ProGrasp (1). Both classes are in the 12-class taxonomy, and the model is
CONFIDENTLY wrong -- 0.992 on the wrong one, 0.059 on the right one. Nothing
structural prevents fixing this. **Worth 0.0691, which is 72% of all reachable
headroom, in one case.**

### case126 — a detection miss, also fixable

    Q     "Was a large needle driver used in this clip?"
    gold  Yes
    ours  No
    perception: needle driver 0.199 (under its cut)

`case_126` contains ONLY Large variants -- `Large Needle Driver` and
`Large SutureCut Needle Driver`, no Mega -- so any needle driver in that clip
IS a large one and the gold is unambiguous. We missed it at 0.199. Worth
0.0271.

### case132 — probably NOT fixable by a 12-class detector

    Q     "Was a large needle driver used during the surgery?"
    gold  No
    ours  Yes
    perception: needle driver 0.997, cadiere 0.986

`case_132` contains BOTH families: `Large Needle Driver`, `Large SutureCut
Needle Driver`, `Mega Needle Driver`, `Mega SutureCut Needle Driver`. For the
gold to be No, the clip must sit in a window whose driver is a MEGA one -- and
the 12-class taxonomy has a single `needle driver` class that cannot express
the difference.

Given a needle driver is detected, answering Yes is the MAXIMUM-LIKELIHOOD bet:
`config/commercial_names.json` puts Large at 62.7% of needle-driver instances
against Mega at ~37%. We made the right bet and lost it. Changing that policy
to chase this case would lose more often than it wins.

**Stated as uncertain, because it is.** The alternative reading is that the
clip contains no needle driver at all and our 0.997 is a false positive, which
WOULD be fixable. Distinguishing them needs the clip's timestamp within
case_132, which the sample does not give. The label structure favours the
first reading -- 13 needle-driver intervals in the case, and a 0.997 detection
-- but that is an argument, not a measurement.

### What this means for where to aim

    reachable   case124 + case126 = 0.0962 of the sample mean
    of which    case124 alone     = 0.0691   (72%)
    likely not  case132           = 0.0271

The single highest-value perception target in this project is **bipolar-versus-
cadiere forceps discrimination**. Not temporal modelling, not more frames, not
a better aggregation -- one confusable pair, in one case, worth more than
everything else available combined.

That is also consistent with the day's null: nothing about a 30-second
timescale helps tell a Cadiere from a Maryland Bipolar in a still frame.

## 2026-08-15, 16:45 — what fixing case124 actually requires, tested rather than assumed

Two claims I made an hour apart, both wrong, both corrected by running the
router instead of reading it:

**Wrong 1:** "fixing case124 needs a router policy too, because both cadiere
and Maryland bipolar are installed in the overlap window." The router already
has `TOOL_PRIOR_ORDER` with cadiere first, so I assumed a policy existed and
would fire.

**Wrong 2:** "the router is already set up to convert a cadiere detection, so
detecting it is worth the full 0.0691." Tested by injecting cadiere at 0.95
into case124's real record and routing it: the answer stayed **Bipolar
Forceps**.

`_answer_tool_identity` ends in `display_name(_best_class(pool, scores))`, and
`_best_class` picks by SCORE. `TOOL_PRIOR_ORDER` is not consulted on this path.
So the requirement is not "detect cadiere" -- it is **"score cadiere above
bipolar"**. Swept against the real record:

    cadiere 0.0590 (actual)  ->  Bipolar Forceps
    cadiere 0.9500           ->  Bipolar Forceps
    cadiere 0.9900           ->  Bipolar Forceps
    cadiere 0.9920           ->  Cadiere Forceps   <- flips exactly at bipolar's score
    cadiere 0.9990           ->  Cadiere Forceps

### Why that is a much harder ask than "improve the tool head"

The model is at **cadiere 0.059, bipolar 0.992** -- a margin of 0.933. This is
not a near-tie that a better tie-break could rescue. The router's own docstring
measured that regime: over 2,563 well-posed forceps windows the argmax is
nearly a coin flip when the top-two margin is under 0.05, and 99.6% correct
when the margin is above 0.70. case124 sits at the confident end and is
confidently WRONG.

So the 0.0691 is reachable in principle and expensive in practice: it requires
inverting a near-total confidence gap on a confusable pair, not a marginal
gain. Any claim that a retrained tool head "would fix case124" should be
checked against this number -- the head must not merely detect cadiere, it
must beat 0.992.

### What that leaves

    case126  0.0271  a MISS at 0.199 -- needs recall, a much easier ask
    case124  0.0691  needs a 0.933-margin inversion
    case132  0.0271  probably outside the 12-class taxonomy

The cheapest real gain on the graded set is **case126**, not case124: raising a
needle driver from 0.199 past its threshold is an ordinary recall improvement,
where case124 needs a confident error reversed. Worth 0.0271 against 0.0691,
but at a fraction of the difficulty.

## 2026-08-15, 17:00 — the cheapest fix is available, measurable, and should NOT be shipped

case126 is the cheapest headroom on the graded set: a needle driver scoring
0.199 against a serving cut of 0.36, worth 0.0271. Lowering that one threshold
is a config edit. Routed end to end over all eleven cases it does exactly what
you would want:

    exactly ONE answer changes -- case126, "No" -> "Yes", which is the gold
    case127 also newly fires the class and its answer is UNCHANGED
    sample mean 0.8767 -> 0.9038

Then the same threshold measured on the 4,635 validation windows it was
originally tuned on:

      thresh    F1      falsePos  falseNeg
       0.19    0.9651      30        97     <- proposed
       0.36    0.9731       0        97     <- shipped

**False negatives do not move. 97 at both.** Lowering the cut from 0.36 to 0.19
recovers ZERO true positives on validation and adds 30 false ones. The
[0.19, 0.36) score band holds 30 negatives and no positives -- on that
population it is pure noise.

So case126 is a true positive sitting inside a band that validation says
contains nothing but false alarms. It is an outlier, and tuning a threshold to
catch it is fitting eleven cases against 4,635.

**Verdict: do not lower it.** +0.0271 on the sample against a change that is
strictly harmful where it was measured -- 30 new false positives, nothing
recovered -- is the x40 checkpoint swap wearing different clothes. That one
scored +0.0180 on validation and failed the sample; this one passes the sample
and fails validation. Same error, opposite direction, and the sample is the
smaller evidence base in both.

### What this leaves for case126

The score has to RISE, not the bar fall. That is a recall improvement on the
tool head at a specific operating point -- and the per-class F1 for needle
driver is already 0.9721, one of the best in the taxonomy, so there is not much
headroom in the class as a whole. case126 is a hard instance rather than a
weak class.

Which means the honest state of the graded set is:

    case124  0.0691  needs a 0.933-margin confidence reversal
    case126  0.0271  needs a hard instance recovered in an already-strong class
    case132  0.0271  probably outside the 12-class taxonomy

None of the three is cheap. That is worth knowing before another perception
run is launched on the assumption that some of it is.

## 2026-08-15, 17:45 — the eyeball gate on case124: the confusable pair is not the one I said

case124 is the largest single item of headroom (0.0691) and the open question
was whether its error is in the MODEL or in the LABEL -- tool labels are
installation intervals, not visibility, so a confidently-wrong 0.992 could be
the model correctly seeing an instrument the logbook had swapped out.

Extracted three frames from `case124.mp4` and looked. The da Vinci overlay
renders the installed instruments by arm, stable across the clip:

    1  CADIERE FORCEPS
    2  MARYLAND BIPOLAR FORCEPS   (L COAG, active)
    3  (endoscope)
    4  MONOPOLAR CURVED SCISSORS

That band is blurred by `surgvu/preprocess.blur_ui_band` before any model sees
it -- REQUIRED BY CHALLENGE RULES ("using the information available in the UI
to make predictions is not allowed ... the UI will be blurred from the test
set"), so this is a legitimate ground-truth check and not an exploitable
signal.

### The model makes TWO errors, and neither is the one I named

    monopolar curved scissors  1.000   CORRECT
    bipolar forceps            0.992   CORRECT -- installed on arm 2
    grasping retractor         0.976   FALSE POSITIVE -- on NO arm
    cadiere forceps            0.059   MISS -- installed on arm 1

So the answer "Bipolar Forceps" is not a misdetection: bipolar genuinely IS
installed and the model is right about it. The gold names Cadiere because the
gold names arm 1. And the model's real failures are a MISS on cadiere and a
FALSE POSITIVE on a grasping retractor that is not present at all.

**One mechanism explains both: the model appears to see the cadiere and call
it a grasping retractor.** Both are graspers; validation per-class F1 is 0.8862
for cadiere and 0.8836 for grasping retractor, the two weakest of the four
classes in play here.

### This retires "bipolar versus cadiere" as the target

Earlier entries in this report called bipolar-vs-cadiere the highest-value
perception target. That framing was wrong -- bipolar is correctly detected.
The confusable pair is **cadiere versus grasping retractor**, and fixing it
would remove a false positive as well as a miss.

It does NOT make case124 cheap. The requirement is still that cadiere outscore
bipolar's 0.992 (17:00 entry), and cadiere is at 0.059 while a wrong class sits
at 0.976. But it names a specific, checkable confusion rather than a vague
"improve the tool head", and it is a hypothesis with a visible mechanism rather
than an inference from scores alone.

### Caveat

Three frames of one clip. The retractor false positive is certain -- no
grasping retractor is on any arm. That the cadiere is what triggers it is the
best available explanation and not a measurement; confirming it needs the
confusion matrix on validation windows where cadiere is installed.

## 2026-08-15, 18:00 — the confusion hypothesis is REFUTED. Cadiere misses are recall, not substitution.

The 17:45 entry proposed that case124's two errors share one mechanism: the
model sees the cadiere and calls it a grasping retractor. Tested on all 4,635
validation windows.

**When cadiere is installed but missed (n=204), does a retractor falsely fire?**

    cadiere installed & DETECTED (n=2104)   retractor false-fires  0.00%
    cadiere installed & MISSED   (n= 204)   retractor false-fires  1.96%
    cadiere NOT installed        (n=2327)   retractor false-fires  4.08%

A missed cadiere is associated with FEWER retractor false positives than
cadiere being absent entirely. The hypothesis predicted the opposite.

**And no class substitutes for it.** On those 204 windows, every class that
fires does so at or below its base rate:

    bipolar forceps      2.94%  vs base 13.45%
    clip applier         2.45%  vs base  2.02%
    grasping retractor   1.96%  vs base  4.08%

Mean cadiere score on the missed windows is **0.104**. The model is not
confidently assigning the instrument to another class -- it is simply not
seeing it.

### What that changes

case124 has **two independent errors**, not one mechanism:

    cadiere 0.059  a RECALL failure. Population-wide, cadiere misses average
                   0.104 and are not substitutions.
    retractor 0.976  an unusually confident false positive, against a 4.08%
                   base rate for that class -- and unexplained.

So the target is cadiere RECALL, which is harder than resolving a confusion:
there is no confusable partner to separate, the model is just blind on these
windows. And "cadiere must beat bipolar's 0.992" (17:00) still stands on top of
that, from a population mean of 0.104 when it misses.

### On being wrong

That is the seventh hypothesis today that measurement overturned -- after the
motion rule's effect on case131, the case132 variant story, both readings of
the router's tie-break, my own handoff's recommended starting point, and the
cheap threshold fix. Each looked right, each was written down as a claim
before being tested, and each cost minutes to check.

The eyeball-gate finding SURVIVES: the model does make two errors on case124,
confirmed against the da Vinci overlay. Only the mechanism connecting them was
invented.

## 2026-08-15, 18:15 — cadiere recall fails PER CASE, which makes it plausibly fixable

If case124's 0.0691 needs cadiere recall (18:00), the next question is whether
that recall is uniformly hard or concentrated. Measured over the 2,308
validation windows where cadiere is installed:

    overall miss rate                8.8%  (204 of 2,308)
    case_133                        50.6%  ( 39/ 77)
    case_072                        44.8%  ( 43/ 96)
    case_115                        29.2%  ( 33/113)
    case_079                        26.2%  ( 27/103)
    ...
    case_058 / case_035 / case_087   0.0%

**Three cases hold 56% of all misses, and 11 of the 26 cases holding cadiere
have none at all.** This is not a class the model is uniformly weak on -- it is
a class the model fails on in specific cases and handles perfectly elsewhere.

That is the encouraging shape. Uniform 8.8% difficulty would mean the class is
near its ceiling; concentration means something about those cases -- lighting,
camera pose, a particular cadiere presentation, an occluding instrument -- is
recoverable with targeted data or augmentation rather than a better
architecture.

**The caveat that stops this from being a plan:** case124 is in the HELDOUT
split, not validation, so nothing here says whether it resembles case_133 or
case_058. The finding is that cadiere recall CAN be fixed case-wise, not that
case124 is one of the fixable ones.

The concrete next step is therefore not "retrain the tool head" but: look at
frames from case_133, case_072 and case_115 where cadiere is installed and
missed, find what they share, and check whether case124 shares it. That is an
afternoon of looking, not a GPU programme, and it decides whether the largest
single item of headroom in the project is reachable.

## 2026-08-15, 18:30 — the largest headroom item reduces to one question a human can answer

Compared frames from the worst and best cadiere cases, both during confirmed
cadiere-installed intervals:

    case_133 (50.6% miss)  instruments at the frame EDGES, largely cut off;
                           the dominant visible instrument is the scissors
    case_058 ( 0.0% miss)  a large, sharply-focused forceps occupying the
                           upper-left quadrant, unmistakable

That is consistent with what this project already knows and the memory already
warns about: **the labels are INSTALLATION intervals, not visibility.** An
instrument can be on an arm and outside the camera's view, and a single-frame
appearance model cannot report it. If the 8.8% cadiere miss rate is dominated
by installed-but-off-camera windows, that portion is IRREDUCIBLE -- no
architecture fixes being asked about something not in the picture.

### Which puts case124 on a knife edge

case124's UI lists three instruments -- CADIERE (arm 1), MARYLAND BIPOLAR
(arm 2, active), MONOPOLAR CURVED SCISSORS (arm 4) -- and its frames show
**two**: a dark-shafted instrument with cream/black jaws entering from the
left, and the grey da Vinci scissors shaft on the right. One of the three
installed instruments is not in view.

Two readings remain and they have opposite consequences:

    (a) the cadiere is NOT in frame
        -> case124's 0.0691 is UNREACHABLE by any appearance model. The gold
           names an instrument the picture does not contain, and the target is
           the logbook rather than the image.

    (b) the left instrument IS the cadiere, and the model is calling it a
        grasping retractor (0.976, on no arm)
        -> reachable, and it explains the false positive that the 18:00
           population test could not explain.

**I cannot resolve this by looking.** Distinguishing a Cadiere Forceps from a
Maryland Bipolar Forceps by their jaws is a domain skill I do not have, and
guessing would be exactly the over-claiming that has already been retracted
seven times today.

### What resolves it

One person who can identify da Vinci instruments, looking at
`/staging/n/nkalthoff/surgvu26/frames/case124_f0.png` and answering: **is the
left-hand instrument a Cadiere Forceps, a Maryland Bipolar, or something
else?** Thirty seconds of expert attention decides whether the largest single
item of headroom in this project is worth pursuing at all.

This is the same eyeball gate that unblocked training on 2026-08-09, when
Noah reviewed 50 annotated case_002 frames. It worked then for the same reason
it would work now: some questions are cheap for a human and expensive for
everything else.

## 2026-08-15, 18:45 — a cadiere reference set, built from the logbook rather than from judgement

To decide whether case124's cadiere is in frame without needing to identify
instruments by expertise, I found windows where **cadiere is the ONLY grasper
installed** -- no bipolar, prograsp, force bipolar, tip-up or retractor -- for
45+ minutes at a stretch. In those windows any grasping instrument visible MUST
be a cadiere, established by the logbook:

    case_040  part1  11842.7 - 14809.2   2,966 s
    case_096  part1  12917.0 - 15769.5   2,852 s
    case_016  part1   5195.7 -  8019.8   2,824 s

Frames are at `/staging/n/nkalthoff/surgvu26/frames/CADIERE_ONLY_*.png`. That is
a reusable asset for any future work on this confusion, and it needs no one to
be trusted -- the alternatives are ruled out by the labels.

### What they show, and what case124 shows

The reference cadieres present as a **dark shaft ending in a bare metallic
wristed jaw**, elongated and fenestrated, with no insulating collar.

case124's visible left-hand instrument has a distinct **cream/white component**
at the tip. Cream or white insulation is characteristic of BIPOLAR
instruments, where the insulator prevents current spread; the Maryland Bipolar
is on arm 2 and the UI shows it ACTIVE (L COAG).

**So the evidence leans toward reading (a): the cadiere is not in frame, and
case124's 0.0691 is unreachable by any appearance model.**

### Confidence, stated honestly

This is pattern-matching on insulator colour across a handful of frames by
someone who is not a surgical-instrument expert. It is a LEAN, not a finding.
Cadiere and Fenestrated Bipolar both have fenestrated jaws, which is exactly
the kind of similarity that makes this hard, and asserting it confidently would
be the eighth over-claim of a day that has already produced seven.

What would settle it, in descending order of cost:

    1. one person who knows da Vinci instruments looking at case124_f0.png
       next to CADIERE_ONLY_case_016_6607.png
    2. sampling more frames across all 30 s of case124 -- the cadiere may
       enter frame at a moment the three sampled frames missed
    3. the same only-grasper-installed trick applied to Maryland Bipolar, to
       build the other half of the reference pair

### Why the lean matters even unconfirmed

If (a) holds, the largest single item of headroom in this project is a
logbook-versus-image mismatch rather than a model failure, and no perception
programme reaches it. Combined with today's other results -- temporal
modelling null, the threshold fix harmful, case132 outside the taxonomy -- the
honest reading is that **the perception path is closer to exhausted than the
0.1233 raw headroom suggests**, and that the remaining leverage is in answer
form and routing.

## 2026-08-15, 19:00 — my own discriminator, tested and REFUTED

The 18:45 lean rested on one visual claim: that the cream/white component on
case124's left-hand instrument marks it as a BIPOLAR, because bipolar
instruments carry insulation and the verified cadieres show bare metal.

Built the other half of the reference pair the same way -- windows where
**Maryland Bipolar is the ONLY grasper installed**, no cadiere, prograsp,
force bipolar, tip-up or retractor, for 3,000+ seconds:

    case_062  part1  mid 2178.8   3,175 s
    case_141  part2  mid 2638.1   3,142 s

`frames/MARYLAND_ONLY_*.png`. The verified Maryland Bipolar presents as a
**metallic grey wristed instrument with a long slender jaw** -- it does NOT
show the cream/white feature I attributed to it.

**So the discriminator is refuted and the 18:45 lean is withdrawn.** I cannot
say from these frames whether case124's cadiere is in view. Reading (a) --
unreachable -- and reading (b) -- present and misread -- remain equally open.

That is the eighth claim of mine today overturned by testing, and the only one
where I tested my own inference method rather than a conclusion drawn from it.
It was worth doing: the lean was about to become the basis for "the perception
path is exhausted", which is a large claim to rest on insulator colour.

### What the day leaves behind here

Two reference sets, both built from the LOGBOOK rather than from anyone's
judgement, covering the confusable pair at the centre of the largest headroom
item:

    frames/CADIERE_ONLY_case_{040,096,016}_*.png    cadiere is the only grasper
    frames/MARYLAND_ONLY_case_{062,141}_*.png       maryland is the only grasper

Anyone can now compare case124 against verified examples of both without
trusting a description. That is a better deliverable than the answer I was
reaching for, because it survives my being wrong -- which today it repeatedly
was.

**The question still needs one person who knows these instruments**, looking at
`frames/case124_f0.png` beside those two sets. It is thirty seconds of expert
attention, and it decides whether the largest single item of headroom in this
project is reachable at all.

## 2026-08-15, 19:10 — confusion vs visibility, settled at population scale

The 19:00 entry left case124 open because my visual discriminator failed. The
question -- is a missed cadiere CONFUSED with something or simply NOT VISIBLE
-- turns out to be answerable without looking at anything.

If confusion with another grasper were the mechanism, recall should collapse
when a rival grasper shares the scene. Split the 2,308 cadiere-installed
validation windows on exactly that:

    cadiere is the ONLY grasper installed  (n=1172)   detected 93.2%
    another grasper also installed          (n=1136)   detected 89.1%
                                                       difference +4.1 points

**Recall barely moves.** A competing grasper in the scene costs four points,
not the collapse a confusion account predicts.

### Three independent lines now agree

    no class substitutes above its base rate when cadiere is missed   (18:00)
    mean cadiere score on missed windows is 0.104, not a confident
      misassignment                                                   (18:00)
    recall is nearly unchanged by a rival grasper being present        (19:10)

All three say the same thing: **cadiere misses are VISIBILITY failures, not
classification errors.** The instrument is installed on an arm and not in the
camera's view, which is exactly what the labels being installation intervals
would produce -- and what the per-case concentration (case_133 50.6%,
case_058 0.0%) looks like when some cases simply keep an arm out of frame more.

### What it means for case124, stated with the right strength

The PRIOR has moved: a missed cadiere is usually a not-visible cadiere. That
favours reading (a) -- case124's cadiere is out of frame and its 0.0691 is
unreachable by any appearance model -- and it does so on population evidence
rather than on the insulator heuristic I refuted an hour ago.

It does NOT settle case124 individually. 91.2% of cadiere-installed windows ARE
detected, so most of the time the instrument is visible; the question is
whether case124 falls in the 8.8%. The visual check against the reference sets
is still what decides that one case.

But the framing for any future work is now different: **"improve cadiere
recall" is largely "detect an instrument that is not in the picture", and that
is not a modelling problem.** Anyone planning a tool-head retrain to chase
case124 should read this section first.

## 2026-08-15, 19:20 — the tool head's error is CASE-structured, not class-structured

The cadiere finding generalises. Recall per class, and how much it varies
across cases holding at least 20 windows of that class:

    class                            recall   best case  worst case  spread
    monopolar curved scissors         98.4%       100%        77%     23 pts
    vessel sealer                     95.3%       100%       100%      0 pts
    grasping retractor                95.2%       100%        77%     23 pts
    needle driver                     94.8%       100%        41%     59 pts
    bipolar forceps                   93.4%       100%        22%     78 pts
    permanent cautery hook/spatula    92.9%       100%        96%      4 pts
    cadiere forceps                   91.2%       100%        49%     51 pts
    force bipolar                     85.2%       100%         0%    100 pts
    prograsp forceps                  84.5%       100%        44%     56 pts

    mean per-case recall spread: 44 points

**Every class reaches 100% recall in its best case.** The model is demonstrably
capable of detecting every instrument in the taxonomy. What varies is the
CASE: bipolar goes 100% to 22%, force bipolar 100% to 0%, needle driver 100%
to 41%.

### Why that reframes perception work

A class the model cannot detect would show a low ceiling everywhere. These
show a perfect ceiling and a collapsing floor, which is the signature of
something about particular cases -- endoscope type, lighting, how much of the
procedure keeps an arm out of frame -- rather than of a representational
limit.

So **a better architecture is unlikely to be the lever.** The model already
achieves perfect recall wherever conditions permit. The gap is between good
cases and bad ones, and the 19:10 result says a large part of what makes a case
bad is instruments installed but not in view -- which no architecture reaches,
because the target is the logbook and the evidence is the picture.

### The caveat

"100% in the best case" is over cases with >=20 windows of that class, and a
case can be easy for uninteresting reasons -- one long stable interval with the
instrument centred. The spread is the robust part of this; the ceiling is
suggestive.

Two classes buck the pattern and are worth noting: vessel sealer (0 pts) and
permanent cautery hook (4 pts) are consistent everywhere, so whatever makes
other classes case-dependent does not apply to them.

### The recommendation this supports

Perception effort should go to **understanding case-level variation** --
which cases fail, what they share, and how much of it is visibility rather
than appearance -- before any retraining. `scripts/reference_frames.py` and
the per-case recall table above are the tools for that, and both are cheap.

Combined with today's other results (temporal null; the threshold fix harmful;
case132 outside the taxonomy; case124's headroom probably a
logbook-versus-image mismatch), the consistent reading is that **the remaining
leverage in this project is not in the perception models.**
