# V2 overnight report -- 2026-08-11 into 2026-08-12

Six experiments, plus one I invented mid-run. **One large positive, one small
positive, five negatives.** Most of the negatives were my own hypotheses,
which is the right ratio for a night of exploration but worth saying plainly.

## SUPERSEDED, 2026-08-12 morning -- read this first

The 20-epoch ResNet landed and **it beats the ensemble on its own**:

    EfficientNet shipped            tools 0.6747   task 0.8923
    ensemble (8-epoch ResNet)       tools 0.7171   task 0.9392
    ResNet-50 @ 20 epochs ALONE     tools 0.7802   task 0.9456
    EfficientNet + ResNet-long      tools 0.7690   task 0.9551

**+0.0631 on tools over the ensemble, from one model.** And ensembling it with
EfficientNet makes tools *worse* (0.7690 vs 0.7802) -- the weaker partner drags
it down. Measurable-class macro-F1 says the same: 0.8511 alone against the
ensemble's 0.7823.

So the overnight framing was wrong in an instructive way. **The ensemble was
compensating for an undertrained ResNet.** Diversity is real -- the seed control
proved it -- but it is worth less than training the better architecture
properly, and it turns *negative* once the partners are far apart in quality.

The task head disagrees: the ensemble still wins there (0.9551 vs 0.9456), so
the right configuration may differ per head.

## The one-line answer

**Experiment 4 (the second spotter) works: +0.0424 macro-F1.** Nothing else
moved the needle, and the ensemble's gain does not show up on the 11 held-out
cases at all -- for a reason we now understand and can measure.

## Every experiment

All numbers are **clip-level macro-F1 on the splits_v2 validation split**,
29 cases / 4,635 windows, with per-class thresholds tuned on one case fold and
scored on the other. That protocol is stricter than the one that produced the
shipped `0.6605`, so the two are not comparable -- see "Numbers that look
comparable and are not".

| # | Experiment | Verdict | Result |
|---|---|---|---|
| 1 | Aggregation (mean vs order statistics) | **POSITIVE** | +0.0142 (`top3`) |
| 1b | Per-class aggregator *(my idea)* | NEGATIVE | 0.6798 vs 0.6889 |
| 2 | Frame count 8 → 30 | NEGATIVE | 8 frames wins; flat |
| 3 | Test-time augmentation | NEGATIVE | every arm loses; hflip −0.0664 |
| 4 | **ResNet-50 second spotter + ensemble** | **POSITIVE** | **+0.0424** |
| 5 | Raise the `pos_weight` ceiling | NEGATIVE | −0.0614 on the CNN |
| 6 | EndoViT surgical foundation model | NEGATIVE *as tested* | 0.6627 vs 0.6889 |

### Reference points

    EfficientNet shipped   id / 16 frames / mean     0.6747
    EfficientNet best      id /  8 frames / top3     0.6889   +0.0142
    ResNet-50 alone        id / 16 frames / q90      0.6834   +0.0087
    EndoViT frozen + MLP   mean                      0.6627   −0.0120
    *** ENSEMBLE ***       id / 16 frames / top5     0.7171   +0.0424

---

## 4 -- The ensemble. The result of the night.

EfficientNet-V2-S and ResNet-50, probabilities averaged **per frame** before
aggregation, then reduced with `top5`.

    macro-F1 (honest, two-fold)     0.6747  ->  0.7171     +0.0424
    macro-F1 (measurable classes)   0.7361  ->  0.7823     +0.0462
    force bipolar                   0.3990  ->  0.7522     our worst class
    stapler                         0.5952  ->  0.6667
    runtime per case                  4.3s  ->    5.0s     of a 600s budget

The instructive detail: **ResNet-50 alone is *worse* than EfficientNet at clip
level** (0.6623 vs 0.6747) despite being *better* per-frame (0.6732 vs
0.6605). An ensemble partner does not have to be better -- it has to be wrong
differently. That is the whole "second spotter" idea, and it is the only thing
tonight that produced a large number.

ResNet also **had not converged at 8 epochs** (0.6301 → 0.6611 → 0.6732, still
climbing when the run ended), which is the opposite of EfficientNet -- that
peaks at epoch 3 and never beats it in a 12-epoch run. The shipped 4-epoch
default turns out to be an EfficientNet fact that had been treated as a
project-wide one. A 20-epoch ResNet is running.

### 4b -- Ensembling the TASK head is worth more, and I nearly skipped it

    single task model     0.8923
    task ensemble         0.9392      +0.047

A **43% cut in task error** (10.8% -> 6.1%), from a partner that was barely
better alone (0.6947 vs 0.6803) and *wildly* unstable across epochs, swinging
between 0.32 and 0.69. I had written that instability off as making the ResNet
task result untrustworthy. It made it a **better ensemble partner** -- an
unstable model is wrong in different places each time, which is exactly what
averaging exploits.

This should have mattered more end to end than the tools ensemble, because
`task_top` feeds the organ and task questions *directly*, with no coarse
presence test in between.

### But it changes nothing on the 11 held-out cases

Byte-identical answers to v1. Ensembling was confirmed active in every case's
log, so this is not a plumbing failure. The mechanism, measured by running
both configs through the identical harness and diffing the *records*:

    perception records differing     6 of 11
    task_top differing               0 of 11
    ANSWERS differing                0 of 11

The router absorbed all six changes, and you can see why case by case.
case125 asks whether a suture is required; both records contain a needle
driver, so both say Yes. case128 asks whether a needle driver is involved --
same. case129 is a procedure question answered from a **constant**, so
perception cannot reach it. case127 is answered from `task_top`, which did not
move.

**The general lesson, and it cuts both ways:** the router asks coarse
questions of a fine-grained record. That is why v1 scored 0.8015 on the
leaderboard while carrying a mediocre 0.6605 tool model -- it is robust to bad
perception. It is also why it does not cash in good perception.

The ensemble's gains sit in the **rare tail** (force bipolar, stapler) while
the 11 sample questions ask about needle driver, forceps and cadiere, where it
barely moves. Eleven questions cannot resolve a difference measured over
4,635 windows.

**Tested twice, both ways.** Re-running with *both* heads ensembled -- tools
+0.0424 and task +0.047 -- still gives byte-identical answers:

    task_top differing    1 of 11   (case126, uterine horn -> rectal artery/vein)
    ANSWERS differing     0 of 11

and even that one change is inert, because case126 is a needle-driver
*presence* question that never consults `task_top`. case127, the organ
question that does, holds the correct `uterine horn` under both.

**The sample set is structurally blind to these improvements.** That is now an
established property of the evaluation, not a one-off observation.

The three v1 failures survive untouched: case124 still answers Bipolar Forceps
against a gold of Cadiere, case126 still No against Yes, case132 still Yes
against No. Moving the sample score requires fixing the *tool* model on
cadiere-versus-bipolar and on a needle driver it never sees above 0.107 --
neither of which an ensemble of two models that share that blind spot can
do.

### The control: diversity is the mechanism, not averaging

I claimed architectural diversity was why the ensemble works, but never
isolated it. A second EfficientNet at a different seed lands at almost exactly
the ResNet's solo quality (0.6722 vs 0.6732 per-frame), so pairing each with
the shipped model separates architecture from partner strength.

    SOLO                             mean     top5     top3
      EfficientNet seed 7 (shipped) 0.6747   0.6847   0.6853
      EfficientNet seed 13          0.6573   0.6671   0.6719
      ResNet-50 (8 epochs)          0.6623   0.6759   0.6829

    PAIRS
      eff7 + eff13   SAME arch      0.6737   0.6707   0.6726
      eff7 + resnet  CROSS arch     0.7054   0.7171   0.7075

      cross - same                  +0.0317  +0.0463  +0.0348

**Two EfficientNets ensemble to 0.6707 -- below the single shipped model's
0.6847.** Averaging two models of the same architecture buys nothing at all;
it is slightly harmful. The entire gain comes from the partner being a
*different kind of model*.

Practical consequence: "train several seeds and average" is not a cheaper
route to this gain. Maintaining two architectures is what the result rests on.

### What does NOT further improve the ensemble

Two obvious knobs, both tested honestly, both flat or worse:

**Frame count.** Flat across 8-30, exactly as for a single model:

    frames        8       16       24       30
    mean     0.6983   0.7054   0.7021   0.7070
    top5     0.7133   0.7171   0.7165   0.7113
    q75      0.7154   0.7148   0.7209   0.7184

The highest cell (24/q75, 0.7209) is +0.0038 over the chosen 16/top5 -- inside
the +-0.008 noise band, and it is the maximum over a 4x3 grid scored on the
folds that report it. Not claimed.

**Member weighting.** Equal weight is arbitrary, so it was swept and selected
on the tuning fold only:

    fold A picks w_eff=0.25  ->  held-out 0.6856   (equal: 0.7329)
    fold B picks w_eff=0.60  ->  held-out 0.7007   (equal: 0.7012)
    honest weighted 0.6932   equal 0.7171   delta -0.0239

The folds disagree, and equal weight wins by a wide margin. **Equal weight is
the right default**, and this is the third swept parameter tonight killed by
the same fold-agreement check.

---

## 1 -- Aggregation. The shipped `mean` is near the bottom.

    top3      0.6889          trim20    0.6784
    top5      0.6847          q75       0.6762
    q90       0.6839          mean      0.6757   <- what we ship
                              noisy_or  0.6272

The training label is *window-level installation state*, so a tool genuinely
installed but visible in six frames of thirty is a positive the mean cannot
see and a top-k can. `noisy_or` is the textbook answer for "present in any
frame" and is the worst of the lot, because it saturates to 1.0 over thirty
frames and destroys the ordering the threshold needs.

**The best aggregator is not a constant of the problem.** EfficientNet wants
`top3`, the ensemble wants `top5`, EndoViT wants `mean` (0.6627 vs 0.6493
under `top3`). Anyone inheriting "top3 is best" onto a new model would lose to
it. It has to be re-measured per model, and the serving config now carries it
per expert for that reason.

## 1b -- Per-class aggregators. My idea, and it is fold noise.

The +0.0142 was 6 classes up and 5 down, and the split looked principled:
intermittently-visible instruments gained, continuously-present ones lost. So
I let each class pick its own aggregator, selected on the tuning fold only.

It scores **0.6798, worse than global `top3`'s 0.6889** -- and the diagnostic
is what makes that conclusive rather than merely disappointing: **only 2 of 12
classes chose the same aggregator on both folds**, and one of those is
unmeasurable anyway. The preferences are properties of the fold, not the
instrument.

One real exception survives: **force bipolar picks `q90` on both folds and
gains +0.073**. The worst class in the taxonomy has a genuine, reproducible
preference. Not enough to justify a per-class vector, but noted for v3.

## 2 -- Frame count. Your intuition, and it does not hold.

    frames    8       16      24      30
    mean    0.6746  0.6747  0.6757  0.6722

Flat, then slightly worse. The single best cell in a 216-cell sweep uses
**8 frames**. A 30-second clip does not contain 30 frames' worth of
independent evidence; the estimate is saturated well below 16.

So "we only use 22 seconds of 600, let's be generous" is right about the
headroom and wrong about where to spend it. The budget is better spent the way
experiment 4 spends it -- on a **second model** rather than more looks from the
same one.

**Hard limit:** shards store 30 frames per window, so this sweep cannot test
above 30. Serving decodes 60 fps video and could take more, but there is no
labelled way to measure it without re-extracting the corpus.

## 3 -- Test-time augmentation. Actively harmful.

    id                   0.6889          id+scale448          0.6726
    scale448             0.6611          id+hflip             0.6679
    hflip                0.6084          id+hflip+scale448    0.6718

Every augmentation combination loses to plain `id`, and **hflip costs
−0.0664**. Instrument *identity* is not chiral but arm *position* is -- in
robotic surgery the left and right arms systematically carry different
instruments, and mirroring destroys that association. This was flagged as "a
question rather than a freebie" when it was built; the answer is emphatically
negative.

## 5 -- The `pos_weight` ceiling. Wrong on two independent measurements.

The hypothesis: `train_tools.py` clips per-class weights at 50 while tip-up's
true negatives/positives ratio is ~175 and stapler's ~200, and tip-up scores
exactly 0.0000 -- never predicted at all.

- On **frozen features**, across 16 configurations, raising the ceiling
  50 → 200 changed macro-F1 by ≤0.008 and sometimes *favoured* 50.
- On the **real CNN**, the ceiling-200 retrain finished at **0.5991** against
  the shipped **0.6605** -- same architecture, same seed, same data, the only
  difference being the ceiling. **−0.0614.** Raising it is not neutral, it is
  actively harmful.

**A caveat that matters more than the result.** tip-up cannot be measured on
this validation split at all: all 68 of its validation windows come from a
**single case**, so no case-level fold can put it on both sides. Its F1 is a
structural zero in one direction regardless of the model. This verdict rests
on stapler and force bipolar, not on the class that motivated it.

And tip-up stays at 0.0000 even when thresholds are tuned self-tuned on *all*
68 windows -- so it is a genuine model failure, not a fold artifact. Neither
the ensemble nor the weighting change touches it.

## 6 -- EndoViT. Negative as tested, but the test was not the fair one.

A ViT-B/16 pretrained with masked autoencoding on 700k endoscopic frames from
nine public datasets, Apache-2.0. Loads into a timm ViT with **zero missing
and zero unexpected keys**.

    EndoViT frozen + MLP head    0.6627      (mean; 0.6493 under top3)
    EfficientNet best            0.6889

Frozen EndoViT loses by 0.0262. **But frozen-versus-fine-tuned is not a fair
fight** -- a completely frozen trunk landing within 0.026 of a fine-tuned CNN
on 115 training cases is a strong showing, and fine-tuning it is the version
that was not tested. That is the clearest v3 item on the list.

Supporting evidence that the features are real: a **nearest-centroid probe**
with no training at all, centroids from one case fold and scored on the other,
gives AUC 0.972 for permanent cautery hook, 0.921 prograsp, 0.889 clip
applier, and **0.850 for stapler from fourteen positive examples**.

### Two near-misses worth recording

This experiment could have produced a confident false null twice over.

1. **Normalisation is not ImageNet.** The checkpoint carries its own
   statistics -- `mean [0.3464, 0.2280, 0.2228]` -- which are endoscopy
   statistics, note the red channel. Feeding ImageNet's would have shifted
   every input off-distribution with no opportunity to adapt, because nothing
   is fine-tuned.
2. **The features need standardising.** EndoViT's CLS vectors sit in a narrow
   cone (mean pairwise cosine **0.995**), so a head fed raw vectors spends its
   early epochs subtracting a constant. Measured as an A/B at matched config:
   **standardised 0.6605 vs raw 0.6321, +0.0284.**

I found the cone property hours before I acted on it, ran the first sweep
without standardising, and got 0.6279 -- which would have read as "the
foundation model is mediocre". It was a preprocessing result wearing a model
result's clothes.

## 6b -- EndoViT as a third ensemble member. Also negative.

Tonight's main lesson is that a member has to be wrong *differently*, not
better -- ResNet is worse alone and still lifts the pair. EndoViT is a
transformer trained by masked autoencoding on endoscopic video, so it is about
as decorrelated from two ImageNet CNNs as anything available. Worth testing.

    2-way  EfficientNet + ResNet              0.7171
    3-way  + EndoViT, equal weight            0.7138     -0.0033

Equal weight is arbitrary, so the weight was swept -- and this is where the
experiment nearly produced a false positive:

    w      0.00    0.10    0.20    0.30    0.40    0.50    0.75    1.00
    top5 0.7171  0.7106  0.7139  0.7253  0.7187  0.7220  0.7129  0.7130

w=0.30 reads as **+0.0083**. But the curve oscillates by ±0.008 between
adjacent weights, and that cell is the *maximum over eight weights scored on
the same folds that report it* -- the identical selection-on-test trap that
killed experiment 1b.

Choosing the weight on one fold and evaluating on the other:

    fold A picks w=0.30  ->  held-out 0.7428   (w=0: 0.7329, +0.0099)
    fold B picks w=1.00  ->  held-out 0.6827   (w=0: 0.7012, -0.0186)

    honest weighted 3-way   0.7127
    honest 2-way (w=0)      0.7171
    delta                  -0.0044

**The two folds choose completely different weights (0.30 against 1.00), so
the choice is fold noise.** EndoViT does not earn a place in the ensemble at
any weight that can be selected reliably. The two-way ensemble stands.

---

## Numbers that look comparable and are not

- **0.6605 is per-frame; everything in this report is clip-level.** The
  shipped checkpoint records per-frame validation macro-F1. Every number here
  reduces frames to a clip first and tunes thresholds on a held-out case fold.
  The shipped configuration under *this* protocol is **0.6747**.
- **`toolsF1` includes tip-up's structural zero; `meas` excludes it.** Both
  are reported. The first is comparable to the shipped macro-F1, which also
  averages that zero in. The second is the one that moves when a model
  improves.
- **Self-tuned numbers are optimistic.** Every table reports the two-fold
  number and the self-tuned number side by side so the gap is visible. The
  shipped `+0.0196` serving-threshold gain was measured self-tuned.

## A method fix I had to make mid-run

The first fold split put **all 68 tip-up windows on one side and none on the
other**, and vessel sealer at 50/186. A class with no positives in a fold
cannot have a threshold tuned for it, so it scores a structural zero -- which
would have made experiment 5's improvements *invisible*, since the rare tail
is exactly what experiment 5 targets. I would have reported a confident null
about nothing.

The split now optimises balance directly (4000 seeded restarts; worst
per-class log-ratio **1.299 → 0.253**). Seeded, so runs stay comparable.

## Corrections to things I said earlier

- **"All three sample misses are model errors."** Only two are. case132 is the
  documented 17.6% tail of `large_needle_driver_policy` -- "Large Needle
  Driver" is a commercial *variant* collapsing into the `needle driver` class,
  unrecoverable from a 12-class taxonomy. No perception work fixes it.
- **"Idle jobs mean a bad resource request."** They meant group GPU
  contention. Dropping memory 64→16GB moved the match count 177→181, and
  lowering the GPU-memory bar only raised "rejected by their own
  requirements" 98→109. Both hypotheses wrong; `condor_q -better-analyze`
  gave the real answer.
- **The ResNet *task* result is not solid.** 0.6947 against the shipped
  0.6803, but the run oscillates between 0.32 and 0.69 across epochs. The
  gain is within its own run-to-run variance. The ResNet *tools* result does
  not have this problem.

## What is still running

- `tools_posw200` -- experiment 5's CNN confirmation, trending negative
- `tools_resnet50_long` -- 20 epochs, since 8 had not converged
- `frame_probs_resnet_both_val` -- for a both-heads ensemble

## The decision for you

Experiment 4 is a **real perception gain with no end-to-end confirmation
available from the sample set**. Shipping it is a bet that the test set asks
about rarer instruments than these 11 questions do.

**For shipping it:** +0.0424 macro-F1 over 4,635 windows is not noise; the
runtime cost is +0.7s against a 600s budget; the serving path is verified
bit-identical on the default config, so v1 is not at risk; and the largest
gains are on the classes v1 is worst at.

**Against:** zero measured change on every case we can actually score, six
perception changes that the router absorbed, and seven submissions left. A
submission spent here buys information about a change we cannot otherwise
observe -- which may itself be the reason to spend it.

`config/perception.json` is untouched. The ensemble lives in
`config/perception_ensemble.json`, so nothing about the submitted v1 has
changed.
