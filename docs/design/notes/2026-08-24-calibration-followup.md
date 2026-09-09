# Motion calibration follow-up (ruling R17) -- do this before trusting config/motion_v2.json

Task 4 shipped the calibration machinery and it works end to end: a 2-case smoke run through
`condor/dump_motion_v2.sub` produced 40 anchors with all nine motion slots populated from real
60fps surgical video, and `scripts/calibrate_motion_v2.py` fitted cuts against the tasks.csv
activity proxy. **The machinery is sound. The reported numbers are not yet trustworthy**, for two
reasons that are defects in the calibration DESIGN, not in the code.

## What the smoke run reported

    micro_short          auc=0.9825   micro_mid   auc=0.9675   micro_long  auc=0.9925
    macro_prev           auc=0.9753   macro_next  auc=0.9753
    flow_mag_mean        auc=0.9875   flow_mag_p90 auc=0.9775
    flow_coherence       auc=0.5075   flow_moving_fraction auc=0.9575

One result here is solid and worth keeping: **flow_coherence measured at chance (0.5075)** while
flow_moving_fraction reached 0.9575. That independently confirms ruling R7 on real video, using a
different objective from the synthetic frames that motivated it (which said 1.33x vs 12x).
Demoting coherence to a secondary signal was correct.

Everything else in that table is suspect.

## Defect 1 -- autocorrelation inflates every AUC

The producer draws a batch of anchors from within ONE span, spaced ~0.8s apart. Measured on the
smoke dump, the idle anchors sat at t = 457, 458, 459, 460, 461, 461, 462, 463, 464, 465 and then
5704, 5705. That is roughly 10 near-duplicate frames inside a single 8-second span.

So "40 anchors" is about **8 independent spans**, and an AUC computed per-anchor treats
near-duplicate neighbours as independent evidence. An AUC of 0.99 over 8 effective samples is not
a generalisation estimate.

**Fix:** draw fewer anchors per span and more spans (a batch of 2-3, not ~10), and/or aggregate to
one value per span before computing AUC. The plan said `--windows-per-case 20` and never said
those windows had to be independent -- that omission is the defect.

## Defect 2 -- the idle class may be trivially separable

Split by label on the smoke dump:

    ACTIVE  micro_short  min 3.706  median 11.447  max 28.686   flow_mag_mean median 1.3956
    IDLE    micro_short  min 0.000  median  1.019  max  8.316   flow_mag_mean median 0.0603
    IDLE had 10 of 20 anchors below micro_short 0.5 -- essentially frozen frames.

If the gaps between annotated tasks are camera-out or paused segments, the fitted cut is detecting
**"is the camera in the body"**, not "is surgery happening". The router question this threshold is
meant to serve ("is tissue being cut?") is only ever asked of clips where the camera is
definitionally in -- so a cut fitted on that distinction would not transfer.

This is checkable: `tools.csv` carries `nan(camera in)` 1277 times, so camera events are recorded.

**Fix:** before the full sweep, confirm what the idle windows actually contain. If they are largely
camera-out, either restrict idle sampling to camera-in segments, or accept the threshold only for
the narrower question it actually answers and say so in the config's `objective` field.

## Why this matters more than it looks

`scripts/sample_motion.py` exists in this repo precisely to price a motion rule rather than estimate
it -- its docstring says the eleven graded sample cases are "the only place the rule can be PRICED
rather than estimated". The same discipline has to apply to its successor. Shipping
`config/motion_v2.json` with a threshold fitted on "camera present" and validated by an AUC inflated
with near-duplicate frames would produce a number that looks like strong evidence and is neither.

The full 155-case sweep should not be run until defect 1 is fixed; it would just produce a
better-looking version of the same unreliable number.
