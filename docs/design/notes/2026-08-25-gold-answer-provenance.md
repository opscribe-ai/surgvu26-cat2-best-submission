# Gold answers do not track visual presence -- measured 2026-08-25

Derived from the first-ever run of the trained detector over the eleven graded sample cases
(cluster 9684659) compared against gold. This refines ruling R24 and should be read before
any further perception work is scoped.

## The evidence

| case | question | gold | on screen (detector max conf) |
|---|---|---|---|
| 122 | "Are there forceps being **used** here?" | **No** | bipolar forceps 0.814, cadiere 0.322 |
| 124 | "What type of forceps is **mentioned**?" | cadiere forceps | bipolar 0.895, **cadiere 0.000** |
| 126 | "Was a **large** needle driver used in this clip?" | Yes | needle driver 0.466 |
| 128 | "Is a needle driver **involved in the procedure**?" | Yes | needle driver 0.721 |
| 132 | "Was a **large** needle driver **used during the surgery**?" | **No** | needle driver 0.865 |

Forceps are clearly visible in case122 and gold is No. A needle driver is clearly visible in
case132 and gold is No. Cadiere is entirely absent from case124 and gold is cadiere.

**A better visual detector cannot fix these by seeing harder.** Two independently-trained models
(our CNN heads and a YOLOv5 detector) agree with each other and disagree with gold.

## Three distinct mechanisms, not one perception problem

Reading the question wording against the gold answers, the failures separate cleanly:

**(a) Size/family -- case126, case132.** Both clips show a needle driver; only "is it *Large*" separates
Yes from No. Addressed by the variant head (Task 10) on the free logbook labels from Task 9.
This is the one that is squarely a perception problem, and it is already the plan's top priority.

**(b) List membership -- case123 "among the **listed** tools", case124 "is **mentioned**".** These ask about a
tool LIST, not about image content. The list is rendered in the bottom UI band, which the challenge
PROHIBITS reading. If gold derives from that list, these questions may be structurally underivable
from the pixels we are allowed to use. We currently score case123 correctly and case124 wrongly,
which is consistent with guessing.

**(c) Activity, not presence -- case122 "being **used**".** A forceps idle in frame is visible but arguably
not "being used". Gold says No while forceps are visible at 0.814. This is exactly the distinction
`src/surgvu/motion.py` was written for: its own docstring says the router answers event questions
with a proxy for PRESENCE, and "the tool head is right, it is being asked the wrong question."

## What this changes

1. **Perception has a lower ceiling on this sample than the plan assumed.** The plan's headline claim
   was that all three remaining failures are tool perception. Mechanism (b) is not perception at all,
   and (c) is motion rather than identification.
2. **The variant head keeps its priority** -- mechanism (a) is real, visual, and covers two of the three.
3. **Motion (W1) gains value it was not credited with.** case122 is currently answered correctly, but
   by a route that does not reason about activity; the same question phrased the other way would be
   answered wrong. Motion evidence is the principled fix.
4. **Do not spend further effort trying to make a detector fix case124.** It is not a detection miss --
   the detector sees zero cadiere with 0.81 class recall. Spending on (b) means finding a legitimate
   non-UI route to list membership, or accepting the loss.

## Caveats, stated honestly

Eleven cases is a small sample and this is a hypothesis fitted to it, not a proven model of how the
organisers generated gold. The alternative reading -- that these clips genuinely contain different
tools than the detector reports -- is not fully excluded, though a 0.000 detection from a class with
0.81 recall argues against it for case124. The leaderboard set is larger and may weight these
mechanisms differently.

The original design document already carried a section titled "Provenance of the question evidence --
read this before trusting any of it". That warning was right and was under-weighted when Plan 1 was
scoped.
