# Version history

What each version changed, and what it measured. The negative results are
included deliberately — three of the last four changes cost points, and the
reasons are more useful than the wins.

**Two scoreboards, never comparable.** The preliminary phase scored 11 cases
whose question set we could see; the final scored 101 unseen cases. Preliminary
0.9128 became final 0.6604 for the *same container*. Preliminary numbers are
useful only against other preliminary numbers.

| version | phase | score | what changed |
|---|---|---|---|
| v2 | prelim | 0.8015 | Router + two perception heads. No VLM on the answer path. |
| v5.2 | prelim | 0.8558 | Instrument variant head (+0.0543, leaderboard-confirmed). A `challenger` arbiter mode was tried and measured **−0.0784**; it was removed. |
| v6 / v6.1 | — | not separately scored | Three-stage VLM curriculum: NVIDIA surgical base → GI corpus → SurgVU 16-frame QA. The VLM was in the container but, as v6.2 later revealed, **never actually ran on the grader**. |
| **v6.2** | **prelim 0.9128 · final 0.660400** | **the submitted system** | Added the missing VRAM term to VLM frame planning. See below. |
| v7 | final 0.660156 | **−0.000244** | Gave the VLM first crack ahead of the router on more intents, with router re-wording. Net zero, and slightly negative. |
| v8 | final ≈ 0.60 | **−0.06** | Full VLM-first redesign: per-intent answer normalisation, five sampled passes with phase jitter, per-intent confidence gates, a much richer evidence prompt. Scored **+0.033 on our internal benchmark and −0.06 on the real test set.** |

## v6.2 — the version in this repository

`select_plan` chose a VLM frame plan on **time budget alone**. There was no VRAM
term at all, so a 420 s budget always reached for the richest plan that fit the
clock — on every card, forever. On the grader's 14.6 GiB Tesla T4 that prefill
OOM'd, `try_vlm_result` caught the exception, and the router answered instead.

The failure was silent and total: **every graded case, through several
submissions.** An OOM and a confident router answer produced byte-identical
output, so nothing in the logs distinguished them.

The fix is a measured token ceiling (`MATH_KERNEL_TOKEN_CEILING = 3072` for
sub-16 GiB cards) plus an OOM step-down. It is a small diff and it is the
difference between shipping a VLM and shipping a VLM-shaped hole.

## Why v7 and v8 lost

Both rest on the same assumption — that the VLM should answer more questions —
and both were checked against an internal 199-item benchmark that **agreed with
them**. v8 scored 0.9418 there against v7's 0.9088, with instrument identity
rising 0.6585 → 0.8675. It then lost about 0.06 on the real set.

The benchmark's references were synthesised from the same operative logbook the
task recogniser was trained on. So it partly measured *"does the model agree with
the classifier"* — which rewards exactly the router-led behaviour v8 was designed
to replace, and inverts the sign of the change it was built to evaluate.

The lesson we would carry forward: **a benchmark built from your own labels
cannot adjudicate a change to who does the labelling.** The intent where the
benchmark was least circular (instrument identity, where a model can genuinely
change the answer by looking) was also the only intent where its verdict looked
plausible.

## What we would do differently

1. **Stamp a git SHA into the container at build time.** The submitted image
   carries no build provenance, so tying an artifact to its source is
   reconstruction rather than lookup.
2. **Hold out a scoring set the perception models never touched**, rather than
   synthesising references from their training labels.
3. **Log an unmissable reason code on every path** where the router answers
   instead of the VLM. Distinguishing "declined" from "crashed" is what would
   have caught the v6.1 OOM immediately instead of several submissions later.
