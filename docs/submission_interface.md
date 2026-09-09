# Submission interface (Grand Challenge algorithm container)

Source: the challenge's algorithm interface definition. Recorded here because
every container decision depends on it and it previously existed only in a
conversation.

## Contract

| Direction | Path | Kind | Slug |
|---|---|---|---|
| read | `/input/endoscopic-robotic-surgery-video.mp4` | MP4 file | `endoscopic-robotic-surgery-video` |
| read | `/input/visual-context-question.json` | String | `visual-context-question` |
| write | `/output/visual-context-response.json` | String | `visual-context-response` |

`endoscopic-robotic-surgery-video` is described as "a sequence of endoscopic
images taken from robotic surgical training".

### The output is a JSON-encoded string, not raw text

The example value is `"Example String"` -- quoted. So a "Yes" answer is the
four bytes `"Yes"` including the quotation marks, which is what
`json.dump("Yes", f)` produces. Writing bare `Yes` is malformed JSON and would
fail regardless of whether the answer is correct.

The same applies on the way in: `visual-context-question.json` holds a JSON
string, so it needs `json.load`, not a raw read. This matches the public
sample set, where `caseNNN_question.json` contains `"Are there forceps being
used here?"` with the quotes.

## What the singular paths imply

One video, one question, one answer per invocation. That reads as **one case
per container run**, which means any model-loading cost is paid per case
rather than amortised across the test set. This is the difference between a
VLM being viable and being disqualified: Qwen3-VL-8B took 668 s to load from
/staging (cluster 9623506). Loading from inside the image will be far faster
than streaming 17.5 GB over ceph, but the per-job time limit decides it.

NOT YET CONFIRMED -- see open questions. If the harness instead loops over
cases inside a single run, the arithmetic changes completely.

## CONFIRMED constraints

From the challenge configuration and the organizers' announcement:

- **10 minutes total run time.** One invocation, one case.
- **32 GB main memory (DRAM) maximum.**
- **Instance is either No GPU, or one NVIDIA T4 (16 GiB VRAM).** T4 is sm_75:
  no bf16, no FlashAttention-2. Everything must run in fp16 or int4 with an
  sdpa/eager attention implementation.
- **One input**: one endoscopic robotic surgery video plus one visual-context
  question. Confirms one case per invocation -- model load is NOT amortised.
- **No internet access once submitted.** Every weight ships inside the image.
- **Ranking metric is BERTScore-F1 with roberta-large**, replacing last year's
  BLEU. `src/surgvu/scoring.py` already reproduces this, so it is the right
  target. It is computed on the organizers' side; we do not ship roberta-large.
- **UI blurring as currently implemented is acceptable.** No change needed to
  `preprocess.blur_ui_band` for the bottom instrument bar.

  **CORRECTED 2026-08-11.** This bullet previously read "the untouched top
  banner is not a problem" on the basis that the banner is constant text. **That
  basis is false and the record is corrected here.** The banner was measured
  (`docs/compliance_audit.md` §2c) by normalised cross-correlation against a
  banner template: it is **present in 5 of the 11 sample cases and absent in 6**,
  and present in **14 of 45 sampled training cases (31%)** in the 512x512 shard
  domain -- that is, it reaches both CNNs unblurred and varies from case to case.

  What was actually measured, and is the real basis for accepting it: banner
  presence is **near-independent of the task label**. Across the same 45-case
  sample it runs **29-43% in every task class with meaningful support, against a
  31% base rate** (the two 0% classes have n=1 and n=2). It is also a single bit,
  so it cannot identify a case; at most it is a weak nuisance variable.

  **Decision: ship as-is, with the record corrected.** Extending the blur to a
  top band is a small edit to `preprocess.py` but invalidates every extracted
  shard and both shipped checkpoints -- a full re-extract and retrain of both
  CNNs, a multi-day cost against a signal measured at near-zero. The residual
  exposure is a train/test distribution question rather than a rules one: if the
  organizers blur the top region in the test set, the model sees a top-of-frame
  distribution it was not trained on. See `docs/compliance_audit.md` §2c for the
  full measurement.
- **Model provenance is fine for an off-the-shelf model.** Qwen3-VL is
  Apache-2.0 and we did not train it, so no training-data disclosure applies.

### What the 16 GiB VRAM ceiling means for the VLM

    Qwen3-VL-8B  fp16   17.5 GB  -> does NOT fit
    Qwen3-VL-8B  NF4    ~5.5 GB  -> fits
    Qwen3-VL-4B  fp16    8.9 GB  -> fits natively, no quantization
    CNNs (both)         165 MB   -> trivial

**Quantize offline and bake the NF4 weights into the image.** Quantizing at
load time means reading the full 17.5 GB fp16 into system RAM and converting,
which against a 32 GB DRAM cap is tight and slow, inside a 10-minute budget.

The 668 s load measured in cluster 9623506 is NOT the relevant figure -- that
was streaming 17.5 GB over ceph at roughly 26 MB/s. From local disk inside the
image, 5.5 GB should be tens of seconds. That estimate is the one number still
worth proving, on an sm_75 card (CHTC's RTX 2080 Ti, 11 GB, hosts 8B-NF4 at
~5.5 GB and 4B-fp16 at 8.9 GB, so both deployment paths are testable).

Rough budget, leaving ~7.5 min of headroom:

    container start + imports        ~30 s
    CNN load + decode + inference    ~30 s   (measured: 16 frames, ~20 s CPU)
    VLM load (NF4, local disk)       ~60 s   <- needs proving
    VLM generate (short answer)      ~30 s   (2.4 s on H200; T4 is slower)

## Still open

1. **Answer length constraints**, if any.
2. **Number of test cases** (does not affect the per-case budget, but informs
   how much total compute the organizers will spend).
3. **Does the 5-reference structure hold at test time?** The answer form is
   tuned to the observed fact that the first reference is a bare token
   ("Yes", "Cadiere Forceps"), which scores 1.0000 while correct prose
   scores lower -- in **9 of the 11** samples, not all 11: case129's first
   reference is a six-word noun phrase and case130's is a full sentence.

   MEASURED 2026-08-11, cluster 9636318, see `docs/NIGHTLY_REPORT.md`. If the
   terse reference disappears the terse form falls **0.8766 -> 0.3355**,
   below the generic constant sentence (0.4358); a terse-plus-clause form
   holds 0.7238 -> 0.7184. Break-even is p = 0.715 on P(a terse reference
   exists). Terse is kept because all 11 samples come from one generator with
   an identical five-reference structure, so p is high -- but it is a bet with
   a priced tail, not a free choice.
