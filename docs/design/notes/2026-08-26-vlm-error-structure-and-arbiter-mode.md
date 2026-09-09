# The fine-tuned VLM's error structure, and what it says about arbiter mode

Source: `/staging/n/nkalthoff/surgvu26/models/vlm_lora/eval_report.json` -- the held-out CASE
eval of the shipped adapter (cluster 9697941), n=300 QA pairs over 285 distinct held-out
cases, `bertscore_f1 = 0.909223216228808`.

That report carries only `bertscore_f1` and `case_id` per item -- no question, no answer,
no intent. Everything below is inferred from the SHAPE of the score distribution against
this project's known score economics, and the inference is bounded accordingly.

## The distribution is not a spread. It is two spikes.

    ~1.000 (exact)                     251   83.7%
    0.90-0.999                          20    6.7%
    0.75-0.90                            2    0.7%
    0.65-0.75  (polar wrong ~0.7015)     0    0.0%
    0.40-0.65                            1    0.3%
    <0.40      (open wrong ~0.2402)     26    8.7%

83.7% exact. The entire 0.0908 of headroom is 26 items sitting at the open-ended
wrong-noun floor. There is no long tail of near-misses to grind down: there is one
failure mode, and it is naming the wrong noun on an open-ended question.

## The empty 0.65-0.75 bucket is the finding

A wrong polar answer scores ~0.7015 (measured, this project). That bucket holds ZERO of
300.

`qa_pairs.jsonl` is 377,557 pairs, **47.1% of them polar** (answer exactly yes/no), so a
uniformly drawn 300 should carry roughly 141 polar questions. Zero landed at the polar-
wrong floor.

**Ruling out the boring explanation.** Polar is NOT so skewed that a constant guess
explains this. The corpus is 56.7% yes / 43.3% no; a constant-"Yes" guesser scores 0.8707
on the polar slice and would put ~61 of ~141 items squarely in the empty bucket. Per
intent, only `tool_presence_polar` is meaningfully skewed (71.3/28.7); `cutting_polar`
(51.8/48.2), `suture_polar` (41.6/58.4), `task_confirmation_polar` (50.1/49.9) and
`variant_presence_polar` (49.8/50.2) are near-balanced. The model is genuinely saying
"No" when "No" is right.

**Bounding it honestly.** A wrong polar answer need not land at exactly 0.7015 under MAX-
over-five-references, so some could hide in the 0.75-0.90 (2 items) or 0.40-0.65 (1 item)
buckets. The strongest defensible claim is therefore: **at most ~3 of ~141 polar questions
wrong, i.e. >=98% polar accuracy on held-out cases.** Not "perfect".

## What this argues about `config/arbiter.json: mode`

Set against the other half of the system: the router needed an entire new perception stack
-- a YOLO detector plus a trained variant head, with a four-condition answer gate -- to fix
**two polar answers** (case126, case132). That was worth +0.0543 and it was the right work.
But it is a lot of machinery aimed at the exact slice the VLM already answers at >=98%.

Meanwhile every one of the VLM's 26 failures is an open-ended noun, and `router.py` records
a wrong noun scoring as low as **-0.086** -- strictly worse than not touching the answer.

So the evidence points at an intent-conditional policy: **take the VLM on polar, never let
it touch the nouns.** That policy is already implemented -- it is `MODE_PRIMARY`
(`_arbitrate_primary`), which reads the VLM's polarity on `router.is_polar_question` and
returns `router_answer` unchanged for every non-polar intent. Its own docstring calls this
"a deliberately conservative reading of a mode whose name suggests the opposite", and the
distribution above is the first measured support for it.

`challenger` (the shipped default) differs in that it gates every override behind VLM
confidence. That is safer per question and, if the >=98% figure holds, leaves polar wins on
the table.

## Do not act on this yet

Three reasons this is a note and not a config edit:

1. Held-out CORPUS cases are not the graded distribution. The graded sample is 11 cases
   and R24 established its gold answers key on things (UI-band list membership, "being
   used" vs present) the corpus QA generator does not model.
2. `primary` takes the VLM's polarity with NO confidence gate in that path. The 98% is a
   sample estimate over ~141 items, not a calibration.
3. Nothing has yet run the merged+quantised model end to end. NF4 is not fp16, and the
   eval above was fp16 + LoRA. Quantisation error has to be measured, not assumed away.

**The test to run** once the NF4 checkpoint is in the image: `scripts/flag_matrix.py` over
the graded sample with `mode=challenger` and `mode=primary`, plus a re-run of the held-out
300 through the QUANTISED model to confirm the polar accuracy survives NF4. Decide on those
numbers, not on this note.
