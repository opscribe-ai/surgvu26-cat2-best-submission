# OpScribe-AI's SurgVU 2026 Category 2 Surgical Visual Question Answering (Submission)

**MICCAI 2026 EndoVis / SurgVU Category 2.**

This repository is the company's best submitted system. It scored **0.9128** on the
preliminary phase via BERTScore-F1.

Category 2 involves giving a pipeline a 30-second surgical clip along with a free-text
question, and then the pipeline will output a free-text answer. Submissions are scored
with BERTScore-F1, taking the best answer graded across five independent human
reference answers.

---

## How it works

As soon as a clip gets ingested into our pipeline, it gets decoded into 16 frames, and
five measurements are taken from those frames. The question, separately, is sorted into one of
thirteen question types, based on the words and the actual string of the question.

The end of the pipeline is concluded by a judge, which decides, based on the question
type, whether the answer should come from the five measurements, from the vision-language
model, or from a fact we already know about the dataset and don't need to look at the
clip to say (ex; if a question is "what does x instrument do?", the answer to this will be the same every time).

That last category only happens in niche scenarios; most questions are answered by the five measurements in this pipeline. 

### The five measurements

| measurement | what it is |
|---|---|
| **Tool recogniser** | A ResNet-50 that tries to identify what instruments are in the window. It outputs all 12 instrument classes, each with its own confidence score, so more than one instrument can be identified as being in the frame. |
| **Task recogniser** | A ResNet-50 that says what surgical step is happening, for example suturing or retraction. Unlike the tool recogniser, the task recogniser outputs just one answer, with eight possible activity classes. |
| **Detector** | Uses YOLO to draw boxes around instruments in the individual frames. The tool recogniser tells you what instruments are probably there; the detector is a second reference for what is there, and it also tells you where they are, with confidence scores. |
| **Motion measurement** | Computes a micro and a macro score. The micro score tells you how much the picture changed between frames 67 milliseconds apart from the target frame, to see if something is moving on a small time scale. The macro score measures the movement from frame to frame across the original 16 frames we took from the 30-second clip, to see if there is larger-scale change. |
| **Agreement measurement** | Deterministic code that checks whether the tool recogniser and the detector named the same instrument. This gives the pipeline more confidence when they agree, and flags things when they disagree. |

Step and instrument probabilities are taken into account to produce probabilities at
the clip level. Those are then compared to thresholds that decide whether a given tool
or step gets approved or denied as actually being in the clip.

### Frame preparation

Before any of the measurements happen, the frames are cropped to our desired size, and
the bottom 8% of the image is Gaussian blurred.

The reason we did this is for consistency. We blurred the bottom 8% in training and
validation so the recognisers couldn't cherry-pick their way to an answer by reading the
UI that the da Vinci robotic system displays at the bottom of the videos. We understand
the UI is blurred for the final, but we wanted to make the frames as similar as possible
to what our pipeline was trained on.

`preprocess.py` is what changes the frame height and does the blurring.

### The vision-language model

```
Qwen2.5-VL-7B, served NF4-quantised (bitsandbytes), 16 frames per call

  Qwen/Qwen2.5-VL-7B-Instruct              base instruction-tuned VLM
    └─ nvidia/Qwen2.5-VL-7B-Surg-CholecT50   surgical continuation, CholecT50
         └─ ~87k-pair GI corpus                abdominal / GI surgery QA
              └─ SurgVU 16-frame QA              this challenge's own clips

  LoRA  r=32, alpha=64  ·  95.2M of 4.79B parameters trained (1.99%)
  Prompt is evidence-conditioned: the model sees the frames AND a rendered
  summary of what all five perception channels found.
```

Two stages of fine-tuning sit on top of a base that had already been adapted for
surgery, so the model arrives knowing what a cadiere forceps looks like. The
adapter touches the language layers and the vision MLPs; the patch embedder and
the positional encoding are left alone.

### The router's thirteen intents

```
OPEN                              POLAR
  tool_identity_open                tool_presence_polar
  task_open                         cutting_polar
  organ_open                        suture_polar
  count_open                        task_confirmation_polar
  procedure_open                    approach_polar
  purpose_open
                                  FALLBACK
                                    unknown_open
                                    unknown_polar
```

`classify_question()` picks one of these from the question text alone: regexes
and keyword rules, no model, no perception. Rule order is the design: COUNT is
tested before the tool rules because "how many instruments" contains the word
*instruments*; ORGAN before PROCEDURE because "what organ is manipulated in this
procedure" contains *procedure*. Polarity gets checked before any open rule,
since "is this laparoscopic?" wants a *Yes*, not the name of a procedure.

### Three ways an answer gets made

Walk three questions through the same pipeline and you get three different
machines doing the work.

**One: the perception channels decide.** Ask *"Is tissue being cut in this
clip?"* and it classifies as `cutting_polar`. The tool recogniser is checked for
a credible cutting instrument, and if there isn't one the answer is No. If there
is, motion gets consulted before committing, because an earlier version of this
answered Yes to a pair of scissors sitting perfectly still in frame. That was a presence
signal standing in for an event, and no amount of improving the tool recogniser
would have fixed. So two channels have to agree: something that can cut is
installed, *and* the scene isn't static. Most intents work roughly like this,
though usually with one channel rather than two.

**Two: the VLM decides.** Ask *"What instrument is the surgeon using?"* and it
classifies as `tool_identity_open`, the one intent where the model's answer ships
instead of the router's. That's a measured choice, not a hedge. Three of the
twelve instrument classes share the head noun *forceps*, and when the classifier's
top two candidates are close its top-1 is barely better than a coin flip, while
the correct answer sits in that top two about 91% of the time. A classifier can't
exploit that, because naming both scores worse than naming one. A model that can
actually look at the frames can. The VLM still receives everything the perception
channels found; it just gets the final word here.

**Three: we already know the answer.** Ask *"What is the purpose of using
forceps?"* and no amount of staring at the clip helps, because that's a question
about what forceps are *for*. It resolves through a lookup keyed on the instrument
the question itself names. Two other intents work the same way: every clip in this
corpus is robotic endoscopic dry-lab surgery, so `procedure_open` returns a
constant and `approach_polar` answers from the question's own wording. Three
intents in total, and on the eleven-case public sample they accounted for 2 of 11
questions. A per-clip guess could only have been worse than a known fact.

Anything the classifier can't place at all falls to `unknown_open` or
`unknown_polar`, and both go to the VLM, because the router has no form for those
questions, so its "answer" would be a generic string written without reference to
what was asked.

---

## Measured results

| | value | notes |
|---|---|---|
| **Final phase (101 cases)** | **0.6604** | BERTScore-F1, max over five references. |
| Preliminary phase | 0.9128 | A different, smaller question set, not comparable to the final. |
| Tool recogniser | 0.7802 macro-F1 | Honest two-fold over validation windows; the number to quote. |
| Task recogniser | 0.9348 accuracy / 0.7920 macro-F1 | |

`tip-up fenestrated grasper` scores 0.0 F1. It is genuinely absent from the
training distribution, and the threshold is pinned at the floor rather than
tuned. It is reported rather than hidden.

---

## Reproducing the container

The evaluation artifact is a single container. Weights are **not** in this
repository; they are fetched into the build context.

```bash
# Apptainer (what was used)
containers/build_submission.sh
apptainer build surgvu26-submission.sif containers/surgvu26-submission.def

# Docker (Grand Challenge upload format)
docker build -f containers/Dockerfile -t surgvu26-cat2 .
docker save surgvu26-cat2 | gzip > surgvu26-cat2.tar.gz
```

Base image is `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`; the VLM layer pins
`transformers==4.57.6`, `accelerate==1.14.0`, `bitsandbytes==0.50.1`. Those pins
are load-bearing, and `containers/surgvu26-submission.def` records why each one is
what it is. `docs/container_build.md` covers the build in full, and
`docs/submission_interface.md` documents the Grand Challenge I/O contract.

Runs on a single 14.6 GiB T4 within the 600 s per-case wall clock, with no
network access at inference.

---

## Repository layout

```
src/surgvu/     pipeline: perception, router, arbiter, VLM, frame planning
scripts/        inference entrypoint, training, evaluation, tuning
config/         model config, serving thresholds, arbiter policy, splits
containers/     Dockerfile, Apptainer definition, build scripts
condor/         HTCondor job files, how every run was actually executed
tests/          test suite
docs/           design notes, build guide, compliance audit, version history
```

- **`docs/VERSIONS.md`**: what each version changed and what it scored, including
  the ones that lost points.
- **`docs/compliance_audit.md`**: pre-submission audit covering data segregation, UI
  blur, split discipline, weight provenance, licensing, secrets.
- **`docs/design/`**: the design plans and measurement notes the build followed.

---

## Data and licensing

The SurgVU 2026 dataset is **not** redistributed here, and no challenge gold
reference answers are included. Access the data through the
[challenge organizers](https://surgvu26.grand-challenge.org/).

Code is Apache-2.0 (`LICENSE`). Third-party components and their licences are
listed in `NOTICE`. Model weights are released separately; see the challenge
report for the link.
