# SurgVU 2026, Category 2 (Surgical Visual Question Answering)

**OpScribe-AI. MICCAI 2026 EndoVis / SurgVU Category 2.**

This repository is the submitted system. It scored **0.9128** on the preliminary
phase and **0.6604** on the 101-case final phase, both BERTScore-F1.

Category 2 gives a model a **30-second robotic surgery clip and a free-text
question**, and asks for a free-text answer. Submissions are scored with
**BERTScore-F1** (roberta-large, `rescale_with_baseline=True`), taking the
**maximum over five independent human reference answers** and averaging across
cases. A one-word answer and a fluent sentence can score very differently against
the same reference, so answer *surface* matters as much as answer *content*, and
that fact shaped most of the design below.

---

## How it works

A clip and a question come in together, and the first thing that happens is that
they go their separate ways. The clip gets decoded into 16 frames and read by
five perception channels; the question, meanwhile, is sorted into one of thirteen
intents purely on its wording. What those two results meet at is an arbiter,
which decides, per intent, whether the answer should come from the perception
channels, from the vision-language model, or from something we already know about
the dataset and don't need to look at the clip to say.

That last category is smaller than the other two, but it's real, and we'd rather
name it than bury it.

### The five perception channels

| channel | what it produces |
|---|---|
| **Tool recogniser** | ResNet-50 at 384px, 12 classes, multi-label. Predicts which instruments are **installed** across the window rather than which are visible in any one frame. |
| **Task recogniser** | ResNet-50 at 384px, 8 classes, single-label. |
| **Detector** | Per-class instrument detections, each with a timestamp and a confidence. |
| **Motion** | Mean absolute inter-frame difference over a 64×64 greyscale reduction, measured at two time bases: 67 ms apart *within* a burst, and 1.9 s apart *across* the clip. |
| **Agreement** | How well the tool recogniser and the detector concur, which is a useful check on either one alone. |

Frame probabilities get averaged into clip-level probabilities, then cut with
per-class thresholds that were tuned against that exact averaging step. Change
the aggregation and those thresholds stop meaning anything, which is why the two
are documented together.

### Frame preparation, and why the UI band is blurred

Before any of that, frames are cropped to the endoscopic image and the on-screen
UI band is Gaussian-blurred.

The reason is distribution consistency. We blurred that band in training and
validation so the recognisers couldn't shortcut their way to an answer by reading
instrument names off the overlay. A model that learns to read the UI validates
beautifully and generalises to nothing. Having trained that way, we run the same
blur at inference so the frames at serving time look like the frames the model
was fitted on. The evaluation clips already arrive with the band obscured; doing
our own blur anyway just means there's one preparation path instead of two.
`preprocess.py` covers the text rows with room to spare: 6.25% of frame height is
what the measurement calls for, and we blur 8%.

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
