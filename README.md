# SurgVU 2026 — Category 2 (Surgical Visual Question Answering)

**OpScribe-AI · 3rd place team, MICCAI 2026 EndoVis / SurgVU Category 2.**

This repository is the submitted system that scored **0.6604 BERTScore-F1** on the
101-case final test set — third of nineteen teams, behind 0.6873 and 0.6871.

Category 2 gives a model a **30-second robotic surgery clip and a free-text
question**, and asks for a free-text answer. Submissions are scored with
**BERTScore-F1** (roberta-large, `rescale_with_baseline=True`), taking the
**maximum over five independent human reference answers** and averaging across
cases. A one-word answer and a fluent sentence can score very differently against
the same reference, so answer *surface* matters as much as answer *content* — a
fact that shaped most of the design below.

---

## How it works

The clip is decoded once into 16 frames at 512×512, with the on-screen UI band
blurred on every path that reaches a model (a challenge rule, not a tuning
choice). Five perception channels then describe the clip, and a **router** turns
that description into an answer:

| channel | what it produces |
|---|---|
| **Tool recogniser** | ResNet-50 @ 384px, 12-class multi-label (sigmoid). Predicts instrument **install state** over the window, not per-frame visibility. |
| **Task recogniser** | ResNet-50 @ 384px, 8-class multi-class (softmax). |
| **Detector** | Per-class instrument detections with timestamps and confidences. |
| **Motion** | Mean absolute inter-frame difference on a 64×64 grayscale reduction, at two time bases — *within-burst* (67 ms apart) and *across-clip* (1.9 s apart). |
| **Agreement** | Concordance between the tool recogniser and the detector, used as an independent check on either alone. |

Per-frame probabilities are reduced to clip-level probabilities by a mean over
frames, then thresholded with **per-class serving thresholds** tuned against that
exact aggregation. The router dispatches on a question **intent** (11 forms:
tool identity, task, organ, procedure, purpose, and the polar variants) and
renders an answer in the surface form measured to score best for that intent.

### Where the VLM fits

A **Qwen2.5-VL-7B** vision-language model runs in NF4 quantisation with a LoRA
adapter trained in three stages (NVIDIA's `Qwen2.5-VL-7B-Surg-CholecT50` surgical
base → a general GI corpus → SurgVU 16-frame QA). It receives the frames *and* a
rendered summary of everything the perception channels found.

The arbiter runs in **`per_intent`** mode with the VLM armed on
**`tool_identity_open` only** (`config/arbiter.json`). This is deliberate and
measured: instrument identity is the one intent where looking at the frames beats
reading a classifier — three of the twelve classes share the head noun "forceps",
and in the tight-margin band the classifier's top-1 is near chance while its top-2
holds the answer ~91% of the time. On the other intents the router reads real
perception and arming the VLM measured worse. A later revision that gave the VLM
first crack at every intent **lost** points on the real test set.

### The fix that defines this version

`select_plan` originally chose a VLM frame plan on **time budget alone**, with no
VRAM term. On the grader's 14.6 GiB Tesla T4 the resulting prefill OOM'd, the
exception was swallowed, and the router answered instead — silently, on every
graded case, through several submissions. `frame_plan.py` now carries a measured
`MATH_KERNEL_TOKEN_CEILING = 3072` for sub-16 GiB cards and steps down on OOM.
This is what made the VLM actually run on the evaluation hardware.

---

## Measured results

| | value | notes |
|---|---|---|
| **Final test set (101 cases)** | **0.6604** | BERTScore-F1, max over 5 references. 3rd team of 19. |
| Preliminary set | 0.9128 | Closed vs. open question sets — the two are not comparable. |
| Tool recogniser | 0.7802 macro-F1 | Honest two-fold over validation windows; the number to quote. |
| Task recogniser | 0.9348 accuracy / 0.7920 macro-F1 | |

`tip-up fenestrated grasper` scores 0.0 F1 — it is genuinely absent from the
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
are load-bearing — `containers/surgvu26-submission.def` records why each one is
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
condor/         HTCondor job files — how every run was actually executed
tests/          test suite
docs/           design notes, build guide, compliance audit, version history
```

- **`docs/VERSIONS.md`** — what each version changed and what it scored, including
  the ones that lost points.
- **`docs/compliance_audit.md`** — pre-submission audit: data segregation, UI
  blur, split discipline, weight provenance, licensing, secrets.
- **`docs/design/`** — the design plans and measurement notes the build followed.

---

## Data and licensing

The SurgVU 2026 dataset is **not** redistributed here, and no challenge gold
reference answers are included. Access the data through the
[challenge organizers](https://surgvu26.grand-challenge.org/).

Code is Apache-2.0 (`LICENSE`). Third-party components and their licences are
listed in `NOTICE`. Model weights are released separately; see the challenge
report for the link.
