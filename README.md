## OpScribe-AI's Official Submission for SurgVU 2026 Category 2 Surgical VQA Challenge

Team Members:
- Noah John Kalthoff
- Ruffin Hager Bryant
- Aadya Ganjigunta
- Tuo Peter Li
- Dhananjay Bhaskar

Our model achieved BERTScore-F1 score of **0.9128** in Category 2 during the [preliminary phase](https://surgvu26.grand-challenge.org/evaluation/category-2-final-phase/leaderboard/) of the MICCAI EndoVis SurgVU Surgical VQA Challenge.

In this challenge, a 30-second surgical clip and a text question are provided; the model must produce a free-text answer. Submissions are scored
using BERTScore-F1, taking the best answer graded across five independent human reference answers.

---

### How it works

As soon as a clip gets ingested into our pipeline, it gets decoded into 16 frames, and
six measurements are taken from those frames. The question, separately, is sorted into one of
thirteen question types, based on the words and the actual string of the question.

The end of the pipeline is concluded by a judge, which decides, based on the question
type, whether the answer should come from the six measurements, from the vision-language
model, or from a fact we already know about the dataset and don't need to look at the
clip to say (ex; if a question is "what does x instrument do?", the answer to this will be the same every time).

That last category only happens in niche scenarios; most questions are answered by the six measurements in this pipeline. 

| measurement | what it is |
|---|---|
| **Tool recogniser** | A ResNet-50 that tries to identify what instruments are in the window. It outputs all 12 instrument classes, each with its own confidence score, so more than one instrument can be identified as being in the frame. |
| **Task recogniser** | A ResNet-50 that says what surgical step is happening, for example suturing or retraction. Unlike the tool recogniser, the task recogniser outputs just one answer, with eight possible activity classes. |
| **Detector** | Uses YOLO to draw boxes around instruments in the individual frames. The tool recogniser tells you what instruments are probably there; the detector is a second reference for what is there, and it also tells you where they are, with confidence scores. |
| **Variant head** | A ResNet-18 that tells a large needle driver apart from a mega one, which is the one thing the tool recogniser can't do, since it only knows a needle driver is there and not which size. It looks at the needle drivers the detector found, cropped out of the frames, and uses the whole frame if the detector didn't find one. |
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
Qwen2.5-VL-7B, served NF4-quantised, 16 frames per call

  Qwen/Qwen2.5-VL-7B-Instruct              base instruction-tuned VLM
    └─ nvidia/Qwen2.5-VL-7B-Surg-CholecT50   surgical continuation, CholecT50
         └─ ~87k-pair GI corpus                abdominal / GI surgery QA
              └─ SurgVU 16-frame QA              this challenge's own clips

  LoRA  r=32, alpha=64  ·  95.2M of 4.79B parameters trained (1.99%)
  Prompt is evidence-conditioned: the model sees the frames AND a rendered
  summary of what all six measurements found.
```

The pre-training and fine-tuning we did sits on top of a model that has already been
trained to thrive in surgical understanding, so the base model arrives with some
understanding of surgical tools and phases.

The training we did touches on how the model outputs and the words that it speaks:
further fine-tuning it on minimally invasive surgery, and then finally showing it
SurgVU-style frames and answers to set it up for success on actual runs.

### The router's thirteen question types

```
OPEN ANSWER                       YES OR NO
  tool_identity_open                tool_presence_polar
  task_open                         cutting_polar
  organ_open                        suture_polar
  count_open                        task_confirmation_polar
  procedure_open                    approach_polar
  purpose_open
                                  NO MATCH
                                    unknown_open
                                    unknown_polar
```

`classify_question()` picks one of these buckets based on the question text alone,
using pattern matching. The main thing we are working out here is the shape of the
answer being asked for. If a question falls in the `tool_identity_open` bucket, we know
we are looking for an answer like "Bipolar Forceps". If it falls under `suture_polar`,
we know it is asking for a yes or no.

It is also worth knowing that the patterns are double-checked so that a question does
not fall into the wrong bucket. For example, "how many instruments are in use?" contains
the word *instrument*, so there is potential for it to be incorrectly put in an
instrument bucket even though we want a number as the output.

### All the possible outcomes from the pipeline

Here are three examples of the pipeline answering questions from each of the three
different mechanisms: the measurements, the VLM, and the hard-coded fact answers.

**Example 1: the measurements pick the answer.** In this scenario the question could be
"Is tissue being cut in this clip?", which would be classified as `cutting_polar`. The
tool recogniser checks for a credible cutting instrument, and if there isn't one, the
answer is No. If there is one, the outputs from the motion measurement are also taken
into account before committing. Motion comes into play if something that cuts is
installed and the scene is very static, which could show that tissue is not actually
being cut in this clip. Most answers end up being fairly complex, with the pipeline
taking in a lot of differing opinions from the measurements and deciding which one is
the most confident.

**Example 2: the VLM outputs the answer.** Here the question could be "What instrument
is the surgeon using?", which classifies as a `tool_identity_open` question. This is the
one question type that is always diverted to the VLM. We did a lot of internal testing
and found that the router had a specifically tough time with this question, and that the
VLM was a lot better at this question type than at a lot of the others, which is why the
measurements are the dominant choice of answer for most question types. Three of the
instruments are all forceps, just different types of forceps, which makes it very hard
for the measurements and the router to be anything better than a coin flip here. So this
is perfect for the VLM to come in, use its own intelligence, and be the final word.

**Example 3: we already know the fact.** The question could be "What is the purpose of
using forceps?". Looking at the clip doesn't help answer this, because the question is
about the definition of what forceps are for. It resolves by looking through what is
essentially a pre-made file to find the answer about an instrument of that type. Two
other question types work this way, because we know every clip in SurgVU is going to be
endoscopic surgery. This is a very simple deterministic approach, and we found it has
the highest probability of success compared with sending these questions to the VLM or
the measurements.

Anything the classifier can't place with the patterns it has falls to `unknown_open` or
`unknown_polar`, and both of those go to the VLM. The router has no form to answer these
questions, so we aren't able to set the answer up for success. The VLM has the potential
to output whatever and answer whatever, whereas the router is constrained to what we
have set it up for.

## Results

| | value | notes |
|---|---|---|
| Preliminary phase | 0.9128 | An 11-question set, way smaller than the final set. |
| Tool recogniser | 0.7802 macro-F1 | |
| Task recogniser | 0.9348 accuracy / 0.7920 macro-F1 | |

---

## Reproducing and running the container

The submission is two artifacts rather than one.

| artifact | contains |
|---|---|
| the container image | the code, both ResNet-50 recognisers, the detector, the variant head, and the VLM's LoRA adapter |
| a model tarball | the NF4-quantised Qwen2.5-VL-7B base, ~5 GB |

We split them because Grand Challenge caps the image at 10 GB and the base model on its
own is about 5 GB. Grand Challenge extracts the model tarball to `/opt/ml/model/` when
the container runs, and `resolve_vlm_model_dir()` checks there before it checks anywhere
else. The LoRA inside the image was trained on that exact base, so the two have to be
used together.

### Getting the weights

We don't keep weights in this repository. All of them are on Hugging Face at
[opscribe-ai/surgvu26-cat2-v6.2](https://huggingface.co/opscribe-ai/surgvu26-cat2-v6.2).
`containers/build_submission.sh` pulls them into the build context, and if yours are
somewhere else you can point `VLM_MODEL_SRC` at a directory that holds
`qwen25vl-7b-nf4/`.

### Building

```bash
# Apptainer (what was used)
containers/build_submission.sh
apptainer build surgvu26-submission.sif containers/surgvu26-submission.def

# Docker (Grand Challenge upload format)
docker build -f containers/Dockerfile -t surgvu26-cat2 .
docker save surgvu26-cat2 | gzip > surgvu26-cat2.tar.gz

# the model tarball. The trailing dot is load-bearing: Grand Challenge uses the
# archive's paths as-is, so packing the parent directory makes every lookup miss.
tar -czf surgvu26-models.tar.gz -C /path/to/models .
```

### Running one case

The container reads and writes fixed paths, and both JSON files hold a JSON-encoded
string rather than raw text.

| direction | path |
|---|---|
| read | `/input/endoscopic-robotic-surgery-video.mp4` |
| read | `/input/visual-context-question.json` |
| write | `/output/visual-context-response.json` |

```bash
mkdir -p input output model
tar -xzf surgvu26-models.tar.gz -C model/          # gives model/qwen25vl-7b-nf4/

cp your_clip.mp4 input/endoscopic-robotic-surgery-video.mp4
echo '"What instrument is being used?"' > input/visual-context-question.json

docker run --rm --gpus all \
  -v "$(pwd)/input:/input:ro" \
  -v "$(pwd)/output:/output" \
  -v "$(pwd)/model:/opt/ml/model:ro" \
  surgvu26-cat2

cat output/visual-context-response.json            # e.g. "Bipolar Forceps"
```

If you forget the `/opt/ml/model` mount the container will still run, but the VLM won't
find its weights and every question ends up being answered by the router instead.

### Environment

The base image is `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`. The VLM layer pins
`transformers==4.57.6`, `accelerate==1.14.0` and `bitsandbytes==0.50.1`. Those pins are
load-bearing, and `containers/surgvu26-submission.def` records why.
`docs/container_build.md` covers the build in full, and `docs/submission_interface.md`
documents the Grand Challenge input/output contract.

The final build ran on a single 14.6 GiB T4 within the 600 second per-case wall clock,
with no network access at inference.

---

## Repository layout

```
src/surgvu/     pipeline: perception, router, arbiter, VLM, frame planning
scripts/        inference entrypoint, training, evaluation, tuning
config/         model config, serving thresholds, arbiter policy, splits
containers/     Dockerfile, Apptainer definition, build scripts
condor/         HTCondor job files, how every run was actually executed
tests/          test suite
docs/           design notes, build guide, compliance audit
```

- **`docs/compliance_audit.md`**: ensures we were working within the rules.
- **`docs/design/`**: the plans and measurements we followed.

---

## Data and licensing

The SurgVU 2026 dataset is not redistributed here. The organizers will likely release
it after the challenge is over.

Code is Apache-2.0 (`LICENSE`). Third-party components and their licences are listed
in `NOTICE`.

## Hugging Face links

- **Base Qwen model** — [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- **NVIDIA then trained that model further on CholecT50** — [nvidia/Qwen2.5-VL-7B-Surg-CholecT50](https://huggingface.co/nvidia/Qwen2.5-VL-7B-Surg-CholecT50)
- **The dataset we then used to train a LoRA on minimally invasive surgery** — [opscribe-ai/mis-abdominal-gi-min-invasive](https://huggingface.co/datasets/opscribe-ai/mis-abdominal-gi-min-invasive)
- **The dataset we fine-tuned that LoRA on to reach the model used in the pipeline** — [opscribe-ai/surgvu-cat2-vqa](https://huggingface.co/datasets/opscribe-ai/surgvu-cat2-vqa)
- **All the artifacts (CNNs, YOLO, the LoRA)** — [opscribe-ai/surgvu26-cat2-v6.2](https://huggingface.co/opscribe-ai/surgvu26-cat2-v6.2)

---

## AI assistance

We used [Claude Code](https://claude.com/claude-code) while building this. It helped
with the pipeline code, the evaluation tooling, the documentation and commits. 
