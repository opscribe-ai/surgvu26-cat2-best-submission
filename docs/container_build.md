# Building and shipping the submission container

## VERDICT, stated plainly

**The shipping artifact must be produced on a machine with Docker. CHTC cannot
produce it.** This is not a workaround away; it is a decision for a human.

Grand Challenge accepts exactly one kind of upload for an algorithm container:
a tarball produced by `docker save`. Their model field validates the extension
against `(".tar", ".tar.gz", ".tar.xz")` and their server-side validator opens
the archive and requires a top-level `manifest.json`:

    manifest = json.loads(open_tarfile.extractfile(
        container_image_files["manifest.json"]).read())
    except KeyError:
        raise ValidationError(
            "Could not find manifest.json in the container image file. "
            "Was this created with docker save?")
    -- comic/grand-challenge.org, app/grandchallenge/components/tasks.py

and the documented command is

    docker save IMAGE | gzip -c > IMAGE.tar.gz
    -- https://grand-challenge.org/documentation/exporting-the-container/

Consequences, each checked rather than assumed:

| Path | Works? | Why |
|---|---|---|
| Upload an Apptainer `.sif` | **No** | Not a tar, no `manifest.json`, fails the extension validator immediately. `singularity`/`apptainer`/`.sif` appear **zero** times in the Grand Challenge codebase. |
| Upload an **OCI** archive (`podman save --format oci-archive`, `buildah push oci-archive:`) | **No** | OCI archives carry `oci-layout` + `index.json` + `blobs/` and no `manifest.json`, which is the exact `KeyError` above. Server-side they push with **crane**, and there is no OCI-archive branch. |
| `docker save` from Docker ≥25 (containerd image store) | Yes | Their validator explicitly handles both the `<25` and `>=25` layouts, and the `sha256:`-prefixed config crane produces. |
| `podman save` **without** `--format` | Probably | It defaults to `docker-archive`, which does write `manifest.json`. Format-compatible, but undocumented by GC — treat as a fallback, not a plan. |
| Pull from ghcr.io / Docker Hub | **No** | No image-URL field exists; the docs offer only "link a GitHub repository" and "upload the container image". |
| Link a GitHub repository, let AWS CodeBuild build it | Yes, **but** | Requires a `Dockerfile` at the repository **root**, the Grand Challenge GitHub app installed, and **an open-source licence GC recognises** (Apache-2.0, MIT, GPLv3, AGPLv3, MPL-2.0, BSL-1.0, Unlicence). Satisfied: this repository is public and carries a root `LICENSE` (Apache-2.0, one of the recognised set). |

### Why CHTC cannot do it

    $ which docker podman buildah
    /usr/bin/which: no docker in (...)
    /usr/bin/which: no podman in (...)
    /usr/bin/which: no buildah in (...)
    $ apptainer --version
    apptainer version 1.5.2-1.el9

Apptainer 1.5.2's `build` writes a SIF or a sandbox directory — those are the
only two output formats it offers. Its `buildkit:` build spec can consume a
`Dockerfile`, but the output is still a SIF and the daemon it needs is not
installed (`/usr/libexec/apptainer/bin/` holds no `buildkitd`/`buildctl`).

**A genuinely open door, deliberately not walked through tonight:**
unprivileged user namespaces DO work here (`unshare -U -r id` returns
`uid=0(root)`) and `newuidmap`/`newgidmap` are present, so a static rootless
`podman` or `buildah` binary could in principle build and `podman save` a
docker-archive on a compute node. That is a multi-hour, uncertain detour
(storage driver, subuid ranges, offline base-image pull) to avoid a
five-minute `docker build` on a laptop. **Recommended only if no Docker host
is available at all.**

## What is prepared here, and what the user does with it

Everything except the one command that needs a Docker daemon.

    containers/Dockerfile              the shipping recipe, complete
    containers/.dockerignore           belongs at the context root
    containers/build_submission.sh     stages the context on a compute node
    containers/build_submission.sub    ... as an HTCondor job
    containers/surgvu26-submission.def the Apptainer twin (content proof only)

The build context is staged as a single downloadable tarball:

    /staging/n/nkalthoff/surgvu26/submission_context.tar.gz

It contains `src/`, `scripts/`, `config/`, `models/tools_v2.pt`,
`models/task_v2.pt`, `Dockerfile` and `.dockerignore` — nothing else. On any
machine with Docker:

    scp <chtc>:/staging/n/nkalthoff/surgvu26/submission_context.tar.gz .
    mkdir surgvu-context && tar -xzf submission_context.tar.gz -C surgvu-context
    cd surgvu-context
    docker build --platform linux/amd64 -t surgvu26-cat2:v1 .
    docker save surgvu26-cat2:v1 | gzip -c > surgvu26-cat2-v1.tar.gz

The build is self-gating: it fails rather than producing a bad image if either
checkpoint does not match the sha256 in `config/perception.json`, if the config
does not bind the re-tuned **serving thresholds** to those same weights, or if
the entrypoint cannot answer a synthetic case end to end on CPU with no
network.

### The serving-threshold gate, and its escape hatch

`config/perception.json` is schema 2 and carries a `serving_thresholds` block:
per-class cuts re-tuned on the CLIP MEAN the container thresholds rather than
on a frame, worth **+0.0195 macro-F1** for no retrain. `scripts/inference.py`
falls back to the checkpoint's per-frame cuts with a WARNING when that block is
missing or bound to other weights — right at serving time, and wrong at build
time, where it means an image ships 0.0195 worse and says so only in a log line
nobody reads. So `scripts/verify_checkpoints.py` FAILS the build on: no block,
a block whose `provenance.checkpoint_sha256` is not the bound checkpoint (or is
absent), a vector of the wrong length or holding non-probabilities, a `by_class`
that disagrees with the served `values`, a `tuned_against_checkpoint_thresholds`
that is not the config's mirror, a block on the softmax task head, and a
`schema_version` the gate does not understand.

Building deliberately WITHOUT the block — the counterpart of
`scripts/build_perception_config.py --drop-serving-thresholds` — has to be
typed:

    docker build --platform linux/amd64 \
        --build-arg SERVING_THRESHOLDS=optional -t surgvu26-cat2:v1 .
    apptainer build --build-arg SERVING_THRESHOLDS=optional out.sif \
        containers/surgvu26-submission.def
    SERVING_THRESHOLDS=optional containers/build_submission.sh

It waives only the block's ABSENCE: a block that is present and unfit still
fails the build, and the log prints `WAIVED` plus a WARNING rather than `OK`.

Test it before uploading, the way the grader runs it:

    mkdir -p in out
    cp <case>.mp4 in/endoscopic-robotic-surgery-video.mp4
    printf '%s' '"Are there forceps being used here?"' > in/visual-context-question.json
    docker run --rm --network none \
        -v "$PWD/in":/input:ro -v "$PWD/out":/output \
        surgvu26-cat2:v1
    cat out/visual-context-response.json      # must be a QUOTED string

`--network none` is the point: it reproduces "no internet access once
submitted" and turns any accidental download into a failure here instead of a
zero there.

## Requirements the Dockerfile satisfies, and how each is enforced

| Requirement | Where it is enforced | How this image satisfies it |
|---|---|---|
| `linux/amd64` | GC validator compares `config["architecture"]` against `COMPONENTS_CONTAINER_PLATFORM = "linux/amd64"` | `FROM --platform=linux/amd64` |
| **Non-root user** | GC validator rejects `user in ["", "root", "0"]` with "The container runs as root. Please add a user, group and USER instruction" | `groupadd`/`useradd` UID 1000, `USER algorithm` |
| An entrypoint | Not validated, but the SageMaker shim re-execs `Entrypoint`/`Cmd`; an image with neither has nothing to run | `ENTRYPOINT ["python", ".../inference.py", "--input-dir", "/input", "--output-dir", "/output", "--models-dir", "/opt/algorithm/models"]` |
| Under 10 GB | Documented on the challenge-submission page | **MEASURE THE REAL TARBALL. The .sif is not the artifact and understates this.** As of 2026-08-26 the image carries the VLM: the built .sif is 8,823,336,960 B (8.82 GB / 8.22 GiB), but Grand Challenge receives `docker save \| gzip`, and the NF4 weights DO NOT COMPRESS -- measured gzip ratio **0.9625** over a 300 MB sample of `model-00001-of-00002.safetensors`. Projected tarball: 5.69 GB (weights) + ~3.46 GB (prior image) + ~0.19 GB (transformers/accelerate/bitsandbytes, 563 MB uncompressed) = **~9.3 GB, roughly 0.7 GB under the ceiling**. Fits, but thin enough that an unmeasured assumption is not safe. If your measured tarball exceeds 10 GB, the fallback is option 3 below (ship the VLM as a separate `model.tar.gz` to /opt/ml/model/), which sidesteps the ceiling entirely. |
| No network at runtime | GC: "Your container will be executed without access to any network resources." | Nothing installs at runtime; `load_expert` builds its backbone with `pretrained=False` so torchvision never fetches ImageNet weights; `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` set so a future VLM fails loudly instead of hanging on DNS inside the budget |
| `/input` read-only, `/output` and `/tmp` writable | GC: "The input directory is read only. `/tmp` and `/output` are fully writable" | The entrypoint only reads `/input` and only writes `/output`; the validation run binds `/input` `:ro` |
| sm_75 (T4) | Deployment instance | torch 2.5.1+**cu121**, whose kernels include sm_75; the CNNs are fp32, so no bf16 and no FlashAttention-2 anywhere |

### Weights are baked in, not uploaded separately

Grand Challenge supports a separate `model.tar.gz` extracted to `/opt/ml/model/`
at runtime, and their guidance is "Model weights should be uploaded
separately." That guidance exists because of the 10 GB image ceiling. Both
CNN checkpoints together are **165 MB**, so baking them in costs nothing and
removes a whole class of deployment failure (a missing or mis-versioned
weights upload).

**The VLM has now landed (Plan 2, Task 4), and this is the SECOND revisit
(2026-08-25) -- now that a LoRA fine-tune checkpoint actually exists**
(`models/vlm_lora/checkpoint-1200` on staging, past the epoch-1 boundary).
`--vlm` stays wired into both the Dockerfile ENTRYPOINT and the .def's
`%runscript`, and `containers/build_submission.sh` now HAS a source-path
variable (`VLM_MODEL_SRC`) ready to stage a self-contained checkpoint
directory the moment one exists -- but its weights are STILL deliberately
NOT staged, and the reason changed from "nothing to stage" to "what exists
does not fit, and what would fit does not exist":

  * **The base model's real on-disk cost was measured, not estimated, this
    time**: the fp16 `Qwen/Qwen2.5-VL-7B-Instruct` HF cache used to fine-tune
    it (`/staging/n/nkalthoff/surgvu26/hf_cache`) is **16 GB** -- 5 safetensors
    shards. 3.46 GB (today's image) + 16 GB + the ~190 MB LoRA adapter +
    ~0.1-0.2 GB of `transformers`/`bitsandbytes` lands at roughly **19.7-19.9
    GB -- nearly 2x the 10 GB ceiling.** This is not a hypothetical; it is
    what actually sits on disk today, and it is the ONLY form of the base
    model that exists anywhere in this project.
  * **A pre-quantised NF4 checkpoint would likely fit** (comparably-sized
    artifacts elsewhere in this project run ~5-6 GB, which would land the
    total around 8.5-9 GB), but **none exists for Qwen2.5-VL-7B-Instruct** --
    the one pre-quantised NF4 directory on disk (`models/qwen3vl-8b-nf4`) is a
    DIFFERENT model, belonging to the retired `surgvu.vlm` fallback, not this
    one. Producing an NF4 checkpoint of this model WITH the fine-tune baked in
    would mean merging the LoRA adapter into the base before quantising --
    which `scripts/train_vlm.py`'s own module docstring explicitly argues
    against ("The LoRA adapter itself is never merged into the base model");
    reversing that design decision is not this task's call to make.
  * **Even setting size aside, `src/surgvu/evidence_vlm.py`'s loader (frozen;
    out of scope to modify) has no code path that applies a LoRA adapter on
    top of a base model, and no code path that requests on-load
    quantisation** (`_load_model` calls a bare
    `AutoModelForImageTextToText.from_pretrained(model_dir, ...)`, unlike
    `scripts/train_vlm.py`'s own loader, which passes a `BitsAndBytesConfig`).
    It only ever correctly loads a checkpoint that is ALREADY, by itself,
    both fine-tuned and quantised -- self-contained, the same shape as
    `models/qwen3vl-8b-nf4`. Staging the fp16 base and the adapter side by
    side, as separate artifacts, would not wire the fine-tune into serving at
    all, on top of not fitting.

Full accounting: `docs/design/2026-08-24-v5-evidence-pipeline/
vlm-weight-staging-report.md`.

Until this is resolved, `--vlm` is a **safe no-op**: on a No-GPU
deployment draw it never even tries (structurally, `available()` gates on
`torch.cuda.is_available()` before any import); on a CUDA draw it tries,
fails fast and offline (`HF_HUB_OFFLINE=1`, no network), and R18 falls back to
the router's answer. Options for whoever resolves this, in descending order
of how much they change today's shipping decision:

  1. **Produce a pre-quantised, adapter-merged NF4 checkpoint** (the shape
     `models/qwen3vl-8b-nf4` already demonstrates for a different model) and
     bake THAT in. Likely fits (~8.5-9 GB total, by that artifact's ratio),
     and is the only option that also solves the "frozen `evidence_vlm.py`
     loader has no adapter/quantisation code path" problem -- but requires
     merging the LoRA adapter into the base before quantising, which
     `scripts/train_vlm.py`'s own docstring argues against, plus a fresh,
     unvalidated quantisation pass for this exact model+adapter pair. Not
     executed here: it reverses a recorded design decision and needs the
     controller's sign-off, not a workaround.
  2. **Bake the raw fp16 base in anyway.** Measured, not estimated: this
     lands at ~19.7-19.9 GB, roughly 2x the ceiling. Not viable regardless of
     the adapter question.
  3. **Use Grand Challenge's separate `model.tar.gz` path** for the VLM
     weights specifically (CNN checkpoints stay baked in, as today), which
     sidesteps the 10 GB image ceiling entirely at the cost of a second
     upload artifact and a runtime read from `/opt/ml/model/` this codebase
     does not currently have any code path for. Still does not by itself fix
     the missing adapter/quantisation-on-load code path in `evidence_vlm.py`.
  4. **Quantise further or prune** (e.g. drop the vision-tower layers this
     project's evidence-packet-driven prompting may not need at full
     resolution) to buy back headroom -- unmeasured, and a genuinely new
     validation burden (4-bit numerics already need re-validating per
     architecture; a second quantisation pass is a second thing to
     re-validate).

None of these is executed by this task -- there is nothing safe to stage
yet. This paragraph is the decision record for whoever takes the next step.

## Two things a human has to decide

1. **Which algorithm API.** Grand Challenge now has an HTTP-server form
   (`GET /health`, `POST /invoke`) selected by
   `LABEL org.grand-challenge.api-method="invoke"`. Absent that label the
   container uses the older `exec` form, which is the file-based
   `/input` → `/output` contract in `docs/submission_interface.md`, and which
   is what the SurgVU **2025** Category 2 template uses. **This image is
   deliberately `exec`-style, with no label.** Confirm against the SurgVU 2026
   submission instructions before uploading; if 2026 requires `invoke`, the
   perception code is unaffected and only a thin HTTP wrapper is needed.
2. **Where the Docker build happens.** A local machine (recommended), or the
   linked-GitHub-repo route, which requires a public repository with a
   recognised open-source licence — both of which now hold.

## The Apptainer twin, and what it does and does not prove

`containers/surgvu26-submission.def` is the same recipe: same base, same two
pip pins, same four copied trees, same build-time checkpoint and end-to-end
gates, same argument vector in `%runscript` as the Dockerfile's `ENTRYPOINT`.
It exists because CHTC has a GPU and no Docker, so it is the only way to prove
the image's **contents** run here.

It does **not** prove the two Docker-side properties: the non-root `USER`
(Apptainer runs as the invoking user regardless) and the `ENTRYPOINT` form.
Those are verified by the `docker run` command above, on the machine that
builds it.

It is **not uploadable**, and must not be mistaken for the deliverable.
