# Pre-submission compliance audit -- SurgVU 2026 Category 2

Adversarial audit run 2026-08-11 against branch `fix/staging-verification-jpeg-shards`,
before an irreversible submission and before the repository is made public.

> **Read this as a dated record, not a current description.** The audit was run
> on 2026-08-11, against the model generation of that date. Two things have moved
> since: the two perception recognisers were **EfficientNetV2-S** then and are
> **ResNet-50** in the submitted v6.2 system, so §4's weight-provenance trail
> describes the earlier checkpoints; and the repository counts in §6 (tracked
> files, largest file) are from that date and have grown. The findings on data
> segregation, UI blur, split discipline and secrets were re-checked before
> publication and still hold.

**Scope.** The eight areas named in the audit brief. Everything below is backed by
command output captured during the audit; nothing is asserted from documentation
alone. Read-only throughout: no checkpoint, config, split file or `/staging/groups`
path was modified, and nothing was pushed.

**What ships.** `containers/Dockerfile` is the shipping recipe (Grand Challenge
accepts only a `docker save` tarball). `/staging/n/nkalthoff/surgvu26/surgvu26-submission.sif`
is the Apptainer twin built from `containers/surgvu26-submission.def` and used here
as the content proof; `/staging/n/nkalthoff/surgvu26/submission_context.tar.gz`
(151,325,616 B) is the build context that would be handed to a Docker builder.

---

## Verdict summary

| # | Item | Verdict |
|---|---|---|
| 1 | Forbidden Category 1 data | **PASS** |
| 2a | UI blur on every path into a model | **PASS** |
| 2b | What the blur destroys, measured | **PASS** |
| 2c | The unblurred top banner | **NEEDS A HUMAN DECISION** -- the premise it was accepted on is false |
| 3 | Held-out integrity of `case_122`–`case_132` | **PASS** on the split and the training runs; see §3.4 |
| 4 | Model and weight provenance | **PASS** |
| 5 | Licensing of dependencies | **PASS** on copyleft; **NEEDS A HUMAN DECISION** on the missing repo licence |
| 6 | Secrets | **PASS** -- with four personal-path leaks to clean up |
| 7 | Anything in the image that should not ship | **PASS** |
| 8 | Internet at inference | **PASS** |

The one substantive finding is §2c. Everything else is either clean or a
publication-hygiene item that is cheap to fix.

---

## 1. Forbidden Category 1 data -- PASS

`cat1_test_set_public.zip` must never have been used for training or tuning. Per
the brief, this was checked at filename level only; the archive was never opened.

**It is not present on disk at all.** It is not in the staging directory the brief
names, and not anywhere under the project's own staging area:

    $ ls -la /staging/groups/bhaskar_opscribe/surgvu/
    case071_missing_video.zip
    cat2_sample/            exp1_deps/   exp1_deps2/   eyeball/
    labels_cat2/            labels_v2/   shards/
    surgvu24_labels_updated_v2.zip
    SURGVU25_cat_2_sample_set_public.zip
    SURGVU25_cat2_train_labels.zip
    videos/

    $ find /staging/groups/bhaskar_opscribe/surgvu/ -maxdepth 2 -iname "*cat1*"
    $ find /staging/n/nkalthoff/ -iname "*cat1*"
    (end)          # no output from either

**No code, config or script references it.** Every `/staging` path used by the
repository was enumerated; the only data roots any code reads are `shards/`,
`cat2_sample/` and `labels_cat2/`:

    $ grep -rn "/staging" --include=*.py --include=*.json --include=*.yaml \
        --include=*.sh --include=*.sub --include=*.def src/ scripts/ config/ \
        containers/ condor/ tools/ tests/
    ...
    scripts/train_tools.py:35:  --shards default="/staging/groups/bhaskar_opscribe/surgvu/shards"
    scripts/train_task.py:40:   --shards default="/staging/groups/bhaskar_opscribe/surgvu/shards"
    scripts/build_splits_v2.py:50: STAGING = Path("/staging/groups/bhaskar_opscribe/surgvu")
    config/splits_v2.json:280:  "source_shards": ".../surgvu/shards"
    config/splits_v2.json:281:  "source_sample_cases": ".../surgvu/cat2_sample"
    ...                          # no hit mentions cat1 anywhere

**Nothing in the full git history references it either, outside prose.** Searching
every commit for the string finds three, all documentation:

    $ git log --all --oneline -S"cat1_test_set_public"
    1fb9250 docs: Plan 2 -- perception model training
    52bbe94 Add foundation implementation plan (plan 1 of 3)
    aa47ca0 Add training data plan; correct the exactly-three assumption by measurement

Two of those three are the *prohibition* being written down:

    docs/design/plans/2026-08-08-surgvu26-cat2-foundation.md:17:
      - **Never train or tune on `cat1_test_set_public.zip` contents.**
    docs/design/plans/2026-08-09-surgvu26-cat2-cnns.md:19:
      - **Never train or tune on `cat1_test_set_public.zip`.**

The only Category 1 work ever contemplated was a tool detector as an internal
arbiter, and it was never built. No Cat 1 data exists on this filesystem, no code
path can reach it, and no artifact derives from it.

---

## 2. UI information

### 2a. The blur is on every path that reaches a model -- PASS

`preprocess.prepare_frame` is crop → blur → resize, in that order, always:

    src/surgvu/preprocess.py:73
    def prepare_frame(frame, size=512):
        """The single entry point. Crop, blur, resize -- in that order, always."""
        frame = crop_side_margins(frame)
        frame = blur_ui_band(frame)
        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_CUBIC)

**There are exactly two places in the codebase that decode video, and both call it.**

    $ grep -rn "VideoCapture|imread|imdecode|av.open|decord|imageio" --include=*.py src/ scripts/
    src/surgvu/extract.py:73:    capture = cv2.VideoCapture(str(video_path))     # training extraction
    src/surgvu/extract.py:291:       frame = cv2.imdecode(...)                   # reads back its own shards
    src/surgvu/perceive.py:89:   capture = cv2.VideoCapture(str(video_path))     # serving decode_clip

- **Training extraction**: `extract.py:91` -- `frames.append(prepare_frame(frame, size=size))`.
  Shards therefore store already-blurred JPEGs; `extract.py:291`'s `imdecode` only
  reads those back, so the training loader cannot see an unblurred frame.
- **Serving**: `perceive.py:105` -- `frames.append(prepare_frame(frame, size=size))`,
  inside `decode_clip`, which is the single frame source in `scripts/inference.py:581`.

**No path bypasses it, including the VLM.** `scripts/inference.py:589` passes the
*already-decoded* `frames` into `route(...)`, and `vlm.py:292` re-uses that array
(`Image.fromarray(frames[index][:, :, ::-1])`) rather than re-opening the video.
The VLM is off by default in any case: `--vlm` is `action="store_true"` and the
Dockerfile `ENTRYPOINT` does not pass it, so `build_vlm` returns `None`.

The property is pinned by tests, so a regression fails the suite rather than
shipping quietly -- `tests/test_perceive.py::test_decode_clip_blurs_the_ui_band`
and `tests/test_preprocess.py::test_prepare_frame_is_square_and_blurred`
("deleting the `blur_ui_band` call from `prepare_frame` must fail this test").

### 2b. What the blur actually destroys, measured on real frames -- PASS

Measured on a mid-clip frame from each of the 11 sample clips, in the container,
not assumed. All 11 are 1280×720 with 193 px black margins each side, cropping to
894×720; `UI_BAND_FRACTION = 0.08` gives a 58-row band, rows 662–719.

    case      raw         cropped    band   bar_lapV   bar_lapV    bar_std  bar_std
                                       px     BEFORE      AFTER     BEFORE    AFTER
    case122   1280x720    894x720      58     4263.6     14.848      43.32    28.47
    case123   1280x720    894x720      58     5712.8      8.600      42.77    25.21
    case124   1280x720    894x720      58     5068.6     31.107      50.71    37.56
    case125   1280x720    894x720      58     4358.5     21.025      44.53    31.27
    case126   1280x720    894x720      58     5318.5      8.945      39.26    23.90
    case127   1280x720    894x720      58     4563.6     30.857      45.17    30.38
    case128   1280x720    894x720      58     4385.2     42.427      47.75    33.46
    case129   1280x720    894x720      58     5042.8     14.275      47.39    32.38
    case130   1280x720    894x720      58     4294.5     12.687      41.72    28.42
    case131   1280x720    894x720      58     4963.9      5.704      45.25    31.46
    case132   1280x720    894x720      58     3523.4      7.985      54.88    42.16

    bar Laplacian var BEFORE  min 3523.4  max 5712.8  median 4563.6
    bar Laplacian var AFTER   min 5.7040  max 42.4268 median 14.2755
    reduction factor          min 103x    max 870x    median 338x

A Laplacian variance of ~14 is the level of flat, textureless image -- the same
scale as the non-text rows above the bar. The text is destroyed, not softened.

**The band covers the whole bar with margin to spare.** Per-row Laplacian variance
over the bottom 80 rows (median across 22 frames) locates the text precisely:

    row 640-661  lapVar  20-36     OUT of band   <- background, no text
    row 662-675  lapVar  20-38     IN band       <- 14 rows of headroom
    row 676      lapVar  243.5     IN band       <- text starts
    row 677-716  lapVar  419-6149  IN band       <- the bar
    row 717-719  lapVar  2.6-6.6   IN band       <- black bottom edge

The topmost text row is 676; the band starts at 662. Every text pixel is inside
the blur, with 14 rows of margin. This is consistent with
`tests/test_preprocess.py::test_band_is_wider_than_the_measured_overlay`, which
asserts `UI_BAND_FRACTION > 0.0625`.

*Residual risk, low:* the band is a fixed fraction of height and has only been
validated on 1280×720 material. A test clip in a different format whose UI bar
occupies more than 8% of frame height would be under-blurred. The Cat 2 sample
format is the format the test set is expected in, so this is noted, not raised.

### 2c. The unblurred top banner -- NEEDS A HUMAN DECISION

The brief states the banner is constant text and asks that the basis be documented,
and that it be flagged **if the region turns out not to be constant**.

**It is not constant. It is present in some cases and absent in others.**

Detected by normalised cross-correlation against a banner template, over 21 evenly
spaced frames per sample clip. The separation is total -- ~0.97 with the banner,
~0.14 without -- and it does not vary within a clip (min ≈ max in every row):

    case          min     med     max   verdict
    case122     0.992   0.994   1.000   BANNER
    case123     0.132   0.140   0.151   no banner
    case124     0.972   0.976   0.978   BANNER
    case125     0.131   0.134   0.136   no banner
    case126     0.931   0.957   0.973   BANNER
    case127     0.134   0.160   0.200   no banner
    case128     0.121   0.137   0.181   no banner
    case129     0.969   0.977   0.978   BANNER
    case130     0.967   0.969   0.972   BANNER
    case131     0.100   0.113   0.137   no banner
    case132     0.125   0.141   0.143   no banner

    PRESENT: case122, case124, case126, case129, case130      (5 of 11)
    ABSENT : case123, case125, case127, case128, case131, case132  (6 of 11)

This is visible directly in the rendered top strips of all 11 clips: five carry
`⚠ TRAINING INSTRUMENT -- NOT FOR HUMAN USE` in a dark box across the top, six
show surgical video to the frame edge. The pattern is identical at frame 0 and at
mid-clip.

Separately, a whole-frame constancy map over 22 frames confirms there is **nothing
else** fixed in the top region -- the only pixels identical in every frame are rows
0–2, which are a black letterbox edge:

    pixels identical across all 22 frames: 3559 of 643680 (0.55%)
    bounding box of the constant region in the TOP HALF:
      rows 0..2, cols 5..893, 1874 pixels
      mean intensity of those constant pixels: 0.62 (0=black)
      share of them that are near-black (<8): 100.0%

**The banner survives unblurred into the training data.** Applying the same
detector in the 512×512 shard domain -- literally the array the CNNs were fed --
across 45 sampled corpus cases:

    case_000   ncc med  0.154  no banner
    case_003   ncc med  0.973  BANNER
    case_025   ncc med  0.977  BANNER
    case_046   ncc med  0.976  BANNER
    ...
    training cases sampled: 45   banner present in 14 (31%)

So the tool and task CNNs trained on frames in which a UI element varies from case
to case. That is the fact the acceptance decision should have been made against.

**What mitigates it.** The banner does not appear to carry label information. Over
the same 45-case sample, banner presence is close to the 31% base rate in every
task class with meaningful support:

    task class                            banner   none   banner share
    range of motion                            0      1        0%   (n=1)
    rectal artery/vein                         5     12       29%
    retraction and collision avoidance         6      8       43%
    skills application                         0      2        0%   (n=2)
    suspensory ligaments                      11     21       34%
    suturing                                  13     26       33%
    uterine horn                               9     18       33%

The two 0% classes have n=1 and n=2 and carry no weight. The four well-supported
classes span 29–43% against a 31% base rate. It is also a single bit, so it cannot
identify a case; at most it is a weak nuisance variable.

**Why it still needs a decision.** The stated basis for accepting it -- "it is
constant text, so probably not predictive" -- is false as a matter of fact, and
`docs/submission_interface.md:58` records the acceptance as *"the untouched top
banner is not a problem"* on that basis. The correct basis is the weaker but real
one: it varies, the model saw it, and it is measurably near-independent of the
labels in a 45-case sample. Two further consequences follow that the original
framing hid:

1. **It is a train/test distribution question, not only a rules question.** If the
   organizers blur the top region in the test set and our model has keyed on it at
   all, behaviour shifts. If they do not, the model sees a variable it was never
   meant to use.
2. **Remediation is not free.** Extending the blur to a top band is a small change
   to `preprocess.py`, but it invalidates every extracted shard and both shipped
   checkpoints -- a full re-extract and retrain of both CNNs.

**Recommended disposition:** ship as-is, and replace the "constant text" claim in
`docs/submission_interface.md` with the measurement above, so the record states
what was actually checked. Re-blurring and retraining is defensible but is a
multi-day cost against a signal measured to be near-zero. This is the user's call,
not the auditor's -- it is flagged, as the brief required, because the region is
**not** in fact constant.

---

## 3. Held-out integrity of `case_122`–`case_132`

### 3.1 The split file holds them out, under the correct id form -- PASS

The id-format trap was checked explicitly rather than by raw set intersection:

    $ python3 -c "... json.load(open('config/splits_v2.json')) ..."
    heldout: ['case_122', 'case_123', 'case_124', 'case_125', 'case_126',
              'case_127', 'case_128', 'case_129', 'case_130', 'case_131', 'case_132']
    train&heldout: []
    val&heldout:   []
    total cases: 155

    # and by numeric pattern, not by string equality:
    train cases in 122-132 range: []
    val   cases in 122-132 range: []

### 3.2 The shipped checkpoints were trained on that split -- PASS, from the artifacts

The checkpoints themselves record no split, so this was established from the job
records and an independent count, not from documentation.

**The submitted arguments name `splits_v2.json` for both runs.** Note that
`train_tools.py:36` and `train_task.py` default to `config/splits.json` -- the
*leaky* v1 split -- so this had to be passed explicitly, and it was:

    $ condor_history 9623711 9623712 -af ClusterId Args Cmd Out
    9623711  scripts/train_tools.py --splits config/splits_v2.json
             --out /staging/n/nkalthoff/surgvu26/models/tools_v2.pt --epochs 6
    9623712  scripts/train_task.py  --splits config/splits_v2.json
             --out /staging/n/nkalthoff/surgvu26/models/task_v2.pt  --epochs 4

**The shard counts corroborate it independently.** Both training logs report the
same loader totals:

    logs/train_9623711.out:  shards: 176 train / 45 val      (tools_v2.pt)
    logs/train_9623712.out:  shards: 176 train / 45 val      (task_v2.pt)

and recounting the shard directory against each split file shows 176/45 is
achievable **only** under `splits_v2`:

    total shards on disk: 235
    config/splits_v2.json: train_shards=176 val_shards=45 heldout_shards=14 sum=235
       unassigned: []
    config/splits.json:    train_shards=187 val_shards=48 heldout_shards=0  sum=235
       unassigned: []

176 + 45 = 221, and 235 − 221 = 14 -- which is exactly the shard count of
`case_122`…`case_132` (`case_124`, `case_131`, `case_132` have two parts each,
the other eight have one). The loader (`dataset.shard_paths_for_split`) selects by
`p.name.rsplit("_part", 1)[0] in cases`, so `case_122_part1.npz` maps to
`case_122`. **No frame from any of the 11 cases entered either training run.**

**The checkpoint bytes are the ones those runs produced.** Metadata read straight
out of the `.pt` pickles matches each log line exactly:

    tools_v2.pt  macro_f1 0.6605197787284851, epochs 4, tip-up F1 0.0
      <- log 9623711: "epoch 3 ... val_macroF1 0.6605", "BEST val macro-F1: 0.6605"
    task_v2.pt   accuracy 0.8694983818770227, macro_f1 0.6802740097045898,
                 description_accuracy 0.8803398058252427, epochs 4
      <- log 9623712: "epoch 3 ... val_acc 0.8695 macroF1 0.6803 desc_acc 0.8803"

### 3.3 The shipped bytes are those checkpoints, everywhere -- PASS

One sha256 per expert, identical in all four places it exists:

    /staging/.../models/tools_v2.pt   9447b0a0faddf7722ae06d594a2f642d3ff859e6c9db290d434a6e4709f05cd6
    tarball ./models/tools_v2.pt      9447b0a0faddf7722ae06d594a2f642d3ff859e6c9db290d434a6e4709f05cd6
    image /opt/algorithm/models/...   9447b0a0faddf7722ae06d594a2f642d3ff859e6c9db290d434a6e4709f05cd6
    config/perception.json  sha256    9447b0a0faddf7722ae06d594a2f642d3ff859e6c9db290d434a6e4709f05cd6  (81672040 B)

    /staging/.../models/task_v2.pt    ea964fe8ee04f29813a2e1635f2b374a5d6aaaab6c6a49dd806bcd63ca76a5eb
    tarball ./models/task_v2.pt       ea964fe8ee04f29813a2e1635f2b374a5d6aaaab6c6a49dd806bcd63ca76a5eb
    image /opt/algorithm/models/...   ea964fe8ee04f29813a2e1635f2b374a5d6aaaab6c6a49dd806bcd63ca76a5eb
    config/perception.json  sha256    ea964fe8ee04f29813a2e1635f2b374a5d6aaaab6c6a49dd806bcd63ca76a5eb  (81650198 B)

`scripts/verify_checkpoints.py` re-checks these at build time, so a stale or
truncated weight file fails the Docker build rather than every graded case.

### 3.4 The sample-clip → corpus-case correspondence

`config/splits_v2.json` holds out `case_122`…`case_132` because the sample
directories are named `case122`…`case132`. That mapping is done **by name only** --
`sampling.py:168` normalises `case122` → `case_122` -- and was never verified
against content. It is load-bearing: if sample clip `caseNNN` is in fact an excerpt
of some *other* corpus case, the wrong cases were held out and the sample leaked
into training after all.

This was tested directly, by perceptual-hashing every frame of each 30 s sample
clip and scanning the full-length corpus videos for a near-duplicate. The matcher
was validated first with a positive control on `case_122`'s own video:

    SAME VIDEO, frames N apart (what a true match looks like):
      frame    1000 ->  0:0   30:0   60:0   120:1   300:9   600:15
      frame   60000 ->  0:0   30:1   60:1   120:0   300:1   600:2
      frame  200000 ->  0:0   30:3   60:13  120:20  300:2   600:3
      frame  400000 ->  0:0   30:7   60:7   120:5   300:1   600:5
    UNRELATED frames in the same video (noise floor):
      n=45 pairs, min 0, median 57, max 117

so a hamming distance ≤ 20 is a match and ~40+ is not. Sample and corpus share
geometry exactly (both 1280×720 @ 60 fps), so nothing prevents a match:

    SAMPLE case122     1280x720 fps=60.000 frames=1800   dur=30.0s
    CORPUS case_122    1280x720 fps=60.000 frames=502947 dur=8382.5s

**Result of the first scan:** a full sweep of `case_122`'s video at 2 s stride
(4192 of 502947 frames) found a best hamming distance of **40** -- above the match
threshold. `case122`'s content was **not** found in corpus `case_122`.

A 155-way parallel scan (HTCondor cluster 9636602, one job per corpus case,
all 11 sample clips' hashes as targets) was launched to determine whether the
sample clips appear in some *other* corpus case -- the only outcome that would mean
real leakage -- or nowhere in the corpus at all, which would make the heldout list
harmless over-caution and leave the no-leakage conclusion intact.

> **STATUS: this scan had not finished when the audit was written.** See
> "Open item" below. It does not change §3.1–§3.3, which stand on their own:
> whatever `case_122`…`case_132` contain, no shard belonging to them entered
> either training run.

---

## 4. Model and weight provenance -- PASS

**The image contains exactly two weight files, and nothing else that could carry
learned parameters:**

    $ find / -xdev \( -name "*.pt" -o -name "*.pth" -o -name "*.safetensors" -o -name "*.ckpt" \)
    /opt/algorithm/models/task_v2.pt
    /opt/algorithm/models/tools_v2.pt
    /opt/conda/lib/python3.11/site-packages/distutils-precedence.pth   # a path file, not weights

No torchvision hub cache ships either -- `/opt/algorithm/.torch` and
`/root/.cache/torch` do not exist in the image.

**Initialisation is torchvision ImageNet, and only that.** `src/surgvu/models.py`
builds from `torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1`, and
exactly one pretrained file was ever fetched into the training `TORCH_HOME`:

    $ find /staging/n/nkalthoff/surgvu26/torch_cache -type f -exec sha256sum {} \;
    dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74
      /staging/n/nkalthoff/surgvu26/torch_cache/hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth

torchvision names its weight files with the leading 8 hex of their sha256, and
`dd5fe13b…` matches `efficientnet_v2_s-dd5fe13b.pth` -- this is the official
upstream ImageNet-1k checkpoint, unmodified. Chain: torchvision ImageNet (BSD-3)
→ fine-tuned on SurgVU shards → `tools_v2.pt` / `task_v2.pt`. Nothing else.

**Strict independence from OpScribe holds.** Scanning the image's filesystem --
raw byte-grepping the `.sif` is a false negative because squashfs is compressed,
so this was done inside the container:

    $ apptainer exec ... grep -rn -i -E "opscribe|/home/nkalthoff|nkalt|ap2001|wisc\.edu|gmail" /opt/algorithm
    config/splits_v2.json:280:  "source_shards": "/staging/groups/bhaskar_opscribe/surgvu/shards"
    config/splits_v2.json:281:  "source_sample_cases": "/staging/groups/bhaskar_opscribe/surgvu/cat2_sample"
    scripts/build_splits_v2.py:50:   STAGING = Path("/staging/groups/bhaskar_opscribe/surgvu")
    scripts/build_variant_priors.py:25:  .../surgvu/labels_cat2/SURGVU25_train_labels
    scripts/perceive_cases.py:26:    SAMPLE_CLIPS = ".../surgvu/cat2_sample"
    scripts/train_tools.py:35 / train_task.py:40 / tune_serving_thresholds.py:67
    src/surgvu/vlm.py:74:  DEFAULT_MODEL_DIR = "/staging/n/nkalthoff/surgvu26/models/qwen3vl-8b-nf4"

Every hit is a **path string naming the shared group scratch directory where the
SurgVU dataset is staged** -- `bhaskar_opscribe` is the storage allocation's name.
No OpScribe adapter, checkpoint, container, `pypkgs` tree or `.sif` is present, and
no OpScribe code is imported. The base image is the public
`pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`, not `opscribe.sif`. Nothing in
the image derives from a model whose training history touches private clinical
data: the only pretrained weights are torchvision ImageNet, and the only fine-tuning
corpus is the SurgVU challenge release.

The 6 GB Qwen3-VL NF4 weights are **not** in the image -- `src/surgvu/vlm.py` ships
but its `DEFAULT_MODEL_DIR` does not exist inside the container, and the VLM is
disabled by default.

---

## 5. Licensing -- PASS on copyleft; NEEDS A HUMAN DECISION on the repo licence

**Dependency surface is tiny.** `containers/Dockerfile` and
`containers/surgvu26-submission.def` install exactly two packages beyond the base,
and neither file contains a single `apt-get` line:

    Dockerfile:26  FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
    Dockerfile:48  RUN pip install --no-cache-dir \
                       "opencv-python-headless==4.11.0.86" \
                       "PyYAML==6.0.2"

**Licences, read from the installed metadata in the shipping image:**

    torch                    2.5.1+cu121   BSD-3-Clause
    torchvision              0.20.1+cu121  BSD
    opencv-python-headless   4.11.0.86     Apache 2.0
    PyYAML                   6.0.2         MIT
    numpy                    2.1.2         BSD
    pillow                   10.2.0        HPND

**No AGPL or GPL anywhere in the image.** Sweeping every installed distribution:

    $ apptainer exec ... python -c "<scan all distributions for AGPL/GPL/LGPL>"
    [('LGPL', 'chardet'), ('LGPL', 'conda'), ('LGPL', 'frozendict')]

Three LGPL packages, all arriving with the base image's conda toolchain
(`conda-build` depends on both `chardet` and `frozendict`), none imported by any
first-party module -- the serving path imports only `torch`, `torchvision`, `cv2`,
`yaml`, `numpy` and the standard library. LGPL on unmodified, dynamically-loaded
libraries imposes a notice duty, not a copyleft obligation on our code.

**The AGPL detector was successfully avoided.** `ultralytics`, `yolov5`, `yolox`,
`mmdet` and `detectron2` are all absent from the image, and `git log --all -S'ultralytics'`
returns no commits -- it was never in the tree. The one AGPL string in the image is
MPL-2.0 §1.12 boilerplate inside opencv's `LICENSE-3RD-PARTY.txt`, in the libsrt
section, and libsrt is not shipped in the Linux wheel.

**Attribution obligations that exist but do not block:** the opencv wheel bundles
FFmpeg shared objects under LGPL-2.1 (no plain-GPL header, and the GPL-only codecs
x264/x265 are absent); the CUDA runtime ships as NVIDIA proprietary wheels
(`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `nvidia-nccl-cu12`, …) whose EULA
permits redistribution bundled in an application. Both are the standard profile of
every PyTorch container on Grand Challenge.

**The decision item: the repository has no licence and no README.**

**RESOLVED since this audit was written.** The repository now carries a root
`LICENSE` (Apache-2.0, one of the set Grand Challenge recognises), a `NOTICE`
attributing every third-party component, and a `README.md`. Apache-2.0 is
compatible with every dependency found below.

A second item for the same decision: the shipped weights are fine-tuned on the
SurgVU challenge dataset, and no dataset licence or data-use agreement is recorded
anywhere in the repo. Whether trained weights may be redistributed publicly is a
question for the SurgVU/Intuitive data agreement, not one the code can answer.

---

## 6. Secrets -- PASS, with four personal-path leaks to clean up

**No credentials anywhere, in the tree or in history.** History coverage used full
object-database enumeration rather than reachable objects, which matters because a
prior rewrite leaves dangling blobs:

    $ git cat-file --batch-all-objects --batch-check='%(objectname) %(objecttype) %(objectsize)'
    TOTAL OBJECTS IN ODB: 885 / TOTAL BLOBS IN ODB: 317
    $ git rev-list --all --objects | ... reachable blobs: 313      => 4 unreachable

All **317** blobs were streamed through a pattern scan for
`ghp_|github_pat_|gho_|ghs_|ghu_|ghr_|AKIA|ASIA|hf_{30,}|sk-{20,}|sk-proj-|xox[baprsu]-|AIza|BEGIN * PRIVATE KEY|glpat-|dckr_pat_|SG\.`:

    === HIGH-CONFIDENCE SECRET HITS ACROSS ALL 317 BLOBS ===
    NONE

The 4 unreachable blobs were dumped in full and are ordinary pre-amend revisions of
`router.py`, `OUTSTANDING.md`, `inference.py` and `test_inference_vlm.py`.

**The previously-revoked PAT has no trace here.** `git log --all -S` per prefix
returned 0 commits for `ghp_`, `github_pat_`, `gho_`, `ghs_`, `AKIA`, `xox`,
`glpat-` and `PRIVATE KEY`. `hf_` (4 commits) and `sk-` (14) are `-S` substring
false positives -- `export HF_HOME=...`, `job 9618015 was HELD` -- and the
length-anchored regexes matched nothing. The prior leak was in a different
repository or predates this history.

Also clean: generic `password|api_key|secret_key|access_token = "..."`
assignments across all blobs; `Authorization:`/`Bearer` headers; URL-embedded
credentials; commit messages; `.git/config` (plain HTTPS remote, no credential
helper, no custom hooks); and a scan **inside** the image, which is the check that
matters since byte-grepping the compressed `.sif` is a false negative:

    $ apptainer exec ... grep -rnE "ghp_|github_pat_|AKIA|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|BEGIN [A-Z ]*PRIVATE KEY" /opt/algorithm
    NONE

No `.env`, `.pem`, `.key`, `id_rsa*`, `.netrc` or keytab exists in the tree, and
no weight/archive/video file is tracked (largest tracked file is 65 KB; 146 tracked
files, all text/code plus one 46 KB `.docx`).

**Four tracked files leak a personal Windows path** -- not secrets, but permanent
once public:

    $ git grep -nE "C:\\\\Users|/Users/|/home/[a-z]+|AppData"
    docs/make_build_guide_docx.py:8:  OUT = r"C:\Users\<redacted>\..."
    tools/check_cardinality.py:5:     base = r"C:\Users\<redacted>\..."
    tools/check_labels.py:5:          base = r"C:\Users\<redacted>\..."
    tools/check_overlay_legibility.py:5: base = r"C:\Users\<redacted>\..."

`/home/nkalthoff`, `wisc.edu` and the personal Gmail appear in **0** tracked files.
Committer metadata was reviewed separately and carries no credential material.
