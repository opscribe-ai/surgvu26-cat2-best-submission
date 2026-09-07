# Label-vocabulary hazards — read before writing W6a QA generation

Measured 2026-08-24 directly against
`/staging/groups/bhaskar_opscribe/surgvu/labels_cat2/SURGVU25_train_labels/case_*/`.

W6a (fine-tuning Qwen2.5-VL-7B on QA pairs generated from the logbook) is the highest-leverage
item in the v5 design: every stock VLM measured so far scores BELOW the 0.6959 zero-perception
baseline (best was 4B-fp16 open at 0.5743), and the cause is structural — gold answers
reconstruct our label taxonomy and a stock model has never seen it. Fine-tuning is what closes
that. Which means the QA generator's output IS the model, and garbage in it is not a cosmetic
problem — it is the whole result.

The raw logbook is NOT clean. Generating QA pairs straight from these columns would produce
training data that teaches the model nonsense, with nothing failing loudly.

## `groundtruth_toolname` — the hazards

| value | count | verdict |
|---|---|---|
| needle driver | 2042 | keep |
| monopolar curved scissors | 1369 | keep |
| **`nan(camera in)`** | **1277** | **DROP — not a tool.** Second most frequent value in the column. A naive generator emits "Is a nan(camera in) being used?" |
| `clip applier ` (TRAILING SPACE) | 882 | keep, but **strip** — otherwise it is a separate class from `clip applier` |
| bipolar forceps | 817 | keep |
| cadiere forceps | 662 | keep |
| prograsp forceps | 509 | keep |
| stapler | 381 | keep |
| vessel sealer | 225 | keep |
| grasping retractor | 217 | keep |
| permanent cautery hook/spatula | 168 | keep |
| **empty string** | **144** | **DROP** |
| force bipolar | 97 | keep |
| **` Single Site"`** | **29** | **DROP — malformed CSV row**, leading space and a stray quote |
| suction irrigator | 23 | out of our 12-class taxonomy — evidence only, never an answer |
| tip-up fenestrated grasper | 22 | keep |
| synchroseal | 15 | out of taxonomy |
| tenaculum forceps / potts scissors / bipolar dissector | 2 each | out of taxonomy |

**Roughly 1450 rows (~19% of the column) are not tools at all.** Whitelist against
`surgvu.taxonomy.TOOL_CLASSES` after stripping; never blacklist, or the next malformed value
walks straight into training.

## `groundtruth_taskname` — case inconsistency splits every class in two

    Suturing 775 / suturing 609
    Rectal artery/vein 291 / rectal artery/vein 286
    Retraction and collision avoidance 262 / retraction and collision avoidance 224
    Uterine horn 220 / uterine horn 186
    Suspensory ligaments 218 / suspensory ligaments 160
    Skills application 93 / skills application 83
    Range of motion 70 / range of motion 84

Every task appears in BOTH cases. Without normalisation the generator produces two labels per
task and the model learns capitalisation as a distinction. Lowercase-normalise, then map onto
`surgvu.taxonomy.TASK_CLASSES`.

Also malformed, drop them: `retraction arm positioning"`, `Retraction and Avoiding Collisions
(w/ bladder in 3rd)"`, `Retraction and Avoiding Collisions (nothing in 3rd arm)"` (1 each,
trailing quotes), ` camera adjustments` (2, leading space), `dissection` (1, not in taxonomy).
`other` (106) is a real value and needs a deliberate decision, not a silent drop.

## Column names and types — already bit us twice

See rulings R14. `tasks.csv` is `start_time`/`stop_time` (float seconds), NOT `start`/`stop`.
`tools.csv` is `install_case_time`/`uninstall_case_time` and those are **`HH:MM:SS.ffffff`
strings**, not floats. Both defects shipped in plan text and both would have produced an empty
result set behind a green test suite.

## The rule this all points at

Every QA generator must **report what it dropped and why**, and the counts must be eyeballed
against this table before a single training run. A generator that silently yields 80% of the
rows looks identical to one that yields 100%, right up until the model is wrong and nobody
knows which stage lost the data.
