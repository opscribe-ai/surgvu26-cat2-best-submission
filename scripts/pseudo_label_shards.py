"""Harvest logbook-constrained YOLO pseudo-labels from the unlabeled shards
(v5 plan4, Task 1).

THE PROBLEM. The groupmate's 14-class `best.pt` was trained on 886 hand-
labeled images. Its weak classes are exactly its RARE classes -- prograsp
forceps, clip applier, force bipolar, grasping retractor, tip-up fenestrated
grasper, vessel sealer all sit at 38-77 training instances against ~222+ for
the strong classes, and their measured recall (0.43-0.62) tracks that
shortage almost exactly. There are 38 GB of unlabeled `.npz` shards sitting
next to `best.pt` -- 235 of them, ~24,500 windows, decoded once already by
`surgvu.extract` for the CNN heads -- and this script is how the detector's
own opinion about them turns into more training data for itself.

WHY THAT IS DANGEROUS UNCONSTRAINED, AND WHY IT IS NOT HERE. Naive self-
training runs a model over unlabeled data, keeps whatever it says above some
confidence, and retrains on that: it amplifies the model's own systematic
errors, because a confident *wrong* answer looks identical to a confident
*right* one from the training loop's perspective. A detector with 0.43
recall on prograsp forceps is exactly the kind of model this failure mode
would hurt worst -- confusing it for, say, cadiere forceps and then
"confirming" that confusion by training on it.

What makes this corpus different is that every second of it already has an
independent, non-model witness: `tools.csv` records which tool was PHYSICALLY
MOUNTED at every timestamp, because that is what a circulating nurse logs in
the OR, not something inferred from pixels. A detection of a tool the
logbook says was not on the patient cart at that instant is not "a low-
confidence guess" -- it is PROVABLY WRONG, independent of how confident the
network was, and can be dropped rather than reinforced. A detection of a
tool the logbook confirms was mounted is not proof the box is in the right
place or even that the right instrument shaft is behind it, but it can no
longer be a wrong-CLASS error, which is the error mode self-training
amplifies fastest. The logbook constraint converts an error-amplifying loop
into an error-SUPPRESSING one: it can only ever throw detections away, never
invent a true positive the video didn't have, which is why Task 2 must look
at the per-class yield this script reports before committing to a 300-epoch
retrain (see the plan's Self-Review -- a class with 0.43 recall generates
few confident detections to harvest in the first place, and the logbook
constraint cannot fix that; it can only stop making it worse).

REUSING `surgvu.labels`, NOT REDERIVING IT. `CaseLabels.tools_at(part,
seconds)` already implements the two rulings that make the logbook safe to
query: R14 (`tools.csv` times are `HH:MM:SS.ffffff` STRINGS, parsed by
`parse_hms`, not floats) and R28 (every interval carries a `part`, and a
tool installed in part 1 must never leak into a part-2 query -- timestamps
RESET at the part boundary). This script never parses a CSV itself; it asks
`CaseLabels` the same question `surgvu.sampling.enumerate_windows` already
asks when it labels a window's ground-truth tool set, but at the RESOLUTION
a detection actually has: one query per (part, timestamp) per detection, not
one per 30-second window. A window's `tools` field in shard metadata is the
midpoint-only answer baked in at extraction time; this script re-derives the
per-second answer instead, because a detection is stamped with its own
anchor's exact second and deserves to be checked against that second, not
against whatever was true 15 seconds away at the window's midpoint.

THE TWO CLASSES THIS CAN NEVER CONFIRM, ON PURPOSE. `best.pt`'s vocabulary
(`surgvu.detect.YOLO_CLASSES`) has 14 entries; `surgvu.taxonomy.TOOL_CLASSES`
has 12. The two extra -- `bipolar dissector`, `suction irrigator` -- are
OUT_OF_SCOPE_TOOLS in `surgvu.taxonomy`, which means `normalize_tool` maps
them to `None` and `CaseLabels._load_tools` never puts them into any
interval at all: `tools_at(...)` can *structurally never* return either
name, regardless of what the video actually shows. A detection of one of
them therefore always fails the "is it mounted" check here and is always
tallied `unmounted` -- not because the constraint doubts the detection, but
because the logbook was never asked to carry that fact for those two
classes. This is harmless by design: both are already STRONG (recall 1.00
and 0.94 respectively) and do not need pseudo-label augmentation, so their
near-zero yield in the report below is the constraint working as intended,
not a bug to chase.

WHAT COUNTS AS "CANNOT BE RESOLVED". An empty `tools_at(...)` result is a
real, resolved fact -- "nothing was mounted at this instant" -- and every
detection at that instant is correctly `unmounted`. "Cannot be resolved" is
a narrower, harder failure: `CaseLabels.from_dir` itself raising because a
case's `tools.csv`/`tasks.csv` is missing or malformed under `--labels-root`.
When that happens, EVERY window of that case's shard is dropped under
`unresolved_window` before the detector is even run over it -- there is no
sound way to accept or reject a detection whose logbook interval does not
exist, and there is no need to pay a GPU forward pass to find that out.

R30 -- SPLIT DISCIPLINE. `config/splits_v2.json`'s `heldout` list is the 11
public sample cases -- the only cases this project ever grades against. They
are excluded from this harvest via `surgvu.sampling.normalize_case_id` on
BOTH the shard's case id and the configured heldout id, never raw string
equality (the public sample spells cases `case122`; the split spells them
`case_122`; a raw comparison finds no overlap and lets every one of them
through silently). `split_heldout_shards` FAILS LOUDLY if the exclusion
removes zero shards -- on the real 235-shard corpus against the real
11-case heldout list, some exclusion is not optional, and a zero-exclusion
result here means the comparison itself is broken, not that there was
nothing to exclude. A detector trained on the cases we grade against is
exactly the contamination that cost this project a full training run once
already (see `config/splits_v2.json`'s own `heldout_rationale`).

OUTPUT LAYOUT. Accepted detections are written in the SAME layout
`yolo_dataset.tar.gz` uses -- `images/<stem>.jpg` + `labels/<stem>.txt`,
`class cx cy w h` normalised -- so Task 2 can concatenate this directory
with the hand-labeled one without any conversion step. Filenames are
prefixed `pseudo_` (the hand-labeled set's own files are all `clip_...`),
so a directory listing alone tells a human which population a file is from
and the two can never collide. Only frames with at least one ACCEPTED
detection are ever written -- no empty/negative label files are produced --
which keeps this a small, signal-dense set rather than a bulk re-encoding of
every frame these shards hold; only the class(es) with an accepted box
appear in a written label file, never a class this window merely happens to
contain per the coarse per-window ground truth.

THE 886 HAND-LABELED IMAGES ARE NEVER TOUCHED HERE. This script only reads
`/staging/groups/bhaskar_opscribe/surgvu/shards`, the corpus the CNN heads
were trained from, which has no overlap with `yolo_dataset.tar.gz`'s frames
(different extraction pipeline entirely). Task 2 measures v2 against the
886 hand labels precisely because this script never contaminates that set
with the detector's own opinions.

WHAT THIS SCRIPT CANNOT DO. It cannot invent a true positive a class with
0.43 recall failed to notice in the first place -- see the plan's own
Self-Review. It reports the yield so Task 2 can decide, with real numbers,
whether a 300-epoch retrain is justified before spending the compute.

WHAT IS AND IS NOT TESTED WITHOUT TORCH. Every pure function below --
`shard_case_id`, `split_heldout_shards`, `classify_detection`,
`box_to_yolo_line`, `pseudo_stem`, `LogbookCache`, the manifest/report
helpers -- is exercised by `tests/test_pseudo_label.py` on the login node,
which has no torch and cannot load `best.pt`. `Detector.detect` itself, and
therefore this script's actual harvesting run over the 235 real shards, has
NOT been run anywhere by this change -- it requires the `surgvu26-train.sif`
container and a GPU, per `condor/pseudo_label.sub`.

USAGE (inside a container with torch -- see condor/pseudo_label.sub):

    python scripts/pseudo_label_shards.py \\
        --shards-dir /staging/groups/bhaskar_opscribe/surgvu/shards \\
        --labels-root /staging/groups/bhaskar_opscribe/surgvu/labels_cat2/SURGVU25_train_labels \\
        --weights /staging/groups/bhaskar_opscribe/surgvu_yolo_detector/best.pt \\
        --yolov5-dir /staging/groups/bhaskar_opscribe/surgvu_yolo_detector/yolov5 \\
        --splits config/splits_v2.json \\
        --out-root /staging/n/nkalthoff/surgvu26/pseudo_labels
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.detect import YOLO_CLASSES  # noqa: E402
from surgvu.extract import part_number, read_shard  # noqa: E402
from surgvu.labels import CaseLabels  # noqa: E402
from surgvu.sampling import normalize_case_id  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

DEFAULT_SHARDS_DIR = "/staging/groups/bhaskar_opscribe/surgvu/shards"
DEFAULT_LABELS_ROOT = (
    "/staging/groups/bhaskar_opscribe/surgvu/labels_cat2/SURGVU25_train_labels")
DEFAULT_WEIGHTS = "/staging/groups/bhaskar_opscribe/surgvu_yolo_detector/best.pt"
DEFAULT_YOLOV5_DIR = "/staging/groups/bhaskar_opscribe/surgvu_yolo_detector/yolov5"
DEFAULT_SPLITS = str(REPO / "config" / "splits_v2.json")
DEFAULT_OUT_ROOT = "/staging/n/nkalthoff/surgvu26/pseudo_labels"

#: The NMS threshold `Detector` itself is loaded at. Kept LOW, not the
#: checkpoint's validated 0.25 operating point, for the same reason
#: `scripts/detect_sample_report.py` loads it low: `conf` discards a
#: candidate BEFORE it ever reaches this script, so a detection sitting at
#: 0.20 would be invisible here rather than merely "below floor" -- and this
#: script's whole job is to tally "below floor" as its own reason, which it
#: cannot do to a detection NMS already threw away.
DEFAULT_DETECTOR_CONF = 0.01

#: The REPORTING/ACCEPTANCE floor applied here, on top of the low internal
#: `conf` above. Defaults to the checkpoint's own validated operating point
#: (0.25, the same default `detect_sample_report.py` and the shipped
#: pipeline use) rather than a stricter, invented self-training threshold --
#: the logbook constraint is what makes wrong-CLASS errors safe to drop
#: automatically; a caller who wants extra conservatism on box confidence can
#: raise this independently via --conf-floor.
DEFAULT_CONF_FLOOR = 0.25

DEFAULT_IOU = 0.45

#: Exact per-class TRAIN-split box-instance counts in the 886-image
#: hand-labeled set (`yolo_dataset.tar.gz`'s `train.txt` against its
#: `labels/*.txt`, counted directly, 2026-08-25) -- this is what `best.pt`
#: actually trained on, per class. NOT the plan document's illustrative
#: "38-77 vs 222" range, which describes measured VAL-split recall context,
#: not a full instance table; this table is the exact, reproducible number
#: Task 2 needs to judge a class's yield against what it started with.
HAND_LABELED_TRAIN_INSTANCES = {
    "bipolar dissector": 79,
    "bipolar forceps": 366,
    "cadiere forceps": 204,
    "clip applier": 90,
    "force bipolar": 126,
    "grasping retractor": 102,
    "monopolar curved scissors": 159,
    "needle driver": 257,
    "permanent cautery hook/spatula": 92,
    "prograsp forceps": 98,
    "stapler": 94,
    "suction irrigator": 99,
    "tip-up fenestrated grasper": 91,
    "vessel sealer": 104,
}
if set(HAND_LABELED_TRAIN_INSTANCES) != set(YOLO_CLASSES):
    # A bare `assert` would be silently stripped under `python -O`; this
    # table's correctness is load-bearing for the yield report (a missing
    # class would print as a bogus 0-baseline "infinite" gain), so the check
    # must survive regardless of how the interpreter is invoked.
    raise ValueError(
        "HAND_LABELED_TRAIN_INSTANCES must carry exactly the detector's 14 "
        "classes; got %s" % (sorted(HAND_LABELED_TRAIN_INSTANCES),))

_YOLO_INDEX = {name: i for i, name in enumerate(YOLO_CLASSES)}


# ---------------------------------------------------------------------------
# Shard / heldout bookkeeping -- pure, torch-free
# ---------------------------------------------------------------------------

def shard_case_id(path):
    """'case_073_part1.npz' -> 'case_073'.

    Mirrors `surgvu.dataset.shard_paths_for_split`'s own `rsplit("_part", 1)`
    idiom rather than inventing a second way to recover a case id from a
    shard filename -- `surgvu.extract.shard_filename` is the one place that
    name is constructed, and this is its inverse.
    """
    return Path(path).name.rsplit("_part", 1)[0]


def split_heldout_shards(shard_paths, heldout_ids):
    """(kept, excluded) shard paths, matching case ids via `normalize_case_id`.

    Comparison is via `surgvu.sampling.normalize_case_id` on BOTH sides,
    never raw string equality -- see the module docstring for why a raw
    comparison silently lets every heldout case through. FAILS LOUDLY
    (ruling R30) if the exclusion removes zero shards: over the real corpus
    against the real heldout list, some exclusion is not optional, and zero
    here means the id comparison itself is broken rather than that nothing
    needed excluding.
    """
    heldout_norm = {normalize_case_id(c) for c in heldout_ids}
    kept, excluded = [], []
    for path in shard_paths:
        case_norm = normalize_case_id(shard_case_id(path))
        (excluded if case_norm in heldout_norm else kept).append(path)
    if not excluded:
        raise RuntimeError(
            "heldout exclusion removed ZERO of %d shard(s) against %d "
            "configured heldout case(s) (%s). A detector trained on the "
            "cases we grade against is the same contamination that cost "
            "this project a full training run once already -- refusing to "
            "trust a 'nothing to exclude' result this suspicious rather "
            "than silently proceed on a possibly-broken comparison."
            % (len(shard_paths), len(heldout_norm), sorted(heldout_norm)))
    return kept, excluded


def load_heldout_ids(splits_path):
    """The raw (possibly `caseNNN`-spelled) `heldout` list from splits_v2.json."""
    payload = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    heldout = payload.get("heldout")
    if not heldout:
        raise ValueError(
            "%s has no non-empty 'heldout' list -- refusing to run a "
            "harvest with nothing configured to exclude." % (splits_path,))
    return heldout


# ---------------------------------------------------------------------------
# Logbook lookups -- pure, torch-free (CaseLabels.from_dir touches disk, not
# torch)
# ---------------------------------------------------------------------------

class LogbookCache:
    """Loads each case's `CaseLabels` at most once, from `labels_root`.

    A case whose `tools.csv`/`tasks.csv` cannot be loaded is not "nothing is
    mounted" -- `tools_at` already reports that fact honestly by returning an
    empty set. It is "the logbook interval cannot be resolved at all", and
    `get` returns `None` for it so every detection in every window of that
    case is tallied `unresolved_window` rather than silently treated as
    unmounted.
    """

    def __init__(self, labels_root):
        self.labels_root = Path(labels_root)
        self._cache = {}

    def get(self, case_id):
        if case_id not in self._cache:
            case_dir = self.labels_root / case_id
            try:
                self._cache[case_id] = CaseLabels.from_dir(case_dir)
            except (OSError, ValueError) as exc:
                print("pseudo_label_shards: cannot resolve the logbook for "
                     "%s (%s): %s" % (case_id, case_dir, exc))
                self._cache[case_id] = None
        return self._cache[case_id]


# ---------------------------------------------------------------------------
# Per-detection classification -- pure, torch-free
# ---------------------------------------------------------------------------

def classify_detection(cls_name, conf, case_labels, part, seconds, conf_floor):
    """One detection -> `("accept", None)` or `("drop", reason)`.

    Confidence is judged BEFORE the logbook lookup, so a detection that
    would be dropped either way never bothers querying `tools_at` -- this is
    a performance choice, not a tallying one (a detection cannot earn two
    reasons).

    The detector's two out-of-taxonomy classes (`bipolar dissector`,
    `suction irrigator`) can NEVER pass the `tools_at` check: see the module
    docstring's "THE TWO CLASSES THIS CAN NEVER CONFIRM" section. Every
    detection of either is tallied `unmounted` here by construction, not
    because the logbook disagrees with the video.
    """
    if conf < conf_floor:
        return "drop", "low_confidence"
    if case_labels is None:
        return "drop", "unresolved_window"
    if cls_name not in case_labels.tools_at(part, seconds):
        return "drop", "unmounted"
    return "accept", None


# ---------------------------------------------------------------------------
# YOLO label formatting -- pure, torch-free
# ---------------------------------------------------------------------------

def box_to_yolo_line(cls_index, box_xyxy, img_w, img_h):
    """`[x1, y1, x2, y2]` pixel box -> `'cls cx cy w h'` normalised, or `None`.

    Clamped to the frame rather than dropped outright: `scale_coords` can
    return a box a fraction of a pixel outside the frame after rounding, and
    a strict in-bounds check would throw away a real, logbook-confirmed
    detection over a rounding artefact. A box that clamps to zero width or
    height (i.e. it was entirely outside the frame to begin with) returns
    `None` -- writing a degenerate box would poison training more than
    dropping a marginal detection would.
    """
    x1, y1, x2, y2 = box_xyxy
    x1 = min(max(x1, 0.0), img_w)
    x2 = min(max(x2, 0.0), img_w)
    y1 = min(max(y1, 0.0), img_h)
    y2 = min(max(y2, 0.0), img_h)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return "%d %.6f %.6f %.6f %.6f" % (
        int(cls_index), cx / img_w, cy / img_h, w / img_w, h / img_h)


def pseudo_stem(case, part, window_index, anchor_index):
    """Deterministic, collision-free basename for one accepted frame.

    Keyed on the window's own POSITION in its shard's `meta` list, not its
    start time rounded to an int: two windows in the same case/part can
    start close enough together (adjacent task segments) for rounded starts
    to collide, while a position in the shard's own list is unique by
    construction. Prefixed `pseudo_`, never `clip_` (the hand-labeled set's
    own prefix), so the two populations cannot collide when Task 2
    concatenates them and a directory listing alone says which is which.
    """
    return "pseudo_%s_part%s_w%04d_f%02d" % (
        case, part_number(part), int(window_index), int(anchor_index))


# ---------------------------------------------------------------------------
# Reporting -- pure, torch-free
# ---------------------------------------------------------------------------

def format_yield_table(per_class_new, baseline=HAND_LABELED_TRAIN_INSTANCES):
    """Human-readable per-class yield table: new pseudo-instances vs baseline.

    This is the decision-relevant number in the whole script (see the module
    docstring): Task 2 should not spend a 300-epoch run on a class whose
    yield is a rounding error next to its hand-labeled count.
    """
    lines = [
        "%-32s %10s %10s %8s" % ("class", "baseline", "new", "x"),
        "-" * 62,
    ]
    for name in YOLO_CLASSES:
        base = baseline.get(name, 0)
        new = per_class_new.get(name, 0)
        ratio = "%.2fx" % ((base + new) / base) if base else "n/a"
        lines.append("%-32s %10d %10d %8s" % (name, base, new, ratio))
    return "\n".join(lines)


def build_report(kept_shards, excluded_shards, tally, per_class_new,
                 windows_total, images_written_total, args):
    return {
        "shards_total_on_disk": len(kept_shards) + len(excluded_shards),
        "shards_heldout_excluded": len(excluded_shards),
        "shards_heldout_excluded_names": sorted(Path(p).name for p in excluded_shards),
        "shards_eligible": len(kept_shards),
        "windows_processed": windows_total,
        "images_written": images_written_total,
        "drop_tally": dict(tally),
        "per_class_new_instances": dict(per_class_new),
        "hand_labeled_train_instances": dict(HAND_LABELED_TRAIN_INSTANCES),
        "conf_floor": args.conf_floor,
        "detector_conf": args.detector_conf,
        "iou": args.iou,
        "shards_dir": str(args.shards_dir),
        "labels_root": str(args.labels_root),
        "splits": str(args.splits),
        "out_root": str(args.out_root),
    }


# ---------------------------------------------------------------------------
# Resume bookkeeping -- pure, torch-free
# ---------------------------------------------------------------------------

def load_processed_shards(manifest_path):
    """Shard filenames already recorded in `manifest_path`, or empty.

    A resubmitted job after a Condor eviction skips every shard already
    accounted for -- one shard is a substantial, atomic unit of GPU work (up
    to ~160 windows), so resuming at shard granularity is cheap to implement
    and avoids redoing hours of work over a corpus this large. A malformed
    trailing line (a partial write from a job killed mid-line) is skipped
    rather than raising, mirroring `scripts/cache_evidence.py`'s
    `load_cached_keys`.
    """
    path = Path(manifest_path)
    done = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = record.get("shard")
            if name:
                done.add(name)
    return done


# ---------------------------------------------------------------------------
# read_shard_safely touches only cv2/numpy (no torch) and is exercised on the
# login node. process_shard touches torch indirectly, via `detector.detect`.
# ---------------------------------------------------------------------------

def read_shard_safely(path):
    """`(frames, meta)` for one shard, or `(None, None)` on any read failure.

    This is the ONLY catch-and-continue boundary in this script's main loop,
    and it exists for exactly one failure mode: a truncated or corrupt
    `.npz` -- a data-integrity problem with ONE shard among 235, safe to
    tally and skip. An exception raised anywhere else in a shard's
    processing (the detector failing to load, torch being absent, a full
    disk on the write) is a bug in the RUN itself, not a property of that
    one shard, and must propagate and fail the job loudly -- see `run`,
    which deliberately does NOT wrap `process_shard` in a try/except. Wrapping
    both failure modes in one `except Exception` would let a broken run (say,
    a bad `--weights` path) silently relabel every one of 235 shards as
    "corrupt" in the report and complete as if nothing were wrong.
    """
    try:
        return read_shard(path)
    except Exception as exc:                                    # noqa: BLE001
        print("pseudo_label_shards: cannot read %s: %s" % (path, exc))
        return None, None


def process_shard(frames, meta, detector, logbook, conf_floor, images_dir,
                  labels_dir):
    """One already-read shard's `(frames, meta)` -> written pseudo-labels.

    Returns `(n_windows, n_images_written, tally, per_class_new)`. Takes the
    decoded shard rather than a path so a read failure (see
    `read_shard_safely`) and everything that can go wrong AFTER a
    successful read stay in clearly separate code paths.
    """
    import cv2
    import numpy as np

    tally = Counter()
    per_class_new = Counter()
    images_written = 0

    for w, row in enumerate(meta):
        case, part = row["case"], row["part"]
        start, fps = float(row["start"]), float(row["fps"])
        depth = len(frames[w])

        case_labels = logbook.get(case)
        if case_labels is None:
            # The whole window is unresolvable; do not even decode/run the
            # detector over frames whose logbook check can only ever fail.
            tally["windows_dropped_unresolved_logbook"] += 1
            continue

        stack = np.stack([frames[w][i] for i in range(depth)])
        detections = detector.detect(stack)
        img_h, img_w = stack.shape[1], stack.shape[2]

        for anchor_idx, found in enumerate(detections):
            seconds = start + anchor_idx / fps
            accepted_lines = []
            for item in found:
                verdict, reason = classify_detection(
                    item["cls"], item["conf"], case_labels, part, seconds,
                    conf_floor)
                if verdict == "drop":
                    tally["detections_dropped_%s" % reason] += 1
                    continue
                line = box_to_yolo_line(
                    _YOLO_INDEX[item["cls"]], item["box"], img_w, img_h)
                if line is None:
                    tally["detections_dropped_degenerate_box"] += 1
                    continue
                accepted_lines.append(line)
                per_class_new[item["cls"]] += 1

            if not accepted_lines:
                continue

            stem = pseudo_stem(case, part, w, anchor_idx)
            frame = stack[anchor_idx]
            # NO params argument. The login node's cv2 4.13.0 accepts
            # `imwrite(path, img, [IMWRITE_JPEG_QUALITY, q])` both positionally
            # and by keyword, but the TRAIN CONTAINER's build rejects it:
            # cluster 9698231 died with "imwrite() takes 2 positional arguments
            # but 3 were given" after processing 5 shards and writing nothing.
            # Rather than guess which build is in the image, drop the parameter
            # -- OpenCV's default JPEG quality is 95, and 95-vs-90 on a training
            # crop is immaterial next to a job that does not run at all.
            ok = cv2.imwrite(str(images_dir / ("%s.jpg" % stem)), frame)
            if not ok:
                raise ValueError(
                    "cv2.imwrite failed for %s (case %s, window %d, anchor "
                    "%d)" % (stem, case, w, anchor_idx))
            (labels_dir / ("%s.txt" % stem)).write_text(
                "\n".join(accepted_lines) + "\n", encoding="utf-8")
            images_written += 1

    n_windows = len(meta)
    return n_windows, images_written, tally, per_class_new


def run(args):
    """The whole harvest: shards -> heldout exclusion -> per-shard pseudo-
    labels -> a drop-count report and per-class yield table."""
    shard_paths = sorted(Path(args.shards_dir).glob("*.npz"))
    if not shard_paths:
        raise SystemExit("no *.npz shards under %s" % (args.shards_dir,))

    heldout_ids = load_heldout_ids(args.splits)
    kept, excluded = split_heldout_shards(shard_paths, heldout_ids)
    print("pseudo_label_shards: %d shard(s) on disk, %d excluded as heldout "
         "(R30), %d eligible" % (len(shard_paths), len(excluded), len(kept)))

    images_dir = Path(args.out_root) / "images"
    labels_dir = Path(args.out_root) / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    already_done = load_processed_shards(manifest_path)
    if already_done:
        print("pseudo_label_shards: resuming, %d shard(s) already recorded "
             "in %s" % (len(already_done), manifest_path))
    to_process = [p for p in kept if p.name not in already_done]
    if args.limit is not None:
        to_process = to_process[:args.limit]
    print("pseudo_label_shards: %d shard(s) to process this run"
         % (len(to_process),))

    device = args.device
    if device in (None, "", "auto"):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("pseudo_label_shards: device=%s" % (device,))

    from surgvu.detect import Detector
    detector = Detector(args.weights, args.yolov5_dir, conf=args.detector_conf,
                        iou=args.iou, device=device)
    logbook = LogbookCache(args.labels_root)

    tally = Counter()
    per_class_new = Counter()
    windows_total = 0
    images_written_total = 0

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as manifest:
        for i, path in enumerate(to_process, start=1):
            frames, meta = read_shard_safely(path)
            if frames is None:
                tally["shards_dropped_unreadable"] += 1
                manifest.write(json.dumps(
                    {"shard": path.name, "status": "unreadable"}) + "\n")
                manifest.flush()
                continue

            # NOT wrapped in try/except, deliberately -- see
            # read_shard_safely's docstring. Anything raised past this point
            # (the detector failing to load, torch missing, a full disk) is
            # a bug in the run and must crash it loudly rather than being
            # miscounted as this shard being corrupt.
            n_windows, n_images, shard_tally, shard_per_class = process_shard(
                frames, meta, detector, logbook, args.conf_floor, images_dir,
                labels_dir)

            tally.update(shard_tally)
            per_class_new.update(shard_per_class)
            windows_total += n_windows
            images_written_total += n_images
            manifest.write(json.dumps(
                {"shard": path.name, "status": "ok", "windows": n_windows,
                 "images": n_images}) + "\n")
            manifest.flush()

            if i % 10 == 0 or i == len(to_process):
                print("pseudo_label_shards: %d/%d shards, %d windows, "
                     "%d images written so far"
                     % (i, len(to_process), windows_total,
                        images_written_total), flush=True)

    report = build_report(kept, excluded, tally, per_class_new, windows_total,
                          images_written_total, args)
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 62)
    print("PER-CLASS YIELD (new pseudo-label instances vs hand-labeled train set)")
    print(format_yield_table(per_class_new))
    print("=" * 62)
    print("DROP TALLY: %s" % dict(tally))
    print("wrote report to %s" % (args.report_out,))
    return report


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", default=DEFAULT_SHARDS_DIR)
    parser.add_argument("--labels-root", default=DEFAULT_LABELS_ROOT)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--yolov5-dir", default=DEFAULT_YOLOV5_DIR)
    parser.add_argument("--splits", default=DEFAULT_SPLITS)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--manifest", default=None,
                        help="default: <out-root>/pseudo_label_manifest.jsonl")
    parser.add_argument("--report-out", default=None,
                        help="default: <out-root>/pseudo_label_report.json")
    parser.add_argument("--conf-floor", type=float, default=DEFAULT_CONF_FLOOR)
    parser.add_argument("--detector-conf", type=float,
                        default=DEFAULT_DETECTOR_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap on NEW shards processed this run (smoke test)")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.manifest is None:
        args.manifest = str(Path(args.out_root) / "pseudo_label_manifest.jsonl")
    if args.report_out is None:
        args.report_out = str(Path(args.out_root) / "pseudo_label_report.json")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
