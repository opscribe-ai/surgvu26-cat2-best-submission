"""Combine the 886-image hand-labeled YOLO set with the logbook-constrained
pseudo-labels into one training corpus for detector v2 (v5 plan4, Task 2).

WHY THIS IS A SEPARATE STEP FROM TRAINING. `train_detector_v2.sh` calls this
before it ever invokes `yolov5/train.py`, so the corpus-construction
decisions below -- what gets kept, what gets capped, what gets excluded --
are inspectable and testable independently of a multi-hour GPU run. Every
function here is pure Python (paths, small text files, one JSON report); the
886 hand-labeled images and the pseudo-labeled pool are read from disk but
never through torch, so this whole module is exercised on the login node by
tests/test_build_detector_v2_dataset.py. It is NOT run against the real
601K-image pool from the login node -- see the module's own CLI docstring.

THE RATIO PROBLEM THIS EXISTS TO SOLVE. 886 hand-labeled images against
601,261 pseudo-labeled ones is a 679:1 imbalance. Concatenating the two
pools verbatim would make an "epoch" over the combined set >99.8% teacher
opinion and <0.2% human ground truth -- the 886 images the detector's
measured 77%/74% P/R came from would vanish into noise, and any signal the
pseudo pool carries (right OR wrong) would dominate purely by volume. This
script's job is to build a corpus where that cannot happen:

  1. HAND-LABELED IMAGES ARE OVERSAMPLED, not the pseudo pool downsampled to
     match them -- oversampling repeats real, human-verified boxes; the
     alternative (throwing away all but 886-worth of pseudo images) would
     discard exactly the volume Task 1 was run to gain. `--hand-oversample`
     (default 20) repeats every hand-labeled train image that many times in
     the manifest. Images that additionally contain a ZERO-GAIN class
     (`bipolar dissector` or `suction irrigator` -- see below) get an EXTRA
     `--zero-gain-oversample-boost` repeats on top of the base factor.

  2. THE PSEUDO POOL IS CAPPED, NOT TAKEN WHOLESALE, but not uniformly:
     classes whose pseudo yield multiplier ((hand-labeled + new) / hand-
     labeled) is at or below `--protect-below-multiplier` (default 100) are
     the genuine long-tail wins Task 1 was run to find -- tip-up fenestrated
     grasper (~6.8x), stapler (~25x), grasping retractor (~66x) -- and every
     pseudo image containing one of them is KEPT, uncapped. Every other
     pseudo image (containing only already-abundant classes: needle driver
     ~1884x, monopolar curved scissors ~1526x, cadiere forceps ~774x,
     bipolar forceps ~574x, and the middling classes 150x-350x) is subject
     to `--pseudo-common-cap` (default 20,000), a seeded uniform subsample.
     A single random subsample over the whole 601K pool would apply the SAME
     shrinkage to a class with 530 new instances as to one with 484,021 --
     which is exactly how a class this project already measured as a weak
     spot could be quietly erased by this step instead of helped by it.

  3. TWO CLASSES GAIN NOTHING FROM PSEUDO-LABELING AT ALL. `bipolar
     dissector` and `suction irrigator` are OUT_OF_TAXONOMY (see
     `surgvu.detect`): the logbook can structurally never confirm either
     (`surgvu.taxonomy.normalize_tool` maps both to `None`, so
     `CaseLabels.tools_at` can never return them), so every pseudo detection
     of either is tallied `unmounted` and zero pseudo images ever carry them.
     Oversampling step 1 above cannot invent pseudo signal for a class that
     structurally has none -- what it CAN do is stop these two classes' per-
     epoch presence from shrinking just because the rest of the corpus grew
     around them. `zero_gain_classes_from_report` reads which classes these
     are directly off the pseudo run's own report (never hardcoded), and the
     extra oversample boost in step 1 is the one lever this script has to
     partially counteract their relative dilution. It is NOT a fix -- see
     the module docstring's own warning against calling this solved -- it is
     the honest, bounded thing oversampling a fixed 886-image pool CAN do.
     Task 2's own per-class recall report on the untouched 240-image
     hand-labeled val split is what actually answers whether it worked.

R30 -- SPLIT DISCIPLINE, TWO INDEPENDENT LAYERS.

  Layer 1 (provenance, not re-derivation): `verify_pseudo_harvest_excluded_
  heldout` reads `pseudo_label_report.json`'s own `shards_heldout_excluded`
  field and FAILS LOUDLY if it is zero or absent. On the real report (14 of
  235 shards excluded, matching the 11-case heldout list) this passes; it
  exists to catch a future re-run of the harvest whose own R30 guard
  regressed, before this script ever touches its output.

  Layer 2 (independent re-derivation, no trust): `assert_no_heldout_pseudo_
  images` parses the case id back out of every candidate pseudo image's own
  filename (`pseudo_<case>_part<N>_w####_f##.jpg`) via `surgvu.sampling.
  normalize_case_id` -- never string equality -- and RAISES if any resolve
  into the heldout set. This does not trust Layer 1's bookkeeping at all: it
  is checking the actual filenames this script is about to add to train.txt.
  On the real (already-clean) pool this finds zero, which is the CORRECT,
  expected outcome here -- unlike `pseudo_label_shards.split_heldout_shards`,
  finding zero here is not itself suspicious, because layer 1 already
  established that some exclusion happened upstream. Finding ANY is
  raised as a hard failure rather than silently dropped and logged: a
  contamination this severe, on a pool whose own report claims to be clean,
  means either the harvest or this check is broken, not merely "a few extra
  images to remove and move on".

THE 240-IMAGE VAL SPLIT IS NEVER TOUCHED. `run` copies the hand-labeled
`val.txt` through verbatim (resolved to absolute paths) and never merges a
single pseudo-labeled path into it -- v2 must be measured against human
labels, not against v1's (or a v2 teacher pass's) own opinions, per the
plan's own Task 2 instruction.

USAGE (offline construction step, run at the start of a compute-node job --
see condor/train_detector_v2.sh; NOT run against the real 601K-image pool
from the login node, per the project's no-heavy-work-on-ap2001 guardrail):

    python scripts/build_detector_v2_dataset.py \\
        --hand-root /staging/n/nkalthoff/surgvu26/detector_v2/hand_labeled \\
        --pseudo-images-dir /staging/n/nkalthoff/surgvu26/pseudo_labels/images \\
        --pseudo-labels-dir /staging/n/nkalthoff/surgvu26/pseudo_labels/labels \\
        --pseudo-report /staging/n/nkalthoff/surgvu26/pseudo_labels/pseudo_label_report.json \\
        --splits config/splits_v2.json \\
        --out-dir /staging/n/nkalthoff/surgvu26/detector_v2/dataset
"""
import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.detect import YOLO_CLASSES  # noqa: E402
from surgvu.sampling import normalize_case_id  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pseudo_label_shards import load_heldout_ids  # noqa: E402

#: Threads used to read the pseudo-label pool's 601,261 label files.
#:
#: Sized from a measurement, not a guess. Serial cold reads on this ceph mount
#: run ~24 files/s (~40 ms each) -- about 7 hours for the pool, and a real run
#: confirmed it: 63 minutes of wall clock for 1 second of CPU. Cold threaded
#: reads measured 276 files/s at 16 threads, 196 at 32, 265 at 64: the ceiling
#: is the filesystem, not the pool, so more threads buy nothing and 16 is
#: chosen as the smallest count that reaches it. ~10x serial, which turns the
#: pass from unusable into ~40 minutes.
LABEL_READ_THREADS = 16

REPO = Path(__file__).resolve().parents[1]

DEFAULT_HAND_ROOT = "/staging/n/nkalthoff/surgvu26/detector_v2/hand_labeled"
DEFAULT_PSEUDO_IMAGES_DIR = "/staging/n/nkalthoff/surgvu26/pseudo_labels/images"
DEFAULT_PSEUDO_LABELS_DIR = "/staging/n/nkalthoff/surgvu26/pseudo_labels/labels"
DEFAULT_PSEUDO_REPORT = "/staging/n/nkalthoff/surgvu26/pseudo_labels/pseudo_label_report.json"
DEFAULT_SPLITS = str(REPO / "config" / "splits_v2.json")
DEFAULT_OUT_DIR = "/staging/n/nkalthoff/surgvu26/detector_v2/dataset"

DEFAULT_HAND_OVERSAMPLE = 20
DEFAULT_ZERO_GAIN_OVERSAMPLE_BOOST = 20
DEFAULT_PSEUDO_COMMON_CAP = 20000
DEFAULT_PROTECT_BELOW_MULTIPLIER = 100.0
DEFAULT_SEED = 0

_PSEUDO_STEM_RE = re.compile(r"^pseudo_(case_\d+)_part\d+_w\d+_f\d+$")


# ---------------------------------------------------------------------------
# Manifest I/O -- pure, torch-free
# ---------------------------------------------------------------------------

def read_txt_list(path):
    """Non-empty, stripped lines of a YOLOv5-style manifest (train.txt/val.txt)."""
    lines = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return lines


def resolve_hand_paths(lines, root):
    """Manifest lines (relative to `root`) -> existing absolute `Path`s.

    Raises if a line does not resolve to a real file. A hand-labeled
    manifest line that silently fails to resolve would shrink the ONE
    population in this build that is not capped or subsampled anywhere --
    the 886/240 human-verified images -- without a trace.

    Breaks if: the `exists()` check is removed and a missing file is
    allowed through as a plain (possibly bogus) path.
    """
    root = Path(root)
    resolved = []
    for line in lines:
        path = (root / line).resolve()
        if not path.exists():
            raise FileNotFoundError(
                "%s (from manifest line %r under %s) does not exist -- a "
                "hand-labeled path that does not resolve to a real file "
                "would silently shrink the human-labeled training set."
                % (path, line, root))
        resolved.append(path)
    return resolved


def image_path_to_label_path(image_path):
    """`.../images/x.jpg` -> `.../labels/x.txt`, replacing the LAST
    '/images/' segment -- mirrors yolov5.utils.dataloaders.img2label_paths's
    own `rsplit(sep, 1)` exactly, so this script's notion of "an image's
    label file" never disagrees with what yolov5's own dataloader will
    resolve for the same path at train time.

    Breaks if: this is changed to replace the FIRST '/images/' occurrence,
    or a plain (non-rsplit) `.replace()`, either of which silently picks the
    wrong label file for a path with more than one 'images' segment.
    """
    text = str(image_path)
    sep = "/images/"
    if sep not in text:
        raise ValueError(
            "%r has no '/images/' path segment; cannot derive its label "
            "path under the yolo_dataset images<->labels convention."
            % (text,))
    head, _, tail = text.rpartition(sep)
    label_tail = Path(tail).with_suffix(".txt").as_posix()
    return Path(head + "/labels/" + label_tail)


def read_label_classes(label_path, class_names=YOLO_CLASSES):
    """The set of class NAMES (not indices) present in one YOLO label file.

    Raises on a class index outside `class_names` -- this label file and
    `YOLO_CLASSES` disagreeing about the vocabulary is a contract violation,
    not a class this function can guess at or skip.

    Breaks if: the range check `0 <= idx < len(class_names)` is removed,
    turning an out-of-range index into an IndexError deep in a dict lookup
    (or, worse, a wraparound via negative indexing) instead of a clear
    contract-violation message.
    """
    names = set()
    text = Path(label_path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        idx = int(line.split()[0])
        if not (0 <= idx < len(class_names)):
            raise ValueError(
                "%s: class index %d is out of range for a %d-class "
                "vocabulary -- this label file and YOLO_CLASSES disagree "
                "about the class contract." % (label_path, idx, len(class_names)))
        names.add(class_names[idx])
    return frozenset(names)


def list_pseudo_pairs(images_dir, labels_dir):
    """Every `(image_path, label_path)` pair under the pseudo-label pool.

    Iterates the LABELS dir, not images -- `pseudo_label_shards.py` only
    ever writes a label file for a frame with at least one accepted
    detection, so the labels dir is the authoritative list of "this frame
    became a pseudo-labeled example". Raises if a label file's matching
    image is missing: the harvest script writes both atomically per
    accepted frame, so a label without its image means the pool itself is
    corrupt, not merely sparse.

    THE EXISTENCE CHECK IS A SET LOOKUP, NOT A stat() PER LABEL, and on this
    filesystem that is the difference between usable and not. The pool holds
    601,261 label files on ceph, where a per-file metadata round trip measures
    ~40 ms; 601k of them is ~7 hours of pure latency. One `os.scandir` of the
    images directory measures ~33,000 entries/s (200k in 6 s), so the whole
    listing costs ~18 seconds and every check after it is a hash lookup. The
    RAISE BEHAVIOUR IS UNCHANGED -- a label whose image is missing still fails
    loudly, which is the invariant that matters.

    Breaks if: the membership check is removed and a label file's missing
    image is silently skipped instead of raised, or if this reverts to
    `image_path.exists()` per label (correct, but ~7 hours slower).
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    image_stems = {
        entry.name[:-4]
        for entry in os.scandir(str(images_dir))
        if entry.name.endswith(".jpg")
    }
    pairs = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = images_dir / (label_path.stem + ".jpg")
        if label_path.stem not in image_stems:
            raise FileNotFoundError(
                "%s has a pseudo-label file but no matching image at %s -- "
                "the harvest script writes both atomically per accepted "
                "frame, so a label without its image means the pool is "
                "corrupt, not merely sparse." % (label_path, image_path))
        pairs.append((image_path, label_path))
    return pairs


def write_lines(path, lines):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "\n".join(str(x) for x in lines) + ("\n" if lines else ""),
        encoding="utf-8")


def write_dataset_yaml(path, train_txt, val_txt, class_names=YOLO_CLASSES):
    lines = ["nc: %d" % len(class_names), "names:"]
    lines += ["- %s" % name for name in class_names]
    lines += ["train: %s" % train_txt, "val: %s" % val_txt]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# R30 -- split discipline, two independent layers (see module docstring)
# ---------------------------------------------------------------------------

def verify_pseudo_harvest_excluded_heldout(report):
    """Layer 1: trust the harvest's OWN recorded exclusion count, verified.

    Breaks if: the `if not excluded` guard is deleted or weakened so a
    missing/zero `shards_heldout_excluded` field is accepted silently.
    """
    excluded = report.get("shards_heldout_excluded")
    if not excluded:
        raise ValueError(
            "pseudo_label_report.json's 'shards_heldout_excluded' is %r -- "
            "the harvest that produced this pool is expected to have "
            "excluded the 11 graded cases (R30). Zero here means either "
            "the harvest's own exclusion guard never ran or this report "
            "was regenerated incorrectly; refusing to build a training "
            "corpus from a pseudo-label pool whose own provenance cannot "
            "vouch for R30 compliance." % (excluded,))


def pseudo_case_id_from_path(path):
    """`pseudo_case_073_part1_w0000_f00.jpg` -> normalized `'case_073'`.

    Raises on anything not matching the harvest script's own `pseudo_stem`
    format -- an unrecoverable case id cannot be checked against the
    heldout list, and silently skipping it would be the same failure shape
    R30 already burned this project on once.
    """
    stem = Path(path).stem
    match = _PSEUDO_STEM_RE.match(stem)
    if match is None:
        raise ValueError(
            "%r does not look like a pseudo-label filename "
            "('pseudo_case_NNN_partP_wWWWW_fFF'); cannot recover its case "
            "id, and an id that cannot be recovered cannot be checked "
            "against the heldout list." % (str(path),))
    return normalize_case_id(match.group(1))


def assert_no_heldout_pseudo_images(pseudo_image_paths, heldout_ids):
    """Layer 2: independent, no-trust re-derivation from filenames.

    Compares via `normalize_case_id` on BOTH sides, never raw string
    equality -- the exact bug shape ('case122' vs 'case_122') that let
    every graded case through a raw comparison once already on this
    project. Finding zero here is the CORRECT, expected outcome on the real
    (already-clean) pool; finding ANY raises rather than silently
    excluding-and-continuing, because a contamination this severe on a pool
    whose own report claims to be clean means something is broken, not
    merely "a few extra images".

    Breaks if: `normalize_case_id` is swapped for a raw string comparison
    (a heldout id spelled 'case122' would then never match a pseudo path
    spelled 'case_122', and this function would silently pass on a
    contaminated pool), or if the `raise` is replaced with a log-and-drop.
    """
    heldout_norm = {normalize_case_id(c) for c in heldout_ids}
    found = sorted({pseudo_case_id_from_path(p) for p in pseudo_image_paths}
                   & heldout_norm)
    if found:
        raise RuntimeError(
            "%d heldout case(s) found among the pseudo-labeled images this "
            "build was about to include (%s) -- pseudo_label_shards.py's "
            "own R30 exclusion should already have removed every one of "
            "these upstream. Refusing to silently drop them and continue: "
            "this indicates either the upstream harvest or this "
            "verification itself is broken." % (len(found), ", ".join(found)))


# ---------------------------------------------------------------------------
# Per-class yield -> which classes get protected from the common-pool cap
# ---------------------------------------------------------------------------

def compute_multipliers(baseline, new_counts, class_names=YOLO_CLASSES):
    """`(hand-labeled + pseudo new) / hand-labeled`, per class.

    Raises if a class has no positive hand-labeled baseline -- a multiplier
    against zero is undefined, and treating it as "infinite yield" would be
    a silent, made-up number standing in for a class this script cannot
    actually rank.

    Breaks if: the `if not base: raise` guard is removed, letting a
    zero/missing baseline divide silently (ZeroDivisionError, or worse, a
    swallowed exception turning it into an arbitrary default).
    """
    multipliers = {}
    for name in class_names:
        base = baseline.get(name)
        if not base:
            raise ValueError(
                "%r has no positive hand-labeled baseline count in the "
                "pseudo report -- a yield multiplier against zero is "
                "undefined." % (name,))
        new = new_counts.get(name, 0)
        multipliers[name] = (base + new) / base
    return multipliers


def protected_classes_from_multipliers(multipliers, threshold):
    """Classes whose yield multiplier is AT OR BELOW `threshold`.

    These are the classes Task 1's report says gained the LEAST relative to
    what they started with -- exactly the ones a uniform random subsample of
    the pseudo pool would be most likely to erase.
    """
    return frozenset(name for name, mult in multipliers.items() if mult <= threshold)


def zero_gain_classes_from_report(report, class_names=YOLO_CLASSES):
    """Classes with literally zero new pseudo instances (never hardcoded).

    On the real report this is `{'bipolar dissector', 'suction irrigator'}`
    -- both OUT_OF_TAXONOMY and structurally unconfirmable by the logbook
    (see `surgvu.detect`/`surgvu.taxonomy`) -- but this function reads the
    fact from the report rather than assuming it, so a future harvest that
    somehow does confirm one of them is not silently mis-handled here.
    """
    new_counts = report.get("per_class_new_instances", {})
    return frozenset(name for name in class_names if new_counts.get(name, 0) == 0)


# ---------------------------------------------------------------------------
# Pseudo-pool classification and capped selection
# ---------------------------------------------------------------------------

def classify_pseudo_pool(pairs, protected_classes):
    """`(image_path, label_path)` pairs -> `(protected_images, common_images)`.

    An image lands in `protected_images` if ANY of its accepted detections
    is a protected class, even if it also contains abundant ones -- keeping
    it costs nothing (it is never capped) and it is the only way a rare
    class's context ever survives the common-pool cap below.

    READ CONCURRENTLY. This is the single most expensive loop in the script:
    one open+read per pseudo-label file, 601,261 of them, on a ceph mount
    where a cold read measures ~40 ms SERIAL -- about 7 hours, nearly all of
    it latency rather than work (a real run burned 63 minutes here and
    accumulated 1 second of CPU). The reads are independent and
    latency-bound, so a thread pool overlaps them: measured cold, 16-64
    threads all reach ~200-280 files/s, roughly 10x serial, bringing the pass
    to well under an hour. Thread COUNT barely matters past 16 -- the ceiling
    is the filesystem, not the pool -- so this does not tune it beyond a
    modest default.

    ORDER IS PRESERVED. `executor.map` yields results in input order, and
    both output lists are appended in that order, so the partition is
    identical to the serial version and `select_pseudo_images`'s seeded
    sampling stays reproducible. That is the property that makes this a
    performance change and not a behaviour change.

    Breaks if: this switches to `as_completed` (which would scramble order
    and silently de-reproducibilise the seeded sample), or the pool is sized
    from the file count rather than a small constant.
    """
    protected_images, common_images = [], []
    pairs = list(pairs)
    if not pairs:
        return protected_images, common_images
    with ThreadPoolExecutor(max_workers=LABEL_READ_THREADS) as executor:
        all_classes = executor.map(
            lambda pair: read_label_classes(pair[1]), pairs)
        for (image_path, _label_path), classes in zip(pairs, all_classes):
            if classes & protected_classes:
                protected_images.append(image_path)
            else:
                common_images.append(image_path)
    return protected_images, common_images


def select_pseudo_images(protected_images, common_images, common_cap, seed=DEFAULT_SEED):
    """All protected images, plus a seeded sample of common images up to `common_cap`.

    Breaks if: `common_cap` stops being applied (the whole point of this
    function -- capping the abundant-class pool -- disappears), or if the
    cap is mistakenly applied to `protected_images` too (the rare classes
    Task 1 found would then be thinned by the very mechanism meant to
    protect them).
    """
    if common_cap < 0:
        raise ValueError("common_cap must be >= 0, got %r" % (common_cap,))
    if len(common_images) <= common_cap:
        selected_common = list(common_images)
    else:
        selected_common = random.Random(seed).sample(list(common_images), common_cap)
    return list(protected_images) + selected_common


# ---------------------------------------------------------------------------
# Hand-labeled oversampling (base factor + zero-gain-class boost)
# ---------------------------------------------------------------------------

def hand_oversample_factors(hand_paths, zero_gain_classes, base_factor, boost_factor):
    """Per-image repeat count: `base_factor`, plus `boost_factor` extra for
    any hand-labeled image containing a zero-gain class.

    This is the one lever available against concern #3 (bipolar dissector /
    suction irrigator gaining literally nothing from pseudo-labeling): it
    cannot invent pseudo signal for a class the logbook can never confirm,
    but it can stop that class's per-epoch presence from shrinking just
    because 20,000+ pseudo images were added around it.

    Breaks if: the `boost_factor` term is dropped, making every hand image
    get the same factor regardless of which classes it contains.
    """
    if base_factor < 1:
        raise ValueError("base_factor must be >= 1, got %r" % (base_factor,))
    if boost_factor < 0:
        raise ValueError("boost_factor must be >= 0, got %r" % (boost_factor,))
    factors = {}
    for path in hand_paths:
        label_path = image_path_to_label_path(path)
        classes = read_label_classes(label_path)
        factors[str(path)] = (base_factor + boost_factor
                               if classes & zero_gain_classes else base_factor)
    return factors


def oversample_hand_labeled(hand_paths, factors):
    """Expand `hand_paths` into a manifest, repeating each per its factor."""
    lines = []
    for path in hand_paths:
        lines.extend([str(path)] * factors[str(path)])
    return lines


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_report(hand_train_paths, hand_val_paths, protected_images, common_images,
                 selected_pseudo, train_manifest, factors, multipliers,
                 protected_classes, zero_gain_classes, args):
    zero_gain_hand_images = sum(
        1 for p in hand_train_paths
        if factors[str(p)] > args.hand_oversample)
    return {
        "hand_train_images": len(hand_train_paths),
        "hand_val_images": len(hand_val_paths),
        "hand_oversample_base_factor": args.hand_oversample,
        "hand_oversample_zero_gain_boost": args.zero_gain_oversample_boost,
        "hand_train_images_with_zero_gain_class": zero_gain_hand_images,
        "hand_train_slots_after_oversample": sum(factors.values()),
        "pseudo_images_available": len(protected_images) + len(common_images),
        "pseudo_images_protected": len(protected_images),
        "pseudo_images_common": len(common_images),
        "pseudo_common_cap": args.pseudo_common_cap,
        "pseudo_images_selected": len(selected_pseudo),
        "protect_below_multiplier": args.protect_below_multiplier,
        "protected_classes": sorted(protected_classes),
        "zero_gain_classes": sorted(zero_gain_classes),
        "class_multipliers": {k: round(v, 2) for k, v in sorted(multipliers.items())},
        "train_manifest_lines": len(train_manifest),
        "val_manifest_lines": len(hand_val_paths),
        "seed": args.seed,
    }


def run(args):
    report = json.loads(Path(args.pseudo_report).read_text(encoding="utf-8"))
    verify_pseudo_harvest_excluded_heldout(report)

    hand_root = Path(args.hand_root)
    hand_train_lines = read_txt_list(args.hand_train)
    hand_val_lines = read_txt_list(args.hand_val)
    hand_train_paths = resolve_hand_paths(hand_train_lines, hand_root)
    hand_val_paths = resolve_hand_paths(hand_val_lines, hand_root)

    pairs = list_pseudo_pairs(args.pseudo_images_dir, args.pseudo_labels_dir)
    pseudo_image_paths = [image_path for image_path, _ in pairs]

    heldout_ids = load_heldout_ids(args.splits)
    assert_no_heldout_pseudo_images(pseudo_image_paths, heldout_ids)

    multipliers = compute_multipliers(
        report["hand_labeled_train_instances"], report["per_class_new_instances"])
    protected = protected_classes_from_multipliers(multipliers, args.protect_below_multiplier)
    zero_gain = zero_gain_classes_from_report(report)

    protected_images, common_images = classify_pseudo_pool(pairs, protected)
    selected_pseudo = select_pseudo_images(
        protected_images, common_images, args.pseudo_common_cap, seed=args.seed)

    factors = hand_oversample_factors(
        hand_train_paths, zero_gain, args.hand_oversample, args.zero_gain_oversample_boost)
    oversampled_hand = oversample_hand_labeled(hand_train_paths, factors)

    train_manifest = oversampled_hand + [str(p) for p in selected_pseudo]

    out_dir = Path(args.out_dir)
    train_txt = out_dir / "train.txt"
    val_txt = out_dir / "val.txt"
    yaml_path = out_dir / "surg_14cls_v2.yaml"
    write_lines(train_txt, train_manifest)
    write_lines(val_txt, hand_val_paths)
    write_dataset_yaml(yaml_path, train_txt, val_txt)

    rep = build_report(hand_train_paths, hand_val_paths, protected_images, common_images,
                       selected_pseudo, train_manifest, factors, multipliers,
                       protected, zero_gain, args)
    rep["train_txt"] = str(train_txt)
    rep["val_txt"] = str(val_txt)
    rep["dataset_yaml"] = str(yaml_path)
    report_out = Path(args.report_out) if args.report_out else out_dir / "build_report.json"
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(rep, indent=2, sort_keys=True), encoding="utf-8")

    print("build_detector_v2_dataset: wrote %d train / %d val lines to %s"
         % (len(train_manifest), len(hand_val_paths), out_dir))
    print(json.dumps(rep, indent=2, sort_keys=True))
    return rep


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-root", default=DEFAULT_HAND_ROOT)
    parser.add_argument("--hand-train", default=None,
                        help="default: <hand-root>/yolo_dataset/train.txt")
    parser.add_argument("--hand-val", default=None,
                        help="default: <hand-root>/yolo_dataset/val.txt")
    parser.add_argument("--pseudo-images-dir", default=DEFAULT_PSEUDO_IMAGES_DIR)
    parser.add_argument("--pseudo-labels-dir", default=DEFAULT_PSEUDO_LABELS_DIR)
    parser.add_argument("--pseudo-report", default=DEFAULT_PSEUDO_REPORT)
    parser.add_argument("--splits", default=DEFAULT_SPLITS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--hand-oversample", type=int, default=DEFAULT_HAND_OVERSAMPLE)
    parser.add_argument("--zero-gain-oversample-boost", type=int,
                        default=DEFAULT_ZERO_GAIN_OVERSAMPLE_BOOST)
    parser.add_argument("--pseudo-common-cap", type=int, default=DEFAULT_PSEUDO_COMMON_CAP)
    parser.add_argument("--protect-below-multiplier", type=float,
                        default=DEFAULT_PROTECT_BELOW_MULTIPLIER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.hand_train is None:
        args.hand_train = str(Path(args.hand_root) / "yolo_dataset" / "train.txt")
    if args.hand_val is None:
        args.hand_val = str(Path(args.hand_root) / "yolo_dataset" / "val.txt")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
