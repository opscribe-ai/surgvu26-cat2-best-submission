"""Train the Large/Mega needle-driver head and fit its abstention cutoff.

WHAT THIS PRODUCES. `--out-weights`: a ResNet-18 state dict, saved as
`{"model": state_dict}` -- the exact shape `surgvu.variant.VariantHead._load`
reads, so the checkpoint this script writes is directly loadable by the class
that serves it, with no adapter step in between. `--out-config` (default
`config/variant_head.json`): the FITTED cutoff plus the validation accuracy
and coverage it achieved and the held-out case ids, per the Task 10 brief.
Both numbers are measurements of this specific run, not constants -- writing
a cutoff into source instead would be a guess wearing the authority of code.

THE SPLIT DISCIPLINE. Held out by CASE, never by frame: frames inside one
install interval are near-duplicates of each other (same arm, same
instrument, seconds apart), so a frame-level split would let the model
memorise a handful of installs and report that as generalisation. 137 of 152
labelled cases contain BOTH families (scripts/build_variant_labels.py), so a
case-level split can starve neither class -- `split_cases` below refuses any
split where either family goes empty on either side, which is possible
precisely because the corpus has that structure.

WHY THIS TALKS TO THE DETECTOR. `surgvu.detect.Detector` (Task 6) is what
turns "is this needle driver Large or Mega" from a whole-frame problem into a
cropped one: the detector finds the needle driver's box, and the classifier
only has to compare its taper and jaw length against the box's own scale
rather than also finding it. When the detector finds nothing (a frame this
sparse can miss it, or the interval genuinely has none of the several
needle-driver classes` moments in shot), the whole frame is used instead --
the same crop-when-available discipline `VariantHead.predict` uses at serving
time, so training and serving never see input the other could not have
produced.

HOW A TIMESTAMP RESOLVES TO A VIDEO FILE (controller ruling R28). Case
timestamps RESET at the part boundary and a single case can have up to two
video files, so "5.0 seconds into the install" is meaningless without
knowing which file it is measured against. An earlier version of this
module did not have that answer -- Task 9's `config/variant_labels.json`
originally omitted the part -- and tried to recover it by DURATION-PROBING:
open each of a case's video files in turn and accept the first whose own
length covers the timestamp. R28 measured that this cannot work: on 6
multi-part cases, 19 of 69 intervals (28%) fit inside more than one part's
duration, so probing would have silently paired the WRONG part's pixels
with a correct-looking family label on roughly a quarter of a multi-part
case's frames -- label noise large enough to plausibly flip this task's own
open question ("are Large and Mega separable at all?") from learnable to
"abstains always", which would look like a result instead of a corrupted
experiment.

The actual fix is upstream, in `scripts/build_variant_labels.py`: every
interval now carries its own recorded `part` (config/variant_labels.json
version >= 2), read from the same `install_case_part` column that function
already used to drop rows spanning a boundary. `resolve_case_video` below
therefore does no probing and no guessing -- it builds the expected filename
directly from the label's `part` and returns None (never a fallback file)
if that exact file is not on disk. `load_variant_labels` refuses to load a
version-1 file at all, so a stale part-less config cannot reach this path
silently.

THE GRADED CASES MUST NEVER ENTER THIS HEAD AT ALL (controller ruling R30).
`config/splits_v2.json`'s `heldout` list -- the 11 public sample cases,
`docs/compliance_audit.md`'s subject -- is the one authoritative list of
cases this repo has already agreed no training run may touch. An earlier
version of this script never consulted it: 7 of the 11 graded cases'
intervals landed in this head's OWN train split (including case_132) and 4
landed in its OWN held-out split (including case_126) -- the exact two
cases this whole component exists to fix, one trained on and the other used
to fit the abstention cutoff. `load_heldout_ids` reads that list and
`exclude_graded_cases` removes every one of those cases from
`config/variant_labels.json`'s cases BEFORE `split_cases` ever sees them, so
they cannot land in this head's train split OR its own held-out split
either. Comparison is via `surgvu.sampling.normalize_case_id`, never raw
string equality -- the public sample directories spell a case `caseNNN`,
this repo's labels spell it `case_NNN`, and the two are the same case but
never equal as strings, which is exactly how the leak went unnoticed the
first time.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.sampling import normalize_case_id  # noqa: E402
from surgvu.variant import FAMILIES, VARIANT_VERSION, variant_record  # noqa: E402

DEFAULT_LABELS = "config/variant_labels.json"
#: NOT config/splits.json -- that is the leaky v1 split
#: docs/compliance_audit.md flags by name as a trap (train_tools.py /
#: train_task.py both default to it and must be passed splits_v2 explicitly).
#: splits_v2's own `heldout` list is the 11 public sample cases; excluding
#: them is the entire point of consulting this file (controller ruling R30).
DEFAULT_SPLITS = "config/splits_v2.json"
DEFAULT_VIDEO_ROOT = "/staging/groups/bhaskar_opscribe/surgvu/videos/surgvu24"
DEFAULT_DETECTOR_WEIGHTS = (
    "/staging/groups/bhaskar_opscribe/surgvu_yolo_detector/best.pt")
DEFAULT_DETECTOR_REPO = (
    "/staging/groups/bhaskar_opscribe/surgvu_yolo_detector/yolov5")

#: variant_record refuses a cutoff at or below chance; the sweep grid starts
#: one step above 0.5 so every candidate it can ever return is already legal.
MIN_CUTOFF = 0.51
#: "the point where accuracy first exceeds 0.75" -- the Task 10 brief's own
#: number, quoted rather than re-derived.
TARGET_ACCURACY = 0.75
#: Matches VariantHead's default `size=224`. A mismatch here would train a
#: model at one resolution and serve it at another, which is a silent
#: accuracy loss that looks like a bad architecture choice (see
#: surgvu.train.prepare_batch's docstring for the same warning elsewhere in
#: this codebase).
FRAME_SIZE = 224

#: config/variant_labels.json versions below this carry no `part` on their
#: intervals (see this module's docstring, controller ruling R28).
#: `load_variant_labels` refuses anything older.
MIN_LABELS_VERSION = 2

#: A part as scripts/build_variant_labels.py's `normalize_part` writes it:
#: '1.0', '2.0', ... `resolve_case_video` matches against this, not against
#: an arbitrary numeric string, so a malformed or hand-written part value
#: fails loudly instead of resolving to a plausible-looking wrong file.
_CANONICAL_PART_RE = re.compile(r"(\d+)\.0")


# ---------------------------------------------------------------------------
# TORCH-FREE. Every function down to build_examples() takes plain python and
# numpy and is exercised directly by tests/test_train_variant.py on a machine
# with no torch installed -- the same discipline surgvu.holdout and
# surgvu.variant follow, and for the same reason: the decision logic (here,
# the split and the cutoff sweep) is what correctness depends on, and it
# should be testable without a GPU, a container, or the video corpus.
# ---------------------------------------------------------------------------

def load_variant_labels(path=DEFAULT_LABELS):
    """`config/variant_labels.json` -> `{case_id: [interval, ...]}`.

    Refuses a file below `MIN_LABELS_VERSION` (controller ruling R28):
    version 1 intervals carry no `part`, and `resolve_case_video` needs one
    to find the right file rather than guess. Checked here, at load time,
    rather than left to surface deep inside `sample_points` on the first
    interval that turns out to be missing one -- a confusing place to
    discover that the wrong config file is on the command line.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    version = data.get("version")
    if not isinstance(version, int) or version < MIN_LABELS_VERSION:
        raise ValueError(
            "%s is version %r; this script requires version >= %d (every "
            "interval must carry 'part' -- see scripts/build_variant_labels.py "
            "and controller ruling R28). Regenerate it with "
            "scripts/build_variant_labels.py." % (path, version, MIN_LABELS_VERSION))
    return data["cases"]


def load_heldout_ids(path=DEFAULT_SPLITS):
    """Normalised case ids in `path`'s `"heldout"` list.

    Controller ruling R30. `config/splits_v2.json`'s `heldout` list is the
    11 public sample cases -- the ONLY cases this task can ever measure a
    result against, and therefore the ones a training run must never touch.
    Every id is passed through `surgvu.sampling.normalize_case_id` rather
    than kept as-is: the public sample directories spell a case `caseNNN`
    and this repo's own labels spell it `case_NNN`, and a raw string
    comparison between the two forms matches nothing, which is exactly how
    7 of the 11 graded cases ended up inside this head's own train split
    before this fix.

    Raises if the list is missing or empty -- a config file that
    accidentally lost its `heldout` key would otherwise exclude nothing
    while `exclude_graded_cases` reports a clean-looking run.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("heldout")
    if not raw:
        raise ValueError(
            "%s has no ('heldout') list, or it is empty. This script relies "
            "on that list to exclude the graded/public-sample cases from "
            "training entirely (controller ruling R30) -- a missing or "
            "empty list would exclude nothing while looking like a clean "
            "run." % (path,))
    return {normalize_case_id(case_id) for case_id in raw}


def exclude_graded_cases(cases, heldout_ids):
    """`cases` with every case named in `heldout_ids` removed.

    Returns `(kept, n_excluded_cases, n_excluded_intervals)`.

    CONTROLLER RULING R30. This removes a graded case from the pool BEFORE
    `split_cases` ever sees it -- not merely from a "train" bucket -- so an
    excluded case can land in NEITHER this head's train split NOR its own
    held-out split used to fit the cutoff. Both routes leak: the audit that
    prompted this fix found case_132 (a graded failure this whole component
    exists to fix) inside the head's train split, and case_126 (the OTHER
    graded failure) inside the head's held-out split, fitting the very
    cutoff a "did this work" measurement would read.

    Ids are compared via `surgvu.sampling.normalize_case_id`, never raw `==`
    or `in` -- see `load_heldout_ids` for why a string comparison silently
    excludes nothing on this corpus's actual spellings.

    Raises if exclusion removes zero cases. On the real corpus this always
    removes exactly the 11 graded cases; a run reporting zero is not
    evidence the corpus was already clean, it is the same normalisation bug
    recurring, and this refuses to proceed rather than risk it silently.
    """
    normalized_heldout = {normalize_case_id(case_id) for case_id in heldout_ids}
    kept = {}
    excluded_cases, excluded_intervals = 0, 0
    for case_id, intervals in cases.items():
        if normalize_case_id(case_id) in normalized_heldout:
            excluded_cases += 1
            excluded_intervals += len(intervals)
            continue
        kept[case_id] = intervals
    if excluded_cases == 0:
        raise ValueError(
            "excluding the graded/held-out cases removed 0 of %d cases. On "
            "the real corpus this always removes the 11 public sample "
            "cases; a zero here means the SAME normalisation bug controller "
            "ruling R30 found (raw id comparison silently matching nothing) "
            "is recurring, not that the corpus was already clean. Refusing "
            "to proceed rather than risk training on the graded cases "
            "again." % (len(cases),))
    return kept, excluded_cases, excluded_intervals


def family_seconds(intervals):
    """{"large": total_seconds, "mega": total_seconds} for one case's rows.

    Duration-weighted, not a bare count of intervals: a single 400 s install
    and a single 8 s install are both "one interval", but they do not carry
    the same amount of visual evidence, and the split below should be
    balanced on the evidence, not on row count.
    """
    totals = {family: 0.0 for family in FAMILIES}
    for interval in intervals:
        family = interval.get("family")
        if family not in totals:
            continue
        duration = float(interval["stop"]) - float(interval["start"])
        if duration > 0:
            totals[family] += duration
    return totals


def split_cases(cases, holdout_fraction=0.2, seed=13, trials=4000):
    """Case ids -> (train_ids, heldout_ids), stratified by family-seconds.

    HOLD OUT BY CASE, NEVER BY FRAME. Frames inside one install interval are
    near-duplicates of each other, so a frame-level split would report
    memorisation as accuracy -- this is the single most important
    correctness property in this task, and it is enforced structurally here
    by never looking inside a case once it has been assigned.

    Random search minimises the worst per-family log-ratio between the two
    sides (so neither side is starved of either class) plus a penalty for
    drifting from `holdout_fraction`, mirroring the objective
    `surgvu.holdout.case_folds` uses for a similar problem -- fixed seed, so
    two runs at the same arguments produce the same split rather than two
    incomparable ones.

    Raises if no split keeps BOTH families non-empty on BOTH sides. 137 of
    152 labelled cases hold both families (see this module's docstring), so
    this is expected to always succeed on the real corpus; it is checked
    rather than assumed because a cutoff fitted against a held-out set
    missing one family would report a coverage number that means nothing for
    that family.
    """
    ids = sorted(cases)
    if len(ids) < 2:
        raise ValueError(
            "need at least 2 cases to hold any out; got %d" % len(ids))
    totals = {cid: family_seconds(cases[cid]) for cid in ids}
    n = len(ids)
    target_heldout = max(1, min(n - 1, int(round(n * holdout_fraction))))

    def imbalance(assign):
        heldout = [c for c, a in zip(ids, assign) if a]
        train = [c for c, a in zip(ids, assign) if not a]
        if not heldout or not train:
            return None
        h = {f: sum(totals[c][f] for c in heldout) for f in FAMILIES}
        t = {f: sum(totals[c][f] for c in train) for f in FAMILIES}
        if any(h[f] <= 0 for f in FAMILIES) or any(t[f] <= 0 for f in FAMILIES):
            return None    # a family missing from one side is disqualifying
        ratio = max(abs(np.log(h[f] / t[f])) for f in FAMILIES)
        size_penalty = abs(len(heldout) - target_heldout) / float(n)
        return ratio + 4.0 * size_penalty

    rng = np.random.default_rng(seed)
    best_assign, best_score = None, None
    for _ in range(trials):
        assign = np.zeros(n, dtype=int)
        assign[rng.choice(n, size=target_heldout, replace=False)] = 1
        score = imbalance(assign)
        if score is None:
            continue
        if best_score is None or score < best_score:
            best_assign, best_score = assign, score

    if best_assign is None:
        raise ValueError(
            "no case-level split of %d cases keeps both large and mega "
            "present on both sides. Splitting further would starve a class "
            "rather than measure it honestly." % n)
    train_ids = [c for c, a in zip(ids, best_assign) if not a]
    heldout_ids = [c for c, a in zip(ids, best_assign) if a]
    return train_ids, heldout_ids


def sample_points(intervals, spacing=8.0, max_per_interval=5, margin=1.0):
    """[(t_seconds, family, arm, part), ...] timestamps to decode, from one
    case's labelled intervals.

    A handful of evenly spaced timestamps per interval, not every second:
    sampling densely would let one long install (some run past 400 s) swamp
    the training set with near-duplicate frames while a short one contributes
    almost nothing, which is the frame-level version of the same
    memorisation risk `split_cases` guards against at the case level.
    `margin` keeps samples away from the install/uninstall edges, where the
    frame may still show the outgoing instrument mid-exchange rather than the
    one the row actually labels.

    `part` is carried straight through from the interval (config/
    variant_labels.json version >= 2, controller ruling R28) so a caller can
    resolve the exact video file a timestamp is measured against
    (`resolve_case_video`) instead of guessing which of a case's up-to-two
    files it belongs to. Raises if an interval has no `part` at all --
    `load_variant_labels` already refuses a file old enough to lack one, so
    reaching this means a caller built the intervals list some other way.
    """
    if max_per_interval < 1:
        raise ValueError("max_per_interval must be >= 1, got %r"
                         % (max_per_interval,))
    out = []
    for interval in intervals:
        family = interval.get("family")
        if family not in FAMILIES:
            continue
        part = interval.get("part")
        if not part:
            raise ValueError(
                "interval %r has no 'part' -- config/variant_labels.json "
                "must be version >= %d (see scripts/build_variant_labels.py, "
                "controller ruling R28); an older, part-less interval cannot "
                "be resolved to a video file without guessing." % (
                    interval, MIN_LABELS_VERSION))
        start = float(interval["start"]) + margin
        stop = float(interval["stop"]) - margin
        if stop <= start:
            continue
        span = stop - start
        count = min(max_per_interval, max(1, int(span // spacing) + 1))
        if count == 1:
            times = [start + span / 2.0]
        else:
            times = [start + i * span / (count - 1) for i in range(count)]
        arm = interval.get("arm")
        out.extend((t, family, arm, part) for t in times)
    return out


def sweep_cutoff(labels, p_large, target_accuracy=TARGET_ACCURACY,
                 min_cutoff=MIN_CUTOFF, step=0.01):
    """(cutoff, accuracy, coverage) fitted on held-out (label, p_large) pairs.

    Walks the grid from `min_cutoff` (already above chance) up to 1.0 and
    returns the FIRST point -- the smallest cutoff, hence the highest
    coverage -- whose accuracy on the samples it decides exceeds
    `target_accuracy`. That is "the accuracy-vs-coverage point where accuracy
    first exceeds 0.75" from the Task 10 brief, made concrete.

    ABSTENTION IS A FEATURE, so a target that is never reached is not an
    error: this returns the highest-accuracy cutoff that decides at least one
    sample instead, and the caller's coverage number then honestly reports
    "not much", rather than the sweep manufacturing a decision the data does
    not support. It raises only when NO cutoff on the grid ever decides
    anything, which would mean every held-out probability is closer to 0.5
    than `min_cutoff` -- degenerate enough to be worth stopping the run over.
    """
    labels = np.asarray(labels)
    p_large = np.asarray(p_large, dtype=float)
    if len(labels) != len(p_large):
        raise ValueError("labels (%d) and p_large (%d) must be the same "
                         "length" % (len(labels), len(p_large)))
    if len(labels) == 0:
        raise ValueError("no held-out examples to fit a cutoff against")

    p_mega = 1.0 - p_large
    confidence = np.maximum(p_large, p_mega)
    predicted = np.where(p_large >= p_mega, "large", "mega")
    correct = predicted == labels

    fallback = None
    for cutoff in np.arange(min_cutoff, 1.0 + 1e-9, step):
        decided = confidence >= cutoff
        if not decided.any():
            continue
        coverage = float(decided.mean())
        accuracy = float(correct[decided].mean())
        if fallback is None or accuracy > fallback[1]:
            fallback = (float(cutoff), accuracy, coverage)
        if accuracy > target_accuracy:
            return float(cutoff), accuracy, coverage
    if fallback is None:
        raise ValueError(
            "no cutoff in (%.2f, 1.0] ever decides on this held-out set -- "
            "every predicted confidence is below %.2f" % (min_cutoff, min_cutoff))
    return fallback


def resolve_case_video(video_root, case_id, part):
    """Path to `case_id`'s video file for its OWN recorded `part`, or None.

    Built directly from the label -- no probing, no "try each part and see
    which duration fits". See this module's docstring (controller ruling
    R28) for why the earlier duration-probing approach was deleted rather
    than kept as a fallback: it produced silently wrong pixels on a
    measured 28% of a multi-part sample, and a fallback that fires
    silently on the cases the primary path cannot resolve is exactly as
    dangerous as never having fixed it.

    `part` must be `normalize_part`'s canonical `'N.0'` form -- which is
    exactly what every interval in a version->=2 `variant_labels.json`
    carries, so a real caller never has to convert it. Raises on anything
    else rather than guessing at a filename from it.

    Returns None, never a different file, when the exact expected filename
    is not on disk -- the caller skips the sample and tallies why, rather
    than falling back to any other part.
    """
    match = _CANONICAL_PART_RE.fullmatch(str(part))
    if not match:
        raise ValueError(
            "cannot resolve a video filename from part %r; expected "
            "normalize_part's canonical 'N.0' form" % (part,))
    filename = "%s_video_part_%03d.mp4" % (case_id, int(match.group(1)))
    path = Path(video_root) / case_id / filename
    return path if path.exists() else None


def build_examples(case_ids, cases, video_root, detector, spacing,
                   max_per_interval, log_prefix="", drops=None):
    """[(crop_uint8, family), ...] decoded from `case_ids`' labelled intervals.

    One decode per sample point (`sample_points`), preprocessed exactly as
    serving preprocesses every frame (`prepare_frame`: crop the black side
    margins, blur the bottom UI band -- a challenge rule, not a choice, so
    there is no bypass here either), then cropped to the detector's
    needle-driver box when it finds one and left whole otherwise -- the same
    discipline `VariantHead.predict` uses, so a model trained on this pool
    sees the same kind of input it will be asked to classify at serving time.

    Returns `(examples, drops)`. `drops` (a `collections.Counter`, created
    fresh if not supplied, and safe to pass shared across calls) tallies
    every sample point this function could not use, by reason:
    `"no_video_for_part"` (the interval's OWN recorded part has no matching
    file on disk -- see `resolve_case_video`; this NEVER falls back to a
    different part), `"unreadable_video"` (the file opened but reports no
    usable fps) and `"unreadable_frame"` (a seek-and-read failed on an
    otherwise-good file). A sample point is skipped and counted rather than
    silently dropped, mirroring `intervals_for_case`'s own discipline one
    layer up in `scripts/build_variant_labels.py`.
    """
    import cv2

    from surgvu.preprocess import prepare_frame

    if drops is None:
        drops = Counter()

    examples = []
    for case_id in case_ids:
        points = sample_points(cases[case_id], spacing=spacing,
                               max_per_interval=max_per_interval)
        resolved = []
        for t, family, _arm, part in points:
            path = resolve_case_video(video_root, case_id, part)
            if path is None:
                drops["no_video_for_part"] += 1
                continue
            resolved.append((path, t, family))
        # Grouped by video file so each part is opened once, not once per
        # sample point -- a case can carry dozens of sample points and
        # cv2.VideoCapture(path) is not free.
        resolved.sort(key=lambda row: (str(row[0]), row[1]))
        capture, current_path, fps = None, None, None
        for path, t, family in resolved:
            if path != current_path:
                if capture is not None:
                    capture.release()
                capture = cv2.VideoCapture(str(path))
                fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
                current_path = path
            if fps <= 0:
                drops["unreadable_video"] += 1
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, frame = capture.read()
            if not ok:
                drops["unreadable_frame"] += 1
                continue
            frame = prepare_frame(frame, size=512)
            box = None
            if detector is not None:
                found = detector.detect([frame])[0]
                needle = [item for item in found if item["cls"] == "needle driver"]
                if needle:
                    box = max(needle, key=lambda item: item["conf"])["box"]
            if box is not None:
                x1, y1, x2, y2 = (int(round(v)) for v in box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 - x1 >= 8 and y2 - y1 >= 8:
                    frame = frame[y1:y2, x1:x2]
            examples.append(
                (cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE)), family))
        if capture is not None:
            capture.release()

    if drops:
        print("%sdrops: %s" % (log_prefix, dict(sorted(drops.items()))))
    print("%s%d cases -> %d examples" % (log_prefix, len(case_ids), len(examples)))
    return examples, drops


# ---------------------------------------------------------------------------
# Training. Torch, cv2 and surgvu.detect are imported inside main(), not at
# module scope, so every function above stays importable (and tested) on a
# machine with none of them installed.
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--splits", default=DEFAULT_SPLITS,
                        help="JSON with a 'heldout' list of graded/public "
                             "sample cases to exclude from BOTH the train "
                             "and held-out split entirely (controller ruling "
                             "R30). Defaults to splits_v2, NOT the leaky v1 "
                             "config/splits.json docs/compliance_audit.md "
                             "warns other training scripts about.")
    parser.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--detector-weights", default=DEFAULT_DETECTOR_WEIGHTS)
    parser.add_argument("--detector-repo", default=DEFAULT_DETECTOR_REPO)
    parser.add_argument("--no-detector", action="store_true",
                        help="skip needle-driver cropping; train on whole "
                             "frames. Degrades exactly the way "
                             "VariantHead.predict(boxes=None) degrades -- "
                             "for a wiring smoke test only, not a result.")
    parser.add_argument("--out-weights", required=True)
    parser.add_argument("--out-config", default="config/variant_head.json")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--spacing-seconds", type=float, default=8.0)
    parser.add_argument("--max-per-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-cases", type=int, default=0,
                        help="smoke-test knob: cap on total cases used before "
                             "the split (0 = all 152). A run capped this way "
                             "is for wiring only -- its accuracy means "
                             "nothing.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    import torch
    from torch import nn

    from surgvu.detect import Detector
    from surgvu.train import seed_everything

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device,
         torch.cuda.get_device_name(0) if device == "cuda" else "")

    cases = load_variant_labels(args.labels)

    # CONTROLLER RULING R30. Excluded BEFORE the case-level split ever sees
    # them, and before --max-cases, so a graded/public-sample case can land
    # in neither this head's train split nor its own held-out split used to
    # fit the cutoff -- not even under a smoke-test cap.
    excluded_ids = load_heldout_ids(args.splits)
    cases, n_excluded_cases, n_excluded_intervals = exclude_graded_cases(
        cases, excluded_ids)
    print("excluded %d graded/held-out case(s) (%d intervals) per %s"
         % (n_excluded_cases, n_excluded_intervals, args.splits))

    if args.max_cases:
        cases = {cid: cases[cid] for cid in list(cases)[:args.max_cases]}
        print("SMOKE TEST: capped at %d cases. Nothing below this line is a "
              "result." % args.max_cases)

    train_ids, heldout_ids = split_cases(cases, args.holdout_fraction, args.seed)

    def interval_counts(ids):
        counts = {family: 0 for family in FAMILIES}
        for cid in ids:
            for interval in cases[cid]:
                if interval.get("family") in counts:
                    counts[interval["family"]] += 1
        return counts

    print("cases: %d train / %d held-out" % (len(train_ids), len(heldout_ids)))
    print("train interval counts:     %s" % interval_counts(train_ids))
    print("held-out interval counts:  %s" % interval_counts(heldout_ids))

    detector = None
    if not args.no_detector:
        detector = Detector(args.detector_weights, args.detector_repo,
                            device=device)

    train_examples, train_drops = build_examples(
        train_ids, cases, args.video_root, detector, args.spacing_seconds,
        args.max_per_interval, log_prefix="train: ")
    heldout_examples, heldout_drops = build_examples(
        heldout_ids, cases, args.video_root, detector, args.spacing_seconds,
        args.max_per_interval, log_prefix="held-out: ")
    if not train_examples:
        raise SystemExit("zero usable training examples; cannot train")
    if not heldout_examples:
        raise SystemExit(
            "zero usable held-out examples; cannot fit a cutoff honestly")

    # CONTROLLER RULING R29. A held-out set is not usable just because it is
    # non-empty -- if every example in it happens to be one family, the
    # cutoff sweep's "accuracy" is measured against a single class and means
    # nothing (predicting that one class every time would score 1.0). Raise
    # loudly rather than let sweep_cutoff quietly report a number that looks
    # like a result.
    present_families = {family for _crop, family in heldout_examples}
    missing_families = [f for f in FAMILIES if f not in present_families]
    if missing_families:
        raise SystemExit(
            "held-out examples contain no %s example(s) at all (controller "
            "ruling R29): a family-collapsed held-out set cannot honestly "
            "measure accuracy. Widen --holdout-fraction, or check for an "
            "upstream case-selection bug, before retrying."
            % (" or ".join(missing_families),))

    if train_drops.get("no_video_for_part") or heldout_drops.get("no_video_for_part"):
        print("NOTE: some intervals' recorded part had no matching video "
             "file on disk (see the drops line above) and were skipped "
             "rather than resolved to a different part. If this count is "
             "large relative to the example counts, investigate the video "
             "corpus layout before trusting the numbers below.")

    label_to_idx = {name: index for index, name in enumerate(FAMILIES)}

    def batches(examples, batch_size, shuffle, seed):
        rng = np.random.default_rng(seed)
        order = (rng.permutation(len(examples)) if shuffle
                 else np.arange(len(examples)))
        for start in range(0, len(examples), batch_size):
            idx = order[start:start + batch_size]
            images = np.stack([examples[i][0] for i in idx])
            labels = np.array([label_to_idx[examples[i][1]] for i in idx])
            yield images, labels

    def to_tensor(images):
        # BGR uint8 -> RGB float [0, 1], matching VariantHead.predict exactly
        # (no ImageNet mean/std normalisation -- same convention
        # surgvu.train.prepare_batch uses for the other two experts).
        return torch.from_numpy(images[:, :, :, ::-1].copy()).permute(
            0, 3, 1, 2).float().div_(255.0).to(device)

    from torchvision.models import resnet18
    model = resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, len(FAMILIES))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    def run_epoch(examples, training, seed):
        model.train(training)
        total_loss, total_correct, total_n = 0.0, 0, 0
        with torch.set_grad_enabled(training):
            for images, labels in batches(examples, args.batch_size, training, seed):
                batch = to_tensor(images)
                target = torch.from_numpy(labels).long().to(device)
                logits = model(batch)
                loss = loss_fn(logits, target)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                total_loss += float(loss) * len(labels)
                total_correct += int((logits.argmax(1) == target).sum())
                total_n += len(labels)
        return total_loss / max(1, total_n), total_correct / max(1, total_n)

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(train_examples, True, args.seed + epoch)
        val_loss, val_acc = run_epoch(heldout_examples, False, args.seed)
        print("epoch %d  train_loss %.4f train_acc %.4f  "
             "val_loss %.4f val_acc %.4f"
             % (epoch, train_loss, train_acc, val_loss, val_acc), flush=True)

    model.eval()
    p_large_all, labels_all = [], []
    with torch.no_grad():
        for images, labels in batches(heldout_examples, args.batch_size, False,
                                      args.seed):
            probs = torch.softmax(model(to_tensor(images)), dim=1).cpu().numpy()
            p_large_all.extend(probs[:, label_to_idx["large"]].tolist())
            labels_all.extend(FAMILIES[label] for label in labels)

    cutoff, val_accuracy, val_coverage = sweep_cutoff(labels_all, p_large_all)

    # The same validation variant_record enforces in production, run here so
    # a bug in the sweep grid cannot write a cutoff the decision layer would
    # itself refuse to use.
    variant_record({"large": 1.0, "mega": 0.0}, cutoff)

    out_weights = Path(args.out_weights)
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, out_weights)

    out_config = {
        "version": VARIANT_VERSION,
        "weights": str(out_weights),
        "cutoff": cutoff,
        "val_accuracy": val_accuracy,
        "val_coverage": val_coverage,
        "held_out_cases": sorted(heldout_ids),
    }
    out_config_path = Path(args.out_config)
    out_config_path.parent.mkdir(parents=True, exist_ok=True)
    out_config_path.write_text(
        json.dumps(out_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("wrote %s and %s" % (out_weights, out_config_path))
    print("cutoff=%.3f  val_accuracy=%.4f  val_coverage=%.4f  "
         "(%d held-out cases, %d held-out examples)"
         % (cutoff, val_accuracy, val_coverage, len(heldout_ids),
            len(heldout_examples)))
    if val_accuracy <= TARGET_ACCURACY:
        print("NOTE: the sweep never exceeded %.2f accuracy anywhere on the "
             "grid. That is a legitimate result, not a bug: it means Large "
             "and Mega needle drivers may not be reliably separable at the "
             "resolution this pipeline serves, and this head should abstain "
             "most of the time. The coverage above says how often." % (
                 TARGET_ACCURACY,))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
