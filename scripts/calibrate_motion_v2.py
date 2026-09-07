"""Fit motion-v2 probe offsets and thresholds against annotated task intervals.

WHY A PROXY OBJECTIVE. The shipped burst offset of 67 ms was chosen and never
swept against anything, so "is 67 ms right" has no answer on record. It does
not need a new annotation to get one: tasks.csv already marks when surgical
work is happening. A timestamp inside an annotated interval is active; one in
the gap between intervals is idle. That is a real label over 155 cases and it
costs nothing.

WHAT THIS OBJECTIVE IS NOT. It is not the answer metric, and a probe offset
that separates task intervals best is not thereby proven to raise BERTScore.
The gaps between annotated tasks also contain real surgery that simply was
not annotated, so "idle" is noisy in a known direction. Recorded here rather
than discovered later: this picks between candidate offsets, it does not
prove the winner helps.

AUC rather than accuracy, because the two classes are unbalanced and the
point of the sweep is to compare statistics before any threshold exists.

NINE SLOTS, NOT EIGHT (controller ruling R8). `slots` below is imported from
`surgvu.motion._VECTOR_KEYS` rather than re-typed, so the two can never drift.
An earlier draft of this script hand-typed eight of the nine names and
silently dropped `flow_moving_fraction` -- which Task 1 measured as the
PRIMARY camera-vs-tool discriminator (12x separation between a camera pan and
local tool motion, versus 1.33x for `flow_coherence`). Dropping it would have
skipped the sweep's most informative feature without raising an error.

THE WRITTEN OFFSETS ARE READ, NOT ASSERTED (controller ruling R16). The
output config's "offsets_ms" used to be the literal `[133, 400, 1200]`,
typed here independently of what scripts/dump_motion_v2.py actually used to
produce `--dump`'s records. `dump_offsets_ms()` reads the offsets each
record itself carries instead: a config that names offsets it was not
actually fitted at is confidently WRONG provenance, which is worse than the
missing provenance MOTION_V2_VERSION exists to prevent, because a wrong
number looks exactly as trustworthy as a right one until something breaks
downstream.

THE REAL tasks.csv SCHEMA, NOT A SIMPLIFIED ONE (controller ruling R14). An
earlier draft of `activity_labels` (and its test fixture) read `start`/`stop`
columns. The real corpus's tasks.csv -- verified directly against
/staging/groups/bhaskar_opscribe/surgvu/labels_cat2/SURGVU25_train_labels/
case_*/tasks.csv -- has `start_part`/`start_time`/`stop_part`/`stop_time`
instead, and is part-aware: a case can have up to two video files and
timestamps RESET at the part boundary (the same fact `src/surgvu/extract.py`
and `src/surgvu/labels.py` already encode). Reading the wrong column names
did not raise; it silently dropped every row via the existing
`except (KeyError, ...): continue`, which would have scored every window
idle and made `separability()` raise ValueError on every slot for a reason
that had nothing to do with the sweep itself. Fixed here rather than
discovered on the first real run.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.labels import normalize_part                         # noqa: E402
from surgvu.motion import _VECTOR_KEYS                            # noqa: E402


def activity_labels(tasks_csv, timestamps, part=None):
    """True where a timestamp falls inside an annotated task interval.

    Reads the REAL tasks.csv schema: `start_part`/`start_time`/`stop_part`/
    `stop_time` (not the simplified `start`/`stop` an earlier draft of this
    function -- and its test fixture -- used, which is exactly how the
    mismatch against the real corpus was missed until controller ruling
    R14). A row whose `start_part` disagrees with its `stop_part` spans a
    video-part boundary; timestamps reset there, so the row's start/stop are
    not comparable and it is dropped, mirroring
    `surgvu.labels.CaseLabels._load_tasks`.

    `part` is optional. When given (any spelling `surgvu.labels.
    normalize_part` accepts -- `'1'`, `1`, `'1.0'`), only rows whose part
    matches contribute; a case with two video files must never have its
    part-2 timestamps compared against a part-1 window. When `None` (the
    default), every row that survives the boundary check is used regardless
    of its own part -- which keeps the common case (most cases have a
    single video part) working exactly as it did before `part` existed.

    Boundaries are inclusive: a frame at the annotated start of a task is
    part of that task.
    """
    intervals = []
    wanted_part = None if part is None else normalize_part(part)
    with open(tasks_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row_part = normalize_part(row.get("start_part"))
            if row_part != normalize_part(row.get("stop_part")):
                continue                              # spans a part boundary
            if wanted_part is not None and row_part != wanted_part:
                continue
            try:
                intervals.append(
                    (float(row["start_time"]), float(row["stop_time"])))
            except (KeyError, TypeError, ValueError):
                continue
    return [any(start <= t <= stop for start, stop in intervals)
            for t in timestamps]


def separability(values, labels):
    """AUC of `values` against boolean `labels`, by rank.

    None values are DROPPED with their label rather than substituted. An
    unavailable measurement scored as 0.0 would look like a quiet frame and
    would be counted as evidence the statistic works.
    """
    pairs = [(v, bool(l)) for v, l in zip(values, labels) if v is not None]
    positives = [v for v, l in pairs if l]
    negatives = [v for v, l in pairs if not l]
    if not positives or not negatives:
        raise ValueError(
            "AUC needs both classes; got %d active and %d idle. A sweep over "
            "one class would report a number that means nothing."
            % (len(positives), len(negatives)))
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positives) * len(negatives))


def best_cut(values, labels):
    """The threshold maximising balanced accuracy, and that accuracy."""
    pairs = sorted((v, bool(l)) for v, l in zip(values, labels)
                   if v is not None)
    if not pairs:
        raise ValueError("no measured values to fit a cut against")
    candidates = sorted({v for v, _ in pairs})
    positives = sum(1 for _, l in pairs if l)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        raise ValueError("a cut needs both classes present")
    best, best_score = candidates[0], -1.0
    for cut in candidates:
        tp = sum(1 for v, l in pairs if l and v >= cut)
        tn = sum(1 for v, l in pairs if not l and v < cut)
        score = 0.5 * (tp / positives + tn / negatives)
        if score > best_score:
            best, best_score = cut, score
    return float(best), float(best_score)


def dump_offsets_ms(records):
    """The probe offsets the dump records were ACTUALLY produced with
    (controller ruling R16).

    Read from the records themselves rather than asserted as a literal in
    this file. An earlier version of this function wrote `[133, 400, 1200]`
    into config/motion_v2.json unconditionally -- a claim about what the
    dump was produced with, not a measurement of it. If dump_motion_v2.py's
    offsets ever changed without this literal being updated to match, the
    config would record confidently WRONG provenance, which is worse than
    recording none: MOTION_V2_VERSION exists precisely so a stored record
    can be told apart from one computed by a different definition, and a
    plausible-looking wrong literal defeats that.

    Every record in one dump must agree, or the vectors it produced are not
    comparable and no single threshold fitted over them means anything.
    """
    if not records:
        raise ValueError("no records in the dump; nothing to calibrate against")
    try:
        seen = {tuple(record["offsets_ms"]) for record in records}
    except KeyError:
        raise ValueError(
            "a dump record has no 'offsets_ms' field; this dump was produced "
            "before scripts/dump_motion_v2.py recorded its own offsets "
            "(controller ruling R16) and its provenance cannot be trusted -- "
            "re-run the dump rather than assume it used the current default")
    if len(seen) > 1:
        raise ValueError(
            "the dump records disagree about their offsets_ms: %s. Every "
            "record must come from the same probe offsets, or the vectors "
            "pooled into one threshold are not comparable"
            % sorted(seen))
    return list(next(iter(seen)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump", required=True,
                        help="JSON produced by scripts/dump_motion_v2.py: a "
                             "list of {case, part, t, vector, offsets_ms} "
                             "records over "
                             "the source videos (NOT scripts/sample_motion.py, "
                             "which emits a different, v1 shape over the "
                             "eleven public sample cases only). `part` "
                             "identifies which of a case's up-to-two video "
                             "files the record's `t` is measured against, "
                             "and is passed to activity_labels() so a "
                             "part-2 timestamp is never checked against "
                             "part-1 task intervals")
    parser.add_argument("--labels-root", required=True,
                        help="SURGVU25_train_labels root, one dir per case")
    parser.add_argument("--split", default="train",
                        help="recorded in the output so a threshold can never "
                             "be quoted without the split it was fitted on")
    parser.add_argument("--out", default="config/motion_v2.json")
    args = parser.parse_args(argv)

    records = json.loads(Path(args.dump).read_text(encoding="utf-8"))
    # Grouped by (case, part) rather than by case alone: a case with two
    # video files has two independent sets of task intervals (start_part/
    # stop_part), and activity_labels() must be handed only the part a
    # record's timestamp actually belongs to.
    by_case_part = {}
    for record in records:
        key = (record["case"], record.get("part"))
        by_case_part.setdefault(key, []).append(record)

    # Imported from surgvu.motion rather than re-typed -- see the module
    # docstring's R8 note. This is the same nine-element ordering
    # `motion_vector` writes into every anchor.
    slots = _VECTOR_KEYS
    columns = {slot: [] for slot in slots}
    labels = []
    for (case, part), rows in sorted(by_case_part.items(),
                                     key=lambda kv: (kv[0][0], str(kv[0][1]))):
        tasks_csv = Path(args.labels_root) / case / "tasks.csv"
        if not tasks_csv.exists():
            print("skipping %s: no tasks.csv" % (case,))
            continue
        stamps = [row["t"] for row in rows]
        labels.extend(activity_labels(tasks_csv, stamps, part=part))
        for slot in slots:
            columns[slot].extend(row["vector"].get(slot) for row in rows)

    thresholds = {}
    for slot in slots:
        try:
            auc = separability(columns[slot], labels)
            cut, balanced = best_cut(columns[slot], labels)
        except ValueError as exc:
            print("%-20s unusable: %s" % (slot, exc))
            continue
        thresholds[slot] = {"cut": cut, "auc": auc,
                            "balanced_accuracy": balanced,
                            "measured": sum(1 for v in columns[slot]
                                            if v is not None)}
        print("%-20s auc=%.4f cut=%.4f balacc=%.4f  n=%d"
              % (slot, auc, cut, balanced, thresholds[slot]["measured"]))

    out = {
        "version": 2,
        # The offsets these records were ACTUALLY produced with (R16), not a
        # literal that has to be kept in sync with dump_motion_v2.py by hand.
        "offsets_ms": dump_offsets_ms(records),
        "thresholds": thresholds,
        "fitted_on": args.split,
        # Distinct CASES, not (case, part) pairs -- a two-part case must
        # count once here, matching what "n_cases" means everywhere else in
        # this codebase (e.g. scripts/calibrate_motion.py's shard count).
        "n_cases": len({case for case, _ in by_case_part}),
        "objective": "tasks.csv interval membership (proxy, not the metric)",
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n",
                              encoding="utf-8")
    print("wrote %s" % (args.out,))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
