"""Large-vs-Mega needle-driver labels, free, from the logbook.

WHY THIS IS THE ONLY WAY TO FIX case132. The question "is a large needle
driver being used" has a different gold answer from the same question about a
mega needle driver, and no model in the pipeline can currently tell them
apart -- the tool heads have one `needle driver` class and so does the
detector.

A PER-CASE PRIOR CANNOT SUBSTITUTE, and this was measured rather than
assumed, against the full 155-case corpus (154 of which contain at least one
needle-driver row): 137 of those 154 cases contain BOTH families, and only 16
(10.4%) are single-family. Guessing from the case is guessing. The variant
has to be resolved visually, per clip, which needs labels.

The labels already exist. tools.csv records `commercial_toolname` for every
install interval, so every frame between an install and its uninstall carries
a family label for free -- no annotation, no boxes, and at the scale of the
whole 155-case corpus.

AN UNRECOGNISED NAME IS DROPPED. Defaulting to the larger family would be
defensible on frequency (Large is 62.7% of needle-driver installs in the
train split) and would be exactly wrong: it would put mislabelled frames into
training and teach the head the corpus prior instead of the appearance,
which is the failure this whole task exists to avoid.

EVERY INTERVAL CARRIES ITS VIDEO PART (controller ruling R28, version 2).
Case timestamps RESET at a part boundary and 126 of 155 cases have more than
one video file, so "seconds since install" means nothing without knowing
which file it is measured against. This function already reads
`install_case_part`/`uninstall_case_part` to drop rows that span a boundary
(see `spans_part_boundary` below) -- version 1 threw that value away instead
of keeping it, on the theory that a consumer could recover the part later by
probing which video file's own duration covers the timestamp. R28 measured
that theory and falsified it: on 6 multi-part cases, 19 of 69 intervals (28%)
fit inside more than one part's duration, which duration-probing cannot
disambiguate even in principle. The fix is upstream, here, where the part is
already known with certainty: every interval below carries `"part"`,
normalised the same way a video-filename lookup expects.

COLUMN NAMES AND TIME FORMAT (verified directly against
/staging/groups/bhaskar_opscribe/surgvu/labels_cat2/SURGVU25_train_labels):
tools.csv's real header is

    install_case_part,install_case_time,uninstall_case_part,
    uninstall_case_time,arm,commercial_toolname,groundtruth_toolname,case

and the two time columns are 'HH:MM:SS.ffffff' strings, e.g.
'00:07:24.796000' -> 444.796 seconds -- not floats. An earlier draft of this
task's plan used `install_time`/`uninstall_time` and `float(...)` directly,
which raises on every real row, is swallowed by a bare except, and yields
zero intervals for all 155 cases behind a fully green test suite. Task 4 and
this plan's own design doc (docs/design/notes/2026-08-24-label-vocab-
hazards.md) both hit that exact defect. Parsing goes through
`surgvu.labels.parse_hms`, the same helper `CaseLabels` uses, instead of a
second hand-rolled copy.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.labels import normalize_part, parse_hms  # noqa: E402

#: Substring tests against the lowercased, stripped commercial name.
#: "suturecut" alone implies Large: the corpus has "Large SutureCut Needle
#: Driver" (776) and "Mega SutureCut Needle Driver" (363 + 2), and the mega
#: ones are caught by the mega rule first (checked before large below), so
#: any remaining SutureCut name is Large-family. Get the order wrong and
#: "Mega SutureCut Needle Driver" misclassifies as large -- Task 8 hit this
#: exact bug for the same corpus.
_MEGA_MARKERS = ("mega",)
_LARGE_MARKERS = ("large", "suturecut")

NEEDLE_DRIVER = "needle driver"
#: 2, not 1 (controller ruling R28). Version 1 intervals carried
#: {arm, family, start, stop} and omitted which VIDEO PART start/stop are
#: measured against -- even though this function already reads
#: install_case_part to implement spans_part_boundary below, it simply did
#: not emit it. Timestamps RESET at a part boundary (surgvu.labels' own
#: docstring), and R28 measured that duration-probing cannot recover the
#: part after the fact: on 6 multi-part cases, 19 of 69 intervals (28%) fit
#: inside more than one part's duration, so guessing from duration alone
#: mislabels roughly a quarter of a multi-part case's frames with the WRONG
#: part's pixels under a correct-looking family label. Version 2 intervals
#: add "part", so a consumer can tell a part-bearing record from the old
#: part-less shape rather than silently trusting a field that might not be
#: there.
VARIANT_LABELS_VERSION = 2


def family_of(commercial_name):
    """"large", "mega", or None for a commercial tool name.

    None covers every name this task does not have a confident answer for:
    empty/missing values, and real-but-rare names like "DeBakey Forceps" or
    "Wristed Needle Driver SingleSite" that show up once each in the whole
    corpus under groundtruth_toolname == "needle driver" but do not name a
    Large or Mega variant. Those rows are dropped by the caller, never
    defaulted.
    """
    if not commercial_name:
        return None
    text = str(commercial_name).strip().lower()
    if not text:
        return None
    if any(marker in text for marker in _MEGA_MARKERS):
        return "mega"
    if any(marker in text for marker in _LARGE_MARKERS):
        return "large"
    return None


def intervals_for_case(tools_csv, drops=None):
    """Needle-driver install intervals with a resolved family.

    Rows are dropped, and the reason is tallied into `drops` (a
    collections.Counter, created fresh if not supplied) when:

      * groundtruth_toolname is not exactly "needle driver" (not this task's
        job -- most of tools.csv is other tools or the `nan(camera in)`
        camera-motion rows described in the label-vocab-hazards note) --
        tallied under "not_needle_driver", not counted as a "drop" in the
        headline sense since it is simply out of scope, but tracked so the
        totals are auditable.
      * the row is an exact duplicate of one already seen (tools.csv is
        cleaner than tasks.csv but not guaranteed unique) -- "duplicate".
      * the (install_case_part, uninstall_case_part) pair does not match --
        the row spans a part boundary, and case timestamps reset at part
        boundaries, so the interval is not a single coherent timeline
        without extra part-duration bookkeeping -- "spans_part_boundary".
      * either time string does not parse as 'HH:MM:SS.ffffff', or the
        parsed stop is not strictly after start -- "unparseable_time".
      * commercial_toolname does not whitelist to a known family --
        "unrecognised_name". This is the drop this whole task exists to get
        right: never default it to the majority family.

    Every dropped row is a row that would otherwise contribute a
    mislabelled, or merely unverifiable, frame to training.
    """
    if drops is None:
        drops = Counter()
    out = []
    seen = set()
    with open(tools_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("install_case_part"), row.get("install_case_time"),
                   row.get("uninstall_case_part"), row.get("uninstall_case_time"),
                   row.get("arm"), row.get("commercial_toolname"),
                   row.get("groundtruth_toolname"))
            if key in seen:
                drops["duplicate"] += 1
                continue
            seen.add(key)

            if (row.get("groundtruth_toolname") or "").strip().lower() \
                    != NEEDLE_DRIVER:
                drops["not_needle_driver"] += 1
                continue

            part_in = normalize_part(row.get("install_case_part"))
            part_out = normalize_part(row.get("uninstall_case_part"))
            if part_in != part_out:
                drops["spans_part_boundary"] += 1
                continue

            start = parse_hms(row.get("install_case_time"))
            stop = parse_hms(row.get("uninstall_case_time"))
            if start is None or stop is None or stop <= start:
                drops["unparseable_time"] += 1
                continue

            family = family_of(row.get("commercial_toolname"))
            if family is None:
                drops["unrecognised_name"] += 1
                continue

            out.append({
                "start": start,
                "stop": stop,
                "family": family,
                "arm": (row.get("arm") or "").strip(),
                # part_in == part_out is already enforced above (a row that
                # disagrees is dropped as spans_part_boundary), so either
                # name is the interval's one true part. normalize_part's
                # canonical form ("1.0", "2.0", ...) is what a consumer
                # matching this against a video filename's part number
                # expects -- the same helper `find_case_video`-style
                # resolvers already import.
                "part": part_in,
            })
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels-root", required=True,
                         help="Directory of case_NNN/ dirs holding tools.csv. "
                              "Use the AUTHORITATIVE root: "
                              "/staging/groups/bhaskar_opscribe/surgvu/"
                              "labels_cat2/SURGVU25_train_labels")
    parser.add_argument("--out", default="config/variant_labels.json")
    args = parser.parse_args(argv)

    labels_root = Path(args.labels_root)
    cases = {}
    total_drops = Counter()
    n_case_dirs = 0
    n_intervals = 0
    both, large_only, mega_only = 0, 0, 0

    for case_dir in sorted(labels_root.iterdir()):
        tools_csv = case_dir / "tools.csv"
        if not tools_csv.exists():
            continue
        n_case_dirs += 1
        drops = Counter()
        intervals = intervals_for_case(tools_csv, drops=drops)
        for reason, count in drops.items():
            total_drops[reason] += count
        if not intervals:
            continue
        cases[case_dir.name] = intervals
        n_intervals += len(intervals)
        families = {i["family"] for i in intervals}
        if families == {"large", "mega"}:
            both += 1
        elif families == {"large"}:
            large_only += 1
        else:
            mega_only += 1

    n_labeled_cases = len(cases)
    print("case dirs scanned: %d   cases with >=1 needle-driver label: %d"
          % (n_case_dirs, n_labeled_cases))
    print("dropped rows by reason: %s" % dict(sorted(total_drops.items())))
    print("intervals kept: %d" % n_intervals)
    print("cases with both families: %d   large-only: %d   mega-only: %d"
          % (both, large_only, mega_only))
    n_single = large_only + mega_only
    print("single-family cases: %d of %d (%.1f%%) -- a per-case prior "
          "resolves only these; the rest need a per-clip visual label"
          % (n_single, n_labeled_cases,
             100.0 * n_single / max(1, n_labeled_cases)))

    out = {"version": VARIANT_LABELS_VERSION, "cases": cases}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    print("wrote %s (%d cases, %d intervals)"
          % (out_path, n_labeled_cases, n_intervals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
