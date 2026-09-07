"""Frames in which an instrument's identity is established BY THE LOGBOOK.

THE PROBLEM THIS SOLVES. Telling a Cadiere Forceps from a Maryland Bipolar by
looking is a domain skill, and the largest single item of headroom in this
project (case124, 0.0691) turns on exactly that judgement. On 2026-08-15 I
tried to make it with an insulator-colour heuristic, and then refuted my own
heuristic against a verified example -- so any conclusion resting on someone's
description of a frame is worth very little.

This removes the judgement. It finds windows where ONE grasper is installed and
every alternative in the family is ruled out by the tool labels, for long
enough to be safe from boundary error. In such a window, whatever grasping
instrument is visible MUST be the named one. Nobody has to be believed.

    python scripts/reference_frames.py --tool "cadiere forceps" --out-dir DIR

WHY THE FAMILY MATTERS. Excluding only the same class is not enough: a frame
holding a cadiere AND a prograsp shows two graspers and identifies neither. The
exclusion set is every class that could be confused for a grasper at a glance,
which is what makes the output usable as ground truth rather than as a hint.

WHAT WAS BUILT WITH IT, and why it is a better deliverable than the answer it
was meant to produce:

    CADIERE_ONLY_case_{040,096,016}    ~3,000 s windows each
    MARYLAND_ONLY_case_{062,141}       ~3,000 s windows each

Those survive being wrong about what they show, which the reasoning they were
built to support did not.
"""
import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

LABELS = "/staging/groups/bhaskar_opscribe/surgvu/labels_cat2/SURGVU25_train_labels"
VIDEOS = "/staging/groups/bhaskar_opscribe/surgvu/videos/surgvu24"

#: Everything that reads as a grasper at a glance. A window is only unambiguous
#: for one of these if none of the others is installed at the same time.
GRASPER_FAMILY = frozenset({
    "cadiere forceps", "bipolar forceps", "prograsp forceps", "force bipolar",
    "tip-up fenestrated grasper", "grasping retractor",
})


def seconds(stamp):
    hours, minutes, rest = stamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


def exclusive_windows(tool, commercial=None, min_seconds=120):
    """(duration, case, part, start, end) where `tool` is the ONLY grasper."""
    found = []
    for case_dir in sorted(glob.glob(os.path.join(LABELS, "case_*"))):
        path = os.path.join(case_dir, "tools.csv")
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            rows = [r for r in csv.DictReader(handle)
                    if r["groundtruth_toolname"] in GRASPER_FAMILY]
        for row in rows:
            if row["groundtruth_toolname"] != tool:
                continue
            if commercial and commercial.lower() not in row["commercial_toolname"].lower():
                continue
            start = seconds(row["install_case_time"])
            end = seconds(row["uninstall_case_time"])
            part = row["install_case_part"]
            overlapping = [
                other for other in rows
                if other["groundtruth_toolname"] != tool
                and other["install_case_part"] == part
                and seconds(other["install_case_time"]) < end
                and start < seconds(other["uninstall_case_time"])]
            if not overlapping and end - start >= min_seconds:
                found.append((end - start, os.path.basename(case_dir), part,
                              start, end))
    found.sort(reverse=True)
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tool", default="cadiere forceps",
                        help="groundtruth_toolname to isolate")
    parser.add_argument("--commercial", default=None,
                        help="also require this substring in "
                             "commercial_toolname, e.g. 'Maryland'")
    parser.add_argument("--top", type=int, default=3,
                        help="how many of the longest windows to sample")
    parser.add_argument("--min-seconds", type=int, default=120)
    parser.add_argument("--out-dir",
                        default="/staging/n/nkalthoff/surgvu26/frames")
    parser.add_argument("--list-only", action="store_true",
                        help="print the windows without decoding anything")
    parser.add_argument("--out", help="JSON report")
    args = parser.parse_args(argv)

    windows = exclusive_windows(args.tool, args.commercial, args.min_seconds)
    if not windows:
        raise SystemExit(
            "no window has %r installed with every other grasper absent for "
            "%d s. That is a result about the corpus, not an error -- some "
            "instruments never appear alone." % (args.tool, args.min_seconds))

    label = (args.commercial or args.tool).upper().replace(" ", "_")
    print("%d exclusive windows for %r" % (len(windows), args.tool))
    picked = windows[:args.top]
    for duration, case, part, start, end in picked:
        print("   %-11s part%-5s %9.1f - %9.1f  (%.0f s)"
              % (case, part, start, end, duration))

    report = {"tool": args.tool, "commercial": args.commercial,
              "windows_found": len(windows), "frames": []}
    if not args.list_only:
        import cv2
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        for duration, case, part, start, end in picked:
            middle = (start + end) / 2.0
            number = part.split(".")[0].zfill(3)
            source = "%s/%s/%s_video_part_%s.mp4" % (VIDEOS, case, case, number)
            capture = cv2.VideoCapture(source)
            fps = capture.get(cv2.CAP_PROP_FPS) or 60.0
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(middle * fps))
            ok, image = capture.read()
            capture.release()
            if not ok:
                print("   FAILED to read %s at %.1f" % (case, middle))
                continue
            out = os.path.join(args.out_dir,
                               "%s_ONLY_%s_%d.png" % (label, case, int(middle)))
            cv2.imwrite(out, image)
            print("   wrote %s" % out)
            report["frames"].append({"case": case, "part": part,
                                     "seconds": middle, "path": out})

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
