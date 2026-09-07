# -*- coding: utf-8 -*-
"""Why does 'exactly three tools' hold 78% or 45% depending on how you count?

Hypothesis: the challenge's 'three instruments installed' fact counts ARM SLOTS.
The tool recogniser predicts DISTINCT CLASSES over a 12-way vocabulary, and two
arms frequently carry the same class (a needle driver on USM1 and a SutureCut
needle driver on USM3 are both `needle driver`). Those collapse to one label.
"""
import collections
import csv
import glob
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from surgvu.labels import parse_hms          # noqa: E402
from surgvu.taxonomy import normalize_tool   # noqa: E402

arms = collections.Counter()
classes = collections.Counter()
dupe_examples = collections.Counter()
rng = random.Random(7)

for tools_path in sorted(glob.glob("data/SURGVU25_train_labels/*/tools.csv")):
    intervals = []
    with open(tools_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tool = normalize_tool(row.get("groundtruth_toolname"))
            if tool is None:
                continue
            p_in = (row.get("install_case_part") or "").strip()
            if p_in != (row.get("uninstall_case_part") or "").strip():
                continue
            start = parse_hms(row.get("install_case_time", ""))
            stop = parse_hms(row.get("uninstall_case_time", ""))
            if start is None or stop is None or stop <= start:
                continue
            intervals.append((p_in, start, stop, tool))

    segments = []
    with open(tools_path.replace("tools.csv", "tasks.csv"), encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            part = (row.get("start_part") or "").strip()
            if part != (row.get("stop_part") or "").strip():
                continue
            try:
                lo, hi = float(row["start_time"]), float(row["stop_time"])
            except (KeyError, TypeError, ValueError):
                continue
            if hi > lo:
                segments.append((part, lo, hi))

    for part, lo, hi in segments:
        for _ in range(6):
            t = rng.uniform(lo, hi)
            live = [tool for p, a, b, tool in intervals if p == part and a <= t <= b]
            arms[len(live)] += 1
            distinct = set(live)
            classes[len(distinct)] += 1
            if len(live) > len(distinct):
                counts = collections.Counter(live)
                for name, n in counts.items():
                    if n > 1:
                        dupe_examples[name] += 1


def show(title, counter):
    total = sum(counter.values())
    print("\n%s  (n=%d)" % (title, total))
    for key in sorted(counter):
        print("   %d : %6d  %5.1f%%" % (key, counter[key], 100.0 * counter[key] / total))


show("Occupied ARM SLOTS (duplicate classes counted separately)", arms)
show("DISTINCT in-scope tool CLASSES (what the model predicts)", classes)

print("\nClasses most often occupying more than one arm at once:")
for name, n in dupe_examples.most_common(6):
    print("   %-32s %6d sampled moments" % (name, n))
