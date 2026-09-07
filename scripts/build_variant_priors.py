"""Write config/variant_priors.json — P(commercial name | class present in a clip).

`config/commercial_names.json` counts INSTALLATIONS. That is the wrong
denominator for answering questions about a 30-second clip, for two reasons:

  1. It weights a tool installed for ten seconds the same as one installed for
     an hour, so it does not describe what a randomly chosen clip contains.
  2. It ignores co-occurrence. Four arms are in play, and a clip that holds any
     needle driver usually holds more than one variant at once. By installation
     share "Large Needle Driver" is only 0.244 of needle-driver installs; but
     given the CLASS is present in a clip, a literal "Large Needle Driver" is
     installed on some arm 0.824 of the time.

That second number is the one the router needs, both to choose the surface form
it emits for a class and to decide whether "was a large needle driver used?"
can be answered from a 12-class output at all. This script measures it directly
by sweeping clip-sized windows over the label tables.

Usage:
    python scripts/build_variant_priors.py <labels_root> [out_path]

<labels_root> is the directory of case_NNN/ dirs holding tools.csv + tasks.csv.
Use the AUTHORITATIVE root:

    /staging/groups/bhaskar_opscribe/surgvu/labels_cat2/SURGVU25_train_labels

NOT `labels_v2/labels`. That is the 2024 release -- the "_v2" is a revision of
the 2024 labels, not a successor to the 2025 ones, and it predates them by nine
months. It lacks the `matched_description` column the Cat 2 answers come from,
and it cannot enumerate 5,565 of the windows the shards actually contain. This
example previously named labels_v2 and that is how the committed
config/variant_priors.json came to be built from the wrong root: it made
`stapler` display as "SureForm Stapler 60" when the correct modal surface form
is "Stapler". See docs/OUTSTANDING.md, "Which label root is authoritative".
"""
import csv
import json
import sys
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.labels import normalize_part, parse_hms  # noqa: E402
from surgvu.taxonomy import TOOL_CLASSES, normalize_task, normalize_tool  # noqa: E402

WINDOW_SECONDS = 30.0

_Interval = namedtuple("_Interval", "part start stop cls commercial")


def load_intervals(case_dir):
    """Tool installation intervals, keeping the commercial name.

    Deliberately mirrors CaseLabels._load_tools -- same dedupe key, same
    part-boundary and ordering guards -- but keeps `commercial_toolname`,
    which CaseLabels discards because the model never predicts it.
    """
    intervals, seen = [], set()
    with open(case_dir / "tools.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("install_case_part"), row.get("install_case_time"),
                   row.get("uninstall_case_part"), row.get("uninstall_case_time"),
                   row.get("arm"), row.get("groundtruth_toolname"))
            if key in seen:
                continue
            seen.add(key)
            cls = normalize_tool(row.get("groundtruth_toolname"))
            if cls is None:
                continue
            part = normalize_part(row.get("install_case_part"))
            if part != normalize_part(row.get("uninstall_case_part")):
                continue
            start = parse_hms(row.get("install_case_time"))
            stop = parse_hms(row.get("uninstall_case_time"))
            if start is None or stop is None or stop <= start:
                continue
            name = (row.get("commercial_toolname") or "").strip()
            if not name:
                continue
            intervals.append(_Interval(part, start, stop, cls, name))
    return intervals


def load_segments(case_dir):
    """(part, start, stop) for every in-vocabulary task segment."""
    segments = []
    with open(case_dir / "tasks.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if normalize_task(row.get("groundtruth_taskname")) is None:
                continue
            part = normalize_part(row.get("start_part"))
            if part != normalize_part(row.get("stop_part")):
                continue
            try:
                start, stop = float(row["start_time"]), float(row["stop_time"])
            except (KeyError, TypeError, ValueError):
                continue
            if stop > start:
                segments.append((part, start, stop))
    return segments


def measure(labels_root, window=WINDOW_SECONDS):
    """{class: {"windows": n, "variants": [{"name":, "p_present":}, ...]}}.

    Windows tile the task segments, matching how the graded clips are drawn.
    A variant is counted once per window no matter how many arms hold it.
    """
    present = Counter()
    variant_windows = defaultdict(Counter)
    total = 0
    for case_dir in sorted(Path(labels_root).iterdir()):
        if not (case_dir / "tools.csv").exists():
            continue
        intervals = load_intervals(case_dir)
        for part, start, stop in load_segments(case_dir):
            edge = start
            while edge + window <= stop:
                middle = edge + window / 2.0
                live = [iv for iv in intervals if iv.part == part
                        and iv.start <= middle <= iv.stop]
                total += 1
                for cls in {iv.cls for iv in live}:
                    present[cls] += 1
                    for name in {iv.commercial for iv in live if iv.cls == cls}:
                        variant_windows[cls][name] += 1
                edge += window

    out = {}
    for cls in TOOL_CLASSES:
        count = present[cls]
        if not count:
            continue
        out[cls] = {
            "windows": count,
            "p_class_present": round(count / total, 4) if total else 0.0,
            "variants": [{"name": name, "p_present": round(n / count, 4)}
                         for name, n in variant_windows[cls].most_common()],
        }
    return {"total_windows": total, "window_seconds": window, "classes": out}


if __name__ == "__main__":
    labels_root = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path(__file__).resolve().parents[1] / "config" / "variant_priors.json")
    report = measure(labels_root)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print("%d windows over %d classes -> %s"
          % (report["total_windows"], len(report["classes"]), out_path))
    for cls, entry in sorted(report["classes"].items()):
        top = entry["variants"][0]
        print("  %-32s P(class)=%.3f  modal %s p=%.3f"
              % (cls, entry["p_class_present"], top["name"], top["p_present"]))
