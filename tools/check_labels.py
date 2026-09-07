# -*- coding: utf-8 -*-
"""Validate the assumptions the training plan rests on."""
import zipfile, csv, io, os, sys, collections, random

# Directory holding SURGVU25_cat2_train_labels.zip. Pass it as the first
# argument, or set SURGVU_LABELS_DIR; defaults to the working directory.
base = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SURGVU_LABELS_DIR", ".")
z = zipfile.ZipFile(os.path.join(base, "SURGVU25_cat2_train_labels.zip"))


def rows(path):
    return list(csv.DictReader(io.StringIO(z.read(path).decode("utf-8", "replace"))))


def hms(s):
    """'00:41:01.008000' -> seconds"""
    try:
        h, m, sec = s.split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)
    except Exception:
        return None


tool_files = sorted(n for n in z.namelist()
                    if n.endswith("tools.csv") and "checkpoint" not in n)
task_files = sorted(n for n in z.namelist()
                    if n.endswith("tasks.csv") and "checkpoint" not in n)

# ---------------------------------------------------------------- 1. exactly three?
print("=" * 68)
print("1. IS IT ALWAYS EXACTLY THREE NON-ENDOSCOPE TOOLS?")
print("=" * 68)

counts = collections.Counter()
parts_seen = collections.Counter()
sampled = 0

for f in tool_files:
    rs = rows(f)
    intervals = []
    for r in rs:
        gt = (r.get("groundtruth_toolname") or "").strip()
        part = (r.get("install_case_part") or "").strip()
        parts_seen[part] += 1
        if gt.startswith("nan"):          # endoscope / camera
            continue
        a, b = hms(r.get("install_case_time", "")), hms(r.get("uninstall_case_time", ""))
        if a is None or b is None or b <= a:
            continue
        intervals.append((part, a, b, gt))
    if not intervals:
        continue
    # sample timestamps within part 1.0 only, to keep the time base consistent
    p1 = [iv for iv in intervals if iv[0] == "1.0"]
    if not p1:
        continue
    lo = min(iv[1] for iv in p1)
    hi = max(iv[2] for iv in p1)
    rnd = random.Random(0)
    for _ in range(40):
        t = rnd.uniform(lo, hi)
        n = sum(1 for _, a, b, _ in p1 if a <= t <= b)
        counts[n] += 1
        sampled += 1

total = sum(counts.values())
for n in sorted(counts):
    print("   %2d tools installed : %6d samples  (%5.1f%%)"
          % (n, counts[n], 100.0 * counts[n] / total))
print("   sampled timestamps:", sampled)

# ---------------------------------------------------------------- 2. parts
print()
print("=" * 68)
print("2. HOW MANY VIDEO PARTS PER CASE?")
print("=" * 68)
print("   distinct install_case_part values:", dict(parts_seen.most_common(8)))

per_case_parts = []
for f in tool_files:
    ps = set((r.get("install_case_part") or "").strip() for r in rows(f))
    per_case_parts.append(len(ps))
print("   parts per case: min %d  max %d  mean %.2f"
      % (min(per_case_parts), max(per_case_parts),
         sum(per_case_parts) / float(len(per_case_parts))))

# ---------------------------------------------------------------- 3. duplicate task rows
print()
print("=" * 68)
print("3. DUPLICATE / OVERLAPPING ROWS IN tasks.csv?")
print("=" * 68)
dupe_cases = 0
tot_rows = 0
tot_unique = 0
for f in task_files:
    rs = rows(f)
    keys = [(r.get("start_time"), r.get("stop_time"), r.get("groundtruth_taskname")) for r in rs]
    tot_rows += len(keys)
    tot_unique += len(set(keys))
    if len(set(keys)) < len(keys):
        dupe_cases += 1
print("   cases with duplicate rows: %d / %d" % (dupe_cases, len(task_files)))
print("   total rows %d  ->  unique %d  (%.0f%% redundant)"
      % (tot_rows, tot_unique, 100.0 * (1 - tot_unique / float(tot_rows))))

# ---------------------------------------------------------------- 4. class balance
print()
print("=" * 68)
print("4. TOOL CLASS BALANCE (install intervals, endoscope excluded)")
print("=" * 68)
cls = collections.Counter()
dur = collections.Counter()
for f in tool_files:
    for r in rows(f):
        gt = (r.get("groundtruth_toolname") or "").strip()
        if not gt or gt.startswith("nan"):
            continue
        a, b = hms(r.get("install_case_time", "")), hms(r.get("uninstall_case_time", ""))
        cls[gt] += 1
        if a is not None and b is not None and b > a:
            dur[gt] += (b - a)
worst = max(cls.values())
for k, v in cls.most_common():
    print("   %-32s %5d intervals  %8.1f h   1:%.0f"
          % (k, v, dur[k] / 3600.0, worst / float(v)))

# ---------------------------------------------------------------- 5. task balance
print()
print("=" * 68)
print("5. TASK CLASS BALANCE (case-normalised)")
print("=" * 68)
tcls = collections.Counter()
tdur = collections.Counter()
for f in task_files:
    seen = set()
    for r in rows(f):
        name = (r.get("groundtruth_taskname") or "").strip().lower()
        key = (r.get("start_time"), r.get("stop_time"), name)
        if not name or key in seen:
            continue
        seen.add(key)
        tcls[name] += 1
        try:
            tdur[name] += float(r["stop_time"]) - float(r["start_time"])
        except Exception:
            pass
for k, v in tcls.most_common():
    print("   %-38s %5d segments   %8.1f h" % (k, v, tdur[k] / 3600.0))
