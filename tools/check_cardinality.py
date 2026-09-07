# -*- coding: utf-8 -*-
"""Corrected check: dedupe rows, sample only inside task segments, respect parts."""
import zipfile, csv, io, os, sys, collections, random

# Directory holding SURGVU25_cat2_train_labels.zip. Pass it as the first
# argument, or set SURGVU_LABELS_DIR; defaults to the working directory.
base = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SURGVU_LABELS_DIR", ".")
z = zipfile.ZipFile(os.path.join(base, "SURGVU25_cat2_train_labels.zip"))


def rows(p):
    return list(csv.DictReader(io.StringIO(z.read(p).decode("utf-8", "replace"))))


def hms(s):
    try:
        h, m, sec = s.split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)
    except Exception:
        return None


cases = sorted(set(n.split("/")[1] for n in z.namelist()
                   if n.count("/") >= 2 and n.endswith(".csv") and "checkpoint" not in n))

counts = collections.Counter()
dup_tool_rows = 0
tot_tool_rows = 0
covered = 0
uncovered = 0
rnd = random.Random(7)

for c in cases:
    tf = "SURGVU25_train_labels/%s/tools.csv" % c
    kf = "SURGVU25_train_labels/%s/tasks.csv" % c
    try:
        trs, krs = rows(tf), rows(kf)
    except KeyError:
        continue

    # --- dedupe tools on the full identity of the interval
    seen = set()
    iv = []
    for r in trs:
        tot_tool_rows += 1
        gt = (r.get("groundtruth_toolname") or "").strip()
        key = (r.get("install_case_part"), r.get("install_case_time"),
               r.get("uninstall_case_part"), r.get("uninstall_case_time"),
               r.get("arm"), gt)
        if key in seen:
            dup_tool_rows += 1
            continue
        seen.add(key)
        if gt.startswith("nan") or not gt:
            continue
        p_in = (r.get("install_case_part") or "").strip()
        p_out = (r.get("uninstall_case_part") or "").strip()
        a, b = hms(r.get("install_case_time", "")), hms(r.get("uninstall_case_time", ""))
        if a is None or b is None:
            continue
        if p_in != p_out:      # spans a part boundary -> skip, needs part durations
            continue
        if b <= a:
            continue
        iv.append((p_in, a, b, gt))

    # --- dedupe task segments, keep only single-part ones
    kseen = set()
    segs = []
    for r in krs:
        key = (r.get("start_part"), r.get("start_time"),
               r.get("stop_part"), r.get("stop_time"),
               r.get("groundtruth_taskname"))
        if key in kseen:
            continue
        kseen.add(key)
        if (r.get("start_part") or "").strip() != (r.get("stop_part") or "").strip():
            continue
        try:
            segs.append(((r.get("start_part") or "").strip(),
                         float(r["start_time"]), float(r["stop_time"])))
        except Exception:
            pass

    # --- sample timestamps INSIDE task segments
    for part, s0, s1 in segs:
        if s1 <= s0:
            continue
        for _ in range(6):
            t = rnd.uniform(s0, s1)
            n = sum(1 for p, a, b, _ in iv if p == part and a <= t <= b)
            counts[n] += 1
            if n > 0:
                covered += 1
            else:
                uncovered += 1

tot = sum(counts.values())
print("=" * 64)
print("TOOLS INSTALLED AT TIMESTAMPS SAMPLED INSIDE TASK SEGMENTS")
print("(duplicate rows removed, part-spanning intervals excluded)")
print("=" * 64)
for n in sorted(counts):
    bar = "#" * int(60.0 * counts[n] / tot)
    print("  %2d tools : %6d  (%5.1f%%) %s" % (n, counts[n], 100.0 * counts[n] / tot, bar))
print()
print("  sampled: %d   with >=1 tool: %.1f%%" % (tot, 100.0 * covered / tot))
print("  duplicate tools.csv rows: %d / %d  (%.0f%%)"
      % (dup_tool_rows, tot_tool_rows, 100.0 * dup_tool_rows / tot_tool_rows))
print()
nz = {k: v for k, v in counts.items() if k > 0}
nzt = sum(nz.values())
print("  Of timestamps with >=1 tool, exactly 3: %.1f%%"
      % (100.0 * counts.get(3, 0) / nzt))
print("  Of timestamps with >=1 tool, <=3:       %.1f%%"
      % (100.0 * sum(v for k, v in nz.items() if k <= 3) / nzt))
