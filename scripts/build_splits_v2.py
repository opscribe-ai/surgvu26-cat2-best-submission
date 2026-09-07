"""Write config/splits_v2.json: the heldout-excluded, tool-stratified case split.

`config/splits.json` (v1) has two defects, both verified against the real
shards:

  1. All 11 public sample cases — the only question-and-answer data that
     exists for Category 2 — are inside it: 8 in train, 3 in val. The sample
     directories are named `case122` and the split names its cases
     `case_122`, so a raw set intersection reports no overlap and the leak
     reads as clean.

  2. It is a plain shuffle, so nothing balanced the tool classes.
     `tip-up fenestrated grasper` landed with 157 train windows and 0 val
     windows. Its per-class F1 is 0.0 by construction, dragging macro-F1 down
     by roughly 0.056 for a purely structural reason; `stapler` got 13 val
     windows, which is noise.

v2 holds the 11 sample cases out as `heldout` and splits the remaining 144 cases
by CASE, stratified so every tool class reaches a usable val count. See
`surgvu.sampling.make_splits_v2` for the assignment and `MIN_VAL_WINDOWS` for
what "usable" means.

config/splits.json is NOT touched and must never be: the current checkpoints
were trained on it, and overwriting it destroys their provenance. This script
refuses to write to any path named splits.json.

Usage:
    python scripts/build_splits_v2.py [--shard-dir DIR] [--sample-dir DIR]
                                      [--out PATH] [--seed N]
                                      [--val-fraction F] [--min-val-windows N]
                                      [--force]

Re-running with the same inputs and seed reproduces the file byte for byte:
nothing in it is a timestamp.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.sampling import (  # noqa: E402
    MIN_VAL_WINDOWS, make_splits_v2, normalize_case_id,
)
from surgvu.taxonomy import TOOL_CLASSES  # noqa: E402

STAGING = Path("/staging/groups/bhaskar_opscribe/surgvu")
DEFAULT_SHARDS = STAGING / "shards"
DEFAULT_SAMPLE = STAGING / "cat2_sample"
DEFAULT_OUT = Path("config/splits_v2.json")

FROZEN = "splits.json"


def read_shard_meta(path):
    """The window table inside one shard.

    It is stored as a JSON string in a 0-d object array, so it must be pulled
    out with `str(z['meta'])` and parsed. Iterating `z['meta']` directly
    iterates a 0-d array and raises; `z['meta'].item()` happens to work but
    reads as if the array were the table. Only the `meta` member is touched,
    so this never decompresses the JPEG payload — 235 shards is 38 GB on disk
    and under a second of reading here.
    """
    with np.load(str(path), allow_pickle=True) as shard:
        return json.loads(str(shard["meta"]))


def read_all_shard_meta(shard_dir):
    """`{shard filename: window table}` for every shard in `shard_dir`."""
    shard_dir = Path(shard_dir)
    paths = sorted(shard_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(
            "no *.npz shards under %s — a split built from nothing would be "
            "an empty file that looks like a successful run" % shard_dir)
    return {p.name: read_shard_meta(p) for p in paths}


def case_windows_from_shard_meta(meta_by_shard):
    """Regroup shard metadata into `{case_id: [[tool, ...], ...]}`.

    A case has one shard per video part, so the parts must be summed back
    together before anything is counted per case — assigning `case_012_part1`
    to train and `case_012_part2` to val would leak near-duplicate frames
    across the split boundary.
    """
    windows = {}
    for name, rows in sorted(meta_by_shard.items()):
        stem = name[:-4] if name.endswith(".npz") else name
        case, sep, _part = stem.rpartition("_part")
        if not sep or not case:
            raise ValueError(
                "shard %r has no _partN suffix, so the case it belongs to "
                "cannot be identified; dropping it would silently shrink the "
                "corpus" % name)
        windows.setdefault(case, []).extend(list(row["tools"]) for row in rows)
    return windows


def heldout_case_ids(names):
    """Sorted, normalised case ids from a listing of sample directories."""
    return sorted({normalize_case_id(name) for name in names})


def _report(result):
    meta = result["meta"]
    lines = ["cases: train %d  val %d  heldout %d"
             % (meta["case_counts"]["train"], meta["case_counts"]["val"],
                meta["case_counts"]["heldout"]),
             "windows: train %d  val %d  heldout %d"
             % (meta["window_totals"]["train"], meta["window_totals"]["val"],
                meta["window_totals"]["heldout"]),
             "",
             "%-32s %8s %8s %8s %8s" % ("tool class", "train", "val", "heldout",
                                        "val%")]
    for name in sorted(TOOL_CLASSES,
                       key=lambda n: sum(meta["window_counts"][n].values())):
        row = meta["window_counts"][name]
        pool = row["train"] + row["val"]
        share = (100.0 * row["val"] / pool) if pool else 0.0
        flag = ""
        if name in meta["absent"]:
            flag = "  ABSENT"
        elif name in meta["underrepresented"]:
            flag = "  UNDER-REPRESENTED"
        elif name in meta["over_represented"]:
            flag = "  OVER-REPRESENTED (in %d cases)" % meta["cases_per_class"][name]
        lines.append("%-32s %8d %8d %8d %7.1f%%%s"
                     % (name, row["train"], row["val"], row["heldout"], share, flag))
    if meta["over_represented"]:
        lines += ["", "NOTE: " + meta["distortion_note"]]
    if meta["underrepresented"]:
        lines += ["", "WARNING: " + meta["underrepresented_note"]]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shard-dir", default=str(DEFAULT_SHARDS))
    parser.add_argument("--sample-dir", default=str(DEFAULT_SAMPLE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-val-windows", type=int, default=MIN_VAL_WINDOWS)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing splits_v2.json")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if out.name == FROZEN:
        raise SystemExit(
            "REFUSING to write %s. config/splits.json is frozen: the current "
            "checkpoints were trained on it, and overwriting it destroys the "
            "only record of what they saw. v2 is a NEW file." % out)
    if out.exists() and not args.force:
        raise SystemExit(
            "%s already exists. Pass --force only if you are certain nothing "
            "has been trained or scored against it yet." % out)

    sample_dir = Path(args.sample_dir)
    if not sample_dir.is_dir():
        raise SystemExit("no sample directory at %s" % sample_dir)
    heldout = heldout_case_ids(p.name for p in sample_dir.iterdir() if p.is_dir())
    if not heldout:
        raise SystemExit(
            "found no sample cases under %s — an empty heldout list would hand "
            "back exactly the leak this script exists to remove" % sample_dir)

    windows = case_windows_from_shard_meta(read_all_shard_meta(args.shard_dir))
    result = make_splits_v2(windows, heldout, val_fraction=args.val_fraction,
                            seed=args.seed,
                            min_val_windows=args.min_val_windows,
                            restarts=args.restarts)
    result["meta"]["source_shards"] = str(Path(args.shard_dir))
    result["meta"]["source_sample_cases"] = str(sample_dir)
    result["meta"]["generated_by"] = (
        "python scripts/build_splits_v2.py --seed %d --val-fraction %s "
        "--min-val-windows %d --restarts %d  (deterministic: same inputs and "
        "seed reproduce this file byte for byte)"
        % (args.seed, args.val_fraction, args.min_val_windows, args.restarts))
    result["meta"]["supersedes"] = (
        "config/splits.json, which is left untouched — the current "
        "checkpoints were trained on it.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("wrote %s" % out)
    print(_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
