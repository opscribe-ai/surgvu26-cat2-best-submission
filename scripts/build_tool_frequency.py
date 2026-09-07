"""Emit config/tool_frequency.json: how often each tool class appears, corpus-wide.

`stratify` favours windows containing rare tools. Rarity only means anything
against the whole dataset, but a per-(case, part) Condor job can only see its
own windows, so left to itself each job computes rarity locally -- and a tool
that is rare across all 155 cases while being common inside one of them gets
deprioritised exactly where it was most needed. That is backwards, and it is
invisible in the output.

The fix is a pre-pass: compute the table once here, ship it to every job the
way config/splits.json is already shipped, and pass it to `stratify`.

Counts come from the TRAIN split only. Validation windows are not part of what
the extraction is selecting for, and folding them in would leak the val
distribution into a training-set decision.

Usage:
    python scripts/build_tool_frequency.py LABELS_ROOT [SPLITS] [OUT]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.labels import load_all_cases          # noqa: E402
from surgvu.sampling import enumerate_windows, tool_frequency  # noqa: E402
from surgvu.taxonomy import TOOL_CLASSES          # noqa: E402


def main(argv):
    if not 1 <= len(argv) <= 3:
        raise SystemExit(__doc__)
    labels_root = Path(argv[0])
    splits_path = Path(argv[1]) if len(argv) > 1 else Path("config/splits.json")
    out = Path(argv[2]) if len(argv) > 2 else Path("config/tool_frequency.json")

    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    train = set(splits["train"])
    cases = load_all_cases(labels_root)

    missing = train - set(cases)
    if missing:
        raise SystemExit(
            "%d case(s) in the train split have no labels on disk: %s"
            % (len(missing), sorted(missing)[:5]))

    windows = []
    for case_id in sorted(train):
        windows.extend(enumerate_windows(case_id, cases[case_id]))
    if not windows:
        raise SystemExit(
            "no windows enumerated from %d train cases -- refusing to write a "
            "frequency table that would make every tool look equally rare"
            % len(train))

    frequency = tool_frequency(windows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(frequency, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")

    print("%d train cases, %d windows -> %s" % (len(train), len(windows), out))
    for name in sorted(TOOL_CLASSES, key=lambda n: frequency[n]):
        share = 100.0 * frequency[name] / len(windows)
        print("  %-32s %7d  %5.1f%%" % (name, frequency[name], share))

    absent = [n for n in TOOL_CLASSES if frequency[n] == 0]
    if absent:
        print("\nNOTE: %d class(es) never appear in a train window: %s\n"
              "Rarity ordering cannot prioritise what is not there; these can "
              "only be learned from the val split or not at all."
              % (len(absent), absent))


if __name__ == "__main__":
    main(sys.argv[1:])
