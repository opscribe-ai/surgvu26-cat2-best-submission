"""Emit config/commercial_names.json: commercial tool names per generic class.

The organizers advised (2026-08-09) that Cat 2 questions may name *commercial*
instruments -- "Large needle driver", "Large SutureCut Needle Driver",
"Cadiere Forceps" -- not only the 12 generic classes the recogniser predicts.

Nothing visual separates a Large needle driver from a Mega one, so the
recogniser stays at class level and this table supplies the commercial prior on
top of it. For most classes that is nearly free: the mapping is close to
one-to-one, so knowing the class effectively names the instrument.

`needle driver` is the exception and the only one worth modelling: it splits
across Large / Mega x plain / SutureCut with no dominant variant.

Rows come from the raw CSVs rather than `CaseLabels`, which drops
`commercial_toolname` at load time. Train split only, for the same reason as
the rarity table: val distribution must not leak into a training-set decision.

Usage:
    python scripts/build_commercial_names.py LABELS_ROOT [SPLITS] [OUT]
"""
import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.taxonomy import normalize_tool     # noqa: E402


def main(argv):
    if not 1 <= len(argv) <= 3:
        raise SystemExit(__doc__)
    labels_root = Path(argv[0])
    splits_path = Path(argv[1]) if len(argv) > 1 else Path("config/splits.json")
    out = Path(argv[2]) if len(argv) > 2 else Path("config/commercial_names.json")

    train = set(json.loads(splits_path.read_text(encoding="utf-8"))["train"])
    counts = collections.defaultdict(collections.Counter)
    for case_id in sorted(train):
        path = labels_root / case_id / "tools.csv"
        if not path.exists():
            raise SystemExit("missing %s" % path)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                generic = normalize_tool(row.get("groundtruth_toolname"))
                if generic is None:                 # endoscope / out of scope
                    continue
                commercial = (row.get("commercial_toolname") or "").strip()
                if commercial:
                    counts[generic][commercial] += 1

    if not counts:
        raise SystemExit("no commercial names found -- refusing to write an "
                         "empty table that would silently disable the prior")

    table = {}
    for generic, variants in counts.items():
        total = sum(variants.values())
        table[generic] = {
            "total": total,
            "variants": [
                {"name": name, "count": n, "share": round(n / total, 4)}
                for name, n in variants.most_common()
            ],
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")

    print("%d generic classes -> %s\n" % (len(table), out))
    for generic in sorted(table, key=lambda g: -len(table[g]["variants"])):
        variants = table[generic]["variants"]
        head = variants[0]
        flag = "  <- AMBIGUOUS" if head["share"] < 0.9 else ""
        print("%-32s %d variant(s), top %.1f%% %r%s"
              % (generic, len(variants), 100 * head["share"], head["name"], flag))


if __name__ == "__main__":
    main(sys.argv[1:])
