"""Write the canonical case-level split. Generate once; never regenerate."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.labels import load_all_cases  # noqa: E402
from surgvu.sampling import make_splits  # noqa: E402

if __name__ == "__main__":
    labels_root = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("config/splits.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        print("REFUSING to overwrite %s — the split must stay stable." % out)
        raise SystemExit(1)

    cases = load_all_cases(labels_root)
    splits = make_splits(list(cases), val_fraction=0.2, seed=7)
    out.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print("train %d  val %d" % (len(splits["train"]), len(splits["val"])))
