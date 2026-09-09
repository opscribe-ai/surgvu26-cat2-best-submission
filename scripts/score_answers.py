# scripts/score_answers.py
"""Score a predictions file against the Cat 2 sample set.

Usage: python scripts/score_answers.py data/samples predictions.json
predictions.json: {"case122": "Yes", "case123": "No", ...}

Note: data/samples is a flat directory (caseNNN.json ground truth,
caseNNN_question.json question, caseNNN.mp4 video -- no per-case
subdirectories), so case discovery below matches on "caseNNN.json"
files directly rather than "caseNNN/caseNNN.json".
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.scoring import Scorer  # noqa: E402

if __name__ == "__main__":
    samples_root = Path(sys.argv[1])
    predictions = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

    pairs = []
    for gt_file in sorted(samples_root.glob("*.json")):
        case_id = gt_file.stem
        if case_id.endswith("_question"):
            continue
        if case_id not in predictions:
            continue
        references = json.loads(gt_file.read_text(encoding="utf-8"))
        pairs.append((case_id, predictions[case_id], references))

    report = Scorer().score_many(pairs)
    for row in report["results"]:
        print("%-10s %.4f" % (row["case_id"], row["bertscore_f1"]))
    print("-" * 22)
    print("%-10s %.4f" % ("MEAN", report["aggregates"]["bertscore_f1"]))
