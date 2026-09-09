"""Score a candidate-answer file against the Cat 2 public sample set.

Usage:
    python scripts/score_sample.py <sample_root> <candidates.json> [--label NAME]

<sample_root> may be either layout:
    nested  -- <root>/caseNNN/caseNNN.json + caseNNN_question.json  (the staged
              cat2_sample set)
    flat    -- <root>/caseNNN.json + caseNNN_question.json

<candidates.json> is a flat {case_id: answer} mapping. Every case found in the
sample root must appear in it: a missing case is an error, not a case worth
0.0 and not a case quietly dropped, because silently scoring 10 of 11 inflates
the mean.

The metric is surgvu.scoring.Scorer -- BERTScore-F1, roberta-large, rescaled
with baseline, MAX over the five references, meaned across cases. Run this
inside the extract container with the scoring venv:

    apptainer exec -B /staging --env HF_HOME=/staging/n/nkalthoff/surgvu26/hf_cache \\
      /staging/n/nkalthoff/surgvu26/surgvu26-extract.sif \\
      /staging/n/nkalthoff/surgvu26/env/bin/python3 scripts/score_sample.py ...
"""
import json
import sys
from collections import namedtuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SampleCase = namedtuple("SampleCase", "case_id question references")

_QUESTION_SUFFIX = "_question"


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_sample_cases(sample_root):
    """Discover every case under sample_root. Returns {case_id: SampleCase}.

    Handles both the nested (caseNNN/caseNNN.json) and flat (caseNNN.json)
    layouts by globbing for reference files at either depth.
    """
    sample_root = Path(sample_root)
    cases = {}
    for gt_path in sorted(list(sample_root.glob("*.json"))
                          + list(sample_root.glob("*/*.json"))):
        case_id = gt_path.stem
        if case_id.endswith(_QUESTION_SUFFIX):
            continue
        question_path = gt_path.with_name("%s%s.json" % (case_id, _QUESTION_SUFFIX))
        if not question_path.exists():
            raise ValueError("case %s has no question file at %s"
                             % (case_id, question_path))
        references = _read_json(gt_path)
        if not isinstance(references, list) or not references:
            raise ValueError("case %s: expected a non-empty list of references"
                             % case_id)
        cases[case_id] = SampleCase(case_id, _read_json(question_path),
                                    [str(r) for r in references])
    if not cases:
        raise ValueError("found no cases under %s" % sample_root)
    return cases


def load_candidates(path):
    """Read a {case_id: answer} mapping, rejecting anything that is not a string."""
    data = _read_json(Path(path))
    if not isinstance(data, dict):
        raise ValueError("%s: expected an object mapping case_id -> answer" % path)
    for case_id, answer in data.items():
        if not isinstance(answer, str):
            raise ValueError("candidate for %s is %s, not a string"
                             % (case_id, type(answer).__name__))
    return data


def build_pairs(cases, candidates):
    """Return [(case_id, candidate, references)] sorted by case_id.

    Fails loudly on any mismatch in either direction: a sample case with no
    candidate, or a candidate for a case that is not in the sample set.
    """
    missing = sorted(set(cases) - set(candidates))
    if missing:
        raise ValueError("no candidate answer for %d case(s): %s"
                         % (len(missing), ", ".join(missing)))
    unknown = sorted(set(candidates) - set(cases))
    if unknown:
        raise ValueError("candidates for %d unknown case(s): %s"
                         % (len(unknown), ", ".join(unknown)))
    return [(case_id, candidates[case_id], list(cases[case_id].references))
            for case_id in sorted(cases)]


def best_references(scorer, pairs):
    """{case_id: the reference that scored highest} -- for the report only.

    BERTScore is computed per (candidate, reference) pair independently, so
    scoring each reference on its own picks the same winner the max in
    Scorer.score_one does.
    """
    best = {}
    for case_id, candidate, references in pairs:
        scored = [(scorer.score_one(candidate, [ref])["bertscore_f1"], ref)
                  for ref in references]
        best[case_id] = max(scored, key=lambda pair: pair[0])[1]
    return best


def _clip(text, width):
    text = " ".join(str(text).split())
    if len(text) <= width:
        return text
    return text[:width - 1] + "…"


def format_table(cases, candidates, report, best_refs, width=34):
    """Render the per-case table plus the aggregate as plain text."""
    header = "%-9s %-*s %-*s %-*s %8s" % (
        "case", width, "question", width, "candidate", width, "best reference",
        "bert_f1")
    lines = [header, "-" * len(header)]
    for row in report["results"]:
        case_id = row["case_id"]
        lines.append("%-9s %-*s %-*s %-*s %8.4f" % (
            case_id,
            width, _clip(cases[case_id].question, width),
            width, _clip(candidates[case_id], width) or "<empty>",
            width, _clip(best_refs.get(case_id, ""), width),
            row["bertscore_f1"]))
    lines.append("-" * len(header))
    lines.append("%-9s %-*s %-*s %-*s %8.4f" % (
        "MEAN", width, "", width, "", width, "",
        report["aggregates"]["bertscore_f1"]))
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    from surgvu.scoring import Scorer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_root", help="directory holding the sample cases")
    parser.add_argument("candidates", help="JSON file of {case_id: answer}")
    parser.add_argument("--label", default=None,
                        help="name printed above the table")
    args = parser.parse_args()

    cases = load_sample_cases(args.sample_root)
    candidates = load_candidates(args.candidates)
    pairs = build_pairs(cases, candidates)

    scorer = Scorer()
    report = scorer.score_many(pairs)
    refs = best_references(scorer, pairs)

    if args.label:
        print("== %s ==" % args.label)
    print(format_table(cases, candidates, report, refs))
