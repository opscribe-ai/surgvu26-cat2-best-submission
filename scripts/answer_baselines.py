"""Emit candidate files for trivial, perception-free answering strategies.

These exist to calibrate the floor of the official metric. None of them look
at the video; several do not even look at the question. Whatever a real system
produces has to beat the best of these, so their scores are the number that
matters before any modelling effort is spent.

Usage:
    python scripts/answer_baselines.py <sample_root> <out_dir>

Writes <out_dir>/<name>.json for every strategy, each a flat
{case_id: answer} mapping that scripts/score_sample.py consumes directly.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_sample import load_sample_cases  # noqa: E402

GENERIC_SENTENCE = "The procedure involves surgical instruments."

# The organizers' polar questions all open with one of these. Matched as a
# whole word so "Isolating ..." is not mistaken for the opener "Is".
_YES_NO_OPENERS = ("is", "are", "was", "does", "do")
_FIRST_WORD = re.compile(r"[a-z']+")


def is_yes_no_question(question):
    """True when the question opens with Is/Are/Was/Does/Do."""
    match = _FIRST_WORD.match((question or "").strip().lower())
    if match is None:
        return False
    return match.group(0) in _YES_NO_OPENERS


def _always_yes(question):
    return "Yes"


def _always_no(question):
    return "No"


def _echo_question(question):
    return question


def _empty(question):
    return ""


def _generic(question):
    return GENERIC_SENTENCE


def _yesno_aware(question):
    """The cheapest non-trivial policy: polarity guess, else a generic sentence."""
    return "Yes" if is_yes_no_question(question) else GENERIC_SENTENCE


BASELINES = {
    "always_yes": _always_yes,
    "always_no": _always_no,
    "echo_question": _echo_question,
    "empty": _empty,
    "generic": _generic,
    "yesno_aware": _yesno_aware,
}


def build_all_baselines(questions):
    """questions: {case_id: question}. Returns {baseline_name: {case_id: answer}}."""
    return {
        name: {case_id: fn(question) for case_id, question in questions.items()}
        for name, fn in BASELINES.items()
    }


if __name__ == "__main__":
    sample_root = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_sample_cases(sample_root)
    questions = {case_id: case.question for case_id, case in cases.items()}

    for name, answers in sorted(build_all_baselines(questions).items()):
        path = out_dir / ("%s.json" % name)
        path.write_text(json.dumps(answers, indent=2, sort_keys=True),
                        encoding="utf-8")
        print("%-14s %d cases -> %s" % (name, len(answers), path))
