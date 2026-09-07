"""Does the terse answer form survive a test set whose references are sentences?

    python scripts/answer_form_eval.py <sample_root> \
        --forms tests/fixtures/answer_forms.json \
        --perception outputs/perception_serving.json \
        --out outputs/answer_form_results.json

WHY THIS EXISTS
---------------
Every answer the router emits is the shortest defensible form -- a bare "Yes",
"Cadiere Forceps", "Uterine horn". That rests on ONE observation about the 11
public sample cases: `reference[0]` is a bare token, BERTScore takes the MAX
over the five references, so a bare token scores exactly 1.0000 while correct
prose scores lower.

If the 2026 test set's references are shaped differently -- one sentence-form
reference instead of five with a terse one first -- that tuning inverts across
EVERY case at once. That is a systemic risk, not a wrong answer here and there,
and it is testable today with the gold we already have.

THE FOUR REFERENCE CONDITIONS
-----------------------------
Each is a SELECTION over the real organizer references. Nothing is invented,
rewritten, or paraphrased; the only thing that changes is which of the five the
metric is allowed to take its max over.

    A_full          all five                today's reality
    B_drop_terse    references[1:]          a test set with no terse reference
    C_one_sentence  references[1:2]         ONE sentence-form reference, the
                                            literal worst case named in the
                                            brief
    D_terse_only    references[:1]          the other extreme: ONE terse
                                            reference. Reported as the anchor
                                            that says how much of today's
                                            0.8766 is riding on reference[0].

A condition that selects nothing is an ERROR, not a zero: `Scorer.score_one`
returns 0.0 for an empty reference list, and a silent 0.0 would look like a
catastrophic form failure instead of a harness bug.

HOW THE NUMBERS ARE COMPUTED, AND THE SELF-CHECK
------------------------------------------------
BERTScore is computed per (candidate, reference) pair independently -- the
metric has no cross-reference term and idf weighting is off -- so every
condition is derivable from ONE vector of five per-reference scores. That is
2.2x less roberta-large work and it also exposes WHICH reference each form
wins on, which is the actual object of study here.

It is not assumed. For every row the derived A_full score is compared against
a direct `Scorer.score_one(candidate, all_references)` call and the run FAILS
if they differ by more than SELF_CHECK_TOLERANCE. If the independence
assumption is ever wrong, this stops rather than reports.

GROUPS
------
Rows are meaned over all 11 cases and also split:

    bare      reference[0] is a bare token (<= BARE_TOKEN_MAX_WORDS words).
              9 of 11 cases. This is the population the terse-form bet is
              actually about.
    phrase    case129 and case130, whose reference[0] is itself a phrase or a
              full sentence. The shipped answers there are already long, so
              they cannot say anything about answer length and would dilute
              the contrast if left mixed in.

THE FALLBACK ARMS
-----------------
The same harness scores the second question: when the router cannot classify a
question it emits a fixed generic sentence. Three arms are built per case --
the shipped constant, a sentence composed from the clip's own detected tools
and task, and a task-only variant -- and scored against the same gold. See
`surgvu.router.perception_sentence` and the honesty note in `fallback_arms`.
"""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from score_sample import load_sample_cases                          # noqa: E402
from surgvu.router import (                                         # noqa: E402
    FALLBACK_OPEN, perception_sentence,
)

# A reference counts as "bare" -- the thing the terse form is tuned to hit --
# when it is at most this many words. "Uterine horn" is two, "Cadiere Forceps"
# is two, "Yes" is one; case129's "Endoscopic surgery or a laparoscopic
# surgery" is six and case130's reference[0] is a whole sentence.
BARE_TOKEN_MAX_WORDS = 3

# Deriving the conditions from one per-reference vector is only valid if
# BERTScore pairs are independent. This is the tolerance on the assertion that
# proves it, per row, at runtime.
#
# 1e-5, not 1e-9. roberta-large runs in float32, which carries about seven
# decimal digits, and scoring a candidate against five references at once
# versus five times separately batches the tensors differently -- so the two
# paths accumulate in a different order and disagree in the last digits. The
# first real run failed at 7.15e-7 (0.689039468765 vs 0.689038753510), which
# is float32 noise rather than a violated assumption.
#
# The tolerance still has teeth: independence failing FOR REAL means the
# metric conditions on the other references, which would move a score by
# order 1e-2, four orders of magnitude above this bar. Loosening to 1e-5 buys
# room for arithmetic and keeps the guard that matters.
SELF_CHECK_TOLERANCE = 1e-5


def condition_indices(condition, n_references):
    """The reference indices a condition selects, as a tuple.

    Index-based rather than content-based on purpose: "drop the terse one" has
    to mean the same thing for every case, and the organizers put the terse
    reference first in all eleven.
    """
    if n_references <= 0:
        raise ValueError("a case with no references cannot be scored")
    indices = tuple(range(n_references))
    if condition == "A_full":
        return indices
    if condition == "B_drop_terse":
        return indices[1:]
    if condition == "C_one_sentence":
        return indices[1:2]
    if condition == "D_terse_only":
        return indices[:1]
    raise ValueError("unknown reference condition %r" % (condition,))


CONDITIONS = ("A_full", "B_drop_terse", "C_one_sentence", "D_terse_only")


def select_references(references, condition):
    """The references a condition leaves in play. Never returns an empty list.

    An empty selection would score 0.0 through `Scorer.score_one` and read as
    a form that failed catastrophically, which is exactly the wrong conclusion
    to draw from a harness bug.
    """
    indices = condition_indices(condition, len(references))
    selected = [references[i] for i in indices]
    if not selected:
        raise ValueError(
            "condition %r selects no reference out of %d; scoring it would "
            "return 0.0 and look like a result"
            % (condition, len(references)))
    return selected


def is_bare_first_reference(references, max_words=BARE_TOKEN_MAX_WORDS):
    """True when reference[0] is a bare token rather than a phrase or sentence."""
    if not references:
        return False
    return 0 < len(str(references[0]).split()) <= max_words


def per_reference_scores(scorer, candidate, references):
    """One BERTScore-F1 per reference, in reference order."""
    return [scorer.score_one(candidate, [reference])["bertscore_f1"]
            for reference in references]


def condition_score(vector, condition):
    """The metric's max, restricted to the references a condition keeps."""
    indices = condition_indices(condition, len(vector))
    kept = [vector[i] for i in indices]
    if not kept:
        raise ValueError("condition %r keeps no reference" % (condition,))
    return max(kept)


def fallback_arms(perception_records, case_ids):
    """{arm: {case_id: answer}} for the generic-vs-perception fallback question.

    THE HONESTY NOTE. The generic sentence fires only on questions the router
    cannot classify, and none of the 11 sample questions is one of them -- so
    this is a COUNTERFACTUAL: for each real case, what would each fallback
    have scored against that case's real gold had the router failed to route
    it? The gold is the organizers'; only the premise is ours. It is thin
    (n=11, of which 4 are open questions) and the report says so rather than
    dressing it up.

    Paraphrases would add nothing: every arm here is a per-CASE constant that
    never looks at the question, so re-asking the same case in other words
    produces the identical candidate string.
    """
    arms = OrderedDict()
    arms["fb_generic"] = {case_id: FALLBACK_OPEN for case_id in case_ids}
    arms["fb_perception"] = {
        case_id: perception_sentence(perception_records.get(case_id, {}))
        for case_id in case_ids}
    arms["fb_task_only"] = {
        case_id: perception_sentence(perception_records.get(case_id, {}),
                                     include_tools=False)
        for case_id in case_ids}
    return arms


def load_forms(path):
    """{form_name: {case_id: answer}} out of the hand-written fixture."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    forms = data.get("forms")
    if not isinstance(forms, dict) or not forms:
        raise ValueError("%s carries no 'forms' object" % (path,))
    out = OrderedDict()
    for name in sorted(forms):
        answers = forms[name]
        if not isinstance(answers, dict) or not answers:
            raise ValueError("form %r in %s is empty" % (name, path))
        out[name] = {case_id: str(answer) for case_id, answer in answers.items()}
    return out


def load_perception(path):
    """{case_id: record}. `records` is tolerated as a wrapper key."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("records"), dict):
        return data["records"]
    return data


def score_rows(scorer, candidate_sets, cases):
    """[{set, case_id, candidate, vector, scores{condition: f1}}], self-checked."""
    rows = []
    for set_name, answers in candidate_sets.items():
        missing = sorted(set(cases) - set(answers))
        if missing:
            raise ValueError("candidate set %r has no answer for %s"
                             % (set_name, ", ".join(missing)))
        for case_id in sorted(cases):
            references = list(cases[case_id].references)
            candidate = answers[case_id]
            vector = per_reference_scores(scorer, candidate, references)
            direct = scorer.score_one(candidate, references)["bertscore_f1"]
            derived = condition_score(vector, "A_full")
            if abs(direct - derived) > SELF_CHECK_TOLERANCE:
                raise AssertionError(
                    "per-reference independence does not hold for %s/%s: "
                    "direct %.12f vs derived %.12f. Every condition in this "
                    "report is derived from that assumption, so this run is "
                    "void." % (set_name, case_id, direct, derived))
            rows.append({
                "set": set_name,
                "case_id": case_id,
                "candidate": candidate,
                "vector": vector,
                "best_index": vector.index(max(vector)),
                "group": "bare" if is_bare_first_reference(references) else "phrase",
                "scores": {condition: condition_score(vector, condition)
                           for condition in CONDITIONS},
            })
    return rows


def aggregate(rows, group=None):
    """{set: {condition: mean f1}} over the rows in a group (None = all)."""
    out = OrderedDict()
    for row in rows:
        if group is not None and row["group"] != group:
            continue
        out.setdefault(row["set"], {c: [] for c in CONDITIONS})
        for condition in CONDITIONS:
            out[row["set"]][condition].append(row["scores"][condition])
    return OrderedDict(
        (set_name, {condition: mean(values[condition]) for condition in CONDITIONS})
        for set_name, values in out.items())


def format_matrix(table, title, n):
    lines = ["== %s (n=%d) ==" % (title, n),
             "%-16s %s" % ("candidate set",
                           " ".join("%14s" % c for c in CONDITIONS)),
             "-" * (16 + 15 * len(CONDITIONS))]
    for set_name, scores in table.items():
        lines.append("%-16s %s" % (set_name,
                                   " ".join("%14.4f" % scores[c]
                                            for c in CONDITIONS)))
    return "\n".join(lines)


def format_rows(rows, cases):
    lines = ["%-16s %-9s %-6s %-46s %s"
             % ("set", "case", "group", "candidate",
                " ".join("%14s" % c for c in CONDITIONS)),
             "-" * 150]
    for row in rows:
        lines.append("%-16s %-9s %-6s %-46s %s"
                     % (row["set"], row["case_id"], row["group"],
                        row["candidate"][:44],
                        " ".join("%14.4f" % row["scores"][c] for c in CONDITIONS)))
    return "\n".join(lines)


def main(argv=None):
    from surgvu.scoring import Scorer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_root")
    parser.add_argument("--forms", default=str(REPO / "tests" / "fixtures"
                                               / "answer_forms.json"))
    parser.add_argument("--perception",
                        default=str(REPO / "outputs" / "perception_serving.json"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    cases = load_sample_cases(args.sample_root)
    forms = load_forms(args.forms)
    perception = load_perception(args.perception)

    candidate_sets = OrderedDict(forms)
    candidate_sets.update(fallback_arms(perception, sorted(cases)))

    print("== the reference structure we actually have ==")
    for case_id in sorted(cases):
        references = cases[case_id].references
        print("%-9s refs=%d  bare_first=%-5s  reference[0]=%r"
              % (case_id, len(references),
                 is_bare_first_reference(references), references[0]))
    print()
    print("== candidate sets ==")
    for set_name, answers in candidate_sets.items():
        print("-- %s" % set_name)
        for case_id in sorted(cases):
            print("   %-9s %r" % (case_id, answers[case_id]))
    print()

    rows = score_rows(scorer=Scorer(), candidate_sets=candidate_sets, cases=cases)

    print(format_rows(rows, cases))
    print()
    print(format_matrix(aggregate(rows), "ALL CASES", len(cases)))
    print()
    bare = [r for r in rows if r["group"] == "bare"]
    print(format_matrix(aggregate(rows, "bare"), "BARE reference[0] only",
                        len(bare) // max(len(candidate_sets), 1)))
    print()
    phrase = [r for r in rows if r["group"] == "phrase"]
    if phrase:
        print(format_matrix(aggregate(rows, "phrase"),
                            "PHRASE/SENTENCE reference[0] only",
                            len(phrase) // max(len(candidate_sets), 1)))

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"rows": rows,
             "aggregate_all": aggregate(rows),
             "aggregate_bare": aggregate(rows, "bare"),
             "aggregate_phrase": aggregate(rows, "phrase")},
            indent=1), encoding="utf-8")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
