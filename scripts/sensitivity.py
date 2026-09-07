"""What perception change would actually change an ANSWER?

Every v2 experiment improved macro-F1 and changed zero answers on the 11
held-out cases. Perception records differed on 6 of 11; the router absorbed
all six. That is not a fluke -- the router asks COARSE questions of a
fine-grained record -- and it means macro-F1 is a proxy the graded metric
mostly ignores.

So stop optimising the proxy. This measures the thing directly: for each
case, which single perception change flips the answer, and in which
direction. Two halves, because 11 cases is too thin to conclude from alone.

PART A -- the 11 real cases. For each, perturb the record one class at a time
and re-route. Reports both directions:

  * REPAIR   a change that turns a wrong answer right. These name exactly
             what a better model would have to get right.
  * BREAK    a change that turns a right answer wrong. These measure
             fragility: a case one threshold-crossing away from being lost is
             a case we are currently winning by luck.

PART B -- generalisation beyond the sample. The router is pure Python and
needs no video, so every question in the paraphrase batteries can be routed
against SYNTHETIC records. That gives, per intent, which classes can change
an answer at all -- a map of where perception effort could ever pay, computed
over hundreds of questions instead of eleven.

WHY EXACT MATCH AGAINST reference[0] IS THE CRITERION. The official metric is
BERTScore-F1 taking the max over five references, and a candidate equal to the
first reference scores exactly 1.0000. So "does this perturbation make the
answer equal to gold[0]" is the right yes/no question, and it needs no scorer,
no roberta-large and no GPU. Answers that merely get CLOSER are not counted;
that is deliberate, because the whole finding this script exists to act on is
that near-misses do not move the graded output.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.router import answer_question, classify_question   # noqa: E402
from surgvu.taxonomy import TASK_CLASSES, TOOL_CLASSES         # noqa: E402

SAMPLE = "/staging/groups/bhaskar_opscribe/surgvu/cat2_sample"


# --------------------------------------------------------------------------
# record surgery
# --------------------------------------------------------------------------

def with_tool(record, tool, present):
    """A copy of `record` with one tool forced present or absent.

    Both `tools_present` and the probability are moved. The router reads the
    presence list for most decisions but falls back to probabilities for
    tool-identity questions and for `credible_tools`, so changing only one of
    them would model a state perception cannot actually produce.
    """
    out = json.loads(json.dumps(record))
    present_set = [t for t in out.get("tools_present", []) if t != tool]
    if present:
        present_set.append(tool)
        out["tools"][tool] = 0.99
    else:
        out["tools"][tool] = 0.01
    out["tools_present"] = sorted(present_set)
    return out


def with_task(record, task):
    """A copy of `record` with the task forced to `task`."""
    out = json.loads(json.dumps(record))
    out["task_top"] = task
    for name in out.get("task", {}):
        out["task"][name] = 0.9 if name == task else 0.01
    return out


def perturbations(record):
    """Every single-change edit worth trying, as (label, record)."""
    edits = []
    for tool in TOOL_CLASSES:
        here = tool in set(record.get("tools_present", []))
        edits.append(("tool:%s=%s" % (tool, "absent" if here else "present"),
                      with_tool(record, tool, not here)))
    for task in TASK_CLASSES:
        if task != record.get("task_top"):
            edits.append(("task:%s" % task, with_task(record, task)))
    return edits


# --------------------------------------------------------------------------
# part A -- the real cases
# --------------------------------------------------------------------------

def part_a(records, sample_root):
    sample_root = Path(sample_root)
    rows = []
    for case in sorted(records):
        record = records[case]
        qpath = sample_root / case / ("%s_question.json" % case)
        gpath = sample_root / case / ("%s.json" % case)
        if not qpath.exists():
            continue
        question = json.loads(qpath.read_text())
        gold = json.loads(gpath.read_text())[0]
        base = answer_question(question, record)
        correct = base.strip().lower() == gold.strip().lower()

        repairs, breaks = [], []
        for label, edited in perturbations(record):
            answer = answer_question(question, edited)
            hit = answer.strip().lower() == gold.strip().lower()
            if not correct and hit:
                repairs.append((label, answer))
            elif correct and not hit:
                breaks.append((label, answer))
        rows.append({"case": case, "question": question, "gold": gold,
                     "answer": base, "correct": correct,
                     "intent": classify_question(question),
                     "repairs": repairs, "breaks": breaks})
    return rows


def report_a(rows):
    print("=" * 78)
    print("PART A -- the 11 held-out cases: what would change the answer")
    print("=" * 78)
    for row in rows:
        mark = "OK  " if row["correct"] else "WRONG"
        print("\n%s %s  [%s]" % (mark, row["case"], row["intent"]))
        print("      Q: %s" % row["question"])
        print("      answer=%r  gold=%r" % (row["answer"], row["gold"]))
        if row["correct"]:
            if row["breaks"]:
                print("      FRAGILE -- %d single change(s) would lose this case:"
                      % len(row["breaks"]))
                for label, answer in row["breaks"][:6]:
                    print("        %-46s -> %r" % (label, answer))
            else:
                print("      robust: no single perception change loses it")
        else:
            if row["repairs"]:
                print("      REPAIRABLE by %d single change(s):" % len(row["repairs"]))
                for label, answer in row["repairs"][:6]:
                    print("        %-46s -> %r" % (label, answer))
            else:
                print("      NOT repairable by any single perception change. "
                      "Perception is not what is costing this case.")

    correct = sum(1 for r in rows if r["correct"])
    fragile = sum(1 for r in rows if r["correct"] and r["breaks"])
    fixable = sum(1 for r in rows if not r["correct"] and r["repairs"])
    unfixable = sum(1 for r in rows if not r["correct"] and not r["repairs"])
    print("\n%d/%d correct | %d of those fragile to a single change | "
          "%d wrong-but-fixable | %d wrong and NOT fixable by perception"
          % (correct, len(rows), fragile, fixable, unfixable))
    return {"correct": correct, "fragile": fragile,
            "fixable": fixable, "unfixable": unfixable}


# --------------------------------------------------------------------------
# part B -- which classes can move an answer at all
# --------------------------------------------------------------------------

def neutral_record():
    """A record with nothing present and a neutral task, as the base state."""
    return {"tools": {t: 0.01 for t in TOOL_CLASSES}, "tools_present": [],
            "task": {t: 0.125 for t in TASK_CLASSES}, "task_top": "other",
            "n_frames": 16}


def part_b(questions):
    """Per intent: how often does each class change the answer?"""
    influence = defaultdict(Counter)
    intent_counts = Counter()
    for question in questions:
        intent = classify_question(question)
        intent_counts[intent] += 1
        base_record = neutral_record()
        base = answer_question(question, base_record)
        for tool in TOOL_CLASSES:
            if answer_question(question, with_tool(base_record, tool, True)) != base:
                influence[intent]["tool:%s" % tool] += 1
        for task in TASK_CLASSES:
            if answer_question(question, with_task(base_record, task)) != base:
                influence[intent]["task:%s" % task] += 1
    return influence, intent_counts


def report_b(influence, intent_counts):
    print("\n" + "=" * 78)
    print("PART B -- which classes can move an answer, over %d questions"
          % sum(intent_counts.values()))
    print("=" * 78)
    print("A class that never appears here cannot change any answer of that "
          "intent,\nso improving it cannot pay however much macro-F1 it adds.\n")
    for intent, total in intent_counts.most_common():
        movers = influence.get(intent, Counter())
        print("%-22s %3d question(s)" % (intent, total))
        if not movers:
            print("     nothing moves it -- answered from a constant or the "
                  "question alone")
            continue
        for name, count in movers.most_common(6):
            print("     %-40s %3d/%-3d (%.0f%%)"
                  % (name, count, total, 100.0 * count / total))
    return


def load_questions(paths, sample_root):
    questions = []
    for case in sorted(Path(sample_root).iterdir()):
        q = case / ("%s_question.json" % case.name)
        if q.exists():
            questions.append(json.loads(q.read_text()))
    for path in paths or []:
        blob = json.loads(Path(path).read_text())
        items = blob if isinstance(blob, list) else blob.get("variants", blob)
        for item in (items if isinstance(items, list) else []):
            text = item.get("question") if isinstance(item, dict) else item
            if isinstance(text, str):
                questions.append(text)
    return questions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", required=True,
                        help="perceive_cases.py output for the sample clips")
    parser.add_argument("--sample-root", default=SAMPLE)
    parser.add_argument("--variants", nargs="*",
                        help="paraphrase battery JSON files, for part B")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    records = json.loads(Path(args.records).read_text())
    if "cases" in records and isinstance(records["cases"], dict):
        records = records["cases"]
    rows = part_a(records, args.sample_root)
    summary = report_a(rows)

    questions = load_questions(args.variants, args.sample_root)
    influence, counts = part_b(questions)
    report_b(influence, counts)

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"summary": summary, "cases": rows,
             "influence": {k: dict(v) for k, v in influence.items()},
             "intent_counts": dict(counts)}, indent=2), encoding="utf-8")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
