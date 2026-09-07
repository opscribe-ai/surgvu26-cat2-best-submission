"""Measure how the question router behaves on phrasings nobody wrote down.

The router in src/surgvu/router.py was designed against the ELEVEN public
sample questions. Eleven items can tell you the rules fire on the corpus you
have; they cannot tell you what happens when a challenge annotator writes
"Are staplers deployed?" instead of "Is a stapler being used?".

tests/fixtures/question_variants.json is a hand-written battery of paraphrases
of those eleven, plus families the eleven never touch (tools that appear in no
sample question, counting questions, questions that should legitimately fall
through). This script runs the battery and reports three numbers, in
increasing strictness:

  INTENT      did classify_question pick the right branch?
  TARGET      did mentioned_tool_classes resolve the right taxonomy classes?
              (a presence question can route correctly and still ask about the
              wrong instrument -- "Is a DeBakey Forceps involved?" routes to
              tool presence and then tests the four FORCEPS classes)
  ANSWER      for the subset that pins a perception scenario, is the emitted
              string right? This is where negation lives: "Is there no needle
              driver present?" has the correct intent and the wrong answer.

Usage:
    python scripts/router_coverage.py [fixture.json] [--verbose]

Exit status is 0 regardless of the score. This is a measurement tool, not a
gate: the number is meant to be reported honestly, including when it is bad.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu import router                                     # noqa: E402
from surgvu.taxonomy import TASK_CLASSES, TOOL_CLASSES        # noqa: E402

DEFAULT_FIXTURE = (Path(__file__).resolve().parents[1]
                   / "tests" / "fixtures" / "question_variants.json")

# Fixture intent strings are the VALUES of the router's INTENT_* constants, so
# a fixture that names an intent the router does not implement yet (counting,
# at the time of writing) simply never matches -- which is the honest result.
KNOWN_INTENTS = sorted({
    value for name, value in vars(router).items()
    if name.startswith("INTENT_") and isinstance(value, str)
})


def build_perception(spec):
    """A full-shape perception dict from a compact scenario spec.

    The fixture only names what matters ("cadiere forceps": 0.601); every other
    class is filled with 0.0 so the dict has exactly the contract's shape.
    """
    spec = spec or {}
    tools = {cls: 0.0 for cls in TOOL_CLASSES}
    for name, value in (spec.get("tools") or {}).items():
        if name in tools:
            tools[name] = float(value)
    task = {cls: 0.0 for cls in TASK_CLASSES}
    for name, value in (spec.get("task") or {}).items():
        if name in task:
            task[name] = float(value)
    top = spec.get("task_top")
    if top and not spec.get("task"):
        task[top] = 1.0
    return {
        "tools": tools,
        "tools_present": list(spec.get("tools_present") or []),
        "task": task,
        "task_top": top,
        "n_frames": 16,
    }


def evaluate(fixture):
    """Run every variant; return a list of per-variant result records."""
    scenarios = {name: build_perception(spec)
                 for name, spec in (fixture.get("scenarios") or {}).items()}
    results = []
    for variant in fixture["variants"]:
        question = variant["question"]
        expected = variant["intent"]
        actual = router.classify_question(question)
        record = {
            "question": question,
            "family": variant.get("family", ""),
            "source": variant.get("source", ""),
            "expected_intent": expected,
            "actual_intent": actual,
            "intent_ok": actual == expected,
            "note": variant.get("note", ""),
            "flagged": bool(variant.get("expected_to_fail_at_authoring")),
        }
        if "classes" in variant:
            got = router.mentioned_tool_classes(question)
            record["expected_classes"] = sorted(variant["classes"])
            record["actual_classes"] = sorted(got)
            record["target_ok"] = got == frozenset(variant["classes"])
        if "answer" in variant:
            scenario = variant.get("scenario")
            perception = scenarios.get(scenario, build_perception({}))
            got = router.answer_question(question, perception)
            record["scenario"] = scenario
            record["expected_answer"] = variant["answer"]
            record["actual_answer"] = got
            record["answer_ok"] = got == variant["answer"]
        results.append(record)
    return results


def _rate(hits, total):
    return "%3d/%-3d %6.1f%%" % (hits, total, 100.0 * hits / total if total else 0.0)


def report(results, verbose=False):
    total = len(results)
    intent_hits = sum(r["intent_ok"] for r in results)

    print("=" * 78)
    print("ROUTER COVERAGE  --  %d paraphrase variants" % total)
    print("=" * 78)
    print()
    print("OVERALL INTENT ACCURACY   %s" % _rate(intent_hits, total))

    targeted = [r for r in results if "target_ok" in r]
    if targeted:
        print("TOOL-TARGET ACCURACY      %s   (%d variants name a tool)"
              % (_rate(sum(r["target_ok"] for r in targeted), len(targeted)),
                 len(targeted)))
    answered = [r for r in results if "answer_ok" in r]
    if answered:
        print("END-TO-END ANSWER         %s   (%d variants pin a perception)"
              % (_rate(sum(r["answer_ok"] for r in answered), len(answered)),
                 len(answered)))
    print()

    # ---- per-intent -----------------------------------------------------
    print("-" * 78)
    print("PER-INTENT RECALL (of the variants that SHOULD route here)")
    print("-" * 78)
    print("%-24s %6s %6s %8s   %s" % ("expected intent", "n", "ok", "recall",
                                      "where the misses went"))
    by_expected = defaultdict(list)
    for r in results:
        by_expected[r["expected_intent"]].append(r)
    for intent in sorted(by_expected):
        rows = by_expected[intent]
        ok = sum(r["intent_ok"] for r in rows)
        misses = Counter(r["actual_intent"] for r in rows if not r["intent_ok"])
        where = ", ".join("%s x%d" % (k, v) for k, v in misses.most_common())
        print("%-24s %6d %6d %7.1f%%   %s"
              % (intent, len(rows), ok, 100.0 * ok / len(rows), where or "-"))
    print()

    # ---- confusion ------------------------------------------------------
    actual_intents = sorted({r["actual_intent"] for r in results})
    expected_intents = sorted(by_expected)
    width = max(len(i) for i in expected_intents) + 1
    print("-" * 78)
    print("CONFUSION TABLE  (rows = expected, columns = actual)")
    print("-" * 78)
    header = " " * width + "".join("%6s" % _abbrev(i) for i in actual_intents)
    print(header)
    for exp in expected_intents:
        counts = Counter(r["actual_intent"] for r in by_expected[exp])
        row = "".join("%6s" % (counts.get(act, "") or ".")
                      for act in actual_intents)
        print("%-*s%s" % (width, _abbrev(exp), row))
    print()
    print("  legend: " + "  ".join("%s=%s" % (_abbrev(i), i)
                                   for i in sorted(set(actual_intents)
                                                   | set(expected_intents))))
    print()

    # ---- the misroutes --------------------------------------------------
    misrouted = [r for r in results if not r["intent_ok"]]
    print("-" * 78)
    print("MISROUTED VARIANTS  (%d)" % len(misrouted))
    print("-" * 78)
    for r in misrouted:
        print("  %-62s" % _clip(r["question"], 62))
        print("      %-22s -> %-22s [%s]"
              % (r["expected_intent"], r["actual_intent"], r["family"]))
        if verbose and r["note"]:
            print("      note: %s" % r["note"])
    if not misrouted:
        print("  (none)")
    print()

    wrong_target = [r for r in targeted if not r["target_ok"]]
    print("-" * 78)
    print("WRONG TOOL TARGET  (%d)  -- right branch, wrong instrument set"
          % len(wrong_target))
    print("-" * 78)
    for r in wrong_target:
        print("  %s" % _clip(r["question"], 70))
        print("      expected %s" % ", ".join(r["expected_classes"]))
        print("      actual   %s" % (", ".join(r["actual_classes"]) or "(none)"))
    if not wrong_target:
        print("  (none)")
    print()

    wrong_answer = [r for r in answered if not r["answer_ok"]]
    print("-" * 78)
    print("WRONG ANSWER  (%d)  -- the string we would actually submit"
          % len(wrong_answer))
    print("-" * 78)
    for r in wrong_answer:
        print("  %s   [%s]" % (_clip(r["question"], 60), r["scenario"]))
        print("      expected %-18r actual %r"
              % (r["expected_answer"], r["actual_answer"]))
    if not wrong_answer:
        print("  (none)")
    print()

    # ---- by family ------------------------------------------------------
    print("-" * 78)
    print("BY FAMILY")
    print("-" * 78)
    by_family = defaultdict(list)
    for r in results:
        by_family[r["family"]].append(r)
    for family in sorted(by_family):
        rows = by_family[family]
        ok = sum(r["intent_ok"] for r in rows)
        print("  %-38s %s" % (family, _rate(ok, len(rows))))
    print()
    return {
        "variants": total,
        "intent_accuracy": intent_hits / total if total else 0.0,
        "intent_hits": intent_hits,
        "target_hits": sum(r["target_ok"] for r in targeted),
        "target_total": len(targeted),
        "answer_hits": sum(r["answer_ok"] for r in answered),
        "answer_total": len(answered),
    }


def _abbrev(intent):
    """'tool_presence_polar' -> 'toolP'-ish, for a table that fits 78 columns."""
    head, _, tail = intent.rpartition("_")
    return (head[:4] + tail[:1]).upper() if head else intent[:5].upper()


def _clip(text, width):
    text = str(text)
    return text if len(text) <= width else text[:width - 1] + "…"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", nargs="?", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--verbose", action="store_true",
                        help="print the authoring note for every misroute")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="print the summary as one JSON line at the end")
    args = parser.parse_args(argv)

    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    results = evaluate(fixture)
    summary = report(results, verbose=args.verbose)
    print("router intents implemented: %s" % ", ".join(KNOWN_INTENTS))
    if args.as_json:
        print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
