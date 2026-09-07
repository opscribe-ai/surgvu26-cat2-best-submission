"""Plural instrument questions: name one tool, or name all of them?

THE GAP. `_answer_tool_identity` emits EXACTLY ONE tool name, and the router
sends plural questions to it -- "What tools are used in this step?", "Which
instruments are used in this step?" are both tool_identity_open and both get a
single noun back. That is a real bet and it has never been measured. Every
answer-form study so far has compared one noun against a SENTENCE, and terse
always won; this compares one noun against a LIST of nouns, which is a
different question. The tokens a second instrument name adds are not padding,
they are content the gold reference may well contain.

WHY IT MATTERS MORE THAN IT LOOKS. The standing finding of this project is
that the router, not perception, is the bottleneck: it asks binary questions
of a 12-dimensional probability vector and discards the rest. A plural
question is the clearest case of that loss. Perception knows three instruments
are installed and the router says one word.

WHAT IS MEASURED. Real co-occurring tool sets, taken from the validation
dump's own labels rather than invented, at each list size. For each set:

    k = 1..m    the first k names, all correct, joined the way a person would
    all + 1     every true name plus one wrong one -- the cost of over-listing
    single      the mean over WHICH single name gets picked, so the k=1 row is
                not an artefact of which tool happened to sort first

References are built in the sample's own style, bare list first, because the
organizers' lists lead with a bare token.

THE POLICY QUESTION, AND WHY THE ANSWER IS DECISIVE EITHER WAY. Combining the
payoff table with the real distribution of m gives the expected score of two
policies: emit one name always, or emit every installed instrument. The second
is scored with PERFECT PERCEPTION -- the true set, handed over free. That is
deliberately generous: real perception is at 0.78 macro-F1 and would list a
wrong tool sometimes. So if "emit all" loses even here, the question is closed
and the router keeps its single noun. If it wins, the margin is an UPPER BOUND
and a second study has to discount it by how often perception's set is right.

SCOPE, as with every study in this family: this measures how roberta-large
treats these surface forms against references we have written in the
organizers' style. It is not a claim about what the 2026 references contain.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.router import display_name                          # noqa: E402
from surgvu.taxonomy import TOOL_CLASSES                        # noqa: E402

DUMP = "/staging/n/nkalthoff/surgvu26/v2/frame_probs_resnetlong_val.npz"

#: Reference templates, bare list first. Mirrors the count study's shape.
TEMPLATES = (
    "{x}",
    "The instruments in use are {x}.",
    "{x} are being used.",
)


def join_names(names):
    """"A", "A and B", "A, B and C" -- how a person writes a short list."""
    names = list(names)
    if len(names) == 1:
        return names[0]
    return "%s and %s" % (", ".join(names[:-1]), names[-1])


def build_refs(names):
    phrase = join_names(names)
    refs = [t.format(x=phrase) for t in TEMPLATES]
    if len(names) == 1:
        # A one-instrument gold would not say "are being used". Fixing the
        # number agreement matters: a grammatical mismatch inside the
        # reference would depress every candidate equally and quietly shift
        # the whole m=1 row.
        refs = [names[0],
                "The instrument in use is %s." % names[0],
                "%s is being used." % names[0]]
    return refs


def observed_sets(dump_path, max_size, per_size):
    """The most common real tool combinations, by size, from the val labels."""
    data = np.load(dump_path, allow_pickle=False)
    target = data["tools_target"].astype(np.int64)
    classes = [str(c) for c in data["tool_classes"]]
    counts = {}
    sizes = []
    for row in target:
        picked = tuple(sorted(np.nonzero(row)[0].tolist()))
        sizes.append(len(picked))
        if picked:
            counts[picked] = counts.get(picked, 0) + 1
    sizes = np.array(sizes)

    by_size = {}
    for size in range(1, max_size + 1):
        ranked = sorted(((n, s) for s, n in counts.items() if len(s) == size),
                        reverse=True)
        by_size[size] = [([classes[i] for i in combo], n)
                         for n, combo in ranked[:per_size]]
    distribution = {int(s): float((sizes == s).mean())
                    for s in range(0, max_size + 1)}
    distribution["over"] = float((sizes > max_size).mean())
    return by_size, distribution, classes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out")
    parser.add_argument("--dump", default=DUMP)
    parser.add_argument("--max-size", type=int, default=4)
    parser.add_argument("--per-size", type=int, default=3,
                        help="how many real combinations to average per size")
    args = parser.parse_args(argv)

    from surgvu.scoring import Scorer
    scorer = Scorer()

    def score(candidate, references):
        return scorer.score_one(candidate, references)["bertscore_f1"]

    by_size, distribution, classes = observed_sets(
        args.dump, args.max_size, args.per_size)

    print("windows by number of installed tools (validation labels):")
    for size in range(0, args.max_size + 1):
        print("  %d tools  %6.2f%%" % (size, 100 * distribution.get(size, 0.0)))
    print("  >%d       %6.2f%%\n" % (args.max_size, distribution["over"]))

    rows = []
    for size in range(1, args.max_size + 1):
        for combo, n in by_size.get(size, []):
            names = [display_name(c) for c in combo]
            refs = build_refs(names)
            entry = {"size": size, "classes": list(combo), "names": names,
                     "windows": int(n), "prefix": {}, "single_mean": None,
                     "over_listed": None}
            for k in range(1, size + 1):
                entry["prefix"][k] = score(join_names(names[:k]), refs)
            entry["single_mean"] = float(np.mean(
                [score(name, refs) for name in names]))
            wrong = next(display_name(c) for c in TOOL_CLASSES
                         if c not in combo)
            entry["over_listed"] = score(join_names(names + [wrong]), refs)
            rows.append(entry)
            print("m=%d  %-52s  n=%5d" % (size, join_names(names)[:52], n))
            print("      " + "  ".join("k=%d %.4f" % (k, v)
                                       for k, v in entry["prefix"].items())
                  + "   single(mean) %.4f   +1 wrong %.4f"
                  % (entry["single_mean"], entry["over_listed"]))

    # Payoff table: mean over the real combinations at each size.
    print("\n%-6s %8s %8s %8s %8s %8s %10s"
          % ("m", "k=1", "k=2", "k=3", "k=4", "k=m", "over-list"))
    payoff = {}
    for size in range(1, args.max_size + 1):
        here = [r for r in rows if r["size"] == size]
        if not here:
            continue
        cells = {k: float(np.mean([r["prefix"][k] for r in here]))
                 for k in range(1, size + 1)}
        # k=1 uses the mean over WHICH name, not the first: with the truth a
        # set rather than a sequence, "the first one" is an ordering artefact.
        cells[1] = float(np.mean([r["single_mean"] for r in here]))
        payoff[size] = {
            "cells": cells,
            "over_listed": float(np.mean([r["over_listed"] for r in here])),
        }
        print("%-6d %8s %8s %8s %8s %8.4f %10.4f"
              % (size,
                 *["%.4f" % cells[k] if k in cells else "-"
                   for k in (1, 2, 3, 4)],
                 cells[size], payoff[size]["over_listed"]))

    # Expected value of the two policies over the real distribution of m,
    # renormalised over the sizes measured (windows with 0 tools are not
    # plural-identity questions and windows above max_size are not measured).
    weights = {s: distribution.get(s, 0.0) for s in payoff}
    total = sum(weights.values())
    if total <= 0:
        raise SystemExit("no measured sizes carry any windows")
    weights = {s: w / total for s, w in weights.items()}

    one = sum(weights[s] * payoff[s]["cells"][1] for s in payoff)
    every = sum(weights[s] * payoff[s]["cells"][s] for s in payoff)
    print("\ncoverage: %.1f%% of tool-bearing windows fall in sizes 1..%d"
          % (100 * total, args.max_size))
    print("EXPECTED SCORE")
    print("  emit ONE name                 %.4f   (what the router does now)"
          % one)
    print("  emit ALL installed names      %.4f   (perfect perception, an "
          "upper bound)" % every)
    print("  delta                         %+.4f" % (every - one))

    print()
    if every <= one:
        print("VERDICT: listing loses even with the true set handed over free. "
              "The extra names cost more than they earn, exactly as the hedge "
              "study found for two-name answers. _answer_tool_identity keeps "
              "its single noun, and the plural phrasing of the question is "
              "not evidence that the answer should be plural.")
    else:
        print("VERDICT: listing wins by %+.4f WITH PERFECT PERCEPTION. That is "
              "an upper bound, not a result: perception is at 0.78 macro-F1 on "
              "tools, so some listed sets would carry a wrong name, and the "
              "over-list row (%.4f against %.4f for the true set at m=2) "
              "prices that error. Discount before wiring anything."
              % (every - one,
                 payoff.get(2, {}).get("over_listed", float("nan")),
                 payoff.get(2, {}).get("cells", {}).get(2, float("nan"))))

    if args.out:
        Path(args.out).write_text(json.dumps({
            "dump": args.dump,
            "size_distribution": distribution,
            "rows": [dict(r, prefix={str(k): v for k, v in r["prefix"].items()})
                     for r in rows],
            "payoff": {str(s): {"cells": {str(k): v for k, v in
                                          payoff[s]["cells"].items()},
                                "over_listed": payoff[s]["over_listed"]}
                       for s in payoff},
            "expected_one": one, "expected_all": every,
            "delta": every - one,
        }, indent=2), encoding="utf-8")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
