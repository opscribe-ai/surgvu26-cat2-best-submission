"""When the tool model is UNSURE, is it better to name two instruments or one?

THE FAILURE THIS COMES FROM. case124 asks "What type of forceps is mentioned?".
The router names the argmax -- Bipolar Forceps, at 0.972, against cadiere at
0.167 -- and the gold is Cadiere Forceps. It scores 0.2402, the worst single
number in the whole 11-case sample, and it is worse than the generic sentence
would have been. The router had no way to express "probably bipolar, possibly
cadiere", because `_answer_tool_identity` returns exactly one name.

THE QUESTION IS AN ARITHMETIC ONE, NOT A MATTER OF TASTE. Hedging trades a
certain loss on the cases we would have got right for a partial recovery on
the ones we would have got wrong, and whether that trade pays depends on two
quantities that can both be MEASURED:

  A. How often is the argmax right, as a function of how close the race was?
     Measured from the validation dump -- real probabilities, real labels.
  B. What does each answer FORM score under the official metric? Measured with
     roberta-large against the organizers' own reference templates.

Multiply and the decision falls out: hedge exactly where the expected score of
the hedge exceeds the expected score of the single name, and nowhere else.

PART A -- SCOPE. Only windows where EXACTLY ONE class of the family is present
count. A window holding two forceps does not have a single right answer to
"what type", so scoring an identity rule on it would measure the question's
ambiguity rather than the model.

PART B -- WHAT IS SIMULATED AND WHAT IS REAL. The five reference templates are
case124's, verbatim, with the instrument noun substituted. The templates are
the organizers'; the substitution is ours. So this measures how the METRIC
treats these answer shapes -- which is a property of roberta-large and is
exactly what we need -- and it does NOT claim to be eleven more graded cases.
Averaged over every ordered pair in the family so no conclusion rests on the
one pair that happens to have been in the sample.

WHAT WOULD MAKE THE ANSWER "DO NOT HEDGE". If a two-name answer scores near
the wrong-single-name floor, the hedge buys nothing and only costs -- an
embedding metric is under no obligation to reward a partially-correct list.
That result is a perfectly good outcome and the script prints it as plainly as
the other one.
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

#: case124's five references, noun removed. Order preserved: reference[0] is
#: the bare token, and the metric takes the MAX over all five, so the bare form
#: dominates whenever the candidate is itself bare.
TEMPLATES = (
    "{x}",
    "The type of forceps mentioned is {x}.",
    "{x} are the type mentioned.",
    "The forceps type is {x}.",
    "{x} is the specific type referenced.",
)

#: The family the sample question actually asked about. Identity questions are
#: routed WITHIN a family -- the router already refuses to answer "what type of
#: forceps" with a stapler -- so the realistic confusion set is this one.
FAMILY = ("bipolar forceps", "cadiere forceps", "force bipolar",
          "prograsp forceps")

MARGIN_EDGES = (0.0, 0.05, 0.10, 0.20, 0.40, 0.70, 1.01)


def accuracy_by_margin(dump_path, family, edges):
    """Part A. Per margin band: n, top-1 accuracy, and P(gold in top-2)."""
    data = np.load(dump_path, allow_pickle=False)
    # Mean over frames is the shipped aggregator, so the margins here are the
    # margins the router would actually see.
    probs = data["tools_id"].mean(axis=1)
    target = data["tools_target"]

    index = [list(TOOL_CLASSES).index(c) for c in family]
    fam_p, fam_t = probs[:, index], target[:, index]

    well_posed = fam_t.sum(axis=1) == 1
    fam_p, fam_t = fam_p[well_posed], fam_t[well_posed]
    gold = fam_t.argmax(axis=1)
    ranked = np.argsort(-fam_p, axis=1)
    ordered = np.sort(fam_p, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]

    bands = []
    for lo, hi in zip(edges, edges[1:]):
        mask = (margin >= lo) & (margin < hi)
        if not mask.any():
            continue
        bands.append({
            "lo": float(lo), "hi": float(hi), "n": int(mask.sum()),
            "top1": float((ranked[mask][:, 0] == gold[mask]).mean()),
            "top2": float(np.mean([g in r[:2]
                                   for g, r in zip(gold[mask], ranked[mask])])),
        })
    overall = {
        "n": int(len(gold)),
        "top1": float((ranked[:, 0] == gold).mean()),
        "top2": float(np.mean([g in r[:2] for g, r in zip(gold, ranked)])),
    }
    return overall, bands


def form_scores(scorer, family):
    """Part B. Mean official-metric score of each answer form, over all pairs.

    Five forms, and the two hedge ORDERINGS are kept apart on purpose: the
    metric is not symmetric in a conjunction, and if naming the likelier tool
    first is worth anything the router should know it.
    """
    names = [display_name(c) for c in family]
    buckets = {k: [] for k in ("single_hit", "single_miss", "hedge_first",
                               "hedge_second", "hedge_miss")}

    for i, gold in enumerate(names):
        references = [t.format(x=gold) for t in TEMPLATES]
        others = [n for j, n in enumerate(names) if j != i]

        def score(candidate):
            return scorer.score_one(candidate, references)["bertscore_f1"]

        buckets["single_hit"].append(score(gold))
        for other in others:
            buckets["single_miss"].append(score(other))
            # gold named first, then the runner-up, and the reverse.
            buckets["hedge_first"].append(score("%s and %s" % (gold, other)))
            buckets["hedge_second"].append(score("%s and %s" % (other, gold)))
        # A hedge that misses entirely: two wrong names. This is what the
        # low-margin band costs when even the top TWO do not contain the truth.
        buckets["hedge_miss"].append(score("%s and %s" % (others[0], others[1])))

    return {k: float(np.mean(v)) for k, v in buckets.items()}, \
           {k: int(len(v)) for k, v in buckets.items()}


def expected_scores(band, forms):
    """Expected metric score of each policy in one margin band.

    single: right with probability top1, otherwise a wrong specific noun.
    hedge : the truth is in the pair with probability top2. When it is, the
            hedge is scored in the order the router would emit -- argmax
            first -- so `hedge_second` is the operative number whenever the
            argmax is the WRONG one, which is precisely the case the hedge
            exists for. Split accordingly rather than using one average.
    """
    top1, top2 = band["top1"], band["top2"]
    single = top1 * 1.0 + (1.0 - top1) * forms["single_miss"]
    # Truth is the argmax (top1) -> "gold and other"     = hedge_first
    # Truth is the runner-up (top2 - top1) -> "other and gold" = hedge_second
    # Truth is in neither (1 - top2) -> two wrong names   = hedge_miss
    hedge = (top1 * forms["hedge_first"]
             + (top2 - top1) * forms["hedge_second"]
             + (1.0 - top2) * forms["hedge_miss"])
    return single, hedge


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump", default=DUMP)
    parser.add_argument("--out")
    parser.add_argument("--family", nargs="+", default=list(FAMILY))
    args = parser.parse_args(argv)

    unknown = [c for c in args.family if c not in TOOL_CLASSES]
    if unknown:
        raise SystemExit("not taxonomy classes: %s" % (unknown,))

    print("PART A -- how often is the argmax right? (validation dump)")
    print("dump: %s" % args.dump)
    overall, bands = accuracy_by_margin(args.dump, args.family, MARGIN_EDGES)
    print("family: %s" % ", ".join(args.family))
    print("well-posed windows (exactly one of the family present): %d"
          % overall["n"])
    print("  top-1 %.4f   gold in top-2 %.4f\n"
          % (overall["top1"], overall["top2"]))
    print("%-14s %7s %8s %8s" % ("margin", "n", "top1", "top2"))
    for band in bands:
        print("%-14s %7d %8.4f %8.4f"
              % ("[%.2f,%.2f)" % (band["lo"], band["hi"]),
                 band["n"], band["top1"], band["top2"]))

    print("\nPART B -- what does each answer FORM score? (roberta-large)")
    from surgvu.scoring import Scorer
    scorer = Scorer()
    forms, counts = form_scores(scorer, args.family)
    print("references are case124's five templates with the noun substituted; "
          "the templates are the organizers', the substitution is ours.")
    print("%-14s %8s %6s" % ("form", "mean F1", "n"))
    for name in ("single_hit", "single_miss", "hedge_first", "hedge_second",
                 "hedge_miss"):
        print("%-14s %8.4f %6d" % (name, forms[name], counts[name]))

    print("\nPART C -- expected score per policy, per margin band")
    print("%-14s %7s %9s %9s %9s  %s"
          % ("margin", "n", "single", "hedge", "delta", "policy"))
    total_single = total_hedge = total_n = 0.0
    cut = None
    for band in bands:
        single, hedge = expected_scores(band, forms)
        better = "HEDGE" if hedge > single else "single"
        if hedge > single:
            cut = band["hi"] if cut is None else max(cut, band["hi"])
        print("%-14s %7d %9.4f %9.4f %+9.4f  %s"
              % ("[%.2f,%.2f)" % (band["lo"], band["hi"]), band["n"],
                 single, hedge, hedge - single, better))
        total_n += band["n"]
        total_single += band["n"] * single
        total_hedge += band["n"] * max(single, hedge)

    print("\nalways-single, over all well-posed windows   %.4f"
          % (total_single / total_n))
    print("hedge only where it wins                     %.4f"
          % (total_hedge / total_n))
    gain = total_hedge / total_n - total_single / total_n
    print("gain                                         %+.4f" % gain)

    print()
    if cut is None:
        print("VERDICT: hedging never wins. A two-name answer does not recover "
              "enough of the metric to pay for what it costs when the argmax "
              "was already right. Leave _answer_tool_identity alone.")
    else:
        print("VERDICT: hedge when the top-two margin is below %.2f. That is "
              "%d of %d well-posed windows (%.1f%%); the rest keep the single "
              "name." % (cut, sum(b["n"] for b in bands if b["hi"] <= cut),
                         int(total_n),
                         100.0 * sum(b["n"] for b in bands if b["hi"] <= cut)
                         / total_n))
        print("CAVEAT BEFORE WIRING: the gain above is per WELL-POSED WINDOW, "
              "not per graded question. It is an upper bound on what the "
              "router can win here, and it only applies to identity questions "
              "-- roughly one in ten of the sample.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "dump": args.dump, "family": args.family,
            "overall": overall, "bands": bands, "forms": forms,
            "form_counts": counts,
            "expected": [dict(band, single=expected_scores(band, forms)[0],
                              hedge=expected_scores(band, forms)[1])
                         for band in bands],
            "always_single": total_single / total_n,
            "hedge_where_it_wins": total_hedge / total_n,
            "gain": gain, "margin_cut": cut,
        }, indent=2), encoding="utf-8")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
