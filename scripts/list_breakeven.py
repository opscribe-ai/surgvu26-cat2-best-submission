"""How sure must we be that a plural question has a plural gold?

THE ONE ASSUMPTION LEFT. `list_policy_eval.py` measured that naming the top
three instruments beats naming one by +0.4964 with real perception -- against
references that LIST the installed set. If the organizers answer "What tools
are used in this step?" with a single instrument name instead, the extra names
are pure cost and the change loses. Nothing in the public sample settles it:
its one identity question is singular.

So the honest form of the result is not a number but a threshold. Let q be
P(a plural question has a plural gold). Listing wins when

    q * gain > (1 - q) * loss     i.e.    q > loss / (gain + loss)

and this script computes both sides from tables already measured rather than
from a fresh guess. No scoring runs here -- it is arithmetic over
list_policy_results.json plus the perception dump.

PRICING THE SINGULAR-GOLD CASE. If the gold names one instrument, it names one
of the m actually installed; which one is unknown, so each member is treated as
equally likely to be the one written down. Then for a policy emitting a set S:

    P(the gold's instrument is in S) = E[ |S & T| / m ]

and the payoff is read off the m=1 row of the measured table, where k=1 means
we named it and e counts the extras we added. That row is the right one
BECAUSE the reference holds a single name; the true set's size only enters
through the probability above.

WHAT THIS IS NOT. It does not estimate q. q is a fact about the 2026
references that we cannot see, and the point of the threshold is that it makes
the decision explicit: below it, keep the single noun.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from list_policy_eval import predicted_sets, resolve_thresholds  # noqa: E402

RESULTS = "list_policy_results.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=RESULTS)
    parser.add_argument("--thresholds")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--cap", type=int, default=3,
                        help="the listing policy: top N above threshold")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    single_ref = {key: value for key, value in payload["payoff"]["1"].items()}
    policies = payload["policies"]

    def cell(k, e):
        return single_ref["%d,%d" % (k, e)]

    rows, _ = predicted_sets(payload["dump"],
                             resolve_thresholds(args.thresholds), args.frames)
    usable = [(t, p, a) for t, p, a in rows if 1 <= len(t) <= 4]

    hit_one, hit_many = [], []
    # The ADVERSARIAL alternative to the uniform assumption: a singular gold
    # naming whichever instrument is most salient, and our ranking agreeing
    # with salience. Then "does our answer contain the gold's instrument"
    # collapses to "is our top name installed at all", which is much easier
    # for the one-name policy and barely easier for the three-name one -- so
    # this is the assumption under which listing looks WORST, and it belongs
    # in the report next to the uniform one rather than after it.
    top1_in, any3_in = [], []
    for true_set, pred, ranked_all in usable:
        one = (pred or ranked_all)[:1]
        many = (pred or ranked_all)[:args.cap]
        # Each installed instrument is equally likely to be the one a singular
        # gold names, so the chance our answer contains it is the fraction of
        # the true set our answer covers.
        hit_one.append(len(set(one) & true_set) / len(true_set))
        hit_many.append(len(set(many) & true_set) / len(true_set))
        top1_in.append(1.0 if set(one) & true_set else 0.0)
        any3_in.append(1.0 if set(many) & true_set else 0.0)
    hit_one = float(np.mean(hit_one))
    hit_many = float(np.mean(hit_many))
    top1_in = float(np.mean(top1_in))
    any3_in = float(np.mean(any3_in))

    # Extras, as the m=1 row indexes them: one name that is not the gold's is
    # (k=0, e=1); three names one of which is the gold's is (k=1, e=2).
    singular_one = hit_one * cell(1, 0) + (1 - hit_one) * cell(0, 1)
    singular_many = hit_many * cell(1, args.cap - 1) \
        + (1 - hit_many) * cell(0, min(args.cap, 2))

    plural_one = policies["one name"]["expected_score"]
    plural_many = policies["top three"]["expected_score"]

    gain = plural_many - plural_one
    loss = singular_one - singular_many
    print("PLURAL GOLD (references list the installed set)")
    print("  one name        %.4f" % plural_one)
    print("  top %d           %.4f" % (args.cap, plural_many))
    print("  gain            %+.4f" % gain)
    print("\nSINGULAR GOLD (references name one instrument)")
    print("  P(our 1 name covers the named one)   %.4f" % hit_one)
    print("  P(our %d names cover the named one)   %.4f" % (args.cap, hit_many))
    print("  one name        %.4f" % singular_one)
    print("  top %d           %.4f" % (args.cap, singular_many))
    print("  loss            %+.4f" % loss)

    if gain <= 0:
        print("\nVERDICT: listing does not even win under plural golds. "
              "Nothing to decide.")
        breakeven = None
    else:
        breakeven = loss / (gain + loss)
        print("\nBREAK-EVEN q = %.3f" % breakeven)
        print("Listing the top %d wins whenever more than %.0f%% of plural "
              "questions have plural golds." % (args.cap, 100 * breakeven))
        for q in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            print("  q=%.1f   one name %.4f   top %d %.4f   %s"
                  % (q, q * plural_one + (1 - q) * singular_one, args.cap,
                     q * plural_many + (1 - q) * singular_many,
                     "list" if (q * plural_many + (1 - q) * singular_many)
                     > (q * plural_one + (1 - q) * singular_one) else "one"))

    # Sensitivity: the same arithmetic under the salience-aligned assumption.
    adv_one = top1_in * cell(1, 0) + (1 - top1_in) * cell(0, 1)
    adv_many = any3_in * cell(1, args.cap - 1) \
        + (1 - any3_in) * cell(0, min(args.cap, 2))
    adv_loss = adv_one - adv_many
    adv_breakeven = (adv_loss / (gain + adv_loss)) if gain + adv_loss > 0 else None
    print("\nSENSITIVITY -- singular gold that names the MOST SALIENT tool, "
          "with our ranking agreeing:")
    print("  P(our top name installed)            %.4f" % top1_in)
    print("  P(any of our %d installed)            %.4f" % (args.cap, any3_in))
    print("  one name        %.4f" % adv_one)
    print("  top %d           %.4f" % (args.cap, adv_many))
    print("  loss            %+.4f" % adv_loss)
    if adv_breakeven is None:
        print("  break-even q: none -- listing still wins at every q.")
    else:
        print("  BREAK-EVEN q = %.3f under this assumption." % adv_breakeven)
        print("  So the decision needs q above %.0f%% in the worst case and "
              "nothing at all in the uniform one. Report the range, not one "
              "end of it." % (100 * adv_breakeven))

    if args.out:
        Path(args.out).write_text(json.dumps({
            "results": args.results, "cap": args.cap,
            "hit_one": hit_one, "hit_many": hit_many,
            "plural_one": plural_one, "plural_many": plural_many,
            "singular_one": singular_one, "singular_many": singular_many,
            "gain": gain, "loss": loss, "breakeven_q": breakeven,
            "salience_top1_installed": top1_in,
            "salience_any_installed": any3_in,
            "salience_one": adv_one, "salience_many": adv_many,
            "salience_breakeven_q": adv_breakeven,
        }, indent=2), encoding="utf-8")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
