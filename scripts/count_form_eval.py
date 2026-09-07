"""Counting questions: should the router emit "Three" or "3"?

THE GAP THIS FILLS. `_answer_count` emits an English word, and that choice has
never been measured -- it was inherited from the observation that every gold
reference in the public sample leads with a bare token, without ever asking
which bare token a COUNTING question's reference would be. None of the eleven
sample questions is a counting question, so there is no gold to look at. The
router is making a surface-form bet with no evidence behind it.

The bet is not small. Under a metric that gives an exact match 1.0000 and a
near-miss much less, "Three" against a gold of "3" is not a rounding error --
it is the difference between a full mark and whatever roberta-large thinks a
number word shares with a numeral.

THE SHAPE OF THE ANSWER. We cannot learn the organizers' format from data we
do not have, so the question is not "which is right" but "which is the more
ROBUST bet", and that is decidable:

    emit word    p * score(word | numeral gold) + (1-p) * 1.0
    emit numeral p * 1.0 + (1-p) * score(numeral | word gold)

with p = P(the gold is a numeral). Whichever form has the higher floor is the
one to emit when p is unknown, and the crossover value of p says how strong a
belief about the format would have to be to justify the other choice. If the
two cross near p=0.5 the choice barely matters and the honest report says so.

BOTH DIRECTIONS ARE MEASURED, NOT ONE AND ASSUMED SYMMETRIC. BERTScore is not
symmetric in candidate and reference -- it is an F1 over a soft alignment, and
a numeral and a number word have different subword tokenisations. Assuming
score(a|b) == score(b|a) here would be assuming away the entire question.

THE SENTENCE FORMS ARE CARRIED TOO. The metric takes the MAX over references,
so a reference list that leads with a bare token but also contains "Three
instruments are visible." changes the calculus: a candidate that misses the
bare token may still match a sentence. Sentence templates mirror the sample's
own reference style, with the count substituted.

SCOPE. This measures how the METRIC treats these surface forms, which is a
property of roberta-large and is exactly what the router needs to know. It is
not a claim about what the 2026 references contain -- that is the unknown p.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.router import COUNT_WORDS                           # noqa: E402

#: The counts a clip can plausibly show. Up to four arms are in play and
#: 97.94% of training windows hold at most three distinct tool classes, so
#: this is the range the router will ever emit -- measuring 0 or 9 would be
#: averaging in cases that never occur.
COUNTS = (1, 2, 3, 4)

#: Reference templates in the sample's own style, count substituted. The bare
#: form is index 0 for the same reason the organizers' lists lead with one.
WORD_TEMPLATES = (
    "{x}",
    "There are {x} instruments visible.",
    "{x} instruments are being used.",
)
NUMERAL_TEMPLATES = WORD_TEMPLATES


def build(templates, token):
    return [t.format(x=token) for t in templates]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out")
    parser.add_argument("--bare-only", action="store_true",
                        help="score against reference[0] alone, i.e. a test "
                             "set whose references are bare tokens only")
    args = parser.parse_args(argv)

    from surgvu.scoring import Scorer
    scorer = Scorer()

    def score(candidate, references):
        return scorer.score_one(candidate, references)["bertscore_f1"]

    rows = []
    for count in COUNTS:
        word = COUNT_WORDS[count]
        numeral = str(count)
        word_refs = build(WORD_TEMPLATES, word)
        numeral_refs = build(NUMERAL_TEMPLATES, numeral)
        if args.bare_only:
            word_refs, numeral_refs = word_refs[:1], numeral_refs[:1]
        # WRONG-VALUE arms. The surface-form question above turned out to be a
        # tie, which makes the far more consequential question the one about
        # getting the NUMBER wrong -- `_answer_count` guesses a modal count
        # when it has no evidence, and how bad that guess is has never been
        # measured. Off by one and off by two, in the same form as the gold,
        # so this isolates the value error from the form error.
        off_one = COUNT_WORDS[count + 1] if count + 1 < len(COUNT_WORDS) else None
        off_two = COUNT_WORDS[count + 2] if count + 2 < len(COUNT_WORDS) else None
        rows.append({
            "count": count, "word": word, "numeral": numeral,
            # The matched cases are carried rather than assumed to be 1.0:
            # they are only exactly 1.0 when the candidate equals a reference
            # verbatim, and printing them is the check that it does.
            "word_vs_word": score(word, word_refs),
            "numeral_vs_numeral": score(numeral, numeral_refs),
            "word_vs_numeral": score(word, numeral_refs),
            "numeral_vs_word": score(numeral, word_refs),
            "off_by_one": score(off_one, word_refs) if off_one else None,
            "off_by_two": score(off_two, word_refs) if off_two else None,
        })

    print("references: %s\n"
          % ("bare token only" if args.bare_only
             else "bare token + %d sentence forms" % (len(WORD_TEMPLATES) - 1)))
    print("%-6s %-7s %10s %10s %10s %10s"
          % ("count", "word", "w|w", "n|n", "w|n", "n|w"))
    for row in rows:
        print("%-6d %-7s %10.4f %10.4f %10.4f %10.4f"
              % (row["count"], row["word"], row["word_vs_word"],
                 row["numeral_vs_numeral"], row["word_vs_numeral"],
                 row["numeral_vs_word"]))

    mean = {k: float(np.mean([r[k] for r in rows]))
            for k in ("word_vs_word", "numeral_vs_numeral",
                      "word_vs_numeral", "numeral_vs_word")}
    print("\nmeans  match(word) %.4f  match(numeral) %.4f  "
          "word|numeral %.4f  numeral|word %.4f"
          % (mean["word_vs_word"], mean["numeral_vs_numeral"],
             mean["word_vs_numeral"], mean["numeral_vs_word"]))

    print("\n%-6s %10s %10s  %s" % ("p(num)", "emit word", "emit numeral",
                                    "better"))
    crossover = None
    previous = None
    for p in [i / 10.0 for i in range(11)]:
        as_word = p * mean["word_vs_numeral"] + (1 - p) * mean["word_vs_word"]
        as_num = p * mean["numeral_vs_numeral"] + (1 - p) * mean["numeral_vs_word"]
        better = "numeral" if as_num > as_word else "word"
        if previous is not None and better != previous and crossover is None:
            crossover = p
        previous = better
        print("%-6.1f %10.4f %10.4f  %s" % (p, as_word, as_num, better))

    # Worst case over the unknown p -- the floor each choice guarantees.
    word_floor = min(mean["word_vs_word"], mean["word_vs_numeral"])
    num_floor = min(mean["numeral_vs_numeral"], mean["numeral_vs_word"])
    print("\nguaranteed floor: word %.4f, numeral %.4f" % (word_floor, num_floor))

    print()
    if crossover is None:
        winner = "numeral" if num_floor > word_floor else "word"
        print("VERDICT: %s wins at every p. The choice does not depend on a "
              "belief about the organizers' format." % winner.upper())
    elif abs(word_floor - num_floor) < 1e-4:
        # Say TIE rather than declaring a winner by a rounding artefact. F1
        # over a one-token candidate against a one-token reference is symmetric
        # -- precision and recall simply swap -- so an exact tie here is the
        # expected result, not a coincidence worth breaking.
        print("VERDICT: EXACT TIE. The two forms score identically against "
              "each other's references (%.4f both ways), so they cross at "
              "p=%.1f and neither has a better floor. There is no evidence "
              "here for changing _answer_count, and no evidence for keeping "
              "it either -- the choice is simply free. Leave it alone."
              % (word_floor, crossover))
    else:
        print("VERDICT: the forms cross at p=%.1f, and the better floor is %s "
              "(%.4f vs %.4f) -- the form to prefer when p is genuinely "
              "unknown."
              % (crossover, "numeral" if num_floor > word_floor else "word",
                 max(num_floor, word_floor), min(num_floor, word_floor)))

    have_off = [r for r in rows if r.get("off_by_one") is not None]
    if have_off:
        one = float(np.mean([r["off_by_one"] for r in have_off]))
        two = float(np.mean([r["off_by_two"] for r in have_off
                             if r.get("off_by_two") is not None]))
        print("\nWRONG VALUE, same form as the gold:")
        print("  exact          1.0000")
        print("  off by one     %.4f" % one)
        print("  off by two     %.4f" % two)
        print("  wrong FORM     %.4f   (right number, other notation)"
              % mean["word_vs_numeral"])
        if one > mean["word_vs_numeral"]:
            print("Getting the NUMBER wrong by one costs LESS than getting the "
                  "notation wrong. Counting accuracy is worth less than it "
                  "looks, and the modal-count fallback is close to free.")
        else:
            print("Getting the number wrong costs more than getting the "
                  "notation wrong, so counting accuracy is the thing to spend "
                  "effort on, not the surface form.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "bare_only": args.bare_only, "rows": rows, "means": mean,
            "word_floor": word_floor, "numeral_floor": num_floor,
            "crossover_p": crossover,
        }, indent=2), encoding="utf-8")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
