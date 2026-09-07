"""How many instrument names should a plural answer carry, given OUR perception?

WHAT THE FIRST STUDY LEFT OPEN. `list_form_eval.py` priced the surface form
and found the single noun the router currently emits is worth 0.4771 against
1.0000 for the full list -- but that 1.0000 was scored with the TRUE set handed
over free, which no serving path has. It is an upper bound and was labelled as
one. The number that decides anything has to be computed with the set our tool
head actually predicts, wrong names and missing names included.

HOW THAT IS DONE HERE, IN TWO PIECES THAT MULTIPLY:

  1. A payoff table over (m, k, e): the true set holds m instruments, the
     answer names k of them correctly and e that are not there. Every cell is
     the official metric against references built from the true set, so the
     cost of a miss and the cost of an extra are measured separately rather
     than assumed symmetric. The first study only measured the k=m diagonal
     and one over-list cell.

  2. The empirical joint distribution of (m, k, e) over the validation split,
     produced by running the SHIPPED serving thresholds across the 2D tool
     dump. This is what our perception really does, not what a 0.78 macro-F1
     might be imagined to do.

Expected score of a policy is then the payoff table averaged over that
distribution. Four policies are compared: the single name the router emits
today, the top two and top three by confidence, and every class that clears
its threshold.

WHY THE ANSWER IS NOT OBVIOUS FROM THE FIRST STUDY. Under-naming and
over-naming are not priced the same. At m=1 an answer with one extra name
scores 0.5805, while at m=2 an answer missing one name scores 0.4982 -- so an
extra name costs LESS than a missing one, and the cheapest policy under
uncertainty may be to name more than we believe rather than fewer. That
asymmetry is exactly what a threshold-driven predicted set gets wrong in the
expensive direction, since thresholds tuned for macro-F1 are tuned to be
CAUTIOUS about the rare classes.

WHAT THIS STILL DOES NOT SETTLE, stated plainly: every reference here is
written by us in the organizers' style, and the whole result is conditional on
plural questions having plural golds. If the 2026 references answer "What
tools are used?" with one instrument name, the table inverts. The sample's one
identity question is singular ("What type of forceps is mentioned?"), so it is
evidence about neither.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from list_form_eval import build_refs, join_names, observed_sets  # noqa: E402
from surgvu.frames import sample_frame_indices                  # noqa: E402
from surgvu.router import display_name                          # noqa: E402

DUMP = "/staging/n/nkalthoff/surgvu26/v2/frame_probs_resnetlong_val.npz"
#: Two candidates because HTCondor FLATTENS transfer_input_files: submitted as
#: `extra=outputs/serving_thresholds.json` the file lands in the scratch root
#: with no directory, so a path that works locally is wrong on the node and
#: vice versa. Resolved rather than guessed.
THRESHOLD_CANDIDATES = ("outputs/serving_thresholds.json",
                        "serving_thresholds.json")
MAX_EXTRA = 2


def resolve_thresholds(explicit=None):
    for path in ([explicit] if explicit else THRESHOLD_CANDIDATES):
        if Path(path).exists():
            return path
    raise SystemExit(
        "no serving thresholds found; looked at %s. Submit with "
        "extra=outputs/serving_thresholds.json."
        % ", ".join(THRESHOLD_CANDIDATES))


def predicted_sets(dump_path, thresholds_path, frames):
    """(true set, ranked predicted classes, ranked all classes) per window."""
    data = np.load(dump_path, allow_pickle=False)
    classes = [str(c) for c in data["tool_classes"]]
    picks = sample_frame_indices(data["tools_id"].shape[1], frames)
    probs = data["tools_id"][:, picks, :].mean(axis=1)
    target = data["tools_target"].astype(np.int64)

    by_class = json.loads(Path(thresholds_path).read_text(encoding="utf-8"))
    cuts = np.array([by_class["serving_thresholds_by_class"][c]
                     for c in classes], dtype=np.float32)

    rows = []
    for index in range(len(probs)):
        true_set = {classes[i] for i in np.nonzero(target[index])[0]}
        order = list(np.argsort(-probs[index]))
        ranked_all = [classes[i] for i in order]
        ranked_pred = [classes[i] for i in order if probs[index][i] >= cuts[i]]
        rows.append((true_set, ranked_pred, ranked_all))
    return rows, classes


def outcome(answer_classes, true_set):
    """(k correct, e extra), with e clamped -- see MAX_EXTRA."""
    k = sum(1 for c in answer_classes if c in true_set)
    e = min(len(answer_classes) - k, MAX_EXTRA)
    return k, e


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out")
    parser.add_argument("--dump", default=DUMP)
    parser.add_argument("--thresholds")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--max-size", type=int, default=4)
    parser.add_argument("--per-size", type=int, default=2,
                        help="real combinations averaged per size; each one "
                             "costs (m+1)*(MAX_EXTRA+1)-1 metric calls")
    args = parser.parse_args(argv)

    from surgvu.scoring import Scorer
    scorer = Scorer()

    def score(candidate, references):
        return scorer.score_one(candidate, references)["bertscore_f1"]

    by_size, distribution, classes = observed_sets(
        args.dump, args.max_size, args.per_size)

    # ---- 1. the payoff table --------------------------------------------
    payoff = {}
    print("PAYOFF: true set of m, answer names k of them plus e others\n")
    for size in range(1, args.max_size + 1):
        cells = {}
        for combo, _ in by_size.get(size, []):
            names = [display_name(c) for c in combo]
            refs = build_refs(names)
            # The extras are the most common NON-members, not random ones: a
            # wrong name our model would actually emit is a frequent class,
            # and a rare one would be an easier mistake than we really make.
            extras = [display_name(c) for c in classes if c not in combo][:MAX_EXTRA]
            for k in range(0, size + 1):
                for e in range(0, MAX_EXTRA + 1):
                    if k + e == 0:
                        continue        # the empty string crashes the scorer
                    answer = join_names(names[:k] + extras[:e])
                    cells.setdefault((k, e), []).append(score(answer, refs))
        payoff[size] = {key: float(np.mean(vals)) for key, vals in cells.items()}
        print("m=%d" % size)
        print("      " + "  ".join("e=%d" % e for e in range(MAX_EXTRA + 1)))
        for k in range(0, size + 1):
            row = ["%.4f" % payoff[size][(k, e)] if (k, e) in payoff[size]
                   else "  --  " for e in range(MAX_EXTRA + 1)]
            print("  k=%d %s" % (k, "  ".join(row)))
        print()

    # ---- 2. what our perception actually produces ------------------------
    thresholds = resolve_thresholds(args.thresholds)
    print("thresholds: %s" % thresholds)
    rows, _ = predicted_sets(args.dump, thresholds, args.frames)
    usable = [(t, p, a) for t, p, a in rows if 1 <= len(t) <= args.max_size]
    print("windows: %d total, %d with 1..%d installed tools (%.1f%%)"
          % (len(rows), len(usable), args.max_size,
             100.0 * len(usable) / max(1, len(rows))))

    policies = {
        "one name": lambda pred, allc: (pred or allc)[:1],
        "top two": lambda pred, allc: (pred or allc)[:2],
        "top three": lambda pred, allc: (pred or allc)[:3],
        "all above threshold": lambda pred, allc: pred or allc[:1],
    }

    results = {}
    print("\n%-22s %9s %9s %9s %9s"
          % ("policy", "score", "names", "correct", "extra"))
    for name, choose in policies.items():
        total, counts, sizes, corrects, extras = 0.0, 0, [], [], []
        for true_set, pred, ranked_all in usable:
            answer = choose(pred, ranked_all)
            k, e = outcome(answer, true_set)
            if k + e == 0:
                # Cannot happen: every policy emits at least one name. Guarded
                # because a silent empty answer would score as if it were free.
                raise SystemExit("policy %r emitted nothing" % name)
            total += payoff[len(true_set)][(k, e)]
            sizes.append(len(answer))
            corrects.append(k)
            extras.append(e)
            counts += 1
        results[name] = {
            "expected_score": total / counts,
            "mean_names": float(np.mean(sizes)),
            "mean_correct": float(np.mean(corrects)),
            "mean_extra": float(np.mean(extras)),
        }
        print("%-22s %9.4f %9.2f %9.2f %9.2f"
              % (name, results[name]["expected_score"],
                 results[name]["mean_names"], results[name]["mean_correct"],
                 results[name]["mean_extra"]))

    current = results["one name"]["expected_score"]
    best = max(results, key=lambda k: results[k]["expected_score"])
    delta = results[best]["expected_score"] - current
    print("\nceiling for reference: naming the true set exactly scores 1.0000, "
          "so the headroom the router is leaving on plural questions is "
          "%.4f." % (1.0 - current))
    print()
    if best == "one name":
        print("VERDICT: the single name the router already emits is the best "
              "policy even against plural references. Listing what perception "
              "predicts costs more in wrong names than it earns in right ones.")
    else:
        print("VERDICT: %r beats the router's single name by %+.4f, WITH REAL "
              "PERCEPTION and its wrong names included. That is not an upper "
              "bound -- it is what the change would be worth if plural "
              "questions have plural golds, which is the one assumption left "
              "standing and cannot be checked against the public sample."
              % (best, delta))

    if args.out:
        Path(args.out).write_text(json.dumps({
            "dump": args.dump, "thresholds": thresholds,
            "frames": args.frames, "max_extra": MAX_EXTRA,
            "size_distribution": distribution,
            "payoff": {str(m): {"%d,%d" % key: value
                                for key, value in cells.items()}
                       for m, cells in payoff.items()},
            "policies": results,
            "best": best, "delta_over_one_name": delta,
        }, indent=2), encoding="utf-8")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
