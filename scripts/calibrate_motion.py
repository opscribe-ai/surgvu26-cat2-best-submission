"""Does frame-difference activity carry ANY signal? Answer before building on it.

THE PREMISE THIS TESTS. `_answer_cutting` returns Yes when a cutting tool is
credible -- when scissors are visible -- with no evidence about whether cutting
is happening. The proposal is to gate that on motion. This script asks whether
the motion statistic is worth gating anything on, and it is designed to be able
to say NO cheaply, before a serving change or a GPU hour is spent.

WHAT THE LABELS LET US ASK, and what they do not. There is no "cutting"
label in this corpus. TASK_CLASSES are procedural: suturing, range of motion,
retraction and collision avoidance, and so on. So the direct question -- "does
activity predict cutting" -- is unanswerable with the data we have, and saying
so is part of the result rather than a reason not to run it.

What IS answerable, in descending order of how much it would tell us:

  1. Does activity vary at all? If every window reads the same, there is no
     signal and everything downstream is noise with a threshold on it.

  2. Does it separate `suturing` from `range of motion`? Suturing is
     sustained bimanual work; range-of-motion exercises are gross movement.
     Both are motion-heavy, so a separation here is about KIND, not presence.

  3. Does it separate the motion-heavy classes from `other`? This is the
     closest available proxy for active-versus-idle, and it is a weak one.

  4. HOW MANY ANSWERS WOULD CHANGE. Count the windows where a cutting tool is
     credible but activity is in the bottom decile -- those are exactly the
     windows where the proposed rule flips Yes to No. If that count is zero
     the rule is inert and not worth shipping whatever its logic; if it is
     most of the corpus the rule is reckless. This number decides the design
     even though no label validates the flips individually.

REPORTED AS A DISTRIBUTION, NOT A VERDICT. Separability is given as AUC with
the class counts beside it, because an AUC over forty windows is not evidence
and should not be able to look like it.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.dataset import shard_paths_for_split                 # noqa: E402
from surgvu.extract import read_shard                            # noqa: E402
from surgvu.motion import macro_activity, micro_activity         # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MULTI16 = "/staging/n/nkalthoff/surgvu26/shards_multi16"

#: Instruments that cut, as the router defines them. Imported rather than
#: restated so this cannot drift from the rule it is evaluating.
from surgvu.router import CUTTING_TOOLS                          # noqa: E402


def auc(positive, negative):
    """Rank AUC, the separability of two samples. 0.5 is no separation.

    Computed from ranks rather than by thresholding, so it needs no cutoff
    and reports the whole ordering at once.
    """
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    joined = np.concatenate([positive, negative])
    order = joined.argsort().argsort().astype(np.float64) + 1.0
    rank_sum = order[:positive.size].sum()
    return float((rank_sum - positive.size * (positive.size + 1) / 2.0)
                 / (positive.size * negative.size))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards", default=MULTI16)
    parser.add_argument("--splits", default=str(REPO / "config" / "splits_v2.json"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--frames-per-burst", type=int, default=3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--windows-per-shard", type=int, default=25,
                        help="sample this many windows per shard. Decoding "
                             "every 48-frame window of the whole pool is "
                             "~1M JPEG decodes; a sample answers all four "
                             "questions and finishes in minutes.")
    args = parser.parse_args(argv)

    shards = shard_paths_for_split(args.shards, args.splits, args.split)
    if args.max_shards:
        shards = shards[:args.max_shards]
    print("%d shards in split %r" % (len(shards), args.split), flush=True)

    micro_all, macro_all, tasks, cutting_present = [], [], [], []
    started = time.time()
    for index, path in enumerate(shards, start=1):
        frames, rows = read_shard(path)
        step = max(1, len(rows) // max(1, args.windows_per_shard))
        for w in range(0, len(rows), step):
            stack = np.stack(list(frames[w]))
            if stack.shape[0] % args.frames_per_burst:
                print("  skip window %d of %s: %d frames"
                      % (w, Path(path).name, stack.shape[0]))
                continue
            micro = micro_activity(stack, args.frames_per_burst)
            macro = macro_activity(stack, args.frames_per_burst)
            micro_all.append(float(micro.mean()))
            macro_all.append(float(macro.mean()))
            tasks.append(str(rows[w].get("task", "")).strip().lower())
            tools = {str(t).strip().lower() for t in rows[w].get("tools", [])}
            cutting_present.append(bool(tools & {t.lower() for t in CUTTING_TOOLS}))
        if index % 10 == 0 or index == len(shards):
            print("  shard %d/%d  windows=%d  elapsed=%ds"
                  % (index, len(shards), len(micro_all),
                     time.time() - started), flush=True)

    micro_all = np.array(micro_all)
    macro_all = np.array(macro_all)
    tasks = np.array(tasks)
    cutting_present = np.array(cutting_present)

    report = {"split": args.split, "windows": int(micro_all.size),
              "shards": len(shards)}

    # 1. Does it vary at all?
    for name, values in (("micro", micro_all), ("macro", macro_all)):
        report[name] = {
            "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()),
            "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            # The one number that decides question 1: a statistic whose spread
            # is a rounding error on its mean is a constant with noise on it.
            "coefficient_of_variation": float(values.std() / (values.mean() or 1)),
        }

    # 2 and 3. Separability, with the counts that say whether to believe it.
    by_task = defaultdict(list)
    for task, value in zip(tasks, micro_all):
        by_task[task].append(value)
    report["per_task_micro"] = {
        task: {"n": len(values), "mean": float(np.mean(values)),
               "median": float(np.median(values))}
        for task, values in sorted(by_task.items())}

    pairs = [("suturing", "range of motion"), ("suturing", "other"),
             ("range of motion", "other")]
    report["separability_auc"] = {}
    for left, right in pairs:
        if left in by_task and right in by_task:
            report["separability_auc"]["%s_vs_%s" % (left, right)] = {
                "micro_auc": auc(by_task[left], by_task[right]),
                "n_left": len(by_task[left]), "n_right": len(by_task[right]),
            }

    # 4. How many answers would the proposed rule actually change?
    cut = np.percentile(micro_all, 10)
    would_flip = int((cutting_present & (micro_all <= cut)).sum())
    report["rule_impact"] = {
        "cutting_tool_credible": int(cutting_present.sum()),
        "of_those_in_bottom_decile_of_activity": would_flip,
        "bottom_decile_threshold": float(cut),
        "fraction_of_cutting_windows_that_would_flip": (
            float(would_flip / cutting_present.sum())
            if cutting_present.sum() else 0.0),
        "NOTE": ("No cutting label exists in this corpus, so this counts how "
                 "many answers the rule would CHANGE, not how many it would "
                 "get right. A count near zero means the rule is inert; a "
                 "count near all of them means it is reckless."),
    }

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
