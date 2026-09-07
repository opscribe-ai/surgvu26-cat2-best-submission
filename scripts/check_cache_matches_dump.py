"""Prove the feature cache is a cache OF the model that will consume it.

WHY THIS EXISTS. `cache_trunk_features.py` and `dump_temporal_probs.py` reach
the same numbers by different routes: the cache stores per-centre 2D logits
computed once, and the dump recomputes them from JPEG through the full
`ResidualTemporal` forward. If those two disagree, every result trained on the
cache is a result about a model nobody serves -- and the disagreement would be
invisible, because both produce plausible probability vectors over the right
classes.

That is not a hypothetical failure mode in this project. Every v4 conversion
arm was built from one checkpoint and compared against a number another
checkpoint produced; the mismatch was worth 0.0337 and survived six hours of
measurement because no one reproduced the reference through the new path
before trusting it. This is that reproduction, for the cache.

WHAT IS COMPARED. At beta=0 the residual model IS the 2D model, so the dump of
the alpha=0 baseline holds exactly `mean_over_bursts(softmax(centre logits))`.
The cache holds those centre logits. Applying the same aggregation to the
cache must give the same probabilities, window for window.

    dump   : decode -> trunk -> per-frame logits -> softmax -> mean
    cache  : (stored logits)                     -> softmax -> mean

Exact equality is not expected -- the two run different batch shapes, and
float32 reduction order depends on batching -- so the test is that the
difference sits at float32 noise and is orders below anything that could move
a score. Both numbers are printed rather than just the verdict.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", required=True,
                        help="cache_trunk_features.py output for a split")
    parser.add_argument("--dump", required=True,
                        help="dump_temporal_probs.py output for the alpha=0 "
                             "baseline on the SAME split")
    parser.add_argument("--arm", default=None,
                        help="which arm in the dump; default the only one")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    # A HALF-WRITTEN NPZ IS NOT A DISAGREEMENT. np.savez_compressed writes
    # incrementally, so a checker triggered on the output file APPEARING --
    # which is how this was first run -- opens a partial zip and dies with
    # `BadZipFile: File is not a zip file`. That traceback reads like the
    # cache is corrupt when the writer was simply still going. Wait on the
    # JOB, not the file; and when someone does not, say which it was.
    def load(path, role):
        try:
            return np.load(path, allow_pickle=False)
        except Exception as error:                        # noqa: BLE001
            raise SystemExit(
                "could not read the %s at %s: %s\n"
                "If the producing job is still in the queue this file is "
                "incomplete, not corrupt -- wait for the job to leave the "
                "queue rather than for the file to exist." % (role, path, error))

    cache = load(args.cache, "cache")
    dump = load(args.dump, "dump")
    meta = json.loads(str(cache["meta"]))
    head = meta["head"]

    arms = [str(a) for a in dump["arms"]]
    arm = args.arm or arms[0]
    key = "%s_%s" % (head, arm)
    if key not in dump.files:
        raise SystemExit(
            "dump has no %r; it holds %s. The cache says head=%r and the arms "
            "are %s." % (key, [f for f in dump.files], head, arms))

    dump_probs = dump[key].astype(np.float64)
    logits = cache["logits"].astype(np.float64)

    # The dump's aggregation, applied to the cache. softmax for the multiclass
    # task head, sigmoid for the multilabel tool head -- applying the wrong one
    # does not raise, it just produces plausible numbers that mean nothing.
    if head == "task":
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        per_frame = exp / exp.sum(axis=-1, keepdims=True)
    else:
        per_frame = 1.0 / (1.0 + np.exp(-logits))
    cache_probs = per_frame.mean(axis=1)

    report = {"cache": args.cache, "dump": args.dump, "arm": arm,
              "head": head,
              "cache_windows": int(cache_probs.shape[0]),
              "dump_windows": int(dump_probs.shape[0])}

    if cache_probs.shape != dump_probs.shape:
        report["verdict"] = "SHAPE MISMATCH"
        report["detail"] = ("cache %s vs dump %s -- these are not the same "
                            "windows, so no comparison is meaningful"
                            % (cache_probs.shape, dump_probs.shape))
        print(json.dumps(report, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2))
        return 1

    # ORDER MUST MATCH TOO. Both walk shards in shard_paths_for_split order and
    # windows in file order, so row i is the same window -- but if that ever
    # stops being true the values would still be plausible, so check the case
    # column rather than assuming.
    same_cases = bool((cache["cases"] == dump["cases"]).all())
    report["cases_aligned"] = same_cases

    diff = np.abs(cache_probs - dump_probs)
    report["max_abs_diff"] = float(diff.max())
    report["mean_abs_diff"] = float(diff.mean())
    report["argmax_agreement"] = float(
        (cache_probs.argmax(axis=1) == dump_probs.argmax(axis=1)).mean())

    # A score moves at the third decimal; float32 noise lives around 1e-6.
    ok = same_cases and report["max_abs_diff"] < 1e-4
    report["verdict"] = "MATCH" if ok else "DISAGREE"
    if not same_cases:
        report["detail"] = ("the case columns differ, so row i is not the "
                            "same window in both -- fix the ordering before "
                            "reading the diffs")

    print(json.dumps(report, indent=2))
    if not ok:
        print("\nThe cache does not reproduce the model it feeds. Anything "
              "trained on it is a result about a different network.")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
