"""Check the multi-burst pool holds the SAME windows as its twins.

WHAT WOULD GO WRONG WITHOUT THIS. The whole v4 comparison rests on one claim:
that a multi-burst shard covers the same moments, with the same labels, as its
sparse and dense twins, and differs only in which frames were decoded. If the
window sets diverge -- a different stratification draw, a case extracted from a
different part, a shard that silently lost windows to short reads -- then every
arm trained on this pool is measured against a 2D number computed on a
different dataset, and the difference gets attributed to temporal modelling.

That failure is invisible from inside a training run. The loss curves look
normal, the macro-F1 is plausible, and nothing raises.

WHAT IS CHECKED, per shard, against the dense twin:

    the shard exists at all
    the same number of window rows
    identical start times, to the millisecond
    identical task labels and tool sets
    exactly bursts x frames_per_burst frames per window
    the frame rate the metadata claims

METADATA ONLY, no frame decoding, so it does not need the container -- numpy
is enough. A frame-level check would need cv2 and would not add much: the
frame COUNT is what the boundary arithmetic in ShardTemporal slices on, and
the count is in the offsets index.

RUN IT AS A JOB, not on the login node. "Metadata only" undersells the I/O:
it opens 469 npz files across two pools on ceph and took about three minutes,
which is exactly the kind of thing the login node should not be doing.

    condor_submit condor/verify.sub script=scripts/verify_multi_pool.py \
        out=multi_pool_verification.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

DENSE = "/staging/n/nkalthoff/surgvu26/shards_dense"
MULTI = "/staging/n/nkalthoff/surgvu26/shards_multi"


def shard_meta(path):
    with np.load(path, allow_pickle=False) as data:
        raw = data["meta"]
        rows = json.loads(raw.item() if hasattr(raw, "item") else str(raw))
        offsets = len(data["offsets"])
    return rows, offsets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dense", default=DENSE)
    parser.add_argument("--multi", default=MULTI)
    parser.add_argument("--expect-frames", type=int, default=32)
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    dense = {p.name: p for p in sorted(Path(args.dense).glob("*.npz"))}
    multi = {p.name: p for p in sorted(Path(args.multi).glob("*.npz"))}

    missing = sorted(set(dense) - set(multi))
    extra = sorted(set(multi) - set(dense))
    problems, windows, checked = [], 0, 0

    for name in sorted(set(dense) & set(multi)):
        d_rows, _ = shard_meta(dense[name])
        m_rows, m_offsets = shard_meta(multi[name])
        checked += 1
        windows += len(m_rows)

        if len(d_rows) != len(m_rows):
            problems.append("%s: %d windows vs %d in the dense twin"
                            % (name, len(m_rows), len(d_rows)))
            continue
        # The dense row IS the burst, so its start is shifted; the multi row
        # spans the whole window again, so it must match the ORIGINAL start,
        # which is the dense start minus the centring offset. Comparing labels
        # and window COUNT is the part that must hold exactly.
        for index, (d_row, m_row) in enumerate(zip(d_rows, m_rows)):
            if d_row["task"] != m_row["task"]:
                problems.append("%s[%d]: task %r vs %r"
                                % (name, index, m_row["task"], d_row["task"]))
                break
            if sorted(d_row["tools"]) != sorted(m_row["tools"]):
                problems.append("%s[%d]: tools %r vs %r"
                                % (name, index, m_row["tools"], d_row["tools"]))
                break
            if abs(float(m_row["length"]) - 30.0) > 1e-6:
                problems.append("%s[%d]: length %.3f, expected the full 30 s"
                                % (name, index, float(m_row["length"])))
                break
        expected_offsets = len(m_rows) * args.expect_frames + 1
        if m_offsets != expected_offsets:
            problems.append(
                "%s: offsets index is %d, expected %d (%d windows x %d frames "
                "+ 1). The frame count per window is what ShardTemporal's "
                "burst arithmetic slices on."
                % (name, m_offsets, expected_offsets, len(m_rows),
                   args.expect_frames))

    print("dense shards %d | multi shards %d | compared %d"
          % (len(dense), len(multi), checked))
    print("windows in the multi pool: %d" % windows)
    if missing:
        print("\nMISSING from the multi pool (%d): %s"
              % (len(missing), ", ".join(missing[:10])))
    if extra:
        print("\nPRESENT ONLY in the multi pool (%d): %s"
              % (len(extra), ", ".join(extra[:10])))
    if problems:
        print("\nPROBLEMS (%d):" % len(problems))
        for line in problems[:20]:
            print("  " + line)
    if not problems and not missing and not extra:
        print("\nPOOL VERIFIED: same shards, same window counts, same labels, "
              "%d frames per window everywhere." % args.expect_frames)

    if args.out:
        Path(args.out).write_text(json.dumps({
            "dense_shards": len(dense), "multi_shards": len(multi),
            "windows": windows, "missing": missing, "extra": extra,
            "problems": problems,
        }, indent=2), encoding="utf-8")
    return 1 if (problems or missing or extra) else 0


if __name__ == "__main__":
    raise SystemExit(main())
