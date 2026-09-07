"""Run the FROZEN trunk once and cache what the sequence branch consumes.

WHY. `ResidualTemporal` freezes the 2D trunk completely -- it is the property
that makes the arm unable to hurt -- and then `train_temporal.py` recomputes
that frozen trunk from JPEG on every epoch. Measured on cluster 9660360, the
pool runs at 1.83 windows/s, which is 1.6 hours per epoch at 24 frames and
3.1 hours at 48. Eight epochs is thirty hours of decoding pixels through a
network whose weights cannot change.

WHAT THE SEQUENCE BRANCH ACTUALLY READS, from ResidualTemporal.forward:

    pooled = tapped[:, :, per_burst // 2]      # the CENTRE frame of each burst
    pooled = pooled.mean(dim=(-2, -1))         # spatially averaged -> (B, bursts, C)
    out    = centre_logits + beta * sequence(pooled)

That is 16 vectors of 2048 per window, plus the 16 per-centre 2D logits. For
the whole 24,578-window pool that is 1.61 GB in fp16 -- and the pass only has
to decode the 16 CENTRE frames, not all 48, so it is three times cheaper than
one training epoch and replaces all of them.

WHAT THIS CANNOT CACHE, stated so the limit is not discovered later. The LOCAL
branch reads spatial feature maps for every frame of every burst, which is
2048x12x12 per frame and about 696 GB across the pool. It is not cacheable at
any useful resolution, so a local-branch arm keeps paying full decode. That is
a real argument for testing the 30 s timescale first, on top of the ones
already recorded: the calibration pointed at the task head, the local branch's
per-burst corrections are order-invariant after the mean, and now it is also
the only one of the two that can be iterated on in minutes.

WHAT IS RECORDED WITH THE FEATURES, so a cache cannot be silently misused:
the checkpoint it came from, the tap, the burst geometry, the image size and
the sampler. A cache built from a different base model, or at a different
resolution, is not a cache of this model -- and the cost of finding that out
by comparing numbers is what the v4 night already paid.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.dataset import (encode_task, encode_tools,             # noqa: E402
                            shard_paths_for_split)
from surgvu.extract import read_shard                              # noqa: E402
from surgvu.frames import sample_frame_indices                     # noqa: E402
from surgvu.taxonomy import TASK_CLASSES, TOOL_CLASSES             # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MULTI16 = "/staging/n/nkalthoff/surgvu26/shards_multi16"
TOOLS_2D = "/staging/n/nkalthoff/surgvu26/models/tools_resnet50_long.pt"
TASK_2D = "/staging/n/nkalthoff/surgvu26/models/task_resnet50_long.pt"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--head", choices=("tools", "task"), default="task")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--shards", default=MULTI16)
    parser.add_argument("--splits", default=str(REPO / "config" / "splits_v2.json"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--frames-per-burst", type=int, default=3)
    parser.add_argument("--bursts", type=int, default=16)
    parser.add_argument("--tap", choices=("layer3", "layer4"), default="layer4")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="CENTRE FRAMES per forward, not windows")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-shards", type=int, default=0)
    args = parser.parse_args(argv)

    import torch
    from surgvu.models import build_model
    from surgvu.temporal import ResidualTemporal
    from surgvu.train import prepare_clip_batch

    checkpoint = args.checkpoint or (TOOLS_2D if args.head == "tools"
                                     else TASK_2D)
    classes = TOOL_CLASSES if args.head == "tools" else TASK_CLASSES
    device = args.device
    if device in (None, "", "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    backbone = build_model(len(classes), "resnet50", pretrained=False)
    backbone.load_state_dict(payload["state_dict"])
    # Built through ResidualTemporal so the trunk walk and the tap are the SAME
    # code the arm will use. Reimplementing the walk here is how a cache ends
    # up holding features from a slightly different network than the one it
    # feeds -- the 0.048 lesson, in a new costume.
    model = ResidualTemporal(backbone, args.bursts, args.frames_per_burst,
                             len(classes), tap=args.tap).to(device).eval()

    shards = shard_paths_for_split(args.shards, args.splits, args.split)
    if args.max_shards:
        shards = shards[:args.max_shards]
    print("device=%s head=%s tap=%s | %d shards in split %r"
          % (device, args.head, args.tap, len(shards), args.split), flush=True)
    print("caching %d bursts x pooled(%s) per window, from the CENTRE frame "
          "of each burst only" % (args.bursts, args.tap), flush=True)

    feats, logits, tools_y, task_y, cases = [], [], [], [], []
    started = time.time()
    for index, path in enumerate(shards, start=1):
        frames, rows = read_shard(path)
        for w in range(len(rows)):
            depth = len(frames[w])
            per_burst = max(1, depth // args.bursts)
            if depth % args.bursts:
                raise SystemExit(
                    "%s window %d holds %d frames, not a whole number of %d "
                    "bursts" % (Path(path).name, w, depth, args.bursts))
            # THE CENTRE OF EACH BURST, which is the frame the 2D path samples.
            centres = [b * per_burst + per_burst // 2
                       for b in range(args.bursts)]
            stack = np.stack([frames[w][i] for i in centres])[None, ...]

            with torch.no_grad():
                batch = prepare_clip_batch(stack, device, args.image_size,
                                           None, None)
                folded = batch.permute(0, 2, 1, 3, 4).reshape(
                    -1, 3, args.image_size, args.image_size)
                tapped, frame_logits = model._trunk(folded)
                pooled = tapped.mean(dim=(-2, -1))       # (bursts, C)
            feats.append(pooled.half().cpu().numpy())
            logits.append(frame_logits.float().cpu().numpy())

            row = rows[w]
            tools_y.append(encode_tools(row["tools"]))
            task_y.append(encode_task(row["task"]))
            cases.append(str(row.get("case",
                                     Path(path).name.rsplit("_part", 1)[0])))
        if index % 10 == 0 or index == len(shards):
            done = time.time() - started
            print("  shard %d/%d  windows=%d  elapsed=%ds  eta=%ds"
                  % (index, len(shards), len(feats), done,
                     done / index * (len(shards) - index)), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        features=np.stack(feats),                 # (windows, bursts, C) fp16
        logits=np.stack(logits),                  # (windows, bursts, classes)
        tools=np.stack(tools_y).astype(np.float32),
        task=np.array(task_y, dtype=np.int64),
        cases=np.array(cases),
        # PROVENANCE. A cache is only a cache OF something.
        meta=json.dumps({
            "checkpoint": str(checkpoint),
            "head": args.head, "tap": args.tap,
            "bursts": args.bursts,
            "frames_per_burst": args.frames_per_burst,
            "image_size": args.image_size,
            "split": args.split, "shards": len(shards),
            "classes": list(classes),
            "sampler": "burst_centres",
            "note": ("pooled trunk features and per-centre 2D logits. The "
                     "LOCAL motion branch cannot be trained from this -- it "
                     "needs spatial maps for all frames, ~696 GB."),
        }))
    print("\nwrote %s: %d windows, %.2f GB"
          % (out, len(feats), out.stat().st_size / 1e9))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
