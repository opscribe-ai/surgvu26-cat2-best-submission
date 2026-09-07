"""One GPU pass over a 3D checkpoint -> per-WINDOW probabilities, scored honestly.

WHY THIS EXISTS. `train_tools_3d.py` prints "BEST val macro-F1", and that
number cannot be compared against anything. It is a maximum over epochs
(biased upward by however many epochs ran) with thresholds tuned on the very
windows it scores. The 2D path's headline 0.7802 is neither of those things:
it is one fixed model, thresholds tuned on one case fold and scored on the
other. Setting 0.7739 beside 0.7802 as though they were the same statistic
would be the single easiest way to reach a wrong conclusion about the whole
3D experiment, so the comparable number gets computed here instead.

WHAT MAKES THE COMPARISON FAIR. The dense shards hold the SAME windows as the
sparse pool -- same cases, same parts, same stratification seed, same labels,
verified by diffing the filename lists. Only the frames inside each window
differ (2 s at 15 fps rather than 30 s at 1 fps). So a per-window macro-F1
from a dense clip and a per-window macro-F1 from aggregated sparse frames are
measured over an identical index set with identical targets, and the folds
land identically. The one thing that changes is what the model was allowed
to see, which is the question.

CLIP OFFSETS. A window holds 30 dense frames and a clip is 16, so there are
several distinct clips per window and the choice is not free:

    center    what `ShardClips` feeds at eval, so it reproduces training
    first     the opening of the burst
    last      the close of it

Serving is not restricted to one of them -- it decodes the whole test clip and
could average several. Dumping all three lets that be MEASURED offline rather
than assumed, and the spread across offsets is itself informative: a model
whose answer swings with the offset is reading something unstable.

CONFOUNDS THAT SURVIVE THIS SCRIPT. An honest number still does not make the
3D-vs-2D gap a clean test of temporality. These backbones are 18 layers
against a ResNet-50, Kinetics-400 pretrained rather than ImageNet, and run at
112px where the 2D path runs 384. Those three all push the same direction.
This script removes the PROTOCOL confound; it does not remove those.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.dataset import encode_tools, shard_paths_for_split   # noqa: E402
from surgvu.extract import read_shard                            # noqa: E402
from surgvu.holdout import (case_folds, honest_macro_f1,         # noqa: E402
                            unmeasurable_classes)
from surgvu.taxonomy import TOOL_CLASSES                         # noqa: E402

DENSE = "/staging/n/nkalthoff/surgvu26/shards_dense"
REPO = Path(__file__).resolve().parents[1]

#: (name, how to pick the start index given depth and clip length).
OFFSETS = (
    ("center", lambda depth, take: (depth - take) // 2),
    ("first", lambda depth, take: 0),
    ("last", lambda depth, take: depth - take),
)


def load_video_expert(path, device):
    """A 3D checkpoint and its metadata, with the metadata trusted for shape.

    The backbone, the image size and the clip length all come from the
    checkpoint rather than from flags. A 112px checkpoint evaluated at 224
    scores badly for a reason that looks exactly like a bad architecture, and
    nothing would raise.
    """
    import torch

    from surgvu.models import build_video_model

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    meta = payload["meta"]
    if not meta.get("temporal"):
        raise ValueError(
            "%s is not a temporal checkpoint (meta.temporal is not set). This "
            "script feeds (B, C, T, H, W); a 2D model would either raise or, "
            "worse, silently convolve across the wrong axis." % path)
    if list(meta.get("classes", [])) != list(TOOL_CLASSES):
        raise ValueError("%s was trained on %r, not the current TOOL_CLASSES"
                         % (path, meta.get("classes")))
    model = build_video_model(len(TOOL_CLASSES), meta["backbone"],
                              pretrained=False)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shards", default=DENSE)
    parser.add_argument("--splits", default=str(REPO / "config" / "splits_v2.json"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True, help="the .npz to write")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-shards", type=int, default=0,
                        help="0 = all; a small number for a smoke test, whose "
                             "macro-F1 is not a result")
    args = parser.parse_args(argv)

    import torch

    from surgvu.train import prepare_clip_batch

    device = args.device
    if device in (None, "", "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, meta = load_video_expert(args.checkpoint, device)
    clip_length = int(meta.get("clip_length") or meta.get("frames_per_window"))
    image_size = int(meta["image_size"])
    kinetics = meta.get("normalisation") == "kinetics"
    mean = std = None
    if kinetics:
        from surgvu.models import VIDEO_MEAN, VIDEO_STD
        mean, std = VIDEO_MEAN, VIDEO_STD

    print("device=%s backbone=%s clip=%d @ %dpx norm=%s"
          % (device, meta["backbone"], clip_length, image_size,
             meta.get("normalisation")), flush=True)
    print("checkpoint reported val macro-F1 %.4f at epoch %s -- that figure is "
          "a max over epochs with self-tuned cuts and is NOT what this script "
          "computes." % (meta.get("macro_f1", float("nan")),
                         meta.get("epochs")), flush=True)

    shards = shard_paths_for_split(args.shards, args.splits, args.split)
    if args.max_shards:
        shards = shards[:args.max_shards]
        print("SMOKE TEST: %d shards. Not a result." % args.max_shards)
    print("%d shards in split %r" % (len(shards), args.split), flush=True)

    probs = {name: [] for name, _ in OFFSETS}
    targets, cases, depths = [], [], set()
    started = time.time()

    for index, path in enumerate(shards, start=1):
        frames, rows = read_shard(path)
        for w in range(len(rows)):
            depth = len(frames[w])
            depths.add(int(depth))
            take = min(clip_length, depth)
            clips = []
            for _, pick in OFFSETS:
                start = max(0, pick(depth, take))
                clip = np.stack([frames[w][start + i] for i in range(take)])
                if take < clip_length:
                    # Repeat the last frame rather than wrapping to the first:
                    # wrapping would splice a jump-cut into the middle of a
                    # motion the model is being asked to read.
                    clip = np.concatenate(
                        [clip, np.repeat(clip[-1:], clip_length - take, axis=0)])
                clips.append(clip)

            with torch.no_grad():
                batch = prepare_clip_batch(np.stack(clips), device,
                                           image_size, mean, std)
                out = torch.sigmoid(model(batch)).float().cpu().numpy()
            for slot, (name, _) in enumerate(OFFSETS):
                probs[name].append(out[slot])

            row = rows[w]
            targets.append(encode_tools(row["tools"]))
            cases.append(str(row.get("case",
                                     Path(path).name.rsplit("_part", 1)[0])))
        print("shard %d/%d %s windows=%d elapsed=%ds"
              % (index, len(shards), Path(path).name, len(rows),
                 time.time() - started), flush=True)

    target = np.stack(targets).astype(np.float32)
    cases = np.array(cases)
    stacked = {name: np.stack(rows_).astype(np.float32)
               for name, rows_ in probs.items()}
    # Aggregating ACROSS offsets is the multi-clip arm: three views of the same
    # window, combined the way serving could combine them.
    order = [name for name, _ in OFFSETS]
    cube = np.stack([stacked[name] for name in order])          # (3, W, C)
    stacked["mean3"] = cube.mean(axis=0)
    stacked["max3"] = cube.max(axis=0)

    print("\nwindows %d | dense depth %s | tool classes %d"
          % (len(target), sorted(depths), target.shape[1]), flush=True)

    fold_a, fold_b = case_folds(cases, target)
    skipped = unmeasurable_classes(target, fold_a, fold_b, list(TOOL_CLASSES))
    print("folds: %d / %d windows" % (len(fold_a), len(fold_b)))
    if skipped:
        print("structurally unmeasurable (no positives in one fold): %s"
              % ", ".join(skipped))

    print("\n%-8s %8s %8s %8s" % ("arm", "honest", "measur.", "self"))
    results = {}
    for name in list(order) + ["mean3", "max3"]:
        scored = honest_macro_f1(target, stacked[name], fold_a, fold_b)
        results[name] = scored
        print("%-8s %8.4f %8.4f %8.4f"
              % (name, scored["honest"], scored["honest_measurable"],
                 scored["self_tuned"]))

    best = max(results, key=lambda k: results[k]["honest"])
    print("\nHONEST CLIP-LEVEL macro-F1: %.4f (%s)"
          % (results[best]["honest"], best))
    print("This is the number comparable to the 2D ResNet-50's 0.7802 -- same "
          "windows, same folds, same protocol.")
    print("Backbone/pretraining/resolution confounds remain: 18 layers vs 50, "
          "Kinetics vs ImageNet, %dpx vs 384." % image_size)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out), tools_target=target, cases=cases,
        tool_classes=np.array(list(TOOL_CLASSES)),
        arms=np.array(list(order) + ["mean3", "max3"]),
        **{"tools_%s" % name: stacked[name] for name in stacked})
    Path(str(out) + ".json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "backbone": meta["backbone"], "image_size": image_size,
        "clip_length": clip_length, "normalisation": meta.get("normalisation"),
        "checkpoint_self_tuned_max_over_epochs": meta.get("macro_f1"),
        "windows": int(len(target)),
        "unmeasurable_classes": skipped,
        "arms": {k: {"honest": v["honest"],
                     "honest_measurable": v["honest_measurable"],
                     "self_tuned": v["self_tuned"],
                     "per_class": dict(zip(TOOL_CLASSES, v["per_class"]))}
                 for k, v in results.items()},
    }, indent=2), encoding="utf-8")
    print("\nwrote %s and %s.json" % (out, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
