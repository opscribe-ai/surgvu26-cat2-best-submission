"""Experiment 6, part 1: frozen EndoViT features for every window.

EndoViT is a ViT-B/16 pretrained with masked autoencoding on Endo700k -- ~700k
endoscopic frames from nine public MIS datasets (cholec80, DSAD, ESAD, GLENDA,
HeiCo, hSDB-instrument and others), Apache-2.0. The bet is that 115 training
cases is a small-data regime, which is exactly where a frozen domain backbone
beats an ImageNet CNN fine-tuned end to end.

Extracting ONCE and caching is what makes the experiment cheap: the forward
passes are paid here, and `scripts/train_head.py` then fits heads on the
cached vectors in seconds, so head architecture, learning rate and class
weighting can all be swept without touching a GPU again.

NORMALISATION IS NOT IMAGENET, AND THIS IS THE EASY WAY TO GET IT WRONG
-----------------------------------------------------------------------
The checkpoint carries the statistics it was pretrained under:

    mean [0.3464, 0.2280, 0.2228]    std [0.2520, 0.2128, 0.2093]

Those are endoscopy statistics -- note the red channel sitting far above the
other two, which is tissue. Feeding ImageNet's [0.485, 0.456, 0.406] would
shift every input off the distribution the frozen weights expect, and because
nothing here is fine-tuned there is no opportunity to adapt. It would show up
as "the foundation model did not help", which is a conclusion about our
preprocessing wearing the costume of a conclusion about the model.

This also differs from `surgvu.train.prepare_batch`, which scales to [0, 1]
and applies no mean/std at all. That is defensible for a backbone being
fine-tuned; it is not defensible for frozen features. The divergence is
deliberate and is the reason this script does its own preprocessing rather
than calling prepare_batch.
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
from surgvu.frames import sample_frame_indices                     # noqa: E402
from surgvu.taxonomy import TASK_CLASSES                         # noqa: E402

SHARDS = "/staging/groups/bhaskar_opscribe/surgvu/shards"
REPO = Path(__file__).resolve().parents[1]
WEIGHTS = "/staging/n/nkalthoff/surgvu26/v2/endovit/pytorch_model.bin"


def load_endovit(weights, img_size, device):
    """The MAE encoder as a feature extractor. Verified to load exactly.

    The checkpoint holds a full MAE -- encoder plus decoder plus the mask
    token. Only the encoder transfers; `decoder_*` and `mask_token` are
    dropped here rather than passed to `load_state_dict(strict=False)`, so
    that the strict load below is a real check. With the drop applied it
    reports zero missing and zero unexpected keys against a timm ViT-B/16,
    which is the evidence that this is the right architecture rather than a
    shape-compatible one.
    """
    import torch
    import timm

    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    encoder = {k: v for k, v in state.items()
               if not k.startswith("decoder_") and k != "mask_token"}

    # dynamic_img_size lets timm interpolate the 197-token position embedding
    # to any input size. Needed because EndoViT pretrained at 224 while our
    # experts run at 384, and whether the extra pixels help small instruments
    # is one of the questions this experiment asks.
    model = timm.models.vision_transformer.VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, num_classes=0, dynamic_img_size=(img_size != 224))
    missing, unexpected = model.load_state_dict(encoder, strict=False)
    missing = [k for k in missing if not k.startswith("head")]
    if missing or unexpected:
        raise ValueError(
            "EndoViT did not load cleanly: missing=%r unexpected=%r. Loading "
            "a partially-initialised backbone would produce features that "
            "look plausible and mean nothing." % (missing[:8], unexpected[:8]))

    mean = checkpoint["args"].mean.view(1, 3, 1, 1).to(device)
    std = checkpoint["args"].std.view(1, 3, 1, 1).to(device)
    return model.to(device).eval(), mean, std


def embed(model, frames, device, img_size, mean, std, flip=False):
    """(F, H, W, 3) uint8 BGR -> (F, 768) float32 CLS features."""
    import torch

    array = np.asarray(frames)
    if flip:
        array = array[:, :, ::-1, :]
    rgb = array[..., ::-1].copy()                      # BGR -> RGB
    with torch.no_grad():
        batch = torch.from_numpy(rgb).to(device)
        batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
        batch = torch.nn.functional.interpolate(
            batch, size=(img_size, img_size), mode="bilinear",
            align_corners=False)
        batch = (batch - mean) / std
        return model(batch).float().cpu().numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--shards", default=SHARDS)
    parser.add_argument("--splits", default=str(REPO / "config" / "splits_v2.json"))
    parser.add_argument("--split", required=True, help="train, val or heldout")
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=8,
                        help="frames per window. 8 matches how the shipped "
                             "experts train; use the full depth for val so "
                             "the aggregation sweep applies here too.")
    parser.add_argument("--img-size", type=int, default=224,
                        help="224 is EndoViT's pretraining size; larger "
                             "interpolates the position embedding")
    parser.add_argument("--flip", action="store_true",
                        help="extract the hflip arm instead of the identity one")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-shards", type=int, default=0)
    args = parser.parse_args(argv)

    import torch

    device = args.device
    if device in (None, "", "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mean, std = load_endovit(args.weights, args.img_size, device)
    print("EndoViT loaded clean | device=%s img_size=%d frames=%d flip=%s"
          % (device, args.img_size, args.frames, args.flip), flush=True)

    shards = shard_paths_for_split(args.shards, args.splits, args.split)
    if args.max_shards:
        shards = shards[:args.max_shards]
    print("%d shards in split %r" % (len(shards), args.split), flush=True)

    task_index = {name: i for i, name in enumerate(TASK_CLASSES)}
    features, tools_target, task_target, cases = [], [], [], []
    started = time.time()

    for index, path in enumerate(shards, start=1):
        frames, meta = read_shard(path)
        depth = frames.depth if hasattr(frames, "depth") else frames.shape[1]
        n_frames = min(args.frames, depth)
        picks = (range(depth) if n_frames == depth
                 else sample_frame_indices(depth, n_frames))
        for window in range(len(meta)):
            stack = np.stack([frames[window][f] for f in picks])
            features.append(embed(model, stack, device, args.img_size,
                                  mean, std, args.flip).astype(np.float16))
            row = meta[window]
            tools_target.append(encode_tools(row["tools"]))
            task_target.append(task_index.get(
                str(row.get("task", "")).strip().lower(), -1))
            cases.append(str(row.get("case", Path(path).name.rsplit("_part", 1)[0])))
        print("shard %d/%d %s windows=%d elapsed=%ds"
              % (index, len(shards), Path(path).name, len(meta),
                 time.time() - started), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out),
        features=np.stack(features),                       # (W, F, 768) fp16
        tools_target=np.stack(tools_target).astype(np.float32),
        task_target=np.array(task_target, dtype=np.int64),
        cases=np.array(cases),
        task_classes=np.array(list(TASK_CLASSES)),
        meta=np.array([json.dumps({
            "backbone": "EndoViT ViT-B/16 (MAE, Endo700k, Apache-2.0)",
            "img_size": args.img_size, "frames": args.frames,
            "flip": bool(args.flip), "split": args.split,
            "splits": str(args.splits),
        })]))
    shape = np.stack(features).shape
    print("wrote %s  windows=%d frames=%d dim=%d  %.1f MB"
          % (out, shape[0], shape[1], shape[2], out.stat().st_size / 1e6),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
