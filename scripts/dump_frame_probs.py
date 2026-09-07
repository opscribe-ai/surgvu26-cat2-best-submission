"""One GPU pass over validation -> every per-frame probability, cached.

Experiments 1 (aggregation), 2 (frame count) and 3 (test-time augmentation)
are all functions of the SAME per-frame probabilities. Running a separate GPU
pass for each would re-derive an identical tensor three times and, worse,
would make the three results incomparable if anything in the loading path
drifted between them. So this pays for the forward passes once and writes the
tensor; `scripts/sweep_aggregation.py` then sweeps all three offline on CPU
in seconds.

WHAT IS DUMPED, AND WHY IT IS THE FULL DEPTH
--------------------------------------------
Every frame of every window, not the 16 the container samples. A shard window
holds 30 frames at 1 fps over the same 30 seconds a serving clip covers, so
sampling 16 of 30 here lands on the same moments serving lands on out of
1800 -- but dumping only those 16 would make a frame-count sweep impossible.
Dumping all 30 lets the sweep subsample offline, which is what makes
experiment 2 free.

CEILING. 30 is a hard cap on any frame count measured this way. Serving
decodes 60 fps video and could take more, but there is no labelled way to
measure above 30 without re-extracting the corpus. Report the cap; do not
quietly present 30 as if it were the maximum useful number of frames.

ARMS
----
Test-time augmentation is dumped as extra leading axis rather than as extra
runs, for the same reason the frames are: an arm is only interesting compared
against the others on identical windows in identical order.

    id        the serving path exactly
    hflip     left-right mirror. Instrument IDENTITY is not chiral, but arm
              POSITION is, so this is a question rather than a freebie.
    scale448  384 -> 448. EfficientNet is fully convolutional and takes it;
              whether more pixels helps small instruments is the question.

CASE IDS ARE DUMPED ON PURPOSE. Thresholds tuned and then scored on the same
windows report the tuning, not the model. The sweep splits val by CASE for an
honest estimate, and it can only do that if it knows which case each window
came from.
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
from surgvu.perceive import load_expert                          # noqa: E402
from surgvu.taxonomy import TASK_CLASSES, TOOL_CLASSES           # noqa: E402

SHARDS = "/staging/groups/bhaskar_opscribe/surgvu/shards"
REPO = Path(__file__).resolve().parents[1]

# (name, horizontal flip, image size). `None` size means "the expert's own".
ARMS = (
    ("id", False, None),
    ("hflip", True, None),
    ("scale448", False, 448),
)


def forward_frames(model, frames, device, image_size, activation):
    """(F, H, W, 3) uint8 BGR -> (F, C) probabilities. No aggregation."""
    import torch

    from surgvu.train import prepare_batch

    model.eval()
    with torch.no_grad():
        batch = prepare_batch(frames, device, image_size)
        logits = model(batch)
        if activation == "sigmoid":
            probs = torch.sigmoid(logits)
        elif activation == "softmax":
            probs = torch.softmax(logits, dim=1)
        else:
            raise ValueError("unknown activation %r" % (activation,))
        return probs.float().cpu().numpy()


def load_bound(entry, models_dir, device):
    path = Path(entry["checkpoint"])
    if models_dir:
        path = Path(models_dir) / path.name
    if not path.exists():
        raise FileNotFoundError("%s checkpoint %s does not exist"
                                % (entry.get("role"), path))
    model, meta = load_expert(path, device)
    if list(meta.get("classes", [])) != list(entry["classes"]):
        raise ValueError(
            "%s checkpoint was trained on %r but the config binds %r"
            % (entry.get("role"), meta.get("classes"), entry["classes"]))
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(REPO / "config" / "perception.json"))
    parser.add_argument("--models-dir")
    parser.add_argument("--shards", default=SHARDS)
    parser.add_argument("--splits", default=str(REPO / "config" / "splits_v2.json"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True, help="the .npz to write")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-shards", type=int, default=0,
                        help="0 = all; a small number for a smoke test")
    parser.add_argument("--arms", default="id,hflip,scale448",
                        help="comma-separated subset of the arm names")
    # Overridable so a ResNet-50 or EndoViT checkpoint can be dumped through
    # the identical path without editing the frozen serving config.
    parser.add_argument("--tools-checkpoint",
                        help="override the config's tools checkpoint")
    parser.add_argument("--task-checkpoint",
                        help="override the config's task checkpoint")
    args = parser.parse_args(argv)

    import torch

    device = args.device
    if device in (None, "", "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device=%s torch=%s" % (device, torch.__version__), flush=True)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tools_entry = dict(config["experts"]["tools"])
    task_entry = dict(config["experts"]["task"])
    if args.tools_checkpoint:
        tools_entry["checkpoint"] = args.tools_checkpoint
    if args.task_checkpoint:
        task_entry["checkpoint"] = args.task_checkpoint

    models_dir = args.models_dir if not (args.tools_checkpoint or
                                         args.task_checkpoint) else None
    tools_model = load_bound(tools_entry, models_dir, device)
    task_model = load_bound(task_entry, models_dir, device)
    print("loaded tools=%s task=%s"
          % (tools_entry["checkpoint"], task_entry["checkpoint"]), flush=True)

    wanted = [name.strip() for name in args.arms.split(",") if name.strip()]
    arms = [arm for arm in ARMS if arm[0] in wanted]
    if len(arms) != len(wanted):
        raise ValueError("unknown arm in %r; known: %s"
                         % (wanted, [a[0] for a in ARMS]))
    print("arms: %s" % ([a[0] for a in arms],), flush=True)

    shards = shard_paths_for_split(args.shards, args.splits, args.split)
    if args.max_shards:
        shards = shards[:args.max_shards]
    print("%d shards in split %r" % (len(shards), args.split), flush=True)

    task_index = {name: i for i, name in enumerate(TASK_CLASSES)}
    tools_out = {arm[0]: [] for arm in arms}
    task_out = {arm[0]: [] for arm in arms}
    tools_target, task_target, cases, depths = [], [], [], set()
    started = time.time()

    for index, path in enumerate(shards, start=1):
        frames, meta = read_shard(path)
        depth = frames.depth if hasattr(frames, "depth") else frames.shape[1]
        depths.add(int(depth))
        for window in range(len(meta)):
            stack = np.stack([frames[window][f] for f in range(depth)])
            for name, flip, size in arms:
                view = stack[:, :, ::-1, :].copy() if flip else stack
                tools_out[name].append(forward_frames(
                    tools_model, view, device,
                    size or tools_entry["image_size"], tools_entry["activation"]))
                task_out[name].append(forward_frames(
                    task_model, view, device,
                    size or task_entry["image_size"], task_entry["activation"]))
            row = meta[window]
            tools_target.append(encode_tools(row["tools"]))
            task_target.append(task_index.get(str(row.get("task", "")).strip().lower(), -1))
            cases.append(str(row.get("case", Path(path).name.rsplit("_part", 1)[0])))
        print("shard %d/%d %s windows=%d elapsed=%ds"
              % (index, len(shards), Path(path).name, len(meta),
                 time.time() - started), flush=True)

    if len(depths) != 1:
        # Ragged depth would make the (W, F, C) stack impossible and a silent
        # pad would double-count a moment. Fail with the fact, not a traceback.
        raise ValueError("shards disagree on frames per window: %s. The "
                         "sweep indexes a fixed depth." % (sorted(depths),))

    payload = {
        "arms": np.array([a[0] for a in arms]),
        "tool_classes": np.array(list(TOOL_CLASSES)),
        "task_classes": np.array(list(TASK_CLASSES)),
        "tools_target": np.stack(tools_target).astype(np.float32),
        "task_target": np.array(task_target, dtype=np.int64),
        "cases": np.array(cases),
        "depth": np.array([depths.pop()], dtype=np.int64),
    }
    for name in tools_out:
        payload["tools_%s" % name] = np.stack(tools_out[name]).astype(np.float32)
        payload["task_%s" % name] = np.stack(task_out[name]).astype(np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out), **payload)
    shape = payload["tools_%s" % arms[0][0]].shape
    print("wrote %s  windows=%d frames=%d tool_classes=%d arms=%d  %.1f MB"
          % (out, shape[0], shape[1], shape[2], len(arms),
             out.stat().st_size / 1e6), flush=True)
    print("provenance: tools=%s task=%s split=%s splits=%s"
          % (tools_entry["checkpoint"], task_entry["checkpoint"],
             args.split, args.splits), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
