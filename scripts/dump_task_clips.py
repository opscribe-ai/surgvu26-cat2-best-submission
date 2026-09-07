"""One GPU pass over a 3D TASK checkpoint -> per-window softmax, scored plainly.

WHY A SECOND DUMP SCRIPT. `dump_clip_probs.py` is a multilabel evaluator: it
sigmoids, sweeps a threshold per tool class, and needs the two-fold protocol
because a threshold tuned and scored on the same windows reports the tuning.
The task head is multiclass. There is no threshold -- the prediction is an
argmax -- so the honest number is just held-out accuracy, and forcing the task
head through the tools evaluator would sweep cuts that do not exist.

WHAT IS AND IS NOT SELF-TUNED HERE. Nothing in this script tunes anything, so
the numbers it prints are clean in the way the tools numbers needed a protocol
to become. One bias survives and it is worth naming: the CHECKPOINT was chosen
as the best of 20 epochs by val macro-F1, on these very windows. The 2D task
head's 0.9456 was selected the same way over the same number of epochs, so the
comparison is like for like -- but neither number is what a truly untouched
split would give, and both would fall by some unmeasured amount on one.

THE THREE NUMBERS, AND WHICH ONE MATTERS
----------------------------------------
    accuracy       exact class hit. What the 2D head's 0.9456 is.
    macro-F1       accuracy's poor relation here: with eight classes and a
                   long tail, it swings on a handful of rare windows.
    desc_acc       whether the DESCRIPTION the router emits is right. Three
                   classes share a modal description, so a confusion inside
                   that group costs the answer nothing. This is the number
                   that survives to the leaderboard, and it is the one to
                   read first.

CLIP OFFSETS, as in dump_clip_probs.py: center reproduces what ShardClips
feeds at eval, first and last bracket the burst. `mean3` averages the three
softmaxes; `max3` takes the elementwise max, which for a multiclass head means
"whichever clip was most confident about any class decides" -- a different and
more brittle rule than averaging, dumped so the difference can be seen rather
than argued about.

CONFOUNDS, unchanged from the tool experiment: 18 layers against a ResNet-50,
Kinetics-400 against ImageNet, 112px against 384, and a 2 s burst standing in
for the 30 s window the label describes. All four push the same way, so a 3D
number BELOW the 2D one is weak evidence about motion, while a 3D number at or
above it would be strong.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.dataset import shard_paths_for_split                # noqa: E402
from surgvu.descriptions import (description_accuracy,          # noqa: E402
                                 load_corpus)
from surgvu.extract import read_shard                           # noqa: E402
from surgvu.holdout import case_folds                           # noqa: E402
from surgvu.metrics import macro_f1                             # noqa: E402
from surgvu.taxonomy import TASK_CLASSES                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dump_clip_probs import OFFSETS                             # noqa: E402

DENSE = "/staging/n/nkalthoff/surgvu26/shards_dense"
REPO = Path(__file__).resolve().parents[1]


def load_task_expert(path, device):
    """A 3D task checkpoint, with its own metadata trusted for shape."""
    import torch

    from surgvu.models import build_video_model

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    meta = payload["meta"]
    if not meta.get("temporal"):
        raise ValueError(
            "%s is not a temporal checkpoint (meta.temporal is not set). This "
            "script feeds (B, C, T, H, W)." % path)
    if list(meta.get("classes", [])) != list(TASK_CLASSES):
        raise ValueError(
            "%s was trained on %r, not the current TASK_CLASSES. Pointing this "
            "at a TOOL checkpoint would produce 12 numbers where 8 are "
            "expected and score them against the wrong labels."
            % (path, meta.get("classes")))
    model = build_video_model(len(TASK_CLASSES), meta["backbone"],
                              pretrained=False)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shards", default=DENSE)
    parser.add_argument("--splits", default=str(REPO / "config" / "splits_v2.json"))
    parser.add_argument("--descriptions",
                        default=str(REPO / "config" / "descriptions.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True, help="the .npz to write")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-shards", type=int, default=0,
                        help="0 = all; a small number for a smoke test, whose "
                             "accuracy is not a result")
    args = parser.parse_args(argv)

    import torch

    from surgvu.train import prepare_clip_batch

    device = args.device
    if device in (None, "", "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, meta = load_task_expert(args.checkpoint, device)
    clip_length = int(meta.get("clip_length") or meta.get("frames_per_window"))
    image_size = int(meta["image_size"])
    mean = std = None
    if meta.get("normalisation") == "kinetics":
        from surgvu.models import VIDEO_MEAN, VIDEO_STD
        mean, std = VIDEO_MEAN, VIDEO_STD

    corpus = load_corpus(args.descriptions)
    index_of = {name: i for i, name in enumerate(TASK_CLASSES)}

    print("device=%s backbone=%s clip=%d @ %dpx norm=%s"
          % (device, meta["backbone"], clip_length, image_size,
             meta.get("normalisation")), flush=True)
    print("checkpoint reported acc %.4f / macroF1 %.4f / desc %.4f at epoch %s "
          "-- selected as the best of the run on these windows."
          % (meta.get("accuracy", float("nan")),
             meta.get("macro_f1", float("nan")),
             meta.get("description_accuracy", float("nan")),
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
                    clip = np.concatenate(
                        [clip, np.repeat(clip[-1:], clip_length - take, axis=0)])
                clips.append(clip)

            with torch.no_grad():
                batch = prepare_clip_batch(np.stack(clips), device,
                                           image_size, mean, std)
                out = torch.softmax(model(batch), dim=1).float().cpu().numpy()
            for slot, (name, _) in enumerate(OFFSETS):
                probs[name].append(out[slot])

            row = rows[w]
            label = str(row.get("task", "")).strip().lower()
            if label not in index_of:
                raise SystemExit(
                    "window %d of %s carries task %r, which is not one of the "
                    "%d TASK_CLASSES. Scoring it as -1 the way the 2D dump "
                    "does would silently count it wrong for every model."
                    % (w, Path(path).name, label, len(TASK_CLASSES)))
            targets.append(index_of[label])
            cases.append(str(row.get("case",
                                     Path(path).name.rsplit("_part", 1)[0])))
        print("shard %d/%d %s windows=%d elapsed=%ds"
              % (index, len(shards), Path(path).name, len(rows),
                 time.time() - started), flush=True)

    truth = np.array(targets, dtype=np.int64)
    cases = np.array(cases)
    order = [name for name, _ in OFFSETS]
    stacked = {name: np.stack(rows_).astype(np.float32)
               for name, rows_ in probs.items()}
    cube = np.stack([stacked[name] for name in order])          # (3, W, C)
    stacked["mean3"] = cube.mean(axis=0)
    stacked["max3"] = cube.max(axis=0)

    print("\nwindows %d | dense depth %s | task classes %d"
          % (len(truth), sorted(depths), len(TASK_CLASSES)), flush=True)

    # The folds do no tuning here. They are reported because a single pooled
    # accuracy hides how much of it rides on a few cases: two fold accuracies
    # far apart mean the headline is a case-sampling number as much as a model
    # number, and that is worth seeing before anything is concluded from it.
    # One-hot, not None: `case_folds` stratifies when it is given targets, and
    # the rare task classes are as case-concentrated as the rare tools were.
    # An alternating split would put a whole class on one side and make its
    # fold accuracy a statement about which cases landed where.
    eye = np.eye(len(TASK_CLASSES), dtype=np.float32)
    fold_a, fold_b = case_folds(cases, eye[truth])

    def evaluate(p):
        pred = p.argmax(axis=1)
        return {
            "accuracy": float((pred == truth).mean()),
            "macro_f1": float(macro_f1(eye[truth], eye[pred])),
            "description_accuracy": float(
                description_accuracy(truth, pred, corpus)),
            "fold_a_accuracy": float((pred[fold_a] == truth[fold_a]).mean()),
            "fold_b_accuracy": float((pred[fold_b] == truth[fold_b]).mean()),
        }

    print("folds: %d / %d windows" % (len(fold_a), len(fold_b)))
    print("\n%-8s %9s %9s %9s   %9s %9s"
          % ("arm", "acc", "macroF1", "desc_acc", "acc(A)", "acc(B)"))
    results = {}
    for name in list(order) + ["mean3", "max3"]:
        scored = results[name] = evaluate(stacked[name])
        print("%-8s %9.4f %9.4f %9.4f   %9.4f %9.4f"
              % (name, scored["accuracy"], scored["macro_f1"],
                 scored["description_accuracy"], scored["fold_a_accuracy"],
                 scored["fold_b_accuracy"]))

    best = max(results, key=lambda k: results[k]["description_accuracy"])
    top = results[best]
    print("\nCLIP-LEVEL, arm %s: accuracy %.4f, description accuracy %.4f"
          % (best, top["accuracy"], top["description_accuracy"]))
    print("The 2D ResNet-50 task head's comparable accuracy is 0.9456 "
          "(delta %+.4f). Both are best-of-20-epochs on these windows."
          % (top["accuracy"] - 0.9456))
    print("A gap smaller than the ~0.012 run-to-run floor is not a difference. "
          "Fusion, not the gap, is the test of whether motion adds anything: "
          "see scripts/fuse_task_2d_3d.py.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out), task_target=truth, cases=cases,
        task_classes=np.array(list(TASK_CLASSES)),
        arms=np.array(list(order) + ["mean3", "max3"]),
        **{"task_%s" % name: stacked[name] for name in stacked})
    Path(str(out) + ".json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "backbone": meta["backbone"], "image_size": image_size,
        "clip_length": clip_length, "normalisation": meta.get("normalisation"),
        "checkpoint_selected_epoch": meta.get("epochs"),
        "checkpoint_reported_accuracy": meta.get("accuracy"),
        "windows": int(len(truth)),
        "arms": results,
    }, indent=2), encoding="utf-8")
    print("\nwrote %s and %s.json" % (out, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
