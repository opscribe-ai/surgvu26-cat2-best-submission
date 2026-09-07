"""Re-tune the tool thresholds for the aggregation SERVING actually uses.

THE DEFECT THIS MEASURES
------------------------
`scripts/train_tools.py` tunes one threshold per class on PER-FRAME validation
probabilities (`tune_thresholds` over `run_epoch`'s output, one row per frame).
`scripts/inference.py` applies those same cuts to the MEAN of the clip's
frames (`predict.aggregate_window`, 16 frames by default). Averaging preserves
the mean of a class's probability distribution and shrinks its variance, so a
cut chosen on the wide per-frame distribution is not the cut that maximises F1
on the narrow clip-level one. The shipped cuts span 0.05 to 0.95, which is
where that mismatch bites hardest.

Nothing here retrains anything and nothing here writes a `.pt`. It runs the
frozen tool checkpoint over the validation split, aggregates exactly the way
the container does, and re-runs the same `tune_thresholds` on the aggregated
probabilities.

FIDELITY TO SERVING IS THE WHOLE POINT
--------------------------------------
A re-tune measured against some other aggregation would be worthless -- it
would fix a mismatch by introducing a second one. So:

  * the frame count comes from the CONFIG's `decode.frames`, not a constant;
  * the frames are chosen by `perceive.sample_frame_indices`, the same
    bin-centre sampler the container uses on a video;
  * the resize is `train.prepare_batch` at the config's `image_size`, via
    the same forward `predict.predict_window` performs;
  * the reducer is `predict.aggregate_window` itself, not a local `.mean()`;
  * and `--self-check` proves the first three by running `predict_window`
    unmodified over the first window and comparing it, element-wise, against
    this script's own per-frame matrix reduced by `aggregate_window`.

WHY VALIDATION SHARDS AND NOT VIDEOS. A shard window holds 30 frames at 1 fps
covering the same 30 seconds a serving clip covers at 60 fps, and every one of
them already went through `preprocess.prepare_frame` at extraction -- the same
crop, the same mandatory UI blur, the same 512. Sampling 16 bin centres out of
those 30 lands on the same 16 moments the serving sampler lands on out of
1800. Re-decoding 45 videos to move those moments by fractions of a second
would cost hours and change nothing.

    condor_submit condor/train.sub \
        script=scripts/tune_serving_thresholds.py \
        args="--report /staging/n/nkalthoff/surgvu26/serving_thresholds.json"

The report it writes is the input to
`scripts/build_perception_config.py --tools-serving-thresholds`.
"""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.dataset import encode_tools, shard_paths_for_split    # noqa: E402
from surgvu.extract import read_shard                             # noqa: E402
from surgvu.metrics import macro_f1, per_class_f1, tune_thresholds  # noqa: E402
from surgvu.perceive import sample_frame_indices                  # noqa: E402
from surgvu.predict import aggregate_window                       # noqa: E402
from surgvu.taxonomy import TOOL_CLASSES                          # noqa: E402

SHARDS = "/staging/groups/bhaskar_opscribe/surgvu/shards"
REPO = Path(__file__).resolve().parents[1]

# The number of leading frames of each window that `train_tools.py` scored to
# produce the thresholds now in the checkpoint. `ShardFrames(shuffle=False)`
# takes `range(k)` with `k = frames_per_window`, so the per-frame baseline
# here has to take exactly those frames in exactly that order -- otherwise it
# would not reproduce the 0.6605 the checkpoint records, and a baseline that
# does not reproduce is not a baseline.
TRAINING_FRAMES_KEY = "frames_per_window"


def evaluate(y_true, probs, thresholds):
    """macro-F1 and per-class F1 of `probs >= thresholds`.

    `>=`, matching `train_tools.py`'s selection rule and
    `perceive.tools_present`. A strict `>` here would score a candidate
    threshold differently than the container will apply it.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    if y_true.shape != probs.shape:
        raise ValueError("truth %r and probabilities %r disagree in shape"
                         % (y_true.shape, probs.shape))
    if thresholds.shape[1] != probs.shape[1]:
        raise ValueError(
            "%d thresholds for %d classes. They are positional, so a short "
            "list would broadcast against the wrong columns rather than fail."
            % (thresholds.shape[1], probs.shape[1]))
    pred = (probs >= thresholds).astype(np.float32)
    return {"macro_f1": macro_f1(y_true, pred),
            "per_class_f1": dict(zip(TOOL_CLASSES,
                                     per_class_f1(y_true, pred).tolist()))}


def aggregate_all(frame_probs):
    """(W, F, C) per-frame probabilities -> (W, C) clip probabilities.

    Reduced one window at a time by `predict.aggregate_window`, the function
    the container calls, rather than by a local `mean(axis=1)` that would be
    free to drift away from it.
    """
    frame_probs = np.asarray(frame_probs, dtype=np.float32)
    if frame_probs.ndim != 3:
        raise ValueError(
            "expected (windows, frames, classes); got shape %r. A 2-D input "
            "is already aggregated and would be averaged across WINDOWS."
            % (frame_probs.shape,))
    return np.stack([aggregate_window(window) for window in frame_probs])


def flatten_per_frame(frame_probs, targets):
    """(W, F, C) + (W, C) -> ((W*F, C), (W*F, C)), one row per frame.

    The per-frame baseline scores frames, not windows: `run_epoch` emits one
    row per frame and every frame of a window carries that window's label,
    because the target is installation state and it is constant across a
    window by construction.
    """
    frame_probs = np.asarray(frame_probs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if frame_probs.ndim != 3:
        raise ValueError("expected (windows, frames, classes); got %r"
                         % (frame_probs.shape,))
    if targets.shape != (frame_probs.shape[0], frame_probs.shape[2]):
        raise ValueError("targets %r do not match probabilities %r"
                         % (targets.shape, frame_probs.shape))
    windows, frames, classes = frame_probs.shape
    return (frame_probs.reshape(windows * frames, classes),
            np.repeat(targets, frames, axis=0))


def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def build_report(classes, shipped, retuned, measurements, provenance):
    """The JSON handed to `build_perception_config.py`.

    Self-describing on purpose: it names the weights it was measured on, the
    split, the aggregation, and both sides of the before/after. A threshold
    vector with no statement of what it was tuned against is indistinguishable
    from a typo.
    """
    shipped = [float(value) for value in shipped]
    retuned = [float(value) for value in retuned]
    if not (len(shipped) == len(retuned) == len(classes)):
        raise ValueError(
            "%d classes, %d shipped thresholds, %d re-tuned. These are "
            "positional and are about to be written into a serving config."
            % (len(classes), len(shipped), len(retuned)))
    return {
        "produced_by": "scripts/tune_serving_thresholds.py",
        "role": "tools",
        "classes": list(classes),
        "checkpoint_thresholds": shipped,
        "serving_thresholds": retuned,
        "serving_thresholds_by_class": dict(zip(classes, retuned)),
        "measurements": measurements,
        "provenance": provenance,
    }


# --------------------------------------------------------------------------
# the pass over validation (torch only from here down)
# --------------------------------------------------------------------------

def frame_probs_for(model, frames, device, image_size, activation="sigmoid"):
    """Per-FRAME probabilities for one window's frame stack.

    This is `predict_window` with the final `aggregate_window` left off, and
    `--self-check` proves that claim against the real function rather than
    asserting it in a comment.
    """
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


def run_pass(model, shards, device, image_size, activation, n_frames,
             training_frames, self_check=True, progress=print):
    """(serving per-frame probs, training per-frame probs, targets).

    Shapes are (W, n_frames, C), (W, training_frames, C) and (W, C).
    """
    import time

    serving, training, targets = [], [], []
    started = time.time()
    checked = not self_check
    for index, path in enumerate(shards, start=1):
        frames, meta = read_shard(path)
        depth = frames.depth if hasattr(frames, "depth") else frames.shape[1]
        if depth < max(n_frames, training_frames):
            raise ValueError(
                "%s stores %d frames per window but the measurement needs %d. "
                "A short window cannot be sampled the way serving samples a "
                "clip, and padding it would double-count a moment."
                % (path, depth, max(n_frames, training_frames)))
        picks = sample_frame_indices(depth, n_frames)
        for window in range(len(meta)):
            stack = np.stack([frames[window][i] for i in picks])
            probs = frame_probs_for(model, stack, device, image_size, activation)
            if not checked:
                _prove_equivalence(model, stack, device, image_size,
                                   activation, probs, progress)
                checked = True
            serving.append(probs)
            training.append(frame_probs_for(
                model,
                np.stack([frames[window][i] for i in range(training_frames)]),
                device, image_size, activation))
            targets.append(encode_tools(meta[window]["tools"]))
        progress("shard %d/%d %s windows=%d elapsed=%ds"
                 % (index, len(shards), Path(path).name, len(meta),
                    time.time() - started))
    return (np.stack(serving).astype(np.float32),
            np.stack(training).astype(np.float32),
            np.stack(targets).astype(np.float32))


def _prove_equivalence(model, stack, device, image_size, activation, probs,
                       progress):
    """`aggregate_window(our per-frame probs)` IS `predict_window(...)`.

    Run once, on the first real window, against the unmodified serving
    function. If this ever fails, every number below is measuring something
    the container does not do, and the run must stop rather than report.
    """
    from surgvu.predict import predict_window

    reference = predict_window(model, stack, device, image_size,
                               activation=activation)
    ours = aggregate_window(probs)
    delta = float(np.max(np.abs(reference - ours)))
    progress("self-check: max |predict_window - aggregate_window(frames)| = %.3e"
             % delta)
    if delta > 1e-6:
        raise ValueError(
            "this script's per-frame forward does not reproduce "
            "predict_window (max delta %.3e). The thresholds it would tune "
            "are calibrated for an aggregation the container does not "
            "perform." % delta)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(REPO / "config" / "perception.json"),
                        help="the frozen binding: frame count, image_size, "
                             "activation and the checkpoint to measure")
    parser.add_argument("--models-dir",
                        help="re-root the config's checkpoint here by basename")
    parser.add_argument("--shards", default=SHARDS)
    parser.add_argument("--splits", default=str(REPO / "config" / "splits_v2.json"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--report", required=True,
                        help="where the JSON report is written")
    parser.add_argument("--probs-out",
                        help="optional .npz of the raw probabilities, so the "
                             "tuning can be re-derived without a second GPU pass")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-shards", type=int, default=0,
                        help="smoke-test knob (0 = all). A capped run's "
                             "thresholds are wiring evidence, not a result.")
    args = parser.parse_args(argv)

    import torch

    from surgvu.perceive import load_expert

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    entry = config["experts"]["tools"]
    n_frames = int(config["decode"]["frames"])
    training_frames = int(entry[TRAINING_FRAMES_KEY])
    checkpoint = Path(entry["checkpoint"])
    if args.models_dir:
        checkpoint = Path(args.models_dir) / checkpoint.name

    device = args.device
    if device in (None, "", "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device %s  checkpoint %s" % (device, checkpoint))
    print("aggregation: %d frames -> predict_window(image_size=%s, %s) -> mean"
          % (n_frames, entry["image_size"], entry["activation"]))

    digest = sha256(checkpoint)
    if digest != entry["sha256"]:
        raise SystemExit(
            "the checkpoint at %s hashes to %s but the config binds %s. "
            "Thresholds tuned on the wrong weights are worse than untuned "
            "ones." % (checkpoint, digest, entry["sha256"]))

    model, meta = load_expert(checkpoint, device)
    if list(meta["classes"]) != list(TOOL_CLASSES):
        raise SystemExit("checkpoint classes %r are not the taxonomy"
                         % (meta["classes"],))
    shipped = [float(value) for value in entry["thresholds"]]

    shards = shard_paths_for_split(args.shards, args.splits, args.split)
    if args.max_shards:
        shards = shards[:args.max_shards]
        print("SMOKE TEST: capped at %d shards. These thresholds are NOT a "
              "result." % args.max_shards)
    print("%s shards: %d" % (args.split, len(shards)))

    serving_probs, training_probs, targets = run_pass(
        model, shards, device, entry["image_size"], entry["activation"],
        n_frames, training_frames)
    print("windows %r  serving %r  training %r"
          % (targets.shape, serving_probs.shape, training_probs.shape))

    clip = aggregate_all(serving_probs)
    flat_probs, flat_targets = flatten_per_frame(training_probs, targets)

    per_frame_shipped = evaluate(flat_targets, flat_probs, shipped)
    clip_shipped = evaluate(targets, clip, shipped)
    retuned = [float(value) for value in tune_thresholds(targets, clip)]
    clip_retuned = evaluate(targets, clip, retuned)

    measurements = {
        "per_frame_checkpoint_thresholds": per_frame_shipped,
        "clip_mean_checkpoint_thresholds": clip_shipped,
        "clip_mean_serving_thresholds": clip_retuned,
        "delta_macro_f1": (clip_retuned["macro_f1"]
                           - clip_shipped["macro_f1"]),
    }
    report = build_report(
        TOOL_CLASSES, shipped, retuned, measurements,
        {
            "checkpoint": str(entry["checkpoint"]),
            "checkpoint_name": checkpoint.name,
            "checkpoint_sha256": digest,
            # The NAME, not the path handed in. This report is about to be
            # embedded in a serving config, and the path it was run under is
            # an HTCondor scratch directory that will not exist tomorrow.
            "splits": Path(args.splits).name,
            "split": args.split,
            "shards": len(shards),
            "windows": int(targets.shape[0]),
            "decode_frames": n_frames,
            "image_size": int(entry["image_size"]),
            "activation": entry["activation"],
            "training_frames_per_window": training_frames,
            "aggregation": "perceive.sample_frame_indices -> "
                           "predict.predict_window -> "
                           "predict.aggregate_window (mean)",
            "date": datetime.date.today().isoformat(),
            "max_shards": int(args.max_shards),
        })

    for label, key in (("PER-FRAME (first %d), checkpoint thresholds"
                        % training_frames, "per_frame_checkpoint_thresholds"),
                       ("CLIP MEAN of %d, checkpoint thresholds (SERVING TODAY)"
                        % n_frames, "clip_mean_checkpoint_thresholds"),
                       ("CLIP MEAN of %d, RE-TUNED" % n_frames,
                        "clip_mean_serving_thresholds")):
        block = measurements[key]
        cuts = retuned if key == "clip_mean_serving_thresholds" else shipped
        print("\n== %s   macroF1 = %.4f" % (label, block["macro_f1"]))
        for name, cut in zip(TOOL_CLASSES, cuts):
            print("   %-32s thr=%.2f  F1=%.4f"
                  % (name, cut, block["per_class_f1"][name]))
    print("\nDELTA macro-F1 from re-tuning for the serving aggregation: %+.4f"
          % measurements["delta_macro_f1"])

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n",
                                 encoding="utf-8")
    print("wrote %s" % args.report)
    if args.probs_out:
        np.savez(args.probs_out, serving_frame_probs=serving_probs,
                 training_frame_probs=training_probs, targets=targets,
                 classes=np.asarray(TOOL_CLASSES))
        print("wrote %s" % args.probs_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
