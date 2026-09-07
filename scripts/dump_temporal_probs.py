"""Honest clip-level probabilities for ANY v4 temporal checkpoint.

WHY NOT dump_clip_probs.py. That script builds its model with
`build_video_model`, which knows only the two Kinetics backbones, and it slices
clips at three fixed offsets inside a single burst. Both assumptions are wrong
for the v4 arms: TSM and I3D are ResNet-50s carrying our surgical weights, and
the multi-burst pool holds four bursts spread across the window rather than one
at its centre. Pointing the old script at a TSM checkpoint would fail on the
backbone name -- or worse, if it did not, it would apply Kinetics normalisation
to weights fitted on [0, 1].

EVERYTHING COMES FROM THE CHECKPOINT. Mechanism, backbone, frame count, image
size, layout, burst count and normalisation are all read from `meta` and none
of them can be overridden by a flag. An arm evaluated at a different
resolution, or with Kinetics statistics it was not trained with, scores badly
for a reason that looks exactly like a bad architecture -- and that mistake
would land on the one question this whole programme exists to answer.

THE ARMS IT DUMPS, and why they differ by layout:

  contiguous  one clip per BURST -- burst0..burstN -- plus their mean and max.
              This is the aggregation the centre-only pool could not do: four
              predictions covering four parts of the labelled window instead of
              one prediction from a 0.53 s sliver.

  spread      a single clip already covering the window, so there is nothing
              to aggregate across. Dumped as one arm, honestly labelled,
              rather than padded out with near-duplicates to look richer.

The centre-only dense pool is a one-burst pool, so a contiguous checkpoint
trained on it yields exactly one arm here and the comparison against the
multi-burst arms is like for like on the same windows.

SCORED THE SAME WAY AS EVERY OTHER ARM: per-class thresholds tuned on one case
fold and scored on the other, both directions, averaged. The 2D ResNet-50's
0.7802 is the number to beat.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.dataset import encode_tools, shard_paths_for_split   # noqa: E402
from surgvu.descriptions import (description_accuracy,           # noqa: E402
                                 load_corpus)
from surgvu.extract import read_shard                            # noqa: E402
from surgvu.holdout import (case_folds, honest_macro_f1,         # noqa: E402
                            unmeasurable_classes)
from surgvu.metrics import macro_f1                              # noqa: E402
from surgvu.taxonomy import TASK_CLASSES, TOOL_CLASSES           # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MULTI = "/staging/n/nkalthoff/surgvu26/shards_multi"


def load_arm(path, device, aggregate="logits"):
    """The model a v4 checkpoint describes, built exactly as it was trained."""
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    meta = payload["meta"]
    if not meta.get("temporal"):
        raise ValueError("%s is not a temporal checkpoint" % path)

    # THE HEAD COMES FROM THE CHECKPOINT TOO. A task checkpoint has 8 outputs
    # and a softmax; scoring it as 12 sigmoid tool probabilities would compare
    # it against the wrong labels entirely. Older tool checkpoints predate the
    # `head` field, so the class list is the fallback -- and the mismatch still
    # raises rather than being coerced.
    head = meta.get("head")
    if head is None:
        head = "task" if list(meta.get("classes", [])) == list(TASK_CLASSES) \
            else "tools"
    expected = TOOL_CLASSES if head == "tools" else TASK_CLASSES
    if list(meta.get("classes", [])) != list(expected):
        raise ValueError("%s claims head %r but was trained on %r"
                         % (path, head, meta.get("classes")))
    mechanism = meta.get("mechanism", "video")
    outputs = len(expected)
    if mechanism == "video":
        from surgvu.models import build_video_model
        model = build_video_model(outputs, meta["backbone"], pretrained=False)
    elif mechanism == "tsm":
        from surgvu.temporal import build_tsm_model
        # `meta.get("fold_div") or 8` would be a bug, and was one: fold_div=0
        # is the NO-SHIFT CONTROL and 0 is falsy, so the control was rebuilt
        # WITH the shift. Both dumps then scored 0.7108 to four decimals --
        # identical, because TSM adds no parameters, so the two checkpoints
        # hold the same weights and differed only in the flag that was being
        # ignored. An arm that silently becomes a different arm is the worst
        # kind of bug here: it produces a plausible number for the wrong model.
        fold = meta.get("fold_div")
        model, _ = build_tsm_model(outputs, checkpoint=None,
                                   backbone=meta["backbone"],
                                   segments=int(meta["frames_per_window"]),
                                   fold_div=8 if fold is None else int(fold),
                                   per_frame=(aggregate == "probs"))
    elif mechanism == "residual":
        # BUILT DIRECTLY rather than through build_residual_model, which
        # requires the 2D checkpoint it is correcting. Here the arm's own
        # state dict supplies every weight, including the trunk's, so loading
        # the 2D file first would only be overwritten -- and would silently
        # succeed if the arm had been trained from a DIFFERENT base, hiding
        # exactly the mismatch that cost 0.048 on the conversions.
        from surgvu.models import build_model
        from surgvu.temporal import ResidualTemporal

        per_burst = meta.get("residual_frames_per_burst")
        if not per_burst:
            raise ValueError(
                "%s is a residual arm with no residual_frames_per_burst. The "
                "clip splits at burst boundaries, so guessing 3 would split "
                "an arm trained at 4 in the wrong places and score a model on "
                "motion it never saw." % path)
        per_burst = int(per_burst)
        clip_frames = int(meta["frames_per_window"])
        if clip_frames % per_burst:
            raise ValueError(
                "%s trained on %d frames, which is not a whole number of %d-"
                "frame bursts" % (path, clip_frames, per_burst))
        tap = meta.get("residual_tap") or "layer4"
        # WHETHER THE SECOND BRANCH EXISTS COMES FROM THE CHECKPOINT. Build it
        # without and the strict load_state_dict raises on the missing
        # sequence.* and beta keys -- loud, but only after a GPU slot has been
        # waited for and the shards opened. Build it WITH when the arm was
        # trained without and the load raises the other way. Neither is a
        # silent wrong answer, which is the standard this file holds, but both
        # are avoidable by reading the flag the trainer already records.
        model = ResidualTemporal(
            build_model(outputs, meta["backbone"], pretrained=False),
            clip_frames // per_burst, per_burst, outputs, tap=tap,
            hidden=int(meta.get("residual_hidden") or 256),
            per_frame=(aggregate == "probs"),
            sequence=bool(meta.get("residual_sequence")))
    elif mechanism == "i3d":
        from surgvu.temporal import build_i3d_model
        extent = meta.get("i3d_temporal")
        model, _ = build_i3d_model(outputs, checkpoint=None,
                                   backbone=meta["backbone"],
                                   temporal=3 if extent is None else int(extent))
    else:
        raise ValueError("unknown mechanism %r in %s" % (mechanism, path))

    model.load_state_dict(payload["state_dict"])
    meta = dict(meta, head=head)
    return model.to(device).eval(), meta


def clip_indices(depth, frames, layout, bursts, sampler="legacy"):
    """(name, frame indices) per arm, matching how the arm was TRAINED.

    Mirrors `ShardTemporal._starts` with its shuffle off. Kept as its own
    function rather than reusing the dataset because the dataset yields one
    clip at a time for training and this needs all of them at once, labelled.
    """
    if layout == "spread":
        # THE SAMPLER IS VERSIONED, and it has to be. ShardTemporal's spread
        # layout was `i * step`, which on a 30-frame window asked for 16 frames
        # and returned the first sixteen. That is fixed -- but every arm
        # trained before the fix learned on the old picks, and evaluating them
        # with the new ones measures a model on input it never saw. On the
        # 32-frame pool the two differ by an offset of 2 frames (0,4,..28
        # against 2,6,..30): small, and exactly the kind of "surely that is
        # close enough" that has been wrong repeatedly tonight.
        #
        # So checkpoints written after the fix carry spread_sampler, and
        # anything without it is scored the way it was trained.
        if sampler == "bin_centres":
            from surgvu.frames import sample_frame_indices
            return [("spread", list(sample_frame_indices(depth, frames)))]
        step = max(1, depth // frames)
        return [("spread",
                 [min(depth - 1, i * step) for i in range(frames)])]
    if layout == "bursts":
        # ONE clip holding whole bursts in time order -- the residual arm reads
        # the boundaries out of it, so this must be `ShardTemporal._starts`
        # frame for frame and not merely "close". Subsetting picks bin centres
        # over the burst indices, the same as the loader, so a model trained on
        # 8 of 16 bursts is scored on the same 8.
        from surgvu.frames import sample_frame_indices
        per_burst = max(1, depth // max(1, bursts))
        if frames % per_burst:
            raise ValueError(
                "%d frames is not a whole number of this pool's %d-frame "
                "bursts" % (frames, per_burst))
        take = min(max(1, bursts), max(1, frames // per_burst))
        picks = []
        for burst in sample_frame_indices(max(1, bursts), take):
            base = burst * per_burst
            picks.extend(range(base, base + per_burst))
        return [("bursts", picks)]
    per_burst = max(1, depth // max(1, bursts))
    take = min(frames, per_burst)
    out = []
    for burst in range(max(1, bursts)):
        base = burst * per_burst
        start = base + (per_burst - take) // 2
        picks = list(range(start, start + take))
        while len(picks) < frames:            # short burst: repeat the last
            picks.append(picks[-1])
        out.append(("burst%d" % burst, picks))
    return out


def freeze_bursts(clips, frames_per_burst):
    """Every burst becomes its own centre frame, repeated. The MOTION control.

    WHY THIS EXISTS. `MotionBranch` reads differences between consecutive
    frames, and on a static burst those differences are exactly zero -- so its
    output collapses to a constant driven by its biases. `alpha * constant` is
    a per-class offset added to every logit, which is threshold recalibration
    wearing a temporal costume. It can move a score with ZERO temporal
    information in it.

    That is mostly harmless on the tools head, where per-class thresholds are
    re-tuned on a held-out fold and absorb a constant. It is NOT harmless on
    the task head, which is an argmax over a softmax: a learned constant offset
    changes predictions outright, and the task head is exactly where the
    temporal hypothesis predicts a gain. So the decomposition has to be
    measurable rather than assumed:

        alpha=0 base     the 2D model
        static control   base + alpha * (bias-only correction)
        the real arm     base + alpha * (motion correction)

    motion  = real - static      recalibration = static - base

    The frames are replaced AFTER selection, so the control scores the same
    windows, the same bursts and the same centre frames as the arm it controls
    -- only the motion inside each burst is destroyed.
    """
    if frames_per_burst < 2:
        raise ValueError("a burst of %d frame(s) has no motion to freeze"
                         % frames_per_burst)
    frozen = clips.copy()
    depth = clips.shape[1]
    if depth % frames_per_burst:
        raise ValueError(
            "clip holds %d frames, which is not a whole number of %d-frame "
            "bursts -- freezing would straddle burst boundaries and destroy "
            "the wrong thing" % (depth, frames_per_burst))
    for start in range(0, depth, frames_per_burst):
        centre = start + frames_per_burst // 2
        frozen[:, start:start + frames_per_burst] = clips[:, centre:centre + 1]
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shards", default=MULTI)
    parser.add_argument("--splits", default=str(REPO / "config" / "splits_v2.json"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--descriptions",
                        default=str(REPO / "config" / "descriptions.yaml"))
    parser.add_argument("--aggregate", choices=("logits", "probs"),
                        default="logits",
                        help="logits: average logits then squash, which is how "
                             "TSM arms were TRAINED. probs: squash per frame "
                             "then average, which is what the 2D reference "
                             "path does. They are not the same number.")
    parser.add_argument("--static-control", action="store_true",
                        help="residual arms: replace every burst with its own "
                             "CENTRE FRAME repeated, so the motion branch sees "
                             "exactly zero differences. Isolates how much of "
                             "an arm's gain is motion and how much is a learned "
                             "per-class offset.")
    parser.add_argument("--max-shards", type=int, default=0)
    args = parser.parse_args(argv)

    import torch

    from surgvu.train import prepare_clip_batch

    device = args.device
    if device in (None, "", "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, meta = load_arm(args.checkpoint, device, args.aggregate)
    head = meta["head"]
    classes = TOOL_CLASSES if head == "tools" else TASK_CLASSES
    corpus = load_corpus(args.descriptions) if head == "task" else None
    frames = int(meta["frames_per_window"])
    image_size = int(meta["image_size"])
    layout = meta.get("layout", "contiguous")
    bursts = int(meta.get("bursts") or 1)
    mean = std = None
    if meta.get("normalisation") == "kinetics":
        from surgvu.models import VIDEO_MEAN, VIDEO_STD
        mean, std = VIDEO_MEAN, VIDEO_STD

    static_per_burst = 0
    if args.static_control:
        if meta.get("mechanism") != "residual":
            raise SystemExit(
                "--static-control only means something for a residual arm: it "
                "exists to separate that arm's motion branch from its biases. "
                "This checkpoint is mechanism=%r." % meta.get("mechanism"))
        static_per_burst = int(meta.get("residual_frames_per_burst") or 0)
        if static_per_burst < 2:
            raise SystemExit(
                "residual_frames_per_burst is %r, so there is no motion in a "
                "burst to freeze." % meta.get("residual_frames_per_burst"))
        print("STATIC CONTROL: every burst replaced by its centre frame, so "
              "the motion branch sees zero differences. Any difference from "
              "the alpha=0 base is a learned per-class offset, NOT motion.",
              flush=True)
    print("alpha recorded in this checkpoint: %s   beta: %s"
          % (meta.get("alpha", "n/a (not a residual arm)"),
             meta.get("beta", "n/a (no sequence branch)")), flush=True)

    print("device=%s mechanism=%s backbone=%s %d frames @ %dpx layout=%s "
          "bursts=%d norm=%s"
          % (device, meta.get("mechanism"), meta["backbone"], frames,
             image_size, layout, bursts, meta.get("normalisation")), flush=True)
    print("trained from %s; checkpoint reports %.4f at epoch %s, which is a max "
          "over epochs with self-tuned cuts and is NOT what this computes."
          % (meta.get("initialised_from"), meta.get("macro_f1", float("nan")),
             meta.get("epochs")), flush=True)

    shards = shard_paths_for_split(args.shards, args.splits, args.split)
    if args.max_shards:
        # SHARDS ARE PER (CASE, PART), SO N SHARDS CAN BE ONE CASE. The honest
        # protocol tunes thresholds on one case fold and scores on the other,
        # and case_folds refuses -- correctly -- to build disjoint folds from a
        # single case. Taking the first 2 shards picked case_009_part1 and
        # case_009_part2 and died forty seconds from the end of the run, after
        # a GPU slot had been waited for and every window had been forwarded.
        # So the cap is on CASES first and shards second: keep taking shards
        # until at least two cases are represented.
        kept, cases_seen = [], set()
        for path in shards:
            if len(kept) >= args.max_shards and len(cases_seen) >= 2:
                break
            kept.append(path)
            cases_seen.add(Path(path).name.rsplit("_part", 1)[0])
        if len(cases_seen) < 2:
            raise SystemExit(
                "split %r offers only %d case(s) (%s), and the honest protocol "
                "needs two to form disjoint folds. This is a property of the "
                "split, not of --max-shards."
                % (args.split, len(cases_seen), sorted(cases_seen)))
        if len(kept) > args.max_shards:
            print("SMOKE TEST: %d shards, not %d -- extended to cover %d cases "
                  "so the folds can be disjoint."
                  % (len(kept), args.max_shards, len(cases_seen)))
        else:
            print("SMOKE TEST: %d shards over %d cases. Not a result."
                  % (len(kept), len(cases_seen)))
        shards = kept
    print("%d shards in split %r" % (len(shards), args.split), flush=True)

    probs, targets, cases, depths = {}, [], [], set()
    started = time.time()
    for index, path in enumerate(shards, start=1):
        shard_frames, rows = read_shard(path)
        for w in range(len(rows)):
            depth = len(shard_frames[w])
            depths.add(int(depth))
            plan = clip_indices(depth, frames, layout, bursts,
                                meta.get("spread_sampler", "legacy"))
            clips = np.stack([np.stack([shard_frames[w][i] for i in picks])
                              for _, picks in plan])
            if args.static_control:
                clips = freeze_bursts(clips, static_per_burst)
            with torch.no_grad():
                batch = prepare_clip_batch(clips, device, image_size, mean, std)
                logits = model(batch)
                # sigmoid for multilabel tools, softmax for the multiclass
                # task head. Applying the wrong one does not raise: it just
                # produces plausible numbers that mean nothing.
                if args.aggregate == "probs":
                    # (clips, frames, classes) -> squash per frame, then mean.
                    per = (torch.sigmoid(logits) if head == "tools"
                           else torch.softmax(logits, dim=2))
                    out = per.mean(dim=1)
                else:
                    out = (torch.sigmoid(logits) if head == "tools"
                           else torch.softmax(logits, dim=1))
                out = out.float().cpu().numpy()
            for slot, (name, _) in enumerate(plan):
                probs.setdefault(name, []).append(out[slot])

            row = rows[w]
            if head == "tools":
                targets.append(encode_tools(row["tools"]))
            else:
                label = str(row.get("task", "")).strip().lower()
                if label not in classes:
                    raise SystemExit("window %d of %s carries task %r, which "
                                     "is not one of the %d TASK_CLASSES"
                                     % (w, Path(path).name, label,
                                        len(classes)))
                targets.append(list(classes).index(label))
            cases.append(str(row.get("case",
                                     Path(path).name.rsplit("_part", 1)[0])))
        print("shard %d/%d %s windows=%d elapsed=%ds"
              % (index, len(shards), Path(path).name, len(rows),
                 time.time() - started), flush=True)

    if head == "tools":
        target = np.stack(targets).astype(np.float32)
    else:
        target = np.array(targets, dtype=np.int64)
    cases = np.array(cases)
    stacked = {name: np.stack(rows_).astype(np.float32)
               for name, rows_ in probs.items()}
    order = list(stacked)
    if len(order) > 1:
        cube = np.stack([stacked[name] for name in order])
        stacked["meanall"] = cube.mean(axis=0)
        stacked["maxall"] = cube.max(axis=0)

    print("\nwindows %d | depth %s | arms %s"
          % (len(target), sorted(depths), list(stacked)), flush=True)

    eye = np.eye(len(classes), dtype=np.float32)
    fold_a, fold_b = case_folds(cases, target if head == "tools"
                                else eye[target])
    print("folds: %d / %d windows" % (len(fold_a), len(fold_b)))
    results = {}

    if head == "tools":
        skipped = unmeasurable_classes(target, fold_a, fold_b, list(classes))
        if skipped:
            print("structurally unmeasurable: %s" % ", ".join(skipped))
        print("\n%-9s %8s %8s %8s" % ("arm", "honest", "measur.", "self"))
        for name in stacked:
            scored = honest_macro_f1(target, stacked[name], fold_a, fold_b)
            results[name] = scored
            print("%-9s %8.4f %8.4f %8.4f"
                  % (name, scored["honest"], scored["honest_measurable"],
                     scored["self_tuned"]))
        best = max(results, key=lambda k: results[k]["honest"])
        print("\nHONEST CLIP-LEVEL macro-F1: %.4f (%s)"
              % (results[best]["honest"], best))
        # "SAME WINDOWS AND FOLDS" WAS TRUE AND MISLEADING. 0.7802 is a
        # hardcoded literal measured on the SPARSE pool, and this dump may be
        # running on shards_multi16, whose burst centres sit at the same
        # FRACTION of each window but up to 0.438 s away from the sparse
        # pool's 1 fps grid. Same windows, same labels, same folds --
        # different FRAMES, which the old wording quietly denied.
        #
        # It matters because the gap is not noise. Measured on the alpha=0
        # baseline: tools 0.7673 against 0.7802, delta -0.0129, while the TASK
        # head reproduced to -0.0004 through the identical code path. The
        # asymmetry is macro-F1 over twelve classes whose support runs from 29
        # (stapler) to 2308 -- one flipped window in the rarest class moves
        # macro-F1 by 0.0029, so four or five of them are the whole delta.
        # Threshold and probability perturbation were both ruled out: +-0.01
        # on the cuts swings 0.0026, and sigma=0.02 noise on the probabilities
        # moves nothing.
        print("2D ResNet-50 on the SPARSE pool: 0.7802. Delta %+.4f."
              % (results[best]["honest"] - 0.7802))
        print("  Same windows and folds, DIFFERENT frames if this ran on a "
              "burst pool. For an arm trained on that pool the base is this "
              "dump's own alpha=0 number, not 0.7802.")
    else:
        # NOTHING IS TUNED for a multiclass head -- the prediction is an argmax
        # -- so there is no honest/self-tuned distinction to draw here. The
        # per-fold accuracies are printed instead, because a single pooled
        # number hides how much of it rides on which cases landed where.
        skipped = []
        print("\n%-9s %9s %9s %9s   %9s %9s"
              % ("arm", "acc", "macroF1", "desc_acc", "acc(A)", "acc(B)"))
        for name in stacked:
            pred = stacked[name].argmax(axis=1)
            results[name] = {
                "accuracy": float((pred == target).mean()),
                "macro_f1": float(macro_f1(eye[target], eye[pred])),
                "description_accuracy": float(
                    description_accuracy(target, pred, corpus)),
                "fold_a_accuracy": float((pred[fold_a] == target[fold_a]).mean()),
                "fold_b_accuracy": float((pred[fold_b] == target[fold_b]).mean()),
                "honest": float(description_accuracy(target, pred, corpus)),
            }
            r = results[name]
            print("%-9s %9.4f %9.4f %9.4f   %9.4f %9.4f"
                  % (name, r["accuracy"], r["macro_f1"],
                     r["description_accuracy"], r["fold_a_accuracy"],
                     r["fold_b_accuracy"]))
        best = max(results, key=lambda k: results[k]["description_accuracy"])
        top = results[best]
        print("\nCLIP-LEVEL, arm %s: accuracy %.4f, description accuracy %.4f"
              % (best, top["accuracy"], top["description_accuracy"]))
        print("2D task head, same windows: accuracy 0.9456, description 0.9581. "
              "Delta %+.4f accuracy, %+.4f description."
              % (top["accuracy"] - 0.9456,
                 top["description_accuracy"] - 0.9581))
    if meta.get("mechanism") in ("tsm", "i3d"):
        print("This arm carries the 2D model's own weights, depth and "
              "preprocessing, so depth, pretraining and resolution are NOT "
              "confounds here. What differs is the temporal mechanism.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prefix = "tools" if head == "tools" else "task"
    np.savez_compressed(
        str(out), cases=cases, arms=np.array(list(stacked)),
        **{"%s_target" % prefix: target,
           "%s_classes" % prefix: np.array(list(classes))},
        **{"%s_%s" % (prefix, name): value for name, value in stacked.items()})
    Path(str(out) + ".json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        # THE POOL IS PART OF THE RESULT, not a detail of how it was produced.
        # The same checkpoint scored on the centre-only pool and on the
        # multi-burst pool differs by ~0.013, and without this field the two
        # dumps are indistinguishable inside the JSON -- scripts/v4_report.py
        # was guessing from the checkpoint FILENAME and labelling multi-pool
        # rows as dense.
        "shards": str(args.shards),
        "split": args.split,
        "meta": {
            k: v for k, v in meta.items() if k != "per_class_f1"},
        "head": head,
        "windows": int(len(target)), "unmeasurable_classes": skipped,
        # The two heads produce different metrics, so the JSON carries whatever
        # the head actually computed rather than a fixed shape. Forcing them
        # into one schema is what made the smoke test fail with a KeyError on
        # 'honest_measurable' -- better than inventing a value for it, which is
        # what a .get(..., 0.0) would have quietly done.
        "arms": {k: (dict(v, per_class=dict(zip(classes, v["per_class"])))
                     if head == "tools" else dict(v))
                 for k, v in results.items()},
        "two_d_reference": (0.7802 if head == "tools"
                            else {"accuracy": 0.9456, "description": 0.9581}),
    }, indent=2, default=str), encoding="utf-8")
    print("\nwrote %s and %s.json" % (out, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
