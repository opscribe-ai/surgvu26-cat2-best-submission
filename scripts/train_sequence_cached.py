"""Train the across-burst branch on CACHED trunk features. Minutes, on CPU.

WHAT THIS REPLACES. `train_temporal.py --mechanism residual --sequence` decodes
JPEG and runs a FROZEN ResNet-50 over every window on every epoch: measured,
1.6 hours per epoch at 24 frames and 3.1 at 48, for a network whose weights
cannot change. `scripts/cache_trunk_features.py` does that once and writes what
the sequence branch actually reads -- 16 pooled vectors and 16 per-centre 2D
logits per window, 1.61 GB for the whole pool. This trains on that.

Two consequences worth stating separately, because the second is the bigger
one:

  Epochs become seconds, so the branch can be swept rather than run once.

  Training stops needing a GPU AT ALL. A 0.5M-parameter 1D convolution over
  16x2048 is a CPU job, and CPU slots are plentiful where GPU slots are not --
  counted live during this work, 0 GPU slots were willing to run the arm while
  14,604 CPU slots were free. The cache decouples the experiment from the
  scarcest resource in the pool.

WHAT IT WRITES IS A NORMAL RESIDUAL CHECKPOINT. The trained branch and beta are
merged into a full `ResidualTemporal` whose trunk comes from the shipped 2D
weights, and the metadata matches what `train_temporal.py` writes. So
`dump_temporal_probs.py` scores this exactly as it scores any other arm, with
no second evaluation path to keep in agreement -- which is the failure mode
that cost 0.048 the first time round.

THE INVARIANT SURVIVES. beta starts at zero, so the merged checkpoint at
initialisation is the 2D model, and `alpha` stays zero throughout because the
local branch is not trained here and has nothing cached to train on.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.descriptions import (description_accuracy,             # noqa: E402
                                 load_corpus)
from surgvu.holdout import case_folds                              # noqa: E402
from surgvu.metrics import (macro_f1, multiclass_accuracy,         # noqa: E402
                            tune_thresholds)
from surgvu.taxonomy import TASK_CLASSES, TOOL_CLASSES             # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TOOLS_2D = "/staging/n/nkalthoff/surgvu26/models/tools_resnet50_long.pt"
TASK_2D = "/staging/n/nkalthoff/surgvu26/models/task_resnet50_long.pt"


def load_cache(path):
    blob = np.load(path, allow_pickle=False)
    meta = json.loads(str(blob["meta"]))
    return blob, meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="1e-3, not the 1e-4 train_temporal uses. Nothing "
                             "pretrained is being disturbed here: the only "
                             "trainable parts are a randomly-initialised "
                             "branch and a zero gate.")
    # REGULARISATION KNOBS, because the first real run overfit hard: with
    # hidden=256 and lr 1e-3 the training loss fell from 0.083 to 0.009 by
    # epoch 5 on 18,412 windows while validation sat at its base. A branch
    # that memorises before it generalises produces a null that is about the
    # CONFIGURATION rather than about the 30 s timescale, and the two are
    # worth separating -- which is affordable now only because a run is two
    # minutes rather than six hours.
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW default; raise it to fight memorisation")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="applied to the pooled features before the branch")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--descriptions", default=str(REPO / "config" / "descriptions.yaml"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import torch
    from surgvu.models import build_model
    from surgvu.temporal import ResidualTemporal
    from surgvu.train import save_checkpoint, seed_everything

    seed_everything(args.seed)
    train, tmeta = load_cache(args.train_cache)
    val, vmeta = load_cache(args.val_cache)

    # THE TWO CACHES MUST DESCRIBE THE SAME MODEL. A train cache from one
    # checkpoint and a val cache from another would train a branch to correct
    # logits it will never see, and the score would be a plausible number
    # about nothing.
    for key in ("checkpoint", "head", "tap", "bursts", "image_size"):
        if tmeta.get(key) != vmeta.get(key):
            raise SystemExit(
                "train and val caches disagree on %r: %r vs %r"
                % (key, tmeta.get(key), vmeta.get(key)))

    head = tmeta["head"]
    classes = TOOL_CLASSES if head == "tools" else TASK_CLASSES
    bursts = int(tmeta["bursts"])
    print("head=%s bursts=%d tap=%s | train %d windows, val %d windows"
          % (head, bursts, tmeta["tap"], train["features"].shape[0],
             val["features"].shape[0]), flush=True)
    print("base checkpoint: %s" % tmeta["checkpoint"], flush=True)

    def tensors(blob):
        feats = torch.from_numpy(blob["features"].astype(np.float32))
        logits = torch.from_numpy(blob["logits"].astype(np.float32))
        target = (torch.from_numpy(blob["tools"]) if head == "tools"
                  else torch.from_numpy(blob["task"]))
        return feats, logits, target

    xtr, ltr, ytr = tensors(train)
    xva, lva, yva = tensors(val)

    from surgvu.temporal import SequenceBranch
    channels = xtr.shape[-1]
    branch = SequenceBranch(channels, bursts, len(classes),
                            hidden=args.hidden).to(args.device)
    beta = torch.nn.Parameter(torch.zeros(1, device=args.device))
    params = list(branch.parameters()) + [beta]
    print("trainable: %d parameters, beta starts at 0"
          % sum(p.numel() for p in params), flush=True)

    loss_fn = (torch.nn.BCEWithLogitsLoss() if head == "tools"
               else torch.nn.CrossEntropyLoss())
    optimizer = torch.optim.AdamW(params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    corpus = load_corpus(args.descriptions) if head == "task" else None
    select = "macro_f1" if head == "tools" else "description"

    def per_burst_logits(feats, logits):
        """(B, bursts, classes) -- corrected, NOT yet aggregated.

        FUNCTIONAL dropout keyed on `branch.training`, deliberately. The first
        version held a standalone nn.Dropout and a comment claiming
        branch.eval() would disable it -- false, because a module that is not
        a submodule of `branch` does not follow its mode. Dropout would have
        stayed ACTIVE through every evaluation, adding noise to the very
        number each epoch is selected on and to the alpha=0 base. Keying off
        branch.training cannot drift that way.
        """
        if args.dropout > 0:
            feats = torch.nn.functional.dropout(
                feats, p=args.dropout, training=branch.training)
        return logits + beta * branch(feats)

    def forward(feats, logits):
        """Mean LOGITS, for the loss only.

        BCEWithLogitsLoss and CrossEntropyLoss both want logits, and averaging
        them before the loss is what train_temporal.py does, so the two
        trainers optimise the same objective. Evaluation does something else
        on purpose -- see `evaluate`.
        """
        return per_burst_logits(feats, logits).mean(dim=1)

    # HONEST EPOCH SELECTION. Picking the best of N epochs on val and then
    # REPORTING that same val number is optimistic by however much the pick
    # exploited val's noise -- and with 40 epochs and a 1.1M-parameter branch
    # that bias is easily worth several thousandths, against an effect size of
    # ~0.009 and a run-to-run noise floor of 0.012. It could manufacture the
    # entire result.
    #
    # So val is split into two CASE folds -- the same split
    # dump_temporal_probs uses for thresholds -- and the headline is the mean
    # of (choose the epoch on A, score it on B) and (choose on B, score on A).
    # Both directions, so neither fold is privileged. The naive
    # best-on-all-val number is still printed beside it, because the gap
    # between them IS the selection bias and is worth seeing.
    eye_t = np.eye(len(classes), dtype=np.float32)
    fold_targets = (yva.numpy() if head == "tools"
                    else eye_t[yva.numpy().astype(np.int64)])
    fold_a, fold_b = case_folds(val["cases"], fold_targets)
    print("val folds: %d / %d windows, split by case"
          % (len(fold_a), len(fold_b)), flush=True)

    def score_subset(probs, index):
        """The selection metric on a subset of val windows."""
        if head == "tools":
            truth = yva.numpy()[index]
            cuts = tune_thresholds(truth, probs[index])
            return macro_f1(truth, (probs[index] >= cuts).astype(np.float32))
        truth = yva.numpy().astype(np.int64)[index]
        pred = probs[index].argmax(axis=1)
        return description_accuracy(truth, pred, corpus)

    def val_probs():
        """Aggregated val probabilities under the CURRENT branch weights."""
        branch.eval()
        with torch.no_grad():
            raw = per_burst_logits(xva, lva)
            if head == "tools":
                return torch.sigmoid(raw).mean(dim=1).numpy()
            return torch.softmax(raw, dim=2).mean(dim=1).numpy()

    def evaluate():
        """Mean PROBABILITIES, because that is how the score is computed.

        branch.eval() below sets branch.training False, which is what
        per_burst_logits keys its dropout off -- so evaluation is
        deterministic.

        THE v4 AGGREGATION BUG, REINTRODUCED BY ME AND CAUGHT BY THE SMOKE.
        This function used to squash the MEAN LOGITS, and reported the alpha=0
        base as accuracy 0.9415 / description 0.9538 when the dump of the same
        checkpoint on the same windows said 0.9454 / 0.9577. The cached logits
        were verified identical to the dump's to 1.8e-07, so the 0.0039 gap
        was entirely the aggregation: softmax(mean(logits)) is not
        mean(softmax(logits)).

        That mismatch cost 0.0145 and most of a night the first time this
        project met it, and the fix then was --aggregate probs in
        dump_temporal_probs. A trainer that selects epochs by a different
        aggregation than the scorer uses would pick the wrong epoch and hand
        it to a dump that then disagrees with it.
        """
        branch.eval()
        with torch.no_grad():
            raw = per_burst_logits(xva, lva)
            if head == "tools":
                # TUNED THRESHOLDS, not 0.5. train_temporal.py selects on
                # macro-F1 with per-class cuts from tune_thresholds, and the
                # 0.7802 reference is computed that way; selecting here on a
                # flat 0.5 would rank epochs by a different quantity than the
                # one this arm is eventually judged on. The honest number
                # still comes from dump_temporal_probs, which tunes on one
                # case fold and scores on the other -- this only has to agree
                # about WHICH EPOCH is best.
                probs = torch.sigmoid(raw).mean(dim=1).numpy()
                truth = yva.numpy()
                cuts = tune_thresholds(truth, probs)
                score = macro_f1(truth,
                                 (probs >= cuts).astype(np.float32))
                return {"macro_f1": score, "selection": score}
            probs = torch.softmax(raw, dim=2).mean(dim=1).numpy()
            truth = yva.numpy().astype(np.int64)
            pred = probs.argmax(axis=1)
            eye = np.eye(len(classes), dtype=np.float32)
            described = description_accuracy(truth, pred, corpus)
            return {"accuracy": multiclass_accuracy(truth, probs),
                    "macro_f1": macro_f1(eye[truth], eye[pred]),
                    "description_accuracy": described,
                    "selection": described}

    # THE BASE, BEFORE ANY TRAINING. beta is zero, so this is the 2D model's
    # own score on these windows -- the number the branch has to beat, printed
    # first so a "gain" can never be measured against a remembered figure.
    base = evaluate()
    base_probs = val_probs()
    print("BASE (beta=0, this IS the 2D model): %s"
          % {k: round(v, 4) for k, v in base.items()}, flush=True)

    best = -1.0
    best_state = None
    per_fold = []
    order = np.arange(xtr.shape[0])
    rng = np.random.default_rng(args.seed)
    started = time.time()
    for epoch in range(args.epochs):
        branch.train()
        rng.shuffle(order)
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            optimizer.zero_grad()
            out = forward(xtr[idx], ltr[idx])
            loss = loss_fn(out, ytr[idx])
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)
        stats = evaluate()
        probs = val_probs()
        per_fold.append((score_subset(probs, fold_a),
                         score_subset(probs, fold_b)))
        print("epoch %d  train_loss %.4f  beta %+.4f  %s  foldA %.4f foldB %.4f"
              % (epoch, total / len(order), float(beta),
                 {k: round(v, 4) for k, v in stats.items()},
                 per_fold[-1][0], per_fold[-1][1]), flush=True)
        if stats["selection"] > best:
            best = stats["selection"]
            best_state = ({k: v.detach().clone()
                           for k, v in branch.state_dict().items()},
                          beta.detach().clone(), dict(stats), epoch)
    print("trained %d epochs in %.1fs" % (args.epochs, time.time() - started))

    if best_state is None:
        raise SystemExit("no epoch produced a usable checkpoint")
    branch_state, beta_value, stats, epoch = best_state
    gain = best - base["selection"]

    # The honest number: choose the epoch on one fold, score it on the other,
    # both ways. base_fold is the same protocol applied to beta=0.
    fold_scores = np.array(per_fold)                       # (epochs, 2)
    honest = 0.5 * (fold_scores[fold_scores[:, 0].argmax(), 1]
                    + fold_scores[fold_scores[:, 1].argmax(), 0])
    base_probs_a = score_subset(base_probs, fold_a)
    base_probs_b = score_subset(base_probs, fold_b)
    honest_base = 0.5 * (base_probs_a + base_probs_b)
    honest_gain = honest - honest_base
    print("\nHONEST (epoch chosen on one case fold, scored on the other, both "
          "directions): %.4f | base %.4f | gain %+.4f"
          % (honest, honest_base, honest_gain))
    print("naive best-on-all-val: %.4f | gain %+.4f  -- the gap %+.4f IS the "
          "selection bias" % (best, gain, best - honest))
    print("\nBEST %s %.4f at epoch %d | base %.4f | gain %+.4f | beta %+.4f"
          % (select, best, epoch, base["selection"], gain, float(beta_value)))
    if gain <= 0:
        print("The branch did not beat its own base. That is a RESULT: the 30 s "
              "context added nothing this head could use.")

    # MERGE INTO A REAL RESIDUAL CHECKPOINT so dump_temporal_probs scores this
    # exactly as it scores every other arm. One evaluation path, not two.
    backbone = build_model(len(classes), "resnet50", pretrained=False)
    payload = torch.load(tmeta["checkpoint"], map_location="cpu",
                         weights_only=False)
    backbone.load_state_dict(payload["state_dict"])
    model = ResidualTemporal(backbone, bursts,
                             int(tmeta["frames_per_burst"]), len(classes),
                             tap=tmeta["tap"], hidden=args.hidden,
                             sequence=True)
    model.sequence.load_state_dict(branch_state)
    with torch.no_grad():
        model.beta.copy_(beta_value)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.out, model, {
        "macro_f1": stats.get("macro_f1", 0.0),
        "selection_value": best, "selected_by": select,
        "base_selection": base["selection"], "gain_over_base": gain,
        "honest_selection": float(honest), "honest_base": float(honest_base),
        "honest_gain": float(honest_gain),
        "selection_bias": float(best - honest),
        "classes": list(classes), "head": head,
        "backbone": "resnet50", "mechanism": "residual",
        "initialised_from": tmeta["checkpoint"],
        "epochs": epoch + 1,
        "clip_length": bursts * int(tmeta["frames_per_burst"]),
        "frames_per_window": bursts * int(tmeta["frames_per_burst"]),
        "layout": "bursts", "spread_sampler": "bin_centres",
        "bursts": bursts, "clips_per_window": 1,
        "image_size": int(tmeta["image_size"]),
        "temporal": True, "fold_div": None, "i3d_temporal": None,
        "residual_tap": tmeta["tap"], "residual_hidden": args.hidden,
        "residual_frames_per_burst": int(tmeta["frames_per_burst"]),
        "residual_sequence": True,
        "alpha": 0.0, "beta": float(beta_value),
        "lr": args.lr, "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "frozen_bn": True, "normalisation": "unit",
        "trained_from_cache": str(args.train_cache),
        **{k: v for k, v in stats.items() if k != "selection"},
    })
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
