"""Does FRAME ORDER carry signal, independent of model capacity?

THE CONFOUND THIS EXISTS TO REMOVE. The 3D runs will come back better or worse
than the 2D ResNet-50, and either way the result is uninterpretable on its own:
those models differ from the 2D path in FOUR ways at once -- temporal
convolution, Kinetics-vs-ImageNet pretraining, 18 layers vs 50, and 112px vs
384. A loss could be motion not helping, or simply a weaker backbone.

So ask the narrow question separately, where it can be isolated. Three arms
over the SAME per-frame probabilities:

  POOLED     the shipped path: an order-invariant aggregator over the frames,
             then tuned thresholds. Every aggregator we have swept -- mean,
             max, q75, q90, trim, topK, noisy-or -- is order-invariant, so
             this is the whole family's ceiling.
  ORDERED    a small GRU reading the frames in order.
  SHUFFLED   the SAME GRU, same capacity, same training budget, on sequences
             whose time axis has been permuted.

ORDERED vs POOLED confounds order with capacity: the GRU has parameters the
aggregator does not, so it can win for reasons that have nothing to do with
time. ORDERED vs SHUFFLED does not -- identical architecture, identical
parameter count, identical optimisation. The ONLY difference is whether the
time axis carries its real ordering.

  ORDERED > SHUFFLED   order carries signal. Temporal modelling is worth it.
  ORDERED ~ SHUFFLED   order carries nothing here, and any gain over POOLED is
                       capacity, not time -- which would predict the 3D runs
                       gain nothing from their temporal dimension either.

WHAT THIS CAN AND CANNOT SEE. The dumps are from the SPARSE 1 fps pool, so
"order" here means 30 seconds of surgical progression, not motion. A null
result therefore does not rule out motion at 67 ms -- it rules out slow
progression, and says the 3D models' advantage would have to come from the
fast timescale the dense shards were built for. Stated plainly because the
temptation to over-read this is exactly why the arm exists.

Honest two-fold throughout: fit on one case fold, score on the other, average.
A sequence model scored on windows it trained on would be a fiction.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.aggregate import AGGREGATORS                       # noqa: E402
from surgvu.frames import sample_frame_indices                 # noqa: E402
from surgvu.holdout import case_folds                          # noqa: E402
from surgvu.metrics import macro_f1, tune_thresholds           # noqa: E402


class SequenceHead(nn.Module):
    """A GRU over per-frame probabilities -> one multi-label logit vector.

    Deliberately small. The question is whether ORDER is usable at all, and a
    large head would answer a different question -- how much capacity helps --
    while making the shuffled control easier to overfit into a tie.
    """

    def __init__(self, n_classes, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_classes, hidden, batch_first=True,
                          bidirectional=True)
        self.head = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):                       # (B, T, C)
        out, _ = self.gru(x)
        return self.head(out.mean(dim=1))       # mean over time, then classify


def fit_and_score(train_x, train_y, score_x, score_y, epochs, lr, seed,
                  device, shuffle_time):
    """Train one head and return its honest macro-F1 on the held-out fold.

    `shuffle_time` permutes the time axis INDEPENDENTLY PER SAMPLE and per
    epoch, on both the training and the scoring split. Permuting once, or with
    one shared permutation, would leave a consistent ordering the GRU could
    still learn -- the destruction has to be total or the control leaks.
    """
    torch.manual_seed(seed)
    model = SequenceHead(train_x.shape[2]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    tx = torch.tensor(train_x, dtype=torch.float32, device=device)
    ty = torch.tensor(train_y, dtype=torch.float32, device=device)
    sx = torch.tensor(score_x, dtype=torch.float32, device=device)

    generator = torch.Generator(device="cpu").manual_seed(seed)

    def maybe_shuffle(batch):
        if not shuffle_time:
            return batch
        idx = torch.argsort(torch.rand(batch.shape[:2], generator=generator),
                            dim=1).to(batch.device)
        return torch.gather(batch, 1, idx.unsqueeze(-1).expand_as(batch))

    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(tx), generator=generator).to(device)
        for start in range(0, len(tx), 256):
            batch = order[start:start + 256]
            logits = model(maybe_shuffle(tx[batch]))
            loss = loss_fn(logits, ty[batch])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(maybe_shuffle(sx))).cpu().numpy()
        train_probs = torch.sigmoid(model(maybe_shuffle(tx))).cpu().numpy()

    # Thresholds come from the TRAINING fold, never the scoring one -- the
    # same rule the pooled arm follows, so the comparison is like for like.
    cuts = tune_thresholds(train_y, train_probs)
    return macro_f1(score_y, (probs >= cuts).astype(np.float32))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probs", required=True, nargs="+")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--aggregation", default="mean", choices=sorted(AGGREGATORS))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seeds", type=int, default=3,
                        help="a single seed cannot distinguish a real gap "
                             "from optimisation noise")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sources = [np.load(p, allow_pickle=False) for p in args.probs]
    data = sources[0]
    target = data["tools_target"].astype(np.float32)
    depth = int(data["depth"][0])
    picks = sample_frame_indices(depth, args.frames)
    stack = np.mean([s["tools_id"] for s in sources], axis=0)
    seq = stack[:, picks, :].astype(np.float32)          # (W, T, C)

    fold_a, fold_b = case_folds(data["cases"], target)
    print("device %s | windows %d | sequence %s" % (device, len(seq), seq.shape[1:]))
    print("shard frames are the SPARSE 1 fps pool: 'order' here is 30 s of "
          "progression, NOT motion.\n")

    # POOLED -- the order-invariant ceiling, same protocol as everything else.
    pooled = AGGREGATORS[args.aggregation](seq)
    pooled_scores = []
    for tune, score in ((fold_a, fold_b), (fold_b, fold_a)):
        cuts = tune_thresholds(target[tune], pooled[tune])
        pooled_scores.append(
            macro_f1(target[score], (pooled[score] >= cuts).astype(np.float32)))
    pooled_f1 = float(np.mean(pooled_scores))

    results = {}
    for name, shuffle_time in (("ordered", False), ("shuffled", True)):
        runs = []
        for seed in range(args.seeds):
            fold_scores = [
                fit_and_score(seq[tune], target[tune], seq[score], target[score],
                              args.epochs, args.lr, seed, device, shuffle_time)
                for tune, score in ((fold_a, fold_b), (fold_b, fold_a))]
            runs.append(float(np.mean(fold_scores)))
        results[name] = runs
        print("%-9s %s  mean %.4f  sd %.4f"
              % (name, " ".join("%.4f" % r for r in runs),
                 float(np.mean(runs)), float(np.std(runs))))

    ordered = float(np.mean(results["ordered"]))
    shuffled = float(np.mean(results["shuffled"]))
    spread = float(np.std(results["ordered"] + results["shuffled"]))

    print()
    print("pooled (%s, order-invariant)  %.4f" % (args.aggregation, pooled_f1))
    print("ordered GRU                    %.4f" % ordered)
    print("shuffled GRU                   %.4f" % shuffled)
    print("ordered - shuffled             %+.4f   <- THE ORDER EFFECT" % (ordered - shuffled))
    print("ordered - pooled               %+.4f   (order AND capacity)" % (ordered - pooled_f1))
    print()
    if abs(ordered - shuffled) <= spread:
        print("VERDICT: order effect is within seed noise (sd %.4f). At this "
              "timescale ordering carries nothing the pooled path is missing, "
              "and any GRU gain is capacity." % spread)
    elif ordered > shuffled:
        print("VERDICT: ordering helps beyond capacity. Temporal modelling has "
              "something to work with.")
    else:
        print("VERDICT: shuffling HELPED, which means the ordered arm is "
              "overfitting a spurious pattern. Treat as a null.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "pooled": pooled_f1, "ordered": results["ordered"],
            "shuffled": results["shuffled"],
            "ordered_mean": ordered, "shuffled_mean": shuffled,
            "order_effect": ordered - shuffled, "seed_sd": spread,
            "aggregation": args.aggregation, "frames": args.frames,
            "probs": list(args.probs),
        }, indent=2), encoding="utf-8")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
