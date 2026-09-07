"""Honest validation scoring, shared so two experiments cannot diverge.

Every v2 experiment reports one number: clip-level macro-F1 on the splits_v2
validation split. That number is only comparable across experiments if the
protocol producing it is byte-identical, which is why it lives here instead
of being reimplemented in each script. `sweep_aggregation.py` (experiments
1-3) and `train_head.py` (experiment 6) both call these.

WHY THRESHOLDS ARE TUNED ON ONE FOLD AND SCORED ON ANOTHER
-----------------------------------------------------------
Per-class thresholds are chosen by search over a 0.05-0.95 grid to maximise
each class's F1. Run that search on the same windows you then score and the
result includes however much of the grid fit noise -- and with a rare tail
(tip-up: 157 positive windows in the entire training corpus) there is a lot of
noise available to fit. The self-tuned number is reported alongside precisely
so the gap is visible rather than assumed small.

FOLDS SPLIT BY CASE, NEVER BY WINDOW. Windows within a case are consecutive
30-second neighbours showing the same instruments in the same scene. A
window-level split puts near-duplicates on both sides of the fold and reports
a threshold that generalises to the next half-minute, which is not the claim
anyone wants to make.
"""
import numpy as np

from .metrics import macro_f1, per_class_f1, tune_thresholds


def case_folds(cases, targets=None):
    """Two window-index arrays, split by case.

    Without `targets` this alternates over the sorted case list. With them it
    STRATIFIES, and it should always be given them -- here is why.

    Measured on the splits_v2 validation split (29 cases), the number of
    distinct cases holding each tool class is wildly uneven:

        needle driver              29 cases
        monopolar curved scissors  28
        bipolar forceps            27
        ...
        vessel sealer              12
        permanent cautery hook      9
        force bipolar               7
        stapler                     5
        tip-up fenestrated grasper  1     <-- unsplittable, see below

    A plain alternating split put ALL 68 tip-up windows on one side and none
    on the other. A threshold cannot be tuned for a class with no positives,
    so that class scores 0 in one direction no matter how good the model is --
    which silently caps macro-F1 and, worse, makes improvements to exactly
    that class invisible. Since raising the rare-tail weight is v2 experiment
    5's entire hypothesis, measuring it through a fold that cannot see the
    rare tail would have produced a confident null result about nothing.

    Stratifying fixes the classes that CAN be split. It cannot fix tip-up:
    all 68 of its validation windows come from one case (case_153), so no
    case-level split can place it on both sides. That is a property of the
    data. `unmeasurable_classes` names such classes so a report can say so
    instead of quietly averaging a structural zero into macro-F1.

    The greedy assignment places the most constrained cases first -- those
    holding the rarest class -- because a case carrying the only stapler
    windows has to land well, while a case carrying the ubiquitous needle
    driver can go anywhere.
    """
    cases = np.asarray(cases)
    unique = sorted(set(cases.tolist()))
    if len(unique) < 2:
        raise ValueError(
            "need at least 2 cases to form disjoint folds; got %r. Scoring "
            "would silently fall back to tuning and scoring on the same "
            "windows." % (unique,))

    if targets is None:
        left = set(unique[0::2])
        mask = np.array([case in left for case in cases])
        return np.where(mask)[0], np.where(~mask)[0]

    targets = np.asarray(targets, dtype=np.float32)
    if len(targets) != len(cases):
        raise ValueError("targets %d rows, cases %d" % (len(targets), len(cases)))

    per_case = np.array([targets[cases == c].sum(axis=0) for c in unique])
    case_size = np.array([int((cases == c).sum()) for c in unique])
    splittable = np.array([((per_case > 0)[:, c]).sum() >= 2
                           for c in range(targets.shape[1])])

    def imbalance(assign):
        """Worst per-class log-ratio between folds, plus a size penalty.

        The WORST class is the objective rather than the average, because one
        starved class is what breaks a macro average -- improving an already
        balanced class buys nothing. Classes held by fewer than two cases are
        excluded: no assignment can help them, so letting them into the
        objective would just add a constant and swamp the comparison.
        """
        a = per_case[assign == 0].sum(axis=0)
        b = per_case[assign == 1].sum(axis=0)
        both = splittable & (a + b > 0)
        if not both.any():
            return 10 ** 9
        ratio = np.abs(np.log((a[both] + 1.0) / (b[both] + 1.0))).max()
        sizes = np.array([case_size[assign == 0].sum(), case_size[assign == 1].sum()])
        size_skew = abs(np.log((sizes[0] + 1.0) / (sizes[1] + 1.0)))
        return float(ratio + 0.5 * size_skew)

    # Random restarts beat a greedy pass here: the objective is a max over
    # classes, which greedy cannot see until the last case is placed. A fixed
    # seed keeps the folds reproducible -- a split that changed between two
    # runs would make two experiments incomparable for no reason.
    rng = np.random.default_rng(1789)
    best = np.array([i % 2 for i in range(len(unique))])
    best_score = imbalance(best)
    for _ in range(4000):
        trial = rng.integers(0, 2, size=len(unique))
        if trial.sum() in (0, len(unique)):
            continue
        score = imbalance(trial)
        if score < best_score:
            best, best_score = trial, score

    assignment = {case: int(best[i]) for i, case in enumerate(unique)}
    mask = np.array([assignment[c] == 0 for c in cases])
    return np.where(mask)[0], np.where(~mask)[0]


def unmeasurable_classes(targets, fold_a, fold_b, class_names=None):
    """Indices (or names) of classes with no positives in one of the folds.

    Their F1 is a structural zero in at least one direction, so including
    them in a macro average reports the split rather than the model. A caller
    that averages over them anyway must at least say which they were.
    """
    targets = np.asarray(targets, dtype=np.float32)
    bad = [c for c in range(targets.shape[1])
           if targets[fold_a][:, c].sum() == 0 or targets[fold_b][:, c].sum() == 0]
    if class_names is None:
        return bad
    return [class_names[c] for c in bad]


def honest_macro_f1(target, clip_probs, fold_a, fold_b):
    """Clip-level macro-F1, as a dict. Keys:

        honest             tune on each fold, score the other, average
        self_tuned         tune and score on everything -- the optimistic
                           number, returned so the gap is visible not assumed
        honest_measurable  `honest` over only the classes that have positives
                           in BOTH folds
        per_class          honest per-class F1, averaged over the two
                           directions, in class order

    `honest` and `honest_measurable` differ by exactly the structural zeros
    described in `case_folds`. Report BOTH: the first is comparable against
    the shipped macro-F1, which also averages a zero in for tip-up, and the
    second is the one that actually moves when a model gets better.
    """
    target = np.asarray(target, dtype=np.float32)
    clip_probs = np.asarray(clip_probs, dtype=np.float32)
    if target.shape != clip_probs.shape:
        raise ValueError("targets %r do not match probabilities %r"
                         % (target.shape, clip_probs.shape))
    scores, per_class = [], []
    for tune_idx, score_idx in ((fold_a, fold_b), (fold_b, fold_a)):
        cuts = tune_thresholds(target[tune_idx], clip_probs[tune_idx])
        pred = (clip_probs[score_idx] >= cuts).astype(np.float32)
        scores.append(macro_f1(target[score_idx], pred))
        per_class.append(per_class_f1(target[score_idx], pred))
    self_cuts = tune_thresholds(target, clip_probs)
    self_pred = (clip_probs >= self_cuts).astype(np.float32)

    per_class = np.mean(per_class, axis=0)
    bad = set(unmeasurable_classes(target, fold_a, fold_b))
    keep = [c for c in range(target.shape[1]) if c not in bad]
    return {
        "honest": float(np.mean(scores)),
        "self_tuned": macro_f1(target, self_pred),
        "honest_measurable": float(per_class[keep].mean()) if keep else float("nan"),
        "per_class": per_class.tolist(),
    }
