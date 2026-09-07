"""Metrics, implemented directly rather than pulled from sklearn.

sklearn is not in the training container and is not worth adding for four
functions. More importantly, `f1_score(..., zero_division=...)` defaults bite
here: a rare class the model never predicts must score 0 and drag macro-F1
down, not vanish into a NaN or be silently skipped.
"""
import numpy as np


def per_class_f1(y_true, y_pred):
    """F1 per column. A class with no true positives scores 0.0, never NaN.
    A class absent from both truth and predictions also scores 0.0 to drag macro-F1
    down on validation splits that omit rare classes, not inflate it."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}")
    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1 - y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1 - y_pred)).sum(axis=0)
    denominator = 2 * tp + fp + fn
    return np.where(denominator > 0, 2 * tp / np.maximum(denominator, 1e-12), 0.0)


def macro_f1(y_true, y_pred):
    return float(per_class_f1(y_true, y_pred).mean())


def tune_thresholds(y_true, probs, grid=None):
    """Per-class threshold maximising that class's F1 on the given data.

    Tuned on VALIDATION and then frozen. Tuning on train would pick cuts that
    fit noise the model already memorised.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)
    if y_true.shape != probs.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} != probs {probs.shape}")
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.01, dtype=np.float32)

    thresholds = np.empty(probs.shape[1], dtype=np.float32)
    for c in range(probs.shape[1]):
        best, best_threshold = -1.0, 0.5
        for threshold in grid:
            pred = (probs[:, c] >= threshold).astype(np.float32)
            score = per_class_f1(y_true[:, c:c + 1], pred[:, None])[0]
            if score > best:
                best, best_threshold = score, float(threshold)
        thresholds[c] = best_threshold
    return thresholds


def multiclass_accuracy(y_true_idx, logits):
    y_true_idx = np.asarray(y_true_idx)
    logits = np.asarray(logits)
    if len(y_true_idx) != len(logits):
        raise ValueError(f"Length mismatch: y_true_idx len={len(y_true_idx)} != logits len={len(logits)}")
    return float((logits.argmax(axis=1) == y_true_idx).mean())
