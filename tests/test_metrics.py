import numpy as np
import pytest

from surgvu.metrics import (
    macro_f1, multiclass_accuracy, per_class_f1, tune_thresholds,
)


def test_per_class_f1_is_one_for_perfect_predictions():
    y = np.array([[1, 0], [0, 1]], dtype=np.float32)
    assert per_class_f1(y, y).tolist() == [1.0, 1.0]


def test_a_class_never_predicted_scores_zero_not_nan():
    """A rare class the model never fires on must drag macro-F1 down. If it
    silently becomes NaN or is skipped, macro-F1 flatters the model exactly
    where the corpus is weakest -- stapler has 137 windows against cadiere's
    15,600."""
    y = np.array([[1, 1], [1, 0]], dtype=np.float32)
    pred = np.array([[1, 0], [1, 0]], dtype=np.float32)
    scores = per_class_f1(y, pred)
    assert scores[1] == 0.0
    assert not np.isnan(scores).any()


def test_a_class_absent_from_both_truth_and_predictions_scores_zero():
    """The denominator-zero branch: a class the validation split never
    contains and the model never predicts.

    It must score 0.0, not 1.0 and not NaN. Scoring it 1.0 would be the
    worst outcome available -- macro-F1 would rise for classes the model
    never demonstrated any ability on, and the rare classes this project
    cares most about (stapler: 137 of the 27,556 enumerated train windows in
    config/tool_frequency.json, and 119 of the 18,412 that survive into the
    splits_v2 train shards) are exactly the ones liable to be absent from a
    31-case validation split. 24,578, which this comment used to cite, is the
    whole extracted corpus rather than either train split.
    """
    y = np.array([[1, 0]], dtype=np.float32)
    pred = np.array([[1, 0]], dtype=np.float32)
    scores = per_class_f1(y, pred)
    assert scores[0] == 1.0
    assert scores[1] == 0.0
    assert not np.isnan(scores).any()


def test_macro_f1_weights_every_class_equally():
    y = np.array([[1, 0]] * 99 + [[0, 1]], dtype=np.float32)
    pred = np.array([[1, 0]] * 99 + [[0, 0]], dtype=np.float32)
    # class 0 perfect, class 1 never predicted -> (1.0 + 0.0) / 2
    assert macro_f1(y, pred) == pytest.approx(0.5)


def test_tune_thresholds_finds_a_better_cut_than_one_half():
    """A class whose probabilities all sit below 0.5 is invisible at the
    default threshold. Rare classes behave exactly like this."""
    y = np.array([[1], [1], [0], [0]], dtype=np.float32)
    probs = np.array([[0.4], [0.35], [0.1], [0.05]], dtype=np.float32)
    thresholds = tune_thresholds(y, probs)
    assert thresholds[0] < 0.5
    assert macro_f1(y, (probs >= thresholds).astype(np.float32)) == 1.0


def test_tune_thresholds_returns_one_threshold_per_class():
    y = np.zeros((4, 3), dtype=np.float32)
    y[0] = 1
    probs = np.random.RandomState(0).rand(4, 3).astype(np.float32)
    assert tune_thresholds(y, probs).shape == (3,)


def test_multiclass_accuracy_uses_argmax():
    logits = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    assert multiclass_accuracy(np.array([1, 0]), logits) == 1.0
    assert multiclass_accuracy(np.array([0, 0]), logits) == 0.5


def test_metrics_reject_shapes_that_would_broadcast_silently():
    """The dangerous mismatch is not one numpy rejects -- it is one numpy
    ACCEPTS. (4,3) against (1,3) broadcasts to a clean [1.0, 1.0, 1.0]: a
    perfect score, no error, nothing to notice. Shapes numpy already refuses
    need no guard from us.
    """
    with pytest.raises(ValueError):
        per_class_f1(np.ones((4, 3), np.float32), np.ones((1, 3), np.float32))
    with pytest.raises(ValueError):
        per_class_f1(np.ones((4, 3), np.float32), np.ones((4, 1), np.float32))
    with pytest.raises(ValueError):
        tune_thresholds(np.ones((4, 3), np.float32), np.ones((1, 3), np.float32))
    with pytest.raises(ValueError):
        multiclass_accuracy(np.zeros(4, np.int64), np.zeros((1, 3), np.float32))
