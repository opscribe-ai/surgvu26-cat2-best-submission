"""Re-tuning the tool thresholds for the aggregation serving actually uses.

The measurement this script produces is only worth anything if it aggregates
the way the container aggregates. Two of the ways it could quietly fail to do
that are shape errors that numpy would not complain about:

  * averaging a (windows, frames, classes) block along the wrong axis gives a
    perfectly well-formed matrix of nonsense;
  * scoring windows against per-frame truth, or the reverse, gives a
    perfectly well-formed macro-F1 of nonsense.

Both are pinned here. The third property -- that this script's forward pass
reproduces `predict_window` exactly -- cannot be pinned with a fixture, so the
script proves it at runtime against the real function and refuses to report if
it does not hold.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tune_serving_thresholds import (                              # noqa: E402
    aggregate_all, build_report, evaluate, flatten_per_frame,
)
from surgvu.taxonomy import TOOL_CLASSES                           # noqa: E402


# ------------------------------------------------------------- aggregation

def test_windows_are_reduced_across_frames_not_across_windows():
    """Two windows, four frames each. The right axis gives one row per window;
    the wrong one averages the two windows together and still returns a
    matrix of the right dtype."""
    probs = np.zeros((2, 4, len(TOOL_CLASSES)), dtype=np.float32)
    probs[0] = 0.2
    probs[1] = 0.8

    clip = aggregate_all(probs)

    assert clip.shape == (2, len(TOOL_CLASSES))
    assert clip[0].tolist() == pytest.approx([0.2] * len(TOOL_CLASSES))
    assert clip[1].tolist() == pytest.approx([0.8] * len(TOOL_CLASSES))


def test_the_mean_is_over_the_frames_of_one_window():
    probs = np.zeros((1, 4, 2), dtype=np.float32)
    probs[0, :, 0] = [0.0, 0.2, 0.4, 0.6]
    probs[0, :, 1] = [1.0, 1.0, 1.0, 0.0]

    assert aggregate_all(probs)[0].tolist() == pytest.approx([0.3, 0.75])


def test_an_already_aggregated_block_is_refused():
    """(W, C) would be averaged across WINDOWS and return one row standing in
    for the whole validation split."""
    with pytest.raises(ValueError, match="already aggregated"):
        aggregate_all(np.zeros((4, len(TOOL_CLASSES)), dtype=np.float32))


# ----------------------------------------------------- the per-frame baseline

def test_every_frame_of_a_window_carries_that_window_s_label():
    """Installation state does not change inside a window by construction, so
    `run_epoch` scores each frame against the window's own label. Reproducing
    its number means repeating the labels the same way."""
    probs = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    targets = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    flat_probs, flat_targets = flatten_per_frame(probs, targets)

    assert flat_probs.shape == (6, 2)
    assert flat_targets.tolist() == [[1, 0], [1, 0], [1, 0],
                                     [0, 1], [0, 1], [0, 1]]


def test_the_frames_stay_in_window_order():
    """`np.repeat`, not `np.tile`: tiling would pair every frame with the
    wrong window's label and still produce a plausible macro-F1."""
    probs = np.zeros((2, 2, 1), dtype=np.float32)
    probs[0] = 0.1
    probs[1] = 0.9

    flat_probs, _ = flatten_per_frame(probs, np.zeros((2, 1), dtype=np.float32))

    assert flat_probs.reshape(-1).tolist() == pytest.approx([0.1, 0.1, 0.9, 0.9])


def test_targets_that_do_not_match_the_probabilities_are_refused():
    with pytest.raises(ValueError, match="targets"):
        flatten_per_frame(np.zeros((2, 3, 2), dtype=np.float32),
                          np.zeros((3, 2), dtype=np.float32))


# ---------------------------------------------------------------- scoring

def test_a_class_is_positive_at_exactly_its_threshold():
    """`>=`, matching `train_tools.py`'s selection and `tools_present`. Under
    a strict `>` a threshold would be scored differently here than the
    container applies it."""
    truth = np.zeros((1, len(TOOL_CLASSES)), dtype=np.float32)
    probs = np.zeros((1, len(TOOL_CLASSES)), dtype=np.float32)
    truth[0, 0] = 1.0
    probs[0, 0] = 0.31
    thresholds = [0.31] + [0.9] * (len(TOOL_CLASSES) - 1)

    assert evaluate(truth, probs, thresholds)["per_class_f1"][
        TOOL_CLASSES[0]] == 1.0


def test_each_class_is_scored_against_its_own_threshold():
    """One vector, twelve different cuts. A single shared cut both drops the
    rare classes and admits the common ones."""
    truth = np.array([[1.0, 0.0]], dtype=np.float32)
    probs = np.array([[0.10, 0.60]], dtype=np.float32)

    scored = evaluate(truth, probs, [0.05, 0.90])

    # 0.10 clears its 0.05 cut and is right; 0.60 does not clear its 0.90 cut
    # and is right to be absent. Under one shared cut anywhere between them,
    # exactly one of those two is wrong.
    assert scored["macro_f1"] == pytest.approx(0.5)


def test_a_short_threshold_vector_is_refused():
    """It would broadcast against the wrong columns rather than fail."""
    with pytest.raises(ValueError, match="positional"):
        evaluate(np.zeros((2, 3), dtype=np.float32),
                 np.zeros((2, 3), dtype=np.float32), [0.5, 0.5])


def test_truth_and_probabilities_must_be_the_same_shape():
    with pytest.raises(ValueError, match="disagree in shape"):
        evaluate(np.zeros((2, 3), dtype=np.float32),
                 np.zeros((4, 3), dtype=np.float32), [0.5] * 3)


# --------------------------------------------------------------- the report

def test_the_report_states_both_vectors_and_is_json_writable():
    """It is about to be embedded in a serving config. A threshold vector with
    no statement of what it was tuned against is indistinguishable from a
    typo."""
    shipped = [0.5] * len(TOOL_CLASSES)
    retuned = [0.25] * len(TOOL_CLASSES)

    report = build_report(TOOL_CLASSES, shipped, retuned,
                          {"delta_macro_f1": 0.0196},
                          {"checkpoint_sha256": "ab" * 32})

    assert report["checkpoint_thresholds"] == shipped
    assert report["serving_thresholds"] == retuned
    assert report["serving_thresholds_by_class"]["stapler"] == 0.25
    assert json.loads(json.dumps(report))["provenance"]["checkpoint_sha256"]


def test_a_report_whose_vectors_do_not_cover_the_taxonomy_is_refused():
    with pytest.raises(ValueError, match="positional"):
        build_report(TOOL_CLASSES, [0.5] * len(TOOL_CLASSES),
                     [0.25] * (len(TOOL_CLASSES) - 1), {}, {})
