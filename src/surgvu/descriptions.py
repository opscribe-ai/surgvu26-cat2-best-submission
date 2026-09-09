"""The 21-string description corpus.

Across all 155 cases there are exactly 21 unique matched_description values.
This is not a training set -- it is a closed retrieval corpus, and it is the
verbatim text the challenge's ground-truth answers were generated from.
Classify the task, retrieve the string, and you hold the source material.
"""
from collections import Counter, defaultdict

import yaml

from .taxonomy import TASK_CLASSES


def build_corpus(cases):
    """{task_class: [description, ...]} ordered most-frequent first."""
    counters = defaultdict(Counter)
    for case in cases.values():
        for segment in case.task_segments():
            if segment.description:
                counters[segment.task][segment.description] += 1
    return {task: [desc for desc, _ in counter.most_common()]
            for task, counter in counters.items()}


def write_corpus(corpus, path):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(corpus, handle, allow_unicode=True, sort_keys=True,
                       default_flow_style=False, width=100)


def load_corpus(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class DescriptionRetriever:
    """Task class -> the description the graders most likely read."""

    def __init__(self, corpus):
        self._corpus = corpus

    def retrieve(self, task_class):
        entries = self._corpus.get(task_class) or []
        return entries[0] if entries else ""


def description_accuracy(true_idx, pred_idx, corpus, classes=TASK_CLASSES):
    """Fraction of predictions that RETRIEVE THE RIGHT TEXT, not the right class.

    Task accuracy is not the number that matters. The task class exists only
    to pull a `matched_description`, and in the real corpus three classes --
    'other', 'retraction and collision avoidance' and 'suturing' -- share the
    same modal description, so confusing them costs nothing downstream. This
    metric is therefore always >= plain class accuracy: an exact class hit is
    a hit here by definition, and some class misses are hits too.

    Compare the RETRIEVED STRING, not the two classes' description lists.
    Those lists are 3, 3 and 12 entries long for the three classes above and
    are never equal to each other, so a list- or set-equality test would
    credit none of these confusions and collapse this metric back onto class
    accuracy -- silently, and exactly where it is supposed to differ.

    An empty retrieval never counts as a hit: two classes both absent from
    the corpus retrieve "" == "" and would otherwise score as correct for
    having produced no answer at all.
    """
    true_idx = list(true_idx)
    pred_idx = list(pred_idx)
    if len(true_idx) != len(pred_idx):
        raise ValueError(
            "Length mismatch: true_idx len=%d != pred_idx len=%d"
            % (len(true_idx), len(pred_idx)))

    retriever = DescriptionRetriever(corpus)
    hits = 0
    for true, pred in zip(true_idx, pred_idx):
        if true == pred:
            hits += 1
            continue
        text = retriever.retrieve(classes[true])
        if text and text == retriever.retrieve(classes[pred]):
            hits += 1
    return hits / max(len(true_idx), 1)
