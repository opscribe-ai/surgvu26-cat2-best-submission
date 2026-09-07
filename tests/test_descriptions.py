import pytest

from surgvu.descriptions import (
    build_corpus, DescriptionRetriever, description_accuracy, load_corpus,
    write_corpus,
)
from surgvu.labels import CaseLabels
from surgvu.taxonomy import TASK_CLASSES
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "case_test"
REAL_CORPUS = Path(__file__).parents[1] / "config" / "descriptions.yaml"

OTHER = TASK_CLASSES.index("other")
RETRACTION = TASK_CLASSES.index("retraction and collision avoidance")
SUTURING = TASK_CLASSES.index("suturing")
RECTAL = TASK_CLASSES.index("rectal artery/vein")
UTERINE = TASK_CLASSES.index("uterine horn")


def test_build_corpus_groups_by_task():
    cases = {"case_test": CaseLabels.from_dir(FIXTURE)}
    corpus = build_corpus(cases)
    assert "suturing" in corpus
    assert "uterine horn" in corpus
    assert corpus["suturing"][0].startswith("Excess bleeding")


def test_retriever_returns_modal_description():
    corpus = {"suturing": ["most common", "rare one"]}
    r = DescriptionRetriever(corpus)
    assert r.retrieve("suturing") == "most common"


def test_build_corpus_orders_by_true_frequency():
    # The fixture gives "skills application" a rare description (appears
    # once, inserted first in the CSV) and a common description (appears
    # three times, inserted after it). If build_corpus ever used insertion
    # order instead of frequency -- e.g. Counter.keys() instead of
    # most_common() -- the rare one would land in slot [0] since it was
    # seen first. This exercises the real end-to-end path, not a hand-built
    # dict, so it would also catch the sort being reversed.
    cases = {"case_test": CaseLabels.from_dir(FIXTURE)}
    corpus = build_corpus(cases)
    entries = corpus["skills application"]
    assert entries[0] == "Common description that appears three times"
    assert entries[1] == "Rare description that appears only once"

    retriever = DescriptionRetriever(corpus)
    assert retriever.retrieve("skills application") == "Common description that appears three times"


def test_retriever_unknown_task_returns_empty_string():
    r = DescriptionRetriever({"suturing": ["x"]})
    assert r.retrieve("range of motion") == ""


def test_corpus_roundtrips_through_yaml(tmp_path):
    corpus = {"suturing": ["a", "b"], "other": ["c"]}
    path = tmp_path / "descriptions.yaml"
    write_corpus(corpus, path)
    assert load_corpus(path) == corpus


def test_description_accuracy_credits_a_class_confusion_that_retrieves_the_same_text():
    # The whole point of this metric. Every prediction here is the WRONG
    # CLASS, so plain class accuracy is 0.0 -- but both classes retrieve the
    # same string, which is all that reaches the answer, so the retrieval is
    # perfect. The lists deliberately differ in length: comparing lists or
    # sets of descriptions instead of the retrieved string would score 0.0,
    # and that is exactly how the real corpus is shaped.
    corpus = {"other": ["shared text"],
              "suturing": ["shared text", "a rarer one", "and another"]}
    assert description_accuracy([OTHER, SUTURING], [SUTURING, OTHER], corpus) == 1.0


def test_description_accuracy_rejects_a_confusion_between_different_texts():
    corpus = {"other": ["one text"], "suturing": ["a different text"]}
    assert description_accuracy([OTHER], [SUTURING], corpus) == 0.0


def test_description_accuracy_does_not_credit_two_empty_retrievals():
    # Neither class is in the corpus, so both retrieve "". Equal strings, but
    # equal to nothing: crediting this would score a model as correct for
    # having produced no answer at all, and would silently inflate the
    # metric on any corpus with a missing class.
    assert description_accuracy([OTHER], [SUTURING], {}) == 0.0


def test_description_accuracy_exceeds_task_accuracy_on_the_real_corpus():
    # Anchored on the shipped corpus, not a hand-built one, so it also fails
    # if config/descriptions.yaml is ever regenerated in a way that destroys
    # the shared modal description this metric exists to exploit.
    # 'other', 'retraction and collision avoidance' and 'suturing' share a
    # modal description; 'rectal artery/vein' and 'uterine horn' do not.
    corpus = load_corpus(REAL_CORPUS)
    truth = [OTHER, OTHER, RETRACTION, RECTAL]
    pred = [RETRACTION, SUTURING, OTHER, UTERINE]

    task_accuracy = sum(t == p for t, p in zip(truth, pred)) / len(truth)
    assert task_accuracy == 0.0
    assert description_accuracy(truth, pred, corpus) == 0.75


def test_description_accuracy_counts_exact_class_hits():
    # The >= task-accuracy invariant rests on this: an exact class hit is a
    # description hit even when that class has no description at all.
    assert description_accuracy([UTERINE], [UTERINE], {}) == 1.0


def test_description_accuracy_rejects_length_mismatch():
    # zip() would silently truncate and report a plausible number over a
    # prefix of the validation set.
    with pytest.raises(ValueError):
        description_accuracy([OTHER, SUTURING], [OTHER], {})
