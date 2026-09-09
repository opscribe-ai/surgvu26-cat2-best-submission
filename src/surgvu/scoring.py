# src/surgvu/scoring.py
"""Wrapper around the official SurgVU 2026 Category 2 metric.

The evaluation container is public, so this reproduces it rather than
approximating it. Primary metric: BERTScore-F1, roberta-large, rescaled with
baseline, MAX over the five references, meaned across cases.

Note the asymmetry, copied deliberately from the organizers' evaluate.py:
BLEU and ROUGE are computed on normalized text, but BERTScore and NLI receive
the RAW candidate and references. Casing and punctuation therefore affect the
metric that ranks us.
"""
import string
from statistics import mean

_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text):
    """Lowercase and strip punctuation -- the organizers' BLEU/ROUGE path."""
    return (text or "").translate(_PUNCT).lower().strip()


class Scorer:
    """Lazily loads roberta-large; reuse one instance across many calls."""

    def __init__(self, device=None, model_type="roberta-large"):
        self._device = device
        self._model_type = model_type
        self._bert = None

    def _bert_scorer(self):
        if self._bert is None:
            from bert_score import BERTScorer
            self._bert = BERTScorer(
                model_type=self._model_type,
                lang="en",
                rescale_with_baseline=True,
                device=self._device,
            )
        return self._bert

    def score_one(self, candidate, references):
        """Max BERTScore-F1 of candidate against every reference."""
        if not references:
            return {"bertscore_f1": 0.0}
        scorer = self._bert_scorer()
        expanded = [candidate] * len(references)
        _p, _r, f1 = scorer.score(expanded, list(references))
        return {"bertscore_f1": float(f1.max().item())}

    def score_many(self, pairs):
        """pairs: iterable of (case_id, candidate, references)."""
        results = []
        for case_id, candidate, references in pairs:
            row = self.score_one(candidate, references)
            row["case_id"] = case_id
            results.append(row)
        aggregate = mean(r["bertscore_f1"] for r in results) if results else 0.0
        return {"results": results, "aggregates": {"bertscore_f1": aggregate}}
