"""Large vs Mega needle driver: the one distinction nothing else can make.

WHAT IT IS WORTH. On the 11-case sample, "large needle driver" appears in 3
questions (27%) and we score 1 of 3. A polar answer is 1.0000 right and
0.7015 wrong, so each of those is 0.2985 -- and the family is the entire
difference between the two gold answers.

WHY IT ABSTAINS. Forcing a binary call on an ambiguous clip trades a certain
0.7015 for a coin flip. That trade is only worth taking when the head is
genuinely better than chance on THIS clip, and the cutoff is where that
judgement is written down. A cutoff of 0.5 never abstains, which is why this
module refuses one: it would be a decision layer that makes no decision.

THE CUTOFF IS FITTED, NOT CHOSEN. scripts/train_variant.py writes it to
config alongside the validation accuracy it achieved. A number typed into
source here would be a guess with the authority of code.
"""
VARIANT_VERSION = 1

FAMILIES = ("large", "mega")


def variant_record(probs, cutoff):
    """Turn a two-class distribution into a decision, or an abstention."""
    if not 0.5 < float(cutoff) <= 1.0:
        raise ValueError(
            "cutoff must be above 0.5 and at most 1.0, got %r. At or below "
            "chance the head never abstains, which removes the only reason "
            "this layer exists." % (cutoff,))
    p_large = float(probs.get("large", 0.0))
    p_mega = float(probs.get("mega", 0.0))
    total = p_large + p_mega
    if abs(total - 1.0) > 1e-3:
        raise ValueError(
            "large+mega must be a distribution, got %.4f. Unnormalised "
            "scores compared against a probability cutoff would abstain or "
            "decide for arithmetic reasons." % (total,))

    top = "large" if p_large >= p_mega else "mega"
    confidence = max(p_large, p_mega)
    decided = confidence >= float(cutoff)
    return {
        "version": VARIANT_VERSION,
        "family": top if decided else None,
        "p_large": p_large,
        "p_mega": p_mega,
        "cutoff": float(cutoff),
        "decided": bool(decided),
    }


class VariantHead:
    """ResNet-18 two-class head over needle-driver crops.

    Torch is imported inside the methods so `variant_record` stays importable
    without it -- that is where the decision logic lives and where the tests
    that matter run.

    CROPS WHEN AVAILABLE, WHOLE FRAME OTHERWISE. The detector's needle-driver
    box is what makes this tractable: a size distinction between two otherwise
    identical instruments is a fine-grained appearance problem, and a
    whole-frame classifier has to find the tool before it can compare it. When
    no box is available the whole frame is used rather than skipping the
    clip -- a degraded measurement beats none, and the abstention path exists
    precisely to catch the cases where that degradation matters.
    """

    def __init__(self, weights, cutoff, device="cpu", size=224):
        self.weights = str(weights)
        self.cutoff = float(cutoff)
        self.device = device
        self.size = int(size)
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        import torch
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(FAMILIES))
        state = torch.load(self.weights, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        model.eval().to(self.device)
        self._model = model
        return model

    def predict(self, frames, boxes=None):
        """Mean two-class distribution over the supplied frames."""
        import cv2
        import numpy as np
        import torch

        model = self._load()
        crops = []
        for index, frame in enumerate(np.asarray(frames)):
            box = (boxes or {}).get(index)
            if box is not None:
                x1, y1, x2, y2 = (int(round(v)) for v in box)
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                if x2 - x1 >= 8 and y2 - y1 >= 8:
                    frame = frame[y1:y2, x1:x2]
            crops.append(cv2.resize(frame, (self.size, self.size)))
        if not crops:
            raise ValueError("no frames to classify")

        batch = torch.from_numpy(
            np.stack(crops)[:, :, :, ::-1].copy()).permute(0, 3, 1, 2).float()
        batch = (batch / 255.0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1).mean(dim=0).tolist()
        return variant_record(dict(zip(FAMILIES, probs)), self.cutoff)
