"""Where the CNN heads and the detector disagree, and by how much.

WHY DISAGREEMENT IS THE USEFUL SIGNAL. Self-consistency -- sampling one model
several times and trusting it when the samples match -- is the confidence
proxy the VLM branch inherited, and it has a measured failure mode:
`surgvu.evidence_vlm`'s module docstring records the groupmate's temperature
sweep finding the model confidently wrong WITH FULL SELF-AGREEMENT on
case122, case127 and case130 at temperature 0.1. Agreement WITHIN one model
measures determinism, not correctness.

Two models trained on different objectives from different label formats fail
differently. When they agree, the evidence is genuinely stronger; when they
diverge, something is actually hard about the frame. That is a confidence
signal a single model cannot produce at any temperature.

WHAT THIS SIGNAL CANNOT DO, MEASURED. This is not a correctness detector: on
case124 the CNN reports "Bipolar Forceps" and the detector agrees --
bipolar forceps at 0.895 confidence, cadiere forceps at 0.000 -- and the gold
answer is Cadiere. Two independently-trained models agreeing on the wrong
answer. `agreement_record` on that frame reports CONFIDENT AGREEMENT, and
that report is honest: both models really did agree, and the agreement is
still wrong. This module measures whether two models concur, not whether
either of them is right. That is why it was sequenced behind the tasks that
directly change a graded answer (see the plan's ruling R25) and shipped
instead as the uncertainty channel for the VLM layer: a caller may treat LOW
agreement as a reason to look harder, but must never treat HIGH agreement as
a proof of correctness.

THE ASYMMETRY IS DELIBERATE. The detector knows `bipolar dissector` and
`suction irrigator`; the CNN heads do not have those classes at all -- they
were never asked. A detection of one of those two classes is NOT a
disagreement, and counting it as one would make every frame containing a
suction irrigator look uncertain for no reason.
"""
from .detect import OUT_OF_TAXONOMY

AGREEMENT_VERSION = 1


def agreement_record(tool_probs, tool_thresholds, yolo_record,
                     yolo_conf_floor=0.25):
    """Compare CNN presence calls against detector presence calls.

    `tool_probs` and `tool_thresholds` are name-keyed maps over the 12-class
    taxonomy. `yolo_record` is the block from `detect.detections_to_record`.

    `tool_agreement` is the Jaccard similarity of the two presence sets, with
    the empty-vs-empty case defined as 1.0: both models saying "nothing here"
    is agreement, not an undefined ratio. A 0.0 there would flag every quiet
    frame as maximally uncertain.
    """
    cnn = {name for name, prob in tool_probs.items()
           if prob >= tool_thresholds.get(name, 1.0)}
    yolo = {name for name, conf in (yolo_record.get("max_conf") or {}).items()
            if conf >= yolo_conf_floor and name not in OUT_OF_TAXONOMY}

    both = sorted(cnn & yolo)
    cnn_only = sorted(cnn - yolo)
    yolo_only = sorted(yolo - cnn)

    union = cnn | yolo
    agreement = 1.0 if not union else len(cnn & yolo) / len(union)

    # The single widest divergence, named so a prompt or a log can quote it
    # instead of a ratio. Ranked by how far past its own bar the lone model
    # went: a class the CNN calls at 0.95 against a 0.5 threshold is a
    # stronger disagreement than one it calls at 0.51.
    candidates = []
    for name in cnn_only:
        margin = tool_probs[name] - tool_thresholds.get(name, 0.5)
        candidates.append((margin, name, "cnn_only"))
    for name in yolo_only:
        margin = yolo_record["max_conf"][name] - yolo_conf_floor
        candidates.append((margin, name, "yolo_only"))
    top = max(candidates)[1:] if candidates else None

    return {
        "version": AGREEMENT_VERSION,
        "tool_agreement": float(agreement),
        "both_present": both,
        "cnn_only": cnn_only,
        "yolo_only": yolo_only,
        "top_disagreement": list(top) if top else None,
    }
