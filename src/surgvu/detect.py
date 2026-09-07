"""YOLOv5 tool detection as a second opinion on the CNN heads.

WHAT THIS ADDS THAT THE CNNs DO NOT. The tool heads are whole-frame
multi-label classifiers: they report that a needle driver is present, not
where or when. A detector reports both, and the "when" is the part this
pipeline has never had -- pooled confidences are a single number for a 30 s
clip, so "needle driver at 3.7 s and 9.4 s but not between" is a statement
the record could not previously make.

WHY IT IS ADDITIVE AND NOT A REPLACEMENT. The CNN path is measured: it earns
0.8766 on the sample against 0.8294 for a blind router. This detector has
never been scored against BERTScore at all. It joins as evidence, and its
DISAGREEMENT with the CNNs is itself a signal (see surgvu/agreement.py) --
arguably the more valuable half, because two independently-wrong models
rarely fail the same way.

FOURTEEN CLASSES, TWELVE ANSWERS. The detector's vocabulary is our twelve
plus `bipolar dissector` and `suction irrigator`. Those two are KEPT in the
record and never emitted as an answer: a confident suction-irrigator
detection constrains what else is in the frame, which is useful even though
it can never be the reply. Mapping happens at answer-formatting time, not
here, so the record stays a faithful account of what was seen.
"""
from .taxonomy import TOOL_CLASSES

DETECT_VERSION = 1

#: Index order from surg_14cls.yaml in the groupmate's yolo_dataset. The
#: ORDER IS THE CONTRACT -- a checkpoint's class indices mean nothing without
#: it, and reordering this silently relabels every detection.
YOLO_CLASSES = (
    "bipolar dissector",             # 0   out of taxonomy
    "bipolar forceps",               # 1
    "cadiere forceps",               # 2
    "clip applier",                  # 3
    "force bipolar",                 # 4
    "grasping retractor",            # 5
    "monopolar curved scissors",     # 6
    "needle driver",                 # 7
    "permanent cautery hook/spatula",# 8
    "prograsp forceps",              # 9
    "stapler",                       # 10
    "suction irrigator",             # 11  out of taxonomy
    "tip-up fenestrated grasper",    # 12
    "vessel sealer",                 # 13
)

#: Present in the detector, absent from the answer taxonomy. Evidence only.
OUT_OF_TAXONOMY = frozenset({"bipolar dissector", "suction irrigator"})

_TOOL_SET = frozenset(TOOL_CLASSES)


def map_to_taxonomy(name):
    """14-class detector name -> 12-class answer name, or None.

    Raises on a name the detector cannot produce. Returning None for an
    unknown string would let a typo in a config silently delete a class's
    detections, and the record would look merely empty rather than wrong.
    """
    if name not in YOLO_CLASSES:
        raise KeyError(
            "%r is not one of the detector's 14 classes. The class list is a "
            "contract with the checkpoint; a name outside it means the "
            "weights and this table disagree." % (name,))
    if name in OUT_OF_TAXONOMY:
        return None
    if name not in _TOOL_SET:
        raise KeyError(
            "%r is in the detector vocabulary but not in TOOL_CLASSES and not "
            "declared out-of-taxonomy. Refusing to guess which it is."
            % (name,))
    return name


def detections_to_record(detections, timestamps):
    """Per-anchor detections -> the record block, time preserved.

    `detections[i]` is the list for anchor i; `timestamps[i]` is that anchor's
    time in seconds within the clip. Every detection is stamped, so downstream
    can ask when a tool appeared and not merely whether.
    """
    if len(detections) != len(timestamps):
        raise ValueError(
            "%d anchors of detections against %d timestamps; a detection "
            "would be stamped with another anchor's time."
            % (len(detections), len(timestamps)))

    per_anchor, by_class, max_conf = [], {}, {}
    for index, (found, when) in enumerate(zip(detections, timestamps)):
        stamped = []
        for item in found:
            entry = {
                "cls": item["cls"],
                "conf": float(item["conf"]),
                "box": [float(v) for v in item["box"]],
                "anchor_idx": index,
                "t_seconds": float(when),
            }
            stamped.append(entry)
            by_class.setdefault(item["cls"], []).append(entry)
            max_conf[item["cls"]] = max(max_conf.get(item["cls"], 0.0),
                                        entry["conf"])
        per_anchor.append(stamped)

    return {
        "version": DETECT_VERSION,
        "classes": list(YOLO_CLASSES),
        "per_anchor": per_anchor,
        "by_class": by_class,
        "max_conf": max_conf,
    }


class Detector:
    """Loads `best.pt` once and runs it over a clip's anchors.

    Torch is imported INSIDE the methods, not at module scope, so the mapping
    and record functions above stay importable on a machine without torch --
    which is where most of their tests run.
    """

    def __init__(self, weights, repo_dir, conf=0.25, iou=0.45, device="cpu"):
        self.weights = str(weights)
        self.repo_dir = str(repo_dir)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self._model = None

    def _ensure_repo_on_path(self):
        """Put the yolov5 checkout on `sys.path`, exactly once.

        BOTH `_load` (imports `models.common`) and `detect` (imports
        `utils.augmentations` / `utils.general`) need this done before their
        OWN `from ... import ...` lines run -- an import resolves against
        `sys.path` at the line that executes it, not against whatever `_load`
        does later. This used to live only inside `_load`, on the assumption
        that `detect` always calls `_load` first; it does not -- `detect`
        imported from `utils` two lines before its own call to `_load()`, so
        on a fresh process the checkout was never on `sys.path` yet and the
        import failed every single time (caught by
        tests/test_detect_weights.py in the container, cluster 9684639; no
        torch-free test calls `detect()` so nothing else could catch it).
        Centralising the check here removes that hidden ordering dependency
        between the two methods instead of relying on both of them calling
        each other in the right order.
        """
        import sys
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

    def _load(self):
        if self._model is not None:
            return self._model
        import torch
        # The local yolov5 checkout, not torch.hub: the container has no
        # internet, and a hub fetch would fail at serving time on a machine
        # nobody can log into.
        self._ensure_repo_on_path()
        from models.common import DetectMultiBackend
        model = DetectMultiBackend(self.weights, device=torch.device(self.device))
        model.eval()
        self._model = model
        return model

    def detect(self, frames, size=640):
        """(N, H, W, 3) uint8 BGR -> list of N detection lists."""
        import numpy as np
        import torch
        # MUST run before the `from utils...` imports directly below: those
        # resolve against sys.path at this point in the method, not against
        # whatever `_load()` does three lines further down. See
        # `_ensure_repo_on_path`'s docstring -- this ordering is the whole
        # fix, do not move it back below the imports.
        self._ensure_repo_on_path()
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_coords

        model = self._load()
        out = []
        for frame in np.asarray(frames):
            # A plain cv2.resize to a square applies two DIFFERENT per-axis
            # ratios on any non-square frame -- 1280x720 (Cat 2) and 640x512
            # (Cat 1) are both the normal case here, not an edge case.
            # scale_coords assumes a single gain plus symmetric padding, i.e.
            # a LETTERBOX resize, and undoes exactly that math. auto=False
            # gives a fixed size x size input rather than one padded only to
            # a stride multiple.
            resized, ratio, pad = letterbox(frame, (size, size), auto=False)
            tensor = torch.from_numpy(
                resized[:, :, ::-1].copy()).permute(2, 0, 1).float()
            tensor = (tensor / 255.0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                raw = model(tensor)
            kept = non_max_suppression(raw, self.conf, self.iou)[0]
            found = []
            if kept is not None and len(kept):
                # Explicit ratio_pad rather than letting scale_coords
                # recompute gain/padding from the two shapes: harder to get
                # wrong later if this method's resize ever changes again.
                boxes = scale_coords(
                    tensor.shape[2:], kept[:, :4].clone(), frame.shape,
                    ratio_pad=(ratio, pad)).round()
                for box, row in zip(boxes.tolist(), kept.tolist()):
                    found.append({"cls": YOLO_CLASSES[int(row[5])],
                                  "conf": float(row[4]),
                                  "box": [float(v) for v in box]})
            out.append(found)
        return out
