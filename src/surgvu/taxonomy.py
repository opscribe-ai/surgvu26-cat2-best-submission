"""Closed vocabularies for SurgVU Category 2.

Only these 12 tool classes appear in the test set. The label tables contain
seven rarer classes plus the endoscope; the challenge states those "will not
be part of the testing set", so they are training signal only and must never
be emitted as a prediction.
"""

TOOL_CLASSES = (
    "bipolar forceps",
    "cadiere forceps",
    "clip applier",
    "force bipolar",
    "grasping retractor",
    "monopolar curved scissors",
    "needle driver",
    "permanent cautery hook/spatula",
    "prograsp forceps",
    "stapler",
    "tip-up fenestrated grasper",
    "vessel sealer",
)

TASK_CLASSES = (
    "other",
    "range of motion",
    "rectal artery/vein",
    "retraction and collision avoidance",
    "skills application",
    "suspensory ligaments",
    "suturing",
    "uterine horn",
)

# Present in the label tables, excluded from the test set.
OUT_OF_SCOPE_TOOLS = frozenset({
    "bipolar dissector",
    "crocodile grasper",
    "curved scissors",
    "potts scissors",
    "suction irrigator",
    "synchroseal",
    "tenaculum forceps",
})

_TOOL_INDEX = {name: i for i, name in enumerate(TOOL_CLASSES)}
_TOOL_SET = frozenset(TOOL_CLASSES)
_TASK_SET = frozenset(TASK_CLASSES)


def normalize_tool(raw):
    """Map a raw groundtruth_toolname to a test-relevant class, or None.

    Returns None for the endoscope (recorded as 'nan(camera in)'), for the
    seven out-of-scope rare classes, and for anything unrecognised.
    """
    if not raw:
        return None
    name = raw.strip().lower()
    if name.startswith("nan"):          # endoscope / camera
        return None
    if name in OUT_OF_SCOPE_TOOLS:
        return None
    return name if name in _TOOL_SET else None


def normalize_task(raw):
    """Map a raw groundtruth_taskname to one of the 8 classes, or None."""
    if not raw:
        return None
    name = raw.strip().lower()
    return name if name in _TASK_SET else None


def tool_index(name):
    """Index of a tool class in TOOL_CLASSES. Raises KeyError if out of scope."""
    return _TOOL_INDEX[name]
