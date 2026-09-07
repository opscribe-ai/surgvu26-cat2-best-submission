import pytest
from surgvu.taxonomy import (
    TOOL_CLASSES, TASK_CLASSES, normalize_task, normalize_tool, tool_index,
)


def test_twelve_tool_classes_in_fixed_order():
    assert len(TOOL_CLASSES) == 12
    assert TOOL_CLASSES[0] == "bipolar forceps"
    assert TOOL_CLASSES == tuple(sorted(TOOL_CLASSES))


def test_eight_task_classes():
    assert len(TASK_CLASSES) == 8
    assert "suturing" in TASK_CLASSES
    assert "other" in TASK_CLASSES


def test_normalize_task_lowercases():
    assert normalize_task("Suturing") == "suturing"
    assert normalize_task("suturing") == "suturing"
    assert normalize_task("Rectal artery/vein") == "rectal artery/vein"


def test_normalize_task_rejects_unknown():
    assert normalize_task("not a task") is None
    assert normalize_task("") is None


def test_normalize_tool_passes_in_scope():
    assert normalize_tool("needle driver") == "needle driver"
    assert normalize_tool("Needle Driver") == "needle driver"


def test_normalize_tool_rejects_endoscope():
    # The camera is not a predictable class and must never enter a label.
    assert normalize_tool("nan(camera in)") is None


def test_normalize_tool_rejects_out_of_scope_rare_classes():
    for rare in ["suction irrigator", "synchroseal", "curved scissors",
                 "potts scissors", "tenaculum forceps", "bipolar dissector",
                 "crocodile grasper"]:
        assert normalize_tool(rare) is None, rare


def test_tool_index_is_stable():
    for i, name in enumerate(TOOL_CLASSES):
        assert tool_index(name) == i


def test_tool_index_raises_on_unknown():
    with pytest.raises(KeyError):
        tool_index("suction irrigator")
