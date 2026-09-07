"""Tests for logbook-derived Large/Mega needle-driver labels.

These labels are free and there are a lot of them, which makes it especially
important that an unrecognised commercial name is DROPPED rather than
assigned to the larger family. A silent default would put hundreds of
mislabelled frames into training and the head would learn the prior instead
of the appearance.

Column names and time format match the REAL tools.csv (verified against
/staging/groups/bhaskar_opscribe/surgvu/labels_cat2/SURGVU25_train_labels),
not the plan's original draft: the header is install_case_part /
install_case_time / uninstall_case_part / uninstall_case_time / arm /
commercial_toolname / groundtruth_toolname / case, and the two time columns
are 'HH:MM:SS.ffffff' strings, e.g. '00:07:24.796000' == 444.796 seconds.
A fixture that used the plan's original made-up header
(install_time/uninstall_time as bare floats) would let a completely broken
parser -- one that raises ValueError on every real row and is swallowed by a
bare except -- pass every test while intervals_for_case returns [] for all
155 real cases.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_variant_labels import family_of, intervals_for_case  # noqa: E402

_HEADER = ("install_case_part,install_case_time,uninstall_case_part,"
           "uninstall_case_time,arm,commercial_toolname,"
           "groundtruth_toolname,case\n")


def _write(tmp_path, rows):
    path = tmp_path / "tools.csv"
    path.write_text(_HEADER + "".join(rows), encoding="utf-8")
    return path


def test_large_suturecut_is_large_family():
    assert family_of("Large SutureCut Needle Driver") == "large"


def test_plain_large_is_large_family():
    assert family_of("Large Needle Driver") == "large"


def test_mega_suturecut_is_mega_family_regardless_of_case():
    assert family_of("Mega Suturecut Needle Driver") == "mega"
    assert family_of("Mega SutureCut Needle Driver") == "mega"


def test_unknown_name_is_dropped_not_defaulted():
    assert family_of("DeBakey Forceps") is None
    assert family_of("Wristed Needle Driver SingleSite") is None
    assert family_of("") is None
    assert family_of(None) is None


def test_real_hms_string_parses_to_expected_seconds(tmp_path):
    """'00:07:24.796000' -> 444.796. If the parser expects a bare float
    (the plan's original, wrong, assumption) this row's time string raises
    ValueError, is swallowed, and the row silently disappears -- covered
    by asserting the row IS kept with the exact expected seconds."""
    path = _write(tmp_path, [
        "1.0,00:07:24.796000,1.0,00:15:54.496000,USM1,"
        "Large Needle Driver,needle driver,case_000\n",
    ])
    intervals = intervals_for_case(path)
    assert len(intervals) == 1
    assert intervals[0]["start"] == pytest.approx(444.796)
    assert intervals[0]["stop"] == pytest.approx(954.496)
    assert intervals[0]["family"] == "large"
    assert intervals[0]["arm"] == "USM1"
    assert intervals[0]["part"] == "1.0"


def test_interval_carries_the_normalised_part_it_was_measured_against(tmp_path):
    """R28: version 1 dropped the part on the floor even though this
    function already reads install_case_part/uninstall_case_part to
    implement spans_part_boundary. A consumer resolving a video file needs
    the CANONICAL form ('2.0'), not whatever spelling ('2', '02', '2.0')
    happened to be in this row -- normalize_part is what makes '2' here
    match a video filename's 'part_002'."""
    path = _write(tmp_path, [
        "2,00:00:05.000000,2,00:00:15.000000,USM1,"
        "Mega Needle Driver,needle driver,case_000\n",
    ])
    intervals = intervals_for_case(path)
    assert len(intervals) == 1
    assert intervals[0]["part"] == "2.0"


def test_intervals_cover_only_needle_drivers(tmp_path):
    path = _write(tmp_path, [
        "1.0,00:00:10.000000,1.0,00:00:20.000000,USM1,"
        "Large SutureCut Needle Driver,needle driver,case_000\n",
        "1.0,00:00:25.000000,1.0,00:00:35.000000,USM2,"
        "Cadiere Forceps,cadiere forceps,case_000\n",
    ])
    intervals = intervals_for_case(path)
    assert len(intervals) == 1
    assert intervals[0]["family"] == "large"
    assert intervals[0]["start"] == pytest.approx(10.0)


def test_both_families_in_one_case_are_both_kept(tmp_path):
    """137 of 154 needle-driver cases contain both families in the train
    corpus. A per-case prior cannot resolve this."""
    path = _write(tmp_path, [
        "1.0,00:00:10.000000,1.0,00:00:20.000000,USM1,"
        "Large Needle Driver,needle driver,case_000\n",
        "1.0,00:00:30.000000,1.0,00:00:40.000000,USM2,"
        "Mega Needle Driver,needle driver,case_000\n",
    ])
    assert {i["family"] for i in intervals_for_case(path)} == {"large", "mega"}


def test_unparseable_time_drops_the_row(tmp_path):
    path = _write(tmp_path, [
        "1.0,,1.0,00:00:20.000000,USM1,"
        "Large Needle Driver,needle driver,case_000\n",
    ])
    assert intervals_for_case(path) == []


def test_reversed_times_are_dropped(tmp_path):
    """stop must be strictly after start; a non-positive duration is not a
    usable interval no matter how it arose."""
    path = _write(tmp_path, [
        "1.0,00:00:30.000000,1.0,00:00:20.000000,USM1,"
        "Large Needle Driver,needle driver,case_000\n",
    ])
    assert intervals_for_case(path) == []


def test_row_spanning_a_part_boundary_is_dropped(tmp_path):
    """Case timestamps reset at part boundaries (see surgvu.labels), so a
    row whose install and uninstall parts differ is not a single coherent
    time interval and must not be kept."""
    path = _write(tmp_path, [
        "1.0,00:00:10.000000,2.0,00:00:20.000000,USM1,"
        "Large Needle Driver,needle driver,case_000\n",
    ])
    assert intervals_for_case(path) == []


def test_unrecognised_commercial_name_is_dropped_not_defaulted(tmp_path):
    """DeBakey Forceps shows up once in the whole corpus under
    groundtruth_toolname == 'needle driver' -- a data-entry artifact, not a
    Large or Mega variant -- and must not be defaulted to the 62.7% majority
    family."""
    path = _write(tmp_path, [
        "1.0,00:00:10.000000,1.0,00:00:20.000000,USM1,"
        "DeBakey Forceps,needle driver,case_000\n",
    ])
    assert intervals_for_case(path) == []


def test_non_needle_driver_rows_are_ignored(tmp_path):
    """A commercial name that would otherwise resolve to a family must
    still be ignored when groundtruth_toolname is not 'needle driver' --
    e.g. the endoscope's 'nan(camera in)' rows must never leak in."""
    path = _write(tmp_path, [
        "1.0,00:00:10.000000,1.0,00:00:20.000000,USM2,"
        "0° Endoscope,nan(camera in),case_000\n",
    ])
    assert intervals_for_case(path) == []


def test_duplicate_row_is_counted_once(tmp_path):
    row = ("1.0,00:00:10.000000,1.0,00:00:20.000000,USM1,"
           "Large Needle Driver,needle driver,case_000\n")
    path = _write(tmp_path, [row, row])
    assert len(intervals_for_case(path)) == 1


def test_drops_are_tallied_by_reason(tmp_path):
    path = _write(tmp_path, [
        "1.0,00:00:10.000000,1.0,00:00:20.000000,USM1,"
        "Large Needle Driver,needle driver,case_000\n",
        "1.0,00:00:10.000000,1.0,00:00:20.000000,USM2,"
        "0° Endoscope,nan(camera in),case_000\n",
        "1.0,00:00:10.000000,1.0,00:00:20.000000,USM3,"
        "DeBakey Forceps,needle driver,case_000\n",
        "1.0,,1.0,00:00:20.000000,USM4,"
        "Large Needle Driver,needle driver,case_000\n",
    ])
    from collections import Counter
    drops = Counter()
    intervals_for_case(path, drops=drops)
    assert drops["not_needle_driver"] == 1
    assert drops["unrecognised_name"] == 1
    assert drops["unparseable_time"] == 1
