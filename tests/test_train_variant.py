"""Tests for the torch-free parts of scripts/train_variant.py.

Everything exercised here -- `family_seconds`, `split_cases`, `sample_points`,
`sweep_cutoff` -- is plain python and numpy, imported without touching torch,
cv2 or surgvu.detect. That mirrors tests/test_variant.py's own discipline:
the decision logic (here, the case-level split and the cutoff fit) is what
correctness depends on, and it has to be checkable on a machine with no GPU,
no container and no video corpus.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_variant import (  # noqa: E402
    build_examples, exclude_graded_cases, family_seconds, load_heldout_ids,
    load_variant_labels, resolve_case_video, sample_points, split_cases,
    sweep_cutoff,
)


# ---------------------------------------------------------------------------
# family_seconds
# ---------------------------------------------------------------------------

def test_family_seconds_sums_durations_by_family():
    intervals = [
        {"start": 0.0, "stop": 10.0, "family": "large", "arm": "USM1"},
        {"start": 20.0, "stop": 25.0, "family": "large", "arm": "USM2"},
        {"start": 30.0, "stop": 33.0, "family": "mega", "arm": "USM3"},
    ]
    assert family_seconds(intervals) == {"large": 15.0, "mega": 3.0}


def test_family_seconds_ignores_unrecognised_family():
    """A family outside {"large", "mega"} must not be silently summed into
    either bucket -- that would be exactly the kind of defaulting
    build_variant_labels.py's own tests guard against one layer up."""
    intervals = [{"start": 0.0, "stop": 10.0, "family": "suturecut",
                 "arm": "USM1"}]
    assert family_seconds(intervals) == {"large": 0.0, "mega": 0.0}


def test_family_seconds_drops_nonpositive_duration():
    intervals = [{"start": 10.0, "stop": 10.0, "family": "large", "arm": "X"}]
    assert family_seconds(intervals) == {"large": 0.0, "mega": 0.0}


# ---------------------------------------------------------------------------
# split_cases
# ---------------------------------------------------------------------------

def _synthetic_cases(n=20, seed=0):
    """`n` cases, each holding both families at varied durations -- close to
    the real corpus's own shape (137 of 152 cases hold both)."""
    rng = np.random.default_rng(seed)
    cases = {}
    for i in range(n):
        large_len = float(rng.uniform(20, 200))
        mega_len = float(rng.uniform(20, 200))
        cases["case_%03d" % i] = [
            {"start": 0.0, "stop": large_len, "family": "large", "arm": "USM1"},
            {"start": 1000.0, "stop": 1000.0 + mega_len, "family": "mega",
             "arm": "USM2"},
        ]
    return cases


def test_split_cases_partitions_every_case_exactly_once():
    cases = _synthetic_cases()
    train_ids, heldout_ids = split_cases(cases, holdout_fraction=0.2, seed=1)
    assert set(train_ids) & set(heldout_ids) == set()
    assert set(train_ids) | set(heldout_ids) == set(cases)


def test_split_cases_keeps_both_families_on_both_sides():
    """The whole point of a case-level split: neither side of the fold is
    starved of a class it needs to be scored on."""
    cases = _synthetic_cases()
    train_ids, heldout_ids = split_cases(cases, holdout_fraction=0.25, seed=2)
    for ids in (train_ids, heldout_ids):
        totals = {"large": 0.0, "mega": 0.0}
        for cid in ids:
            case_totals = family_seconds(cases[cid])
            for family in totals:
                totals[family] += case_totals[family]
        assert totals["large"] > 0
        assert totals["mega"] > 0


def test_split_cases_is_deterministic_for_a_fixed_seed():
    cases = _synthetic_cases()
    first = split_cases(cases, holdout_fraction=0.2, seed=7)
    second = split_cases(cases, holdout_fraction=0.2, seed=7)
    assert first == second


def test_split_cases_raises_when_no_split_keeps_both_families_both_sides():
    """Two cases, each single-family: any non-trivial split puts one family
    entirely on one side. A cutoff fitted against a held-out set missing a
    family would report a coverage number that means nothing for it, so this
    must raise rather than silently return an unbalanced split."""
    cases = {
        "case_a": [{"start": 0.0, "stop": 10.0, "family": "large",
                    "arm": "USM1"}],
        "case_b": [{"start": 0.0, "stop": 10.0, "family": "mega",
                    "arm": "USM1"}],
    }
    with pytest.raises(ValueError):
        split_cases(cases, holdout_fraction=0.5, seed=3)


# ---------------------------------------------------------------------------
# sample_points
# ---------------------------------------------------------------------------

def test_sample_points_respects_the_margin():
    """A margin of 1.0s on a 10s->12s interval leaves an 11s->11s point --
    zero span -- so it must be dropped, not sampled at the edge."""
    intervals = [{"start": 10.0, "stop": 12.0, "family": "large", "arm": "A",
                 "part": "1.0"}]
    points = sample_points(intervals, spacing=8.0, max_per_interval=5, margin=1.0)
    assert points == []


def test_sample_points_caps_at_max_per_interval():
    """A very long install must not flood the pool with near-duplicate
    frames -- that is the same memorisation risk split_cases guards against
    one level up, applied inside a single case."""
    intervals = [{"start": 0.0, "stop": 1000.0, "family": "mega", "arm": "A",
                 "part": "1.0"}]
    points = sample_points(intervals, spacing=8.0, max_per_interval=5, margin=1.0)
    assert len(points) == 5
    times = [t for t, _family, _arm, _part in points]
    assert times == sorted(times)
    assert all(0.0 <= t <= 1000.0 for t in times)


def test_sample_points_drops_unrecognised_family():
    intervals = [{"start": 0.0, "stop": 50.0, "family": None, "arm": "A",
                 "part": "1.0"}]
    assert sample_points(intervals) == []


def test_sample_points_short_interval_yields_one_centred_point():
    intervals = [{"start": 10.0, "stop": 16.0, "family": "large", "arm": "A",
                 "part": "2.0"}]
    points = sample_points(intervals, spacing=8.0, max_per_interval=5, margin=1.0)
    assert len(points) == 1
    t, family, arm, part = points[0]
    assert t == pytest.approx(13.0)
    assert family == "large"
    assert arm == "A"
    assert part == "2.0"


def test_sample_points_raises_when_a_labelled_interval_has_no_part():
    """R28: a config/variant_labels.json below version 2 (or any hand-built
    intervals list) that omits `part` must fail loudly here, not resolve
    silently to some default video file downstream in build_examples."""
    intervals = [{"start": 0.0, "stop": 50.0, "family": "large", "arm": "A"}]
    with pytest.raises(ValueError):
        sample_points(intervals)


# ---------------------------------------------------------------------------
# sweep_cutoff
# ---------------------------------------------------------------------------

def test_sweep_cutoff_finds_the_smallest_cutoff_clearing_the_target():
    rng = np.random.default_rng(4)
    n = 400
    labels = np.array(["large"] * (n // 2) + ["mega"] * (n // 2))
    # Confident and (mostly) correct: p_large near 1 for large, near 0 for
    # mega, with a little noise so accuracy is not a trivial 1.0 everywhere.
    p_large = np.concatenate([
        np.clip(rng.normal(0.9, 0.05, n // 2), 0.0, 1.0),
        np.clip(rng.normal(0.1, 0.05, n // 2), 0.0, 1.0),
    ])
    cutoff, accuracy, coverage = sweep_cutoff(labels, p_large, target_accuracy=0.75)
    assert cutoff > 0.5
    assert accuracy > 0.75
    assert 0.0 < coverage <= 1.0


def test_sweep_cutoff_never_returns_a_cutoff_at_or_below_chance():
    """Every value sweep_cutoff can return must already be legal input to
    variant_record -- a single-line change to MIN_CUTOFF that let it drift to
    0.5 would defeat the abstention layer downstream."""
    rng = np.random.default_rng(5)
    labels = np.array(["large", "mega"] * 100)
    p_large = np.clip(rng.normal(0.5, 0.2, 200), 0.0, 1.0)
    cutoff, _accuracy, _coverage = sweep_cutoff(labels, p_large,
                                                target_accuracy=0.75)
    assert cutoff > 0.5


def test_sweep_cutoff_falls_back_honestly_when_target_is_unreachable():
    """Near-chance predictions everywhere: no cutoff can clear 0.75 accuracy.
    ABSTENTION IS A FEATURE, so this must not raise -- it returns the best
    achievable point instead of manufacturing a decision the data does not
    support."""
    rng = np.random.default_rng(6)
    labels = np.array(["large", "mega"] * 200)
    p_large = np.clip(rng.normal(0.5, 0.02, 400), 0.0, 1.0)
    cutoff, accuracy, coverage = sweep_cutoff(labels, p_large, target_accuracy=0.75)
    assert cutoff > 0.5
    assert 0.0 <= accuracy <= 1.0
    assert coverage > 0.0


def test_sweep_cutoff_raises_when_nothing_ever_decides():
    """Every confidence sits exactly at 0.5 -- the strictly-above-chance grid
    can never fire, and there is nothing honest to report."""
    labels = np.array(["large", "mega"])
    p_large = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        sweep_cutoff(labels, p_large, target_accuracy=0.75)


# ---------------------------------------------------------------------------
# resolve_case_video -- pure pathlib, no cv2 and no video decoding. R28
# deleted this module's earlier duration-probing approach (which DID need
# cv2, to open every candidate file and read its length) precisely because
# it could not reliably tell two on-disk parts apart; the replacement can be
# tested with empty placeholder files, since it never opens anything.
# ---------------------------------------------------------------------------

def test_resolve_case_video_returns_none_for_a_missing_case_dir(tmp_path):
    assert resolve_case_video(str(tmp_path), "case_999", "1.0") is None


def test_resolve_case_video_returns_none_when_the_named_part_is_absent(tmp_path):
    """The recorded part (2.0) has no file, even though part 1 exists --
    must return None, never silently fall back to the part that IS there."""
    case_dir = tmp_path / "case_x"
    case_dir.mkdir()
    (case_dir / "case_x_video_part_001.mp4").write_bytes(b"part 1")
    assert resolve_case_video(str(tmp_path), "case_x", "2.0") is None


def test_resolve_case_video_picks_the_recorded_part_not_the_first_on_disk(tmp_path):
    """R28's whole point: a "just take the first video file for this case"
    implementation would return part 1 regardless of which part the label
    actually names, silently pairing the wrong part's pixels with a
    correct-looking family label. This assertion is verified to actually
    catch that -- see the pasted failing output in the Task 10 fix report
    for a naive `sorted(case_dir.glob(...))[0]` resolver run against this
    exact test."""
    case_dir = tmp_path / "case_x"
    case_dir.mkdir()
    (case_dir / "case_x_video_part_001.mp4").write_bytes(b"part 1")
    (case_dir / "case_x_video_part_002.mp4").write_bytes(b"part 2")

    resolved = resolve_case_video(str(tmp_path), "case_x", "2.0")

    assert resolved is not None
    assert resolved.name == "case_x_video_part_002.mp4"


def test_resolve_case_video_rejects_a_non_canonical_part():
    """Only normalize_part's 'N.0' form is accepted -- resolving a filename
    from anything else would be guessing at a format instead of reading the
    one config/variant_labels.json actually writes."""
    with pytest.raises(ValueError):
        resolve_case_video("/irrelevant", "case_x", "2")


# ---------------------------------------------------------------------------
# build_examples -- the skip-and-tally path only (no real video content is
# needed for this: when the recorded part has no file at all,
# resolve_case_video returns None before build_examples ever tries to open
# anything, so this exercises the exact path a real corrupted-corpus run
# would hit without needing a decodable .mp4 fixture).
# ---------------------------------------------------------------------------

def test_build_examples_skips_and_tallies_an_interval_with_no_video_for_its_part(tmp_path):
    """An interval whose recorded part has no matching video file must be
    skipped, not crash, and the reason must be countable -- not merged into
    some other drop reason where it would be invisible in the run's log."""
    (tmp_path / "case_x").mkdir()   # the case dir exists; no .mp4 inside it
    cases = {"case_x": [{"start": 0.0, "stop": 5.0, "family": "large",
                         "arm": "A", "part": "1.0"}]}

    examples, drops = build_examples(
        ["case_x"], cases, str(tmp_path), detector=None,
        spacing=8.0, max_per_interval=5)

    assert examples == []
    assert drops["no_video_for_part"] == 1


# ---------------------------------------------------------------------------
# load_variant_labels -- version guard
# ---------------------------------------------------------------------------

def test_load_variant_labels_refuses_a_file_below_the_minimum_version(tmp_path):
    """A version-1 config/variant_labels.json carries no `part` on its
    intervals (R28). Loading it here must fail immediately and say why,
    rather than let sample_points raise deep inside the first interval of
    the first case, which is a confusing place to discover the wrong file
    is on the command line."""
    import json as json_module
    path = tmp_path / "variant_labels.json"
    path.write_text(json_module.dumps({
        "version": 1,
        "cases": {"case_000": [{"start": 0.0, "stop": 5.0, "family": "large",
                                "arm": "A"}]},
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        load_variant_labels(str(path))


# ---------------------------------------------------------------------------
# load_heldout_ids / exclude_graded_cases -- controller ruling R30. The
# graded/public-sample cases (config/splits_v2.json's "heldout" list) must
# never enter this head's train split or its own held-out split. See
# scripts/train_variant.py's module docstring for what went wrong before
# this existed: 7 of the 11 graded cases landed in TRAIN, including
# case_132, and 4 landed in the head's OWN held-out split, including
# case_126 -- the two graded failures this component exists to fix.
# ---------------------------------------------------------------------------

def _write_splits(tmp_path, heldout):
    import json as json_module
    path = tmp_path / "splits_v2.json"
    path.write_text(json_module.dumps({"heldout": heldout}), encoding="utf-8")
    return path


def test_load_heldout_ids_normalises_every_id(tmp_path):
    """The public-sample spelling ('case122') and this repo's own spelling
    ('case_005') must both come back in the one canonical form."""
    path = _write_splits(tmp_path, ["case122", "case_005"])
    assert load_heldout_ids(str(path)) == {"case_122", "case_005"}


def test_load_heldout_ids_raises_when_the_list_is_missing():
    import json as json_module
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "splits.json"
        path.write_text(json_module.dumps({"train": [], "val": []}),
                        encoding="utf-8")
        with pytest.raises(ValueError):
            load_heldout_ids(str(path))


def test_load_heldout_ids_raises_when_the_list_is_empty(tmp_path):
    path = _write_splits(tmp_path, [])
    with pytest.raises(ValueError):
        load_heldout_ids(str(path))


def test_exclude_graded_cases_removes_them_from_the_pool_entirely():
    """Not merely 'excluded from train' -- removed from `cases` before
    split_cases ever runs, so the excluded case cannot land in EITHER split
    train produces."""
    cases = {
        "case_001": [{"start": 0.0, "stop": 5.0, "family": "large", "arm": "A"}],
        "case_002": [{"start": 0.0, "stop": 5.0, "family": "mega", "arm": "A"}],
    }
    kept, n_cases, n_intervals = exclude_graded_cases(cases, {"case_001"})
    assert "case_001" not in kept
    assert "case_002" in kept
    assert n_cases == 1
    assert n_intervals == 1


def test_exclude_graded_cases_matches_across_id_spellings():
    """THE R30 REGRESSION GUARD. `cases` is keyed the way
    config/variant_labels.json spells it ('case_005'); the heldout set is
    spelled the way the public sample directories spell it ('case5', no
    zero-padding, no underscore). A raw string-equality/`in` implementation
    would find no overlap and exclude nothing -- which is exactly the bug
    that let 7 of the 11 graded cases into this head's train split. Verified
    to actually catch a naive implementation: see the pasted failing output
    in the Task 10 fix report for a plain `case_id in heldout_ids`
    resolver run against this exact test."""
    cases = {
        "case_005": [{"start": 0.0, "stop": 5.0, "family": "large", "arm": "A"}],
        "case_009": [{"start": 0.0, "stop": 5.0, "family": "mega", "arm": "A"}],
    }
    kept, n_cases, n_intervals = exclude_graded_cases(cases, {"case5"})
    assert "case_005" not in kept
    assert "case_009" in kept
    assert n_cases == 1
    assert n_intervals == 1


def test_exclude_graded_cases_raises_when_nothing_is_excluded():
    """A zero-exclusion is not evidence the corpus was already clean -- on
    the real corpus it is the same normalisation bug recurring, so this
    must fail loudly rather than silently proceed with an unfiltered pool."""
    cases = {
        "case_001": [{"start": 0.0, "stop": 5.0, "family": "large", "arm": "A"}],
    }
    with pytest.raises(ValueError):
        exclude_graded_cases(cases, {"case_999"})
