"""Pure-logic tests for the v2 split: heldout exclusion + tool stratification.

Nothing here opens a real shard. The corpus is described as
`{case_id: [[tool, ...], ...]}` -- one list of tool names per 30-second
window -- which is exactly the shape `build_splits_v2` reduces the real shard
metadata to, so the assignment logic is exercised on synthetic data that
takes microseconds to build.

The two defects this file exists to prevent, both of which were live in
`config/splits.json`:

  1. The 11 public sample cases (the only question/answer data that exists)
     sat inside train and val. They are named `case122` on disk and
     `case_122` in the split file, so `set(sample) & set(split)` returns the
     empty set and the leak looks clean. That mistake has already been made
     here once.

  2. `tip-up fenestrated grasper` had 157 train windows and 0 val windows, so
     its per-class F1 was 0.0 by construction and macro-F1 was ~0.056 lower
     than the model deserved, for a reason that had nothing to do with the
     model.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from surgvu.sampling import (  # noqa: E402
    MIN_VAL_WINDOWS, make_splits_v2, normalize_case_id,
)
from surgvu.taxonomy import TOOL_CLASSES  # noqa: E402

from build_splits_v2 import (  # noqa: E402
    case_windows_from_shard_meta, heldout_case_ids, main,
)

COMMON = "cadiere forceps"
RARE = "tip-up fenestrated grasper"


def _case(total, tool_counts):
    """One case: `total` windows, of which `tool_counts[tool]` contain `tool`."""
    windows = [[] for _ in range(total)]
    for tool, n in tool_counts.items():
        assert n <= total, "a case cannot have more %s windows than windows" % tool
        for i in range(n):
            windows[i].append(tool)
    return windows


def _corpus(n_cases=20, per_case=100, first_id=0):
    """`n_cases` interchangeable cases, every window carrying COMMON."""
    return {"case_%03d" % (first_id + i): _case(per_case, {COMMON: per_case})
            for i in range(n_cases)}


def _granular_corpus():
    """Cases of three different lengths carrying four classes on different
    strides -- no two are interchangeable, so the assignment has real local
    optima to get stuck in. `_corpus` cannot stand in: identical cases make
    almost every arrangement locally optimal."""
    corpus = {}
    for i in range(30):
        length = [40, 80, 160][i % 3]
        tools = {COMMON: length}
        if i % 7 == 0:
            tools[RARE] = length // 3
        if i % 5 == 0:
            tools["stapler"] = length // 4
        if i % 3 == 0:
            tools["clip applier"] = length // 2
        corpus["case_%03d" % i] = _case(length, tools)
    return corpus


# --------------------------------------------------------------------------
# id normalisation -- the reason the leak was invisible
# --------------------------------------------------------------------------

def test_normalize_case_id_maps_both_spellings_to_one_form():
    """`case122` (sample directory) and `case_122` (splits.json) are the same
    case. A set intersection over the raw strings says otherwise."""
    assert normalize_case_id("case122") == "case_122"
    assert normalize_case_id("case_122") == "case_122"
    assert normalize_case_id("  case122  ") == "case_122"
    assert normalize_case_id("case7") == "case_007"


def test_normalize_case_id_refuses_to_guess():
    """A name it cannot parse must raise, not fall through as itself: a heldout
    case that quietly fails to normalise is a heldout case that quietly stays in
    the training set."""
    for bad in ["", "case", "cat2_sample", "case_12a", "casey_12", "12"]:
        with pytest.raises(ValueError):
            normalize_case_id(bad)


def test_two_spellings_of_one_case_in_the_corpus_are_an_error():
    """`case122` and `case_122` normalise to the same case. Silently keeping
    the last one seen would drop half its windows from every count; keeping
    both would let one case sit in train and val at the same time."""
    corpus = _corpus(5)
    corpus["case3"] = _case(50, {COMMON: 50})
    with pytest.raises(ValueError, match="case_003"):
        make_splits_v2(corpus, heldout_ids=[])


def test_heldout_case_ids_normalises_a_directory_listing():
    assert heldout_case_ids(["case131", "case122", "case130"]) == [
        "case_122", "case_130", "case_131"]


# --------------------------------------------------------------------------
# heldout exclusion
# --------------------------------------------------------------------------

def test_heldout_cases_are_held_out_even_when_spelled_the_other_way():
    corpus = _corpus(20)
    result = make_splits_v2(corpus, heldout_ids=["case000", "case001", "case002"])

    assert result["heldout"] == ["case_000", "case_001", "case_002"]
    for case in result["heldout"]:
        assert case not in result["train"]
        assert case not in result["val"]


def test_a_heldout_case_that_is_not_in_the_corpus_is_an_error():
    """Silently dropping an unmatched heldout id is how "no overlap" gets
    reported for a split that is 100% leaking."""
    with pytest.raises(ValueError, match="case_999"):
        make_splits_v2(_corpus(20), heldout_ids=["case999"])


def test_every_case_lands_in_exactly_one_split():
    corpus = _corpus(20)
    result = make_splits_v2(corpus, heldout_ids=["case_000"])
    train, val, heldout = set(result["train"]), set(result["val"]), set(result["heldout"])

    assert train | val | heldout == set(corpus)
    assert not train & val
    assert not train & heldout
    assert not val & heldout
    assert len(train) + len(val) + len(heldout) == len(corpus)


def test_heldout_windows_are_counted_as_heldout_and_nowhere_else():
    corpus = _corpus(20, per_case=100)
    result = make_splits_v2(corpus, heldout_ids=["case_000", "case_001"])
    counts = result["meta"]["window_counts"][COMMON]

    assert counts["heldout"] == 200
    assert counts["train"] + counts["val"] + counts["heldout"] == 2000


# --------------------------------------------------------------------------
# stratification
# --------------------------------------------------------------------------

def test_val_holds_roughly_the_requested_fraction_of_cases():
    result = make_splits_v2(_corpus(100), heldout_ids=[], val_fraction=0.2)
    assert len(result["val"]) == 20
    assert len(result["train"]) == 80

    # 23 * 0.2 = 4.6 cases: rounded, not truncated. Truncating loses a whole
    # case out of val every time the fraction does not divide evenly.
    assert len(make_splits_v2(_corpus(23), heldout_ids=[])["val"]) == 5


def test_the_val_window_share_follows_the_fraction_when_cases_differ_in_length():
    """Cases differ in length by more than an order of magnitude, so 20% of
    the cases is not 20% of the windows -- and val is a window count in every
    metric computed from it.

    Every case here carries the same 10 tool windows and differs only in how
    many untooled windows follow, so the per-class terms cannot tell these
    cases apart at all. Only a term on the total window count can."""
    corpus = {}
    for i in range(40):
        length = 10 if i % 2 else 400
        corpus["case_%03d" % i] = _case(length, {COMMON: 10})

    for seed in (1, 11, 21):
        totals = make_splits_v2(corpus, heldout_ids=[], val_fraction=0.2,
                                seed=seed)["meta"]["window_totals"]
        share = totals["val"] / float(totals["train"] + totals["val"])
        assert 0.19 <= share <= 0.21, (seed, totals)


def test_the_published_split_is_a_local_optimum_and_says_so():
    """A swap search that ran out of passes and one that finished look
    identical from the outside. The file records which happened, so "the
    greedy worked" is a checkable claim rather than an assumption."""
    corpus = _granular_corpus()

    assert make_splits_v2(corpus, heldout_ids=[], seed=13)["meta"]["local_optimum"]

    cut_short = make_splits_v2(corpus, heldout_ids=[], seed=13, max_passes=1)
    assert cut_short["meta"]["local_optimum"] is False


def test_the_val_floor_never_takes_more_than_half_a_class():
    """A floor of 25 against a class with 20 windows in the whole corpus is
    unreachable without emptying train of it. Being able to measure a class
    is worth nothing if the model was never taught it."""
    corpus = _corpus(20)
    corpus["case_005"] = _case(100, {COMMON: 100, RARE: 10})
    corpus["case_012"] = _case(100, {COMMON: 100, RARE: 10})

    result = make_splits_v2(corpus, heldout_ids=[], seed=9)
    counts = result["meta"]["window_counts"][RARE]

    assert counts["val"] == 10
    assert counts["train"] == 10
    assert result["meta"]["min_val_windows_per_class"][RARE] == 10
    assert RARE not in result["meta"]["underrepresented"]


def test_a_class_confined_to_two_cases_still_reaches_val():
    """The tip-up failure in miniature: a class that lives in a handful of
    cases is exactly what an unstratified shuffle drops entirely."""
    corpus = _corpus(20)
    corpus["case_000"] = _case(100, {COMMON: 100, RARE: 40})
    corpus["case_001"] = _case(100, {COMMON: 100, RARE: 40})

    result = make_splits_v2(corpus, heldout_ids=[], seed=3)
    counts = result["meta"]["window_counts"][RARE]

    assert counts["val"] >= MIN_VAL_WINDOWS
    assert counts["train"] > 0           # and it is still learnable
    assert RARE not in result["meta"]["underrepresented"]


def test_no_class_present_in_the_corpus_is_left_with_an_empty_val():
    corpus = _corpus(20)
    corpus["case_004"] = _case(100, {COMMON: 100, RARE: 60})
    corpus["case_009"] = _case(100, {COMMON: 100, RARE: 60})
    corpus["case_014"] = _case(100, {COMMON: 100, "stapler": 50})
    corpus["case_015"] = _case(100, {COMMON: 100, "stapler": 50})

    result = make_splits_v2(corpus, heldout_ids=[], seed=1)

    for tool in (RARE, "stapler", COMMON):
        assert result["meta"]["window_counts"][tool]["val"] > 0, tool


def test_a_class_living_in_a_single_case_is_reported_not_forced():
    """One case cannot be in both splits. Dragging it into val to satisfy the
    floor would take the class's *only* training windows with it, so the
    honest outcome is to keep it trainable and say out loud that it cannot be
    measured."""
    corpus = _corpus(20)
    corpus["case_003"] = _case(100, {COMMON: 100, RARE: 30})

    result = make_splits_v2(corpus, heldout_ids=[], seed=5)
    meta = result["meta"]

    assert RARE in meta["unsplittable"]
    assert meta["window_counts"][RARE]["train"] == 30
    assert meta["window_counts"][RARE]["val"] == 0
    assert RARE in meta["underrepresented"]


def test_a_class_that_had_to_be_over_sampled_into_val_is_flagged():
    """Cases move whole, so a class confined to a couple of long cases cannot
    land on 20% -- the reachable val shares here are 0% or 50%. Reaching the
    floor at 50% is the right trade, but it is a distortion of the split and
    the file has to say so instead of presenting 50% as if it were 20%."""
    corpus = _corpus(20)
    corpus["case_006"] = _case(100, {COMMON: 100, RARE: 40})
    corpus["case_017"] = _case(100, {COMMON: 100, RARE: 40})

    result = make_splits_v2(corpus, heldout_ids=[], seed=4)
    meta = result["meta"]

    assert meta["window_counts"][RARE]["val"] == 40
    assert RARE in meta["over_represented"]
    assert RARE in meta["distortion_note"]
    assert COMMON not in meta["over_represented"]


def test_cases_per_class_explains_the_granularity_and_ignores_heldout():
    """The number of cases a class lives in is the whole reason a val share
    can or cannot be hit, so it is recorded next to the counts. Heldout cases are
    not available to the split and must not be counted as if they were."""
    corpus = _corpus(20)
    corpus["case_000"] = _case(100, {COMMON: 100, RARE: 10})
    corpus["case_001"] = _case(100, {COMMON: 100, RARE: 10})
    corpus["case_002"] = _case(100, {COMMON: 100, RARE: 10})

    meta = make_splits_v2(corpus, heldout_ids=["case_002"], seed=6)["meta"]

    assert meta["cases_per_class"][RARE] == 2
    assert meta["cases_per_class"][COMMON] == 19
    assert meta["cases_per_class"]["stapler"] == 0


def test_a_class_absent_from_the_whole_corpus_is_reported_as_absent():
    result = make_splits_v2(_corpus(20), heldout_ids=[])
    meta = result["meta"]

    assert meta["window_counts"]["stapler"] == {"train": 0, "val": 0, "heldout": 0}
    assert "stapler" in meta["absent"]
    assert "stapler" not in meta["unsplittable"]


def test_the_split_is_reproducible_and_seed_dependent():
    corpus = _corpus(30)
    corpus["case_007"] = _case(100, {COMMON: 100, RARE: 50})
    corpus["case_011"] = _case(100, {COMMON: 100, RARE: 50})

    a = make_splits_v2(corpus, heldout_ids=[], seed=7)
    b = make_splits_v2(corpus, heldout_ids=[], seed=7)
    assert a == b

    c = make_splits_v2(corpus, heldout_ids=[], seed=8)
    assert c["meta"]["window_counts"][RARE]["val"] >= MIN_VAL_WINDOWS


def test_more_restarts_can_only_improve_the_published_split():
    """A swap search stops at the first arrangement no single exchange
    improves, and which one that is depends entirely on where it started.
    Restarts exist because one climb settles for a visibly worse split -- on
    the real corpus, a single start left `tip-up fenestrated grasper` at
    75/82 instead of 68/89. Each restart is an independent climb and the best
    is kept, so the achieved cost must never go up when restarts do."""
    corpus = _granular_corpus()

    def cost(seed, restarts):
        return make_splits_v2(corpus, heldout_ids=[], seed=seed,
                              restarts=restarts)["meta"]["search_cost"]

    for seed in range(5):
        assert cost(seed, 8) <= cost(seed, 1) + 1e-12, seed
    assert cost(3, 8) < cost(3, 1)          # and here it strictly does


def test_window_counts_are_the_real_counts_not_an_assumption():
    """The greedy pass is not trusted: the meta block is recomputed from the
    assignment, so a reader auditing the file is auditing the split."""
    corpus = _corpus(20)
    corpus["case_002"] = _case(100, {COMMON: 100, RARE: 45})
    corpus["case_013"] = _case(100, {COMMON: 100, RARE: 45})

    result = make_splits_v2(corpus, heldout_ids=["case_019"], seed=2)
    counts = result["meta"]["window_counts"]

    for tool in TOOL_CLASSES:
        for split in ("train", "val", "heldout"):
            expected = sum(
                sum(1 for w in corpus[case] if tool in w)
                for case in result[split])
            assert counts[tool][split] == expected, (tool, split)


def test_meta_records_the_parameters_that_produced_the_split():
    result = make_splits_v2(_corpus(20), heldout_ids=["case_000"],
                            val_fraction=0.25, seed=42, min_val_windows=11)
    meta = result["meta"]

    assert meta["seed"] == 42
    assert meta["val_fraction"] == 0.25
    assert meta["min_val_windows"] == 11
    assert "case_000" in meta["heldout_rationale"]


def test_an_unknown_tool_name_is_rejected():
    """`encode_tools` raises on a class outside the twelve; a split built on
    counts that silently ignored it would be counting something else."""
    corpus = _corpus(20)
    corpus["case_000"] = _case(100, {COMMON: 100})
    corpus["case_000"][0].append("suction irrigator")

    with pytest.raises(ValueError, match="suction irrigator"):
        make_splits_v2(corpus, heldout_ids=[])


def test_a_corpus_too_small_to_split_is_an_error():
    """Each of these would hand back a split with an empty side, which trains
    or validates on nothing while looking like a successful build."""
    with pytest.raises(ValueError, match="val"):
        make_splits_v2(_corpus(2), heldout_ids=[])                # rounds to 0 val

    with pytest.raises(ValueError, match="val"):
        make_splits_v2(_corpus(20), heldout_ids=[], val_fraction=1.0)   # 0 train

    with pytest.raises(ValueError, match="val"):
        make_splits_v2(_corpus(2), heldout_ids=["case_000", "case_001"])  # 0 left


# --------------------------------------------------------------------------
# shard metadata -> per-case windows
# --------------------------------------------------------------------------

def test_shard_meta_is_grouped_by_case_across_parts():
    meta = {
        "case_012_part1.npz": [{"tools": ["needle driver"], "task": "suturing"},
                               {"tools": [], "task": "other"}],
        "case_012_part2.npz": [{"tools": ["stapler"], "task": "other"}],
        "case_013_part1.npz": [{"tools": ["stapler"], "task": "other"}],
    }
    windows = case_windows_from_shard_meta(meta)

    assert sorted(windows) == ["case_012", "case_013"]
    assert len(windows["case_012"]) == 3
    assert sorted(t for w in windows["case_012"] for t in w) == [
        "needle driver", "stapler"]


def test_shard_meta_grouping_survives_a_case_id_that_contains_part():
    """`rsplit('_part', 1)` on the *name*, not a `split`: a left-anchored
    split would truncate at the first occurrence."""
    meta = {"case_012_part1_part2.npz": [{"tools": [], "task": "other"}]}
    assert sorted(case_windows_from_shard_meta(meta)) == ["case_012_part1"]


def test_a_shard_name_without_a_part_suffix_is_an_error():
    with pytest.raises(ValueError, match="stray.npz"):
        case_windows_from_shard_meta({"stray.npz": []})


# --------------------------------------------------------------------------
# the frozen file must stay frozen
# --------------------------------------------------------------------------

def test_main_refuses_to_write_over_the_frozen_split(tmp_path):
    """The current checkpoints were trained on config/splits.json. Overwriting
    it destroys their provenance, so the path is refused by name."""
    frozen = tmp_path / "splits.json"
    frozen.write_text('{"train": [], "val": []}', encoding="utf-8")

    # --force, so that the refusal has to come from the name and cannot be
    # the ordinary "file already exists" guard passing for the wrong reason.
    with pytest.raises(SystemExit, match="REFUSING"):
        main(["--shard-dir", str(tmp_path), "--sample-dir", str(tmp_path),
              "--out", str(frozen), "--force"])

    assert json.loads(frozen.read_text(encoding="utf-8")) == {"train": [], "val": []}


# --------------------------------------------------------------------------
# the real corpus (needs /staging)
# --------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
SHARDS = Path("/staging/groups/bhaskar_opscribe/surgvu/shards")
SAMPLE = Path("/staging/groups/bhaskar_opscribe/surgvu/cat2_sample")
SPLITS_V2 = REPO / "config" / "splits_v2.json"


@pytest.mark.slow
def test_committed_split_matches_the_real_shards():
    """Recount every window from /staging and check the committed file's meta
    block against it, so the audit trail in the file is the truth."""
    if not SHARDS.is_dir() or not SPLITS_V2.exists():
        pytest.skip("needs /staging shards and a built config/splits_v2.json")
    from build_splits_v2 import read_all_shard_meta

    committed = json.loads(SPLITS_V2.read_text(encoding="utf-8"))
    windows = case_windows_from_shard_meta(read_all_shard_meta(SHARDS))

    assert set(committed["heldout"]) == set(heldout_case_ids(
        p.name for p in SAMPLE.iterdir() if p.is_dir()))
    assert (set(committed["train"]) | set(committed["val"])
            | set(committed["heldout"])) == set(windows)

    for tool in TOOL_CLASSES:
        for split in ("train", "val", "heldout"):
            expected = sum(sum(1 for w in windows[case] if tool in w)
                           for case in committed[split])
            assert committed["meta"]["window_counts"][tool][split] == expected
        assert committed["meta"]["window_counts"][tool]["val"] > 0, tool


@pytest.mark.slow
def test_committed_split_shares_no_case_with_the_frozen_heldout_leak():
    if not SPLITS_V2.exists():
        pytest.skip("needs a built config/splits_v2.json")
    committed = json.loads(SPLITS_V2.read_text(encoding="utf-8"))
    heldout = set(committed["heldout"])

    assert len(heldout) == 11
    assert not heldout & set(committed["train"])
    assert not heldout & set(committed["val"])


# --------------------------------------------------------------------------
# the name itself
# --------------------------------------------------------------------------
# This split used to be called `dev`. "dev" conventionally names the set you
# TUNE against -- the one you are allowed to look at repeatedly. This one is
# the exact opposite: 11 sealed cases that are the only question-and-answer
# data in existence for this task, and the only thing standing behind the
# 0.8766 end-to-end number being a measurement rather than a memory. A name
# that invites the one use it must never have is a defect in the artefact, not
# a matter of taste.

def test_the_sealed_split_is_not_called_dev():
    committed = json.loads(SPLITS_V2.read_text(encoding="utf-8"))

    assert "heldout" in committed
    assert "dev" not in committed
    assert "dev_rationale" not in committed["meta"]
    assert "heldout_rationale" in committed["meta"]
    assert set(committed["meta"]["case_counts"]) == {"train", "val", "heldout"}
    assert set(committed["meta"]["window_totals"]) == {"train", "val", "heldout"}
    for row in committed["meta"]["window_counts"].values():
        assert set(row) == {"train", "val", "heldout"}


def test_the_generator_never_emits_a_dev_key():
    """The config could be renamed by hand and silently reverted by the next
    regeneration. The name has to come from the generator."""
    result = make_splits_v2(_corpus(20), heldout_ids=["case_000"])
    meta = result["meta"]

    assert "dev" not in result
    assert result["heldout"] == ["case_000"]
    assert "dev_rationale" not in meta
    assert "heldout_rationale" in meta
    # Every block keyed by split, not just the top level: the counts are what
    # a reader audits the split with, and one stale key there reintroduces the
    # name in the place it is most read.
    assert set(meta["case_counts"]) == {"train", "val", "heldout"}
    assert set(meta["window_totals"]) == {"train", "val", "heldout"}
    for row in meta["window_counts"].values():
        assert set(row) == {"train", "val", "heldout"}


def test_config_splits_json_is_untouched_by_the_rename():
    """v1 is frozen: the v1 checkpoints' provenance is this file. It has two
    keys and neither of them is the renamed one."""
    frozen = json.loads((REPO / "config" / "splits.json")
                        .read_text(encoding="utf-8"))

    assert set(frozen) == {"train", "val"}
    assert len(frozen["train"]) == 124 and len(frozen["val"]) == 31
