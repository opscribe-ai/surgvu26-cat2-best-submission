"""The frame planner decides how much evidence the VLM gets, per case.

Its failure mode is asymmetric and that shapes every test here: reserving too
much costs a few frames of evidence, reserving too little means a prefill that
cannot finish, no response written, and a score of 0 -- worse than any wrong
answer.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu import frame_plan as fp                             # noqa: E402


def test_plans_are_ordered_richest_first():
    """select_plan stops at the FIRST fit, so a cheaper plan appearing before
    a more expensive one would silently cap the whole system at the cheap one."""
    costs = [fp.estimated_seconds(n, s, m) for _l, n, s, m in fp.FRAME_PLANS]
    assert costs == sorted(costs, reverse=True), costs


def test_a_generous_budget_selects_a_rich_plan():
    plan = fp.select_plan(600)
    assert plan is not None
    assert plan["n_frames"] >= 16


def test_a_tight_budget_steps_down_rather_than_failing():
    plan = fp.select_plan(150)
    assert plan is not None
    assert plan["n_frames"] < 16


def test_no_budget_returns_None_so_the_caller_skips_entirely():
    """THE LOAD-BEARING CASE. Returning the cheapest plan here would start a
    prefill there is no time for -- and the deadline StoppingCriteria stops
    GENERATION token by token, so it cannot interrupt a prefill at all. On a
    contended node the router path alone has measured 317s of the 600s
    budget, so this is a real outcome, not a defensive one."""
    assert fp.select_plan(10) is None
    assert fp.select_plan(0) is None


def test_multiscale_costs_more_than_the_same_frames_flat():
    flat = fp.vision_tokens(16, 512, multiscale=False)
    multi = fp.vision_tokens(16, 512, multiscale=True)
    assert multi > flat


def test_resolution_dominates_frame_count_in_token_cost():
    """768 is 2.25x the tokens of 512 per frame, so 8 frames at 768 costs more
    than 16 at 512 -- the reason the planner cannot treat 'more frames' and
    'bigger frames' as interchangeable."""
    assert fp.vision_tokens(8, 768) > fp.vision_tokens(16, 512)


def test_the_calibration_constant_is_pessimistic_against_the_l40_measurement():
    """An L40 ran 1,620 tokens inside a 13.11s VLM step, ~8.1 s/1k. The
    constant must stay well ABOVE that, because the grader's T4 is 3-5x slower
    and under-reserving risks a 0 while over-reserving costs a few frames."""
    l40_measured = 13.11 / 1.620
    assert fp.SECONDS_PER_1K_TOKENS > l40_measured * 3


def test_a_faster_measured_rate_unlocks_richer_plans():
    """The pessimistic default is a PLACEHOLDER until sm_75 is measured.
    Passing a real rate must let the planner reach plans it otherwise refuses,
    or measuring would buy nothing."""
    default_plan = fp.select_plan(600)
    measured_plan = fp.select_plan(600, seconds_per_1k=10.0)
    assert measured_plan["tokens"] > default_plan["tokens"]


def test_describe_never_raises_on_None():
    assert "skipping" in fp.describe(None)
