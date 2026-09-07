"""The plan ladder must not select evidence the card cannot hold.

MEASURED, NOT MODELLED. Every number here comes from condor probe 9716652,
which reproduced a T4 on an H200 by capping the allocator to 14.16 GiB AND
forcing the math SDPA kernel -- the kernel sm_75 is stuck with because flash
needs sm_80+. The simulator was calibrated first: forced-math, uncapped,
5184 tokens peaked at 15.88 GiB against the 16.5 GiB a real sm_75 Quadro RTX
8000 measured on the identical image (job 9716098). Within 4%.

    tokens   peak GiB   plan        under a 14.16 GiB T4 ceiling
    -----------------------------------------------------------
      2592      8.37    8x512       fits
      3072      9.42    12x448      fits
      2912      9.59    16x384      fits
      5184     15.88    16x512      OOM  <- what shipped, and what the
                                          grader ran on all 11 cases

The bug this pins: select_plan chose on TIME ALONE. With a 420 s budget it
always reached for 16x512, on every card, forever -- so the VLM loaded,
prefilled, and died on the grader's T4 in all eleven graded cases while
`try_vlm_result` swallowed the exception and kept the router's answer. Eleven
silent failures that looked exactly like eleven agreements.
"""
import pytest

from surgvu import frame_plan


#: A T4's nameplate, and the card Grand Challenge grades on.
T4_GIB = 14.56

#: Peak VRAM measured under the math kernel, by probe 9716652.
MEASURED = [(2592, 8.37), (3072, 9.42), (2912, 9.59), (5184, 15.88)]


def test_the_shipped_plan_is_the_one_that_oomed():
    """16x512 is 5184 tokens -- the plan that failed on the grader."""
    assert frame_plan.vision_tokens(16, 512) == 5184


@pytest.mark.parametrize("tokens,peak", MEASURED)
def test_the_token_ceiling_admits_exactly_what_fit(tokens, peak):
    """The ceiling must accept every measured fit and reject the OOM.

    Not a fitted curve: a fitted quadratic through these four points
    mispredicts 16x384 by 0.5 GiB, and a memory model that is wrong by half a
    gigabyte on the data it was fitted to has no business deciding whether
    the grader's card survives.
    """
    admitted = tokens <= frame_plan.MATH_KERNEL_TOKEN_CEILING
    assert admitted == (peak < T4_GIB)


def test_a_small_card_never_selects_the_oom_plan():
    """The regression, stated as the grader would experience it."""
    plan = frame_plan.select_plan(420.0, vram_gib=T4_GIB)
    assert plan is not None, "a T4 must still get SOME evidence"
    assert plan["tokens"] <= frame_plan.MATH_KERNEL_TOKEN_CEILING
    assert plan["tokens"] < 5184


def test_a_big_card_is_unchanged():
    """An H200 has no reason to be punished for the T4's kernel."""
    rich = frame_plan.select_plan(420.0, vram_gib=139.8)
    assert rich["tokens"] == frame_plan.select_plan(420.0)["tokens"]


def test_vram_unaware_calls_behave_exactly_as_before():
    """Every existing caller passes no vram_gib and must not move."""
    for budget in (60.0, 120.0, 420.0, 3000.0):
        assert (frame_plan.select_plan(budget)
                == frame_plan.select_plan(budget, vram_gib=None))


def test_more_vram_never_returns_poorer_evidence():
    prev = 0
    for vram in (14.56, 16.0, 24.0, 48.0, 139.8):
        plan = frame_plan.select_plan(420.0, vram_gib=vram)
        assert plan["tokens"] >= prev
        prev = plan["tokens"]


def test_the_ceiling_leaves_real_headroom_on_a_t4():
    """9.42 GiB of 14.56 is not a coincidence to be tightened later.

    The margin absorbs what the probe could not see: a judge model held
    alongside, allocator fragmentation, and the driver's own context.
    """
    fits = [p for t, p in MEASURED if t <= frame_plan.MATH_KERNEL_TOKEN_CEILING]
    assert max(fits) < T4_GIB - 4.0


def test_the_ladder_offers_a_rung_at_the_ceiling():
    """Without a 3072-token rung the T4 drops to 8x512 and loses resolution.

    12x448 measured LOWER peak than 16x384 (9.42 vs 9.59) while carrying MORE
    tokens (3072 vs 2912) at higher resolution, so it dominates it outright.
    """
    tokens = {frame_plan.vision_tokens(n, s, m)
              for _, n, s, m in frame_plan.FRAME_PLANS}
    assert frame_plan.MATH_KERNEL_TOKEN_CEILING in tokens


def test_an_absurdly_small_card_gets_none_not_an_oom():
    """Better no VLM answer than a crash the seam silently absorbs."""
    assert frame_plan.select_plan(420.0, vram_gib=2.0) is None


# --------------------------------------------------------------------------
# stepping down after an OOM
# --------------------------------------------------------------------------

def test_the_step_down_is_strictly_cheaper_every_time():
    """Repeated OOMs must terminate, not cycle."""
    plan = frame_plan.select_plan(420.0, vram_gib=139.8)
    seen = [plan["tokens"]]
    while plan is not None:
        plan = frame_plan.next_cheaper_plan(plan)
        if plan is not None:
            assert plan["tokens"] < seen[-1]
            seen.append(plan["tokens"])
    assert len(seen) > 1, "there must be somewhere to step down to"


def test_the_cheapest_plan_has_nowhere_left_to_go():
    cheapest = min(frame_plan.FRAME_PLANS,
                   key=lambda p: frame_plan.vision_tokens(p[1], p[2], p[3]))
    plan = {"tokens": frame_plan.vision_tokens(*cheapest[1:])}
    assert frame_plan.next_cheaper_plan(plan) is None


def test_a_t4_plan_can_still_step_down():
    """The retry path must be available on the card that actually needs it."""
    plan = frame_plan.select_plan(420.0, vram_gib=T4_GIB)
    assert frame_plan.next_cheaper_plan(plan) is not None
