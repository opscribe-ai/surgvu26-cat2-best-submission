"""How many frames, at what resolution, chosen from the time actually left.

WHY THIS IS NOT A CONSTANT.

The grader allows 600 s per case and the shipped v5 uses ~18 s of it on an
L40 -- 3%. The obvious response is to spend the rest on more visual evidence,
and the obvious way to do that is to raise the frame count and the resolution
until the budget is full. Two things make a FIXED setting the wrong shape for
that:

  * THE HARDWARE IS UNKNOWN AND SLOWER. Every timing this project has is from
    an L40 or a developer's own GPU. The grader runs a T4: sm_75, ~320 GB/s
    against the L40's ~864, roughly 3-5x slower on the memory-bound part.
    Projected, 24 frames at 768px costs 452-753 s of a 600 s budget on a T4 --
    a configuration that is comfortable where it was measured and fatal where
    it runs.

  * THE DEADLINE CANNOT SAVE A PREFILL. `evidence_vlm._deadline_stopping_
    criteria` stops GENERATION token by token, which is the only hook
    transformers offers inside a call already in progress. Prefill happens
    BEFORE the first token, so a 17,496-token prefill that takes five minutes
    is uninterruptible -- the deadline fires after it, too late. A missing
    response scores 0, worse than any wrong answer.

So the plan is chosen UP FRONT from the seconds remaining, at the moment the
VLM is about to run. On a fast card with a fast router pass, that selects the
richest plan and genuinely fills the window. On a slow card, or a case that
already spent 300 s in perception, it steps down instead of gambling.

CALIBRATION IS DELIBERATELY PESSIMISTIC. `SECONDS_PER_1K_TOKENS` is set from
the L40 measurement multiplied by the worst end of the T4 penalty, because
being wrong slow costs a few frames of evidence and being wrong fast costs the
whole answer.
"""

#: (label, n_frames, size, multiscale) richest first.
#:
#: Resolution before frame count in the ordering: case124's failure was
#: "Bipolar Forceps" vs gold "Cadiere Forceps", a fine-grained instrument
#: discrimination where pixels-on-target plausibly matters more than another
#: view of the same scene. That is a hypothesis, not a measurement, and the
#: ordering is the cheapest place to encode it -- swapping two tuples is the
#: whole cost of being wrong.
FRAME_PLANS = (
    ("rich",     24, 768, True),
    ("high-res", 16, 768, False),
    ("wide",     16, 512, False),
    ("t4-safe",  12, 448, False),
    ("standard",  8, 512, False),
    ("minimal",   5, 512, False),
)

#: The most vision tokens a 16 GiB-class card can prefill, MEASURED.
#:
#: THIS IS THE NUMBER THAT WAS MISSING, and its absence cost eleven graded
#: cases. `select_plan` chose on time alone, so a 420 s budget always reached
#: for "wide" (5184 tokens) -- on every card, forever. On the grader's Tesla
#: T4 that prefill OOMs, `try_vlm_result` swallows the exception, and the
#: router's answer ships. Eleven silent failures that look exactly like
#: eleven agreements.
#:
#: WHY A TABLE AND NOT A FORMULA. Condor probe 9716652 measured peak VRAM at
#: four token counts under the math SDPA kernel -- the kernel sm_75 is stuck
#: with, since flash attention requires sm_80+ and that single fact is the
#: whole bug:
#:
#:     tokens   peak GiB   plan
#:       2592      8.37    8x512
#:       3072      9.42    12x448
#:       2912      9.59    16x384
#:       5184     15.88    16x512   <- OOM on a 14.56 GiB T4
#:
#: A quadratic fitted through those points mispredicts 16x384 by 0.5 GiB. A
#: memory model wrong by half a gigabyte on its own training data has no
#: business deciding whether the grader's card survives, so this is the
#: largest MEASURED fit instead, with 5.1 GiB of headroom left for the judge,
#: fragmentation and the driver context.
#:
#: The probe is trustworthy because it was calibrated before it was used:
#: forced-math and uncapped, 5184 tokens peaked at 15.88 GiB on an H200
#: against the 16.5 GiB a real sm_75 Quadro RTX 8000 measured on the
#: identical image (job 9716098). Within 4%.
#:
#: Raise this ONLY against a new measurement on sm_75 hardware, or on the
#: grader's own try-out runner. An H200 left to choose its own kernel reports
#: 6.89 GiB for the plan that OOMs on a T4, and believing that number is what
#: shipped v6.
MATH_KERNEL_TOKEN_CEILING = 3072

#: Below this, decline the VLM outright rather than pick the cheapest rung.
#:
#: The weights alone measured 7.22 GiB resident after load (probe 9716652),
#: before a single vision token is prefilled, and the smallest plan measured
#: 8.37 GiB in total. A card that cannot clear that will OOM on EVERY rung,
#: and an OOM is strictly worse than declining: `try_vlm_result` absorbs the
#: exception, so the run looks identical to a VLM that agreed with the
#: router, which is precisely the blindness that hid this bug through six
#: submissions. Declining is at least legible in the log.
MIN_VLM_VRAM_GIB = 10.0

#: Vision-token cost, matching Qwen2.5-VL's patch 14 and 2x2 spatial merge.
PATCH = 14
MERGE = 2

#: Seconds per 1,000 vision tokens, measured then made pessimistic.
#:
#: An L40 ran 1,620 tokens inside a 13.11 s VLM step, i.e. ~8.1 s/1k including
#: generation and fixed overhead. Multiplied by 5 -- the worst end of the T4
#: penalty -- gives ~40 s/1k. On the L40 this over-reserves by 5x and simply
#: selects a slightly smaller plan than it could have; on a T4 it is roughly
#: right. That asymmetry is the point: under-reserving risks a 0.
SECONDS_PER_1K_TOKENS = 40.0

#: Fixed cost of a VLM step regardless of frames: model already warm, prompt
#: assembly, generation of a short answer.
FIXED_OVERHEAD_SECONDS = 15.0


def vision_tokens(n_frames, size, multiscale=False):
    """Vision tokens for a plan. Multi-scale adds a half-count dense burst."""
    per_frame = ((size // PATCH) ** 2) // (MERGE ** 2)
    total = per_frame * n_frames
    if multiscale:
        total += per_frame * (n_frames // 2)
    return total


def estimated_seconds(n_frames, size, multiscale=False,
                      seconds_per_1k=SECONDS_PER_1K_TOKENS):
    """Pessimistic wall-clock estimate for one VLM step under this plan."""
    tokens = vision_tokens(n_frames, size, multiscale)
    return FIXED_OVERHEAD_SECONDS + (tokens / 1000.0) * seconds_per_1k


def select_plan(remaining_seconds, plans=FRAME_PLANS,
                seconds_per_1k=SECONDS_PER_1K_TOKENS, vram_gib=None):
    """The richest plan fitting `remaining_seconds` AND `vram_gib`, or None.

    TWO BUDGETS, NOT ONE. Time was the only one for v1..v6 and it is the
    reason the VLM never spoke on the grader: a plan can fit comfortably in
    600 s and still be unable to prefill on a 14.56 GiB card. `vram_gib=None`
    keeps the old time-only behaviour byte-for-byte, for callers (and tests)
    that have no device to ask.

    None means even the cheapest plan does not fit, and the caller must skip
    the VLM entirely rather than start something it cannot finish. That is a
    real outcome, not a defensive one: on a contended node the router path
    alone has measured 317 s of the 600 s budget.

    Breaks if: this returns the cheapest plan instead of None when nothing
    fits (the caller would then start a prefill it has no time for, and the
    deadline cannot interrupt a prefill), or if the plans are reordered so
    that a later entry is more expensive than an earlier one -- the search
    stops at the first fit and assumes richest-first.
    """
    # A card at or below 16 GiB is assumed to be running the math kernel,
    # because that is what every sm_75 card does and the ceiling costs a
    # larger card nothing: 24 GiB and up keeps the full ladder.
    if vram_gib is not None and vram_gib < MIN_VLM_VRAM_GIB:
        return None
    ceiling = (MATH_KERNEL_TOKEN_CEILING
               if vram_gib is not None and vram_gib < 16.0 else None)
    for label, n_frames, size, multiscale in plans:
        if ceiling is not None and vision_tokens(n_frames, size, multiscale) > ceiling:
            continue
        if estimated_seconds(n_frames, size, multiscale, seconds_per_1k) <= remaining_seconds:
            return {"label": label, "n_frames": n_frames, "size": size,
                    "multiscale": multiscale,
                    "tokens": vision_tokens(n_frames, size, multiscale),
                    "estimate": estimated_seconds(n_frames, size, multiscale,
                                                   seconds_per_1k)}
    return None


def describe(plan):
    if plan is None:
        return "no plan fits the remaining budget; skipping the VLM"
    return ("plan=%s frames=%d size=%d multiscale=%s tokens=%d est=%.0fs"
            % (plan["label"], plan["n_frames"], plan["size"],
               plan["multiscale"], plan["tokens"], plan["estimate"]))


def next_cheaper_plan(plan, plans=FRAME_PLANS,
                      seconds_per_1k=SECONDS_PER_1K_TOKENS):
    """The richest plan strictly cheaper than `plan`, or None.

    THE SECOND LINE OF DEFENCE, and it exists because getting a new container
    onto Grand Challenge is expensive. `MATH_KERNEL_TOKEN_CEILING` is measured
    rather than modelled, but it was measured on ONE model with ONE judge
    configuration; a card carrying something this probe did not see could
    still OOM at a plan the table calls safe. Without this the seam absorbs
    that OOM and the VLM goes silent for the whole run -- the exact failure
    being fixed -- and the next chance to correct it is another upload.

    With it, an OOM costs one wasted prefill and the answer still arrives.
    """
    if not plan:
        return None
    current = plan.get("tokens")
    if current is None:
        return None
    for label, n_frames, size, multiscale in plans:
        tokens = vision_tokens(n_frames, size, multiscale)
        if tokens < current:
            return {"label": label, "n_frames": n_frames, "size": size,
                    "multiscale": multiscale, "tokens": tokens,
                    "estimate": estimated_seconds(n_frames, size, multiscale,
                                                  seconds_per_1k)}
    return None
