# The measurements that decided v5.1 (2026-08-26/27)

Every number here was produced by `surgvu.scoring.Scorer` -- BERTScore-F1,
roberta-large, rescaled, MAX over five references -- the same metric Grand
Challenge uses. Kept because each one CLOSED a decision, and because the
per-case detail is what makes a future disagreement resolvable.

## The graded 11 predict leaderboard DELTAS to ~0.004

    11-case: base 0.8766 -> +yolo+variant 0.9309 (+0.0543)
                         -> +VLM challenger 0.8525 (-0.0784)
    predicted v5 = 0.8015 + 0.0543 - 0.0784 = 0.7774
    ACTUAL    v5 = 0.7737                     error -0.0037

Second confirmed prediction: the flag matrix measured `--motion-v2` at exactly
zero, and v4 -- which added the motion gate -- scored exactly 0.8015, unmoved.

They are a POOR estimator of the absolute score (0.93 local vs 0.80
leaderboard) and an EXCELLENT estimator of deltas, which is the only thing a
decision needs.

## Files

    cand_router_only.json        the 11 answers with the VLM unable to override
    score_router_only_v5.txt     -> MEAN 0.9309, case124 the only miss (0.2402)
    cand_vlm_challenger.json     the same 11 with the VLM overriding
    score_vlm_challenger_v5.txt  -> MEAN 0.8525
    judge_probe_result.txt       the decision VLM on the 4 disagreement cases
    evidence_probe_partition.json 651 agreeing / 45 contradicting held-out records

## What each closed

    challenger  -0.0784  rejected, then CONFIRMED by the leaderboard
    judge       -0.0486  rejected: picked the VLM's wrong answer on case122/124
    evidence    +0.0049  rejected: 0.30 standard errors, noise
    detector v2     n/a  killed: 300 epochs cannot finish before Sep 6

## The number that reframes all of it

Only **2.6%** of corpus questions (207/8000) reach an UNKNOWN_* intent, which
is the only place `fallback` consults the VLM at all. So ~97.4% of the score
is router + perception, and the +0.0543 variant-head gain is what carries
v5.1. v5 handed the VLM 100% of questions when it was better on 2.6%.
