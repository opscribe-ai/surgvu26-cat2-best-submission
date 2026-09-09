# Split-discipline audit -- after R30

R30 found the variant head had been trained on the eleven graded evaluation cases. That prompted a
sweep of every script in this build that FITS or TRAINS anything. The problem was not isolated.

## The audit

| script | consults a split? | verdict |
|---|---|---|
| `scripts/train_variant.py` | `config/splits_v2.json` | **OK** -- fixed under R30 |
| `scripts/dump_motion_v2.py` | none | **BLOCKING** -- see below |
| `scripts/calibrate_motion_v2.py` | none | **BLOCKING** -- see below |
| `scripts/build_commercial_names.py` | defaults to `config/splits.json` | leaky v1 -- pre-existing, mild |
| `scripts/build_variant_labels.py` | none | acceptable, see below |
| `scripts/build_variant_priors.py` | none | pre-existing, unreviewed here |

## The two split files are NOT interchangeable

    config/splits.json     keys: ['val', 'train']          -- NO heldout key at all
    config/splits_v2.json  heldout: case_122 .. case_132   -- the 11 graded sample cases

All eleven graded cases sit inside `splits.json`'s train/val. `docs/compliance_audit.md` §3.2 already
warned that `train_tools.py` and `train_task.py` DEFAULT to this leaky v1 file and had to be passed
`splits_v2` explicitly for the shipped checkpoints to be clean. Any new script defaulting to
`config/splits.json` reproduces that trap.

## BLOCKING: do not run the full motion sweep yet

`dump_motion_v2.py` samples windows from all 155 cases, and `calibrate_motion_v2.py` fits cuts on
whatever dump it is handed. Neither excludes the graded eleven. Running the full 155-case sweep as
things stand would write `config/motion_v2.json` with thresholds fitted partly on the cases we grade
ourselves against -- the identical failure R30 caught in the variant head, in a second artifact.

Required before the sweep: `dump_motion_v2.py` (or the calibrator) must exclude
`config/splits_v2.json`'s heldout list, comparing with `surgvu.sampling.normalize_case_id` rather than
string equality, and must fail loudly if the exclusion removes zero cases. A silent zero-exclusion IS
the bug.

This stacks with ruling R17, which already said the sweep should not run until the autocorrelation
defect is fixed. Two independent reasons to hold.

## `build_commercial_names.py` -- pre-existing, mild, but the docstring is wrong

It defaults to the leaky v1 split, so `config/commercial_names.json` was built including the graded
cases, and its own docstring's claim of "Train split only ... val distribution must not leak into a
training-set decision" is false with respect to `splits_v2`.

Severity is LOW because that file is a SYNONYM TABLE -- commercial name to class -- used for parsing
question text, not for predicting from video. What leaks is vocabulary, not labels. But note that
Task 8 quoted counts from this file as "train-split" figures, and those counts include the graded
cases. Not worth regenerating on its own; worth correcting the claim.

## `build_variant_labels.py` -- acceptable as-is

It emits labels for all 152 cases with no split filter. That is defensible: it is a DATASET, not a
fitted artefact, and its consumer (`train_variant.py`) now performs the exclusion. Anything else that
consumes it must do the same. Worth one docstring line saying so, so the next consumer does not
inherit the assumption that the file is already clean.

## The general rule this build keeps rediscovering

Every artefact fitted on corpus data must name the split it was fitted on and must exclude the graded
eleven, and the exclusion must be loud when it removes nothing. Three separate components in this
plan have now produced, or nearly produced, a confident number that measured itself.
