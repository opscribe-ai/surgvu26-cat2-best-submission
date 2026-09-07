"""Choose which 30-second windows become training data.

The unit is a 30-second window sampled at 1 fps, because that is exactly the
test-time format. Windows are enumerated only inside task segments — that is
where the evaluation clips come from, and it is where the labels are defined.

Stratification is by TOOL CLASS rather than duration. Intervals and hours
disagree sharply in this dataset: clip applier appears in 895 intervals but
only 12.5 hours, because it is installed briefly and often. Sampling by
duration would under-represent it badly.
"""
import random
import re
from collections import namedtuple

from .taxonomy import TOOL_CLASSES

#: A window is `length` seconds of `case`'s part `part` starting at `start`.
#:
#: `length` is carried on the tuple rather than left to a module constant
#: because it is the only thing that says how many frames the window is: the
#: extractor derives its frame count from `window.length`, and the shard
#: writer derives its expected stack depth from the same field. Before
#: `length` existed, `enumerate_windows(length=15)` produced windows that were
#: then decoded as 30 seconds, and nothing in the pipeline could notice.
Window = namedtuple("Window", "case part start length task description tools")

WINDOW_SECONDS = 30.0


def enumerate_windows(case_id, labels, length=WINDOW_SECONDS, stride=WINDOW_SECONDS):
    """Every non-overlapping window that fits inside a task segment.

    Segments overlap in the real data, so a moment can be covered by more
    than one. `CaseLabels.task_at` resolves that by taking the SHORTEST
    covering segment — it is the most specific — and this function defers to
    it rather than reading `segment.task` off whichever segment it happens to
    be iterating. Reading the iterated segment produced windows whose task
    contradicted `task_at` at their own midpoint (155 of them across the 155
    real cases), which meant a window's label depended on CSV row order.

    Because overlapping segments can enumerate the same window twice, exact
    duplicates are dropped: decoding and storing the identical clip twice
    costs decode time and silently over-weights that clip in training.
    """
    windows = []
    seen = set()
    for segment in labels.task_segments():
        t = segment.start
        while t + length <= segment.stop:
            midpoint = t + length / 2.0
            resolved = labels.task_at(segment.part, midpoint)
            if resolved is None:        # cannot happen: midpoint is in segment
                t += stride
                continue
            task, description = resolved
            window = Window(
                case=case_id,
                part=segment.part,
                start=t,
                length=length,
                task=task,
                description=description,
                tools=frozenset(labels.tools_at(segment.part, midpoint)),
            )
            if window not in seen:
                seen.add(window)
                windows.append(window)
            t += stride
    return windows


def tool_frequency(windows):
    """Count of window-appearances per tool class, over all 12 classes.

    Every class is initialised to 0, including classes absent from
    `windows`, so a caller can rely on the shape of the returned dict — this
    is meant to be built once over a whole dataset's windows and handed to
    `stratify` as its `frequency` argument.
    """
    frequency = {name: 0 for name in TOOL_CLASSES}
    for w in windows:
        for tool in w.tools:
            frequency[tool] += 1
    return frequency


def stratify(windows, per_case_cap, seed=0, frequency=None):
    """Down-sample to per_case_cap, favouring rare tool classes.

    Windows are visited in ascending order of their rarest tool's count, so a
    window containing a tip-up fenestrated grasper is taken before one
    containing only needle drivers.

    `frequency` should be a dataset-wide table (e.g. from `tool_frequency`
    called over every case's windows) whenever the caller has one available —
    that is what "rarest tool" is supposed to mean. If `frequency` is
    omitted, it is computed from the `windows` passed in here instead, which
    makes rarity CASE-LOCAL: a tool that is rare across the dataset but
    happens to appear often in this particular case's windows will be
    treated as common and may be dropped, which is backwards. The fallback
    is only a faithful stand-in for dataset-wide rarity when `windows`
    already spans the whole dataset (e.g. in tests); per-case callers should
    pass a precomputed `frequency` table.
    """
    if len(windows) <= per_case_cap:
        return list(windows)

    if frequency is None:
        frequency = tool_frequency(windows)

    def rarity(window):
        if not window.tools:
            return 10 ** 9
        return min(frequency[tool] for tool in window.tools)

    rng = random.Random(seed)
    shuffled = list(windows)
    rng.shuffle(shuffled)                       # break ties reproducibly
    shuffled.sort(key=rarity)
    return shuffled[:per_case_cap]


def make_splits(case_ids, val_fraction=0.2, seed=7):
    """Split by CASE. Frames from one session are near-duplicates, so a
    frame-level split leaks and every number measured afterwards is fiction."""
    ids = sorted(case_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_fraction))
    return {"val": sorted(ids[:n_val]), "train": sorted(ids[n_val:])}


# ---------------------------------------------------------------------------
# v2: heldout exclusion + tool-stratified case assignment
# ---------------------------------------------------------------------------

#: How many val windows a tool class needs before its per-class F1 is a
#: measurement rather than a coin flip.
#:
#: With n val windows containing a class, flipping one window moves that
#: class's recall by 1/n. At n=25 that is 4 points; at n=13 (what `stapler`
#: had in config/splits.json) it is 7.7; at n=0 the F1 is not small but
#: UNDEFINED, and every scorer in this repo reports undefined as 0.0 — which
#: is how `tip-up fenestrated grasper` silently cost macro-F1 ~0.056 for a
#: reason that had nothing to do with the model. 25 is the smallest count at
#: which a per-class number is worth printing.
MIN_VAL_WINDOWS = 25

#: Cost of missing a class's val floor, per unit of relative deficit. Large
#: enough that no amount of proportional tidiness elsewhere can buy back a
#: class whose val count is zero.
_FLOOR_WEIGHT = 100.0

#: Weight on holding the overall val WINDOW share near `val_fraction`. Cases
#: differ in length, so an exactly-20%-of-cases split is not a 20%-of-windows
#: split; this term is what keeps the two from drifting apart.
_WINDOW_WEIGHT = 4.0

#: How far above its proportional val share a class may sit before the split
#: is reporting something other than a 1-in-5 sample of that class.
_DISTORTION = 1.5

_CASE_ID = re.compile(r"^case_?(\d+)$")


def normalize_case_id(raw):
    """`case122` and `case_122` are the same case; return the `case_122` form.

    The public sample directories are named `case122`; `config/splits.json`
    names its cases `case_122`. `set(sample_dirs) & set(split)` over the raw
    strings is therefore empty for a split that contains every one of them,
    which is exactly the reading that let all 11 sample cases sit inside
    train and val.

    Anything unparseable raises. Returning the input unchanged would be the
    same failure in a new place: a heldout id that quietly fails to normalise is
    a heldout id that quietly stays in the training set.
    """
    match = _CASE_ID.match(str(raw).strip())
    if match is None:
        raise ValueError(
            "cannot read %r as a case id. Expected 'case122' or 'case_122'; "
            "a name that does not normalise cannot be excluded, and an "
            "un-excluded heldout case is a leak that looks clean." % (raw,))
    return "case_%03d" % int(match.group(1))


def _tool_counts(windows):
    """Per-class window counts for one case, as a tuple over TOOL_CLASSES."""
    counts = [0] * len(TOOL_CLASSES)
    for window in windows:
        for tool in set(window):
            if tool not in _TOOL_POSITION:
                raise ValueError(
                    "window carries tool %r, which is not one of the %d "
                    "classes in TOOL_CLASSES. Counting it as nothing would "
                    "stratify the split against a vocabulary that is not the "
                    "one being scored." % (tool, len(TOOL_CLASSES)))
            counts[_TOOL_POSITION[tool]] += 1
    return tuple(counts)


_TOOL_POSITION = {name: i for i, name in enumerate(TOOL_CLASSES)}


def _split_cost(val_counts, val_windows, targets, floors, target_windows):
    """Lower is better. Two terms, in strict priority order.

    The floor term is a hard requirement expressed as a very large linear
    penalty: a class below its floor dominates everything else, so the search
    buys val positives for a rare class before it tidies up a common one. The
    proportional term is a squared RELATIVE deviation, so `stapler` at 2x its
    target counts the same as `cadiere forceps` at 2x its target — an
    absolute deviation would let the rare classes drift arbitrarily far in
    exchange for a rounding error on the common ones.
    """
    cost = 0.0
    for i, target in enumerate(targets):
        if target is None:                      # class absent from the corpus
            continue
        count = val_counts[i]
        floor = floors[i]
        if floor is not None and count < floor:
            cost += _FLOOR_WEIGHT * (floor - count) / floor
        cost += ((count - target) / target) ** 2
    cost += _WINDOW_WEIGHT * ((val_windows - target_windows) / target_windows) ** 2
    return cost


def _hill_climb(eligible, counts, totals, targets, floors, target_windows,
                n_val, seed, restarts, max_passes):
    """Best-improvement swap search over which cases are in val.

    Whole cases move, never windows: frames inside one session are
    near-duplicates, so a window-level split leaks. That makes this a
    constrained assignment problem rather than a sampling problem, and a
    single greedy pass has no way to undo an early choice that later turns
    out to have been the only case holding a rarer class. Swapping until no
    single exchange helps, from several seeded starts, does.
    """
    rng = random.Random(seed)
    best_val, best_cost, best_converged = None, None, False

    for _ in range(max(1, restarts)):
        order = list(eligible)
        rng.shuffle(order)
        val = set(order[:n_val])
        vector = [0] * len(TOOL_CLASSES)
        for case in val:
            for i, n in enumerate(counts[case]):
                vector[i] += n
        windows = sum(totals[case] for case in val)
        cost = _split_cost(vector, windows, targets, floors, target_windows)

        converged = False
        for _pass in range(max_passes):
            move, move_cost = None, cost
            for out in sorted(val):
                for into in sorted(c for c in eligible if c not in val):
                    trial = [vector[i] - counts[out][i] + counts[into][i]
                             for i in range(len(TOOL_CLASSES))]
                    trial_windows = windows - totals[out] + totals[into]
                    trial_cost = _split_cost(trial, trial_windows, targets,
                                             floors, target_windows)
                    if trial_cost < move_cost - 1e-12:
                        move, move_cost = (out, into), trial_cost
            if move is None:
                converged = True         # no single swap improves this split
                break
            out, into = move
            val.discard(out)
            val.add(into)
            vector = [vector[i] - counts[out][i] + counts[into][i]
                      for i in range(len(TOOL_CLASSES))]
            windows = windows - totals[out] + totals[into]
            cost = move_cost

        if best_cost is None or cost < best_cost - 1e-12:
            best_val, best_cost, best_converged = set(val), cost, converged

    return best_val, best_cost, best_converged


def make_splits_v2(case_windows, heldout_ids, val_fraction=0.2, seed=7,
                   min_val_windows=MIN_VAL_WINDOWS, restarts=8,
                   max_passes=200):
    """Split by CASE into train/val/heldout, stratified over TOOL_CLASSES.

    `case_windows` maps a case id to that case's windows, each window being
    the list of tool classes installed during it —
    `{"case_012": [["needle driver"], [], ...]}`. `heldout_ids` are the cases to
    hold out entirely; they may be spelled either way (`case122` or
    `case_122`) and every one of them must exist in `case_windows`, because a
    heldout id that matches nothing is indistinguishable from a heldout id that was
    successfully excluded.

    Returns `{"train": [...], "val": [...], "heldout": [...], "meta": {...}}`
    where `meta["window_counts"]` is RECOUNTED from the finished assignment,
    not carried over from the search. The point of the meta block is that a
    reader can audit the split without re-running the search, which it can
    only do if the numbers in it are observations rather than intentions.
    """
    corpus = {}
    for raw, windows in case_windows.items():
        case = normalize_case_id(raw)
        if case in corpus:
            raise ValueError(
                "two spellings of %s in the corpus (%r); its windows would be "
                "counted twice and the case could land in two splits at once"
                % (case, raw))
        corpus[case] = list(windows)

    heldout = sorted({normalize_case_id(d) for d in heldout_ids})
    missing = [d for d in heldout if d not in corpus]
    if missing:
        raise ValueError(
            "%d case(s) to hold out are not in the corpus: %s. An id that "
            "matches nothing excludes nothing, and the split then reports a "
            "clean 'no overlap' while every one of them is still in train."
            % (len(missing), ", ".join(missing)))

    counts = {case: _tool_counts(windows) for case, windows in corpus.items()}
    totals = {case: len(windows) for case, windows in corpus.items()}

    eligible = sorted(set(corpus) - set(heldout))
    n_val = int(round(len(eligible) * val_fraction))
    if not eligible or n_val < 1 or n_val >= len(eligible):
        raise ValueError(
            "cannot cut a val split from %d eligible case(s) at fraction %r: "
            "it would leave train or val empty" % (len(eligible), val_fraction))

    totals_by_class = [sum(counts[c][i] for c in eligible)
                       for i in range(len(TOOL_CLASSES))]
    cases_by_class = [sum(1 for c in eligible if counts[c][i])
                      for i in range(len(TOOL_CLASSES))]
    target_windows = val_fraction * sum(totals[c] for c in eligible)
    if target_windows <= 0:
        raise ValueError("no windows in the %d eligible cases" % len(eligible))

    targets, floors, wanted = [], [], []
    for i, total in enumerate(totals_by_class):
        if total == 0:
            targets.append(None)
            floors.append(None)
            wanted.append(None)
            continue
        targets.append(val_fraction * total)
        # Half the class's windows is the ceiling on what val may demand: a
        # floor above that would strip the training set of the very class it
        # was trying to make measurable.
        floor = max(1, min(min_val_windows, total // 2))
        wanted.append(floor)
        # A class living in ONE case cannot be in both splits. Forcing that
        # case into val to satisfy the floor takes the class's only training
        # windows with it, so the floor is not enforced — it is reported.
        floors.append(floor if cases_by_class[i] >= 2 else None)

    val, search_cost, converged = _hill_climb(
        eligible, counts, totals, targets, floors, target_windows, n_val,
        seed, restarts, max_passes)
    train = sorted(set(eligible) - val)
    val = sorted(val)

    members = {"train": train, "val": val, "heldout": heldout}
    window_counts = {}
    for i, name in enumerate(TOOL_CLASSES):
        window_counts[name] = {
            split: sum(sum(1 for w in corpus[c] if name in w) for c in cases)
            for split, cases in members.items()}

    # "absent" is absent from the SPLITTABLE pool: a class that appears only
    # inside heldout cases is unavailable to train and val alike, which is the
    # same problem for anything measured on them.
    absent = [n for i, n in enumerate(TOOL_CLASSES) if totals_by_class[i] == 0]
    unsplittable = [n for i, n in enumerate(TOOL_CLASSES)
                    if cases_by_class[i] == 1]
    underrepresented = [n for i, n in enumerate(TOOL_CLASSES)
                        if wanted[i] is not None
                        and window_counts[n]["val"] < wanted[i]]
    # Cases move whole, so a class that lives in two long cases can only take
    # a val share of 0% or 50% — there is no 20% to hit. Meeting the floor at
    # 50% is the right trade against an F1 of 0.0, but it is a distortion of
    # the split, and presenting it beside the classes that did land on 20%
    # without saying so would misrepresent what the val number measures.
    over_represented = []
    for i, name in enumerate(TOOL_CLASSES):
        pool = window_counts[name]["train"] + window_counts[name]["val"]
        if pool and window_counts[name]["val"] > _DISTORTION * val_fraction * pool:
            over_represented.append(name)

    meta = {
        "generator": "surgvu.sampling.make_splits_v2",
        "seed": seed,
        "val_fraction": val_fraction,
        "min_val_windows": min_val_windows,
        "restarts": restarts,
        # The objective value the published split achieved, so a later
        # attempt can be compared against it instead of argued about. Each
        # restart is an independent climb and the best is kept, so raising
        # `restarts` can only lower this number, never raise it.
        "search_cost": search_cost,
        # True means the search stopped because NO single case swap improves
        # the split any further — the published assignment is a local optimum
        # of the objective, not wherever the loop happened to run out of
        # passes. A heuristic search that was cut short is exactly the thing
        # a reader of this file cannot otherwise detect.
        "local_optimum": converged,
        "method": (
            "Cases — never windows — are assigned by best-improvement swap "
            "search from %d seeded starts, minimising a cost that (1) "
            "penalises any tool class holding fewer than min_val_windows val "
            "windows and (2) squares each class's relative deviation from a "
            "val share of val_fraction, plus a term holding the overall val "
            "window share near val_fraction." % restarts),
        "heldout_rationale": (
            "Held out entirely: %s. These are the 11 public sample cases, the "
            "only question-and-answer data that exists for this task. All of "
            "them were inside config/splits.json (8 train, 3 val); the sample "
            "directories spell them 'case122' and the split spells them "
            "'case_122', so a raw set intersection reports no overlap. "
            "Excluding them makes them a legitimate held-out QA set."
            % ", ".join(heldout)),
        "case_counts": {"train": len(train), "val": len(val), "heldout": len(heldout)},
        "window_counts": window_counts,
        "window_totals": {
            split: sum(totals[c] for c in cases)
            for split, cases in members.items()},
        "min_val_windows_per_class": {
            name: wanted[i] for i, name in enumerate(TOOL_CLASSES)
            if wanted[i] is not None},
        "cases_per_class": {name: cases_by_class[i]
                            for i, name in enumerate(TOOL_CLASSES)},
        "absent": absent,
        "unsplittable": unsplittable,
        "underrepresented": underrepresented,
        "over_represented": over_represented,
    }
    if over_represented:
        meta["distortion_note"] = (
            "%s take a val share more than %gx the %g target. Each lives in "
            "too few cases for any whole-case assignment to hit the target: "
            "see cases_per_class. The alternative was a val count below "
            "min_val_windows, which scores as an F1 of 0.0 and measures "
            "nothing, so the over-sampling is deliberate — but the val "
            "numbers for these classes rest on fewer distinct cases than the "
            "rest, and their train counts are correspondingly thinner."
            % (", ".join(over_represented), _DISTORTION, val_fraction))
    if underrepresented:
        meta["underrepresented_note"] = (
            "%s did not reach min_val_windows val windows. A class is capped "
            "at half its windows (it must stay learnable) and a class living "
            "in a single case cannot be in two splits at once, so for these "
            "the val count is the best the corpus allows — the number is "
            "recorded here rather than silently accepted."
            % ", ".join(underrepresented))

    return {"train": train, "val": val, "heldout": heldout, "meta": meta}
