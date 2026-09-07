"""Motion evidence at two timescales, computed rather than learned.

WHAT THIS IS FOR. The router answers "is tissue being cut?" with
`_answer_cutting`, which returns Yes if a cutting tool is CREDIBLE -- that is,
if scissors are visible. A scissors sitting idle in frame answers Yes. The
same shape appears in `_answer_suture`. Those are questions about an EVENT,
answered by a proxy for PRESENCE, and no amount of improving the tool head
fixes that: the tool head is right, it is being asked the wrong question.

Motion is the missing evidence, and the cheapest useful form of it needs no
model at all. Frame differencing over the burst pool answers "is anything
happening" directly, with no labels, no training, nothing to overfit, and
nothing to go stale when a checkpoint is retrained.

TWO TIMESCALES, because the pool holds two and they answer different questions.

    micro   within one burst -- 3 frames 67 ms apart, 0.2 s total. Is there
            motion AT this sampled moment. Sixteen of these per window.

    macro   between burst centres -- 1.875 s apart, spanning the full 30 s.
            Is the scene changing ACROSS the window. Fifteen of these.

They are not redundant. A window can be micro-still and macro-active (the
camera cut to a new view between samples, nothing moving within any sample) or
micro-active and macro-still (sustained work in one place, the scene as a
whole unchanged). The second is what active surgery looks like; the first is
what a repositioning or a scene change looks like.

WHAT THIS DOES NOT DISTINGUISH, stated plainly because it bounds every
conclusion drawn from it: camera motion from instrument motion. A scope push
moves every pixel and reads as high activity. Separating them needs flow or a
learned model, which is what the learned branches in surgvu/temporal.py are
for. This is the floor, not the ceiling -- but it is a floor available today,
against a router rule that currently has no motion evidence at all.

NORMALISATION. Frames are reduced to small grayscale before differencing: the
statistic should describe the surgical field, not JPEG noise, and downsampling
is a cheap low-pass that also makes the whole thing fast enough to be free.
Values are mean absolute difference in 0-255 units, so they are comparable
across cases without a calibration step -- and a calibration THRESHOLD, when
one is needed, is fitted on the training split and recorded, never guessed.

THE V2 RECORD (`motion_vector`, `motion_record_v2`, below) closes the gap the
paragraph above describes, by consuming `surgvu.flow.flow_features` alongside
the frame-difference statistics. Task 1 measured flow on two cases built to
be unambiguous: a global camera pan reads coherence=1.0000,
moving_fraction=1.0000; local tool motion reads coherence=0.7513,
moving_fraction=0.0822. moving_fraction separates the two cases by 12x,
coherence by only 1.33x, so moving_fraction is carried as the PRIMARY
camera-vs-tool discriminator in the v2 vector and coherence rides along as a
secondary signal -- the measurement is recorded here rather than a stronger
claim than the measurement supports.
"""
import numpy as np

#: Frames are reduced to this before differencing. 64x64 keeps instrument-scale
#: motion (an instrument tip crosses many of these pixels in 67 ms) while
#: discarding the JPEG ringing that dominates a raw pixel difference at 512.
WORK_SIZE = 64

#: Recorded in every motion block so a stored record can be told apart from one
#: computed by a later, different definition. The v4 lesson: a number whose
#: provenance is not written down gets compared to a number it is not
#: comparable with.
MOTION_VERSION = 1


def _to_work(frames):
    """(N, H, W, 3) uint8 -> (N, WORK_SIZE, WORK_SIZE) float32 grayscale.

    Strided subsampling rather than an interpolating resize: this needs no
    cv2, runs on any array, and for a low-pass ahead of a DIFFERENCE the exact
    resampling kernel does not matter. What matters is that every frame in a
    comparison gets the SAME treatment, which striding guarantees.
    """
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(
            "expected (N, H, W, 3) frames, got %r. Motion is computed on "
            "whole frames; handing this a batch axis would difference across "
            "windows." % (array.shape,))
    if array.shape[0] == 0:
        raise ValueError("no frames to compute motion from")
    # Rec. 601 luma. Any fixed weighting works for a difference statistic; this
    # one is the standard and keeps the numbers interpretable as brightness.
    gray = (0.299 * array[..., 2] + 0.587 * array[..., 1]
            + 0.114 * array[..., 0]).astype(np.float32)
    height, width = gray.shape[1], gray.shape[2]
    row = max(1, height // WORK_SIZE)
    col = max(1, width // WORK_SIZE)
    return gray[:, ::row, ::col]


def _pair_activity(work):
    """Mean |difference| between consecutive frames of one small stack."""
    if work.shape[0] < 2:
        return np.zeros(0, dtype=np.float32)
    return np.abs(np.diff(work, axis=0)).mean(axis=(1, 2)).astype(np.float32)


def micro_activity(frames, frames_per_burst):
    """Motion INSIDE each burst: one value per burst.

    `frames` is the window's full frame list in time order, bursts contiguous.
    Returns (bursts,) of mean absolute inter-frame difference within a burst --
    motion at the 67 ms scale, at each of the sampled moments.
    """
    work = _to_work(frames)
    depth = work.shape[0]
    if frames_per_burst < 2:
        raise ValueError(
            "a burst of %d frame(s) has no within-burst motion to measure"
            % frames_per_burst)
    if depth % frames_per_burst:
        raise ValueError(
            "%d frames is not a whole number of %d-frame bursts; differencing "
            "would straddle a boundary and call a 1.9 s gap a 67 ms motion"
            % (depth, frames_per_burst))
    bursts = depth // frames_per_burst
    out = np.empty(bursts, dtype=np.float32)
    for index in range(bursts):
        start = index * frames_per_burst
        out[index] = _pair_activity(work[start:start + frames_per_burst]).mean()
    return out


def macro_activity(frames, frames_per_burst):
    """Change BETWEEN burst centres: one value per adjacent pair.

    The centre frame is the one the 2D path samples, so this measures how the
    window changes across exactly the moments the appearance model reports on.
    Returns (bursts - 1,).
    """
    work = _to_work(frames)
    depth = work.shape[0]
    if depth % frames_per_burst:
        raise ValueError(
            "%d frames is not a whole number of %d-frame bursts"
            % (depth, frames_per_burst))
    bursts = depth // frames_per_burst
    if bursts < 2:
        raise ValueError(
            "a window of %d burst(s) has no across-burst change to measure"
            % bursts)
    centres = work[frames_per_burst // 2::frames_per_burst]
    return _pair_activity(centres)


def motion_record_from_bursts(bursts, centres=None):
    """Motion from a list of bursts, any of which may be missing.

    WHY IT TOLERATES A MISSING BURST. A training shard yields a clean
    (bursts x frames_per_burst) stack -- the extractor drops a window entirely
    rather than write a ragged one. Serving cannot do that: `decode_clip`
    already skips a frame whose seek fails rather than ending the clip, on the
    reasoning that one bad index late in a file should cost one frame and not
    every frame after it, and that reasoning does not stop applying because
    motion was added.

    So a burst whose flanking frame failed to read arrives as None, and it is
    EXCLUDED from the micro statistic rather than counted as stillness. A
    failed read is not evidence that nothing moved.

    `centres` is the sampled moments themselves, used for the macro statistic.
    It is passed separately because the centre frame survives even when its
    flanks do not -- which is the whole point of the serving contract: the
    appearance model must be exactly as robust as it is today.
    """
    # A burst with fewer than two frames is UNMEASURABLE, not still, and it is
    # excluded for the same reason None is. Differencing it yields an empty
    # array whose mean is NaN, and NaN is not valid JSON -- a strict parser
    # rejects the whole record, so the failure would surface as an unreadable
    # response rather than as a wrong number. Reachable only via per_burst=1,
    # which `decode_clip_bursts` permits; the serving default is 3.
    usable = [b for b in bursts
              if b is not None and np.asarray(b).shape[0] >= 2]
    micro = np.array(
        [float(_pair_activity(_to_work(np.asarray(b))).mean()) for b in usable],
        dtype=np.float32)

    macro = np.zeros(0, dtype=np.float32)
    if centres is not None:
        centre_array = np.asarray(centres)
        if centre_array.shape[0] >= 2:
            macro = _pair_activity(_to_work(centre_array))

    def summarise(values):
        if values.size == 0:
            return {"per": [], "mean": None, "max": None, "std": None}
        return {"per": [float(v) for v in values],
                "mean": float(values.mean()), "max": float(values.max()),
                "std": float(values.std())}

    micro_block = summarise(micro)
    macro_block = summarise(macro)
    return {
        "version": MOTION_VERSION,
        "bursts": len(bursts),
        # How much of the window the micro statistic actually saw. A record
        # summarising three of sixteen bursts is not the same measurement as
        # one summarising all sixteen, and a reader must be able to tell.
        "bursts_measured": len(usable),
        "micro": {"per_burst": micro_block["per"], "mean": micro_block["mean"],
                  "max": micro_block["max"], "std": micro_block["std"]},
        "macro": {"per_gap": macro_block["per"], "mean": macro_block["mean"],
                  "max": macro_block["max"], "std": macro_block["std"]},
    }


from .flow import flow_features

#: Distinct from MOTION_VERSION so a v2 record can never be read as a v1 one.
#: The v4 lesson, applied again: a number whose provenance is not written down
#: gets compared to a number it is not comparable with.
MOTION_V2_VERSION = 2

#: THE single source of truth for the probe offsets (controller ruling R16).
#: Chosen so the three micro slots span the 67ms-1875ms gap the shipped
#: sampler leaves. These are DEFAULTS; scripts/calibrate_motion_v2.py fits
#: them and writes the fitted set to config, and the record names the offsets
#: it actually used.
#:
#: This module is the home for that single definition -- not
#: `surgvu.perceive`, which imports torch at module scope, and not either
#: script, which deliberately avoids importing perceive for the same reason.
#: `perceive.DEFAULT_PROBE_OFFSETS_MS` and `scripts/dump_motion_v2.OFFSETS_MS`
#: both import this tuple rather than repeating the literal, so the three can
#: never drift against each other. tests/test_inference_motion_v2.py asserts
#: that directly; tests/test_dump_motion_v2.py checks the dump-side half.
#:
#: Before this, the same three numbers were typed as a literal in FOUR
#: places, one of which -- config/motion_v2.json's "offsets_ms" -- recorded
#: what the calibrator was ASSUMED to have used rather than what it was
#: actually handed. A config that asserts the wrong provenance is worse than
#: one that asserts none, because the number that exists to prevent exactly
#: this (MOTION_V2_VERSION) cannot catch a value that looks plausible and is
#: wrong. scripts/calibrate_motion_v2.py now carries the real offsets through
#: from the dump records instead of writing that literal.
PROBE_OFFSETS_MS = (133, 400, 1200)

#: Which probe offset feeds which slot of the vector. Built from
#: PROBE_OFFSETS_MS rather than typed alongside it, so the two cannot list
#: the same three numbers in different orders.
VECTOR_SLOTS = (("micro_short", PROBE_OFFSETS_MS[0]),
                ("micro_mid", PROBE_OFFSETS_MS[1]),
                ("micro_long", PROBE_OFFSETS_MS[2]))


def _mad(frame_a, frame_b):
    """Mean absolute difference between two frames, in 0-255 units."""
    work = _to_work(np.stack([np.asarray(frame_a), np.asarray(frame_b)]))
    return float(np.abs(work[1] - work[0]).mean())


def motion_vector(centre_before, centre, centre_after, probe_entry):
    """The nine-element motion description of one anchor.

    Every element is a float or None. NONE MEANS UNAVAILABLE, never still --
    the distinction v1 already makes and the one a threshold fitted over these
    numbers depends on. A missing flank averaged in as 0.0 would drag a case
    toward "quiet" on the strength of a failed disk read.

    `centre_before` / `centre_after` are the neighbouring ANCHORS (1.875 s
    away at the shipped sampling rate), so the macro slots describe how the
    scene changes across the moments the appearance model reports on. Either
    may be None at the ends of the window.

    Flow is computed on the SHORTEST available probe pair. Flow's assumption
    is small displacement; at 1.2 s a tool can cross the frame and the field
    stops meaning anything.

    Four of the nine slots come from flow_features: flow_mag_mean,
    flow_mag_p90, flow_coherence, and flow_moving_fraction. Task 1 measured
    that moving_fraction, not coherence, is the statistic that actually
    separates camera motion from instrument motion -- a global pan reads
    coherence=1.0000, moving_fraction=1.0000, while local tool motion reads
    coherence=0.7513, moving_fraction=0.0822. That is 12x separation on
    moving_fraction against 1.33x on coherence, so moving_fraction is carried
    here as the primary camera-vs-tool discriminator and coherence rides
    along as a secondary signal. Both are None together whenever no probe
    pair is available, for the same reason the other flow slots are.
    """
    vector = {}
    for name, offset in VECTOR_SLOTS:
        pair = (probe_entry or {}).get(offset)
        vector[name] = None if pair is None else _mad(pair[0], pair[1])

    vector["macro_prev"] = (None if centre_before is None
                            else _mad(centre_before, centre))
    vector["macro_next"] = (None if centre_after is None
                            else _mad(centre, centre_after))

    flow_pair = None
    for _, offset in VECTOR_SLOTS:
        pair = (probe_entry or {}).get(offset)
        if pair is not None:
            flow_pair = pair
            break
    if flow_pair is None:
        vector.update(flow_mag_mean=None, flow_mag_p90=None,
                      flow_coherence=None, flow_moving_fraction=None)
    else:
        flow = flow_features(flow_pair[0], flow_pair[1])
        vector.update(flow_mag_mean=flow["mag_mean"],
                      flow_mag_p90=flow["mag_p90"],
                      flow_coherence=flow["coherence"],
                      flow_moving_fraction=flow["moving_fraction"])
    return vector


_VECTOR_KEYS = ("micro_short", "micro_mid", "micro_long",
                "macro_prev", "macro_next",
                "flow_mag_mean", "flow_mag_p90", "flow_coherence",
                "flow_moving_fraction")


def motion_record_v2(centres, probes):
    """Per-anchor motion vectors plus a summary that excludes what it lacks.

    `centres` is the (n, H, W, 3) stack from decode_clip_multiscale; `probes`
    is its companion list. Returns a record whose "summary" reports, for every
    slot, the mean/max over the anchors that HAVE a measurement together with
    how many that was. A mean over three of sixteen anchors is not the same
    measurement as a mean over sixteen and a reader must be able to tell --
    the same reasoning behind v1's `bursts_measured`.
    """
    array = np.asarray(centres)
    if array.ndim != 4:
        raise ValueError("expected (N, H, W, 3) centres, got %r"
                         % (array.shape,))
    if len(probes) != array.shape[0]:
        raise ValueError(
            "%d probe entries for %d centres; a vector would be built from "
            "another anchor's neighbours"
            % (len(probes), array.shape[0]))

    per_anchor = []
    for index in range(array.shape[0]):
        before = array[index - 1] if index > 0 else None
        after = array[index + 1] if index + 1 < array.shape[0] else None
        per_anchor.append(
            motion_vector(before, array[index], after, probes[index]))

    summary = {}
    for key in _VECTOR_KEYS:
        values = [v[key] for v in per_anchor if v[key] is not None]
        summary[key] = {
            "mean": float(np.mean(values)) if values else None,
            "max": float(np.max(values)) if values else None,
            "measured": len(values),
        }

    return {
        "version": MOTION_V2_VERSION,
        "anchors": int(array.shape[0]),
        "offsets_ms": [offset for _, offset in VECTOR_SLOTS],
        "per_anchor": per_anchor,
        "summary": summary,
    }
