"""Cache REAL, MODEL-PRODUCED evidence packets for every distinct window in
Task 3's `qa_frames_manifest.jsonl` (v5 plan3's Task 4 follow-up).

WHY THIS SCRIPT EXISTS IN THIS FORM -- read this before touching the block
list below. `scripts/train_vlm.py` trains against an EMPTY evidence context
(`build_sampling_prompt(question, {})`) because Task 3 extracted frames only
and never ran the CNN/YOLO/motion/variant stack over the 15,087 sampled
training windows -- see that script's module docstring, "THE EVIDENCE PACKET
IS DELIBERATELY EMPTY AT TRAIN TIME". Synthesising an evidence packet from
the GROUND-TRUTH tool/task labels this corpus was generated from was
considered there and REJECTED: `evidence_vlm._render_tools_block` would then
render the LITERAL ANSWER to the tool_presence/tool_identity/count question
being asked about that SAME window, and the model would learn to read the
evidence line instead of the pixels -- the opposite of what a perception
fine-tune is for.

This script is the honest alternative that rejection left open: run the
SAME perception models `scripts/inference.py` runs at serving time --
`surgvu.perceive` (the CNN tool/task heads), `surgvu.detect.Detector` (the
YOLO second opinion), `surgvu.variant.VariantHead` (Large-vs-Mega),
`surgvu.motion.motion_record_v2` (the nine-slot motion vector) and
`surgvu.agreement.agreement_record` (CNN-vs-YOLO disagreement) -- over every
window, and cache whatever they actually produce. THE CRITICAL PROPERTY, and
it is the whole point of writing this as a separate perception pass instead
of a label-lookup: if the detector misses a tool, or the CNN heads call the
wrong task, the cached evidence says so, exactly as it would at serving on
an ungraded video with no ground truth anywhere nearby. A follow-up fine-tune
trained on this noisy, model-produced evidence learns to be robust to
evidence that is sometimes wrong -- which is the only kind of evidence it
will ever see at inference time. A fine-tune trained on evidence read
straight out of tools.csv/tasks.csv would instead learn to trust the
evidence line unconditionally, and that trust is misplaced on every window
the perception stack gets wrong -- which, per this project's own measured
macro-F1s (config/perception.json), is not rare. This script therefore never
opens tools.csv, tasks.csv, or any other ground-truth label file; its only
inputs are the manifest (for which windows to run) and the video corpus
itself.

THE OUTPUT is one JSONL record per DISTINCT window -- `{case, part, t_start,
t_stop, evidence}` -- where `evidence` carries EXACTLY the shape
`surgvu.perceive.clip_record` returns at serving time: `tools`/
`tools_present`/`task`/`task_top`/`n_frames`, plus whichever of
`motion_v2`/`yolo`/`variant`/`agree` actually ran on that window. That is
what makes `evidence_vlm.build_sampling_prompt(question, evidence)` render
IDENTICALLY to how it renders `perception` at serving -- the entire reason a
follow-up `scripts/train_vlm.py` retrain against this cache is what finally
lets `config/arbiter.json`'s `vlm_evidence_context` flip to `true` with
training and serving still matching. `motion` (v1, burst-based) is
deliberately NOT part of that shape: the task this cache exists for names
only the CNN probabilities, YOLO, variant and motion_v2 among the evidence
blocks, and v1 motion would need its own separate `decode_clip_bursts` pass
this cache never pays for.

REUSE, NOT REIMPLEMENTATION. `scripts/inference.py`'s `infer()` (the
mandatory CNN blocks) and `add_evidence()` (the best-effort yolo/variant/
agree blocks) already assemble exactly these blocks in exactly this order,
with the R18 best-effort wrappers this file mirrors line for line --
`cache_one_window` below IS that same sequence, with every granular
primitive (`predict_window_frames`, `clip_record`, `expert_meta`,
`reduce_frames`, `Detector.detect`, `detections_to_record`,
`VariantHead.predict`, `_needle_driver_boxes`, `motion_record_v2`,
`agreement_record`, `CLIP_SECONDS`) imported from `scripts/inference.py` and
`src/surgvu/*` UNMODIFIED, never re-derived. Divergence between how evidence
is built here and how it is built at serving would silently reintroduce the
exact train/serve mismatch this whole exercise exists to remove. The ONE
structural difference from `inference.py` is deliberate and explained where
it happens (`load_experts`): that script loads every checkpoint fresh on
every call because it runs once per case, inside one container, against a
10-minute-per-case budget that explicitly does not amortise model load (see
its own module docstring, BUDGET); this script instead processes 15,087
windows against the SAME two CNN experts (plus one detector, one variant
head) in a single long-running process, so it loads each of those exactly
once and reuses them across every window. Nothing about WHICH function does
the loading, or what it does once loaded, differs.

TWO GUARANTEES THIS SCRIPT RE-VERIFIES INDEPENDENTLY, NOT TRUSTED FROM
UPSTREAM (ruling R30). `qa_frames_manifest.jsonl` was already built by
`scripts/build_qa_pairs.py --extract-frames` from a corpus that had
`config/splits_v2.json`'s 11 heldout (graded) cases excluded -- so by
construction this manifest should contain ZERO of them. "Should" is exactly
the word ruling R30 says not to trust: the variant head was once trained on
the graded cases (0.9011 contaminated vs 0.8681 clean) because an earlier
exclusion step silently matched nothing. `verify_manifest_clean` (reused,
unmodified, from `scripts/train_vlm.py` -- the SAME independent check that
script runs before training) normalises every case id with
`surgvu.sampling.normalize_case_id` (never raw string equality: the public
sample dirs spell a case `case122`, this repo's labels spell it `case_122`)
and raises if any manifest case normalises into the heldout set or into a
case `config/splits_v2.json` does not know at all. `verify_heldout_excluded`
below is the complementary half ruling R30 asks this script to add: it
raises loudly if that exclusion removed ZERO of the 11 configured heldout
cases -- i.e. if `heldout_norm - present_norm` (the heldout ids this
manifest does NOT contain) is empty. A zero-exclusion here is
indistinguishable from a `normalize_case_id` comparison that silently
matches nothing at all -- exactly the failure mode that produced the
contaminated variant-head run -- and would otherwise pass its parent
manifest through as "no leakage" for the wrong reason.

R28 -- how a window resolves to a video file. Reused, unmodified, from
`scripts/build_qa_pairs.py`: `resolve_window_video` builds the expected
filename directly from the window's OWN recorded `case`/`part` fields (never
by probing a case's video files and accepting whichever one's duration
happens to cover the timestamp -- measured to silently pair the WRONG part's
pixels with a window on 28% of a multi-part sample; see
`scripts/train_variant.py`'s module docstring for the full measurement) and
returns `(None, reason)` rather than a fallback file when the exact expected
file is not on disk.

WHAT COULD NOT BE RUN OR TESTED HERE. This login node has no `torch`
installed (confirmed: `import torch` fails), and running heavy work on it is
forbidden regardless (see this repo's own "no heavy work on the login node"
guardrail). Every function that imports torch, `surgvu.perceive`,
`surgvu.detect`, `surgvu.variant`, `surgvu.motion`, `surgvu.agreement`, or
`scripts/inference.py` lives INSIDE a function body below the "TORCH-
TOUCHING" section header, never at module scope, so everything above that
header -- manifest loading, window dedup, the R30 guard, the record shape,
`window_key`/resume bookkeeping, and R28's video-path construction (via
`build_qa_pairs.resolve_window_video`/`frame_index_range`, both torch-free:
`build_qa_pairs.py` only imports `cv2` at module scope, deferring its own
`surgvu.perceive` import to inside `_decode_window_frames`) -- is importable
and exercised directly by `tests/test_cache_evidence.py` on this login node.
What is NOT, and cannot be, exercised here: actually loading the tool/task/
detector/variant checkpoints, decoding real video frames
(`decode_clip_multiscale`), and every model forward pass. `condor/
cache_evidence.sub` is what actually exercises those, on a GPU execute node.
"""
import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_qa_pairs import (                                        # noqa: E402
    VIDEO_ROOT, distinct_windows, frame_index_range, resolve_window_video,
)
from train_vlm import (                                             # noqa: E402
    load_case_universe, load_manifest, verify_manifest_clean,
)

REPO = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = "/staging/n/nkalthoff/surgvu26/qa_frames_manifest.jsonl"
DEFAULT_SPLITS = str(REPO / "config" / "splits_v2.json")
DEFAULT_PERCEPTION_CONFIG = str(REPO / "config" / "perception.json")
DEFAULT_VARIANT_CONFIG = str(REPO / "config" / "variant_head.json")
DEFAULT_OUT = "/staging/n/nkalthoff/surgvu26/evidence_cache.jsonl"
DEFAULT_REPORT_OUT = "/staging/n/nkalthoff/surgvu26/evidence_cache_report.json"

# Same corpus, same detector weights, as every other job in this repo that
# already runs the detector/variant head against real clips (condor/
# detect_sample.sub, condor/variant_sample.sub) -- not re-derived.
DEFAULT_YOLO_WEIGHTS = "/staging/groups/bhaskar_opscribe/surgvu_yolo_detector/best.pt"
DEFAULT_YOLO_REPO = "/staging/groups/bhaskar_opscribe/surgvu_yolo_detector/yolov5"
DEFAULT_VARIANT_WEIGHTS = "/staging/n/nkalthoff/surgvu26/models/variant_head.pt"


def log(message):
    """Everything this file says goes to stderr -- same convention as
    scripts/inference.py's own `log`, so a container's combined stdout/
    stderr log reads the same way whether it is running that file or this
    one."""
    print("[cache_evidence] %s" % (message,), file=sys.stderr, flush=True)


class WindowDecodeError(Exception):
    """A window's frames could not be decoded at all (a bad seek, a
    corrupted span, an out-of-range index). Distinguished from
    `WindowCnnError` so the drop report can tell "never got pixels" apart
    from "got pixels, the appearance models failed" -- two different bugs
    with different fixes."""


class WindowCnnError(Exception):
    """The MANDATORY tool/task CNN heads failed on an otherwise-decoded
    window. Unlike a yolo/variant/motion_v2/agree failure (R18: absent, not
    fatal), there is no perception record at all without these two blocks --
    `surgvu.perceive.clip_record` takes them as required arguments, not
    optional ones -- so a window that raises here is dropped, mirroring how
    a CNN failure in `scripts/inference.py`'s `infer_with_retry` falls
    through to the whole-pipeline fallback rather than producing a partial
    perception record."""


# ============================================================================
# TORCH-FREE: manifest loading, window dedup, the R30 guard, the record
# shape, resume bookkeeping. Every function in this section is exercised
# directly by tests/test_cache_evidence.py on this login node.
# ============================================================================


def enumerate_distinct_windows(records):
    """One dict per distinct `(case, part, t_start, t_stop)` window among
    `records`, sorted for a reproducible, resumable processing order.

    Reuses `build_qa_pairs.distinct_windows` -- Task 3's own window-dedup
    primitive, built for the identical reason (many QA records share one
    30 s clip: tool presence, task, organ, count, ... are all asked about
    the SAME window) -- rather than re-deriving the `(case, part, t_start,
    t_stop)` grouping key a second, possibly-divergent way. 23,355 manifest
    records collapse onto 15,087 distinct windows this way (confirmed
    against the real manifest at /staging/n/nkalthoff/surgvu26/
    qa_frames_manifest.jsonl).
    """
    groups = distinct_windows(records)
    windows = []
    for key in sorted(groups, key=lambda k: (k[0], str(k[1]), float(k[2]), float(k[3]))):
        case, part, t_start, t_stop = key
        windows.append({"case": case, "part": part,
                        "t_start": t_start, "t_stop": t_stop})
    return windows


def verify_heldout_excluded(present_norm, heldout_norm):
    """FAIL LOUDLY (ruling R30) if comparing this manifest's own case set
    against the configured heldout list -- via `normalize_case_id` on both
    sides (already applied by the caller; see `train_vlm.load_case_universe`
    / `verify_manifest_clean`), never raw string equality -- shows that the
    exclusion removed ZERO cases.

    `verify_manifest_clean` (called immediately before this, its return
    value is `present_norm`) already raises if any heldout case IS present
    in the manifest -- that catches exclusion failing outright. This is the
    complementary half: `heldout_norm - present_norm` is the set of
    configured heldout ids this manifest does NOT contain, i.e. the ones
    actually excluded. On a correctly built manifest that is all 11 of
    them. If it is EMPTY, either every configured heldout id also happens
    to be present (which `verify_manifest_clean` would already have raised
    on) or -- the case this function exists to catch on its own --
    `heldout_norm`/`present_norm` failed to line up at all (an empty
    `heldout_norm`, already guarded by `load_case_universe`, or a
    normalisation mismatch), which would make "no leakage found" trivially,
    vacuously true rather than a real verification. Mirrors
    `build_qa_pairs.select_cases`'s own `if not excluded: raise` guard,
    which catches the identical failure shape one step upstream, over the
    raw label directory rather than this manifest.

    Breaks if: this is replaced by a bare `assert heldout_norm` (which
    `load_case_universe` already guarantees and would pass even when this
    manifest's own case set happens to equal the full heldout set), or the
    `-` is flipped to `&` (which would raise on the NORMAL, healthy case
    instead of the broken one).
    """
    excluded_norm = heldout_norm - present_norm
    if not excluded_norm:
        raise RuntimeError(
            "heldout exclusion check against %d configured heldout case(s) "
            "found NONE of them absent from this manifest's %d distinct "
            "case(s) -- ruling R30: a zero-exclusion here is "
            "indistinguishable from a broken normalize_case_id comparison "
            "(or a corrupted splits file) and this refuses to trust a "
            "'no leakage' verdict that would otherwise be vacuous."
            % (len(heldout_norm), len(present_norm)))
    return excluded_norm


def window_key(case, part, t_start, t_stop):
    """The identity of one window, for resume/dedup bookkeeping.

    Plain field equality on `(case, part, t_start, t_stop)` -- the same
    tuple `enumerate_distinct_windows`/`build_qa_pairs.distinct_windows`
    already group by. `t_start`/`t_stop` round-trip through
    `json.dumps`/`json.loads` exactly (Python's float repr is the shortest
    decimal string that reads back to the identical float), so a key built
    from a manifest record and one built from a line THIS script already
    wrote for the same window compare equal.
    """
    return (str(case), str(part), float(t_start), float(t_stop))


def load_cached_keys(out_path):
    """Window keys already written to `out_path`, or an empty set if it
    does not exist yet.

    A resubmitted job after eviction skips the (expensive, GPU-bound) work
    for every window it already cached -- mirroring `condor/train_vlm.sub`'s
    own resume-from-checkpoint convention, for the identical reason: this
    job runs for hours over 15,087 windows and eviction is a real, observed
    risk on this cluster (see condor/train_vlm.sub's own comment on
    cluster 9629572 and the dbrundage-chtcgpu5000 staging-mount failure).
    A malformed line (a partial write from a job killed mid-line) is
    skipped rather than raising -- `handle.flush()` after every write in
    `process_manifest` keeps this rare, but a corrupt trailing line must
    never make an otherwise-valid resume refuse to start.
    """
    path = Path(out_path)
    keys = set()
    if not path.exists():
        return keys
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                log("WARNING: %s has a malformed line; skipping it for "
                    "resume purposes (it will not block the rest of the "
                    "cache from loading)" % (out_path,))
                continue
            keys.add(window_key(record["case"], record["part"],
                                record["t_start"], record["t_stop"]))
    return keys


def build_record(case, part, t_start, t_stop, evidence):
    """The one JSONL record this script writes per window.

    `evidence` carries EXACTLY the shape `surgvu.perceive.clip_record`
    returns at serving time -- `tools`/`tools_present`/`task`/`task_top`/
    `n_frames` plus whichever of `motion_v2`/`yolo`/`variant`/`agree`
    actually ran -- so `evidence_vlm.build_sampling_prompt(question,
    evidence)` renders identically to how it renders `perception` at
    serving. This function does not shape `evidence` itself; it only
    assembles the four window-identity fields around whatever
    `cache_one_window` (or a test) hands it, so there is exactly one place
    where the packet's shape is decided.
    """
    return {"case": case, "part": part, "t_start": t_start,
            "t_stop": t_stop, "evidence": evidence}


# ============================================================================
# TORCH-TOUCHING: model loading and per-window perception. NOT importable or
# exercisable on this login node -- every import of torch, surgvu.perceive,
# surgvu.detect, surgvu.variant, surgvu.motion, surgvu.agreement, or
# scripts/inference.py lives inside a function body below, never at module
# scope, so the section above (and its tests) stays torch-free regardless of
# whether this section could ever run here.
# ============================================================================


def load_experts(config, device, models_dir=None):
    """Every checkpoint `inference.expert_checkpoints` names for both the
    `tools` and `task` experts, loaded EXACTLY ONCE and reused across every
    window.

    `scripts/inference.py`'s own `infer()` reloads these from disk on every
    call because it runs once per case, inside a single container, against
    a 10-minute-per-case budget that explicitly does not amortise model
    load (see its module docstring, BUDGET). This script instead processes
    15,087 windows against the SAME two experts in one long-running
    process; reloading either checkpoint's weights from disk 15,087 times
    would be pure waste that job's design has no reason to pay for and this
    one does not need to. `inference.load_bound_expert` itself -- the
    binding checks (class order, image_size, threshold drift) it runs on
    every load -- is imported and called unmodified.
    """
    import inference
    loaded = {}
    for role in ("tools", "task"):
        entry = config["experts"][role]
        loaded[role] = [
            inference.load_bound_expert(dict(entry, checkpoint=checkpoint),
                                        device, models_dir)
            for checkpoint in inference.expert_checkpoints(entry)]
    return loaded


def load_detector(weights, repo, device):
    """A `surgvu.detect.Detector` bound to `weights`/`repo`, or `None` if
    `weights` is falsy -- the same "off unless configured" contract
    `inference.py`'s `--yolo` flag gives it, resolved once here rather than
    gated behind a per-case flag."""
    if not weights:
        return None
    from surgvu.detect import Detector
    return Detector(weights, repo, device=device)


def load_variant_head(weights, variant_config_path, device):
    """A `surgvu.variant.VariantHead` bound to `weights` and the FITTED
    cutoff in `variant_config_path`, or `None` if `weights` is falsy.
    Reads the cutoff from config at call time -- never hardcodes it --
    exactly as `inference.add_evidence` does."""
    if not weights:
        return None
    from surgvu.variant import VariantHead
    head_config = json.loads(
        Path(variant_config_path).read_text(encoding="utf-8"))
    return VariantHead(weights, head_config["cutoff"], device=device)


def cache_one_window(video_path, first, last, config, experts, detector,
                     variant_head, device, decode_frames, decode_size):
    """The evidence packet for one window.

    Mirrors `scripts/inference.py`'s `infer()` (the mandatory tools/task
    CNN blocks, immediately followed by `clip_record`) and then
    `add_evidence()` (the best-effort yolo/agree/variant blocks), in the
    SAME order, with the SAME failure semantics: a decode or CNN failure
    raises (`WindowDecodeError`/`WindowCnnError`) and the caller drops the
    window entirely -- there is no perception record at all without pixels
    or without both CNN heads, `clip_record` takes them as required
    arguments, not optional ones. A yolo/variant/motion_v2/agree failure is
    caught here, logged (a traceback plus a human-readable WARNING), and
    leaves the corresponding key simply ABSENT -- R18, copied line for line
    from `add_evidence`'s own try/except blocks around each block, never
    partial and never able to cost the case its CNN-backed evidence.

    `motion` (v1, burst-based) is deliberately not computed here -- see the
    module docstring's "THE OUTPUT" section for why only motion_v2 is part
    of this cache's shape.
    """
    from surgvu.perceive import clip_record, decode_clip_multiscale
    from surgvu.predict import predict_window_frames
    import inference

    try:
        frames, probes = decode_clip_multiscale(
            str(video_path), n_frames=decode_frames, size=decode_size,
            index_range=(first, last))
    except Exception as exc:                                   # noqa: BLE001
        raise WindowDecodeError(str(exc)) from exc

    # MANDATORY: the tool/task CNN heads. NOT wrapped R18-style -- see this
    # function's docstring and WindowCnnError's.
    try:
        per_role = {}
        for role in ("tools", "task"):
            entry = config["experts"][role]
            per_model = [predict_window_frames(model, frames, device,
                                               entry["image_size"],
                                               activation=entry["activation"])
                        for model in experts[role]]
            per_role[role] = inference.reduce_frames(per_model, entry, role)
    except Exception as exc:                                   # noqa: BLE001
        raise WindowCnnError(str(exc)) from exc

    motion_v2 = None
    try:
        from surgvu.motion import motion_record_v2
        motion_v2 = motion_record_v2(frames, probes)
    except Exception:                            # noqa: BLE001 - R18
        traceback.print_exc(file=sys.stderr)
        motion_v2 = None
        log("WARNING: motion_v2 failed for this window; continuing without it")

    perception = clip_record(
        per_role["tools"], inference.expert_meta(config["experts"]["tools"]),
        per_role["task"], inference.expert_meta(config["experts"]["task"]),
        len(frames), motion_v2=motion_v2)

    yolo_record = None
    if detector is not None:
        try:
            from surgvu.detect import detections_to_record
            found = detector.detect(frames)
            stamps = [i * (inference.CLIP_SECONDS / max(1, len(frames)))
                     for i in range(len(frames))]
            yolo_record = detections_to_record(found, stamps)
            perception["yolo"] = yolo_record
        except Exception:                        # noqa: BLE001 - R18
            traceback.print_exc(file=sys.stderr)
            yolo_record = None
            log("WARNING: the detector failed for this window; continuing "
                "without it")

        # Its OWN try/except, deliberately not folded into the yolo block
        # above (same reasoning as inference.add_evidence): an agreement
        # failure must never discard a yolo record that already succeeded.
        # Only attempted when a yolo record exists at all.
        if yolo_record is not None:
            try:
                from surgvu.agreement import agreement_record
                tool_meta = inference.expert_meta(config["experts"]["tools"])
                tool_thresholds = dict(zip(tool_meta["classes"],
                                           tool_meta["thresholds"]))
                perception["agree"] = agreement_record(
                    perception["tools"], tool_thresholds, yolo_record)
            except Exception:                    # noqa: BLE001 - R18
                traceback.print_exc(file=sys.stderr)
                log("WARNING: agreement computation failed for this "
                    "window; continuing without it")

    if variant_head is not None:
        try:
            boxes = (inference._needle_driver_boxes(yolo_record)
                    if yolo_record else {})
            perception["variant"] = variant_head.predict(frames, boxes or None)
        except Exception:                        # noqa: BLE001 - R18
            traceback.print_exc(file=sys.stderr)
            log("WARNING: the variant head failed for this window; "
                "continuing without it")

    return perception


def process_manifest(args):
    """The whole run: manifest -> distinct windows -> one evidence-cache
    JSONL record per window, plus a drop-count report. The only function
    that touches torch (via the imports inside the functions it calls)."""
    records = load_manifest(args.manifest)
    train_norm, val_norm, heldout_norm = load_case_universe(args.splits)
    present_norm = verify_manifest_clean(records, train_norm, val_norm, heldout_norm)
    excluded_norm = verify_heldout_excluded(present_norm, heldout_norm)
    log("R30: %d configured heldout case(s), %d confirmed absent from this "
        "manifest's %d distinct case(s); verify_manifest_clean raised "
        "already if any were present."
        % (len(heldout_norm), len(excluded_norm), len(present_norm)))

    windows = enumerate_distinct_windows(records)
    log("%d manifest record(s) collapse onto %d distinct window(s)"
        % (len(records), len(windows)))

    already = load_cached_keys(args.out)
    if already:
        log("resuming: %d window(s) already cached in %s"
            % (len(already), args.out))

    import inference
    config = inference.load_config(args.perception_config)
    decode_frames = args.frames or config["decode"]["frames"]
    decode_size = args.size or config["decode"]["size"]

    device = args.device
    if device in (None, "", "auto"):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log("device=%s decode_frames=%d decode_size=%d"
        % (device, decode_frames, decode_size))

    experts = load_experts(config, device, args.models_dir)
    detector = (load_detector(args.yolo_weights, args.yolo_repo, device)
               if args.yolo else None)
    variant_head = (load_variant_head(args.variant_weights, args.variant_config, device)
                    if args.variant_head else None)
    log("yolo=%s variant_head=%s" % (detector is not None, variant_head is not None))

    drops = Counter()
    video_cache = {}
    written = 0
    processed_this_run = 0
    started = time.time()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as handle:
        for index, window in enumerate(windows):
            case, part = window["case"], window["part"]
            t_start, t_stop = window["t_start"], window["t_stop"]
            key = window_key(case, part, t_start, t_stop)
            if key in already:
                continue
            if args.limit is not None and processed_this_run >= args.limit:
                log("--limit %d reached; stopping with %d/%d window(s) "
                    "left unprocessed this run"
                    % (args.limit, len(windows) - index, len(windows)))
                break
            processed_this_run += 1

            info, reason = resolve_window_video(args.video_root, case, part,
                                                video_cache)
            if info is None:
                drops[reason] += 1
                continue

            span = frame_index_range(t_start, t_stop, info.fps, info.total)
            if span is None:
                drops["index_range_invalid"] += 1
                continue
            first, last = span

            try:
                evidence = cache_one_window(
                    info.path, first, last, config, experts, detector,
                    variant_head, device, decode_frames, decode_size)
            except WindowDecodeError:
                traceback.print_exc(file=sys.stderr)
                drops["decode_failed"] += 1
                log("WARNING: %s part %s [%.3f, %.3f) dropped: decode "
                    "failed" % (case, part, t_start, t_stop))
                continue
            except WindowCnnError:
                traceback.print_exc(file=sys.stderr)
                drops["cnn_failed"] += 1
                log("WARNING: %s part %s [%.3f, %.3f) dropped: the "
                    "mandatory tool/task CNN heads failed"
                    % (case, part, t_start, t_stop))
                continue

            record = build_record(case, part, t_start, t_stop, evidence)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            written += 1

            if written % 100 == 0 or (index + 1) == len(windows):
                elapsed = time.time() - started
                log("%d/%d windows visited (%d written, %d dropped) in "
                    "%.1fs" % (index + 1, len(windows), written,
                              sum(drops.values()), elapsed))

    report = {
        "manifest": str(args.manifest),
        "distinct_windows": len(windows),
        "already_cached_at_start": len(already),
        "written_this_run": written,
        "drops": dict(drops),
        "heldout_confirmed_absent": sorted(excluded_norm),
    }
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log("wrote %d record(s) to %s this run; %d dropped (%s); report -> %s"
        % (written, args.out, sum(drops.values()), dict(drops), args.report_out))
    return report


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Cache real, model-produced evidence packets for every "
                    "distinct window in the VLM training manifest.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--splits", default=DEFAULT_SPLITS)
    parser.add_argument("--video-root", default=VIDEO_ROOT)
    parser.add_argument("--perception-config", default=DEFAULT_PERCEPTION_CONFIG)
    parser.add_argument("--models-dir", default=None,
                        help="re-roots every checkpoint path in "
                             "--perception-config by basename, mirroring "
                             "scripts/inference.py's --models-dir.")
    parser.add_argument("--frames", type=int, default=None,
                        help="override --perception-config's decode.frames.")
    parser.add_argument("--size", type=int, default=None,
                        help="override --perception-config's decode.size.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--report-out", default=DEFAULT_REPORT_OUT)
    # ON by default -- unlike inference.py's --yolo/--variant-head (optional
    # serving-time evidence, reviewed into the container's command line one
    # flag at a time), this script's entire purpose is to have yolo/variant/
    # motion_v2 evidence to train on. --no-yolo/--no-variant-head are the
    # opt-outs, kept for debugging and partial/smoke runs.
    parser.add_argument("--yolo", dest="yolo", action="store_true", default=True)
    parser.add_argument("--no-yolo", dest="yolo", action="store_false")
    parser.add_argument("--yolo-weights", default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--yolo-repo", default=DEFAULT_YOLO_REPO)
    parser.add_argument("--variant-head", dest="variant_head",
                        action="store_true", default=True)
    parser.add_argument("--no-variant-head", dest="variant_head",
                        action="store_false")
    parser.add_argument("--variant-weights", default=DEFAULT_VARIANT_WEIGHTS)
    parser.add_argument("--variant-config", default=DEFAULT_VARIANT_CONFIG)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of NEW windows processed this "
                             "run (already-cached windows do not count "
                             "against it) -- for smoke-testing before "
                             "paying for the full 15,087-window run.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    process_manifest(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
