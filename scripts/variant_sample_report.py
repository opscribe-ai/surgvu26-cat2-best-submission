"""Run the trained Large-vs-Mega needle-driver head against the 11 public
Cat 2 sample clips and report, per case, what it would have said.

THE QUESTION THIS ANSWERS. A detector run over the 11 sample clips shows a
needle driver visibly present in BOTH case126 ("Was a LARGE needle driver
used in this clip?", gold Yes, we currently answer No) and case132 ("Was a
LARGE needle driver used during the surgery?", gold No, we currently answer
Yes). The detector alone cannot separate these two golds -- only the SIZE
FAMILY does. `surgvu.variant.VariantHead` was just trained (val_accuracy
0.9011 at val_coverage 0.9990 on 30 held-out corpus cases) to make exactly
that call. This script is the first time it is pointed at the actual graded
clips rather than its own held-out validation pool. If it says LARGE on
case126 and MEGA on case132, both cases become answerable and the family is
worth roughly double what the detector offers alone on this pair; if it says
the same family for both, or abstains on both, the chain does not pay off.

WHY DECODE_CLIP AND NOT A HAND-ROLLED FRAME READER. Same reasoning as
scripts/detect_sample_report.py: `surgvu.perceive.decode_clip` is the real
serving decoder (16 evenly spaced frames, size 512, crop-side-margins +
blur-UI-band + resize, in that order). The UI-band blur is a challenge rule,
not a tuning choice, so every frame handed to the detector and the variant
head here has gone through exactly the same preprocessing the shipped
pipeline uses. There is no code path in this script that reads a frame
before `prepare_frame` has touched it.

WHY THE DETECTOR IS LOADED AT ITS NORMAL OPERATING THRESHOLD, NOT A LOW ONE.
Unlike detect_sample_report.py (which wants the full confidence landscape to
survive NMS for a "did anything come close" report), this script needs the
SAME box-selection convention `scripts/train_variant.py:build_examples` used
to build the training/validation pool: `Detector(weights, repo, device=...)`
at its class defaults (conf=0.25, iou=0.45), taking the max-confidence
`needle driver` detection as the crop box and falling back to the whole
frame when none clears NMS. Loading the detector at a different operating
point here would feed the head input drawn from a different distribution
than the one it was trained and cutoff-fitted against.

WHY THE CROP-VS-WHOLE-FRAME COUNT MATTERS (R29). `VariantHead.predict`
silently falls back to the whole frame when the detector's box is missing or
smaller than 8x8 after clamping to the frame -- a whole-frame decision is a
strictly harder problem (find the tool AND judge its size) than a cropped
one (compare the box's own taper and jaw length against its own scale). The
head's aggregate val_accuracy says nothing about which regime produced any
given case's decision. This script computes, independently and read-only,
which regime each of the 16 frames actually fell into, by reproducing
`VariantHead.predict`'s own clamp-and-8px-floor check on the same boxes this
script hands it -- not by modifying variant.py to report it itself.

A SANITY CHECK THIS SCRIPT INCLUDES, AND WHY IT IS NOT AS TRIVIAL AS IT
SOUNDS. The 11 sample cases and the head's training corpus are supposed to
be disjoint pools, so `held_out_cases` (from variant_head.json) containing
none of them should hold on a plain string comparison. But this exact repo
already documents an id-format trap for this exact number range: the public
sample directories are named `caseNNN`, the corpus that `config/
variant_labels.json` (and `config/splits_v2.json`, for the OTHER heads) uses
is named `case_NNN`, and `surgvu.sampling.normalize_case_id` exists
specifically because a raw string check over the two forms is silently
empty even when the corpus and the sample overlap by number (see
docs/compliance_audit.md section 3, "Held-out integrity of case_122-
case_132"). This script therefore runs BOTH checks: the literal one asked
for (raw string membership in `held_out_cases`, hard-asserted, since a
failure here would contaminate every number below) and a normalized one
(via `normalize_case_id`, reported but not hard-asserted, since resolving it
fully -- whether corpus `case_126` is the SAME video content as sample
`case126` -- needs the perceptual-hash scan docs/compliance_audit.md section
3.4 already started and left unfinished for a different split; re-running
that scan is out of this script's scope). Both results are printed plainly.

USAGE (inside a container with torch -- see condor/variant_sample.sub):

    python scripts/variant_sample_report.py \\
        --sample-root /staging/groups/bhaskar_opscribe/surgvu/cat2_sample \\
        --weights /staging/groups/bhaskar_opscribe/surgvu_yolo_detector/best.pt \\
        --yolov5-dir /staging/groups/bhaskar_opscribe/surgvu_yolo_detector/yolov5 \\
        --candidates baselines/shipped_candidates.json \\
        --variant-weights /staging/n/nkalthoff/surgvu26/models/variant_head.pt \\
        --variant-config /staging/n/nkalthoff/surgvu26/models/variant_head.json \\
        --out-json variant_sample_results.json

This is a READ-ONLY consumer of src/surgvu/variant.py, src/surgvu/detect.py,
src/surgvu/perceive.py, src/surgvu/router.py and src/surgvu/sampling.py. It
decides nothing and changes no answer -- it reports what the trained head
sees on the graded clips so a human can judge whether it belongs in the
pipeline.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reused rather than re-parsing the sample layout / video-path layout by
# hand, per the controller's brief -- same precedent as detect_sample_report
# itself reusing score_sample's helpers.
from detect_sample_report import _video_path  # noqa: E402
from score_sample import build_pairs, load_candidates, load_sample_cases  # noqa: E402

from surgvu.detect import Detector  # noqa: E402
from surgvu.router import variant_qualifier  # noqa: E402
from surgvu.sampling import normalize_case_id  # noqa: E402
from surgvu.variant import FAMILIES  # noqa: E402

DEFAULT_SAMPLE_ROOT = "/staging/groups/bhaskar_opscribe/surgvu/cat2_sample"
DEFAULT_WEIGHTS = "/staging/groups/bhaskar_opscribe/surgvu_yolo_detector/best.pt"
DEFAULT_YOLOV5_DIR = "/staging/groups/bhaskar_opscribe/surgvu_yolo_detector/yolov5"
DEFAULT_CANDIDATES = "baselines/shipped_candidates.json"
DEFAULT_VARIANT_WEIGHTS = "/staging/n/nkalthoff/surgvu26/models/variant_head.pt"
DEFAULT_VARIANT_CONFIG = "/staging/n/nkalthoff/surgvu26/models/variant_head.json"
DEFAULT_VARIANT_LABELS = "config/variant_labels.json"

#: The two cases the controller's brief names by id -- printed first, and
#: flagged inline in the per-case table, so a human never has to hunt for
#: them among the other nine.
HEADLINE_CASES = ("case126", "case132")


def _box_regime(frame_shape, box):
    """"crop" or "whole_frame" for one frame -- mirrors VariantHead.predict.

    Reproduces `VariantHead.predict`'s own box-acceptance check (clamp to
    the frame, then require at least 8px on each side) rather than importing
    a private helper from variant.py, because variant.py exposes no such
    helper: the check lives inline inside `predict`'s loop. Duplicating five
    lines of arithmetic here is cheaper and safer than adding a new public
    seam to src/ for a read-only reporting need -- and this script is
    forbidden from touching src/ regardless.
    """
    if box is None:
        return "whole_frame"
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(frame_shape[1], x2)
    y2 = min(frame_shape[0], y2)
    if x2 - x1 >= 8 and y2 - y1 >= 8:
        return "crop"
    return "whole_frame"


def _needle_boxes(detections):
    """{frame_index: box} for the max-confidence `needle driver` detection.

    Mirrors `scripts/train_variant.py:build_examples` exactly: only the
    `needle driver` class is a candidate crop (a detector guess at some
    OTHER tool would crop the wrong object), and when a frame has more than
    one needle-driver detection the highest-confidence one wins. A frame
    with no needle-driver detection at all is simply absent from the
    returned dict -- `VariantHead.predict` and `_box_regime` above both
    treat a missing key the same as an explicit `None`.
    """
    boxes = {}
    for index, found in enumerate(detections):
        needle = [item for item in found if item["cls"] == "needle driver"]
        if needle:
            boxes[index] = max(needle, key=lambda item: item["conf"])["box"]
    return boxes


def sanity_check_held_out(sample_ids, held_out_cases, labels_path):
    """Assert no sample case is literally in `held_out_cases`; also report
    the normalized (case_NNN) picture, which this exact id range has a
    documented history of hiding a real overlap behind. See module
    docstring for the full rationale.

    Raises AssertionError on the raw check -- the one the controller's brief
    asks for -- because a failure there means every number the rest of this
    script prints is contaminated, and an assertion is cheaper than the
    argument. The normalized/train-split findings are informational only:
    they identify a KNOWN id-coincidence already flagged in
    docs/compliance_audit.md for a different split (splits_v2.json), not a
    new leak this script has verified as real video-content overlap -- that
    verification is a perceptual-hash scan, out of scope here, and already
    left unfinished elsewhere for the same reason (compute cost).
    """
    sample_ids = sorted(sample_ids)
    held_out_set = set(held_out_cases)

    raw_overlap = sorted(set(sample_ids) & held_out_set)
    assert not raw_overlap, (
        "SANITY CHECK FAILED: sample case id(s) %r appear VERBATIM in the "
        "trained head's held_out_cases (%d entries). Every number in this "
        "report is contaminated -- stopping rather than printing them."
        % (raw_overlap, len(held_out_cases)))

    normalized = {cid: normalize_case_id(cid) for cid in sample_ids}
    normalized_in_held_out = sorted(
        cid for cid, norm in normalized.items() if norm in held_out_set)

    labelled_corpus = None
    normalized_in_train_split = []
    if labels_path and Path(labels_path).exists():
        labelled = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        labelled_corpus = set((labelled.get("cases") or {}).keys())
        normalized_in_train_split = sorted(
            cid for cid, norm in normalized.items()
            if norm in labelled_corpus and norm not in held_out_set)

    return {
        "sample_ids": sample_ids,
        "held_out_cases_count": len(held_out_cases),
        "raw_overlap": raw_overlap,
        "raw_overlap_ok": not raw_overlap,
        "normalized_ids": normalized,
        "normalized_overlap_with_held_out": normalized_in_held_out,
        "normalized_overlap_with_train_split": normalized_in_train_split,
        "labels_path_checked": str(labels_path) if labelled_corpus is not None else None,
        "labelled_corpus_size": len(labelled_corpus) if labelled_corpus is not None else None,
    }


def build_report(sample_root, weights, yolov5_dir, candidates_path,
                 variant_weights, variant_config, labels_path,
                 detector_conf, iou, n_frames, size, variant_size, device):
    from surgvu.variant import VariantHead  # torch import deferred inside

    cases = load_sample_cases(sample_root)
    candidates = load_candidates(candidates_path)
    pairs = build_pairs(cases, candidates)     # fails loudly on any mismatch

    cfg = json.loads(Path(variant_config).read_text(encoding="utf-8"))
    cutoff = float(cfg["cutoff"])
    held_out_cases = list(cfg.get("held_out_cases") or [])

    sanity = sanity_check_held_out(
        [case_id for case_id, _, _ in pairs], held_out_cases, labels_path)

    # Loaded once, reused across all 11 cases -- both Detector._load() and
    # VariantHead._load() cache their model on first call, and re-loading
    # per case would pay the checkpoint cost 11 times for no benefit.
    detector = Detector(weights, yolov5_dir, conf=detector_conf, iou=iou,
                       device=device)
    head = VariantHead(variant_weights, cutoff=cutoff, device=device,
                       size=variant_size)

    results = []
    for case_id, pipeline_answer, references in pairs:
        from surgvu.perceive import decode_clip     # torch import deferred

        video = _video_path(sample_root, case_id)
        frames = decode_clip(video, n_frames=n_frames, size=size)

        detections = detector.detect(frames)
        boxes = _needle_boxes(detections)
        detector_found_box = len(boxes) > 0

        regimes = [_box_regime(frames[i].shape, boxes.get(i))
                  for i in range(len(frames))]
        n_crop = sum(1 for r in regimes if r == "crop")
        n_whole_frame = len(regimes) - n_crop

        record = head.predict(frames, boxes=boxes)

        question = cases[case_id].question
        asked_family = variant_qualifier(question)
        gold = references[0]

        # Purely interpretive, NOT a pipeline change: what a yes/no answer
        # would be if the router consumed this head's decision for a
        # question naming one family. None when the question names no
        # family (nothing to hypothesize about) or the head abstained
        # (decided=False) -- an abstention has no opinion to report.
        hypothetical_answer = None
        if asked_family is not None and record["decided"]:
            hypothetical_answer = (
                "Yes" if record["family"] == asked_family else "No")

        results.append({
            "case_id": case_id,
            "question": question,
            "gold": gold,
            "pipeline_answer": pipeline_answer,
            "asked_family": asked_family,
            "detector_found_needle_driver_box": detector_found_box,
            "n_frames": len(frames),
            "n_frames_crop": n_crop,
            "n_frames_whole_frame": n_whole_frame,
            "frame_regimes": regimes,
            "variant": record,
            "hypothetical_answer": hypothetical_answer,
            "hypothetical_matches_gold": (
                None if hypothetical_answer is None
                else hypothetical_answer.strip().lower() == gold.strip().lower()),
        })
    return results, sanity, cutoff, held_out_cases


def _format_sanity(sanity):
    lines = []
    lines.append("=" * 78)
    lines.append("SANITY CHECK: do any of the 11 sample cases leak into the head's "
                 "held-out set?")
    lines.append("  raw string match (the literal check):     %s  (overlap=%r)"
                 % ("PASS" if sanity["raw_overlap_ok"] else "FAIL",
                    sanity["raw_overlap"]))
    norm_hits = sanity["normalized_overlap_with_held_out"]
    lines.append("  normalized match (case122 -> case_122):   %s  (%d/11: %s)"
                 % ("clean" if not norm_hits else "OVERLAP FOUND",
                    len(norm_hits), norm_hits or "none"))
    if norm_hits:
        lines.append("    -- these sample case NUMBERS also name a case in the head's")
        lines.append("       own training corpus (config/variant_labels.json) that landed")
        lines.append("       in ITS held-out split. Per docs/compliance_audit.md section 3,")
        lines.append("       the identical number-coincidence was checked for a DIFFERENT")
        lines.append("       split (splits_v2.json) and case122's video content did NOT")
        lines.append("       match corpus case_122's video -- but that scan was never")
        lines.append("       finished for the full 122-132 range, and was never run at all")
        lines.append("       against THIS corpus/split. Not re-verified here (out of scope:")
        lines.append("       needs the perceptual-hash scan, not a login-node job).")
    train_hits = sanity["normalized_overlap_with_train_split"]
    if train_hits:
        lines.append("  normalized match against the head's TRAIN split (not held-out):")
        lines.append("    %d/11: %s -- these case NUMBERS were in the corpus this head"
                     % (len(train_hits), train_hits))
        lines.append("    actually TRAINED on, same unverified-content caveat as above.")
    lines.append("=" * 78)
    return "\n".join(lines)


def _format_headline(results):
    by_id = {row["case_id"]: row for row in results}
    lines = []
    lines.append("=" * 78)
    lines.append("HEADLINE: case126 and case132 -- the pair this report exists to answer")
    families_seen = []
    for case_id in HEADLINE_CASES:
        row = by_id.get(case_id)
        if row is None:
            lines.append("  %s: NOT FOUND in this sample set" % case_id)
            continue
        v = row["variant"]
        wrong = "WRONG" if row["pipeline_answer"].strip().lower() != row["gold"].strip().lower() else "right"
        lines.append("  %-7s  Q: %s" % (case_id, row["question"]))
        lines.append("           gold=%-4s  pipeline(current)=%-4s [%s]  asks-family=%s"
                     % (row["gold"], row["pipeline_answer"], wrong, row["asked_family"]))
        lines.append("           head: family=%-5s decided=%-5s p_large=%.3f p_mega=%.3f "
                     "cutoff=%.3f  detbox=%s  crop=%d/%d whole=%d/%d"
                     % (v["family"], v["decided"], v["p_large"], v["p_mega"],
                        v["cutoff"], row["detector_found_needle_driver_box"],
                        row["n_frames_crop"], row["n_frames"],
                        row["n_frames_whole_frame"], row["n_frames"]))
        if row["hypothetical_answer"] is not None:
            verdict = "MATCHES GOLD" if row["hypothetical_matches_gold"] else "still wrong"
            lines.append("           IF the router consumed this: answer='%s' vs gold=%r -> %s"
                         % (row["hypothetical_answer"], row["gold"], verdict))
        else:
            lines.append("           IF the router consumed this: head ABSTAINED -- no "
                         "hypothetical answer")
        families_seen.append(v["family"] if v["decided"] else None)
        lines.append("")

    if len(families_seen) == 2 and None not in families_seen:
        pays_off = families_seen[0] != families_seen[1]
        lines.append("  DOES THE HEAD SEPARATE THE TWO CASES? family(case126)=%s vs "
                     "family(case132)=%s -> %s"
                     % (families_seen[0], families_seen[1],
                        "YES, different families -- the chain can pay off"
                        if pays_off else
                        "NO, same family -- the chain does NOT pay off on this pair"))
    else:
        lines.append("  DOES THE HEAD SEPARATE THE TWO CASES? at least one case abstained "
                     "-- the chain cannot pay off on an abstention")
    lines.append("=" * 78)
    return "\n".join(lines)


def _format_case(row):
    v = row["variant"]
    flag = "  <== TARGET" if row["case_id"] in HEADLINE_CASES else ""
    lines = []
    lines.append("%-8s asks=%-6s gold=%-4s pipeline=%-4s%s"
                 % (row["case_id"], row["asked_family"] or "-", row["gold"],
                    row["pipeline_answer"], flag))
    lines.append("         head: family=%-5s decided=%-5s p_large=%.3f p_mega=%.3f  "
                 "detbox=%-5s crop=%2d/%-2d whole=%2d/%-2d"
                 % (v["family"], v["decided"], v["p_large"], v["p_mega"],
                    row["detector_found_needle_driver_box"],
                    row["n_frames_crop"], row["n_frames"],
                    row["n_frames_whole_frame"], row["n_frames"]))
    return "\n".join(lines)


def format_report(results, sanity, cutoff, held_out_cases):
    lines = [_format_sanity(sanity), "", _format_headline(results), ""]
    lines.append("cutoff=%.3f (fitted, from variant_head.json)  "
                 "held_out_cases in training run: %d" % (cutoff, len(held_out_cases)))
    lines.append("-" * 78)
    lines.append("ALL 11 CASES (scan top-to-bottom; TARGET marks case126/case132):")
    lines.append("-" * 78)
    for row in results:
        lines.append(_format_case(row))
    lines.append("-" * 78)

    n_decided = sum(1 for r in results if r["variant"]["decided"])
    n_crop_frames = sum(r["n_frames_crop"] for r in results)
    n_total_frames = sum(r["n_frames"] for r in results)
    n_no_box_at_all = sum(1 for r in results
                          if not r["detector_found_needle_driver_box"])
    lines.append("SUMMARY: %d/%d cases decided (not abstained). "
                 "%d/%d cases had NO needle-driver detection at all (100%% "
                 "whole-frame). Frames: %d/%d (%.1f%%) used a detector crop, "
                 "%d/%d (%.1f%%) fell back to the whole frame."
                 % (n_decided, len(results), n_no_box_at_all, len(results),
                    n_crop_frames, n_total_frames,
                    100.0 * n_crop_frames / n_total_frames,
                    n_total_frames - n_crop_frames, n_total_frames,
                    100.0 * (n_total_frames - n_crop_frames) / n_total_frames))
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", default=DEFAULT_SAMPLE_ROOT,
                        help="directory holding the 11 public sample cases")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="path to the detector's best.pt")
    parser.add_argument("--yolov5-dir", default=DEFAULT_YOLOV5_DIR,
                        help="the yolov5 checkout Detector puts on sys.path")
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES,
                        help="{case_id: answer} JSON -- the CURRENT shipped answers")
    parser.add_argument("--variant-weights", default=DEFAULT_VARIANT_WEIGHTS,
                        help="trained VariantHead ResNet-18 state dict")
    parser.add_argument("--variant-config", default=DEFAULT_VARIANT_CONFIG,
                        help="variant_head.json: fitted cutoff, val metrics, "
                             "held_out_cases")
    parser.add_argument("--variant-labels", default=DEFAULT_VARIANT_LABELS,
                        help="config/variant_labels.json -- the head's full "
                             "labelled corpus, used only to classify a "
                             "normalized id-overlap as train-split vs "
                             "held-out-split vs not-in-corpus-at-all")
    parser.add_argument("--detector-conf", type=float, default=0.25,
                        help="Detector's NMS confidence threshold -- kept at "
                             "the class default / what build_examples used, "
                             "for train/serve box-selection parity (see "
                             "module docstring)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="Detector's NMS IoU threshold (detect.py default)")
    parser.add_argument("--n-frames", type=int, default=16,
                        help="frames per clip (decode_clip's own default)")
    parser.add_argument("--size", type=int, default=512,
                        help="frame side length (decode_clip's own default)")
    parser.add_argument("--variant-size", type=int, default=224,
                        help="VariantHead's crop resize side (its own default, "
                             "and train_variant.py's FRAME_SIZE)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-json", default="variant_sample_results.json",
                        help="where to write the full per-case record as JSON")
    args = parser.parse_args()

    results, sanity, cutoff, held_out_cases = build_report(
        sample_root=args.sample_root, weights=args.weights,
        yolov5_dir=args.yolov5_dir, candidates_path=args.candidates,
        variant_weights=args.variant_weights,
        variant_config=args.variant_config, labels_path=args.variant_labels,
        detector_conf=args.detector_conf, iou=args.iou,
        n_frames=args.n_frames, size=args.size,
        variant_size=args.variant_size, device=args.device)

    print(format_report(results, sanity, cutoff, held_out_cases))

    out_path = Path(args.out_json)
    out_path.write_text(
        json.dumps({
            "sample_root": args.sample_root,
            "weights": args.weights,
            "yolov5_dir": args.yolov5_dir,
            "candidates": args.candidates,
            "variant_weights": args.variant_weights,
            "variant_config": args.variant_config,
            "variant_labels": args.variant_labels,
            "detector_conf": args.detector_conf,
            "iou": args.iou,
            "n_frames": args.n_frames,
            "size": args.size,
            "variant_size": args.variant_size,
            "cutoff": cutoff,
            "held_out_cases": held_out_cases,
            "sanity_check": sanity,
            "results": results,
        }, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print("\nwrote %s" % out_path)
