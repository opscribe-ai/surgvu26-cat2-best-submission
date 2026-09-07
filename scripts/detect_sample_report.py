"""Run the groupmate's YOLOv5 tool detector against the 11 public Cat 2
sample clips and report, per case, whether it would have caught case124.

THE QUESTION THIS ANSWERS. case124 is 62% of everything recoverable on the
11-case public sample: gold is "Cadiere Forceps", the shipped pipeline
answers "Bipolar Forceps". The detector (`src/surgvu/detect.py`) confuses
those two classes only ~1% of the time in each direction on its OWN
validation set -- but nobody has ever pointed it at the graded clip itself.
Everything before this script is inference from val-set metrics on a
different pool of frames. This script is the first time detect.py's
`Detector.detect` is run against the actual 11 cases that get scored.

WHY DECODE_CLIP AND NOT A HAND-ROLLED FRAME READER. `surgvu.perceive.
decode_clip` is the real serving decoder: 16 evenly spaced frames, size 512,
crop-side-margins + blur-UI-band + resize, in that order. The UI-band blur is
a challenge rule ("using the information available in the UI to make
predictions is not allowed... the UI will be blurred"), not a tuning choice,
so every frame handed to the detector here goes through exactly the same
preprocessing the shipped pipeline uses. There is no code path in this script
that reads a frame before `prepare_frame` has touched it.

WHY THE DETECTOR IS LOADED AT A LOW INTERNAL CONFIDENCE. `Detector`'s own
`conf` argument is the NMS threshold BELOW which a candidate detection is
discarded before it ever reaches this script -- so if it were left at
detect.py's own default (0.25, the checkpoint's validated operating point),
a class sitting at 0.20 for cadiere forceps would be invisible here, not
merely "below floor". This script loads the detector at a much lower
`--detector-conf` (default 0.01) so the full confidence landscape survives
NMS, and applies `--conf-floor` (default 0.25, the checkpoint's validated
operating point) purely as a REPORTING cutoff on top of that. The two are
independent knobs on purpose.

WHAT COUNTS AS "THE CASE IS A TOOL-NAMING QUESTION". Only case124's gold
answer is itself a tool noun ("Cadiere Forceps"); case122/123/126/128/131/132
ask yes/no questions that MENTION a tool family ("forceps", "large needle
driver") without gold answering with a specific class name, and matching the
detector's top pick against a yes/no answer is a category error. The
`_gold_tool_noun` check below is exact-match against the FIRST reference
(the short-form canonical answer, e.g. "Cadiere Forceps", "Uterine horn",
"No") -- not a substring search, because a substring search over all five
reference sentences catches "needle driver" inside "No, a large needle
driver is not listed." and would misclassify the yes/no cases as
tool-identity cases.

USAGE (inside a container with torch -- see condor/detect_sample.sub):

    python scripts/detect_sample_report.py \\
        --sample-root /staging/groups/bhaskar_opscribe/surgvu/cat2_sample \\
        --weights /staging/groups/bhaskar_opscribe/surgvu_yolo_detector/best.pt \\
        --yolov5-dir /staging/groups/bhaskar_opscribe/surgvu_yolo_detector/yolov5 \\
        --candidates baselines/shipped_candidates.json \\
        --conf-floor 0.25 \\
        --out-json detect_sample_results.json

This is a READ-ONLY consumer of src/surgvu/detect.py and src/surgvu/
perceive.py. It decides nothing and changes no answer -- it reports what the
detector sees so a human can judge whether it belongs in the pipeline.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Same directory as this script, so a plain `python scripts/detect_sample_
# report.py` (script's own dir is sys.path[0]) resolves this with no path
# surgery. Reused rather than re-parsing the sample layout by hand, per the
# controller's brief.
from score_sample import build_pairs, load_candidates, load_sample_cases  # noqa: E402

from surgvu.detect import (  # noqa: E402
    Detector, YOLO_CLASSES, detections_to_record, map_to_taxonomy)
from surgvu.taxonomy import TOOL_CLASSES  # noqa: E402

DEFAULT_SAMPLE_ROOT = "/staging/groups/bhaskar_opscribe/surgvu/cat2_sample"
DEFAULT_WEIGHTS = "/staging/groups/bhaskar_opscribe/surgvu_yolo_detector/best.pt"
DEFAULT_YOLOV5_DIR = "/staging/groups/bhaskar_opscribe/surgvu_yolo_detector/yolov5"
DEFAULT_CANDIDATES = "baselines/shipped_candidates.json"

_TOOL_SET = frozenset(TOOL_CLASSES)


def _video_path(sample_root, case_id):
    """Nested (root/caseNNN/caseNNN.mp4) or flat (root/caseNNN.mp4).

    Mirrors the two layouts score_sample.load_sample_cases already handles
    for the reference/question JSON, since the video sits next to them in
    both layouts observed on staging.
    """
    sample_root = Path(sample_root)
    nested = sample_root / case_id / ("%s.mp4" % case_id)
    if nested.exists():
        return nested
    flat = sample_root / ("%s.mp4" % case_id)
    if flat.exists():
        return flat
    raise FileNotFoundError(
        "no clip for %s under %s (looked for %s and %s)"
        % (case_id, sample_root, nested, flat))


def _gold_tool_noun(references):
    """The gold answer's tool class name, or None if it does not name one.

    Exact match against the FIRST reference only (the short canonical form),
    not a substring search over all five -- see the module docstring for why
    a substring search over the long-form sentences would misclassify the
    yes/no tool-presence cases (case123/126/128/132's references contain the
    literal words "needle driver" inside an explanatory sentence whose gold
    answer is "Yes"/"No", not a tool name).
    """
    if not references:
        return None
    canonical = str(references[0]).strip().lower()
    return canonical if canonical in _TOOL_SET else None


def _frame_timestamps(video_path, requested_n_frames, actual_n_frames):
    """Seconds for each anchor `decode_clip` returned, best effort.

    `decode_clip` skips a frame whose seek-and-read fails rather than ending
    the clip, so it can return fewer frames than `requested_n_frames` without
    saying which sampled indices were the casualties (see its own
    docstring). This is independent, read-only bookkeeping -- NOT a
    modification of perceive.py -- that re-opens the same file to recover
    total-frame-count and fps exactly the way `decode_clip` / `_frame_count`
    do, then reproduces `sample_frame_indices`.

    When nothing was skipped (`actual_n_frames == requested_n_frames`, the
    observed case for every one of these 11 clips), the indices line up with
    `decode_clip`'s output 1:1 and the returned timestamps are exact. When a
    frame WAS skipped, which index was dropped cannot be recovered from
    outside `decode_clip` without changing it, so this falls back to
    re-spacing `actual_n_frames` anchors evenly across the same clip and
    marks the result approximate -- close enough for a "when did the tool
    appear" read but flagged rather than presented as exact.
    """
    import cv2

    from surgvu.frames import sample_frame_indices

    capture = cv2.VideoCapture(str(video_path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            count = 0
            while capture.grab():
                count += 1
            total = count
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 0:      # 0, None, or NaN
            fps = 60.0
    finally:
        capture.release()

    indices = sample_frame_indices(total, requested_n_frames)
    if len(indices) == actual_n_frames:
        return [round(i / fps, 3) for i in indices], False

    approx_indices = sample_frame_indices(total, actual_n_frames)
    return [round(i / fps, 3) for i in approx_indices], True


def _classes_by_confidence(max_conf):
    """All 14 detector classes, sorted by max confidence descending.

    Classes with no detection at all get 0.0 -- included rather than omitted
    so a near-miss (say cadiere forceps sitting at 0.20, just under the
    reporting floor) is visible instead of silently absent from the table.
    """
    ordered = sorted(YOLO_CLASSES, key=lambda c: max_conf.get(c, 0.0),
                     reverse=True)
    return [(c, float(max_conf.get(c, 0.0))) for c in ordered]


def _top_in_taxonomy(ranked):
    """First (class, conf) in `ranked` that is one of the 12 answer classes.

    `ranked` is already sorted descending, so this is the detector's best
    guess at an ANSWERABLE tool -- excluding bipolar dissector / suction
    irrigator, which detect.py's own map_to_taxonomy keeps as evidence-only
    and never lets become an answer.
    """
    for cls, conf in ranked:
        if map_to_taxonomy(cls) is not None:
            return cls, conf
    return None, 0.0


def build_report(sample_root, weights, yolov5_dir, candidates_path,
                 conf_floor, detector_conf, iou, n_frames, size, device):
    cases = load_sample_cases(sample_root)
    candidates = load_candidates(candidates_path)
    pairs = build_pairs(cases, candidates)     # fails loudly on any mismatch

    # Loaded once, reused across all 11 cases: Detector._load() caches the
    # model on first call, and re-instantiating per case would pay best.pt's
    # load cost 11 times for no benefit.
    detector = Detector(weights, yolov5_dir, conf=detector_conf, iou=iou,
                       device=device)

    results = []
    for case_id, pipeline_answer, references in pairs:
        from surgvu.perceive import decode_clip     # torch import deferred

        video = _video_path(sample_root, case_id)
        frames = decode_clip(video, n_frames=n_frames, size=size)
        timestamps, approx = _frame_timestamps(video, n_frames, len(frames))

        detections = detector.detect(frames)
        record = detections_to_record(detections, timestamps)

        ranked = _classes_by_confidence(record["max_conf"])
        above_floor = [c for c, conf in ranked if conf >= conf_floor]
        top_cls, top_conf = _top_in_taxonomy(ranked)

        gold_noun = _gold_tool_noun(references)
        if gold_noun is None:
            verdict = "NOT_A_TOOL_QUESTION"
        elif top_cls == gold_noun:
            verdict = "MATCH"
        else:
            verdict = "DISAGREE"

        results.append({
            "case_id": case_id,
            "question": cases[case_id].question,
            "gold_references": references,
            "gold_tool_noun": gold_noun,
            "pipeline_answer": pipeline_answer,
            "n_frames_decoded": len(frames),
            "n_frames_requested": n_frames,
            "timestamps_approximate": approx,
            "detector_conf_threshold": detector_conf,
            "conf_floor": conf_floor,
            "classes_ranked": [{"cls": c, "max_conf": conf} for c, conf in ranked],
            "classes_above_floor": above_floor,
            "top_in_taxonomy_class": top_cls,
            "top_in_taxonomy_conf": top_conf,
            "verdict": verdict,
            "detection_record": record,
        })
    return results


def _format_case(row, width=None):
    lines = []
    tag = {"MATCH": "[TOOL-ID: MATCH]",
          "DISAGREE": "[TOOL-ID: DISAGREE]",
          "NOT_A_TOOL_QUESTION": "[not a tool-ID question]"}[row["verdict"]]
    lines.append("%s  %s" % (row["case_id"], tag))
    lines.append("  Q:        %s" % row["question"])
    lines.append("  gold:     %s" % (row["gold_tool_noun"] or row["gold_references"][0]))
    lines.append("  pipeline: %s" % row["pipeline_answer"])

    top5 = row["classes_ranked"][:5]
    top_bits = []
    for entry in top5:
        cls, conf = entry["cls"], entry["max_conf"]
        marker = "*" if conf >= row["conf_floor"] else " "
        oot = " (out-of-taxonomy)" if cls not in _TOOL_SET else ""
        top_bits.append("%s%-32s %.3f%s" % (marker, cls, conf, oot))
    lines.append("  detector top classes (max conf; * clears floor %.2f):"
                 % row["conf_floor"])
    for bit in top_bits:
        lines.append("      %s" % bit)

    if row["verdict"] != "NOT_A_TOOL_QUESTION":
        lines.append("  top in-taxonomy pick: %s (%.3f) vs gold %r -> %s"
                     % (row["top_in_taxonomy_class"], row["top_in_taxonomy_conf"],
                        row["gold_tool_noun"], row["verdict"]))
    if row["timestamps_approximate"]:
        lines.append("  NOTE: decode_clip returned %d/%d requested frames; "
                     "per-detection timestamps below are re-spaced, not exact."
                     % (row["n_frames_decoded"], row["n_frames_requested"]))
    return "\n".join(lines)


def format_report(results):
    lines = []
    tool_id_rows = [r for r in results if r["verdict"] != "NOT_A_TOOL_QUESTION"]
    lines.append("=" * 72)
    lines.append("HEADLINE: does the detector get the tool-naming case(s) right?")
    for row in tool_id_rows:
        lines.append("  %s: gold=%r  pipeline=%r  detector-top=%r (%.3f)  -> %s"
                     % (row["case_id"], row["gold_tool_noun"],
                        row["pipeline_answer"], row["top_in_taxonomy_class"],
                        row["top_in_taxonomy_conf"], row["verdict"]))
    if not tool_id_rows:
        lines.append("  (no case in this sample has a tool-noun gold answer)")
    lines.append("=" * 72)
    lines.append("")

    for row in results:
        lines.append(_format_case(row))
        lines.append("-" * 72)

    n_match = sum(1 for r in tool_id_rows if r["verdict"] == "MATCH")
    n_disagree = sum(1 for r in tool_id_rows if r["verdict"] == "DISAGREE")
    n_na = len(results) - len(tool_id_rows)
    lines.append("SUMMARY: %d tool-ID case(s) -- %d MATCH, %d DISAGREE; "
                 "%d case(s) not a tool-naming question."
                 % (len(tool_id_rows), n_match, n_disagree, n_na))
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", default=DEFAULT_SAMPLE_ROOT,
                        help="directory holding the 11 public sample cases")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="path to best.pt")
    parser.add_argument("--yolov5-dir", default=DEFAULT_YOLOV5_DIR,
                        help="the yolov5 checkout Detector puts on sys.path")
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES,
                        help="{case_id: answer} JSON -- the CURRENT shipped answers")
    parser.add_argument("--conf-floor", type=float, default=0.25,
                        help="reporting cutoff for 'clears the floor' (default: "
                             "the checkpoint's own validated operating point)")
    parser.add_argument("--detector-conf", type=float, default=0.01,
                        help="Detector's internal NMS confidence threshold -- "
                             "kept low so the full confidence landscape survives "
                             "NMS for this report; see module docstring")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="Detector's NMS IoU threshold (detect.py default)")
    parser.add_argument("--n-frames", type=int, default=16,
                        help="frames per clip (decode_clip's own default)")
    parser.add_argument("--size", type=int, default=512,
                        help="frame side length (decode_clip's own default)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-json", default="detect_sample_results.json",
                        help="where to write the full per-case record as JSON")
    args = parser.parse_args()

    results = build_report(
        sample_root=args.sample_root, weights=args.weights,
        yolov5_dir=args.yolov5_dir, candidates_path=args.candidates,
        conf_floor=args.conf_floor, detector_conf=args.detector_conf,
        iou=args.iou, n_frames=args.n_frames, size=args.size,
        device=args.device)

    print(format_report(results))

    out_path = Path(args.out_json)
    out_path.write_text(
        json.dumps({
            "sample_root": args.sample_root,
            "weights": args.weights,
            "yolov5_dir": args.yolov5_dir,
            "candidates": args.candidates,
            "conf_floor": args.conf_floor,
            "detector_conf": args.detector_conf,
            "iou": args.iou,
            "n_frames": args.n_frames,
            "size": args.size,
            "results": results,
        }, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print("\nwrote %s" % out_path)
