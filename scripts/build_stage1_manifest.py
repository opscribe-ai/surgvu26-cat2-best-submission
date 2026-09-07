#!/usr/bin/env python3
"""Stage 1 of the v6 curriculum: a SurgVU-shaped manifest from SSG-VQA.

WHY A STAGE 1 AT ALL
----------------------
The SurgVU corpus this project trains on is templated: 164 distinct question
strings and 116 distinct answers over 144 cases. A model fitted only on that
learns the TEMPLATES. The graded questions are human-written and varied
("Are there forceps being used here?", "What is the purpose of using forceps
in this procedure?"), so when the test set phrases something a way the
generator never produced, a template-fitted model has nothing to fall back on.
Stage 1 is the cure: real surgical frames, real question variety, a different
procedure family.

THE ORDER IS THE WHOLE POINT: stage 1 FIRST, SurgVU LAST. BERTScore grades
against SurgVU's answer forms. Stage 1 is ~2x the SurgVU corpus, so mixing
them lets it dominate and drag the output distribution toward its own
vocabulary. Run last, SurgVU re-establishes the form on top of stage 1's
perception.

ONLY 5.5% OF SSG-VQA IS USED, AND THAT IS DELIBERATE
------------------------------------------------------
Censused over all 27,489 annotation files (1,004,907 QA lines, 50 videos):

    other_spatial        731,665   72.8%   <- dropped
    count_spatial        177,927   17.7%   <- dropped
    anatomy_at_position   40,337    4.0%   <- dropped
    anatomy_present       27,489    2.7%   <- KEPT
    tools_present         27,489    2.7%   <- KEPT

The dropped 94.5% is templated spatial reasoning -- "What number of anatomys
are both above the bottom-mid anatomy and below the white top-left gallbladder
anatomy?" -- whose answers are bare integers and underscore-lowercase nouns.
That is a different task in a different register, and 900k of it would teach
the model a phrasing stage 2 then has to unteach. The two kept shapes are
exactly SurgVU's `tool_identity_open` and `organ_open`.

THE ANSWER FORM IS NORMALISED, THE ANSWER VOCABULARY IS NOT
-------------------------------------------------------------
SSG-VQA writes `grasper, hook` and `cystic_plate`. This module rewrites those
to `Grasper and Hook` and `Cystic Plate` -- SurgVU's SHAPE (Title Case, comma
list with a final "and") -- while leaving the NOUNS alone.

That split is the point. CholecT45's tool taxonomy (grasper, hook, scissors,
clipper, irrigator, bipolar) is not SurgVU's twelve classes, and pretending
otherwise would teach wrong names. But the FORM is free to transfer: stage 1
teaches "list what you see, like this", stage 2 teaches "and here are the
twelve names". Teaching the form twice wastes stage 2's gradient on
punctuation.

16-FRAME WINDOWS, NOT SINGLE IMAGES
-------------------------------------
CholecT45 frames are sequential at 1fps (`data/VID01/000000.png`, `000001`,
...), so an annotated frame N becomes a real 16-frame window over its
neighbours -- the same shape `evidence_vlm.DEFAULT_FRAMES_PER_CALL` samples at
serving. A single-image stage 1 would train the model on a prompt shape it
never sees in production, which is the exact mismatch v6 exists to fix (both
previous adapters were fitted on 4 frames and served 16).

CASE IDS ARE case_900+, SO THEY CANNOT COLLIDE
------------------------------------------------
`train_vlm.assign_case_split` splits BY CASE and raises on any case that is in
neither split list, so stage 1 needs ids `surgvu.sampling.normalize_case_id`
accepts. SurgVU occupies case_000..case_154; VID01..VID80 map to case_901..
case_980, which cannot overlap, so a stage-1 record can never be mistaken for
a SurgVU one -- or silently land in a SurgVU split.
"""
import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_SSG_QA = ("/staging/groups/bhaskar_opscribe/benchmarking_datasets/"
                  "cholecystectomy_eval/SSG-VQA/ssg-qa.zip")
DEFAULT_FRAMES_ZIP = ("/staging/groups/bhaskar_opscribe/benchmarking_datasets/"
                      "cholecystectomy_eval/SSG-VQA/CholecT45.zip")

#: 1 fps in CholecT45, so 16 frames is a 16-second window -- comparable to the
#: 30-second SurgVU window that `build_qa_pairs.py` samples 16 frames from.
FRAMES_PER_WINDOW = 16

#: The two SSG-VQA question shapes that map onto SurgVU intents. Matched on the
#: question PREFIX because SSG-VQA's generator emits these two verbatim.
KEPT_SHAPES = (
    ("Which tools are present", "tool_identity_open"),
    ("Which anatomical structures are present", "organ_open"),
)

#: SurgVU's own question phrasings for the two kept intents, so stage 1 asks in
#: the register stage 2 will. Cycled per record (index into the tuple) rather
#: than picked randomly, so a rebuild is reproducible without a seed.
QUESTION_FORMS = {
    "tool_identity_open": (
        "What instruments are being used in this clip?",
        "Which surgical tools are visible here?",
        "What tools does this clip show?",
    ),
    "organ_open": (
        "What anatomical structures are visible in this clip?",
        "What anatomy is shown here?",
        "Which structures can be seen in this segment?",
    ),
}

_VID_RE = re.compile(r"VID(\d+)")


def case_id_for_video(video):
    """`VID01` -> `case_901`. See the module docstring on why 900+."""
    match = _VID_RE.match(str(video))
    if match is None:
        raise ValueError("unparseable video id %r" % (video,))
    return "case_%d" % (900 + int(match.group(1)))


def display_noun(raw):
    """`cystic_plate` -> `Cystic Plate`. Form only -- the noun is untouched."""
    return " ".join(w.capitalize() for w in str(raw).strip().split("_") if w)


def format_answer(raw):
    """`grasper, hook` -> `Grasper and Hook`; `a, b, c` -> `A, B and C`.

    Mirrors SurgVU's own list form. Returns None for an EMPTY annotation --
    SSG-VQA writes a bare `` when no tool is present, and 2,780 of the kept
    lines are that. An empty answer is not a training target: it would teach
    the model to emit nothing, and a missing response scores 0 at grading.
    """
    items = [display_noun(p) for p in str(raw).split(",") if p.strip()]
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    return "%s and %s" % (", ".join(items[:-1]), items[-1])


def parse_annotations(ssg_qa_zip):
    """[(video, frame_index, intent, answer)] for the kept shapes only."""
    out = []
    with zipfile.ZipFile(ssg_qa_zip) as zf:
        names = sorted(n for n in zf.namelist()
                       if n.endswith(".txt") and "/VID" in n)
        for name in names:
            parts = Path(name).parts
            video, stem = parts[-2], Path(name).stem
            if not stem.isdigit():
                continue
            frame_index = int(stem)
            text = zf.read(name).decode("utf-8", "replace")
            for line in text.splitlines():
                if "|" not in line:
                    continue
                question, answer = line.split("|", 1)[0], line.split("|")[1]
                for prefix, intent in KEPT_SHAPES:
                    if question.strip().startswith(prefix):
                        formatted = format_answer(answer)
                        if formatted is not None:
                            out.append((video, frame_index, intent, formatted))
                        break
    return out


def window_frame_indices(centre, available):
    """The 16 frame indices for a window centred on `centre`.

    CLAMPED TO WHAT EXISTS, NOT WRAPPED. `available` is the sorted set of
    frame indices actually present for that video; near a video's start or end
    there are fewer than 8 neighbours on one side, so the window slides inward
    rather than wrapping around to the far end of the operation -- which would
    put frames from a completely different phase in the same clip.

    Returns None when the video has fewer than FRAMES_PER_WINDOW frames at all.
    """
    if len(available) < FRAMES_PER_WINDOW:
        return None
    position = available.index(centre)
    start = position - FRAMES_PER_WINDOW // 2
    start = max(0, min(start, len(available) - FRAMES_PER_WINDOW))
    return available[start:start + FRAMES_PER_WINDOW]


def build_records(annotations, frames_index, frames_root):
    """Manifest records in `build_qa_pairs.py --extract-frames`'s own shape."""
    records = []
    per_intent_seen = {}
    for video, centre, intent, answer in annotations:
        available = frames_index.get(video)
        if not available or centre not in frames_index[video]:
            continue
        idxs = window_frame_indices(centre, available)
        if idxs is None:
            continue
        forms = QUESTION_FORMS[intent]
        n = per_intent_seen.get(intent, 0)
        per_intent_seen[intent] = n + 1
        case = case_id_for_video(video)
        out_dir = Path(frames_root) / video
        records.append({
            "case": case,
            "part": "1.0",
            "t_start": float(centre),
            "t_stop": float(centre) + float(FRAMES_PER_WINDOW),
            "question": forms[n % len(forms)],
            "answer": answer,
            "intent": intent,
            "frame_dir": str(out_dir),
            "frame_paths": [str(out_dir / ("%06d.jpg" % i)) for i in idxs],
            "provenance": {
                "generator": "scripts/build_stage1_manifest.py",
                "source": "SSG-VQA",
                "video": video,
                "centre_frame": centre,
            },
        })
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ssg-qa", default=DEFAULT_SSG_QA)
    parser.add_argument("--frames-zip", default=DEFAULT_FRAMES_ZIP)
    parser.add_argument("--frames-root", required=True,
                        help="where the converted 512px JPEGs live/will live")
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--splits-out", required=True,
                        help="stage-1 splits JSON; train_vlm needs one and it "
                             "must NOT be config/splits_v2.json")
    parser.add_argument("--val-videos", type=int, default=6,
                        help="how many videos are held out for val. Split BY "
                             "VIDEO, never by frame: neighbouring frames of "
                             "one operation are near-duplicates, so a "
                             "frame-level split leaks val into train.")
    args = parser.parse_args(argv)

    with zipfile.ZipFile(args.frames_zip) as zf:
        frames_index = {}
        for name in zf.namelist():
            if not name.endswith(".png") or "/data/VID" not in name:
                continue
            parts = Path(name).parts
            video, stem = parts[-2], Path(name).stem
            if stem.isdigit():
                frames_index.setdefault(video, []).append(int(stem))
    for video in frames_index:
        frames_index[video].sort()
    print("frame index: %d video(s), %d frame(s)"
          % (len(frames_index), sum(len(v) for v in frames_index.values())))

    annotations = parse_annotations(args.ssg_qa)
    print("kept annotations: %d" % len(annotations))

    records = build_records(annotations, frames_index, args.frames_root)
    print("records: %d" % len(records))

    videos = sorted({r["provenance"]["video"] for r in records})
    val_videos = set(videos[-args.val_videos:]) if args.val_videos else set()
    splits = {
        "train": sorted(case_id_for_video(v) for v in videos if v not in val_videos),
        "val": sorted(case_id_for_video(v) for v in val_videos),
        # train_vlm.load_case_universe REQUIRES a non-empty heldout list (R30).
        # Stage 1 has no graded cases to hold out, so the SurgVU graded eleven
        # go here: they are not in this corpus at all, which makes the list
        # both non-empty and true.
        "heldout": ["case_%d" % n for n in range(122, 133)],
        "meta": {
            "generator": "scripts/build_stage1_manifest.py",
            "source": "SSG-VQA (CholecT45 frames), CC BY-NC-SA 4.0",
            "split_unit": "video",
            "note": "Stage 1 of the v6 curriculum. Case ids are 900+ so they "
                    "cannot collide with SurgVU's case_000..case_154.",
        },
    }

    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_out, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    Path(args.splits_out).write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print("wrote %s (%d records)" % (args.manifest_out, len(records)))
    print("wrote %s (train=%d val=%d)"
          % (args.splits_out, len(splits["train"]), len(splits["val"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
