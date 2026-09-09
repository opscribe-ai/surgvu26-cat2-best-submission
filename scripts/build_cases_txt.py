"""Emit condor/cases.txt from the videos present on staging.

One row per VIDEO FILE, not per case: `case_id, part, video_path`. A case
splits into up to two video files and timestamps reset at the boundary, so the
unit of work is a `(case, part)` pair. Collapsing both files of a case to one
`case_id` queued two jobs that decoded part-2 windows from the part-1 video and
then raced to write the same shard path.

Assumed filename convention (the real staging layout is authoritative -- check
the SKIPPED report before submitting):

    case_056_video_part_001.mp4   -> case_056, part 1
    case_056_video_part_002.mp4   -> case_056, part 2

Anything matching `case_<digits>` followed later by `part` and a number is
accepted, so `case_056_part2.mp4` and `case_056-part-02.mp4` work too. A file
whose name yields no part number is SKIPPED and reported -- the part is never
guessed, not even for a case that has only one video file, because guessing
"part 1" for an unrecognised name is precisely how a whole part's worth of
windows gets decoded against the wrong timestamps.

Usage:
    python scripts/build_cases_txt.py VIDEO_ROOT LABELS_ROOT [OUT]
"""
import re
import sys
from pathlib import Path

# case id, then anything, then 'part' and its number. The part group is
# mandatory: no part, no row.
VIDEO_NAME = re.compile(r"^(case_\d+).*?part[_\-\s]*(\d+)", re.IGNORECASE)


def parse_video_name(name):
    """('case_056', '1') from a video filename, or None if it has no part."""
    match = VIDEO_NAME.match(name)
    if not match:
        return None
    return match.group(1), str(int(match.group(2)))


def main(argv):
    if not 2 <= len(argv) <= 3:
        raise SystemExit(__doc__)
    video_root = Path(argv[0])
    labels_root = Path(argv[1])
    out = Path(argv[2]) if len(argv) > 2 else Path("condor/cases.txt")
    out.parent.mkdir(parents=True, exist_ok=True)

    known = {d.name for d in labels_root.iterdir() if d.is_dir()}
    rows, unparsed, unknown = [], [], []
    claimed = {}
    # Videos may sit flat in VIDEO_ROOT, but the staged layout nests one
    # directory per case (`surgvu24/case_056/case_056_video_part_001.mp4`).
    # A flat glob silently found zero files there and queued an empty run.
    videos = sorted(set(video_root.glob("*.mp4")) | set(video_root.glob("*/*.mp4")))
    if not videos:
        raise SystemExit("no .mp4 found under %s (checked %s/*.mp4 and %s/*/*.mp4)"
                         % (video_root, video_root, video_root))
    for video in videos:
        parsed = parse_video_name(video.name)
        if parsed is None:
            unparsed.append(video.name)
            continue
        case_id, part = parsed
        if case_id not in known:
            unknown.append(video.name)
            continue
        if (case_id, part) in claimed:
            raise SystemExit(
                "two videos claim %s part %s: %r and %r. One shard path per "
                "(case, part). Resolve this before queuing."
                % (case_id, part, claimed[(case_id, part)], video.name))
        claimed[(case_id, part)] = video.name
        rows.append("%s, %s, %s" % (case_id, part, video))

    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("queued %d videos across %d cases"
          % (len(rows), len({c for c, _ in claimed})))

    if unparsed:
        print("\nSKIPPED %d videos with NO READABLE PART in the filename. "
              "the part was NOT guessed, so these cases will have missing "
              "windows. Fix the convention or this script's regex before "
              "submitting:" % len(unparsed))
        for name in unparsed[:20]:
            print("  %s" % name)
    if unknown:
        print("\nSKIPPED %d videos with no matching label directory:"
              % len(unknown), unknown[:5])
    if unparsed:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
