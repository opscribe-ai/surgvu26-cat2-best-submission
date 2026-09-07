"""Extract one (case, part) to a shard. One HTCondor job runs one of these.

The job unit is a `(case, part)` pair, not a case. A case can have two video
files and **timestamps reset at the part boundary**, so part-2 windows are only
meaningful against the part-2 video. Running one job per case meant part-2
windows were decoded from the part-1 video at part-local timestamps, and both
of a case's jobs wrote the same shard path.

Usage:
    python scripts/extract_case.py CASE_ID PART LABELS_DIR VIDEO_PATH OUT_PATH \
                                   FREQUENCY_JSON

PART is the part the VIDEO_PATH file contains ('1', '2', '1.0', ... all work).
OUT_PATH's filename must carry the part, so two jobs for one case cannot
collide; `surgvu.extract.shard_filename` produces the canonical name.

FREQUENCY_JSON is the corpus-wide tool-rarity table from
`scripts/build_tool_frequency.py`, and it is required rather than optional.
Without it `stratify` computes rarity from this job's own windows, and a tool
that is rare across the corpus while being common inside this one case gets
deprioritised precisely where it was most needed -- with no error and no
visible difference in the shard.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.extract import (  # noqa: E402
    extract_window, part_number, shard_filename, write_shard,
)
from surgvu.labels import CaseLabels, normalize_part  # noqa: E402
from surgvu.sampling import enumerate_windows, stratify  # noqa: E402

PER_CASE_CAP = 160


def main(argv):
    if len(argv) != 6:
        raise SystemExit(__doc__)
    case_id, part, labels_dir, video_path, out_path, frequency_path = argv
    part = normalize_part(part)

    token = "_part%s" % part_number(part)
    if token not in Path(out_path).name:
        raise SystemExit(
            "refusing to write %r: a shard filename must carry its part "
            "(it must contain %r; the canonical name is %r). Two parts of one "
            "case writing the same path is exactly the collision this "
            "argument exists to prevent."
            % (out_path, token, shard_filename(case_id, part)))

    labels = CaseLabels.from_dir(labels_dir)
    all_windows = enumerate_windows(case_id, labels)
    in_part = [w for w in all_windows if normalize_part(w.part) == part]
    if not in_part:
        raise SystemExit(
            "%s part %s: no windows. %d windows exist for this case but none "
            "in this part (parts present: %s). Check the video filename's part "
            "against the label CSVs before assuming this case is empty."
            % (case_id, part, len(all_windows),
               sorted({w.part for w in all_windows})))

    # Corpus-wide rarity, not this job's local view. Required: see __doc__.
    frequency = json.loads(Path(frequency_path).read_text(encoding="utf-8"))
    if not any(frequency.values()):
        raise SystemExit(
            "%s has every tool at zero, which makes all windows equally rare "
            "and silently disables stratification." % frequency_path)

    windows = stratify(in_part, per_case_cap=PER_CASE_CAP, seed=7,
                       frequency=frequency)
    print("%s part %s: %d windows selected (of %d in this part, %d in the case)"
          % (case_id, part, len(windows), len(in_part), len(all_windows)))

    payload = []
    for i, window in enumerate(windows):
        # extract_window raises if `part` disagrees with `window.part`.
        frames = extract_window(video_path, window, part)
        payload.append((window, frames))     # write_shard drops short windows
        if (i + 1) % 20 == 0:
            print("  %d/%d" % (i + 1, len(windows)), flush=True)

    written = write_shard(payload, out_path)
    print("%s part %s: wrote %d of %d windows -> %s"
          % (case_id, part, written, len(payload), out_path))


if __name__ == "__main__":
    main(sys.argv[1:])
