"""Extract one (case, part) as a DENSE shard: short bursts that carry motion.

The sparse shards sample 30 seconds at 1 fps because that mirrors what serving
samples -- but not what the video holds. The source is 60 fps and the graded
test clips are 1800 frames over 30 seconds, so the 1 fps pool has already
discarded 59 of every 60 frames of motion before any model sees it. A 3D
convolution over frames a second apart is looking at scene changes.

This writes the same windows with the same labels, narrowed to a 2-second
burst at the window's centre sampled at 15 fps. Same 30 frames per window,
same window count, same case split, same stratification -- so a dense shard is
a drop-in A/B against its sparse twin rather than a different dataset that
also happens to be denser. The only thing that changes is what the frames are
spaced by: 67 ms instead of 1000 ms.

Usage mirrors scripts/extract_case.py exactly, plus the two burst knobs:

    python scripts/extract_dense.py CASE_ID PART LABELS_DIR VIDEO_PATH \
                                    OUT_PATH FREQUENCY_JSON [SECONDS] [FPS]

WHY NOT JUST ENUMERATE 2-SECOND WINDOWS. That would multiply the window count
by fifteen and change which moments are represented, so any difference against
v2 would confound "denser frames" with "more and different training data".
Narrowing existing windows keeps every other variable fixed.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.extract import (  # noqa: E402
    dense_window, extract_window_dense, part_number, shard_filename,
    write_shard,
)
from surgvu.labels import CaseLabels, normalize_part  # noqa: E402
from surgvu.sampling import enumerate_windows, stratify  # noqa: E402

PER_CASE_CAP = 160
BURST_SECONDS = 2.0
BURST_FPS = 15


def main(argv):
    if len(argv) not in (6, 7, 8):
        raise SystemExit(__doc__)
    case_id, part, labels_dir, video_path, out_path, frequency_path = argv[:6]
    seconds = float(argv[6]) if len(argv) > 6 else BURST_SECONDS
    fps = int(argv[7]) if len(argv) > 7 else BURST_FPS
    part = normalize_part(part)

    token = "_part%s" % part_number(part)
    if token not in Path(out_path).name:
        raise SystemExit(
            "OUT_PATH %r does not carry %r. Two jobs for one case would write "
            "the same file and the survivor would be whichever finished last. "
            "Use surgvu.extract.shard_filename." % (out_path, token))

    labels = CaseLabels.from_dir(labels_dir)

    # THE ORDER HERE IS LOad-BEARING and must match scripts/extract_case.py
    # exactly: enumerate, filter to THIS PART, and only then stratify.
    # Stratifying before the part filter draws from a different candidate pool
    # and returns a different window set, which would leave every dense shard
    # holding different moments from its sparse twin -- and a 3D-vs-2D result
    # would then be measuring the sampler.
    all_windows = enumerate_windows(case_id, labels)
    in_part = [w for w in all_windows if normalize_part(w.part) == part]
    if not in_part:
        raise SystemExit(
            "%s part %s: no windows. %d windows exist for this case but none "
            "in this part (parts present: %s)."
            % (case_id, part, len(all_windows),
               sorted({w.part for w in all_windows})))

    frequency = json.loads(Path(frequency_path).read_text(encoding="utf-8"))
    if not any(frequency.values()):
        raise SystemExit(
            "%s has every tool at zero, which makes all windows equally rare "
            "and silently disables stratification." % frequency_path)

    # Same cap, same seed: `stratify` is seeded, so this reproduces the sparse
    # run's selection exactly rather than merely resembling it.
    windows = stratify(in_part, per_case_cap=PER_CASE_CAP, seed=7,
                       frequency=frequency)
    print("%s part %s: %d windows selected (of %d in this part, %d in the case)"
          % (case_id, part, len(windows), len(in_part), len(all_windows)))

    payload = []
    for i, window in enumerate(windows):
        burst = dense_window(window, seconds=seconds)
        frames = extract_window_dense(video_path, burst, part, fps=fps)
        payload.append((burst, frames))      # write_shard drops short windows
        if (i + 1) % 20 == 0:
            print("  %d/%d" % (i + 1, len(windows)), flush=True)

    written = write_shard(payload, out_path, fps=fps)
    print("%s part %s: wrote %d of %d windows -> %s (%d frames each, "
          "%.1fs at %d fps)"
          % (case_id, part, written, len(payload), out_path,
             int(round(seconds * fps)), seconds, fps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
