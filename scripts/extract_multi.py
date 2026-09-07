"""Extract one (case, part) as a MULTI-BURST shard: motion, spread across the window.

WHAT THIS FIXES. The centre-only dense pool gave every 3D model 16 frames --
1.07 s at 15 fps -- of a 30-second labelled window, while the 2D path sampled
16 frames spread across the whole of it. Every 3D result we have was measured
under that handicap, so "temporal modelling does not help" and "the model saw
3.6% of the window" are not separated by anything we ran.

This writes the same windows with the same labels, as FOUR bursts of 8 frames
at 15 fps, evenly spread at 12.5 / 37.5 / 62.5 / 87.5% of each window:

    centre only (v3)  [              ####              ]   30 frames
    spread     (v4)   [   ##      ##      ##      ##   ]   32 frames

Same window set, same case split, same stratification seed, same frame
spacing. 32 frames per window against the old 30, so the pool costs ~41 GB
against 38 and roughly 1.3x the decode time -- four seeks per window instead
of one, and a seek is the expensive part.

WHAT THE EXTRA BURSTS BUY, beyond coverage: four bursts are four genuinely
different clips of one window, so the trainer can yield several per epoch.
`ShardClips` yields ONE clip per window per epoch while `ShardFrames` yields
thirty frames, which means that at 20 epochs each the 3D models got ~30x fewer
gradient samples per window than the 2D model did. That asymmetry is not a
property of 3D convolution; it is a property of how the loader was written.

Usage mirrors scripts/extract_dense.py, plus the burst count:

    python scripts/extract_multi.py CASE_ID PART LABELS_DIR VIDEO_PATH \\
                                    OUT_PATH FREQUENCY_JSON [BURSTS] [SECONDS] [FPS]

BURST LENGTH. 8 frames at 15 fps is 0.5334 s, chosen because Kinetics-400
weights were fitted on 16 frames at 25 fps = 0.64 s. A burst far shorter than
that hands the pretrained temporal filters input from outside the regime they
were trained in, which would be a second confound rather than a control.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.extract import (  # noqa: E402
    extract_window_multi, part_number, shard_filename, write_shard,
)
from surgvu.labels import CaseLabels, normalize_part  # noqa: E402
from surgvu.sampling import enumerate_windows, stratify  # noqa: E402

PER_CASE_CAP = 160
BURSTS = 4
BURST_SECONDS = 0.5334
BURST_FPS = 15


def main(argv):
    if len(argv) not in (6, 7, 8, 9):
        raise SystemExit(__doc__)
    case_id, part, labels_dir, video_path, out_path, frequency_path = argv[:6]
    bursts = int(argv[6]) if len(argv) > 6 else BURSTS
    seconds = float(argv[7]) if len(argv) > 7 else BURST_SECONDS
    fps = int(argv[8]) if len(argv) > 8 else BURST_FPS
    part = normalize_part(part)

    token = "_part%s" % part_number(part)
    if token not in Path(out_path).name:
        raise SystemExit(
            "OUT_PATH %r does not carry %r. Two jobs for one case would write "
            "the same file and the survivor would be whichever finished last. "
            "Use surgvu.extract.shard_filename." % (out_path, token))

    labels = CaseLabels.from_dir(labels_dir)

    # THE ORDER HERE IS LOAD-BEARING and must match scripts/extract_case.py and
    # scripts/extract_dense.py exactly: enumerate, filter to THIS PART, and
    # only then stratify. Stratifying before the part filter draws from a
    # different candidate pool, and every shard would then hold different
    # moments from its sparse and dense twins -- turning a 3D-vs-2D result
    # into a measurement of the sampler.
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

    windows = stratify(in_part, per_case_cap=PER_CASE_CAP, seed=7,
                       frequency=frequency)
    print("%s part %s: %d windows selected (of %d in this part, %d in the case)"
          % (case_id, part, len(windows), len(in_part), len(all_windows)))

    per_window = bursts * int(round(seconds * fps))
    payload = []
    dropped = 0
    for i, window in enumerate(windows):
        frames = extract_window_multi(video_path, window, part, bursts=bursts,
                                      seconds=seconds, fps=fps)
        if not frames:
            dropped += 1
            continue
        # The row keeps its ORIGINAL 30 s length, unlike the dense pool where
        # the row was the burst. The frames now span the whole window, so a
        # row claiming 0.53 s would be a false description of its own content
        # -- and `read_shard` consumers use `length` to reason about what a
        # window covers.
        payload.append((window, frames))
        if (i + 1) % 20 == 0:
            print("  %d/%d" % (i + 1, len(windows)), flush=True)

    # frames_per_window is passed EXPLICITLY here, which production callers of
    # write_shard normally must not do. It is required: the writer derives the
    # expected depth from window.length * fps, which for a 30 s row at 15 fps
    # is 450 -- not the 32 a multi-burst window actually holds. Without this
    # every window would be judged short and the shard would come back empty.
    written = write_shard(payload, out_path, fps=fps,
                          frames_per_window=per_window)
    print("%s part %s: wrote %d of %d windows -> %s (%d bursts x %d frames = "
          "%d per window, %.4fs each at %d fps, %d dropped for short reads)"
          % (case_id, part, written, len(windows), out_path, bursts,
             int(round(seconds * fps)), per_window, seconds, fps, dropped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
