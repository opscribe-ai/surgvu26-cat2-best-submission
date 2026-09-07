"""Run both perception experts over a directory of clips, once, to JSON.

This is the whole perception half of inference: video in, one record per case
out. It is deliberately thin -- every decision it could get wrong (which
frames, which threshold, which activation, which resolution) lives in
`surgvu.perceive` where it is unit-tested. What is left here is argument
parsing, a loop, and a progress line.

The output file is the contract with the question router, which never opens a
video. Written once at the end, after every clip has succeeded: a partial file
is worse than no file, because the router cannot tell the difference between
"this case was not perceived" and "this case has no tools".
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.perceive import (                                   # noqa: E402
    DEFAULT_FRAMES, decode_clip, find_clips, load_expert, perceive_clip,
)

SAMPLE_CLIPS = "/staging/groups/bhaskar_opscribe/surgvu/cat2_sample"
MODELS = "/staging/n/nkalthoff/surgvu26/models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", default=SAMPLE_CLIPS,
                        help="directory holding caseNNN/caseNNN.mp4")
    parser.add_argument("--tools-checkpoint",
                        default="%s/tools_efficientnet_v2_s.pt" % MODELS)
    parser.add_argument("--task-checkpoint",
                        default="%s/task_efficientnet_v2_s.pt" % MODELS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help="frames sampled evenly across each clip; recorded "
                             "in each record as n_frames")
    # 512 matches the training shards: frames were stored at 512 and the model
    # saw them resized from there to its own image_size. Preparing at some
    # other size would resample twice differently than training did.
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    tool_model, tool_meta = load_expert(args.tools_checkpoint, args.device)
    task_model, task_meta = load_expert(args.task_checkpoint, args.device)
    print("tools: image_size=%s thresholds=%s"
          % (tool_meta["image_size"],
             [round(t, 3) for t in tool_meta["thresholds"]]))
    print("task:  image_size=%s" % (task_meta["image_size"],))

    clips = find_clips(args.clips)
    print("clips: %d under %s" % (len(clips), args.clips), flush=True)

    records = {}
    for case, path in clips:
        started = time.time()
        frames = decode_clip(path, n_frames=args.frames, size=args.size)
        record = perceive_clip(frames, tool_model, tool_meta,
                               task_model, task_meta, device=args.device)
        records[case] = record
        print("%s  %2d frames  %5.1fs  task=%-34s tools=%s"
              % (case, record["n_frames"], time.time() - started,
                 record["task_top"], record["tools_present"] or "[]"),
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("wrote %d records to %s" % (len(records), out))


if __name__ == "__main__":
    main()
