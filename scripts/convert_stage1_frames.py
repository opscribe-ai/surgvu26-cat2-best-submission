#!/usr/bin/env python3
"""CholecT45's PNGs -> 512px JPEGs, streamed from the zip.

WHY STREAM RATHER THAN UNZIP
------------------------------
CholecT45.zip is 59 GB across 90,728 files. Unzipping it costs that much
scratch and 90k inodes before a single frame is converted, and the PNGs are
then deleted anyway. `zipfile` reads members individually, so this decodes
each entry, writes one JPEG, and never materialises the PNG.

WHY 512px JPEG AND NOT THE ORIGINAL PNG
-----------------------------------------
Stage 2's frames are 512px JPEGs (`build_qa_pairs.py --size 512`, quality set
by JPEG_QUALITY). Stage 1 must match: a curriculum whose two stages differ in
resolution or codec teaches the model that those differences carry meaning.
It also collapses 59 GB to roughly 4.5 GB, which is the difference between
fitting beside the stage-2 rebuild and not.

GEOMETRY MATCHES `preprocess.prepare_frame`, WHICH SQUASHES TO A SQUARE
-------------------------------------------------------------------------
Stage 2's frames come out of `prepare_frame`, whose last line is
`cv2.resize(frame, (size, size), INTER_CUBIC)` -- a SQUARE 512x512, aspect
deliberately not preserved. This module squashes the same way, with the same
interpolation. Written the obvious way first it letterboxed to 512 on the long
side (512x288 for 16:9), which would have handed the model two different frame
geometries across the curriculum and taught it the difference meant something.

WHAT IS DELIBERATELY NOT COPIED from `prepare_frame`: `crop_side_margins` and
`blur_ui_band`. Both are SurgVU-specific -- the side margins and the bottom UI
band exist in da Vinci recordings, and the blur is there because the challenge
forbids reading that band. CholecT45 has neither, so applying them would crop
and blur real surgical content, destroying the very instrument views the tool
questions ask about. Stage 1 matches the geometry, not the redactions.
"""
import argparse
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# IMPORTED, NOT RESTATED. Stage 2 encodes through this same constant
# (build_qa_pairs.py imports it from here too), and the two stages' frames
# must be byte-comparable. Written as a literal first, this said 92; the real
# value is 90, which is the whole argument for importing it.
from surgvu.extract import (  # noqa: E402
    JPEG_QUALITY, looks_like_a_written_jpeg,
)

DEFAULT_SIZE = 512

#: How many zip members are read into memory at once. See the loop below:
#: ThreadPoolExecutor.map drains its iterable eagerly, so the batching is what
#: bounds memory, not the pool size. 256 x ~500KB is about 128MB live.
BATCH_SIZE = 256


def convert_one(payload, size=DEFAULT_SIZE):
    """PNG bytes -> JPEG bytes, squashed to `size` x `size`, or None.

    Returns None (rather than raising) for an undecodable member: one corrupt
    frame in 90,728 must not kill a multi-hour job, and the caller tallies it.
    """
    buf = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        return None
    if image.shape[:2] != (size, size):
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    ok, out = cv2.imencode(".jpg", image,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
    return out.tobytes() if ok else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-zip", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.frames_zip) as zf:
        members = [n for n in zf.namelist()
                   if n.endswith(".png") and "/data/VID" in n]
        members.sort()
        print("members: %d" % len(members), flush=True)

        # One ZipFile handle is NOT thread-safe for concurrent reads, so the
        # READ stays on this thread and only the decode+encode+write fans out.
        # That is the expensive half anyway: cv2 releases the GIL in both.
        written = [0]
        failed = []

        def handle(item):
            name, payload = item
            video = Path(name).parts[-2]
            stem = Path(name).stem
            out_dir = out_root / video
            out_path = out_dir / (stem + ".jpg")
            # Size-checked, not just existence: an interrupted write leaves a
            # zero-byte file that a plain .exists() would skip forever.
            if looks_like_a_written_jpeg(out_path):
                return None
            jpeg = convert_one(payload, args.size)
            if jpeg is None:
                return name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(jpeg)
            return None

        # BOUNDED BATCHES, NOT `pool.map` OVER A GENERATOR.
        #
        # ThreadPoolExecutor.map() drains its iterable EAGERLY -- it submits
        # every item before yielding the first result. Handed a generator that
        # reads each member as it goes, that pulls all 90,728 PNGs (59GB) into
        # memory at once. Measured: the job reached 9,766MB against an 8GB
        # request and was held by the scheduler.
        #
        # Reading one bounded batch, converting it, then reading the next caps
        # live payloads at BATCH_SIZE (~128MB at 500KB/frame) whatever the zip
        # size.
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for start in range(0, len(members), BATCH_SIZE):
                batch = members[start:start + BATCH_SIZE]
                payloads = [(name, zf.read(name)) for name in batch]
                for bad in pool.map(handle, payloads):
                    if bad is not None:
                        failed.append(bad)
                    else:
                        written[0] += 1
                done += len(batch)
                del payloads
                if done % 5000 < BATCH_SIZE:
                    print("  %d/%d converted (%d undecodable)"
                          % (done, len(members), len(failed)), flush=True)

    print("done: %d processed, %d undecodable" % (written[0], len(failed)))
    for name in failed[:10]:
        print("  undecodable: %s" % name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
