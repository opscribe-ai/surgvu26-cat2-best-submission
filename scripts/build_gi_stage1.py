#!/usr/bin/env python3
"""Stage 1 of the v6 curriculum from the OpScribe GI corpus.

WHY THIS AND NOT SSG-VQA
--------------------------
Both were built as stage-1 candidates. This one wins on every axis that
matters, measured rather than assumed:

                            GI corpus        SSG-VQA (kept subset)
    pairs                   86,991           54,978
    procedure families      9                1 (cholecystectomy)
    question style          natural language templated generator output
    SurgVU tool names       15,345 mentions  ~0

That last row is the decisive one. The GI corpus's answers name SurgVU's OWN
twelve-class taxonomy -- stapler 4,625, bipolar forceps 4,114, monopolar
curved scissors 3,190, prograsp forceps 2,086, clip applier 861, needle driver
469 -- because `surg396k`/EndoVis and the da Vinci sources share it. SSG-VQA's
frames come from CholecT45, whose taxonomy is grasper/hook/clipper/irrigator:
correct for its own domain, and none of SurgVU's names. Stage 1 on this corpus
teaches the model the vocabulary stage 2 will grade it on.

THE PUBLISHED TRAIN/VAL SPLIT LEAKS AND IS NOT USED
-----------------------------------------------------
Measured: 2,143 images appear in BOTH train.json and val.json (25% of val),
and 469 of 615 val directories are also in train. The split was made by
RECORD, so the same frame lands on both sides.

This module therefore ignores it and splits by IMAGE DIRECTORY -- a real
video/sequence -- so no directory's frames can appear on both sides. Note in
passing that the 72B model's reported eval loss (0.0037) was measured against
the leaking split and is optimistic; nothing here depends on it.

STAGE 1 IS SINGLE-IMAGE, STAGE 2 IS 16-FRAME, AND THAT IS DELIBERATE
----------------------------------------------------------------------
These are stills from nine unrelated corpora; most have no sequential
neighbours to build a window from. Qwen2.5-VL is natively multi-image and
processes each image identically, so a 1-image stage followed by a 16-frame
stage is an ordinary curriculum rather than a mismatch -- and stage 2, which
runs LAST, is the one whose shape has to match serving. What is NOT acceptable
is a mixture inside one stage, which is why this is uniformly single-image
rather than windowed-where-possible.

Frames ARE converted to 512x512 (`preprocess.prepare_frame`'s geometry), so
the two stages agree on everything except frame count.

CASE IDS ARE case_300..case_499
---------------------------------
`train_vlm.assign_case_split` splits by case and raises on any case in neither
list. Directories are hashed into 200 buckets, each bucket entirely inside one
split, so the no-leak property survives bucketing. SurgVU holds case_000..154
and the SSG-VQA builder holds case_900+, so this range collides with neither.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.extract import (  # noqa: E402
    JPEG_QUALITY, looks_like_a_written_jpeg,
)

N_BUCKETS = 200
CASE_BASE = 300
DEFAULT_SIZE = 512
VAL_FRACTION_BUCKETS = 20          # 20 of 200 buckets -> ~10% val


def stable_bucket(text, n=N_BUCKETS):
    """Deterministic bucket for a directory path.

    `hashlib`, not `hash()`: Python's built-in string hash is randomised per
    process (PYTHONHASHSEED), so `hash()` here would put the same directory in
    a different split on every run -- silently reshuffling train and val
    between the manifest build and any later rebuild.
    """
    digest = hashlib.sha1(str(text).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n


def case_for_dir(directory):
    return "case_%d" % (CASE_BASE + stable_bucket(directory))


def extract_pair(record):
    """(question, answer) from a `conversations` list, or None.

    Takes the FIRST human turn and the FIRST gpt turn. Records with neither --
    or with an empty answer -- are dropped: an empty target teaches the model
    to emit nothing, and a missing response scores 0 at grading.
    """
    question = answer = None
    for turn in record.get("conversations") or []:
        who, value = turn.get("from"), (turn.get("value") or "").strip()
        if who == "human" and question is None:
            question = value
        elif who == "gpt" and answer is None:
            answer = value
    if not question or not answer:
        return None
    return question, answer


def convert_image(src, dst, size=DEFAULT_SIZE):
    """Write `src` to `dst` as a `size`x`size` JPEG. False if unreadable.

    Squashes rather than letterboxes, matching `preprocess.prepare_frame`'s
    final `cv2.resize(frame, (size, size))`, so stage 1 and stage 2 frames have
    the same geometry.
    """
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        return False
    if image.shape[:2] != (size, size):
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", image,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
    if not ok:
        return False
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_bytes(buf.tobytes())
    return True


def out_path_for(image_path, out_root):
    """A collision-free flattened destination.

    The nine source corpora have overlapping basenames (`frame000.png` appears
    under many sequences), so the destination is keyed by a hash of the FULL
    source path rather than by its basename.
    """
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()
    return Path(out_root) / digest[:2] / (digest + ".jpg")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gi-json", nargs="+", required=True,
                        help="gi_train.json and gi_val.json -- BOTH, since the "
                             "published split leaks and is replaced here")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--splits-out", required=True)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--workers", type=int, default=8,
                        help="threads converting images. The cost is ceph read "
                             "latency plus PNG decode and cv2 releases the GIL "
                             "in both, so threads recover nearly all of it.")
    parser.add_argument("--limit", type=int, default=0,
                        help="debug only: stop after N records")
    args = parser.parse_args(argv)

    records = []
    seen = set()
    for path in args.gi_json:
        for record in json.load(open(path)):
            image = record.get("image")
            pair = extract_pair(record)
            if not image or pair is None:
                continue
            key = (image, pair[0])
            if key in seen:          # the two files overlap by 2,143 images
                continue
            seen.add(key)
            records.append((image, pair[0], pair[1], record.get("source")))
    print("records after dedupe: %d" % len(records), flush=True)
    if args.limit:
        records = records[:args.limit]

    # -- PASS 1: convert the UNIQUE images, on a thread pool.
    #
    # Serially this ran at ~220 images/min -- 5.6 hours for 74k -- because the
    # cost is ceph read latency plus PNG decode, not CPU. cv2 releases the GIL
    # in imread/imencode, so threads recover almost all of it.
    #
    # UNIQUE images, not records: the same frame carries several questions
    # (86,991 records over ~74k images), and converting per record would redo
    # the same decode repeatedly.
    #
    # Bounded batches rather than one pool.map over the whole list -- map()
    # drains its iterable eagerly, which is what got the SSG-VQA converter
    # held at 9,766MB of 8GB.
    unique = {}
    for image, _, _, _ in records:
        if image not in unique:
            unique[image] = out_path_for(image, args.out_root)
    print("unique images: %d" % len(unique), flush=True)

    # THE `dst.exists()` CHECK BELONGS INSIDE THE WORKER, NOT IN A LIST
    # COMPREHENSION OUT HERE.
    #
    # Written the obvious way -- `todo = [... if not dst.exists()]` -- this is
    # 74,000 SERIAL stat() calls on cephfs before a single image is converted.
    # Measured on job 9710274: ~15 minutes at roughly 0% CPU (7s of user CPU
    # across 15 minutes of wall clock), because stat on a cold cephfs inode is
    # pure latency and nothing was overlapping it.
    #
    # Inside the worker the same 74,000 stats run `--workers`-ways parallel,
    # alongside the conversions they gate. `convert_stage1_frames.py` already
    # did it this way; this file hoisted it out and paid for the difference.
    todo = list(unique.items())
    print("candidates: %d (already-present ones are skipped in-worker)"
          % len(todo), flush=True)

    bad = set()
    skipped = [0]
    present = set()               # sources whose destination is known good

    def _convert(pair):
        src, dst = pair
        # NOT `dst.exists()`: write_bytes is not atomic, so a killed job
        # leaves a ZERO-BYTE file that exists, gets skipped forever, and
        # kills training hours later with UnidentifiedImageError. See
        # surgvu.extract.looks_like_a_written_jpeg.
        if looks_like_a_written_jpeg(dst):
            skipped[0] += 1
            present.add(src)      # set.add is atomic under the GIL
            return None
        if convert_image(src, dst, args.size):
            present.add(src)
            return None
        return src

    BATCH = 512
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(todo), BATCH):
            for failed_src in pool.map(_convert, todo[start:start + BATCH]):
                if failed_src is not None:
                    bad.add(failed_src)
            if (start // BATCH) % 20 == 0:
                print("  %d/%d processed (%d skipped, %d unreadable)"
                      % (min(start + BATCH, len(todo)), len(todo),
                         skipped[0], len(bad)), flush=True)

    # -- PASS 2: the manifest, in RECORD order so it does not depend on which
    # thread finished first.
    # NO `dst.exists()` HERE. PASS 1 already knows which destinations are good
    # -- it either wrote them or skipped them because they were present -- so
    # re-statting is 86,991 SERIAL cephfs calls (~27 minutes at the measured 54
    # stats/second) to rediscover what `present` already holds.
    #
    # This is the same mistake as the conversion loop, one pass later: PASS 1
    # was parallelised and PASS 2 was left statting. Observed on job 9710294,
    # which finished converting and then sat at 1,279 CPU-seconds across nearly
    # three hours of wall clock.
    manifest, missing, converted = [], len(bad), len(todo) - len(bad) - skipped[0]
    for i, (image, question, answer, source) in enumerate(records, 1):
        dst = unique[image]
        if image not in present:
            continue
        directory = str(Path(image).parent)
        manifest.append({
            "case": case_for_dir(directory),
            "part": "1.0",
            "t_start": float(i),
            "t_stop": float(i) + 1.0,
            "question": question,
            "answer": answer,
            "intent": "gi_stage1",
            "frame_dir": str(dst.parent),
            "frame_paths": [str(dst)],
            "provenance": {"generator": "scripts/build_gi_stage1.py",
                           "source": source, "image": image},
        })


    buckets = sorted({stable_bucket(Path(m["provenance"]["image"]).parent)
                      for m in manifest})
    val_buckets = set(buckets[-VAL_FRACTION_BUCKETS:])
    splits = {
        "train": sorted("case_%d" % (CASE_BASE + b) for b in buckets if b not in val_buckets),
        "val": sorted("case_%d" % (CASE_BASE + b) for b in val_buckets),
        # R30 requires a non-empty heldout list. The SurgVU graded eleven are
        # not in this corpus at all, which makes the list both non-empty and
        # true.
        "heldout": ["case_%d" % n for n in range(122, 133)],
        "meta": {"generator": "scripts/build_gi_stage1.py",
                 "split_unit": "image directory, bucketed",
                 "note": "The published train/val split leaks (2,143 images on "
                         "both sides) and is deliberately not used."},
    }

    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_out, "w", encoding="utf-8") as handle:
        for record in manifest:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    Path(args.splits_out).write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print("wrote %s (%d records, %d unreadable images)"
          % (args.manifest_out, len(manifest), missing))
    print("wrote %s (train=%d val=%d cases)"
          % (args.splits_out, len(splits["train"]), len(splits["val"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
