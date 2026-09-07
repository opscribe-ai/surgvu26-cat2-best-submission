"""ONE (attention backend x frame plan) measurement, on a fake T4.

WHY THIS EXISTS. v6.1's NF4 checkpoint was validated on an H200 (139.8 GiB,
peak 6.89 GiB) and called fixed. It then OOM'd on the grader's T4. The H200
measurement was not weak evidence, it was the WRONG evidence: sm_90 selects a
flash SDPA kernel that never materialises the LxL attention score matrix,
and sm_75 -- T4, and every sm_75 card in this pool -- cannot. Same weights,
same image: 6.89 GiB on sm_90, 16.5 GiB on sm_75.

    28 heads x 5184 tokens^2 x 4 B = 3.0 GiB per score matrix.

The pool has no literal T4, so this CAPS THE ALLOCATOR to a T4's capacity on
whatever sm_75 card we land on. A 47 GiB Quadro RTX 8000 then OOMs exactly
where the grader's 14.56 GiB card does, which turns a card we can actually
schedule into a faithful T4.

ONE CONFIG PER PROCESS, ON PURPOSE. A CUDA OOM leaves the allocator
fragmented and the cache poisoned; measuring a second config in the same
process would report that damage as the second config's cost. The driver
(probe_t4_memory.sh) re-execs this file per config.

Exercises the REAL call path -- surgvu.evidence_vlm.call_vlm, the same
function inference.py reaches through try_vlm_result -- rather than a
synthetic forward pass, so a pass here is a statement about the pipeline.
"""
import argparse
import json
import os
import sys
import time
import traceback

#: The grader's card, from its own log line:
#:     gpu=Tesla T4 vram=14.6 GiB capability=7.5
#: and its OOM text, "GPU 0 has a total capacity of 14.56 GiB".
T4_TOTAL_GIB = 14.56

#: set_per_process_memory_fraction bounds the CACHING ALLOCATOR, which does
#: not include the ~0.3-0.4 GiB CUDA context that exists on a real T4 before
#: a single tensor is allocated. Cap below the nameplate so "fits" here means
#: "fits there" rather than "fits by less than the context we forgot".
CONTEXT_RESERVE_GIB = 0.40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/opt/ml/model/qwen25vl-7b-nf4")
    ap.add_argument("--video", required=True)
    ap.add_argument("--question", default="What type of forceps is mentioned?")
    ap.add_argument("--attn", default="sdpa",
                    help="attn_implementation passed to from_pretrained")
    ap.add_argument("--no-math-sdp", action="store_true",
                    help="forbid the math SDPA backend, forcing mem-efficient "
                         "or a hard error -- the Phase 1 question")
    ap.add_argument("--force-math", action="store_true",
                    help="allow ONLY the math SDPA backend. This is what makes "
                         "a T4 out of any GPU: sm_75 cannot use flash, so it "
                         "materialises the score matrix, and forcing the same "
                         "kernel on sm_90 reproduces that cost exactly.")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--no-cap", action="store_true",
                    help="measure the card as it really is, no T4 ceiling")
    args = ap.parse_args()

    record = {
        "attn": args.attn, "no_math_sdp": args.no_math_sdp,
        "force_math": args.force_math,
        "frames": args.frames, "size": args.size,
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
    }

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from surgvu import evidence_vlm as ev
    from surgvu import frame_plan

    if not torch.cuda.is_available():
        record.update(status="no-cuda")
        print(json.dumps(record)); return 0

    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / (1024 ** 3)
    record.update(gpu=props.name, capability="%d.%d" % (props.major, props.minor),
                  device_gib=round(total_gib, 2))

    # THE EFFECTIVE CEILING IS THE SMALLER OF THE TWO, and which one binds
    # changes what a pass MEANS.
    #
    #   simulated  -- a card bigger than a T4, capped down to one. A pass is
    #                 an exact statement about the grader's card.
    #   card       -- an 11 GiB 2080 Ti, which is sm_75 but SMALLER than a
    #                 T4. Capping is impossible and unnecessary: the card is
    #                 already a harsher ceiling than the one we care about,
    #                 so a pass here PROVES a T4 fit with GiB to spare. A
    #                 failure here proves nothing about a T4 -- it may still
    #                 fit in 14.56 GiB. This asymmetry is why the row records
    #                 which bound applied instead of just a number.
    if not args.no_cap:
        budget = T4_TOTAL_GIB - CONTEXT_RESERVE_GIB
        if budget < total_gib:
            torch.cuda.set_per_process_memory_fraction(budget / total_gib)
            record.update(cap_gib=round(budget, 2), bound="simulated")
        else:
            record.update(cap_gib=round(total_gib, 2), bound="card")
    else:
        record.update(cap_gib=None, bound="none")
    # The Phase 1 lever. Flash needs sm_80+ and is unavailable here either
    # way; the live question is whether the cutlass mem-efficient kernel
    # (sm_50+) can take Qwen2.5-VL's mask, or whether transformers hands it
    # something that forces the math fallback.
    # THE OTHER HALF OF THE T4, and the reason this no longer needs sm_75
    # hardware. A T4 differs from an H200 in exactly two ways that matter:
    # how much memory it has, and which SDPA kernel it is allowed to use.
    # The cap above reproduces the first. This reproduces the second.
    #
    #   H200 (sm_90), free choice ....... peak  6.89 GiB   (job 9716130)
    #   Quadro RTX 8000 (sm_75) ......... peak 16.50 GiB   (job 9716098)
    #
    # Both ran this identical image at 5184 tokens. The 9.6 GiB difference is
    # the [28 heads x 5184 x 5184] score matrix that flash never builds and
    # the math fallback must. Forcing math on ANY card pays that same cost,
    # so `--force-math --no-cap` at 5184 tokens MUST land near 16.5 GiB.
    # That row is the calibration: if it does not reproduce the sm_75
    # measurement we already hold, this simulator is not trustworthy and no
    # other row in the sweep means anything.
    if args.force_math:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        record["kernel"] = "math (T4-equivalent)"
    elif args.no_math_sdp:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        record["kernel"] = "mem-efficient only"
    else:
        record["kernel"] = "free choice"

    plan = {"label": "probe", "n_frames": args.frames, "size": args.size,
            "multiscale": False,
            "tokens": frame_plan.vision_tokens(args.frames, args.size, False)}
    record["tokens"] = plan["tokens"]

    started = time.time()
    try:
        # Load with the attention implementation under test and seed the
        # module's own cache, so call_vlm below reuses THIS model rather than
        # loading a second copy through the unparameterised _load_model.
        processor = AutoProcessor.from_pretrained(args.model_dir)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_dir, device_map={"": "cuda"},
            attn_implementation=args.attn)
        model.eval()
        ev._MODEL_CACHE[(str(args.model_dir), "cuda")] = (model, processor)
        record["load_s"] = round(time.time() - started, 1)
        record["weights_gib"] = round(
            torch.cuda.max_memory_allocated() / (1024 ** 3), 2)

        frames = ev.sample_frames_planned({"path": args.video}, plan)
        answer = ev.call_vlm(frames, args.question, "",
                             model_dir=args.model_dir, device="cuda")
        record.update(status="ok", answer=str(answer)[:120])
    except Exception as exc:                    # noqa: BLE001 - the measurement
        record.update(status="OOM" if "OutOfMemory" in type(exc).__name__
                      or "out of memory" in str(exc).lower() else "error",
                      error=type(exc).__name__,
                      detail=" ".join(str(exc).split())[:220])
        traceback.print_exc(file=sys.stderr)

    record["peak_gib"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
    record["reserved_gib"] = round(torch.cuda.max_memory_reserved() / (1024 ** 3), 2)
    record["wall_s"] = round(time.time() - started, 1)
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
