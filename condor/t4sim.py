"""Make the container believe it is running on the grader's Tesla T4.

Bind-mounted over the container's `sitecustomize.py`, so CPython imports it
before any pipeline code runs. NOTHING IN THE IMAGE CHANGES -- this is
validation scaffolding that lives outside the artefact being validated, which
is the point: a test that edits the thing under test proves nothing.

Inert unless SURGVU_T4SIM=1 is set.

WHAT A T4 IS, operationally, and all three parts matter:

  1. 14.56 GiB of VRAM.               -> cap the caching allocator
  2. sm_75, so flash SDPA is          -> force the math kernel, which
     unavailable and attention must      materialises [28 x 5184 x 5184]
     fall back.                          = 3.0 GiB, held twice
  3. It REPORTS 14.56 GiB and         -> spoof get_device_properties, or
     capability 7.5 when asked.          frame_plan.select_plan reads the
                                         host card's real size, picks the
                                         5184-token plan, and we would be
                                         testing the OOM path instead of the
                                         fix.

Part 3 is the one that is easy to leave out and fatal to leave out. The fix
being validated is a decision made FROM the reported VRAM, so a simulator
that caps memory without spoofing the report exercises none of it.

Calibrated: forced-math at 5184 tokens measured 15.88 GiB on an H200 against
16.5 GiB on real sm_75 silicon (job 9716098). Within 4%.
"""
import os
import sys

_T4_TOTAL_BYTES = int(14.56 * 1024 ** 3)
_CONTEXT_RESERVE = int(0.40 * 1024 ** 3)

if os.environ.get("SURGVU_T4SIM") == "1":
    try:
        import torch

        if torch.cuda.is_available():
            _real = torch.cuda.get_device_properties
            _actual = _real(0).total_memory

            class _T4Props(object):
                """Enough of a device-properties object for our callers."""
                name = "Tesla T4 (simulated)"
                total_memory = _T4_TOTAL_BYTES
                major, minor = 7, 5
                multi_processor_count = 40

                def __getattr__(self, item):     # anything else, defer
                    return getattr(_real(0), item)

            torch.cuda.get_device_properties = lambda *a, **k: _T4Props()

            budget = _T4_TOTAL_BYTES - _CONTEXT_RESERVE
            if budget < _actual:
                torch.cuda.set_per_process_memory_fraction(budget / _actual)

            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)

            sys.stderr.write(
                "[t4sim] posing as a Tesla T4: reports %.2f GiB / sm_75, "
                "allocator capped to %.2f GiB of a real %.2f GiB, "
                "math SDPA forced\n"
                % (_T4_TOTAL_BYTES / 1024 ** 3, budget / 1024 ** 3,
                   _actual / 1024 ** 3))
    except Exception as exc:                     # noqa: BLE001 - never fatal
        sys.stderr.write("[t4sim] FAILED to engage: %r\n" % (exc,))
