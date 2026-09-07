"""A VLM that is only ever allowed to answer what the router could not.

WHAT THIS IS NOT
----------------
It is not an answerer. Qwen3-VL was measured as the SOLE answerer across all
11 public sample cases and lost decisively to a policy that never looks at a
pixel:

    CNN + router                0.8766
    zero-perception baseline    0.6959
    4B-fp16 open                0.5743      8B-NF4 closed   0.5501
    4B-fp16 closed              0.5216      8B-NF4 open     0.4923

The reason is structural rather than a prompting failure: the gold answers in
this benchmark are reconstructions of the LABEL TAXONOMY -- 12 instrument
classes and 8 task classes -- which is exactly what the two CNNs are trained
to predict and exactly what a general-purpose VLM has never seen. So this
module may not touch a question the router routes. `scripts/inference.py`
offers it INTENT_UNKNOWN_OPEN and nothing else, and there are tests in both
files saying so.

WHAT IT IS
----------
The router answers taxonomy questions from the CNNs and a handful of known
question shapes from constants. Anything it cannot classify falls to one
generic sentence -- `router.FALLBACK_OPEN` -- worth a measured 0.35-0.48. This
is a candidate replacement for THAT sentence, on THOSE questions only. The
floor it has to beat is low; the ceiling it can damage is zero, because a
question that reached it had no better answer waiting.

FAIL-SAFE, NOT BEST-EFFORT
--------------------------
Every method here returns None rather than raising, and the caller absorbs
what leaks anyway. A VLM that OOMs on a T4, times out, or emits an empty
string must cost its own answer and nothing else: the generic sentence is
still written and the process still exits 0. That is worth restating in
numbers -- a crash that writes no response scores 0 for the case, so the
downside of this module misbehaving is ~50x its best upside.

THE DEPLOYMENT INSTANCE MAY NOT HAVE A GPU
------------------------------------------
The weights are NF4-quantised, which is a bitsandbytes format and requires
CUDA. The challenge instance is documented as either No GPU or a single T4,
so on half the possible instance types this module CANNOT run at all. It
detects that and declines, which is the fail-safe path and not an error --
but it means shipping ~6 GB of image buys nothing on a No-GPU draw. That is a
cost-side fact for the ship/do-not-ship decision, recorded here because it is
invisible from the code alone.

FRAMES, NOT VIDEO
-----------------
`answer` takes the frames `perceive.decode_clip` already produced and never
re-opens the video. Two reasons, and the first is a rule rather than an
optimisation: `preprocess.prepare_frame` blurs the bottom UI band, and
"using the information available in the UI to make predictions is not
allowed". A VLM handed the raw file would read the instrument names straight
off the on-screen UI, which is precisely the thing the blur exists to stop.
The second reason is that the decode is already paid for.

MEASURED COST on sm_75 (RTX 2080 Ti, stricter than the T4):

    processor 2.4s | model load 5.9s from local disk | generate 4.1s
    peak GPU 7.24 GB | footprint 6.26 GB | 6.0 GB on disk

against a 600 s per-case budget of which the CNN path uses 6.2 s. Loading is
LAZY -- nothing is imported or read until an unrouted question actually
arrives -- so a case the router handles pays exactly zero of that.
"""
import re
import time

# Where the NF4 weights live on /staging. Inside the submission image they are
# baked in somewhere else, so this is a default and not a binding.
DEFAULT_MODEL_DIR = "/staging/n/nkalthoff/surgvu26/models/qwen3vl-8b-nf4"

# How many of the decoded frames to show it. Four is what the sm_75 gate
# measured at 4.1 s of generation; the frames are 30 s apart in a 30 s clip,
# so more of them buy redundancy rather than coverage.
DEFAULT_FRAMES = 4

# A hard cap on generated tokens. The gold answers this module competes with
# are 6 to 10 words, and the metric penalises elaboration -- a long answer is
# both slower and worse.
DEFAULT_MAX_NEW_TOKENS = 48

# Wall-clock seconds this module may consume, measured from the moment it is
# asked. The budget is 600 s per case and the CNN path uses 6.2-16.0 s, so
# this is generous by design: it exists to bound a pathological node (a cold
# read of 6 GB of weights over shared storage took 219 s once), not to shave
# a fast one.
DEFAULT_BUDGET_SECONDS = 240.0

# Longer than this and we are no longer answering, we are explaining.
MAX_ANSWER_WORDS = 24

# Openings that mean "I decline". The generic fallback is a MEASURED 0.35-0.48
# on an open question; a hedge about being an AI is not, and its shape (long,
# first-person, no surgical content) is the worst case for an embedding
# metric scored against a bare noun phrase. Matched at the start only, so an
# answer that happens to contain "unable" further in is kept.
_REFUSAL_RE = re.compile(
    r"^(i\s*(?:'m|am|can(?:not|'t)?|do(?:n't| not)?|would|could)\b"
    r"|as an ai\b|sorry\b|unable to\b|it is not possible\b"
    r"|there is not enough\b|unfortunately\b)")

# Chat models like to restate the task before answering.
_LABEL_RE = re.compile(r"^(answer|response|a)\s*[:\-]\s*", re.IGNORECASE)


# --------------------------------------------------------------------------
# prompt construction -- pure, and testable without torch
# --------------------------------------------------------------------------

def perception_context(perception):
    """One sentence naming what the CNNs found, or "" when they found nothing.

    The two classifiers are the only part of this system with any measured
    skill on this corpus, so their output is offered to the VLM as evidence
    rather than being hidden from it. It is stated as a report from a
    classifier, not as ground truth: `tools_present` is a thresholded list and
    is wrong often enough (case124's cadiere false negative, case126's needle
    driver) that a prompt asserting it as fact would be teaching the model to
    repeat our own errors with more confidence than we hold them.
    """
    if not isinstance(perception, dict):
        return ""
    parts = []
    tools = perception.get("tools_present")
    if isinstance(tools, (list, tuple)) and tools:
        parts.append("an instrument classifier reports %s"
                     % (", ".join(str(t) for t in tools),))
    task = perception.get("task_top")
    if isinstance(task, str) and task.strip():
        parts.append("an activity classifier reports %s" % (task.strip(),))
    if not parts:
        return ""
    return "For context, %s. These may be wrong." % ("; ".join(parts),)


def build_prompt(question, perception, use_context=True):
    """The text half of the message. Images are attached by the caller.

    Short and closed-ended on purpose. The metric is BERTScore against
    references that are mostly a bare noun phrase, so every sentence of
    preamble the model emits costs score; "at most 12 words" is the shortest
    instruction that still permits the one-sentence gold answers in the
    purpose family.
    """
    lines = ["These are frames from a robotic endoscopic surgery video."]
    if use_context:
        context = perception_context(perception)
        if context:
            lines.append(context)
    lines.append("Question: %s" % (" ".join(str(question or "").split()),))
    lines.append("Answer in at most 12 words. Give only the answer, with no "
                 "explanation and no preamble.")
    return "\n".join(lines)


def sanitize(text):
    """The model's raw generation -> a submittable string, or None.

    None means "keep the router's answer". Returning something unusable would
    be worse than returning nothing: `write_response` would collapse an empty
    string back to the generic fallback anyway, and a paragraph would score
    below it.

    Everything here is a bound, not a rewrite. We do not correct the model's
    content -- there is nothing to correct it against -- we only refuse output
    whose SHAPE is known to score badly.
    """
    if not isinstance(text, str):
        return None
    # First line only: a model that ignores "no explanation" usually obeys it
    # for one line and then keeps going.
    line = ""
    for candidate in text.splitlines():
        if candidate.strip():
            line = candidate.strip()
            break
    line = _LABEL_RE.sub("", line).strip()
    line = line.strip('"').strip("'").strip()
    line = " ".join(line.split())
    if not line:
        return None
    if _REFUSAL_RE.match(line.lower()):
        return None
    words = line.split()
    if len(words) > MAX_ANSWER_WORDS:
        line = " ".join(words[:MAX_ANSWER_WORDS])
    return line or None


def even_indices(total, wanted):
    """`wanted` indices evenly spaced over `total`, at bin centres.

    Mirrors `perceive.sample_frame_indices`, which cannot be imported here
    without dragging torch and cv2 into a module that is meant to be
    importable and testable without either. tests/test_inference_vlm.py pins
    the two against each other over a range of inputs so the copy cannot
    drift.
    """
    total = int(total)
    wanted = int(wanted)
    if total <= 0 or wanted <= 0:
        return []
    if total <= wanted:
        return list(range(total))
    step = total / float(wanted)
    return [int((i + 0.5) * step) for i in range(wanted)]


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

class QwenVlmFallback(object):
    """Qwen3-VL behind a gate, a deadline, and a promise never to raise.

    Construction is free: no import, no file read, no CUDA context. Everything
    expensive happens on the first `answer` call, so enabling this module and
    never hitting an unrouted question costs nothing measurable.
    """

    def __init__(self, model_dir=DEFAULT_MODEL_DIR, device=None,
                 n_frames=DEFAULT_FRAMES,
                 max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                 budget_seconds=DEFAULT_BUDGET_SECONDS,
                 use_context=True, log=None):
        self.model_dir = str(model_dir)
        self.device = device
        self.n_frames = int(n_frames)
        self.max_new_tokens = int(max_new_tokens)
        self.budget_seconds = float(budget_seconds)
        self.use_context = bool(use_context)
        self._log = log or (lambda message: None)
        self._model = None
        self._processor = None

    # -- loading -----------------------------------------------------------

    def available(self):
        """False when this instance cannot possibly run the weights.

        The NF4 checkpoint is a bitsandbytes artifact and bitsandbytes 4-bit
        needs CUDA. On the No-GPU deployment draw the honest answer is "not
        available", reported once and then declined -- not a stack trace on
        every case.
        """
        try:
            import torch
        except Exception:                       # noqa: BLE001 - see docstring
            return False
        return bool(torch.cuda.is_available())

    def _load(self):
        """(model, processor), cached. Raises; `answer` is what absorbs it."""
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        started = time.time()
        processor = AutoProcessor.from_pretrained(self.model_dir)
        self._log("vlm: processor in %.1fs" % (time.time() - started,))
        started = time.time()
        # device_map pins the whole model to one device. The alternative,
        # "auto", would silently offload layers to CPU on a card too small for
        # them and turn a 4 s generation into minutes -- a timeout dressed up
        # as a success.
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_dir, device_map={"": device})
        model.eval()
        self._log("vlm: weights in %.1fs on %s" % (time.time() - started, device))
        self._model, self._processor = model, processor
        return model, processor

    # -- generation --------------------------------------------------------

    def _images(self, frames):
        """The frames the VLM sees: RGB PIL images, evenly spaced.

        `decode_clip` hands back OpenCV BGR. Feeding that to a model trained
        on RGB is a silent, plausible-looking failure -- everything runs and
        the colours are wrong -- so the conversion is here and not left to the
        caller.
        """
        from PIL import Image

        picked = even_indices(len(frames), self.n_frames)
        return [Image.fromarray(frames[index][:, :, ::-1]) for index in picked]

    def _generate(self, images, prompt, deadline):
        """Raw decoded text from one forward pass. Torch lives here and only here."""
        import torch

        model, processor = self._load()
        if time.time() >= deadline:
            self._log("vlm: out of budget after loading; declining")
            return None

        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        inputs = inputs.to(model.device)

        started = time.time()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                stopping_criteria=_deadline_criteria(deadline))
        self._log("vlm: generate in %.1fs" % (time.time() - started,))
        prompt_length = inputs["input_ids"].shape[1]
        return processor.batch_decode(generated[:, prompt_length:],
                                      skip_special_tokens=True)[0]

    # -- the one public entry point ---------------------------------------

    def answer(self, question, perception, frames):
        """A better answer for an unrouted open question, or None.

        Never raises. Every early return is a decline, and a decline is the
        router's calibrated sentence, which is the outcome we already measured
        and already accept.
        """
        deadline = time.time() + self.budget_seconds
        try:
            if not question or not str(question).strip():
                return None
            if frames is None or len(frames) == 0:
                self._log("vlm: no frames; declining")
                return None
            if not self.available():
                self._log("vlm: no CUDA device; the NF4 weights cannot be "
                          "loaded on this instance. Declining -- the router's "
                          "answer stands.")
                return None
            prompt = build_prompt(question, perception, self.use_context)
            raw = self._generate(self._images(frames), prompt, deadline)
            cleaned = sanitize(raw)
            self._log("vlm: raw=%r -> %r" % (raw, cleaned))
            return cleaned
        except Exception as error:              # noqa: BLE001 - see docstring
            self._log("vlm: declined after %r" % (error,))
            return None


def _deadline_criteria(deadline):
    """A StoppingCriteria that stops generating once the deadline passes.

    `max_new_tokens` already bounds a healthy run to ~4 s. This bounds an
    UNhealthy one -- a node where each token takes a second -- and it is the
    only lever that works once `generate` has been entered. A partial answer
    is an acceptable outcome: `sanitize` will either make something of it or
    return None, and both are better than blowing the case's budget.
    """
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            import torch
            return torch.full((input_ids.shape[0],), time.time() >= deadline,
                              dtype=torch.bool, device=input_ids.device)

    return StoppingCriteriaList([_Deadline()])
