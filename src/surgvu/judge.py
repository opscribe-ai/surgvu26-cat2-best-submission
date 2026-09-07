"""The decision VLM: a second pass that SEES the candidates and picks.

THE ARCHITECTURE, AND WHY IT IS NOT THE SAME AS EVIDENCE-CONDITIONING.

v5 answers a question twice, independently:

    frames -> CNNs + YOLO + variant head -> router  -> answer A
    frames -> Evidence VLM (frames only)            -> answer B

and `arbiter.arbitrate` picks between them with a fixed rule. This module
replaces that rule with a MODEL that sees the question, the perception
evidence, and BOTH candidates, and returns the final answer.

The alternative -- feeding the evidence into the answering VLM's own prompt
(`scripts/train_vlm.py --evidence-cache`) -- risks something this design does
not. A VLM shown the detector's findings can learn to READ THEM instead of the
pixels, and a VLM that re-encodes the perception stack cannot usefully
disagree with the router; it just agrees more confidently, and the second
opinion stops being a second opinion. Here, PASS 1 STAYS BLIND: it sees frames
and the question, nothing else, exactly as the shipped adapter was trained.
Dependence on the evidence is confined to the judging stage, which is supposed
to be dependent on everything.

WHY ZERO-SHOT FIRST. A trained judge needs (question, evidence, candidates) ->
gold triples, which means running the router AND pass 1 over ~23k training
windows; the evidence cache alone took 18 hours for 15k. Judging between two
given candidates is a far easier task than answering from scratch, so it is
worth finding out whether the base instruction-tuned model can already do it
before spending days generating data for one that is trained to.

WHAT WOULD MAKE THIS A BAD IDEA, stated up front so the measurement is not
read charitably. On the eleven graded cases the VLM's polar judgement was 5/7
against the router's 7/7, and it answered "Yes" to 6 of 7 where gold was 4/3 --
a Yes-lean inherited from a corpus that is 56.7% Yes (71.3% for tool
presence). A judge built on the same base model inherits that prior. It must
be measured against `fallback` and `challenger` on the same eleven cases
before it goes anywhere near a submission.

COST. One VLM pass measures 42.3 s steady and 171.4 s cold, and every Grand
Challenge invocation is cold. A second pass is not free: expect ~250-350 s of
the 600 s budget, cutting today's ~3.5x headroom to roughly 2x. That is why
`should_consult` exists -- the judge is skipped outright when the two
candidates already agree, which is most cases.
"""
import re

#: Rendered into the prompt for each candidate. Deliberately NOT "the router"
#: and "the VLM": naming the sources invites the model to pick by reputation
#: ("the neural one sounds smarter") instead of by evidence. Neutral labels
#: keep the comparison about the answers.
CANDIDATE_LABELS = ("Answer 1", "Answer 2")

#: What the judge must reply with to choose a candidate verbatim. Anything
#: else it writes is treated as a free-text answer (see `parse_judgement`).
#:
#: MATCHES ANY DIGIT, not just 1-2, so that an out-of-range label is caught
#: HERE and range-checked, instead of falling through to the free-text branch.
#: Restricted to [12] this pattern let "Answer 3" -- a confused reply naming a
#: candidate that does not exist -- be returned as the literal ANSWER, which
#: would put the string "Answer 3" in /output and score near zero. A reply
#: shaped like a choice is a choice attempt, and a failed one must fall back,
#: not be shipped.
CHOICE_PATTERN = re.compile(r"^\s*answer\s*(\d+)\s*$", re.IGNORECASE)

#: Same reasoning for the labelled-restatement branch ("Answer 1: Yes").
LABELLED_PATTERN = re.compile(r"^\s*answer\s*(\d+)\s*[:.\-]", re.IGNORECASE)


def normalize(text):
    """Lowercased, punctuation-trimmed, whitespace-collapsed."""
    return " ".join(str(text or "").split()).strip().rstrip(".").lower()


def should_consult(router_answer, vlm_answer):
    """False when consulting the judge cannot change the answer.

    THE COST CONTROL, AND IT IS THE DIFFERENCE BETWEEN VIABLE AND NOT. A
    second VLM pass costs ~42 s warm and ~171 s cold. When both candidates
    already say the same thing there is nothing to arbitrate, and paying that
    to confirm an agreement would spend most of the per-case budget for no
    possible change in output.

    Compares NORMALIZED text, so "Yes" and "yes." do not count as a
    disagreement worth 171 seconds.

    Breaks if: this returns True on agreement (the judge then runs on every
    case and the budget argument above stops holding), or if it compares raw
    strings (case and trailing punctuation would manufacture disagreements).
    """
    if not str(router_answer or "").strip():
        return False
    if not str(vlm_answer or "").strip():
        return False
    return normalize(router_answer) != normalize(vlm_answer)


def build_judge_prompt(question, evidence_lines, candidates):
    """The text the decision VLM is shown, alongside the same frames.

    `evidence_lines` is the already-rendered evidence text -- this module does
    not re-implement the renderers in `evidence_vlm._EVIDENCE_RENDERERS`,
    because a second copy of that formatting would drift from the first and
    the model would be shown two different descriptions of the same detector
    output depending on which stage rendered it.

    The instruction asks for a verbatim label OR a better answer. Allowing a
    third option matters: on the graded sample BOTH candidates were wrong for
    case124 (router "Bipolar Forceps", VLM "Clip applier", gold "Cadiere
    Forceps"), and a judge restricted to picking one of two would have been
    unable to do anything but choose the less wrong.
    """
    lines = []
    if evidence_lines:
        lines.append(evidence_lines.rstrip())
        lines.append("")
    lines.append("Question: %s" % (" ".join(str(question or "").split()),))
    lines.append("")
    lines.append("Two answers have been proposed:")
    for label, candidate in zip(CANDIDATE_LABELS, candidates):
        lines.append("  %s: %s" % (label, " ".join(str(candidate or "").split())))
    lines.append("")
    lines.append(
        "Look at the images and the findings above. Reply with exactly "
        "'%s' or '%s' if one of them is correct. If both are wrong, reply "
        "with the correct answer instead, as briefly as possible."
        % CANDIDATE_LABELS)
    return "\n".join(lines)


#: Whether the judge may substitute its OWN answer for both candidates.
#:
#: OFF, on measured evidence. The judge is a BASE Qwen3-VL-4B, not fine-tuned
#: on this corpus, and its own verification run answered "3" where the gold
#: answer was "Three" (cluster 9707616). That is a correct answer in the wrong
#: register, and BERTScore-F1 punishes register: the router's phrasing was
#: tuned against the reference answers and the judge's has not been.
#:
#: So a judge that writes its own answer can swap a well-phrased candidate for
#: a badly-phrased one and LOSE points while being more right -- the same
#: mechanism that cost case130 0.0288 for a missing full stop.
#:
#: The case FOR allowing it is real and stays recorded: on case124 both
#: candidates were wrong (router "Bipolar Forceps", VLM "Clip applier", gold
#: "Cadiere Forceps"), and a judge restricted to picking one could only choose
#: the less wrong. That is worth revisiting IF the judge is ever fine-tuned on
#: this corpus, or if free-text judgements are measured to help. Neither has
#: happened, so the default is the conservative one.
ALLOW_FREETEXT_ANSWER = False


def parse_judgement(reply, candidates, allow_freetext=ALLOW_FREETEXT_ANSWER):
    """(answer, source) from the judge's reply.

    source is 'choice' when it named a candidate, 'freetext' when it wrote
    its own answer, or 'unparseable' when it returned nothing usable -- and
    that third case returns None so the caller can fall back rather than ship
    an empty string, which scores 0, worse than any wrong answer.

    Breaks if: a reply naming a candidate is returned as free text (the
    candidate's exact wording matters -- the router's phrasing is tuned to the
    reference answers, and paraphrasing it costs BERTScore).
    """
    text = str(reply or "").strip()
    if not text:
        return None, "unparseable"
    match = CHOICE_PATTERN.match(text)
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < len(candidates):
            return candidates[index], "choice"
        return None, "unparseable"
    # A judge that writes "Answer 1: Yes" has still chosen; take the label
    # rather than its restatement, because the candidate's exact wording is
    # what was tuned against the reference answers.
    labelled = LABELLED_PATTERN.match(text)
    if labelled:
        index = int(labelled.group(1)) - 1
        if 0 <= index < len(candidates):
            return candidates[index], "choice"
        return None, "unparseable"
    if not allow_freetext:
        # Fall back rather than paraphrase. The caller treats None as "the
        # judge did not decide" and keeps the arbiter's own answer, which is
        # a well-phrased candidate rather than this model's register.
        return None, "freetext-suppressed"
    return text, "freetext"
