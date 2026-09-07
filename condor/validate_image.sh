#!/bin/bash
set -uo pipefail
# The SHIPPED IMAGE over all eleven public sample cases, /input read-only.
#
#   condor/validate_image.sh [image.sif]
#
# WHY THIS EXISTS. Nothing validated the artifact that ships.
#
#   containers/build_submission.sh   runs ONE case (case127) through the image
#   condor/validate.sh               runs all eleven, but through
#                                    scripts/inference.py inside the TRAINING
#                                    image, which is a different filesystem, a
#                                    different python, and a different set of
#                                    installed packages
#
# Both are useful and neither answers "does the thing we upload produce the
# right eleven answers". A container can pass a one-case smoke and still carry
# a missing font, an unwritable temp dir that only a second invocation hits, or
# a model file that resolves for one expert and not the other. The submission
# is two per phase; discovering that from a leaderboard result is expensive.
#
# `apptainer run --containall` with `-B in:/input:ro -B out:/output` is exactly
# how containers/build_submission.sh invokes it and exactly how the grader
# mounts it: read-only input, writable output, no host filesystem leaking in.
# A container that works only because it can see $HOME passes every test but
# the real one.

echo "host: $(hostname)  start: $(date)"
IMAGE="${1:-/staging/n/nkalthoff/surgvu26/surgvu26-submission.sif}"
# OVERRIDABLE, and the reason is the most expensive lesson of this project.
#
#   condor_submit condor/validate_image.sub gpus=1 \
#       sample=/staging/n/nkalthoff/surgvu26/cat2_sample_graded
#
# `cat2_sample`'s question files are NOT the questions Grand Challenge asks of
# these clips. Comparing the sample files against the grader's own per-case
# logs (v6 run, 2026-08-30) shows FOUR of the eleven differ:
#
#   case125  sample "...in this surgical step?"   graded "...in this surgical procedure?"
#   case126  sample "...used in this clip?"       graded "...used in the procedure?"
#   case127  sample "What organ is being manipulated?"
#            graded "What is the location of the surgical procedure?"
#   case131  sample "Is tissue being cut during this clip?"
#            graded "Is the surgical procedure being performed an open surgery?"
#
# The first two reword harmlessly and route the same way. The last two route
# somewhere else entirely, and BOTH were answered wrong on the leaderboard --
# case127 answered a location question with a procedure type, case131 called
# robotic endoscopic footage an open surgery. Neither was visible here,
# because this harness scored a question the grader never posed.
#
# RETIRED FIGURE: "0.9309 on the graded sample, 10/11 exact" was measured
# against the SAMPLE questions, and four of the eleven differ from what the
# grader asks -- two of them (127, 131) routed somewhere else entirely and
# were wrong on the leaderboard while this harness reported 11/11. The real
# number is the prelim leaderboard's 0.8558, and 0.9128 after those two were
# fixed on 2026-08-31. Validate against cat2_sample_graded, not the default:
#
#   condor_submit condor/validate_image.sub gpus=1 t4sim=1 \
#       sample=/staging/n/nkalthoff/surgvu26/cat2_sample_graded
#
# So "0.9309 on the graded sample, 10/11 exact" -- the number EXPECTED below
# is pinned to, and the number this project has steered by since v4 -- was
# measured on the WRONG QUESTIONS for two of the eleven cases. Run BOTH
# sample sets before shipping; EXPECTED only applies to the default one.
SAMPLE="${SAMPLE_DIR:-/staging/groups/bhaskar_opscribe/surgvu/cat2_sample}"

echo '{}' > image_validation.json

[ -f "$IMAGE" ]  || { echo "FATAL: no image at $IMAGE"; exit 42; }
[ -d "$SAMPLE" ] || { echo "FATAL: $SAMPLE not visible on this node"; exit 75; }
command -v apptainer >/dev/null || { echo "FATAL: no apptainer"; exit 42; }

echo "image:  $IMAGE ($(stat -c %s "$IMAGE") bytes)"
echo "sample: $SAMPLE"
echo

PASS=0; FAIL=0
printf '%-9s %-8s %8s  %s\n' case exit wall answer
printf -- '---------------------------------------------------------------\n'
: > image_answers.tsv
: > image_vlm.tsv

# ---- --nv, AND ONLY WHEN A GPU WAS ACTUALLY ASSIGNED ------------------------
# WITHOUT THIS THE GPU DRAW IS UNTESTABLE. `apptainer run --containall` does
# not expose the host's NVIDIA driver or libraries unless `--nv` is passed, so
# a job that condor gave a real card to still sees torch.cuda.is_available()
# == False INSIDE the image. Validation 9698741 was assigned GPU-7990e801,
# logged "VLM: no CUDA device available" for all eleven cases, and reported
# "passed 11, failed 0" -- a green GPU validation that could not, structurally,
# have exercised the GPU path it exists to exercise.
#
# CONDITIONAL, because the No-GPU draw is a real deployment case that must keep
# working: `--nv` on a node with no NVIDIA libraries is at best a warning and
# at worst a failure to start, and it would be absurd to break the CPU
# validation in order to fix the GPU one. _CONDOR_AssignedGPUs is set by the
# startd only when GPUs were actually granted -- not merely requested -- so it
# answers exactly the question that matters here.
# _CONDOR_AssignedGPUs ONLY. CUDA_VISIBLE_DEVICES was in this test and had to
# come out: CHTC sets it to the sentinel "10000" on NON-GPU nodes -- a device
# index that deliberately does not exist, so nothing can see a GPU. Job
# 9705191 requested no GPU, was assigned none (condor_q: RequestGPUs and
# AssignedGPUs both undefined), and still printed "GPU assigned (10000)".
#
# The run was still correct -- no device means torch.cuda.is_available() is
# False and the VLM declines, which is the No-GPU draw behaving exactly as it
# should. But the LOG WAS WRONG, and a log that misreports which draw was
# tested is the same failure as a test that measures the wrong thing: someone
# reading "GPU assigned" on the CPU run concludes the No-GPU path was never
# validated, or worse, that it was and the VLM silently did nothing.
# ---- OPTIONAL: POSE AS THE GRADER'S T4 --------------------------------------
# The pool has no T4 and the one schedulable sm_75 card is PI-owned (0 slots
# willing, ~14 h of backfill). So instead of finding the hardware, impose it:
# condor/t4sim.py is bind-mounted over the container's sitecustomize.py and
# makes the process report 14.56 GiB / sm_75, cap its allocator to match, and
# use the math SDPA kernel that sm_75 is stuck with.
#
#   condor_submit condor/validate_image.sub gpus=1 t4sim=1 \
#       sample=/staging/n/nkalthoff/surgvu26/cat2_sample_graded
#
# THIS IS THE GATE BEFORE AN UPLOAD. Getting a container and a 5 GB model
# tarball onto Grand Challenge is expensive enough that "it passed on an
# H200" is not worth the round trip -- that is precisely how v6 shipped with
# a VLM that OOM'd on all eleven graded cases. A green run here means the
# whole pipeline, not a probe, survived the grader's memory and the grader's
# attention kernel.
#
# The image itself is untouched: the simulator is bound in from outside, so
# what passes validation is byte-identical to what gets uploaded.
# INJECTED AS A DIRECTORY ON PYTHONPATH, NOT AS A FILE OVER site-packages.
# Checked against the built image rather than assumed:
#
#   PYTHONPATH=[]                                        (empty -- nothing to clobber)
#   sitepkgs: ['/opt/conda/lib/python3.11/site-packages']
#   ls .../site-packages/sitecustomize.py -> No such file or directory
#
# There is no sitecustomize.py to bind OVER, and `apptainer --containall`
# cannot create a missing FILE mountpoint in a read-only image -- that bind
# would have failed and taken the whole validation with it. A directory bind
# plus PYTHONPATH works either way: CPython's site.py imports `sitecustomize`
# from any sys.path entry, and PYTHONPATH entries are on sys.path by then.
T4SIM_BIND=""
T4SIM_ENV=""
if [ "${SURGVU_T4SIM:-0}" = "1" ]; then
    _t4sim="$PWD/condor/t4sim.py"
    [ -f "$_t4sim" ] || { echo "FATAL: SURGVU_T4SIM=1 but $_t4sim is missing"; exit 42; }
    rm -rf t4simdir && mkdir -p t4simdir
    cp "$_t4sim" t4simdir/sitecustomize.py
    T4SIM_BIND="-B $PWD/t4simdir:/t4sim:ro"
    T4SIM_ENV="--env PYTHONPATH=/t4sim"
    echo "T4 SIMULATION ENGAGED -- this run is not measuring this node's GPU"
fi

# ---- the MODEL SIDECAR ------------------------------------------------------
# WITHOUT THIS THE BUNDLE IS UNTESTABLE, and it would fail quietly. Grand
# Challenge extracts the optional model tarball to /opt/ml/model/ at runtime;
# nothing mounts it here unless we say so. A validation run without the bind
# resolves to the image's own NF4 weights and finds no judge -- so it would
# measure v5's models while reporting on a build whose whole point is int8 plus
# a decision VLM, and every line of its output would look correct.
#
# That is the same shape as the --nv bug earlier today: a correct pipeline
# making correct decisions from a premise the harness set wrong.
#
#   condor_submit condor/validate_image.sub gpus=1 #       sidecar=/staging/n/nkalthoff/surgvu26/models_sidecar
#
# Absent, the run tests the image alone -- which is a REAL deployment state
# (a submission uploaded without the tarball) and worth validating on purpose.
SIDECAR_BIND=""
SIDECAR_DIR="${SIDECAR_DIR:-}"
if [ -n "$SIDECAR_DIR" ]; then
    if [ ! -d "$SIDECAR_DIR" ]; then
        echo "FATAL: sidecar requested but $SIDECAR_DIR does not exist"
        exit 1
    fi
    SIDECAR_BIND="-B $SIDECAR_DIR:/opt/ml/model"
    echo "model sidecar: $SIDECAR_DIR -> /opt/ml/model"
    ls "$SIDECAR_DIR" | sed 's/^/  /'
else
    echo "no model sidecar bound; testing the image's own weights (a real"
    echo "  deployment state: a submission uploaded without the tarball)"
fi

# Extract the image's OWN arbiter mode before the case loop, so the comparison
# below is against what this container actually does rather than what the repo
# currently says. One apptainer exec, no GPU needed.
apptainer exec --containall "$IMAGE" \
    python -c "import json;print(json.load(open('/opt/algorithm/config/arbiter.json'))['mode'])" \
    > image_arbiter_mode.txt 2>/dev/null \
    || echo "challenger" > image_arbiter_mode.txt
echo "image arbiter mode: $(cat image_arbiter_mode.txt)"

NV_FLAG=""
if [ -n "${_CONDOR_AssignedGPUs:-}" ]; then
    NV_FLAG="--nv"
    echo "GPU assigned ($_CONDOR_AssignedGPUs); running with --nv"
else
    echo "no GPU assigned; running without --nv (the No-GPU deployment draw)" \
         "[CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}, ignored:" \
         "CHTC sets it to a nonexistent sentinel index on non-GPU nodes]"
fi

for dir in "$SAMPLE"/case*/; do
    case_id=$(basename "$dir")
    [ -f "$dir/$case_id.mp4" ] || continue
    rm -rf vin vout && mkdir -p vin vout
    cp "$dir/$case_id.mp4" vin/endoscopic-robotic-surgery-video.mp4
    cp "$dir/${case_id}_question.json" vin/visual-context-question.json

    start=$(date +%s.%N)
    # PER-CASE STDERR, not just the combined file. Appending every case to one
    # image_validation.err is how job 9716130 recorded eight "VLM: 2 call(s)
    # ... answer='yes'" lines that CANNOT BE ATTRIBUTED TO A CASE. That log is
    # the only record of what the VLM said, and without the pairing it cannot
    # answer the one question worth asking of it -- would arming this intent
    # have helped or hurt? -- so the run has to be repeated to learn anything.
    # The combined file is still written, because the report below greps it.
    mkdir -p caseerr
    apptainer run --containall $NV_FLAG $SIDECAR_BIND $T4SIM_BIND $T4SIM_ENV \
        --env "SURGVU_T4SIM=${SURGVU_T4SIM:-0}" \
        -B "$PWD/vin":/input:ro -B "$PWD/vout":/output \
        "$IMAGE" >/dev/null 2>"caseerr/$case_id.err"
    rc=$?
    cat "caseerr/$case_id.err" >> image_validation.err
    wall=$(echo "$(date +%s.%N) - $start" | bc)

    # What the VLM actually did on THIS case, in one field. `absorbed` is the
    # dangerous one: try_vlm_result swallows every exception and keeps the
    # router's answer, so a crashed VLM and an agreeing VLM produce identical
    # output. This is the only place the difference survives.
    ce="caseerr/$case_id.err"
    vlm_answer=$(sed -n "s/.*VLM: [0-9]* call(s).*answer='\(.*\)'.*/\1/p" "$ce" | tail -1)
    if   grep -q "OutOfMemoryError" "$ce"; then vlm_state="OOM"
    elif grep -q "the VLM seam raised" "$ce"; then vlm_state="absorbed"
    elif [ -n "$vlm_answer" ];             then vlm_state="spoke"
    elif grep -q "no CUDA device available" "$ce"; then vlm_state="no-gpu"
    else vlm_state="silent"; fi
    printf '%s\t%s\t%s\n' "$case_id" "$vlm_state" "$vlm_answer" >> image_vlm.tsv

    answer=$(cat vout/visual-context-response.json 2>/dev/null)
    # A MISSING OR EMPTY RESPONSE IS A FAILURE, not a blank row. The grader
    # scores a missing response as zero, so it must not look like a pass here.
    if [ "$rc" -ne 0 ] || [ -z "$answer" ] || [ "$answer" = "null" ]; then
        FAIL=$((FAIL+1)); mark="FAIL"
    else
        PASS=$((PASS+1)); mark="ok"
    fi
    printf '%-9s %-8s %7.1fs  %s\n' "$case_id" "$mark($rc)" "$wall" "${answer:0:44}"
    printf '%s\t%s\n' "$case_id" "$answer" >> image_answers.tsv
done

echo
echo "passed $PASS, failed $FAIL"

python3 - <<'PY'
import json, os
rows = {}
if os.path.exists("image_answers.tsv"):
    for line in open("image_answers.tsv"):
        parts = line.rstrip("\n").split("\t", 1)
        if len(parts) == 2:
            rows[parts[0]] = parts[1]

# THE ANSWERS THE SHIPPED IMAGE MUST PRODUCE.
#
# Generation 2 (2026-08-25), the --yolo --variant-head build that measures
# 0.9309 on the graded sample, 10/11 exact. Recorded from a real run of the
# real surgvu26-submission.sif with /input read-only, and independently
# reproduced by the 88-invocation scripts/flag_matrix.py sweep.
#
# Generation 1 (2026-08-15) was identical EXCEPT case126 '"No"' and case132
# '"Yes"'. Those two are not drift -- they are the entire measured gain
# (0.8766 -> 0.9309), the two tool-perception failures R24 traced to gold
# answers keying on instrument SIZE/FAMILY rather than raw presence, which
# the variant head resolves. This block was left pinned to generation 1 after
# the fix landed, so the CORRECT image validated as `clean: false, exit 1` on
# exactly the two cases that prove it works -- see
# baselines/image_validation_gen2.json, that very run's own output, whose
# "differs_from_verified": ["case126", "case132"] is the record of this harness
# calling the good image bad. Reading that as a regression and reverting
# the variant head would have thrown away the whole gain. Hence GAIN below.
EXPECTED = {
    "case122": '"No"', "case123": '"No"', "case124": '"Bipolar Forceps"',
    "case125": '"Yes"', "case126": '"Yes"', "case127": '"Uterine horn"',
    "case128": '"Yes"',
    "case129": '"Endoscopic surgery or a laparoscopic surgery"',
    "case130": '"To grasp and hold tissues or objects during the surgery."',
    "case131": '"No"', "case132": '"No"',
}
# case131 CHANGED FROM "Yes" TO "No" ON 2026-08-31, and it is the correction,
# not a regression. "Is the surgical procedure being performed an open
# surgery?" used to fall to `unknown_polar`, whose answer is the CONSTANT
# FALLBACK_POLAR = "Yes" -- asserting that robotic endoscopic footage was open
# surgery, while case129 in the SAME run called it endoscopic. It now routes
# to `approach_polar`. Together with case127 this took the prelim leaderboard
# from 0.8558 to 0.9128.

# The two cases whose values ARE the gain. Called out separately from the
# other nine because a diff here has a specific, expensive meaning -- the
# variant head or the YOLO detector went inert in this build (the answer gate
# needs BOTH: a DETECTED needle driver and a decided variant, so either one
# failing silently reverts both cases to generation 1) -- and because the
# pre-submission checklist is written in terms of these two by name.
GAIN = {"case126": '"Yes"', "case132": '"No"'}

# THE GPU DRAW HAS A DIFFERENT CORRECT ANSWER SET, AND WITHOUT THIS EVERY
# CORRECT GPU VALIDATION EXITS 1.
#
# EXPECTED above is the ROUTER-ONLY answer set. On a No-GPU draw that is what
# the image produces, because NF4 is CUDA-only and try_vlm_result declines. On
# a GPU draw the VLM runs and, under the shipped `challenger` mode, wins the
# arbitration on four cases -- measured 2026-08-26 (job 9705192) and scored at
# 0.8525 against the router's 0.9309.
#
# Those four differences are the SHIPPED CONFIGURATION WORKING. Comparing a GPU
# run against EXPECTED reports them as failures and exits 1, which means the
# expected outcome of a correct GPU validation is a red one -- and a red result
# that is always red tells you nothing when something genuinely breaks. That is
# the same defect as this file's stale EXPECTED had this morning, one draw over.
#
# So a GPU draw is compared against what the VLM actually produces. A diff HERE
# is real: it means the VLM changed its mind, which for a greedy-ish sampler on
# fixed frames should not happen without a cause worth knowing about.
EXPECTED_GPU = dict(EXPECTED)
EXPECTED_GPU.update({
    # ONE DIFFERENCE, NOT FOUR. The four-way override above described the
    # RETIRED `challenger` mode, which scored 0.8525 against the router's
    # 0.9309 and was abandoned for exactly that reason. The shipped mode is
    # `per_intent` with vlm_intents = ["tool_identity_open"], so the VLM may
    # win exactly one of these eleven.
    #
    # Measured on the fixed image under T4 simulation, job 9716683:
    # spoke=11, OOM=0, and every other case identical to the router.
    "case124": '"Bipolar forceps"',     # VLM override on its one armed
                                        # intent -- lowercase "f" is the
                                        # VLM's own casing, and it is the
                                        # cheapest single proof that the VLM
                                        # is alive. Both this and the
                                        # router's "Bipolar Forceps" are
                                        # wrong vs gold "Cadiere Forceps",
                                        # so the override is score-neutral
                                        # here; it is a LIVENESS signal.
})

# Which set applies is decided by the RUN, not guessed: the script printed
# "GPU assigned"/"no GPU assigned" above from _CONDOR_AssignedGPUs.
import os
# THE EXPECTED SET DEPENDS ON THE ARBITER MODE, NOT JUST THE DRAW. This keyed
# on the GPU draw alone and assumed a GPU implies the VLM overrides -- true
# under `challenger`, false under `fallback`, which is what ships since
# 2026-08-26. Validation 9707945 ran the bundle on a GPU in fallback mode,
# produced the correct ROUTER answers for all eleven cases, and exited 1
# against a table of challenger answers: a green run reported as a failure,
# on the four cases that prove fallback is doing its job.
# READ THE MODE FROM INSIDE THE IMAGE, via a file the wrapper extracted.
#
# This first read ./config/arbiter.json from the job's working directory --
# which validate_image.sub does not transfer (transfer_input_files = condor).
# The open failed, the except branch defaulted to "challenger", and the run
# compared a correctly-behaving `fallback` image against the CHALLENGER
# expectation table: exit 1 on four cases that were right (cluster 9709901).
#
# The repo's config would have been the wrong source anyway. The image carries
# its own copy, baked in at build time, and that is the one that decides what
# the grader sees -- they can differ whenever the repo moves ahead of a build,
# which is exactly the situation this project is in most of the time.
try:
    with open("image_arbiter_mode.txt", encoding="utf-8") as _fh:
        MODE = _fh.read().strip() or "challenger"
except Exception:
    MODE = "challenger"

# `challenger` and `primary` let the VLM override ANY intent the router covers,
# which is what EXPECTED_GPU's four differences encode. `per_intent` overrides
# only the intents named in `vlm_intents`, and `fallback` never overrides.
VLM_MAY_OVERRIDE = MODE in ("challenger", "primary")

# REPORT THE HARDWARE DRAW FROM THE HARDWARE, NOT FROM THE MODE.
#
# Until 2026-08-29 this line was
#     ON_GPU = bool(os.environ.get("_CONDOR_AssignedGPUs")) and VLM_MAY_OVERRIDE
# and the two ideas -- "was a GPU granted" and "does this mode let the VLM
# win" -- were collapsed into one flag that then drove the printed narrative.
# Under `per_intent` (added after this code was written, and not in the tuple
# above) that made validation 9714458 print "No-GPU draw ... the VLM cannot run
# without CUDA and declines" on a run where condor had granted GPU-a6385676,
# the image logged `vlm=31.2s` on all eleven cases, and the VLM actually won
# case124. Every answer recorded was still correct; the EXPLANATION printed
# beside them was false.
#
# That is the same defect this file already warns about 140 lines up, where a
# GPU-granted run reported a green GPU validation it could not structurally
# have performed -- a log that misreports which draw ran is worse than no log,
# because it is believed. So the draw is reported from _CONDOR_AssignedGPUs
# alone, and the choice of expectation table is a SEPARATE decision below.
GPU_ASSIGNED = bool(os.environ.get("_CONDOR_AssignedGPUs"))
ON_GPU = GPU_ASSIGNED and VLM_MAY_OVERRIDE
print("(arbiter mode %r; VLM may override any router intent: %s)"
      % (MODE, VLM_MAY_OVERRIDE))
print("(hardware draw: %s)"
      % ("GPU assigned -- the VLM can run" if GPU_ASSIGNED
         else "no GPU -- NF4 is CUDA-only, the VLM declines"))
if ON_GPU:
    EXPECTED = EXPECTED_GPU
    print("(comparing against the VLM-active answer set --")
    print(" four cases differ from the router-only set BY DESIGN under")
    print(" `challenger`/`primary`)")
elif MODE == "per_intent" and GPU_ASSIGNED:
    # The router-only table is still the right baseline: `per_intent` leaves
    # every UNARMED intent to the router, so those cases must match exactly.
    # The armed intents are the ones expected to differ, and that is the whole
    # signal this run exists to produce -- do not read such a diff as a fault
    # without first checking it against config/arbiter.json's `vlm_intents`.
    print("(comparing against the router-only answer set -- correct for")
    print(" `per_intent`, which leaves unarmed intents to the router.")
    print(" A diff is EXPECTED on exactly the armed `vlm_intents`; check any")
    print(" diff against that list before calling it a regression.)")
else:
    print("(comparing against the router-only answer set)")

diffs = [k for k in sorted(EXPECTED)
         if k in rows and rows[k].strip() != EXPECTED[k]]
missing = [k for k in sorted(EXPECTED) if k not in rows]
# On a GPU draw the VLM legitimately overrides case132, so the router-side
# GAIN check does not apply -- it would report the shipped behaviour as a loss
# every single run. What still matters there is that the DETECTOR is alive,
# which case126 shows: the VLM agrees with the variant head on that one, so a
# flip there means the gate really did go inert.
GAIN_APPLICABLE = {"case126": GAIN["case126"]} if ON_GPU else GAIN
lost_gain = [k for k in sorted(GAIN_APPLICABLE)
             if k in rows and rows[k].strip() != GAIN_APPLICABLE[k]]

print("\n== the shipped image vs the verified answers ==")
if missing:
    print("  MISSING from this run: %s" % missing)
for k in diffs:
    print("  %s\n     image:    %s\n     expected: %s" % (k, rows[k], EXPECTED[k]))
if not diffs and not missing:
    print("  all eleven identical. The container answers what the code answers.")

print("\n== the +0.0543 gain cases ==")
if lost_gain:
    print("  *** GAIN LOST on %s ***" % lost_gain)
    print("  TWO CAUSES ARE POSSIBLE AND THEY LOOK IDENTICAL HERE. Read the")
    print("  IMAGE STDERR section below before concluding anything:")
    print("    1. --yolo or --variant-head went INERT. The answer gate needs")
    print("       both (a DETECTED needle driver AND a decided variant), so")
    print("       either failing silently reverts these cases. Look for a")
    print("       swallowed WARNING and for missing 'yolo max_conf'/'variant")
    print("       family=' lines.")
    print("    2. The VLM OVERRODE the router. This is not a malfunction --")
    print("       it is arbiter mode 'challenger' doing exactly its job, with")
    print("       a confident VLM answer winning. Look for a 'VLM: N call(s)")
    print("       ... answer=' line on these cases. Measured 2026-08-26 on the")
    print("       GPU draw: the VLM flipped case132 No->Yes at confidence 1.00,")
    print("       undoing the variant head. If that is what happened, the fix")
    print("       is an arbiter-mode decision, NOT a detector investigation.")
else:
    held = ", ".join("%s=%s" % (k, GAIN_APPLICABLE[k]) for k in sorted(GAIN_APPLICABLE))
    print("  %s. The gain survives." % held)
    if ON_GPU:
        print("  (case132 is NOT checked on a GPU draw: the VLM legitimately")
        print("   overrides it under `challenger`. case126 is the live probe --")
        print("   the VLM agrees with the variant head there, so a flip means")
        print("   the detector/variant gate really did go inert.)")

json.dump({"answers": rows, "differs_from_verified": diffs,
           "missing": missing, "lost_gain": lost_gain,
           "clean": bool(rows) and not diffs and not missing},
          open("image_validation.json", "w"), indent=2)
raise SystemExit(1 if (diffs or missing or not rows) else 0)
PY
DIFF_RC=$?

# The comparison's exit code must reach the job, or a container that answers
# differently reports success. Same lesson as condor/motion_ab.sh.
[ "$FAIL" -gt 0 ] && DIFF_RC=1

# ---- WHAT THE VLM SAID, PER CASE --------------------------------------------
# The arbiter's whole design is that the VLM must EARN its say: per_intent
# arms a named list, and router_confidence_floor / vlm_confidence_ceiling
# gate the override. Deciding which intents to arm needs the VLM's answer
# next to the shipped answer, case by case. Printing it costs nothing and is
# the difference between measuring that decision and guessing it.
if [ -s image_vlm.tsv ]; then
    echo
    echo "== what the VLM said, per case =="
    printf '%-9s %-9s %-30s %s\n' case vlm_state "vlm answer" "shipped answer"
    while IFS=$'\t' read -r c st va; do
        fa=$(awk -F'\t' -v k="$c" '$1==k{print $2}' image_answers.tsv)
        printf '%-9s %-9s %-30s %s\n' "$c" "$st" "${va:0:30}" "${fa:0:40}"
    done < image_vlm.tsv
    echo
    echo "  spoke=$(grep -c 'spoke' image_vlm.tsv 2>/dev/null || echo 0)" \
         "OOM=$(grep -c 'OOM' image_vlm.tsv 2>/dev/null || echo 0)" \
         "absorbed=$(grep -c 'absorbed' image_vlm.tsv 2>/dev/null || echo 0)" \
         "no-gpu=$(grep -c 'no-gpu' image_vlm.tsv 2>/dev/null || echo 0)"
fi

# ---- surface the container's own stderr -------------------------------------
# The per-case stderr is appended to image_validation.err above. Without this
# dump it is never transferred back, so a component that failed INSIDE the image
# -- and was swallowed by scripts/inference.py's best-effort wrappers, which log
# a WARNING and continue -- leaves NO trace in the condor .out. That is exactly
# how an enabled-but-inert --yolo/--variant-head looks: eleven valid answers,
# exit 0, "passed 11, failed 0", and no evidence at all.
# This block must sit BEFORE the exit; appended after it, it is dead code.
echo
echo "== IMAGE STDERR (WARNING lines say why an evidence block is absent) =="
if [ -s image_validation.err ]; then
    # THE VLM GETS ITS OWN SECTION, FIRST AND UNTRUNCATED.
    #
    # The general dump below greps WARNING|Traceback|Error|error|yolo|variant|
    # motion and tails 60 lines. `scripts/inference.py`'s two VLM lines --
    # "VLM enabled: ..." from build_vlm and "VLM: no CUDA device available..."
    # from try_vlm_result -- match NONE of those words, so they were being
    # filtered out entirely. The 2026-08-26 CPU-draw validation of the first
    # VLM-carrying image therefore showed no VLM lines at all, which reads as
    # "the VLM never ran" and is indistinguishable from it. Absence was an
    # artifact of this filter, not a fact about the image.
    #
    # That distinction is the whole point of the GPU draw: the flag is only
    # proven live by SEEING it load. So these lines are pulled out separately,
    # before the general dump, where nothing can tail them off the end.
    echo "-- VLM lines (build_vlm / try_vlm_result / arbiter) --"
    if grep -aE "VLM|vlm|arbiter" image_validation.err >/dev/null 2>&1; then
        grep -aE "VLM|vlm|arbiter" image_validation.err | tail -40
    else
        echo "   NONE. On a No-GPU draw expect \"VLM: no CUDA device"
        echo "   available\"; on a GPU draw expect \"VLM enabled: ...\" AND a"
        echo "   per-case \"VLM: N call(s) ...\". Neither appearing on a GPU"
        echo "   draw means --vlm is INERT in this image -- do not ship it."
    fi
    echo "-- general --"
    # gpu=/vram lines get their own line, for the same reason the VLM ones do:
    # they match none of the words below, so the general dump silently hides
    # exactly the hardware readings this project keeps having to guess at.
    echo "-- hardware --"
    grep -aE "gpu=|vram peak|frame plan" image_validation.err | sort -u | head -8
    echo "-- general --"
    grep -aE "WARNING|Traceback|Error|error|yolo|variant|motion" image_validation.err | tail -60
    echo "-- full stderr: $(wc -l < image_validation.err) lines --"
else
    echo "(empty -- the container wrote nothing to stderr)"
fi

# ---- THE SILENT-VLM GATE ----------------------------------------------------
# THIS IS THE CHECK THAT WOULD HAVE CAUGHT v6, AND ITS ABSENCE COST A
# SUBMISSION.
#
# v6 was validated green -- 11/11, exit 0 -- and then scored 0.8558 on the
# leaderboard, byte-identical to v5.2. Grand Challenge's own per-case logs
# showed why: the int8 checkpoint took 12.5 GiB of the T4's 14.56, generation
# needed 3.1 GiB more, and EVERY ONE of the eleven cases died with
# `torch.OutOfMemoryError` inside the vision tower's attention. R18 absorbed
# each one and kept the router's answer, exactly as designed.
#
# That design is right -- a crash that writes no response scores 0 on every
# question, far worse than a wrong answer. But it means "the VLM crashed",
# "the VLM never loaded" and "the VLM agreed with the router" are INDIST-
# INGUISHABLE from the answers alone. A validation that only compares answers
# therefore cannot see a dead VLM. Ours could not, for three builds.
#
# So: if a GPU was granted AND a sidecar was bound, the VLM is expected to
# actually run. Not to WIN a case -- `per_intent` may legitimately leave every
# answer to the router -- but to get far enough to produce a candidate. If it
# never did, that is a failure no matter how the answers came out.
if [ -n "$SIDECAR_DIR" ] && [ -n "${_CONDOR_AssignedGPUs:-}" ] \
   && [ -s image_validation.err ]; then
    echo
    echo "== SILENT-VLM GATE =="
    VLM_SPOKE=$(grep -acE "VLM: [0-9]+ call\(s\) agreed=" image_validation.err || true)
    VLM_ABSORBED=$(grep -acF "the VLM seam raised" image_validation.err || true)
    VLM_OOM=$(grep -acF "OutOfMemoryError" image_validation.err || true)
    echo "  cases where the VLM produced a candidate: $VLM_SPOKE"
    echo "  cases where the seam absorbed an exception: $VLM_ABSORBED"
    echo "  CUDA out-of-memory events: $VLM_OOM"

    if [ "$VLM_OOM" -gt 0 ]; then
        echo
        echo "FATAL: the VLM ran out of GPU memory. This is the v6 failure"
        echo "  exactly: the answers may all be 'correct' because the router"
        echo "  produced every one of them, and the model you shipped never"
        echo "  spoke. Peak usage is in the 'hardware' section above."
        grep -aF "OutOfMemoryError" image_validation.err | head -1
        DIFF_RC=1
    elif [ "$VLM_SPOKE" -eq 0 ]; then
        echo
        echo "FATAL: a GPU was granted and a sidecar was bound, but the VLM"
        echo "  never produced a candidate on any case. The answers below are"
        echo "  the ROUTER's alone. Something upstream of the arbiter is"
        echo "  failing silently -- check the seam-absorbed count above."
        DIFF_RC=1
    elif [ "$VLM_ABSORBED" -gt 0 ]; then
        echo
        echo "WARNING: the VLM spoke on $VLM_SPOKE case(s) but the seam also"
        echo "  absorbed $VLM_ABSORBED exception(s). Partial failure is still"
        echo "  failure on the cases it hit; read the traceback above."
    else
        echo "  OK: the VLM ran on every case it was asked to."
    fi
fi

echo "exit: $DIFF_RC  end: $(date)"
exit "$DIFF_RC"
