"""R22/R34/R35/R37 EXPERIMENT: can pandas/matplotlib/seaborn be
`sys.modules`-stubbed instead of installed, so `--yolo` can run inside
`containers/surgvu26-submission.def` with no rebuild?

THE DECISION THIS INFORMS. `containers/surgvu26-submission.def` installs
only `opencv-python-headless` and `PyYAML` on top of the torch base -- Grand
Challenge has no internet, so a runtime pip install there is impossible.
`yolov5/models/common.py` (the staged checkout at
`/staging/groups/bhaskar_opscribe/surgvu_yolo_detector/yolov5`) and its own
import chain (`utils/dataloaders.py`, `utils/general.py`, `utils/plots.py`)
hard-import `pandas`, `requests`, `PIL`, `tqdm`, `matplotlib` and `seaborn`
at MODULE SCOPE, and `DetectMultiBackend` -- the class
`surgvu.detect.Detector._load` loads through -- lives in
`models/common.py`. None of the six is actually CALLED on the
`.pt`-checkpoint detection path (see the per-name trace below), so a
`sys.modules` stub, rather than a container rebuild, might be enough to let
`--yolo` run.

FOUR RULINGS, EACH CORRECTING THE LAST -- READ IN ORDER, because each one
changed what this file actually does, not just what it says:

  R22 (three names): pandas, requests, PIL are imported at module scope in
  `models/common.py` and none is called on the `.pt` detection path. Traced
  and confirmed correct, line numbers included -- see "THE PER-NAME CALL
  TRACE" below.

  R34 (six names, corrected R22's SCOPE): the rest of `models/common.py`'s
  import chain also hard-imports tqdm, matplotlib and a second pandas site,
  equally module-scope, equally required just for the `import` statement to
  succeed. The version of this file built for R34 stubbed all six, and its
  own up-front guard assertion -- "fail loudly if any of the six is
  genuinely importable here" -- did exactly its job: the job failed on
  that assertion with

      AssertionError: ['requests', 'PIL', 'tqdm'] are genuinely importable
      in this environment

  which is what R35 is about.

  R35 (three names again, but a DIFFERENT three, corrected which SIX were
  actually missing): all three container definitions
  (`containers/surgvu26-train.def`, `containers/surgvu26-submission.def`,
  `containers/Dockerfile`) share one base --
  `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` -- and that base image
  already ships `requests`, `PIL` and `tqdm`. The submission `.def` adds
  only `opencv-python-headless` and `PyYAML` on top. So the submission
  image's genuinely MISSING set, as inferred from the shared base, is
  `pandas`, `matplotlib`, `seaborn` -- three, and a different three than
  either R22 or R34 named. THIS FILE NOW STUBS EXACTLY THOSE THREE, and lets
  `requests`, `PIL` and `tqdm` be the real, installed packages -- stubbing
  them would misrepresent the container this experiment is standing in for.

  What this DOES verify: whether `pandas`/`matplotlib`/`seaborn` can be
  bare-stubbed without touching anything `Detector.detect` actually needs,
  in an environment shaped like "pytorch/pytorch base + real
  requests/PIL/tqdm + real opencv-python-headless/PyYAML + no pandas/
  matplotlib/seaborn". What this does NOT verify: that the built
  `surgvu26-submission.sif` actually has exactly that package set. The
  inference chain is "all three `.def`/Dockerfile files declare the same
  `FROM`/`From:` base, and the submission `.def` adds nothing beyond
  opencv-python-headless and PyYAML, therefore the submission image's
  package set equals the base's plus those two." If the built `.sif` was
  ever produced from a different base tag, or had packages added or removed
  after the fact outside what `containers/surgvu26-submission.def` shows,
  that inference breaks and this experiment's environment would silently
  stop matching the real one. This test's own two-sided guard (STEP 1)
  catches the "requests/PIL/tqdm are actually missing" half of that failure
  mode for THIS job's environment (`surgvu26-train.sif`, which shares the
  same base); it cannot catch a divergence between `surgvu26-train.sif` and
  the actual built `surgvu26-submission.sif` that this job never touches.

  R37 (a THIRD obstacle, not a rescope -- torch's own import machinery,
  not yolov5's): the R35-scoped job actually ran (cluster 9686702) and got
  past Step 1 and past `models.common`'s own import chain, then failed
  inside TORCH:

      torch/_dynamo/trace_rules.py:3359   module_spec = find_spec(import_name)
      ValueError: pandas.__spec__ is None

  `torch._dynamo.trace_rules` enumerates module specs at IMPORT time (not
  only when `torch.compile` is actually invoked), via
  `importlib.util.find_spec(import_name)`. For a name already in
  `sys.modules`, that function returns `sys.modules[name].__spec__` and
  RAISES `ValueError` when that is `None` -- and a bare `types.ModuleType`
  genuinely HAS a `__spec__` attribute defaulting to `None` (confirmed
  torch-free: `types.ModuleType("x").__spec__ is None`; it is not simply
  absent, which is why this surfaced as a `ValueError` from inside
  `find_spec`, not an `AttributeError` from our own stub). The fix is a
  real `importlib.machinery.ModuleSpec(name, loader=None)` on every stub --
  see `_TrackedStubModule.__init__` below, which applies it uniformly to
  every top-level stub AND every submodule stub (`matplotlib.pyplot`),
  since all of them are instances of that one class. `__file__`, `__path__`
  and `__version__` were each considered and deliberately NOT added -- see
  "ATTRIBUTES CONSIDERED AND RULED OUT" below for why, per name.

THE PER-NAME CALL TRACE (unchanged since R22 -- re-verified against the
same staged checkout, not re-derived):

  * `pandas`  -- read/used only at models/common.py:738, inside
    `Detections.pandas()` (the `AutoShape` results wrapper's `.pandas()`
    accessor). `Detector.detect` never constructs a `Detections`. A SECOND
    import site is `utils/general.py:31`, and a THIRD is `utils/plots.py:16`
    -- both are plain `import pandas as pd` for the same reason.
  * `matplotlib` -- `utils/plots.py:13-14` (`import matplotlib` /
    `import matplotlib.pyplot as plt`) and `utils/metrics.py:10`
    (`import matplotlib.pyplot as plt` again, cached by the time it runs).
    Every `plt.`-prefixed call in both files (plotting/saving figures) is
    inside a function body, never at module or class scope, EXCEPT two
    calls described below.
  * `seaborn` -- `utils/plots.py:17` (`import seaborn as sn`). Every
    `sn.`-prefixed call found (histplot/pairplot for label-distribution
    plots) is inside a function body, never at module or class scope.

None of the three is on `Detector.detect`'s CALL path -- but, per the AST
scan below, two of them ARE touched once at IMPORT time regardless of
whether anything calls them later.

WHAT AN AST SCAN (NOT A TEXT GREP) FOUND AT MODULE SCOPE. A text grep for
"pd.", "matplotlib." or "sn." matched only `import` lines the first time
this experiment was built, because indentation-based pattern matching does not
reliably distinguish "runs at import time" (module body or class body) from
"only runs if called" (inside a function/method body). A proper AST walk --
flagging any `Name` node referencing one of the stubbed names that sits
outside a `FunctionDef`/`AsyncFunctionDef`/`Lambda` body -- found two real
module-scope CALLS, still true for both of the two names still stubbed
here:

  * `utils/general.py:53`   `pd.options.display.max_columns = 10`
  * `utils/plots.py:27`     `matplotlib.rc('font', **{'size': 11})`
  * `utils/plots.py:28`     `matplotlib.use('Agg')`

(A fourth call this same scan found, `for orientation in
ExifTags.TAGS.keys():` at `utils/dataloaders.py:45`, needed a stub under
R34's six-name scope; under R35 `PIL` is the REAL package, so that call
site resolves against genuine `PIL.ExifTags.TAGS` and needs no
accommodation here.)

A bare `types.ModuleType("pandas")` with no `.options` attribute, or a bare
`matplotlib` with no callable `.rc`/`.use`, would `AttributeError` before
`DetectMultiBackend` is even defined -- for reasons that have nothing to do
with whether pandas/matplotlib are called on the detection path, only that
the stubs would be too bare to survive import. The stubs below are built to
tolerate exactly these three call sites and no more: this is still "insert
placeholders", not "reimplement pandas/matplotlib", and it is still built
entirely in this test file, not under `src/`.

ATTRIBUTES CONSIDERED AND RULED OUT (ruling R37 asked this explicitly: set
what is genuinely needed, not a broad guess, and name the import path that
demanded it). Three more module attributes were considered for every stub
besides `__spec__`/`__loader__`, and none was added:

  * `__file__` -- NOT added. No call site in the reachable-at-import-time
    chain was found that reads `pandas.__file__`/`matplotlib.__file__`/
    `seaborn.__file__` (checked the same way as the AST scan above -- no
    hit). `find_spec` itself does not need it: for an already-imported
    module it only reads `.__spec__`, never `.__file__` directly. Setting
    a FAKE path would be worse than leaving it absent -- a fake,
    nonexistent path handed to something that later tries to open or stat
    it produces a confusing, hard-to-attribute failure ("file not found:
    <made-up path>"), whereas leaving `__file__` genuinely unset produces a
    clean `AttributeError: module 'pandas' has no attribute '__file__'`
    that names exactly what is missing, if anything ever does turn out to
    need it. That would be actionable evidence for a genuine round 4; a
    silently-wrong fake path would not be.
  * `__path__` -- NOT added, including on the `matplotlib` stub even though
    it is package-like (it carries the `matplotlib.pyplot` submodule). Same
    reasoning: no call site was found reading it, and Python's import
    system itself never needs it here because every submodule
    (`matplotlib.pyplot`) is inserted into `sys.modules` directly by this
    test, not discovered by walking a parent's `__path__` the way a real
    package's finder would.
  * `__version__` -- NOT added. Grepped the whole reachable-at-import-time
    tree (`models/`, `utils/`) for `.__version__` reads: the only hits are
    `trt.__version__` (TensorRT, the `elif engine:` branch, unreachable for
    a `.pt` checkpoint), `A.__version__` (Albumentations, only inside
    augmentation classes `Detector.detect` never constructs), and
    `wandb.__version__` (`utils/loggers/__init__.py`, never imported by
    this chain at all) -- none references pandas, matplotlib or seaborn.

The general principle applied throughout this file (not new to R37, but
sharpest here): a stub should be exactly as complete as something on the
reachable import path is shown to demand, confirmed by reading the actual
source or by a mechanics dry run -- never rounded up "to be safe" against
an import path nobody has pointed to. Rounding up is how a stub quietly
turns into a reimplementation.

THIS TEST'S OWN INSTRUMENTATION. Each of the three stub modules tracks
which of its attributes get read at all (`_TrackedStubModule.__getattribute__`),
and the two no-op callables (`matplotlib.rc`, `matplotlib.use`) count how
many times they are actually invoked. The test prints a per-module report
at the end: "imported but never touched" vs. "an attribute was read" vs.
"actually called N time(s)" are three different, distinguishable outcomes,
and only the report at the bottom of a passing run says which one applies
to each of the three -- not this docstring, which is a prediction, not a
result. With only three stubs (down from six under R34), this report is
now a complete, legible account of everything the real detection path asked
of everything this test faked -- the strongest form of evidence this
experiment can produce for the decision.

THIS TEST IS THE EMPIRICAL HALF OF THE ARGUMENT. `tests/test_detect_weights.py`
+ `condor/detect_smoke.sub` already prove the detector works when
pandas/requests/tqdm/matplotlib/seaborn/Pillow are genuinely, really
installed (`condor/detect_smoke.sh` pip-installs all six). This test proves
-- or disproves -- that it *still* works when pandas/matplotlib/seaborn are
NOT installed and are stood in for by stub modules inserted into
`sys.modules`, from this test file alone, while requests/PIL/tqdm stay real.
`condor/detect_stub.sh`, the executable behind `condor/detect_stub.sub`
(the only job that actually runs this test's body -- see the skip logic
below), installs nothing but pytest: the same footprint as
`condor/pytest.sh`. It does NOT install pandas/matplotlib/seaborn (the
three genuinely missing from the submission image, per R35) and relies on
`surgvu26-train.sif`'s shared base already providing requests/PIL/tqdm.

THE STUBBING ITSELF LIVES ONLY HERE, NOT UNDER src/. Nothing in
`src/surgvu/detect.py` (or anywhere else under `src/`) inserts stub modules,
imports a stand-in for pandas/matplotlib/seaborn, or otherwise knows this
experiment exists. This test is deciding WHETHER to adopt stubbing as a
strategy; building it into the shipped module would pre-empt that decision.
Nothing here (or in `condor/detect_stub.sh`) pip-installs pandas,
matplotlib or seaborn either -- if this test can only be made to pass by
installing one of them for real, that is this experiment's answer (stubbing
is not viable; the fallback is a container rebuild or an export-based
approach), and it must be reported as that, not routed around.

SKIPPED, NOT FAILED, when the weights or the yolov5 checkout are absent --
same convention as `tests/test_detect_weights.py`, so this suite stays green
on a machine (a laptop, this login node, CI without a staging mount) that has
neither, with no network access required to decide that.
"""
import importlib.machinery
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Same corpus paths as tests/test_detect_weights.py's skipif re-check.
_DETECTOR_ROOT = Path(
    "/staging/groups/bhaskar_opscribe/surgvu_yolo_detector")
WEIGHTS = _DETECTOR_ROOT / "best.pt"
YOLOV5_DIR = _DETECTOR_ROOT / "yolov5"

# The controller-named surgvu24 corpus clip: 60 fps, 1280x720, matching the
# geometry `prepare_frame`/`letterbox` were written for -- not synthetic
# noise, so a detector finding nothing would actually mean something.
_VIDEO = Path(
    "/staging/groups/bhaskar_opscribe/surgvu/videos/surgvu24/case_000/"
    "case_000_video_part_001.mp4")


def _staging_available():
    return WEIGHTS.exists() and YOLOV5_DIR.exists() and _VIDEO.exists()


_SKIP_REASON = (
    "weights (%s), yolov5 checkout (%s) or the sample video not present -- "
    "this test only runs where /staging/groups/bhaskar_opscribe is mounted "
    "(condor/detect_stub.sub); it is skipped rather than failed everywhere "
    "else so the suite stays green off-staging." % (WEIGHTS, YOLOV5_DIR))

# Ruling R35: the three names genuinely MISSING from the submission image,
# as inferred from the shared pytorch/pytorch base plus
# containers/surgvu26-submission.def's own additions (opencv-python-headless,
# PyYAML -- neither pandas, matplotlib nor seaborn). Do not silently widen
# or narrow this without updating the docstring above -- the R22 -> R34 ->
# R35 history is exactly the kind of thing a future reader needs to see,
# not rediscover.
_STUB_TOP_LEVEL = ("pandas", "matplotlib", "seaborn")

# Ruling R35: the three names the shared base image (pytorch/pytorch:
# 2.5.1-cuda12.1-cudnn9-runtime) already ships for real. STEP 1's guard
# fails loudly if any of these is actually MISSING here -- not just if a
# stubbed name is actually present -- because a missing one would mean this
# job's environment no longer matches the submission image this experiment
# is standing in for, in the other direction.
_REQUIRED_REAL = ("requests", "PIL", "tqdm")


class _TrackedStubModule(types.ModuleType):
    """A `sys.modules` stand-in that records every attribute NAME accessed
    on it, so this test can report exactly what the reachable-at-import-time
    yolov5 code asked of each stub -- not just whether the bare `import`
    statement succeeded. Dunder/internal names (leading `_`) are not
    recorded: those are import-machinery bookkeeping (`__name__`,
    `__spec__`, `__loader__`, `__path__`, ...), not evidence of the
    detection code path touching the stub.
    """

    def __init__(self, name):
        super().__init__(name)
        object.__setattr__(self, "_touched", [])
        # Ruling R37: torch._dynamo.trace_rules enumerates module specs at
        # IMPORT time via `importlib.util.find_spec(name)`, which for an
        # already-imported module returns `sys.modules[name].__spec__` and
        # RAISES `ValueError` when that is `None` -- not absent, `None`: a
        # bare `types.ModuleType` genuinely has a `__spec__` attribute that
        # defaults to `None` (confirmed torch-free:
        # `types.ModuleType("x").__spec__ is None`), so `find_spec` on any
        # of these stubs raised before this fix existed. Observed directly
        # in cluster 9686702: `torch/_dynamo/trace_rules.py:3359` ->
        # `ValueError: pandas.__spec__ is None`. A real `ModuleSpec` with
        # `loader=None` is enough to satisfy `find_spec` itself (re-verified
        # torch-free after adding this: a `sys.modules` entry with exactly
        # this spec resolves through `importlib.util.find_spec` cleanly) --
        # it does not claim the stub has a real loader or a file on disk, it
        # only stops `__spec__` from being `None`. `__loader__` already
        # defaults to `None` on a bare `ModuleType` (confirmed the same way)
        # -- set explicitly anyway, matching the fix as specified, so it is
        # not left to an implicit default a future reader has to re-derive.
        self.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        self.__loader__ = None

    def __getattribute__(self, attr_name):
        if not attr_name.startswith("_"):
            object.__getattribute__(self, "_touched").append(attr_name)
        return object.__getattribute__(self, attr_name)

    def _touched_names(self):
        """De-duplicated, first-seen-order list of attribute names read."""
        seen = []
        for name in object.__getattribute__(self, "_touched"):
            if name not in seen:
                seen.append(name)
        return seen


class _NoOpCallable:
    """A callable stub: does nothing, counts how many times it was called,
    and returns its first positional argument unchanged (or an empty
    iterator if there is none) rather than `None` or a bare placeholder --
    so a caller that treats this as an iteration wrapper would still get
    its input back rather than silence. (No name stubbed under R35 is
    actually used this way -- `matplotlib.rc`/`.use` are fire-and-forget --
    but the behaviour costs nothing to keep and matches the convention
    established for tqdm under the R34 version of this file.)
    """

    def __init__(self, label):
        self.label = label
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return args[0] if args else iter(())


@pytest.mark.slow  # excluded from condor/pytest.sub's "-m 'not slow'" run,
# same reasoning as tests/test_detect_weights.py: that job's container also
# has +WantStagingMount (weights/checkout/video all resolve there too) but
# condor/pytest.sh installs nothing but pytest -- no stubbing happens there
# either -- so without this marker the main suite would hit exactly the
# ModuleNotFoundErrors this file's stubs exist to prevent, misattributed to
# a thousand-test run instead of showing up as its own red job.
@pytest.mark.skipif(not _staging_available(), reason=_SKIP_REASON)
def test_detector_runs_with_pandas_matplotlib_seaborn_stubbed():
    # ================================================================
    # STEP 1 -- THE LOAD-BEARING ASSERTION, NOW TWO-SIDED. Must run BEFORE
    # anything below touches sys.modules. Both directions can invalidate
    # this run:
    #
    #   (a) if pandas/matplotlib/seaborn is ALREADY importable here, the
    #       stub is never exercised and every assertion below could be
    #       passing against the real package instead -- this is exactly
    #       what happened under R34's six-name version, which is why R35
    #       exists: the guard fired on ['requests', 'PIL', 'tqdm'] being
    #       unexpectedly present, which is what revealed the base image
    #       already ships them.
    #
    #   (b) if requests/PIL/tqdm is NOT importable here, this job's
    #       environment no longer matches the submission image this
    #       experiment is standing in for (R35's whole premise is that
    #       those three are real, shipped by the shared base) -- a pass
    #       under a decayed environment would not transfer to the real
    #       submission container.
    #
    # Fail loudly, by name, either way, rather than let a false pass or a
    # misleading pass through.
    # ================================================================
    def _really_importable(name):
        try:
            __import__(name)
        except ImportError:
            return False
        return True

    should_be_stubbed_but_present = [
        n for n in _STUB_TOP_LEVEL if _really_importable(n)]
    assert not should_be_stubbed_but_present, (
        "%r %s genuinely importable in this environment "
        "(condor/detect_stub.sh installs only pytest -- see its docstring, "
        "and neither this test nor condor/detect_stub.sh installs "
        "pandas/matplotlib/seaborn). This run proves NOTHING about "
        "stubbing: any pass below could be because the stub is safe, or "
        "simply because the real package was there all along and the stub "
        "was never exercised. Investigate the base image "
        "(containers/surgvu26-train.def / "
        "containers/surgvu26-submission.def / containers/Dockerfile) "
        "directly instead of trusting this job's exit code." % (
            should_be_stubbed_but_present,
            "is" if len(should_be_stubbed_but_present) == 1 else "are"))

    should_be_real_but_missing = [
        n for n in _REQUIRED_REAL if not _really_importable(n)]
    assert not should_be_real_but_missing, (
        "%r %s NOT importable in this environment, but ruling R35's whole "
        "premise is that requests/PIL/tqdm are genuinely present here "
        "(shipped by the pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime "
        "base all three container definitions share). This job's "
        "environment (surgvu26-train.sif) no longer matches the submission "
        "image this experiment is meant to stand in for, so a pass below "
        "would not transfer to it -- fix the environment (or re-derive "
        "which names are genuinely missing from the actual base image) "
        "before trusting this job's exit code." % (
            should_be_real_but_missing,
            "is" if len(should_be_real_but_missing) == 1 else "are"))

    # ================================================================
    # STEP 2 -- insert exactly the three stubs R35 scoped this to. Bare
    # ModuleType-derived objects, except for the two call sites the
    # docstring's AST scan actually found at MODULE SCOPE
    # (pandas.options.display, matplotlib.rc/use): those need enough real
    # behaviour to survive being touched once, at import time, by code this
    # test never calls directly. Nothing here reimplements pandas or
    # matplotlib -- every value below is either a namespace that tolerates
    # arbitrary attribute assignment, or a no-op. requests/PIL/tqdm are
    # deliberately left alone: they resolve against the real, installed
    # packages, exactly as they would in the actual submission container.
    # ================================================================
    def _install_stub(name):
        # _TrackedStubModule.__init__ gives every stub -- including
        # matplotlib.pyplot below -- a real __spec__/__loader__ (ruling
        # R37) automatically; nothing further is needed here for that.
        module = _TrackedStubModule(name)
        sys.modules[name] = module
        return module

    stubs = {}

    stubs["pandas"] = _install_stub("pandas")
    # utils/general.py:53 -- `pd.options.display.max_columns = 10` runs at
    # IMPORT time (module scope, confirmed by AST scan, not inside a
    # function): a bare stub with no `.options` attribute would
    # AttributeError here, long before DetectMultiBackend is even defined.
    # types.SimpleNamespace tolerates the assignment for free -- it is not
    # pandas.options, it is just an object with a settable `.display`.
    stubs["pandas"].options = types.SimpleNamespace(
        display=types.SimpleNamespace())

    matplotlib_stub = _install_stub("matplotlib")
    # utils/plots.py:27-28 -- `matplotlib.rc(...)` and `matplotlib.use(...)`
    # both run at IMPORT time (module scope, confirmed by AST scan). A bare
    # stub has no callable `.rc`/`.use` at all; these no-ops let the two
    # statements execute and do nothing, which is exactly what "not on the
    # detection call path" should mean in practice.
    matplotlib_rc = _NoOpCallable("matplotlib.rc")
    matplotlib_use = _NoOpCallable("matplotlib.use")
    matplotlib_stub.rc = matplotlib_rc
    matplotlib_stub.use = matplotlib_use
    pyplot_stub = _install_stub("matplotlib.pyplot")
    matplotlib_stub.pyplot = pyplot_stub
    stubs["matplotlib"] = matplotlib_stub

    stubs["seaborn"] = _install_stub("seaborn")

    # ================================================================
    # STEP 3/4 -- load the real detector against the real weights, and run
    # it on a real decoded frame. Imported here, not at module scope: torch
    # and surgvu.perceive must never be required to even COLLECT this file
    # on a machine without torch -- the skipif above has already decided
    # whether this body runs before any of these names are touched.
    # ================================================================
    from surgvu.detect import YOLO_CLASSES, Detector
    from surgvu.perceive import decode_clip

    # Defaults (16 frames, 512x512): the exact preprocessing
    # (`prepare_frame`'s crop-margins + blur-UI-band + square resize) the
    # serving path actually uses, not a test-only shape.
    frames = decode_clip(_VIDEO)

    detector = Detector(WEIGHTS, YOLOV5_DIR)
    # Nothing has touched any stub yet: Detector.__init__ only stores
    # config (see src/surgvu/detect.py), it does not import yolov5. Every
    # `_touched` list and `_NoOpCallable.call_count` below is still at its
    # starting value, so whatever changes during `.detect()` is exactly,
    # and only, what the real detection path (not merely module import
    # elsewhere) asked of each stub.
    result = detector.detect(frames)

    # ================================================================
    # STEP 5 -- the SAME properties tests/test_detect_weights.py asserts.
    # A stub that silently corrupted the geometry path (wrong class table,
    # wrong letterbox math) would still return well-typed data; only these
    # checks catch that. `scale_coords` itself lives in utils/general.py --
    # one of the two modules whose import now actually touches a stub
    # (pandas.options) rather than merely importing an unreached name -- so
    # this assertion is the strongest evidence available that the stub did
    # not perturb the geometry path.
    # ================================================================
    assert isinstance(result, list)
    assert len(result) == len(frames)

    height, width = frames.shape[1], frames.shape[2]
    total_detections = 0
    found_any = False
    for per_frame in result:
        assert isinstance(per_frame, list)
        for item in per_frame:
            found_any = True
            total_detections += 1
            assert isinstance(item, dict)
            assert set(item) == {"cls", "conf", "box"}

            # A name outside the 14-class contract means the weights and
            # detect.py's YOLO_CLASSES table disagree -- or that a stub
            # somehow reached a code path that changed what got decoded.
            assert item["cls"] in YOLO_CLASSES

            conf = item["conf"]
            assert 0 < conf <= 1, "conf %r out of (0, 1] for %r" % (
                conf, item["cls"])

            # THE IMPORTANT ONE (see test_detect_weights.py's docstring):
            # the empirical check on the letterbox/scale_coords geometry.
            box = item["box"]
            assert len(box) == 4
            x1, y1, x2, y2 = box
            assert 0 <= x1 < x2 <= width, (
                "box %r has x1/x2 outside [0, %d]" % (box, width))
            assert 0 <= y1 < y2 <= height, (
                "box %r has y1/y2 outside [0, %d]" % (box, height))

    # ================================================================
    # STEP 6 -- report, per stub, whether it was merely imported or
    # actually touched, so a silent no-op (stub imports "succeed" but the
    # forward pass quietly returns nothing) is visible either way, and so
    # the decision this experiment informs gets more than a pass/fail bit.
    # With only three stubs, this is a complete account, not a sample.
    # ================================================================
    print(
        "detect_stub: %d detection(s) across %d frame(s) from %s "
        "(conf threshold %.2f)" % (
            total_detections, len(frames), _VIDEO, detector.conf))

    print("detect_stub: per-stub touch report "
          "(imported vs. attribute-accessed vs. called) --")
    for name in _STUB_TOP_LEVEL:
        module = stubs[name]
        touched = module._touched_names()
        if touched:
            print("  %-10s ATTRIBUTE-ACCESSED: %s" % (name, touched))
        else:
            print("  %-10s imported only, no attribute ever read" % (name,))
    pyplot_touched = sys.modules["matplotlib.pyplot"]._touched_names()
    if pyplot_touched:
        print("  matplotlib.pyplot ATTRIBUTE-ACCESSED: %s" % (
            pyplot_touched,))
    else:
        print("  matplotlib.pyplot imported only, no attribute ever read")
    print("  matplotlib.rc(...) called %d time(s)" % (
        matplotlib_rc.call_count,))
    print("  matplotlib.use(...) called %d time(s)" % (
        matplotlib_use.call_count,))
    print("detect_stub: requests/PIL/tqdm were left REAL (not stubbed, not "
          "instrumented) per ruling R35 -- they are the genuine, installed "
          "packages, exactly as they would be in the actual submission "
          "container.")

    # Same reasoning as tests/test_detect_weights.py: a real, tool-bearing
    # 30s surgical clip producing zero detections across all 16 sampled
    # anchors is far more likely to mean a stub silently broke something
    # (or the checkpoint/class table/preprocessing disagree) than that this
    # clip truly shows no tool. Fail loudly rather than let that look like
    # a passing experiment.
    assert found_any, (
        "Detector found zero detections across all %d frames sampled from "
        "%s at conf=%.2f with pandas/matplotlib/seaborn stubbed. This does "
        "not necessarily mean the stubs are unsafe, but it means this run "
        "gives no positive evidence for the stubbing decision either -- "
        "investigate before trusting it." % (
            len(frames), _VIDEO, detector.conf))
