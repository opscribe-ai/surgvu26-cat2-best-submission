import cv2
import numpy as np
from surgvu.preprocess import (
    UI_BAND_FRACTION, detect_side_margins, crop_side_margins, blur_ui_band,
    prepare_frame,
)


def _frame_with_margins(width=1280, height=720, margin=192):
    f = np.zeros((height, width, 3), dtype=np.uint8)
    f[:, margin:width - margin, :] = 200          # bright content
    return f


def test_detect_side_margins_finds_black_bars():
    left, right = detect_side_margins(_frame_with_margins())
    assert left == 192
    assert right == 192


def test_detect_side_margins_returns_zero_when_already_cropped():
    f = np.full((512, 640, 3), 200, dtype=np.uint8)
    assert detect_side_margins(f) == (0, 0)


def test_detect_side_margins_ignores_single_bright_edge_column():
    # A specular highlight, vignette artifact, or compression edge can leave a
    # single bright column right at the frame border. A single bright pixel
    # must not be mistaken for the start of real content — that requires a
    # run of consecutive bright columns. Against the old argmax-based
    # implementation this fails: bright[0] and bright[-1] are True, so
    # argmax(bright) == 0 and argmax(bright[::-1]) == 0, reporting (0, 0)
    # instead of the true (192, 192) margin.
    f = _frame_with_margins()
    f[:, 0, :] = 200
    f[:, -1, :] = 200
    left, right = detect_side_margins(f)
    assert left == 192
    assert right == 192


def test_detect_side_margins_all_black_frame_returns_zero():
    f = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert detect_side_margins(f) == (0, 0)

    cropped = crop_side_margins(f)
    assert cropped.shape == f.shape
    assert cropped.shape[1] > 0


def test_detect_side_margins_asymmetric_margin_right_only():
    width, height, margin = 1280, 720, 192
    f = np.zeros((height, width, 3), dtype=np.uint8)
    f[:, :width - margin, :] = 200         # content flush to the left edge
    left, right = detect_side_margins(f)
    assert left == 0
    assert right == margin


def test_crop_removes_margins_only_when_present():
    cropped = crop_side_margins(_frame_with_margins())
    assert cropped.shape[1] == 1280 - 384

    already = np.full((512, 640, 3), 200, dtype=np.uint8)
    assert crop_side_margins(already).shape[1] == 640


def test_blur_ui_band_changes_bottom_and_preserves_top():
    f = np.random.RandomState(0).randint(0, 255, (720, 896, 3), dtype=np.uint8)
    out = blur_ui_band(f)
    band = int(round(720 * UI_BAND_FRACTION))
    assert np.array_equal(out[: 720 - band], f[: 720 - band])   # top untouched
    assert not np.array_equal(out[720 - band:], f[720 - band:])  # bottom changed


def test_blur_ui_band_reduces_bottom_variance():
    f = np.random.RandomState(1).randint(0, 255, (720, 896, 3), dtype=np.uint8)
    out = blur_ui_band(f)
    band = int(round(720 * UI_BAND_FRACTION))
    assert out[720 - band:].var() < f[720 - band:].var() / 2


def test_band_is_wider_than_the_measured_overlay():
    # The bar is ~45 px of 720 (0.0625). We deliberately blur more, because
    # under-blurring leaves readable text and reintroduces shortcut learning,
    # while over-blurring costs a sliver of mostly-black bottom edge.
    assert UI_BAND_FRACTION > 0.0625


def test_reblurring_is_stable():
    # Test clips arrive already blurred by the organizers, so our blur runs on
    # top of theirs. Re-applying must not error, reshape, or keep degrading:
    # the second application should change far less than the first.
    f = np.random.RandomState(2).randint(0, 255, (512, 640, 3), dtype=np.uint8)
    once = blur_ui_band(f)
    twice = blur_ui_band(once)
    assert twice.shape == once.shape
    assert twice.dtype == once.dtype
    first_delta = np.abs(once.astype(int) - f.astype(int)).mean()
    second_delta = np.abs(twice.astype(int) - once.astype(int)).mean()
    assert second_delta < first_delta / 2


def _detailed_frame_with_margins(width=1280, height=720, margin=192, seed=5):
    """Margins plus high-frequency content, so blurring is measurable.

    `_frame_with_margins` is a flat block of 200s: its bottom band has zero
    variance blurred or not, so it cannot witness anything about blurring.
    """
    rng = np.random.RandomState(seed)
    f = np.zeros((height, width, 3), dtype=np.uint8)
    f[:, margin:width - margin, :] = rng.randint(
        0, 256, (height, width - 2 * margin, 3), dtype=np.uint8)
    return f


def test_prepare_frame_is_square_and_blurred():
    """The UI band must be genuinely blurred by `prepare_frame`, not merely
    resized. Using the burned-in UI overlay to make predictions is prohibited
    by challenge rules and the overlay is blurred out of the test set, so a
    model trained on unblurred frames both breaks the rules and collapses at
    test time. This asserts the blur, not just the geometry: deleting the
    `blur_ui_band` call from `prepare_frame` must fail this test.
    """
    source = _detailed_frame_with_margins()
    out = prepare_frame(source, size=512)

    assert out.shape == (512, 512, 3)
    assert out.dtype == np.uint8

    # The same crop and resize, with no blur — the exact thing `prepare_frame`
    # would reduce to if the blur call were removed.
    unblurred = cv2.resize(crop_side_margins(source), (512, 512),
                           interpolation=cv2.INTER_CUBIC)

    # Stay well inside the band: it is blurred at source height (0.08 * 720
    # rows) and then resized, so its footprint in the output is ~41 rows.
    band = 30
    assert out[-band:].var() < unblurred[-band:].var() / 3, (
        "bottom band variance %.1f vs unblurred %.1f — the UI band is not "
        "being blurred" % (out[-band:].var(), unblurred[-band:].var()))

    # And the band is materially smoother than the untouched upper region of
    # the very same output frame.
    assert out[-band:].var() < out[:256].var() / 3


def test_prepare_frame_handles_test_geometry():
    # 640x512 at 1 fps is the Cat 1 test-set geometry; must not crash.
    f = np.full((512, 640, 3), 180, dtype=np.uint8)
    assert prepare_frame(f).shape == (512, 512, 3)
