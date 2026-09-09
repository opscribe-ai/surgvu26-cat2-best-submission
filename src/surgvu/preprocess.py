"""Frame preparation, identical at training and inference.

Blurring the bottom UI band is REQUIRED BY CHALLENGE RULES, not a tuning
choice: "using the information available in the UI to make predictions is not
allowed. To enforce this, the UI will be blurred from the test set". A model
trained on unblurred frames uses UI information whether or not that was the
intent, and would collapse at test where the band is gone.

Geometry is detected rather than assumed. Observed formats differ:
  Cat 2 sample clips   1280x720 @ 60 fps, black side margins present
  Cat 1 test clips      640x512 @  1 fps, already cropped
so cropping is conditional and the band is expressed as a fraction of height.
"""
import cv2
import numpy as np

UI_BAND_FRACTION = 0.08        # 45 px of 720 is 0.0625; widened for margin
BLUR_KERNEL = 51


def detect_side_margins(frame, threshold=12, run=8):
    """(left, right) width in pixels of near-black vertical margins.

    A margin boundary is only accepted once `run` consecutive columns clear
    the threshold. A single bright column at the frame edge -- a specular
    highlight, a vignette artifact, a compression edge -- must not be mistaken
    for the start of real content: real margins are ~193 px wide, so there is
    enormous separation between signal and noise at run=8.
    """
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    column_mean = grey.mean(axis=0)
    bright = column_mean > threshold

    def first_run(mask):
        streak = 0
        for i, is_bright in enumerate(mask):
            streak = streak + 1 if is_bright else 0
            if streak == run:
                return i - run + 1
        return None

    left = first_run(bright)
    right = first_run(bright[::-1])
    if left is None or right is None:
        return 0, 0
    return left, right


def crop_side_margins(frame):
    """Remove black side bars if present; return unchanged if already cropped."""
    left, right = detect_side_margins(frame)
    if left == 0 and right == 0:
        return frame
    width = frame.shape[1]
    return frame[:, left:width - right, :]


def blur_ui_band(frame, band_fraction=UI_BAND_FRACTION, kernel=BLUR_KERNEL):
    """Gaussian-blur the bottom band where the instrument UI is rendered.

    Safe to apply to an already-blurred frame -- blurring is idempotent enough
    that re-applying costs nothing and guarantees compliance regardless of
    what the organizers shipped.
    """
    height = frame.shape[0]
    band = max(1, int(round(height * band_fraction)))
    out = frame.copy()
    k = kernel if kernel % 2 == 1 else kernel + 1
    out[height - band:] = cv2.GaussianBlur(out[height - band:], (k, k), 0)
    return out


def prepare_frame(frame, size=512):
    """The single entry point. Crop, blur, resize -- in that order, always."""
    frame = crop_side_margins(frame)
    frame = blur_ui_band(frame)
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_CUBIC)
