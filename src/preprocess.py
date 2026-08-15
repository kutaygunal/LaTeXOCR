"""Shared image loading and preprocessing module.

This is the ONE shared input contract that both recognition approaches
(AI-001 and OWN-001) call. It turns a raw LaTeX equation image (PNG/JPG)
into a clean, normalized, binarized grayscale array ready for downstream
recognition.

Public API
----------
- ``load(path) -> np.ndarray``        : load an image as grayscale.
- ``preprocess(path) -> np.ndarray``  : full pipeline (see below).
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# Canonical output height (pixels) for the resized, preprocessed image.
CANONICAL_HEIGHT = 64

# How much sharper the horizontal projection must get before a rotation is
# considered a real skew correction rather than scoring noise.
SKEW_MIN_GAIN = 0.05

# A foreground blob smaller than this fraction of the largest one is noise.
SPECK_AREA_FRACTION = 0.008

# A size jump this wide, with at least this many components below it, marks the
# boundary between the glyphs on a page and a field of noise specks.
SPECK_GAP_RATIO = 6.0
SPECK_MIN_NOISE_COUNT = 10

# Supported image extensions (lowercase, without dot).
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class PreprocessError(Exception):
    """Raised when an image cannot be loaded or preprocessed."""


def load(path: str) -> np.ndarray:
    """Load an image from ``path`` and return it as a grayscale array.

    Parameters
    ----------
    path : str
        Path to a PNG/JPG/BMP/TIFF image.

    Returns
    -------
    np.ndarray
        Grayscale image, shape ``(H, W)``, dtype ``uint8``.

    Raises
    ------
    PreprocessError
        If the file does not exist, has an unsupported extension, or cannot
        be decoded by OpenCV.
    """
    if not os.path.isfile(path):
        raise PreprocessError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise PreprocessError(
            f"Unsupported image extension {ext!r}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise PreprocessError(f"OpenCV could not decode image: {path}")

    return img


def _normalize(img: np.ndarray) -> np.ndarray:
    """Normalize pixel values to the full [0, 255] range."""
    lo, hi = int(img.min()), int(img.max())
    if hi <= lo:
        # Degenerate (constant) image: return as-is to avoid div-by-zero.
        return img
    return ((img.astype(np.float32) - lo) * (255.0 / (hi - lo))).astype(np.uint8)


def _binarize(img: np.ndarray) -> np.ndarray:
    """Binarize with Otsu's method, normalizing polarity to black-on-white.

    Otsu assumes a bimodal histogram. We normalize first so the threshold is
    stable regardless of the original contrast. After thresholding we detect
    the background polarity and invert if needed so the output is ALWAYS
    black text on a white background, regardless of the input polarity
    (e.g. the ``black_bg`` tier has white text on a black background).
    """
    norm = _normalize(img)
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _normalize_polarity(binary)


def _normalize_polarity(binary: np.ndarray) -> np.ndarray:
    """Ensure the output is black text on a white background.

    Text typically covers a minority of the image area, so the background is
    the majority color. If the majority of pixels are black, the polarity is
    inverted (white text on black) and we flip it.
    """
    black_fraction = float(np.count_nonzero(binary < 128)) / binary.size
    if black_fraction > 0.5:
        return cv2.bitwise_not(binary)
    return binary


def _denoise(binary: np.ndarray) -> np.ndarray:
    """Remove salt-and-pepper noise with a median filter.

    A 3x3 median filter removes isolated noise pixels while preserving the
    stroke edges of glyphs far better than a mean (blur) filter.
    """
    return cv2.medianBlur(binary, 3)


def _noise_size_gap(areas: np.ndarray) -> float:
    """Find the size below which components are a noise field, not glyphs.

    A noisy page is bimodal: a handful of glyphs of ordinary size and a crowd of
    tiny specks, with a wide empty band between the two. Sorting the components
    by size and cutting at the largest jump finds that band without a fixed
    threshold, which matters because how big a speck is depends on the
    resolution. The cut is only taken when the jump is genuinely wide *and*
    there is a crowd below it — one small component below a big jump is a
    period, not a noise field.
    """
    if areas.size < SPECK_MIN_NOISE_COUNT + 1:
        return 0.0
    ordered = np.sort(areas)[::-1].astype(np.float64)
    ratios = ordered[:-1] / np.maximum(ordered[1:], 1.0)
    cut = int(np.argmax(ratios))
    below = ordered.size - (cut + 1)
    if ratios[cut] < SPECK_GAP_RATIO or below < SPECK_MIN_NOISE_COUNT:
        return 0.0
    return float(ordered[cut])


def _remove_specks(binary: np.ndarray) -> np.ndarray:
    """Erase foreground blobs far too small to be part of a glyph.

    A median filter clears isolated noise pixels but not the clumps of two or
    three that survive it, and those clumps are ruinous out of proportion to
    their size: scattered across the page they defeat the auto-crop, which then
    frames the noise instead of the equation and shrinks the formula into a
    corner of the output.

    Two rules decide what is too small, and the stricter one wins: a fixed
    fraction of the largest component, which handles a few stray pixels, and
    the bimodal cut of ``_noise_size_gap``, which handles a whole field of
    them. Both are relative to what else is on the page, so they hold at any
    resolution and keep genuinely small glyphs — a period, the dot of an ``i``.
    """
    fg = (binary < 128).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if count <= 1:
        return binary

    areas = stats[1:, cv2.CC_STAT_AREA]
    threshold = max(
        2.0,
        SPECK_AREA_FRACTION * float(areas.max()),
        _noise_size_gap(areas),
    )
    speck_labels = np.flatnonzero(areas < threshold) + 1
    if speck_labels.size == 0:
        return binary

    cleaned = binary.copy()
    cleaned[np.isin(labels, speck_labels)] = 255
    return cleaned


def _rotate(
    img: np.ndarray, angle: float, border: int = 255, flags: int = cv2.INTER_CUBIC
) -> np.ndarray:
    """Rotate an image about its center by ``angle`` degrees."""
    h, w = img.shape
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        img, matrix, (w, h), flags=flags,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border,
    )


def _profile_sharpness(binary: np.ndarray) -> float:
    """How concentrated the foreground is into distinct horizontal rows.

    When a line of text is horizontal, every row is either dense with glyph
    pixels or empty, so the row-sum profile is spiky and its sum of squares is
    large. Tilting the text smears ink across neighbouring rows and flattens
    the profile. Maximizing this is the standard projection-profile way to find
    the skew angle.

    The score is normalized by the total amount of ink, so that rotations which
    merely change how many pixels survive thresholding cannot win by adding
    ink rather than by aligning anything.
    """
    rows = (binary < 128).sum(axis=1).astype(np.float64)
    total = rows.sum()
    if total <= 0:
        return 0.0
    return float((rows * rows).sum() / (total * total))


def _skew_angle(binary: np.ndarray, limit: float = 12.0) -> float:
    """Estimate the image's skew in degrees, searching coarse then fine.

    The estimate runs on a downscaled copy — skew is a global property and does
    not need full resolution — which keeps the search to a couple of
    milliseconds.
    """
    h, w = binary.shape
    scale = min(1.0, 320.0 / max(h, w))
    small = (
        cv2.resize(binary, (max(1, int(w * scale)), max(1, int(h * scale))),
                   interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else binary
    )

    # Nearest-neighbour sampling during the search: interpolating a binary
    # image would blur strokes into new pixels and change the ink count, which
    # the score would read as an improvement.
    upright = _profile_sharpness(small)
    best_angle = 0.0
    best_score = upright
    for step, span, center in ((2.0, limit, 0.0), (0.25, 2.0, None)):
        origin = best_angle if center is None else center
        angle = origin - span
        while angle <= origin + span + 1e-9:
            if abs(angle) > 1e-9:
                rotated = _rotate(small, angle, flags=cv2.INTER_NEAREST)
                score = _profile_sharpness(rotated)
                if score > best_score:
                    best_score = score
                    best_angle = angle
            angle += step
    # Only accept a rotation that is clearly better than leaving the image
    # alone. A stacked expression such as a fraction has no single text line to
    # straighten, and small scoring wobbles must not tilt it.
    if best_score < (1.0 + SKEW_MIN_GAIN) * upright:
        return 0.0
    return best_angle


def _deskew(binary: np.ndarray) -> np.ndarray:
    """Rotate the image so text lines are horizontal.

    Returns the input unchanged if the skew is negligible or the image is
    effectively blank, so a straight image is never degraded by a needless
    interpolation pass.
    """
    # Foreground = black text pixels (value 0) on a white background.
    if not np.any(binary < 128):
        return binary

    angle = _skew_angle(binary)
    if abs(angle) < 0.5:
        return binary
    return _rotate(binary, angle)


def _auto_crop(binary: np.ndarray) -> np.ndarray:
    """Crop away uniform white borders around the text content."""
    fg = np.argwhere(binary < 128)
    if fg.size == 0:
        return binary

    y0, x0 = fg.min(axis=0)
    y1, x1 = fg.max(axis=0)
    return binary[y0 : y1 + 1, x0 : x1 + 1]


def _resize_to_height(img: np.ndarray, height: int = CANONICAL_HEIGHT) -> np.ndarray:
    """Resize preserving aspect ratio so the image has the given height."""
    h, w = img.shape
    if h == 0:
        return img
    scale = height / float(h)
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)
    return resized


def preprocess(path: str, height: int = CANONICAL_HEIGHT) -> np.ndarray:
    """Full preprocessing pipeline for a LaTeX equation image.

    Steps
    -----
    1. Load as grayscale.
    2. Normalize to full contrast range.
    3. Binarize with Otsu (text becomes black on white).
    4. Denoise with a median filter.
    5. Deskew to horizontal.
    6. Auto-crop uniform white borders.
    7. Resize to a canonical height (aspect ratio preserved).

    Parameters
    ----------
    path : str
        Path to the input image.
    height : int
        Target output height in pixels (default ``CANONICAL_HEIGHT``).

    Returns
    -------
    np.ndarray
        Binarized, denoised, deskewed, cropped, resized grayscale image.
        Text is black (0) on a white (255) background.

    Raises
    ------
    PreprocessError
        If the image cannot be loaded or is empty after preprocessing.
    """
    img = load(path)
    if img.size == 0:
        raise PreprocessError(f"Image is empty: {path}")

    binary = _binarize(img)
    binary = _denoise(binary)
    binary = _remove_specks(binary)
    binary = _deskew(binary)
    binary = _auto_crop(binary)
    binary = _resize_to_height(binary, height)
    # Geometric transforms (deskew/resize) interpolate and introduce gray
    # values; snap back to a strict binary image at the end.
    _, binary = cv2.threshold(binary, 128, 255, cv2.THRESH_BINARY)

    if binary.size == 0:
        raise PreprocessError(f"Image became empty after preprocessing: {path}")

    return binary
