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


def _deskew(binary: np.ndarray) -> np.ndarray:
    """Rotate the image so text lines are horizontal.

    Uses the minimum-area bounding rectangle of the foreground (text) pixels.
    Returns the input unchanged if the skew is negligible or the image is
    effectively blank.
    """
    # Foreground = black text pixels (value 0) on a white background.
    fg = np.argwhere(binary < 128)
    if fg.size == 0:
        return binary

    rect = cv2.minAreaRect(fg)
    angle = rect[2]

    # Normalize the angle to a small rotation in degrees.
    if angle > 45:
        angle -= 90
    if angle < -45:
        angle += 90

    # Skip rotation for negligible skew (avoids unnecessary interpolation).
    if abs(angle) < 0.5:
        return binary

    h, w = binary.shape
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(
        binary, matrix, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    return rotated


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
    binary = _deskew(binary)
    binary = _auto_crop(binary)
    binary = _resize_to_height(binary, height)
    # Geometric transforms (deskew/resize) interpolate and introduce gray
    # values; snap back to a strict binary image at the end.
    _, binary = cv2.threshold(binary, 128, 255, cv2.THRESH_BINARY)

    if binary.size == 0:
        raise PreprocessError(f"Image became empty after preprocessing: {path}")

    return binary
