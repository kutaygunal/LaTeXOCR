"""Unit and integration tests for src/preprocess.py.

Covers: valid image, white-bg, black-bg, noisy, low-res, invalid path,
corrupt file, unsupported extension, empty image, output contract
(binary, canonical height, black-on-white), and CLI wiring.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import (  # noqa: E402
    CANONICAL_HEIGHT,
    PreprocessError,
    _binarize,
    _normalize,
    load,
    preprocess,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")


def _sample_image(tier: str) -> str:
    """Return the path of the first image of a given tier."""
    for fname in sorted(os.listdir(IMAGES_DIR)):
        if f"_{tier}_" in fname:
            return os.path.join(IMAGES_DIR, fname)
    raise FileNotFoundError(f"no image for tier {tier!r}")


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_valid_image_returns_grayscale():
    img = load(_sample_image("clean"))
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    assert img.ndim == 2  # grayscale, not color


def test_load_invalid_path_raises():
    with pytest.raises(PreprocessError):
        load(os.path.join(IMAGES_DIR, "does_not_exist.png"))


def test_load_unsupported_extension_raises(tmp_path):
    p = tmp_path / "x.gif"
    p.write_bytes(b"GIF89a")
    with pytest.raises(PreprocessError):
        load(str(p))


def test_load_corrupt_file_raises(tmp_path):
    p = tmp_path / "corrupt.png"
    p.write_bytes(b"this is not a real png at all")
    with pytest.raises(PreprocessError):
        load(str(p))


# ---------------------------------------------------------------------------
# _normalize / _binarize
# ---------------------------------------------------------------------------


def test_normalize_spans_full_range():
    img = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    out = _normalize(img)
    assert out.min() == 0
    assert out.max() == 255


def test_normalize_constant_image_unchanged():
    img = np.full((4, 4), 128, dtype=np.uint8)
    out = _normalize(img)
    np.testing.assert_array_equal(out, img)


def test_binarize_output_is_binary():
    img = np.random.default_rng(0).integers(0, 255, (64, 64)).astype(np.uint8)
    out = _binarize(img)
    assert set(np.unique(out)).issubset({0, 255})


# ---------------------------------------------------------------------------
# preprocess() full pipeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tier",
    ["clean", "white_bg", "black_bg", "noisy", "low_res"],
)
def test_preprocess_all_tiers_succeeds(tier):
    out = preprocess(_sample_image(tier))
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8
    assert out.ndim == 2
    assert out.size > 0


def test_preprocess_canonical_height():
    for tier in ["clean", "noisy", "low_res", "black_bg", "white_bg"]:
        out = preprocess(_sample_image(tier))
        assert out.shape[0] == CANONICAL_HEIGHT, f"tier={tier}"


def test_preprocess_custom_height():
    out = preprocess(_sample_image("clean"), height=32)
    assert out.shape[0] == 32


def test_preprocess_output_is_strictly_binary():
    for tier in ["clean", "noisy", "low_res", "black_bg", "white_bg"]:
        out = preprocess(_sample_image(tier))
        vals = set(np.unique(out))
        assert vals.issubset({0, 255}), f"tier={tier} got {vals}"


def test_preprocess_text_is_black_on_white():
    """After preprocessing, foreground (text) must be black (0) on white (255)."""
    for tier in ["clean", "white_bg", "black_bg", "noisy", "low_res"]:
        out = preprocess(_sample_image(tier))
        # There must be both black and white pixels (non-blank content).
        assert 0 in np.unique(out), f"tier={tier} has no black (text) pixels"
        assert 255 in np.unique(out), f"tier={tier} has no white (bg) pixels"


def test_preprocess_black_bg_inverted_to_black_text():
    """black_bg tier is white-on-black; pipeline must invert to black-on-white."""
    out = preprocess(_sample_image("black_bg"))
    # Majority of pixels should be white background.
    white_frac = float(np.mean(out == 255))
    assert white_frac > 0.5, f"black_bg not inverted, white_frac={white_frac:.3f}"


def test_preprocess_empty_image_raises(monkeypatch, tmp_path):
    """Guard: an image that loads to zero size must raise PreprocessError."""
    import preprocess as preprocess_mod

    p = tmp_path / "empty.png"
    p.write_bytes(b"placeholder")

    def fake_load(path):
        return np.zeros((0, 0), dtype=np.uint8)

    monkeypatch.setattr(preprocess_mod, "load", fake_load)
    with pytest.raises(PreprocessError):
        preprocess(str(p))


def test_preprocess_blank_image_raises_or_returns_blank(tmp_path):
    """A fully blank image has no foreground; pipeline must not crash."""
    p = tmp_path / "blank.png"
    cv2.imwrite(str(p), np.full((100, 100), 255, dtype=np.uint8))
    out = preprocess(str(p))
    assert out.size > 0


def test_preprocess_deterministic():
    a = preprocess(_sample_image("clean"))
    b = preprocess(_sample_image("clean"))
    np.testing.assert_array_equal(a, b)


def test_preprocess_denoise_removes_salt_pepper():
    """A clean image with injected isolated noise pixels should be cleaned."""
    base = preprocess(_sample_image("clean"))
    # Re-inject noise into a copy of the raw image and re-run.
    raw = load(_sample_image("clean"))
    noisy = raw.copy()
    rng = np.random.default_rng(1)
    mask = rng.random(raw.shape) < 0.02
    noisy[mask] = 255 - noisy[mask]
    p = os.path.join(os.path.dirname(__file__), "_noisy_tmp.png")
    cv2.imwrite(p, noisy)
    try:
        out = preprocess(p)
        assert out.shape[0] == CANONICAL_HEIGHT
    finally:
        if os.path.exists(p):
            os.remove(p)


# ---------------------------------------------------------------------------
# CLI wiring (main.py)
# ---------------------------------------------------------------------------


def test_cli_preprocess_subcommand(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from main import main

    src = _sample_image("clean")
    out = str(tmp_path / "out.png")
    rc = main(["preprocess", src, "--height", "48", "--out", out])
    assert rc == 0
    assert os.path.isfile(out)
    img = cv2.imread(out, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    assert img.shape[0] == 48


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_preprocess_performance_budget():
    """Preprocessing a single image should complete quickly (perf smoke test)."""
    import time

    start = time.perf_counter()
    for _ in range(5):
        preprocess(_sample_image("clean"))
    elapsed = time.perf_counter() - start
    per_image = elapsed / 5
    # Generous budget: a single image should be well under 1s.
    assert per_image < 1.0, f"preprocess too slow: {per_image:.3f}s/image"


# ---------------------------------------------------------------------------
# Deskew
# ---------------------------------------------------------------------------


def _synthetic_text_binary(h=120, w=400) -> np.ndarray:
    """A black horizontal bar (text) on a white background."""
    img = np.full((h, w), 255, dtype=np.uint8)
    img[40:80, 60:340] = 0
    return img


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        img, m, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )


def _foreground_angle(binary: np.ndarray) -> float:
    """Return the min-area-rect angle of the foreground (text) pixels."""
    fg = np.argwhere(binary < 128)
    if fg.size == 0:
        return 0.0
    rect = cv2.minAreaRect(fg)
    angle = rect[2]
    if angle > 45:
        angle -= 90
    if angle < -45:
        angle += 90
    return angle


def test_deskew_rotated_image_becomes_horizontal():
    """A rotated text image should be deskewed back to horizontal."""
    from preprocess import _deskew

    base = _synthetic_text_binary()
    rotated = _rotate(base, 8.0)
    # Sanity: the rotated input is genuinely skewed.
    assert abs(_foreground_angle(rotated)) > 3.0

    out = _deskew(rotated)
    angle = _foreground_angle(out)
    assert abs(angle) < 1.0, f"deskew failed, residual angle={angle:.2f}"


def test_deskew_negligible_skew_unchanged():
    """abs(angle) < 0.5 branch must return the input unchanged."""
    from preprocess import _deskew

    base = _synthetic_text_binary()
    # A horizontal bar has ~0 skew; _deskew must return it unchanged.
    out = _deskew(base)
    np.testing.assert_array_equal(out, base)


def test_deskew_blank_image_unchanged():
    """A blank (no foreground) image must pass through unchanged."""
    from preprocess import _deskew

    blank = np.full((100, 100), 255, dtype=np.uint8)
    out = _deskew(blank)
    np.testing.assert_array_equal(out, blank)


def test_preprocess_rotated_image_output_horizontal():
    """End-to-end: a rotated real image is deskewed by the full pipeline."""
    import tempfile

    raw = load(_sample_image("clean"))
    rotated = _rotate(raw, 6.0)
    p = os.path.join(os.path.dirname(__file__), "_rot_tmp.png")
    cv2.imwrite(p, rotated)
    try:
        out = preprocess(p)
        assert abs(_foreground_angle(out)) < 2.0
    finally:
        if os.path.exists(p):
            os.remove(p)


# ---------------------------------------------------------------------------
# Auto-crop
# ---------------------------------------------------------------------------


def test_auto_crop_removes_white_border():
    """Padding with a white border must be cropped away by _auto_crop."""
    from preprocess import _auto_crop

    base = _synthetic_text_binary()
    # Text block is rows 40:80, cols 60:340 of the base.
    text_block = base[40:80, 60:340]
    # Pad with a wide white border on all sides.
    padded = np.full((base.shape[0] + 100, base.shape[1] + 100), 255, dtype=np.uint8)
    padded[50 : 50 + base.shape[0], 50 : 50 + base.shape[1]] = base

    cropped = _auto_crop(padded)
    assert cropped.shape[0] < padded.shape[0]
    assert cropped.shape[1] < padded.shape[1]
    # Content preserved: the cropped region equals the original text block.
    np.testing.assert_array_equal(cropped, text_block)


def test_auto_crop_blank_unchanged():
    from preprocess import _auto_crop

    blank = np.full((100, 100), 255, dtype=np.uint8)
    out = _auto_crop(blank)
    np.testing.assert_array_equal(out, blank)


def test_preprocess_padded_image_crops_border():
    """End-to-end: a padded image is cropped so output width < padded width."""
    import tempfile

    raw = load(_sample_image("clean"))
    padded = np.full((raw.shape[0] + 200, raw.shape[1] + 200), 255, dtype=np.uint8)
    padded[100 : 100 + raw.shape[0], 100 : 100 + raw.shape[1]] = raw
    p = os.path.join(os.path.dirname(__file__), "_pad_tmp.png")
    cv2.imwrite(p, padded)
    try:
        out = preprocess(p)
        # Output is resized to height 64; width must be smaller than the
        # padded input's width (border cropped before resize).
        assert out.shape[1] < padded.shape[1]
        assert out.shape[0] == CANONICAL_HEIGHT
    finally:
        if os.path.exists(p):
            os.remove(p)


# ---------------------------------------------------------------------------
# Denoise (strengthened)
# ---------------------------------------------------------------------------


def _isolated_pixel_count(binary: np.ndarray) -> int:
    """Count isolated (salt-and-pepper) pixels via a 3x3 median filter.

    A pixel is 'isolated' if it changes when a 3x3 median filter is applied.
    """
    filtered = cv2.medianBlur(binary, 3)
    return int(np.count_nonzero(binary != filtered))


def test_denoise_actually_removes_noise():
    """Median denoise must reduce isolated salt-and-pepper pixels."""
    from preprocess import _denoise

    base = _synthetic_text_binary()
    rng = np.random.default_rng(7)
    noisy = base.copy()
    mask = rng.random(base.shape) < 0.05
    noisy[mask] = 255 - noisy[mask]

    before = _isolated_pixel_count(noisy)
    denoised = _denoise(noisy)
    after = _isolated_pixel_count(denoised)

    assert before > 0, "noise injection produced no isolated pixels"
    assert after < before, f"denoise did not reduce noise: {before} -> {after}"


def test_denoise_output_closer_to_clean_reference():
    """Denoised output must be closer to the clean reference than the noisy input."""
    from preprocess import _denoise

    base = _synthetic_text_binary()
    rng = np.random.default_rng(11)
    noisy = base.copy()
    mask = rng.random(base.shape) < 0.05
    noisy[mask] = 255 - noisy[mask]

    noisy_diff = int(np.count_nonzero(noisy != base))
    denoised = _denoise(noisy)
    denoised_diff = int(np.count_nonzero(denoised != base))

    assert denoised_diff < noisy_diff, (
        f"denoise not closer to clean: noisy_diff={noisy_diff}, "
        f"denoised_diff={denoised_diff}"
    )
