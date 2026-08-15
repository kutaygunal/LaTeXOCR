"""Tests for the noise and skew handling in the shared preprocessing.

These cover the two failure modes that quietly destroy a page before any
recognizer sees it: a field of noise specks that drags the auto-crop across the
whole image, and a skew estimate that rotates an image which was never crooked.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import (  # noqa: E402
    _auto_crop,
    _deskew,
    _noise_size_gap,
    _profile_sharpness,
    _remove_specks,
    _skew_angle,
    preprocess,
)


def _page_with_glyphs(h=400, w=900):
    """A blank page with three glyph-sized blocks in the middle."""
    img = np.full((h, w), 255, dtype=np.uint8)
    img[160:240, 380:440] = 0
    img[160:240, 460:520] = 0
    img[160:240, 540:600] = 0
    return img


def _sprinkle(img, count=300, size=3, seed=0):
    """Scatter tiny speck clusters across the page."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    for _ in range(count):
        y = int(rng.integers(0, img.shape[0] - size))
        x = int(rng.integers(0, img.shape[1] - size))
        out[y : y + size, x : x + size] = 0
    return out


# ---------------------------------------------------------------------------
# Speck removal
# ---------------------------------------------------------------------------


def test_remove_specks_clears_scattered_noise():
    noisy = _sprinkle(_page_with_glyphs())
    cleaned = _remove_specks(noisy)
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (cleaned < 128).astype(np.uint8), connectivity=8
    )
    assert count - 1 == 3, "only the three glyph blocks should survive"


def test_remove_specks_keeps_a_small_glyph():
    """A period beside full-size glyphs is small, but it is not noise."""
    page = _page_with_glyphs()
    page[228:240, 620:632] = 0  # a period on the baseline
    cleaned = _remove_specks(page)
    count, _, _, _ = cv2.connectedComponentsWithStats(
        (cleaned < 128).astype(np.uint8), connectivity=8
    )
    assert count - 1 == 4


def test_remove_specks_leaves_a_clean_page_untouched():
    page = _page_with_glyphs()
    assert np.array_equal(_remove_specks(page), page)


def test_remove_specks_handles_a_blank_page():
    blank = np.full((50, 50), 255, dtype=np.uint8)
    assert np.array_equal(_remove_specks(blank), blank)


def test_noise_size_gap_finds_the_bimodal_cut():
    areas = np.array([1200, 1000, 340] + [20, 18, 15, 12, 10, 9, 8, 6, 5, 4, 3])
    assert _noise_size_gap(areas) == pytest.approx(340)


def test_noise_size_gap_ignores_a_lone_small_component():
    """One small glyph under a big jump is a period, not a noise field."""
    areas = np.array([4000, 3000, 2000, 100])
    assert _noise_size_gap(areas) == 0.0


def test_noise_size_gap_ignores_a_smooth_size_range():
    areas = np.array([4000, 3000, 2200, 1800, 1400, 1000, 800, 600, 400, 300, 200])
    assert _noise_size_gap(areas) == 0.0


def test_noisy_page_crops_to_its_content():
    """The whole point: noise must not blow the crop up to the full page."""
    noisy = _sprinkle(_page_with_glyphs())
    cropped = _auto_crop(_remove_specks(noisy))
    assert cropped.shape[0] <= 90
    assert cropped.shape[1] <= 240


# ---------------------------------------------------------------------------
# Skew estimation
# ---------------------------------------------------------------------------


def test_skew_angle_reports_zero_for_a_straight_page():
    assert _skew_angle(_page_with_glyphs()) == pytest.approx(0.0, abs=0.3)


def test_skew_angle_reports_the_correcting_rotation():
    """The angle returned is the one that puts the page back, so it is negated."""
    page = _page_with_glyphs()
    rotated = cv2.warpAffine(
        page, cv2.getRotationMatrix2D((450.0, 200.0), 6.0, 1.0), (900, 400),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    assert _skew_angle(rotated) == pytest.approx(-6.0, abs=1.5)


def test_deskew_straightens_a_rotated_page():
    page = _page_with_glyphs()
    rotated = cv2.warpAffine(
        page, cv2.getRotationMatrix2D((450.0, 200.0), 6.0, 1.0), (900, 400),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    straightened = _deskew(rotated)
    # Straightening recovers nearly all of the sharpness the tilt cost.
    assert _profile_sharpness(straightened) > _profile_sharpness(rotated)
    assert _profile_sharpness(straightened) > 0.95 * _profile_sharpness(page)


def test_deskew_leaves_a_stacked_expression_alone():
    """A fraction is a vertical stack, not a text line: never tilt it."""
    page = np.full((300, 200), 255, dtype=np.uint8)
    page[60:110, 80:120] = 0  # numerator
    page[140:148, 60:140] = 0  # bar
    page[180:230, 80:120] = 0  # denominator
    assert np.array_equal(_deskew(page), page)


def test_profile_sharpness_prefers_aligned_rows():
    aligned = _page_with_glyphs()
    tilted = cv2.warpAffine(
        aligned, cv2.getRotationMatrix2D((450.0, 200.0), 8.0, 1.0), (900, 400),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    assert _profile_sharpness(aligned) > _profile_sharpness(tilted)


def test_preprocess_of_a_noisy_page_keeps_the_glyphs_large(tmp_path):
    """End to end: a noisy page must not come out as specks around a tiny formula."""
    path = str(tmp_path / "noisy.png")
    cv2.imwrite(path, _sprinkle(_page_with_glyphs()))
    out = preprocess(path, height=128)
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (out < 128).astype(np.uint8), connectivity=8
    )
    heights = sorted(int(s[cv2.CC_STAT_HEIGHT]) for s in stats[1:])
    assert count - 1 == 3
    assert heights[-1] > 64, "the glyphs should fill the normalized height"
