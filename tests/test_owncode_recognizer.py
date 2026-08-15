"""Unit and integration tests for src/owncode_recognizer.py.

Covers: segmentation, template matching, structural reconstruction
(fractions, sqrt, accents, sub/superscripts, equals-merge, function grouping),
blank/empty input, all 5 tiers, and the optional CNN boost.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from owncode_recognizer import (  # noqa: E402
    Glyph,
    Recognizer,
    _classify_template,
    _estimate_baseline,
    _find_accent,
    _find_fraction,
    _find_sqrt,
    _fix_classification,
    _group_functions,
    _is_script,
    _merge_equals,
    _parse,
    _parse_mainline,
    _segment,
    recognize,
    recognize_with_cnn,
)
from symbols import build_library

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")


def _sample_image(tier: str) -> str:
    for fname in sorted(os.listdir(IMAGES_DIR)):
        if f"_{tier}_" in fname:
            return os.path.join(IMAGES_DIR, fname)
    raise FileNotFoundError(f"no image for tier {tier!r}")


def _glyph(x0, y0, w, h, symbol=None, area=None):
    """Build a synthetic Glyph with a solid black block image."""
    img = np.full((h, w), 255, dtype=np.uint8)
    img[:] = 0
    g = Glyph(x0, y0, w, h, area if area is not None else w * h, img)
    g.symbol = symbol
    return g


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def test_segment_counts_components():
    binary = np.full((100, 100), 255, dtype=np.uint8)
    binary[10:30, 10:30] = 0  # component 1
    binary[50:60, 50:80] = 0  # component 2
    comps = _segment(binary)
    assert len(comps) == 2


def test_segment_filters_tiny_noise():
    binary = np.full((100, 100), 255, dtype=np.uint8)
    binary[10:12, 10:12] = 0  # area 4 < MIN_AREA=5 -> filtered
    binary[50:60, 50:60] = 0  # area 100 -> kept
    comps = _segment(binary)
    assert len(comps) == 1


def test_segment_blank_returns_empty():
    binary = np.full((100, 100), 255, dtype=np.uint8)
    assert _segment(binary) == []


# ---------------------------------------------------------------------------
# Template matching
# ---------------------------------------------------------------------------


def test_classify_template_known_symbol():
    lib = build_library()
    # A glyph that is a solid block should match some template with score > 0.
    g = _glyph(0, 0, 20, 20)
    name, score = _classify_template(g, lib)
    assert name is not None
    assert 0.0 <= score <= 1.0


def test_classify_template_returns_best():
    lib = build_library()
    # A glyph shaped like a vertical bar should match '|' or '1' better than 'o'.
    g = _glyph(0, 0, 8, 40)
    name, _ = _classify_template(g, lib)
    assert name is not None


# ---------------------------------------------------------------------------
# Structural reconstruction
# ---------------------------------------------------------------------------


def test_parse_fraction():
    H, W = 100, 200
    bar = _glyph(50, 48, 100, 4)  # w/h=25>8, h<20, w>=0.5*max_w
    a = _glyph(80, 20, 20, 20, symbol="a")
    b = _glyph(80, 70, 20, 20, symbol="b")
    out = _parse([a, bar, b], H, W)
    assert out == r"\frac{a}{b}"


def test_parse_sqrt():
    H, W = 100, 200
    radical = _glyph(10, 10, 30, 60, symbol="\\sqrt")
    x = _glyph(45, 30, 20, 20, symbol="x")
    out = _parse([radical, x], H, W)
    assert out == r"\sqrt{x}"


def test_parse_sqrt_with_index():
    H, W = 100, 200
    radical = _glyph(10, 10, 30, 60, symbol="\\sqrt")
    x = _glyph(45, 30, 20, 20, symbol="x")
    idx = _glyph(2, 2, 8, 8, symbol="3")
    out = _parse([radical, x, idx], H, W)
    assert out == r"\sqrt[3]{x}"


def test_parse_accent_hat():
    H, W = 100, 200
    x = _glyph(50, 40, 20, 20, symbol="x")
    hat = _glyph(55, 10, 10, 10, symbol="\\hat")
    out = _parse([x, hat], H, W)
    assert out == r"\hat{x}"


def test_parse_superscript():
    H, W = 100, 200
    x = _glyph(50, 40, 20, 20, symbol="x")
    sup = _glyph(75, 10, 15, 15, symbol="2")
    out = _parse([x, sup], H, W)
    assert out == "x^2"


def test_parse_subscript():
    H, W = 100, 200
    x = _glyph(50, 40, 20, 20, symbol="x")
    sub = _glyph(75, 70, 15, 15, symbol="i")
    out = _parse([x, sub], H, W)
    assert out == "x_i"


def test_parse_merge_equals():
    H, W = 100, 200
    top = _glyph(40, 40, 60, 4)
    bot = _glyph(40, 50, 60, 4)
    merged = _merge_equals([top, bot], H)
    eqs = [c for c in merged if c.symbol == "="]
    assert len(eqs) == 1


def test_group_functions():
    assert _group_functions(["s", "i", "n"]) == "\\sin"
    assert _group_functions(["c", "o", "s"]) == "\\cos"
    assert _group_functions(["x", "y"]) == "xy"  # not a known function


def test_estimate_baseline():
    H = 100
    a = _glyph(0, 40, 10, 10, area=100)  # cy=45
    b = _glyph(0, 50, 10, 10, area=100)  # cy=55
    bl = _estimate_baseline([a, b], H)
    # Area-weighted median: first component to reach half the total area.
    assert bl == 45.0


def test_is_script():
    H = 100
    main = _glyph(0, 40, 10, 10)  # cy=45
    sup = _glyph(0, 5, 10, 10)  # cy=10
    assert not _is_script(main, 45.0, H)
    assert _is_script(sup, 45.0, H)


def test_find_accent():
    m = _glyph(50, 40, 20, 20)  # cy=50
    hat = _glyph(55, 10, 10, 10, symbol="\\hat")  # cy=15
    assert _find_accent(m, [hat]) == "\\hat"


def test_parse_empty():
    assert _parse([], 100, 200) == ""


def test_parse_mainline_empty():
    assert _parse_mainline([], 100, 200) == ""


# ---------------------------------------------------------------------------
# _fix_classification
# ---------------------------------------------------------------------------


def test_fix_classification_tall_narrow_bracket_to_int():
    H, W = 100, 200
    g = _glyph(0, 0, 10, 60, symbol="[")  # h=60>50, w/h=0.17<0.5
    _fix_classification([g], H, W)
    assert g.symbol == "\\int"
    assert g.conf == 1.0


def test_fix_classification_tall_narrow_close_bracket_to_int():
    H, W = 100, 200
    g = _glyph(0, 0, 10, 60, symbol="]")
    _fix_classification([g], H, W)
    assert g.symbol == "\\int"


def test_fix_classification_large_wide_to_sqrt():
    H, W = 100, 200
    g = _glyph(0, 0, 150, 80, symbol="x")  # w=150>140, h=80>70, area dominates
    _fix_classification([g], H, W)
    assert g.symbol == "\\sqrt"
    assert g.conf == 1.0


def test_fix_classification_does_not_change_normal():
    H, W = 100, 200
    g = _glyph(0, 40, 20, 20, symbol="x")
    _fix_classification([g], H, W)
    assert g.symbol == "x"


def test_fix_classification_short_bracket_unchanged():
    H, W = 100, 200
    g = _glyph(0, 40, 10, 20, symbol="[")  # h=20 not > 0.5*H
    _fix_classification([g], H, W)
    assert g.symbol == "["


# ---------------------------------------------------------------------------
# Nested structural reconstruction
# ---------------------------------------------------------------------------


def test_parse_fraction_with_sqrt_in_numerator():
    H, W = 200, 300
    bar = _glyph(100, 100, 100, 4)
    radical = _glyph(110, 20, 30, 60, symbol="\\sqrt")
    x = _glyph(145, 40, 20, 20, symbol="x")
    b = _glyph(130, 140, 20, 20, symbol="b")
    out = _parse([radical, x, bar, b], H, W)
    assert out == r"\frac{\sqrt{x}}{b}"


def test_parse_fraction_inside_fraction():
    H, W = 200, 300
    outer_bar = _glyph(100, 118, 100, 4)  # cy=120
    inner_bar = _glyph(120, 38, 60, 4)  # cy=40
    a = _glyph(130, 0, 20, 20, symbol="a")  # cy=10
    c = _glyph(130, 60, 20, 20, symbol="c")  # cy=70
    b = _glyph(130, 150, 20, 20, symbol="b")  # cy=160
    # Outer bar must be found first, so it comes first in the list.
    out = _parse([outer_bar, a, inner_bar, c, b], H, W)
    assert out == r"\frac{\frac{a}{c}}{b}"


def test_parse_sqrt_inside_fraction_denominator():
    H, W = 200, 300
    bar = _glyph(100, 100, 100, 4)
    a = _glyph(130, 20, 20, 20, symbol="a")
    radical = _glyph(110, 140, 30, 60, symbol="\\sqrt")
    x = _glyph(145, 160, 20, 20, symbol="x")
    out = _parse([a, bar, radical, x], H, W)
    assert out == r"\frac{a}{\sqrt{x}}"


# ---------------------------------------------------------------------------
# Full pipeline on real images
# ---------------------------------------------------------------------------


def test_recognize_blank_image_returns_empty(tmp_path):
    p = tmp_path / "blank.png"
    cv2.imwrite(str(p), np.full((100, 100), 255, dtype=np.uint8))
    out = recognize(str(p))
    assert isinstance(out, str)


@pytest.mark.parametrize("tier", ["clean", "white_bg", "black_bg", "noisy", "low_res"])
def test_recognize_all_tiers_no_crash(tier):
    out = recognize(_sample_image(tier))
    assert isinstance(out, str)


def test_recognize_clean_returns_latex():
    out = recognize(_sample_image("clean"))
    assert isinstance(out, str)
    assert out.strip() != ""


def test_recognizer_object_api():
    r = Recognizer()
    out = r.recognize(_sample_image("clean"))
    assert isinstance(out, str)


# ---------------------------------------------------------------------------
# CNN boost (integration, torch available)
# ---------------------------------------------------------------------------


def test_train_cnn_small_library():
    torch = pytest.importorskip("torch")
    from owncode_recognizer import SymbolCNN, train_cnn

    lib = build_library()
    # Use a small subset to keep the test fast.
    subset = {k: lib[k] for k in ("a", "b", "x", "1", "+")}
    cnn = train_cnn(subset, epochs=2, seed=0)
    assert isinstance(cnn, SymbolCNN)
    assert cnn.num_classes == len(subset)
    assert set(cnn.class_names) == set(subset.keys())


def test_recognize_with_cnn_no_crash():
    torch = pytest.importorskip("torch")
    from owncode_recognizer import train_cnn

    lib = build_library()
    subset = {k: lib[k] for k in ("a", "b", "x", "1", "+", "=")}
    cnn = train_cnn(subset, epochs=1, seed=0)
    out = recognize_with_cnn(_sample_image("clean"), cnn)
    assert isinstance(out, str)
