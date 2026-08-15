"""Tests for the layout analysis added to the own-code recognizer.

Covers the pieces that turn a bag of connected components into structured
LaTeX: glyph re-assembly (dotted letters, the two halves of a plus-minus),
splitting glyphs that were printed touching, the stacked-script rule, the
binomial reader, the script/main-line decision, and the spacing conventions of
the emitted source.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from owncode_recognizer import (  # noqa: E402
    Glyph,
    TemplateBank,
    Token,
    _build_units,
    _find_binom,
    _insert_thin_spaces,
    _join,
    _merge_dots,
    _merge_signs,
    _needs_space,
    _reads_as_script,
    _script,
    _script_owner,
    _split_merged,
    _stacked_scripts,
)
from symbols import build_bank  # noqa: E402


@pytest.fixture(scope="module")
def bank():
    return TemplateBank()


def _block(x0, y0, w, h, symbol=None):
    """A synthetic glyph: a solid black rectangle."""
    img = np.zeros((h, w), dtype=np.uint8)
    g = Glyph(x0, y0, w, h, w * h, img)
    g.symbol = symbol
    return g


# ---------------------------------------------------------------------------
# Glyph re-assembly
# ---------------------------------------------------------------------------


def test_merge_dots_joins_tittle_to_stem():
    stem = _block(10, 20, 4, 30)
    dot = _block(10, 12, 4, 4)
    merged = _merge_dots([stem, dot])
    assert len(merged) == 1
    assert merged[0].h == 38  # from the top of the dot to the foot of the stem


def test_merge_dots_joins_point_below_for_exclamation():
    bar = _block(10, 10, 4, 24)
    point = _block(10, 38, 4, 4)
    merged = _merge_dots([bar, point])
    assert len(merged) == 1
    assert merged[0].y1 == 42


def test_merge_dots_leaves_wide_base_alone():
    """An accent over a round letter is not a tittle: the base is not a stem."""
    base = _block(10, 20, 30, 30)
    accent = _block(15, 8, 20, 6)
    merged = _merge_dots([base, accent])
    assert len(merged) == 2


def test_merge_dots_leaves_operator_limit_alone():
    """The limit above a big operator must not be glued on as its dot."""
    sigma = _block(10, 30, 46, 65)
    limit = _block(20, 4, 22, 18)
    merged = _merge_dots([sigma, limit])
    assert len(merged) == 2


def test_merge_signs_joins_plus_and_bar():
    cross = _block(10, 10, 20, 20)
    bar = _block(10, 34, 20, 4)
    merged = _merge_signs([cross, bar])
    assert len(merged) == 1
    assert merged[0].h == 28


def test_merge_signs_ignores_distant_bar():
    cross = _block(10, 10, 20, 20)
    bar = _block(10, 80, 20, 4)
    assert len(_merge_signs([cross, bar])) == 2


# ---------------------------------------------------------------------------
# Splitting glyphs printed touching
# ---------------------------------------------------------------------------


def test_split_merged_separates_two_stacked_symbols(bank):
    """A sigma with its limit touching is split back into two glyphs."""
    templates = {t.name: t.image for t in build_bank() if t.style == "italic"}
    upper, lower = templates["n"], templates["\\sum"]
    width = max(upper.shape[1], lower.shape[1])
    canvas = np.full((upper.shape[0] + lower.shape[0], width), 255, dtype=np.uint8)
    canvas[: upper.shape[0], : upper.shape[1]] = upper
    canvas[upper.shape[0] :, : lower.shape[1]] = lower
    blob = Glyph(0, 0, width, canvas.shape[0], int((canvas < 128).sum()), canvas)

    parts = _split_merged([blob], bank)
    assert len(parts) >= 2, "a sigma stacked on its limit should be split"


def test_split_merged_keeps_a_clean_glyph_whole(bank):
    """A glyph the bank recognizes outright is never split."""
    template = next(t for t in build_bank() if t.name == "\\sum")
    image = template.image
    glyph = Glyph(0, 0, image.shape[1], image.shape[0], int((image < 128).sum()), image)
    assert len(_split_merged([glyph], bank)) == 1


# ---------------------------------------------------------------------------
# Stacked scripts
# ---------------------------------------------------------------------------


def test_stacked_scripts_marks_limit_under_operator():
    sigma = _block(20, 30, 46, 65)
    lower = _block(25, 100, 30, 20)
    stacked = _stacked_scripts([sigma, lower])
    assert id(lower) in stacked
    assert id(sigma) not in stacked


def test_stacked_scripts_ignores_side_by_side_glyphs():
    a = _block(0, 20, 20, 30)
    b = _block(30, 20, 20, 30)
    assert _stacked_scripts([a, b]) == set()


def test_stacked_scripts_ignores_equal_sized_stack():
    """Two glyphs of the same size stacked are a structure, not a script."""
    a = _block(0, 0, 20, 30)
    b = _block(0, 40, 20, 30)
    assert _stacked_scripts([a, b]) == set()


# ---------------------------------------------------------------------------
# Binomial coefficients
# ---------------------------------------------------------------------------


def test_find_binom_reads_stacked_pair_in_parens():
    opening = _block(0, 0, 10, 100, symbol="(")
    n = _block(20, 10, 20, 20, symbol="n")
    k = _block(20, 60, 20, 20, symbol="k")
    closing = _block(50, 0, 10, 100, symbol=")")
    found = _find_binom([opening, n, k, closing])
    assert found is not None
    _, _, upper, lower, _ = found
    assert [g.symbol for g in upper] == ["n"]
    assert [g.symbol for g in lower] == ["k"]


def test_find_binom_ignores_ordinary_parentheses():
    """`\\sin(x)` holds one glyph on one line, not a stack."""
    opening = _block(0, 0, 10, 60, symbol="(")
    x = _block(20, 20, 20, 20, symbol="x")
    closing = _block(50, 0, 10, 60, symbol=")")
    assert _find_binom([opening, x, closing]) is None


# ---------------------------------------------------------------------------
# Script attachment and classification
# ---------------------------------------------------------------------------


def test_script_owner_prefers_the_preceding_unit():
    """The `1` of `\\int_0^1` belongs to the integral, not the following x."""
    integral = _build_units([_block(0, 10, 25, 90, symbol="\\int")])[0]
    x = _build_units([_block(85, 55, 40, 40, symbol="x")])[0]
    one = _block(64, 0, 20, 36)
    assert _script_owner(one, [integral, x]) is integral


def test_script_owner_prefers_overlapping_unit():
    sigma = _build_units([_block(20, 30, 46, 65, symbol="\\sum")])[0]
    other = _build_units([_block(120, 40, 20, 30, symbol="x")])[0]
    limit = _block(25, 100, 30, 20)
    assert _script_owner(limit, [sigma, other]) is sigma


def test_reads_as_script_accepts_a_raised_digit(bank):
    """A small digit raised off the line reads as a superscript."""
    template = next(t for t in build_bank() if t.name == "2" and t.style == "italic")
    scale = 0.75
    height = int(template.image.shape[0] * scale)
    width = int(template.image.shape[1] * scale)
    import cv2

    image = cv2.resize(template.image, (width, height), interpolation=cv2.INTER_AREA)
    x_height = template.image.shape[0] / template.metrics.rel_h
    baseline = 200.0
    raised = Glyph(
        0, int(baseline - 0.95 * x_height - height), width, height,
        int((image < 128).sum()), image,
    )
    assert _reads_as_script(raised, bank, x_height, baseline)


def test_reads_as_script_rejects_a_glyph_on_the_line(bank):
    template = next(t for t in build_bank() if t.name == "2" and t.style == "italic")
    image = template.image
    x_height = image.shape[0] / template.metrics.rel_h
    baseline = 200.0
    on_line = Glyph(
        0, int(baseline - image.shape[0]), image.shape[1], image.shape[0],
        int((image < 128).sum()), image,
    )
    assert not _reads_as_script(on_line, bank, x_height, baseline)


# ---------------------------------------------------------------------------
# Emission: spacing conventions
# ---------------------------------------------------------------------------


def _tokens(*pairs):
    return [Token(text, kind) for text, kind in pairs]


def test_join_spaces_infix_operators():
    tokens = _tokens(("x", "atom"), ("+", "operator"), ("y", "atom"))
    assert _join(tokens) == "x + y"


def test_join_keeps_scripts_tight():
    tokens = _tokens(("i", "atom"), ("=", "operator"), ("1", "atom"))
    assert _join(tokens, script=True) == "i=1"


def test_join_spaces_arrow_even_inside_a_script():
    tokens = _tokens(("x", "atom"), ("\\to", "operator"), ("0", "atom"))
    assert _join(tokens, script=True) == "x \\to 0"


def test_join_separates_command_from_following_letter():
    """`\\pi r` must not run together into the unknown command `\\pir`."""
    tokens = _tokens(("\\pi", "atom"), ("r^2", "atom"))
    assert _join(tokens) == "\\pi r^2"


def test_join_keeps_plain_parentheses_tight():
    tokens = _tokens(("\\log", "limit"), ("(", "open"), ("x", "atom"), (")", "close"))
    assert _join(tokens) == "\\log (x)"


def test_join_spaces_angle_brackets():
    tokens = _tokens(("\\langle", "open"), ("a", "atom"), ("\\rangle", "close"))
    assert _join(tokens) == "\\langle a \\rangle"


def test_join_spaces_after_a_limit_operator():
    tokens = _tokens(("\\sum_{i=1}^{n}", "limit"), ("i", "atom"))
    assert _join(tokens) == "\\sum_{i=1}^{n} i"


def test_needs_space_leaves_comma_attached_to_its_left():
    left = Token("a", "atom")
    comma = Token(",", "comma")
    right = Token("b", "atom")
    assert not _needs_space(left, comma, False)
    assert _needs_space(comma, right, False)


def test_script_braces_single_character_only_when_asked():
    assert _script("^", "n") == "^n"
    assert _script("^", "n", braces=True) == "^{n}"
    assert _script("_", "i=1") == "_{i=1}"


def test_insert_thin_spaces_marks_a_wide_gap():
    tokens = [Token("x^2", "atom", 0, 40), Token("dx", "atom", 80, 120)]
    out = _insert_thin_spaces(tokens, x_height=40.0)
    assert [t.text for t in out] == ["x^2", "\\,", "dx"]


def test_insert_thin_spaces_ignores_normal_letter_spacing():
    tokens = [Token("a", "atom", 0, 20), Token("b", "atom", 22, 42)]
    out = _insert_thin_spaces(tokens, x_height=40.0)
    assert [t.text for t in out] == ["a", "b"]


def test_insert_thin_spaces_skips_gaps_around_operators():
    tokens = [
        Token("x", "atom", 0, 20),
        Token("+", "operator", 60, 80),
        Token("y", "atom", 120, 140),
    ]
    out = _insert_thin_spaces(tokens, x_height=20.0)
    assert [t.text for t in out] == ["x", "+", "y"]
