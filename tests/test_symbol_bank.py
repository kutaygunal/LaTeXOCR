"""Tests for the template bank: font variants and font metrics.

``build_library`` gives one bitmap per symbol; ``build_bank`` additionally
renders the styles a symbol can appear in and measures each against the
baseline, which is what lets the recognizer separate glyphs that look alike.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from symbols import (  # noqa: E402
    SYMBOL_SOURCES,
    SymbolMetrics,
    Template,
    build_bank,
    build_library,
)


@pytest.fixture(scope="module")
def bank():
    return build_bank()


def _find(bank, name, style="italic"):
    return next(t for t in bank if t.name == name and t.style == style)


def test_bank_covers_every_symbol(bank):
    assert {t.name for t in bank} == set(SYMBOL_SOURCES)


def test_bank_entries_are_templates_with_binary_images(bank):
    for template in bank:
        assert isinstance(template, Template)
        assert template.image.size > 0
        assert set(np.unique(template.image)).issubset({0, 255})
        assert 0 in np.unique(template.image), f"{template.name} has no ink"


def test_bank_is_cached_between_calls():
    first, second = build_bank(), build_bank()
    assert len(first) == len(second)
    assert first[0].image is second[0].image  # same arrays, not re-rendered


def test_library_is_cached_between_calls():
    assert build_library()["x"] is build_library()["x"]


def test_letters_have_italic_and_roman_variants(bank):
    styles = {t.style for t in bank if t.name == "g"}
    assert {"italic", "roman"} <= styles


def test_uppercase_letters_also_have_a_bold_variant(bank):
    styles = {t.style for t in bank if t.name == "F"}
    assert {"italic", "roman", "bold"} <= styles


def test_operators_have_a_single_variant(bank):
    assert len([t for t in bank if t.name == "\\pm"]) == 1


# ---------------------------------------------------------------------------
# Metrics: the measurements that separate look-alike glyphs
# ---------------------------------------------------------------------------


def test_metrics_are_measured_in_x_heights(bank):
    x = _find(bank, "x").metrics
    assert isinstance(x, SymbolMetrics)
    # 'x' *is* the x-height, and it sits on the baseline.
    assert x.rel_h == pytest.approx(1.0, abs=0.05)
    assert x.desc == pytest.approx(0.0, abs=0.05)


def test_capitals_measure_taller_than_lowercase(bank):
    assert _find(bank, "X").metrics.rel_h > _find(bank, "x").metrics.rel_h * 1.2
    assert _find(bank, "O").metrics.rel_h > _find(bank, "o").metrics.rel_h * 1.2


def test_descender_hangs_below_the_baseline(bank):
    assert _find(bank, "g").metrics.desc > 0.2
    assert _find(bank, "9").metrics.desc < 0.1


def test_period_and_cdot_differ_only_in_height(bank):
    """The cue that separates `.` from `\\cdot`: one sits on the line."""
    period = _find(bank, ".").metrics
    cdot = _find(bank, "\\cdot").metrics
    assert period.rel_h == pytest.approx(cdot.rel_h, abs=0.05)
    assert period.desc > cdot.desc + 0.3  # the cdot floats above the baseline


def test_big_operators_measure_taller_than_the_line(bank):
    for name in ("\\int", "\\sum", "\\prod"):
        assert _find(bank, name).metrics.rel_h > 1.8, name


def test_radical_has_no_metrics(bank):
    """The radical is found structurally, so it is scored on shape alone."""
    assert _find(bank, "\\sqrt").metrics is None


def test_accents_sit_above_the_baseline(bank):
    for name in ("\\hat", "\\bar", "\\vec"):
        assert _find(bank, name).metrics.desc < -0.5, name


def test_bank_does_not_read_the_dataset(tmp_path):
    """Building templates must not depend on data/ (no test-set leakage)."""
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        assert len(build_bank()) >= len(SYMBOL_SOURCES)
    finally:
        os.chdir(old)
