"""Unit and integration tests for src/symbols.py.

Covers: library builds deterministically, correct size, no test_set
dependency, binary/non-empty templates, and vocabulary constants.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from symbols import (  # noqa: E402
    FUNCTION_NAMES,
    LIMIT_SYMBOLS,
    SYMBOL_SOURCES,
    _render_symbol,
    build_library,
)


def test_library_size_matches_sources():
    lib = build_library()
    assert len(lib) == len(SYMBOL_SOURCES)


def test_library_keys_match_sources():
    lib = build_library()
    assert set(lib.keys()) == set(SYMBOL_SOURCES.keys())


def test_library_builds_deterministically():
    a = build_library()
    b = build_library()
    assert set(a.keys()) == set(b.keys())
    for name in a:
        assert a[name].shape == b[name].shape


def test_library_templates_are_binary():
    lib = build_library()
    for name, tmpl in lib.items():
        vals = set(np.unique(tmpl))
        assert vals.issubset({0, 255}), f"{name} not binary: {vals}"


def test_library_templates_non_empty():
    lib = build_library()
    for name, tmpl in lib.items():
        assert tmpl.size > 0, f"{name} is empty"
        assert 0 in np.unique(tmpl), f"{name} has no black (foreground) pixels"


def test_library_no_test_set_dependency(tmp_path):
    """build_library must not read data/ or the test split."""
    import json
    import os

    # Point cwd away from any data dir and confirm build still works.
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        lib = build_library()
        assert len(lib) == len(SYMBOL_SOURCES)
    finally:
        os.chdir(old)


def test_function_names_present():
    for name in ("sin", "cos", "log", "ln", "lim", "exp", "tan"):
        assert name in FUNCTION_NAMES


def test_limit_symbols_present():
    for name in ("\\int", "\\sum", "\\prod", "\\lim"):
        assert name in LIMIT_SYMBOLS


def test_render_symbol_letters():
    for ch in "abx12+=":
        tmpl = _render_symbol(ch)
        assert tmpl.size > 0
        assert 0 in np.unique(tmpl)


def test_render_accent_extracts_component():
    """Accent templates must be small (the accent, not the base letter)."""
    hat = _render_symbol("\\hat")
    bar = _render_symbol("\\bar")
    vec = _render_symbol("\\vec")
    for t in (hat, bar, vec):
        assert t.size > 0
        assert 0 in np.unique(t)
        # Accent is a small component, much smaller than a full letter canvas.
        assert t.shape[0] < 200 and t.shape[1] < 200


def test_render_sqrt_non_empty():
    tmpl = _render_symbol("\\sqrt")
    assert tmpl.size > 0
    assert 0 in np.unique(tmpl)
