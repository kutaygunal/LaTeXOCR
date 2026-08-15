"""Symbol library / template rendering helper for the own-code OCR pipeline.

Renders a curated set of LaTeX symbols to binary images using **matplotlib
mathtext** (no LaTeX engine is installed), so the own-code recognizer can
classify segmented glyphs by template matching.

The library is built from the symbol vocabulary that appears in the
ground-truth expressions. It is rendered deterministically and does **not**
depend on the test split, so building it never leaks test data.

Besides the plain bitmap templates, this module exposes two things the
recognizer needs to tell visually similar glyphs apart:

* **Font variants** (``build_variants``) — the same symbol rendered in the
  shapes mathtext actually produces in an equation: italic for variables,
  roman for function names (``\\log``), bold for ``\\mathbf``. A roman ``g``
  looks nothing like an italic one, so one template per name is not enough.
* **Font metrics** (``symbol_metrics``) — each symbol's height and vertical
  position relative to the x-height of the line it sits on. Shape alone cannot
  separate ``.`` from ``\\cdot`` or ``o`` from ``O``; their position and size
  on the baseline can.

Rendering is expensive (each template is a matplotlib figure), so the library,
the variants and the metrics are all built once and cached per process.

Public API
----------
- ``build_library() -> dict[str, np.ndarray]`` : render every symbol template.
- ``build_bank() -> list[Template]`` : every template variant with metrics.
- ``SYMBOL_SOURCES`` : name -> mathtext source mapping.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Symbol vocabulary
# ---------------------------------------------------------------------------

# name -> mathtext source. Single-glyph symbols render as one connected
# component. Accents (hat/bar/vec) are rendered over a base letter and the
# accent component is extracted separately (see _render_symbol).
SYMBOL_SOURCES: dict[str, str] = {
    # lowercase letters
    "a": r"a", "b": r"b", "c": r"c", "d": r"d", "e": r"e", "f": r"f",
    "g": r"g", "h": r"h", "i": r"i", "j": r"j", "k": r"k", "l": r"l",
    "m": r"m", "n": r"n", "o": r"o", "p": r"p", "q": r"q", "r": r"r",
    "s": r"s", "t": r"t", "u": r"u", "v": r"v", "w": r"w", "x": r"x",
    "y": r"y", "z": r"z",
    # uppercase letters
    "A": r"A", "B": r"B", "C": r"C", "D": r"D", "E": r"E", "F": r"F",
    "G": r"G", "H": r"H", "I": r"I", "J": r"J", "K": r"K", "L": r"L",
    "M": r"M", "N": r"N", "O": r"O", "P": r"P", "Q": r"Q", "R": r"R",
    "S": r"S", "T": r"T", "U": r"U", "V": r"V", "W": r"W", "X": r"X",
    "Y": r"Y", "Z": r"Z",
    # digits
    "0": r"0", "1": r"1", "2": r"2", "3": r"3", "4": r"4", "5": r"5",
    "6": r"6", "7": r"7", "8": r"8", "9": r"9",
    # operators and symbols
    "+": r"+", "-": r"-", "=": r"=", "\\pm": r"\pm", "\\cdot": r"\cdot",
    "\\times": r"\times", "\\to": r"\to", "\\langle": r"\langle",
    "\\rangle": r"\rangle", "(": r"(", ")": r")", "[": r"[", "]": r"]",
    "|": r"|", ",": r",", ".": r".", "!": r"!", "\\nabla": r"\nabla",
    "\\partial": r"\partial", "\\infty": r"\infty", "\\alpha": r"\alpha",
    "\\beta": r"\beta", "\\gamma": r"\gamma", "\\pi": r"\pi",
    # Rendered bare: attaching empty limits (\sum_{}^{}) makes mathtext switch
    # to the smaller inline form, which is not the shape these operators have
    # in the equations being read.
    "\\sum": r"\sum", "\\prod": r"\prod",
    "\\int": r"\int",
    "\\sqrt": r"\sqrt{x}",
    # accents (rendered over a base letter; accent component is extracted)
    "\\hat": r"\hat{x}", "\\bar": r"\bar{x}", "\\vec": r"\vec{x}",
}

# Multi-letter function names recognized during reconstruction. They render as
# a sequence of roman letters, so they are detected by grouping consecutive
# baseline letters and matching against this set.
FUNCTION_NAMES = {
    "sin", "cos", "log", "ln", "lim", "exp", "tan", "sec", "csc", "cot",
    "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan", "det", "gcd",
    "min", "max", "sup", "inf", "arg", "dim", "mod", "deg", "ker", "hom",
}

# Symbols that take limits (sub/superscripts) in the ground-truth expressions.
LIMIT_SYMBOLS = {"\\int", "\\sum", "\\prod", "\\lim"}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_latex(
    latex: str,
    fontsize: int = 40,
    dpi: int = 200,
) -> np.ndarray:
    """Render a LaTeX expression to a binary black-on-white numpy array."""
    fig = plt.figure(figsize=(8, 2))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")
    ax.axis("off")
    ax.text(
        0.5, 0.5, f"${latex}$",
        ha="center", va="center", fontsize=fontsize, color="black",
    )

    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.1,
        facecolor="white",
    )
    plt.close(fig)

    buf.seek(0)
    img = Image.open(buf).convert("L")
    arr = np.array(img)
    _, binary = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY)
    return binary


def _extract_accent(binary: np.ndarray) -> np.ndarray:
    """Extract the accent component (hat/bar/vec) from a rendered base image.

    The accent is the topmost connected component of the rendered ``\\hat{x}``
    style image (the base letter is the large component below it).
    """
    fg = (binary < 128).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    comps = [
        (stats[i][1], stats[i][0], stats[i][2], stats[i][3], stats[i][4])
        for i in range(1, n)
        if stats[i][4] >= 5
    ]
    if not comps:
        return binary
    # Topmost component (smallest y0) is the accent.
    comps.sort(key=lambda c: c[0])
    y0, x0, w, h, _ = comps[0]
    return binary[y0 : y0 + h, x0 : x0 + w]


def _crop_to_content(binary: np.ndarray) -> np.ndarray:
    """Crop a binary image to its foreground (black) bounding box."""
    fg = np.argwhere(binary < 128)
    if fg.size == 0:
        return binary
    y0, x0 = fg.min(axis=0)
    y1, x1 = fg.max(axis=0)
    return binary[y0 : y1 + 1, x0 : x1 + 1]


def _largest_component(binary: np.ndarray) -> np.ndarray:
    """Return the largest connected component of a binary image, cropped."""
    fg = (binary < 128).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    best = None
    best_area = -1
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area > best_area:
            best_area = area
            best = (x, y, w, h)
    if best is None:
        return binary
    x, y, w, h = best
    return binary[y : y + h, x : x + w]


def _render_symbol(name: str) -> np.ndarray:
    """Render a single symbol template as a tight binary image."""
    source = SYMBOL_SOURCES[name]
    binary = _render_latex(source)
    if name in ("\\hat", "\\bar", "\\vec"):
        return _crop_to_content(_extract_accent(binary))
    if name == "\\sqrt":
        # The radical sign + vinculum is the largest component of \sqrt{x}.
        return _crop_to_content(_largest_component(binary))
    return _crop_to_content(binary)


_LIBRARY_CACHE: dict[str, np.ndarray] | None = None


def build_library() -> dict[str, np.ndarray]:
    """Render every symbol template.

    The result is cached per process: rendering ~90 matplotlib figures takes
    seconds, and the library is identical on every call.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of symbol name -> binary template image (black on white).
    """
    global _LIBRARY_CACHE
    if _LIBRARY_CACHE is None:
        _LIBRARY_CACHE = {name: _render_symbol(name) for name in SYMBOL_SOURCES}
    # Hand out a copy of the mapping so callers cannot mutate the cache. The
    # template arrays themselves are treated as read-only.
    return dict(_LIBRARY_CACHE)


# ---------------------------------------------------------------------------
# Template bank: font variants + metrics
# ---------------------------------------------------------------------------

# Font styles rendered for letters. mathtext sets variables in italic but
# function names (\log, \sin) in roman and \mathbf in bold, so the same letter
# has visibly different shapes depending on where it appears in an equation.
_ROMAN = "roman"
_ITALIC = "italic"
_BOLD = "bold"

_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_UPPERCASE = _LETTERS.upper()


@dataclass(frozen=True)
class SymbolMetrics:
    """Font metrics of one template, in units of the line's x-height.

    Measured by rendering the symbol next to a reference ``x``: the bottom of
    that ``x`` is the baseline and its height is the x-height. Expressing the
    metrics as ratios makes them independent of the font size, so they can be
    compared against a glyph in an image of any resolution.
    """

    rel_h: float  # symbol height / x-height
    rel_w: float  # symbol width / x-height
    desc: float  # (symbol bottom - baseline) / x-height; + = descends
    aspect: float  # width / height


@dataclass(frozen=True)
class Template:
    """One rendered template: a symbol name, its bitmap and its metrics."""

    name: str
    style: str
    image: np.ndarray
    metrics: SymbolMetrics | None


def _components(binary: np.ndarray, min_area: int = 5) -> list[tuple[int, int, int, int, int]]:
    """Connected components of a binary image as ``(x0, y0, w, h, area)``."""
    fg = (binary < 128).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    return [
        (int(stats[i][0]), int(stats[i][1]), int(stats[i][2]), int(stats[i][3]), int(stats[i][4]))
        for i in range(1, n)
        if stats[i][4] >= min_area
    ]


def _measure(
    binary: np.ndarray,
    parts: list[tuple[int, int, int, int, int]],
    baseline: float,
    x_height: float,
) -> tuple[np.ndarray, SymbolMetrics]:
    """Crop ``parts`` out of ``binary`` and measure them against a baseline."""
    x0 = min(p[0] for p in parts)
    y0 = min(p[1] for p in parts)
    x1 = max(p[0] + p[2] for p in parts)
    y1 = max(p[1] + p[3] for p in parts)
    image = binary[y0:y1, x0:x1]
    h = max(1, y1 - y0)
    w = max(1, x1 - x0)
    metrics = SymbolMetrics(
        rel_h=h / x_height,
        rel_w=w / x_height,
        desc=(y1 - baseline) / x_height,
        aspect=w / h,
    )
    return image, metrics


def _render_measured(source: str) -> tuple[np.ndarray, SymbolMetrics] | None:
    """Render ``source`` beside a reference ``x`` and measure the symbol.

    The reference ``x`` fixes the baseline (its bottom edge) and the x-height
    (its height); everything to its left is the symbol being measured.
    Returns None if the render produced no symbol component.
    """
    binary = _render_latex(source + r"\;x")
    parts = _components(binary)
    if len(parts) < 2:
        return None
    reference = max(parts, key=lambda p: p[0])  # the rightmost component is 'x'
    symbol_parts = [p for p in parts if p is not reference]
    if not symbol_parts:
        return None
    baseline = float(reference[1] + reference[3])
    x_height = float(reference[3])
    if x_height <= 0:
        return None
    return _measure(binary, symbol_parts, baseline, x_height)


def _render_measured_accent(source: str) -> tuple[np.ndarray, SymbolMetrics] | None:
    """Measure an accent from its own ``\\hat{x}``-style render.

    The base letter *is* the reference here: it is the largest component, and
    the accent is everything above it.
    """
    binary = _render_latex(source)
    parts = _components(binary)
    if len(parts) < 2:
        return None
    reference = max(parts, key=lambda p: p[4])
    accent_parts = [p for p in parts if p is not reference and p[1] < reference[1]]
    if not accent_parts:
        return None
    baseline = float(reference[1] + reference[3])
    x_height = float(reference[3])
    return _measure(binary, accent_parts, baseline, x_height)


def _variant_sources(name: str) -> list[tuple[str, str]]:
    """Return the ``(style, mathtext source)`` pairs to render for ``name``."""
    source = SYMBOL_SOURCES[name]
    if name in _LETTERS:
        # Italic (a variable) and roman (inside a function name like \log).
        return [(_ITALIC, source), (_ROMAN, rf"\mathrm{{{source}}}")]
    if name in _UPPERCASE:
        # Uppercase also appears in bold via \mathbf (e.g. \mathbf{F}).
        return [
            (_ITALIC, source),
            (_ROMAN, rf"\mathrm{{{source}}}"),
            (_BOLD, rf"\mathbf{{{source}}}"),
        ]
    return [(_ITALIC, source)]


_BANK_CACHE: list[Template] | None = None


def build_bank() -> list[Template]:
    """Render every template variant with its font metrics.

    Each symbol contributes one template per font style it can appear in (see
    ``_variant_sources``), measured against a reference ``x`` so the recognizer
    can score a glyph on size and baseline position as well as shape.

    Symbols whose metrics cannot be measured (the ``\\sqrt`` radical, which is
    detected structurally rather than matched) get ``metrics=None`` and are
    scored on shape alone.

    Returns
    -------
    list[Template]
        All rendered templates, cached per process.
    """
    global _BANK_CACHE
    if _BANK_CACHE is not None:
        return list(_BANK_CACHE)

    bank: list[Template] = []
    for name in SYMBOL_SOURCES:
        if name == "\\sqrt":
            bank.append(Template(name, _ITALIC, _render_symbol(name), None))
            continue
        if name in ("\\hat", "\\bar", "\\vec"):
            measured = _render_measured_accent(SYMBOL_SOURCES[name])
            if measured is not None:
                bank.append(Template(name, _ITALIC, measured[0], measured[1]))
            continue
        for style, source in _variant_sources(name):
            measured = _render_measured(source)
            if measured is None:
                continue
            bank.append(Template(name, style, measured[0], measured[1]))

    _BANK_CACHE = bank
    return list(bank)
