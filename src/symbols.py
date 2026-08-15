"""Symbol library / template rendering helper for the own-code OCR pipeline.

Renders a curated set of LaTeX symbols to binary images using **matplotlib
mathtext** (no LaTeX engine is installed), so the own-code recognizer can
classify segmented glyphs by template matching.

The library is built from the symbol vocabulary that appears in the
ground-truth expressions. It is rendered deterministically and does **not**
depend on the test split, so building it never leaks test data.

Public API
----------
- ``build_library() -> dict[str, np.ndarray]`` : render every symbol template.
- ``SYMBOL_SOURCES`` : name -> mathtext source mapping.
"""

from __future__ import annotations

import io

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
    "|": r"|", ",": r",", ".": r".", "\\nabla": r"\nabla",
    "\\partial": r"\partial", "\\infty": r"\infty", "\\alpha": r"\alpha",
    "\\beta": r"\beta", "\\gamma": r"\gamma", "\\pi": r"\pi",
    "\\sum": r"\sum_{}^{}", "\\prod": r"\prod_{}^{}",
    "\\int": r"\int_{}^{}",
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


def build_library() -> dict[str, np.ndarray]:
    """Render every symbol template.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of symbol name -> binary template image (black on white).
    """
    return {name: _render_symbol(name) for name in SYMBOL_SOURCES}
