"""Own-code OCR pipeline: LaTeX equation image -> LaTeX string.

This is the hand-written recognition approach (no external AI). It converts a
preprocessed LaTeX equation image into a LaTeX string using:

1. **Glyph segmentation** — connected components on the binarized image.
2. **Symbol classification** — template matching against a symbol library
   rendered with matplotlib mathtext (see ``symbols.py``), optionally boosted
   by a small CNN (torch) trained on rendered symbols.
3. **Structural reconstruction** — a recursive layout parser that handles
   fractions, square roots, accents, sub/superscripts, limit symbols
   (integral/sum/product/limit) and multi-letter function names.

The symbol library and any CNN are built from rendered symbols only; the
pipeline never trains on the test split (data-leakage safe).

Public API
----------
- ``recognize(image_path) -> str`` : full pipeline entry point.
- ``Recognizer`` : object-oriented wrapper with optional CNN boost.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from preprocess import preprocess
from symbols import FUNCTION_NAMES, LIMIT_SYMBOLS, build_library

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fixed size used to normalize glyphs and templates before comparison.
MATCH_SIZE = 32

# Minimum connected-component area to keep (removes noise specks).
MIN_AREA = 5

# Fraction-bar geometry thresholds (relative to image height H).
FRAC_MIN_ASPECT = 8.0
FRAC_MAX_HEIGHT_FRAC = 0.2
FRAC_MIN_WIDTH_FRAC = 0.5  # fraction bar must be >= this * max component width

# Script (sub/superscript) vertical tolerance, as a fraction of image height.
SCRIPT_TOL_FRAC = 0.18

# Accent detection: a script whose x-center overlaps the base is an accent.
ACCENT_OVERLAP_FRAC = 0.0

# Lowercase letters (used for function-name grouping).
LETTERS = set("abcdefghijklmnopqrstuvwxyz")


# ---------------------------------------------------------------------------
# Glyph container
# ---------------------------------------------------------------------------


class Glyph:
    """A single connected component (glyph) with its bounding box and image."""

    __slots__ = ("x0", "y0", "w", "h", "area", "image", "cx", "cy", "x1", "y1",
                 "symbol", "conf")

    def __init__(self, x0, y0, w, h, area, image):
        self.x0 = x0
        self.y0 = y0
        self.w = w
        self.h = h
        self.area = area
        self.image = image
        self.cx = x0 + w / 2.0
        self.cy = y0 + h / 2.0
        self.x1 = x0 + w
        self.y1 = y0 + h
        self.symbol = None
        self.conf = 0.0


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def _segment(binary: np.ndarray) -> list[Glyph]:
    """Split a binary image into connected components (glyphs)."""
    fg = (binary < 128).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    comps: list[Glyph] = []
    for i in range(1, n):
        x0, y0, w, h, area = stats[i]
        if area < MIN_AREA:
            continue
        img = binary[y0 : y0 + h, x0 : x0 + w]
        comps.append(Glyph(x0, y0, w, h, area, img))
    return comps


# ---------------------------------------------------------------------------
# Template matching
# ---------------------------------------------------------------------------


def _resize_to_square(img: np.ndarray, size: int = MATCH_SIZE) -> np.ndarray:
    """Resize a binary image to a ``size x size`` canvas preserving aspect."""
    h, w = img.shape
    if h == 0 or w == 0:
        return np.full((size, size), 255, dtype=np.uint8)
    scale = size / float(max(h, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    _, resized = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY)
    canvas = np.full((size, size), 255, dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two binary images' foreground pixels."""
    fa = a < 128
    fb = b < 128
    inter = int(np.logical_and(fa, fb).sum())
    union = int(np.logical_or(fa, fb).sum())
    return inter / union if union else 0.0


def _classify_template(glyph: Glyph, library: dict[str, np.ndarray]) -> tuple[str, float]:
    """Classify a glyph by template matching (IoU over resized images)."""
    g = _resize_to_square(glyph.image)
    best_name = None
    best_score = -1.0
    for name, tmpl in library.items():
        t = _resize_to_square(tmpl)
        score = _iou(g, t)
        if score > best_score:
            best_name = name
            best_score = score
    return best_name, best_score


# ---------------------------------------------------------------------------
# Optional CNN boost
# ---------------------------------------------------------------------------


class SymbolCNN:
    """A small CNN classifier for glyphs, trained on rendered symbols.

    This is an optional boost on top of template matching. It is trained only
    on symbols rendered from the library (never on the test split). If no
    trained model is available, the recognizer falls back to template matching.
    """

    def __init__(self, num_classes: int, class_names: list[str]):
        import torch
        import torch.nn as nn

        self.class_names = class_names
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        ).to(self.device)
        self.net.eval()

    def _tensor(self, glyph: Glyph) -> "torch.Tensor":
        import torch

        img = _resize_to_square(glyph.image)
        arr = (img < 128).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)

    def classify(self, glyph: Glyph) -> tuple[str, float]:
        import torch

        with torch.no_grad():
            logits = self.net(self._tensor(glyph))
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            conf = float(probs[idx].item())
        return self.class_names[idx], conf


def train_cnn(
    library: dict[str, np.ndarray],
    epochs: int = 12,
    seed: int = 0,
) -> SymbolCNN:
    """Train a small CNN on rendered symbol templates.

    Parameters
    ----------
    library : dict[str, np.ndarray]
        Symbol library from ``symbols.build_library()``.
    epochs : int
        Number of training epochs.
    seed : int
        Deterministic seed for reproducibility.

    Returns
    -------
    SymbolCNN
        A trained classifier over the library's symbol names.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    names = list(library.keys())
    name_to_idx = {name: i for i, name in enumerate(names)}
    cnn = SymbolCNN(len(names), names)

    # Build a small augmented dataset from each template.
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for name, tmpl in library.items():
        base = _resize_to_square(tmpl)
        for shift in range(4):
            xs.append(np.roll(base, shift, axis=1))
            ys.append(name_to_idx[name])
    X = np.stack(xs).astype(np.float32)
    X = (X < 128).astype(np.float32)
    Y = np.array(ys, dtype=np.int64)

    optimizer = torch.optim.Adam(cnn.net.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    cnn.net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i : i + 64]
            xb = torch.from_numpy(X[idx]).unsqueeze(1).to(cnn.device)
            yb = torch.from_numpy(Y[idx]).to(cnn.device)
            optimizer.zero_grad()
            out = cnn.net(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
    cnn.net.eval()
    return cnn


# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------


def _is_hbar(c: Glyph, H: int) -> bool:
    """True if the glyph is a horizontal bar (wide, short)."""
    return c.h > 0 and c.w / c.h > 3.0 and c.h < 0.25 * H


def _merge_equals(comps: list[Glyph], H: int) -> list[Glyph]:
    """Merge two vertically-adjacent horizontal bars into a single '=' glyph."""
    bars = [c for c in comps if _is_hbar(c, H)]
    merged: set[int] = set()
    result: list[Glyph] = []
    for i, a in enumerate(bars):
        if id(a) in merged:
            continue
        for b in bars[i + 1 :]:
            if id(b) in merged:
                continue
            x_overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
            if x_overlap <= 0:
                continue
            x_union = max(a.x1, b.x1) - min(a.x0, b.x0)
            if x_union <= 0 or x_overlap / x_union < 0.5:
                continue
            if abs(a.cy - b.cy) > 0.3 * H:
                continue
            # Merge a and b into one '=' glyph.
            x0 = min(a.x0, b.x0)
            y0 = min(a.y0, b.y0)
            x1 = max(a.x1, b.x1)
            y1 = max(a.y1, b.y1)
            img = np.full((y1 - y0, x1 - x0), 255, dtype=np.uint8)
            img[a.y0 - y0 : a.y1 - y0, a.x0 - x0 : a.x1 - x0] = a.image
            img[b.y0 - y0 : b.y1 - y0, b.x0 - x0 : b.x1 - x0] = b.image
            merged.add(id(a))
            merged.add(id(b))
            eq = Glyph(x0, y0, x1 - x0, y1 - y0, a.area + b.area, img)
            eq.symbol = "="
            eq.conf = 1.0
            result.append(eq)
            break
    for c in comps:
        if id(c) not in merged:
            result.append(c)
    return result


def _find_fraction(
    comps: list[Glyph], H: int
) -> tuple[Glyph, list[Glyph], list[Glyph]] | None:
    """Find a fraction bar and split components into numerator/denominator."""
    if not comps:
        return None
    max_w = max(c.w for c in comps)
    band = 0.1 * H
    for c in comps:
        if c.h <= 0 or c.w / c.h < FRAC_MIN_ASPECT:
            continue
        if c.h > FRAC_MAX_HEIGHT_FRAC * H:
            continue
        if c.w < FRAC_MIN_WIDTH_FRAC * max_w:
            continue
        above = [x for x in comps if x.cy < c.cy - band]
        below = [x for x in comps if x.cy > c.cy + band]
        if above and below:
            return c, above, below
    return None


def _find_sqrt(
    comps: list[Glyph], H: int
) -> tuple[Glyph, list[Glyph], list[Glyph], list[Glyph]] | None:
    """Find a sqrt radical and split into (radical, radicand, index, rest)."""
    for c in comps:
        if c.symbol != "\\sqrt":
            continue
        # Radicand: components to the right of the radical sign and within its
        # vertical span. The radical sign is roughly as wide as it is tall.
        sign_w = 0.4 * c.h
        radicand = [
            x for x in comps
            if x is not c
            and x.x0 > c.x0 + sign_w
            and c.y0 <= x.cy <= c.y1
        ]
        # Index (e.g. \sqrt[3]): small component in the upper-left of radical.
        index = [
            x for x in comps
            if x is not c
            and x.x0 < c.x0 + sign_w
            and x.y0 < c.y0 + 0.5 * c.h
        ]
        rest = [x for x in comps if x is not c and x not in radicand and x not in index]
        return c, radicand, index, rest
    return None


def _estimate_baseline(comps: list[Glyph], H: int) -> float:
    """Estimate the main baseline (center y) using an area-weighted median.

    Very tall structural symbols (integrals, sums, radicals, delimiters) are
    excluded because they distort the baseline.
    """
    cands = [c for c in comps if c.h < 0.6 * H and c.area >= MIN_AREA]
    if not cands:
        return H / 2.0
    ordered = sorted(cands, key=lambda c: c.cy)
    total = sum(c.area for c in ordered)
    acc = 0
    for c in ordered:
        acc += c.area
        if acc >= total / 2.0:
            return float(c.cy)
    return float(ordered[-1].cy)


def _is_script(c: Glyph, baseline: float, H: int) -> bool:
    """True if the glyph is a sub/superscript (off the main baseline)."""
    tol = SCRIPT_TOL_FRAC * H
    return abs(c.cy - baseline) > tol


def _find_accent(m: Glyph, scripts: list[Glyph]) -> str | None:
    """Return the accent symbol (\\hat/\\bar/\\vec) over a base glyph, if any."""
    for s in scripts:
        if s.symbol not in ("\\hat", "\\bar", "\\vec"):
            continue
        if s.cx < m.x0 or s.cx > m.x1:
            continue
        if s.cy < m.cy - 0.1 * m.h:
            return s.symbol
    return None


def _find_accent_glyph(m: Glyph, scripts: list[Glyph]) -> Glyph | None:
    """Return the accent glyph (\\hat/\\bar/\\vec) over a base glyph, if any."""
    for s in scripts:
        if s.symbol not in ("\\hat", "\\bar", "\\vec"):
            continue
        if s.cx < m.x0 or s.cx > m.x1:
            continue
        if s.cy < m.cy - 0.1 * m.h:
            return s
    return None


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def _script(prefix: str, content: str) -> str:
    """Format a sub/superscript, omitting braces for a single character."""
    if len(content) == 1 and content not in "\\":
        return prefix + content
    return prefix + "{" + content + "}"


def _group_functions(tokens: list[str]) -> str:
    """Merge consecutive plain-letter tokens into known function names."""
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        j = i
        letters: list[str] = []
        while j < n and tokens[j] in LETTERS:
            letters.append(tokens[j])
            j += 1
        if len(letters) >= 2:
            name = "".join(letters)
            if name in FUNCTION_NAMES:
                out.append("\\" + name)
                i = j
                continue
        out.append(tokens[i])
        i += 1
    return "".join(out)


def _parse_mainline(comps: list[Glyph], H: int, W: int) -> str:
    """Parse a flat sequence of glyphs into a LaTeX string."""
    if not comps:
        return ""
    # Limit symbols (integral/sum/product/limit) are always main-line glyphs
    # even though they are tall; estimate the baseline from the rest.
    limit_main = [c for c in comps if c.symbol in LIMIT_SYMBOLS]
    others = [c for c in comps if c.symbol not in LIMIT_SYMBOLS]
    baseline = _estimate_baseline(others, H) if others else H / 2.0
    main = list(limit_main) + [c for c in others if not _is_script(c, baseline, H)]
    scripts = [c for c in others if _is_script(c, baseline, H)]
    main.sort(key=lambda c: c.x0)

    # Assign each script to the nearest main glyph.
    assigned: dict[int, list[Glyph]] = {id(m): [] for m in main}
    for s in scripts:
        best = None
        best_d = float("inf")
        for m in main:
            d = abs(s.cx - m.cx)
            if d < best_d:
                best_d = d
                best = m
        if best is not None:
            assigned[id(best)].append(s)

    tokens: list[str] = []
    for m in main:
        sym = m.symbol
        my_scripts = assigned[id(m)]
        if m.symbol in LIMIT_SYMBOLS:
            # Limits: everything above is superscript, below is subscript.
            sup = [s for s in my_scripts if s.cy < m.cy - 0.1 * H]
            sub = [s for s in my_scripts if s.cy > m.cy + 0.1 * H]
            sup.sort(key=lambda s: s.x0)
            sub.sort(key=lambda s: s.x0)
            if sub:
                sym += _script("_", _parse(sub, H, W))
            if sup:
                sym += _script("^", _parse(sup, H, W))
        else:
            accent = _find_accent(m, my_scripts)
            if accent is not None:
                sym = accent + "{" + sym + "}"
                # The accent glyph is consumed by the accent; exclude it from
                # the sub/superscript lists so it is not double-counted.
                accent_glyph = _find_accent_glyph(m, my_scripts)
                if accent_glyph is not None:
                    my_scripts = [s for s in my_scripts if s is not accent_glyph]
            sup = [s for s in my_scripts if s.cy < m.cy - 0.1 * H]
            sub = [s for s in my_scripts if s.cy > m.cy + 0.1 * H]
            sup.sort(key=lambda s: s.x0)
            sub.sort(key=lambda s: s.x0)
            if sub:
                sym += _script("_", _parse(sub, H, W))
            if sup:
                sym += _script("^", _parse(sup, H, W))
        tokens.append(sym)

    return _group_functions(tokens)


def _parse(comps: list[Glyph], H: int, W: int) -> str:
    """Recursive layout parser over a set of glyphs."""
    if not comps:
        return ""
    comps = _merge_equals(comps, H)

    frac = _find_fraction(comps, H)
    if frac is not None:
        bar, above, below = frac
        rest = [c for c in comps if c is not bar and c not in above and c not in below]
        left = [c for c in rest if c.x1 <= bar.x0]
        right = [c for c in rest if c.x0 >= bar.x1]
        frac_str = (
            r"\frac{" + _parse(above, H, W) + r"}{" + _parse(below, H, W) + r"}"
        )
        return _parse(left, H, W) + frac_str + _parse(right, H, W)

    sqrt = _find_sqrt(comps, H)
    if sqrt is not None:
        radical, radicand, index, rest = sqrt
        inner = _parse(radicand, H, W)
        if index:
            idx_str = _parse(index, H, W)
            return r"\sqrt[" + idx_str + r"]{" + inner + r"}" + _parse(rest, H, W)
        return r"\sqrt{" + inner + r"}" + _parse(rest, H, W)

    return _parse_mainline(comps, H, W)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _fix_classification(comps: list[Glyph], H: int, W: int) -> None:
    """Correct known template-matching failures using structural cues.

    - A tall, narrow glyph that matches a bracket is almost certainly an
      integral sign (brackets do not appear in the ground-truth vocabulary).
    - A very large, wide glyph is a square-root radical (radical + vinculum).
    """
    total_area = sum(c.area for c in comps) or 1
    for c in comps:
        if c.h > 0.5 * H and c.w / c.h < 0.5 and c.symbol in ("[", "]"):
            c.symbol = "\\int"
            c.conf = 1.0
        elif c.area > 0.3 * total_area and c.w > 0.7 * W and c.h > 0.7 * H:
            c.symbol = "\\sqrt"
            c.conf = 1.0


class Recognizer:
    """Own-code OCR recognizer with optional CNN boost.

    Parameters
    ----------
    cnn : SymbolCNN | None
        Optional trained CNN used to boost classification. If None, only
        template matching is used.
    """

    def __init__(self, cnn: SymbolCNN | None = None):
        self.library = build_library()
        self.cnn = cnn

    def _classify(self, glyph: Glyph) -> tuple[str, float]:
        name, conf = _classify_template(glyph, self.library)
        if self.cnn is not None:
            cnn_name, cnn_conf = self.cnn.classify(glyph)
            if cnn_conf > conf:
                name, conf = cnn_name, cnn_conf
        return name, conf

    def _recognize_binary(self, binary: np.ndarray) -> str:
        H, W = binary.shape
        comps = _segment(binary)
        for c in comps:
            c.symbol, c.conf = self._classify(c)
        _fix_classification(comps, H, W)
        return _parse(comps, H, W)

    def recognize(self, image_path: str) -> str:
        """Convert a LaTeX equation image into a LaTeX string."""
        binary = preprocess(image_path)
        return self._recognize_binary(binary)


def recognize(image_path: str) -> str:
    """Convert a LaTeX equation image into a LaTeX string (template matching)."""
    return Recognizer().recognize(image_path)


def recognize_with_cnn(image_path: str, cnn: SymbolCNN) -> str:
    """Recognize using template matching boosted by a trained CNN."""
    return Recognizer(cnn=cnn).recognize(image_path)
