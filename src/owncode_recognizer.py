"""Own-code OCR pipeline: LaTeX equation image -> LaTeX string.

This is the hand-written recognition approach (no external AI). It converts a
preprocessed LaTeX equation image into a LaTeX string using:

1. **Glyph segmentation** — connected components on the binarized image, with
   multi-part glyphs (``=``, ``i``, ``j``, ``!``, ``\\pm``) merged back
   together and glyphs that were printed touching cut apart again.
2. **Symbol classification** — every glyph is scored against a bank of
   rendered templates (see ``symbols.build_bank``) on four independent cues:
   shape overlap, blurred correlation, aspect ratio, and — once the line's
   font metrics have been estimated — glyph size and baseline position.
   Shape alone cannot separate ``.`` from ``\\cdot`` or ``o`` from ``O``;
   metrics can.
3. **Structural reconstruction** — a recursive layout parser that handles
   fractions, square roots and their indices, binomials, accents,
   sub/superscripts, limit symbols (integral/sum/product/limit) and
   multi-letter function names.
4. **LaTeX emission** — tokens are joined with the spacing conventions real
   LaTeX source uses (spaces around infix operators, thin spaces before
   differentials, none inside sub/superscripts).

The template bank is built from rendered symbols only; the pipeline never
trains on the test split (data-leakage safe).

Public API
----------
- ``recognize(image_path) -> str`` : full pipeline entry point.
- ``Recognizer`` : object-oriented wrapper with optional CNN boost.
"""

from __future__ import annotations

import cv2
import numpy as np

from preprocess import preprocess
from symbols import FUNCTION_NAMES, LIMIT_SYMBOLS, build_bank

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fixed size used to normalize glyphs and templates before comparison.
MATCH_SIZE = 32

# Minimum connected-component area to keep (removes noise specks).
MIN_AREA = 5

# Working height (pixels) the input image is normalized to before
# segmentation. Taller than the shared preprocessing default: template
# matching on 32x32 patches needs glyphs that are more than a few pixels wide.
WORK_HEIGHT = 128

# Fraction-bar geometry thresholds (relative to image height H).
FRAC_MIN_ASPECT = 4.0
FRAC_MAX_HEIGHT_FRAC = 0.2
FRAC_MIN_WIDTH_FRAC = 0.5  # fraction bar must be >= this * max component width
FRAC_MIN_WIDTH_RATIO = 1.05  # ... and wider than the glyphs it divides

# Script (sub/superscript) vertical tolerance, as a fraction of image height.
# Used only when the region is too small to estimate font metrics from.
SCRIPT_TOL_FRAC = 0.18

# The geometry of a sub/superscript relative to the line it hangs off: type
# size, how far a superscript is raised, how far a subscript drops (all in
# x-heights), and how much better the script reading must be to be believed.
SCRIPT_SCALE = 0.75
SUP_RAISE = 0.95
SUB_DROP = 0.42
SCRIPT_MARGIN = 0.07

# Lowercase letters (used for function-name grouping).
LETTERS = set("abcdefghijklmnopqrstuvwxyz")

# Accents are written above their base letter rather than on the line.
ACCENTS = ("\\hat", "\\bar", "\\vec")

# Relative weights of the four classification cues. Shape carries the most
# weight; the metric cues break ties between glyphs that look alike.
SHAPE_WEIGHT = 1.0
ASPECT_WEIGHT = 0.35
SIZE_WEIGHT = 0.50
POSITION_WEIGHT = 0.50

# Score assigned to a cue that cannot be evaluated (e.g. size for a template
# with no metrics), so such templates are neither rewarded nor punished.
NEUTRAL_PRIOR = 0.5

# How sharply a glyph is punished for differing from the height its template
# predicts at the line's scale.
SIZE_PENALTY = 3.0

# Up to this many glyphs, every glyph's own baseline vote is tried; beyond it
# the median vote is reliable on its own (see _baseline_candidates).
MAX_INDIVIDUAL_BASELINE_VOTES = 4

# Gaussian sigma used to blur glyph masks before correlation. Blurring makes
# the match tolerant of the one-pixel stroke shifts that binarization and
# rescaling introduce.
BLUR_SIGMA = 1.6

# A main-line gap wider than this many x-heights is a typeset thin space (\,).
THIN_SPACE_MIN_GAP = 0.55

# How well a run of glyphs must read as a known function name, relative to how
# well those glyphs read on their own, for the name to be accepted.
FUNCTION_MATCH_RATIO = 0.92

# How many template matches per glyph propose an x-height during estimation.
METRIC_CANDIDATES_PER_GLYPH = 3

# Splitting a touching pair of glyphs is only attempted when the blob matches
# nothing well (score out of the ~1.35 that shape + aspect can award), and is
# only kept when both halves clear SPLIT_MIN_PART_SCORE and beat the blob.
SPLIT_MAX_WHOLE_SCORE = 0.85
SPLIT_MIN_PART_SCORE = 0.85
SPLIT_MIN_GAIN = 0.05
SPLIT_CANDIDATE_ROWS = 6
SPLIT_ROUNDS = 3

# Two glyphs count as stacked (base and script) when they share this much of
# the narrower one's width, and the upper/lower one is this much shorter.
STACK_MIN_OVERLAP = 0.6
STACK_MAX_SIZE_RATIO = 0.85

# A script counts as sitting *on* a unit (rather than after it) only when it
# overlaps this much of its own width; a one-pixel overlap means nothing.
SCRIPT_MIN_OVERLAP = 0.3

# The stem of an i/j/! is a narrow vertical bar and its dot is tiny. Anything
# looser and the sigma of `\sum_{k=0}^{\infty}` counts as a stem, with the
# infinity above it glued on as its dot.
DOT_STEM_MAX_ASPECT = 0.45
DOT_MAX_HEIGHT_RATIO = 0.4

# How much smaller (or larger) than the expression around it a nested region —
# a fraction half, a script — is allowed to be set.
NESTED_MIN_SCALE = 0.45
NESTED_MAX_SCALE = 1.15


# Infix operators that take a space on either side in conventional LaTeX.
INFIX_OPERATORS = {"+", "-", "=", "\\pm", "\\cdot", "\\times", "\\to"}

# Operators that keep their spacing even inside a sub/superscript. LaTeX source
# is written `\sum_{i=1}^{n}` (tight) but `\lim_{x \to 0}` (spaced).
SCRIPT_SPACED_OPERATORS = {"\\to"}

# Operators that set their limits above and below themselves. Their limits are
# conventionally braced — `\sum_{i=1}^{n}` — where a side-set `\int_0^1` is not.
STACKED_LIMIT_SYMBOLS = {"\\sum", "\\prod", "\\lim"}


# ---------------------------------------------------------------------------
# Glyph container
# ---------------------------------------------------------------------------


class Glyph:
    """A single connected component (glyph) with its bounding box and image."""

    __slots__ = ("x0", "y0", "w", "h", "area", "image", "cx", "cy", "x1", "y1",
                 "symbol", "conf", "style", "locked", "shape_index", "base_scores")

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
        self.style = "italic"
        # Set when the symbol comes from a structural rule (a merged '=', a
        # radical) rather than from matching, so refinement leaves it alone.
        self.locked = False
        # Caches filled in by the classifier (see TemplateBank.base).
        self.shape_index = -1
        self.base_scores = None


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def _segment(binary: np.ndarray, min_area: int = MIN_AREA) -> list[Glyph]:
    """Split a binary image into connected components (glyphs)."""
    fg = (binary < 128).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    comps: list[Glyph] = []
    for i in range(1, n):
        x0, y0, w, h, area = stats[i]
        if area < min_area:
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


def _mask(img: np.ndarray) -> np.ndarray:
    """Normalized foreground mask of a glyph image, as a float square."""
    return (_resize_to_square(img) < 128).astype(np.float32)


def _fill_mask(img: np.ndarray, size: int = MATCH_SIZE) -> np.ndarray:
    """Foreground mask stretched to fill the square, ignoring aspect ratio.

    Padding a very thin glyph into a square leaves a mostly empty canvas, and
    every thin glyph then looks alike — an integral sign scores as well against
    a bold ``I`` as against itself. Stretching each glyph to fill the square
    instead compares their internal structure (the integral's S-curve against
    the ``I``'s straight stem); the aspect-ratio cue, scored separately, is
    what keeps the discarded proportions from being lost.
    """
    if img.size == 0:
        return np.zeros((size, size), dtype=np.float32)
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return (resized < 128).astype(np.float32)


def _unit(vec: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-norm copy of a vector (for correlation scoring)."""
    centered = vec - vec.mean()
    norm = float(np.linalg.norm(centered))
    return centered / norm if norm > 0 else centered


def _blurred_vector(mask: np.ndarray) -> np.ndarray:
    """Blur a glyph mask and flatten it into a correlation vector."""
    return _unit(cv2.GaussianBlur(mask, (0, 0), BLUR_SIGMA).ravel())


class TemplateBank:
    """Vectorized scorer over every rendered template variant.

    All templates are pre-rendered, pre-resized and stacked into matrices once,
    so classifying a glyph is two matrix-vector products rather than a Python
    loop over ~170 templates. That is what makes the pipeline fast enough to
    run the whole benchmark in seconds.
    """

    def __init__(self):
        self.templates = build_bank()
        masks = [_mask(t.image) for t in self.templates]
        self.flat = np.stack([m.ravel() for m in masks])
        self.areas = self.flat.sum(axis=1)
        self.blurred = np.stack(
            [_blurred_vector(_fill_mask(t.image)) for t in self.templates]
        )
        self.names = [t.name for t in self.templates]
        self.styles = [t.style for t in self.templates]
        self.aspects = np.array(
            [t.image.shape[1] / max(1, t.image.shape[0]) for t in self.templates],
            dtype=np.float32,
        )
        # Metrics are missing for structurally-detected symbols (the radical).
        self.rel_h = np.array(
            [t.metrics.rel_h if t.metrics else np.nan for t in self.templates],
            dtype=np.float32,
        )
        self.desc = np.array(
            [t.metrics.desc if t.metrics else np.nan for t in self.templates],
            dtype=np.float32,
        )
        self.has_metrics = ~np.isnan(self.rel_h)
        self._by_name: dict[str, list[int]] = {}
        for i, name in enumerate(self.names):
            self._by_name.setdefault(name, []).append(i)

    def base(self, glyph: Glyph) -> np.ndarray:
        """Shape and aspect scores for a glyph, cached on the glyph itself.

        These do not depend on the line metrics, so they are computed once per
        glyph and reused through the repeated re-scoring that metric estimation
        performs. The cache lives on the glyph, not on the bank, so it dies
        with the image rather than growing for the life of the process.
        """
        if glyph.base_scores is not None:
            return glyph.base_scores

        mask = _mask(glyph.image)
        flat = mask.ravel()
        intersection = self.flat @ flat
        union = np.maximum(self.areas + flat.sum() - intersection, 1.0)
        iou = intersection / union
        correlation = np.clip(
            self.blurred @ _blurred_vector(_fill_mask(glyph.image)), 0.0, None
        )
        shape = 0.5 * iou + 0.5 * correlation

        aspect = glyph.w / max(glyph.h, 1)
        aspect_score = np.exp(-2.0 * np.abs(np.log(aspect / self.aspects)))

        glyph.base_scores = SHAPE_WEIGHT * shape + ASPECT_WEIGHT * aspect_score
        return glyph.base_scores

    def size_bonus(self, glyph: Glyph, x_height: float) -> np.ndarray:
        """How well each template's height matches the glyph at this scale."""
        expected = np.where(self.has_metrics, self.rel_h * x_height, 1.0)
        return SIZE_WEIGHT * np.where(
            self.has_metrics,
            np.exp(-SIZE_PENALTY * np.abs(np.log(glyph.h / np.maximum(expected, 1e-6)))),
            NEUTRAL_PRIOR,
        )

    def position_bonus(
        self, glyph: Glyph, x_height: float, baseline: float
    ) -> np.ndarray:
        """How well each template's baseline position matches the glyph."""
        expected_bottom = baseline + self.desc * x_height
        return POSITION_WEIGHT * np.where(
            self.has_metrics,
            np.exp(-np.abs(glyph.y1 - expected_bottom) / (0.35 * x_height)),
            NEUTRAL_PRIOR,
        )

    def scores(
        self,
        glyph: Glyph,
        x_height: float | None = None,
        baseline: float | None = None,
    ) -> np.ndarray:
        """Score every template against ``glyph``.

        Shape and aspect are always available. Size and baseline position are
        scored only when the caller has estimated the line's x-height and
        baseline; without them those cues fall back to a neutral value.
        """
        total = self.base(glyph)
        if x_height and x_height > 0:
            total = total + self.size_bonus(glyph, x_height)
            if baseline is not None:
                total = total + self.position_bonus(glyph, x_height, baseline)
        return total

    def best(
        self,
        glyph: Glyph,
        x_height: float | None = None,
        baseline: float | None = None,
        exclude: tuple[str, ...] = (),
    ) -> tuple[str, str, float]:
        """Return the best ``(symbol, style, confidence)`` for a glyph."""
        scores = self.scores(glyph, x_height, baseline)
        if exclude:
            scores = scores.copy()
            for name in exclude:
                for i in self._by_name.get(name, ()):
                    scores[i] = -np.inf
        index = int(np.argmax(scores))
        maximum = SHAPE_WEIGHT + ASPECT_WEIGHT
        if x_height and x_height > 0:
            maximum += SIZE_WEIGHT
            if baseline is not None:
                maximum += POSITION_WEIGHT
        return self.names[index], self.styles[index], float(scores[index]) / maximum

    def index_of(self, glyph: Glyph) -> int:
        """Index of the best shape-only match (used for metric estimation)."""
        return int(np.argmax(self.scores(glyph)))

    def score_of(
        self,
        glyph: Glyph,
        name: str,
        x_height: float | None = None,
        baseline: float | None = None,
        style: str | None = None,
    ) -> float:
        """Best score for one specific symbol, ignoring all the others.

        Used for lexicon-constrained matching: "how well does this glyph read
        as the letter 'o'?", regardless of what it matched on its own.
        """
        indices = self._by_name.get(name)
        if not indices:
            return 0.0
        if style is not None:
            styled = [i for i in indices if self.styles[i] == style]
            indices = styled or indices
        scores = self.scores(glyph, x_height, baseline)
        return float(max(scores[i] for i in indices))


# ---------------------------------------------------------------------------
# Line metrics
# ---------------------------------------------------------------------------


def _metric_candidates(glyphs: list[Glyph], bank: TemplateBank) -> list[float]:
    """Plausible x-heights for a group, proposed by its best-matching shapes."""
    candidates: list[float] = []
    for glyph in glyphs:
        base = bank.base(glyph)
        for index in np.argsort(-base)[:METRIC_CANDIDATES_PER_GLYPH]:
            if bank.has_metrics[index] and bank.rel_h[index] > 0:
                candidates.append(glyph.h / float(bank.rel_h[index]))
    # Round before de-duplicating: near-identical proposals score identically
    # and only cost time.
    return sorted({round(c, 1) for c in candidates if c > 0})


def _baseline_candidates(votes: list[float]) -> list[float]:
    """Baselines worth testing, given each glyph's vote for where it lies.

    The median is the right answer for a line with several glyphs on it. For a
    handful it is the wrong shape of answer: two glyphs — a letter and its
    superscript — put the median *between* them, on a line neither of them sits
    on, and the superscript then looks no more displaced than the letter. So
    each individual vote is tried as well, and the group decides which one it
    actually stands on.
    """
    candidates = [float(np.median(votes))]
    if len(votes) <= MAX_INDIVIDUAL_BASELINE_VOTES:
        candidates.extend(votes)
    return sorted(set(candidates))


def _estimate_metrics(
    glyphs: list[Glyph], bank: TemplateBank, hint: float | None = None
) -> tuple[float | None, float | None]:
    """Estimate a glyph group's x-height and baseline, in pixels.

    The two numbers are found jointly rather than voted on independently. Each
    glyph proposes the x-heights that would explain its own size; for each
    proposal the implied baseline is read off the glyphs, and the pair that
    makes the *whole group* score highest wins. Estimating them together is
    what keeps one oversized symbol — an integral sign that matched a bold
    ``I`` — from dragging the scale of the entire line with it.
    """
    if not glyphs:
        return None, None
    candidates = _metric_candidates(glyphs, bank)
    if hint:
        # A nested region is set smaller than the expression around it, but
        # never wildly so. Scales outside that band are the artefacts of a
        # region too small to measure on its own — two glyphs cannot say
        # between themselves which of them is the superscript — so the
        # enclosing scale rules them out.
        plausible = [
            c for c in candidates
            if NESTED_MIN_SCALE * hint <= c <= NESTED_MAX_SCALE * hint
        ]
        candidates = plausible or candidates
    if not candidates:
        return None, None

    best_height: float | None = None
    best_baseline: float | None = None
    best_total = -np.inf
    for x_height in candidates:
        votes = []
        for glyph in glyphs:
            scores = bank.base(glyph) + bank.size_bonus(glyph, x_height)
            index = int(np.argmax(scores))
            if bank.has_metrics[index]:
                votes.append(glyph.y1 - float(bank.desc[index]) * x_height)
        if not votes:
            continue
        for baseline in _baseline_candidates(votes):
            # Weight each glyph by its size: a sum sign carries more evidence
            # about the line's scale than the small digits stacked around it,
            # and without the weighting a formula with many script glyphs
            # settles on the script's scale and inverts the whole layout.
            total = sum(
                glyph.h * float(np.max(bank.scores(glyph, x_height, baseline)))
                for glyph in glyphs
            )
            if total > best_total:
                best_total = total
                best_height = x_height
                best_baseline = baseline

    return best_height, best_baseline


def _classify_group(
    glyphs: list[Glyph],
    bank: TemplateBank,
    x_height: float | None = None,
    baseline: float | None = None,
    allow_accents: bool = True,
) -> None:
    """Classify a group of glyphs that share one font size and baseline.

    The parser calls this per layout region (main line, each script, each
    fraction part), because a superscript is small in absolute terms but
    full-size relative to its own group.
    """
    pending = [g for g in glyphs if not g.locked]
    if not pending:
        return
    if x_height is None and len(pending) >= 2:
        x_height, baseline = _estimate_metrics(pending, bank)
    exclude = () if allow_accents else ACCENTS
    for glyph in pending:
        glyph.symbol, glyph.style, glyph.conf = bank.best(
            glyph, x_height, baseline, exclude
        )


# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------


def _is_hbar(c: Glyph, H: int) -> bool:
    """True if the glyph is a horizontal bar (wide, short)."""
    return c.h > 0 and c.w / c.h > 3.0 and c.h < 0.25 * H


def _merge_glyphs(a: Glyph, b: Glyph, symbol: str | None = None) -> Glyph:
    """Combine two glyphs into one, pasting both images onto a shared canvas."""
    x0 = min(a.x0, b.x0)
    y0 = min(a.y0, b.y0)
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    img = np.full((y1 - y0, x1 - x0), 255, dtype=np.uint8)
    for part in (a, b):
        region = img[part.y0 - y0 : part.y1 - y0, part.x0 - x0 : part.x1 - x0]
        np.minimum(region, part.image, out=region)
    merged = Glyph(x0, y0, x1 - x0, y1 - y0, a.area + b.area, img)
    if symbol is not None:
        merged.symbol = symbol
        merged.conf = 1.0
        merged.locked = True
    return merged


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
            merged.add(id(a))
            merged.add(id(b))
            result.append(_merge_glyphs(a, b, symbol="="))
            break
    for c in comps:
        if id(c) not in merged:
            result.append(c)
    return result


def _merge_dots(comps: list[Glyph]) -> list[Glyph]:
    """Merge the loose dot of an 'i', 'j' or '!' back onto its stem.

    Segmentation splits these letters into two components; left alone, the dot
    is read as an accent, a stray superscript or a comma. An accent
    (hat/bar/vec) is wide relative to its base, whereas a dot is small, roughly
    square, and sits directly over — or, for ``!``, directly under — a narrow
    stem, which is how the two are told apart here.
    """
    used: set[int] = set()
    result: list[Glyph] = []
    for base in comps:
        if id(base) in used:
            continue
        if base.h <= 0 or base.w / base.h > DOT_STEM_MAX_ASPECT:
            continue  # a stem is a narrow vertical bar, not merely tallish
        best = None
        for dot in comps:
            if dot is base or id(dot) in used:
                continue
            if not 0.4 <= dot.w / max(dot.h, 1) <= 2.5:
                continue  # a dot is roughly square; a fraction bar is not
            if dot.w > 1.15 * base.w or dot.h > DOT_MAX_HEIGHT_RATIO * base.h:
                continue  # too big to be a dot
            if dot.area > 0.45 * base.area:
                continue
            if dot.y1 <= base.y0:
                gap = base.y0 - dot.y1  # tittle above an i/j
            elif dot.y0 >= base.y1:
                gap = dot.y0 - base.y1  # point below the bar of a !
            else:
                continue  # overlaps the stem: part of some other glyph
            if gap > 0.6 * base.h:
                continue  # too far away: a script, not a dot
            if dot.cx < base.x0 - 0.3 * base.w or dot.cx > base.x1 + 0.3 * base.w:
                continue  # must be centered on the stem
            if best is None or abs(dot.cy - base.cy) < abs(best.cy - base.cy):
                best = dot
        if best is not None:
            used.add(id(base))
            used.add(id(best))
            result.append(_merge_glyphs(base, best))
    for c in comps:
        if id(c) not in used:
            result.append(c)
    return result


def _sub_glyph(glyph: Glyph, row0: int, row1: int) -> Glyph | None:
    """Cut rows ``[row0, row1)`` out of a glyph, cropped to their content."""
    band = glyph.image[row0:row1]
    ink = band < 128
    if not ink.any():
        return None
    rows = np.flatnonzero(ink.any(axis=1))
    cols = np.flatnonzero(ink.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    image = band[y0:y1, x0:x1]
    return Glyph(
        glyph.x0 + x0,
        glyph.y0 + row0 + y0,
        x1 - x0,
        y1 - y0,
        int((image < 128).sum()),
        image,
    )


def _try_split(glyph: Glyph, bank: TemplateBank) -> list[Glyph] | None:
    """Split a glyph that is really two glyphs printed touching.

    ``\\sum_{i=1}^{n}`` sets its limits hard against the operator, and after
    binarization the sigma and the ``n`` above it are frequently one connected
    component that matches nothing in the bank. Rather than guess from the
    shape alone, each near-empty row is tried as a cut and the split is kept
    only when both halves are recognized *better* than the blob was — the
    classifier decides where the glyph ends.
    """
    whole = float(np.max(bank.base(glyph)))
    if whole >= SPLIT_MAX_WHOLE_SCORE or glyph.h < 8:
        return None

    ink = (glyph.image < 128).sum(axis=1).astype(np.float32)
    low, high = int(0.2 * glyph.h), int(0.8 * glyph.h)
    if high <= low:
        return None
    candidates = sorted(range(low, high), key=lambda r: ink[r])[:SPLIT_CANDIDATE_ROWS]

    best: list[Glyph] | None = None
    best_score = whole + SPLIT_MIN_GAIN
    for row in candidates:
        top = _sub_glyph(glyph, 0, row)
        bottom = _sub_glyph(glyph, row, glyph.h)
        if top is None or bottom is None:
            continue
        if min(top.h, bottom.h) < 0.15 * glyph.h:
            continue
        scores = [float(np.max(bank.base(part))) for part in (top, bottom)]
        if min(scores) < SPLIT_MIN_PART_SCORE:
            continue
        combined = min(scores)
        if combined > best_score:
            best_score = combined
            best = [top, bottom]
    return best


def _split_merged(comps: list[Glyph], bank: TemplateBank) -> list[Glyph]:
    """Repeatedly split glyphs that turn out to be two touching glyphs."""
    result = list(comps)
    for _ in range(SPLIT_ROUNDS):
        changed = False
        expanded: list[Glyph] = []
        for glyph in result:
            parts = _try_split(glyph, bank)
            if parts:
                expanded.extend(parts)
                changed = True
            else:
                expanded.append(glyph)
        result = expanded
        if not changed:
            break
    return result


def _merge_signs(comps: list[Glyph]) -> list[Glyph]:
    """Rejoin a ``\\pm`` sign, whose bar and cross are separate components.

    The merged blob is left unclassified so the template bank identifies it
    normally; without the merge the bar below the cross reads as a subscript.
    """
    used: set[int] = set()
    result: list[Glyph] = []
    for upper in comps:
        if id(upper) in used or upper.h <= 0:
            continue
        if not 0.7 <= upper.w / upper.h <= 1.4:
            continue  # the cross of a plus sign is roughly square
        for lower in comps:
            if lower is upper or id(lower) in used or lower.h <= 0:
                continue
            if lower.w / lower.h < 3.0:
                continue  # the bar underneath is flat and wide
            if not 0.8 <= lower.w / max(upper.w, 1) <= 1.25:
                continue  # the bar of a plus-minus matches the cross's width
            gap = lower.y0 - upper.y1
            if gap < 0 or gap > 0.3 * upper.h:
                continue  # further apart: a numerator over a fraction bar
            overlap = min(upper.x1, lower.x1) - max(upper.x0, lower.x0)
            if overlap < 0.7 * min(upper.w, lower.w):
                continue
            used.add(id(upper))
            used.add(id(lower))
            result.append(_merge_glyphs(upper, lower))
            break
    for c in comps:
        if id(c) not in used:
            result.append(c)
    return result


def _find_fraction(
    comps: list[Glyph], H: int
) -> tuple[Glyph, list[Glyph], list[Glyph]] | None:
    """Find a fraction bar and split components into numerator/denominator.

    Only glyphs that sit *over or under the bar itself* belong to the fraction:
    in ``\\frac{d}{dx} \\left( x^2 \\right)`` the exponent 2 is higher than the
    bar but well to its right, and pulling it into the numerator would corrupt
    both halves.
    """
    if not comps:
        return None
    max_w = max(c.w for c in comps)
    band = 0.1 * H
    bars = [
        c
        for c in comps
        if c.h > 0
        and c.w / c.h >= FRAC_MIN_ASPECT
        and c.h <= FRAC_MAX_HEIGHT_FRAC * H
        and c.w >= FRAC_MIN_WIDTH_FRAC * max_w
        and not c.locked
    ]
    # Widest bar first: in a nested fraction that is the outer one, and
    # splitting there first keeps the recursion well-formed.
    for c in sorted(bars, key=lambda b: -b.w):
        span = 0.15 * c.w  # tolerance for glyphs that overhang the bar slightly
        over_bar = [
            x for x in comps
            if x is not c and c.x0 - span <= x.cx <= c.x1 + span
        ]
        above = [x for x in over_bar if x.cy < c.cy - band]
        below = [x for x in over_bar if x.cy > c.cy + band]
        if not (above and below):
            continue
        # A fraction bar is drawn at least as wide as what it divides, which is
        # how it is told apart from a minus sign that happens to have glyphs
        # above and below it elsewhere in the expression.
        widest = max(x.w for x in above + below)
        if c.w < FRAC_MIN_WIDTH_RATIO * widest:
            continue
        return c, above, below
    return None


def _find_binom(
    comps: list[Glyph],
) -> tuple[Glyph, Glyph, list[Glyph], list[Glyph], list[Glyph]] | None:
    """Find a binomial coefficient: two glyphs stacked inside parentheses.

    ``\\binom{n}{k}`` looks exactly like a fraction without its bar, so it is
    recognized the same way — by the stacking — with the delimiters standing in
    for the missing rule. Ordinary parentheses hold glyphs that sit side by
    side on one line, which is what keeps ``\\sin(x)`` out of here.
    """
    opens = [c for c in comps if c.symbol == "("]
    closes = [c for c in comps if c.symbol == ")"]
    for opening in opens:
        for closing in closes:
            if closing.x0 <= opening.x1:
                continue
            inside = [
                c for c in comps
                if c is not opening and c is not closing
                and opening.x1 <= c.cx <= closing.x0
            ]
            if len(inside) < 2:
                continue
            ordered = sorted(inside, key=lambda c: c.cy)
            for cut in range(1, len(ordered)):
                upper, lower = ordered[:cut], ordered[cut:]
                if max(c.y1 for c in upper) > min(c.y0 for c in lower):
                    continue  # the two halves share rows: one line, not a stack
                rest = [
                    c for c in comps
                    if c is not opening and c is not closing and c not in inside
                ]
                return opening, closing, upper, lower, rest
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
        # Index (e.g. \sqrt[3]): a small glyph tucked into the radical's upper
        # left. It must be small and sit above the radical, otherwise the terms
        # written to the left of the root (the "-b \pm" of the quadratic
        # formula) get swallowed as an index.
        index = [
            x for x in comps
            if x is not c
            and c.x0 - 0.2 * c.h <= x.x0
            and x.x1 <= c.x0 + sign_w
            and x.cy < c.y0 + 0.4 * c.h
            and x.h < 0.6 * c.h
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


def _stacked_scripts(comps: list[Glyph]) -> set[int]:
    """Ids of glyphs printed above or below another glyph rather than beside it.

    Nothing is written on top of anything else on a line of maths unless it is
    a script: ``\\sum_{i=1}^{n}`` stacks its limits over and under the sigma,
    and ``\\lim_{x \\to 0}`` tucks its condition underneath. Ordinary text never
    stacks, so a pair of glyphs that share horizontal space but no vertical
    space is a base and its script — and the smaller of the two is the script.
    Fractions and radicals are resolved before this point, so their stacking is
    already gone.
    """
    stacked: set[int] = set()
    for i, a in enumerate(comps):
        for b in comps[i + 1 :]:
            if min(a.y1, b.y1) > max(a.y0, b.y0):
                continue  # they share rows, so they sit side by side
            overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
            if overlap < STACK_MIN_OVERLAP * min(a.w, b.w):
                # A big operator sets its limits centered on itself, and the
                # stack is often wider than the operator, so the outer glyphs
                # of `\sum_{i=1}` overhang the sigma without overlapping it.
                operator = (
                    a if a.symbol in LIMIT_SYMBOLS
                    else b if b.symbol in LIMIT_SYMBOLS
                    else None
                )
                other = b if operator is a else a
                if operator is None or abs(other.cx - operator.cx) > 1.2 * operator.w:
                    continue
            # The script is the shorter glyph — height, not area, identifies it
            # when a wide subscript sits under a narrow letter, as with the `x`
            # beneath the `l` of `\lim`. It then has to be *smaller* by one
            # measure or the other: a subscript digit is nearly as tall as the
            # letter it hangs under, but nothing like as heavy.
            shorter = a if a.h <= b.h else b
            taller = b if shorter is a else a
            ratio = min(
                shorter.h / max(taller.h, 1),
                shorter.area / max(taller.area, 1),
            )
            if ratio > STACK_MAX_SIZE_RATIO:
                continue  # comparable sizes: not a base with its script
            stacked.add(id(shorter))
    return stacked


def _shape_index(glyph: Glyph, bank: TemplateBank) -> int:
    """Best shape-only template index for a glyph, computed once and cached."""
    if glyph.shape_index < 0:
        glyph.shape_index = bank.index_of(glyph)
    return glyph.shape_index


def _reads_as_script(
    glyph: Glyph, bank: TemplateBank, x_height: float, baseline: float
) -> bool:
    """Decide whether a glyph is a script by asking which reading fits better.

    Rather than thresholding how far the glyph sits from where it "should",
    both readings are scored with the same classifier: once against the line
    itself, and once against a script — smaller type, raised or dropped off the
    line. Whichever reading explains the glyph better wins. A threshold has to
    be picked to suit some average glyph; this comparison adapts to whatever
    the glyph actually is, which is what keeps the ``2`` of ``z^2`` a script
    and the ``i`` after ``\\sum_{i=1}^{n}`` on the line.
    """
    on_line = float(np.max(bank.scores(glyph, x_height, baseline)))
    script_height = SCRIPT_SCALE * x_height
    raised = float(
        np.max(bank.scores(glyph, script_height, baseline - SUP_RAISE * x_height))
    )
    dropped = float(
        np.max(bank.scores(glyph, script_height, baseline + SUB_DROP * x_height))
    )
    return max(raised, dropped) > on_line + SCRIPT_MARGIN


def _split_scripts(
    comps: list[Glyph], H: int, bank: TemplateBank | None,
    hint: float | None = None,
) -> tuple[list[Glyph], list[Glyph], float | None, float | None]:
    """Split a region into main-line glyphs and sub/superscripts.

    A glyph is a script when it sits away from where its own shape says it
    should sit. Comparing against the *predicted* position rather than a fixed
    fraction of the image height is what makes this work at every nesting
    depth: ``b`` is tall and ``,`` hangs below the baseline, yet neither is a
    script, while a superscript ``2`` is displaced from wherever a ``2``
    belongs. The fixed-height rule is kept as a fallback for regions too small
    to estimate metrics from.
    """
    limits = [c for c in comps if c.symbol in LIMIT_SYMBOLS]
    others = [c for c in comps if c.symbol not in LIMIT_SYMBOLS]
    stacked = _stacked_scripts(comps)

    metrics = None
    if bank is not None and len(others) >= 2:
        metrics = _estimate_metrics(others, bank, hint)

    if metrics is None or metrics[0] is None or metrics[1] is None:
        center = _estimate_baseline(others, H) if others else H / 2.0
        main = limits + [c for c in others if not _is_script(c, center, H)]
        scripts = [c for c in others if _is_script(c, center, H)]
        return main, scripts, None, None

    x_height, baseline = metrics
    main: list[Glyph] = []
    scripts: list[Glyph] = []
    # Two passes: the first baseline estimate is pulled off-line by the scripts
    # themselves, so re-estimate once from the main-line glyphs only.
    for _ in range(2):
        main, scripts = [], []
        for c in others:
            if id(c) in stacked:
                scripts.append(c)
                continue
            index = _shape_index(c, bank)
            if bank.names[index] in ACCENTS:
                # An accent sits exactly where its template says it should —
                # above the line — so the residual test would call it main-line.
                scripts.append(c)
                continue
            if not bank.has_metrics[index]:
                main.append(c)
                continue
            if _reads_as_script(c, bank, x_height, baseline):
                scripts.append(c)
            else:
                main.append(c)
        if not main:
            break
        refined = _estimate_metrics(main, bank, hint)
        if refined[0] is None or refined[1] is None:
            break
        x_height, baseline = refined

    if not main and not limits:
        # Every glyph looked displaced, which means the metric fit was wrong,
        # not that the region is made entirely of scripts. Scripts are only
        # ever emitted attached to a main-line glyph, so leaving the main line
        # empty would silently drop the whole region. Glyphs printed above or
        # below another glyph keep their script status even here: that came
        # from the layout, not from the metric fit that just failed.
        unstacked = [c for c in others if id(c) not in stacked]
        if unstacked:
            main = unstacked
            scripts = [c for c in others if id(c) in stacked]
        else:
            main, scripts = others, []

    return limits + main, scripts, x_height, baseline


def _find_accent(m: Glyph, scripts: list[Glyph]) -> str | None:
    """Return the accent symbol (\\hat/\\bar/\\vec) over a base glyph, if any."""
    glyph = _find_accent_glyph(m, scripts)
    return glyph.symbol if glyph is not None else None


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
# Emission
# ---------------------------------------------------------------------------


class Token:
    """One emitted LaTeX fragment, plus what the emitter needs to space it."""

    __slots__ = ("text", "kind", "x0", "x1")

    def __init__(self, text: str, kind: str = "atom", x0: float = 0.0, x1: float = 0.0):
        self.text = text
        self.kind = kind  # atom | operator | comma | open | close | limit | space
        self.x0 = x0
        self.x1 = x1


def _needs_space(left: Token, right: Token, script: bool) -> bool:
    """Decide whether two adjacent tokens are separated by a space.

    Mirrors how the expressions are written by hand: spaces around infix
    operators and stretchy delimiters, none inside sub/superscripts (LaTeX
    source says ``\\sum_{i=1}^{n}`` but ``a + b``), and always a space where
    running two tokens together would change the meaning — ``\\pi r`` must not
    collapse into the unknown command ``\\pir``.
    """
    if left.kind == "space" or right.kind == "space":
        return True
    if left.kind == "comma":
        return True
    if right.kind == "comma":
        return False
    if left.kind == "limit":
        return True
    # Only stretchy delimiters are set off by spaces: LaTeX source reads
    # `\left( x^2 \right)` but `\log(x)`.
    if left.kind == "open":
        return left.text.startswith("\\")
    if right.kind == "close":
        return right.text.startswith("\\")
    if right.kind == "open" and right.text.startswith("\\left"):
        return True
    if left.kind == "close" and left.text.startswith("\\right"):
        return True
    if left.kind == "operator" or right.kind == "operator":
        operator = left if left.kind == "operator" else right
        if script and operator.text not in SCRIPT_SPACED_OPERATORS:
            return False
        return True
    # A command ending in a letter would swallow a following letter.
    if left.text.startswith("\\") and left.text[-1].isalpha() and right.text[:1].isalpha():
        return True
    # Outside scripts, a command that follows a symbol is written detached:
    # `c \rangle`, not `c\rangle`. Inside a script the source stays tight
    # (`e^{i\pi}`), so the rule is skipped there.
    if not script and right.text.startswith("\\") and left.text[-1:].isalnum():
        return True
    return False


def _join(tokens: list[Token], script: bool = False) -> str:
    """Join tokens into LaTeX source with conventional spacing."""
    if not tokens:
        return ""
    out = [tokens[0].text]
    for previous, token in zip(tokens, tokens[1:]):
        if _needs_space(previous, token, script):
            out.append(" ")
        out.append(token.text)
    return "".join(out)


def _script(prefix: str, content: str, braces: bool = False) -> str:
    """Format a sub/superscript, omitting braces for a single character.

    ``braces`` forces them anyway, for the stacked limits of ``\\sum`` and
    ``\\prod``: those are conventionally written ``\\sum_{i=1}^{n}`` even when
    the limit is a single character, while a side-set ``\\int_0^1`` is not.
    """
    if not braces and len(content) == 1 and content not in "\\":
        return prefix + content
    return prefix + "{" + content + "}"


def _group_function_tokens(tokens: list[str]) -> list[str]:
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
        # Longest match wins, so "arcsin" is preferred over "arc" + "sin".
        matched = False
        for end in range(len(letters), 1, -1):
            name = "".join(letters[:end])
            if name in FUNCTION_NAMES:
                out.append("\\" + name)
                i += end
                matched = True
                break
        if matched:
            continue
        out.append(tokens[i])
        i += 1
    return out


def _group_functions(tokens: list[str]) -> str:
    """Merge consecutive plain-letter tokens into known function names."""
    return "".join(_group_function_tokens(tokens))


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


class _Unit:
    """A main-line glyph (or merged function name) with its attached scripts."""

    __slots__ = ("glyphs", "symbol", "style", "x0", "x1", "cx", "cy", "h", "scripts")

    def __init__(self, glyphs: list[Glyph], symbol: str, style: str = "italic"):
        self.glyphs = glyphs
        self.symbol = symbol
        self.style = style
        self.x0 = min(g.x0 for g in glyphs)
        self.x1 = max(g.x1 for g in glyphs)
        self.cx = (self.x0 + self.x1) / 2.0
        self.cy = sum(g.cy for g in glyphs) / len(glyphs)
        self.h = max(g.h for g in glyphs)
        self.scripts: list[Glyph] = []


def _function_match(
    glyphs: list[Glyph],
    bank: TemplateBank | None,
    x_height: float | None,
    baseline: float | None,
) -> str | None:
    """Return the function name these glyphs spell, if they spell one.

    Function names are set in roman while variables are italic, and a roman
    ``l`` is a bare stem that matches ``I`` or ``1`` about as well as itself.
    Rather than demand that every letter be read correctly on its own, each
    candidate name from the lexicon is scored as a whole: ``\\log`` wins if
    reading the three glyphs as l-o-g is nearly as good as reading them as
    whatever they matched individually. This is the OCR equivalent of a
    spellchecker, and it is what makes ``\\lim`` and ``\\log`` survive noise.
    """
    if bank is None or len(glyphs) < 2:
        return None
    unconstrained = [
        float(np.max(bank.scores(g, x_height, baseline))) for g in glyphs
    ]
    best_name = None
    best_score = 0.0
    for name in FUNCTION_NAMES:
        if len(name) != len(glyphs):
            continue
        scores = [
            bank.score_of(g, ch, x_height, baseline, style="roman")
            for g, ch in zip(glyphs, name)
        ]
        ratio = sum(scores) / max(sum(unconstrained), 1e-6)
        if ratio >= FUNCTION_MATCH_RATIO and ratio > best_score:
            best_score = ratio
            best_name = name
    return best_name


def _is_name_candidate(symbol: str) -> bool:
    """True if a glyph could be one letter of a function name."""
    return (
        len(symbol) == 1
        and symbol not in INFIX_OPERATORS
        and symbol not in ",.()[]{}"
    )


def _build_units(
    main: list[Glyph],
    bank: TemplateBank | None = None,
    x_height: float | None = None,
    baseline: float | None = None,
) -> list[_Unit]:
    """Group main-line glyphs into units, merging function names into one."""
    main = sorted(main, key=lambda c: c.x0)
    symbols = [c.symbol or "" for c in main]
    units: list[_Unit] = []
    i = 0
    while i < len(main):
        # A function name is a run of letter-like glyphs; try the longest run
        # first so "arcsin" is preferred over "arc" + "sin". A glyph that was
        # *misread* as punctuation still counts — a roman `l` matches `!`, `I`
        # and `1` as readily as itself, and excluding it here would deny the
        # lexicon the chance to put `\log` back together.
        j = i
        while j < len(main) and _is_name_candidate(symbols[j]):
            j += 1
        matched = False
        for end in range(j, i + 1, -1):
            run = main[i:end]
            name = "".join(symbols[i:end])
            if name in FUNCTION_NAMES:
                matched = True
            else:
                found = _function_match(run, bank, x_height, baseline)
                name = found if found else name
                matched = found is not None
            if matched:
                units.append(_Unit(run, "\\" + name, "roman"))
                i = end
                break
        if matched:
            continue
        units.append(_Unit([main[i]], symbols[i], main[i].style))
        i += 1
    return units


def _script_owner(script: Glyph, units: list[_Unit]) -> _Unit | None:
    """Find the main-line unit a script belongs to.

    A script is written after the thing it modifies, so the owner is the unit
    it overlaps horizontally (limits stacked on a big operator, an accent over
    its letter) or, failing that, the closest unit that starts before it. The
    old rule — nearest center — hands the ``1`` of ``\\int_0^1`` to the ``x``
    that follows it, because that ``x`` happens to be closer.
    """
    if not units:
        return None
    overlapping = [
        u for u in units
        if min(u.x1, script.x1) - max(u.x0, script.x0)
        > SCRIPT_MIN_OVERLAP * max(1, script.w)
    ]
    if overlapping:
        return max(
            overlapping,
            key=lambda u: min(u.x1, script.x1) - max(u.x0, script.x0),
        )
    preceding = [u for u in units if u.x0 <= script.x0]
    if preceding:
        return max(preceding, key=lambda u: u.x1)
    return min(units, key=lambda u: abs(u.cx - script.cx))


def _decorate(symbol: str, style: str) -> str:
    """Wrap a symbol in the font command its rendered style implies."""
    if style == "bold" and symbol.isalpha():
        return r"\mathbf{" + symbol + "}"
    return symbol


def _token_kind(symbol: str) -> str:
    """Classify a symbol for spacing purposes."""
    if symbol in INFIX_OPERATORS:
        return "operator"
    if symbol == ",":
        return "comma"
    if symbol in ("\\langle", "("):
        return "open"
    if symbol in ("\\rangle", ")"):
        return "close"
    return "atom"


def _parse_mainline(
    comps: list[Glyph], H: int, W: int, bank: TemplateBank | None = None,
    x_height: float | None = None,
) -> str:
    """Parse a flat sequence of glyphs into a LaTeX string."""
    tokens, x_height = _mainline_tokens(comps, H, W, bank, x_height)
    return _join(_insert_thin_spaces(tokens, x_height), script=False)


def _mainline_tokens(
    comps: list[Glyph], H: int, W: int, bank: TemplateBank | None = None,
    hint: float | None = None,
) -> tuple[list[Token], float | None]:
    """Turn a flat sequence of glyphs into spaced-but-unjoined tokens."""
    x_height = hint
    if not comps:
        return [], x_height
    # Limit symbols (integral/sum/product/limit) are always main-line glyphs
    # even though they are tall; the baseline is estimated from the rest.
    main, scripts, estimated_height, baseline = _split_scripts(comps, H, bank, hint)
    main.sort(key=lambda c: c.x0)
    if estimated_height:
        x_height = estimated_height

    if bank is not None:
        # Nothing on the main line is an accent: an accent is always printed
        # above a base letter, which puts it in the script set. Saying so keeps
        # the minus of `e^{-x}` from being read as the arrow of `\vec`.
        _classify_group(main, bank, x_height, baseline, allow_accents=False)

    units = _build_units(main, bank, x_height, baseline)

    for s in scripts:
        unit = _script_owner(s, units)
        if unit is not None:
            unit.scripts.append(s)

    tokens: list[Token] = []
    for unit in units:
        symbol = unit.symbol
        my_scripts = unit.scripts
        is_limit = symbol in LIMIT_SYMBOLS
        if not is_limit:
            accent_glyph = _find_accent_glyph(unit, my_scripts)
            if accent_glyph is not None:
                symbol = accent_glyph.symbol + "{" + symbol + "}"
                # The accent glyph is consumed by the accent; exclude it from
                # the sub/superscript lists so it is not double-counted.
                my_scripts = [s for s in my_scripts if s is not accent_glyph]
            else:
                symbol = _decorate(symbol, unit.style)
        # Everything here is already known to be a script, so it is above the
        # base or below it — there is no third option. An earlier version left
        # a dead band scaled to the whole image, and any script that fell in it
        # was dropped from the output entirely.
        sup = sorted(
            (s for s in my_scripts if s.cy < unit.cy), key=lambda s: s.x0
        )
        sub = sorted(
            (s for s in my_scripts if s.cy >= unit.cy), key=lambda s: s.x0
        )
        braces = unit.symbol in STACKED_LIMIT_SYMBOLS
        if sub:
            symbol += _script(
                "_", _parse(sub, H, W, bank, True, x_height), braces
            )
        if sup:
            symbol += _script(
                "^", _parse(sup, H, W, bank, True, x_height), braces
            )

        kind = "limit" if is_limit else _token_kind(unit.symbol)
        if symbol.startswith("\\left"):
            kind = "open"
        elif symbol.startswith("\\right"):
            kind = "close"
        if kind == "operator" and not tokens and unit.symbol == "-":
            kind = "atom"  # a leading '-' is unary: "-b", not "- b"
        tokens.append(Token(symbol, kind, unit.x0, unit.x1))

    return tokens, x_height


def _insert_thin_spaces(tokens: list[Token], x_height: float | None) -> list[Token]:
    """Insert ``\\,`` where the typeset gap between tokens is unusually wide.

    mathtext renders ``\\,`` as a visible gap, so a main-line gap much wider
    than normal letter spacing is evidence the source contained one — as in
    ``\\int x^2 \\, dx``.
    """
    if not x_height or len(tokens) < 2:
        return tokens
    out = [tokens[0]]
    for previous, token in zip(tokens, tokens[1:]):
        gap = token.x0 - previous.x1
        spaced = previous.kind in ("operator", "comma", "limit", "open") or token.kind in (
            "operator",
            "close",
        )
        if not spaced and gap > THIN_SPACE_MIN_GAP * x_height:
            out.append(Token("\\,", "space"))
        out.append(token)
    return out


def _parse(
    comps: list[Glyph],
    H: int,
    W: int,
    bank: TemplateBank | None = None,
    script: bool = False,
    hint: float | None = None,
) -> str:
    """Recursive layout parser over a set of glyphs."""
    tokens, x_height = _parse_tokens(comps, H, W, bank, script, hint)
    if not script:
        tokens = _insert_thin_spaces(tokens, x_height)
    return _join(tokens, script)


def _parse_tokens(
    comps: list[Glyph],
    H: int,
    W: int,
    bank: TemplateBank | None = None,
    script: bool = False,
    hint: float | None = None,
) -> tuple[list[Token], float | None]:
    """Recursive layout parser, returning tokens with their positions.

    Structures (fractions, radicals) collapse into a single token that keeps
    the horizontal extent of the glyphs it came from, so the spacing rules
    downstream treat ``\\frac{1}{x}`` exactly like any other atom and can still
    see the gap between it and whatever follows.
    """
    if not comps:
        return [], None
    comps = _merge_equals(comps, H)
    if bank is not None and hint is None and len(comps) >= 2:
        # Measure the region before splitting it up: a part that is too small
        # to measure on its own then falls back to the scale of the expression
        # it came out of, which is what the spacing rules need.
        hint = _estimate_metrics(comps, bank)[0]

    frac = _find_fraction(comps, H)
    if frac is not None:
        bar, above, below = frac
        rest = [c for c in comps if c is not bar and c not in above and c not in below]
        # Anything not stacked on the bar sits beside the fraction; the side is
        # decided by the glyph's center so no glyph is ever dropped.
        left = [c for c in rest if c.cx < bar.x0]
        right = [c for c in rest if c.cx >= bar.x0]
        token = Token(
            r"\frac{" + _parse(above, H, W, bank, script, hint)
            + r"}{" + _parse(below, H, W, bank, script, hint) + r"}",
            "atom", bar.x0, bar.x1,
        )
        return _splice(left, token, right, H, W, bank, script, hint)

    binom = _find_binom(comps)
    if binom is not None:
        opening, closing, upper, lower, rest = binom
        token = Token(
            r"\binom{" + _parse(upper, H, W, bank, script, hint)
            + r"}{" + _parse(lower, H, W, bank, script, hint) + r"}",
            "atom", opening.x0, closing.x1,
        )
        left = [c for c in rest if c.cx < opening.x0]
        right = [c for c in rest if c.cx >= closing.x1]
        return _splice(left, token, right, H, W, bank, script, hint)

    sqrt = _find_sqrt(comps, H)
    if sqrt is not None:
        radical, radicand, index, rest = sqrt
        inner = _parse(radicand, H, W, bank, script, hint)
        if index:
            text = r"\sqrt[" + _parse(index, H, W, bank, script, hint) + r"]{" + inner + r"}"
        else:
            text = r"\sqrt{" + inner + r"}"
        x0 = min([radical.x0] + [c.x0 for c in radicand + index])
        x1 = max([radical.x1] + [c.x1 for c in radicand + index])
        token = Token(text, "atom", x0, x1)
        left = [c for c in rest if c.cx < x0]
        right = [c for c in rest if c.cx >= x0]
        return _splice(left, token, right, H, W, bank, script, hint)

    return _mainline_tokens(comps, H, W, bank, hint)


def _splice(
    left: list[Glyph],
    token: Token,
    right: list[Glyph],
    H: int,
    W: int,
    bank: TemplateBank | None,
    script: bool,
    hint: float | None = None,
) -> tuple[list[Token], float | None]:
    """Parse the glyphs on either side of a structure and splice them around it."""
    left_tokens, left_height = _parse_tokens(left, H, W, bank, script, hint)
    right_tokens, right_height = _parse_tokens(right, H, W, bank, script, hint)
    x_height = left_height or right_height
    return left_tokens + [token] + right_tokens, x_height


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _looks_like_radical(c: Glyph, comps: list[Glyph]) -> bool:
    """True if a component has the shape of a radical sign with its vinculum.

    A radical is a wide component whose top edge is an almost unbroken
    horizontal rule and whose box covers other glyphs (the radicand) without
    touching them.
    """
    if c.h < 3 or c.w < 3 or c.w < 0.8 * c.h:
        return False
    covered = [
        x for x in comps
        if x is not c and c.x0 < x.cx < c.x1 and c.y0 <= x.cy <= c.y1
    ]
    if not covered:
        return False
    top_band = c.image[: max(1, c.h // 8)] < 128
    if top_band.size == 0:
        return False
    # The vinculum spans nearly the whole width of the component.
    return float(top_band.any(axis=0).mean()) > 0.85


def _fix_classification(comps: list[Glyph], H: int, W: int) -> None:
    """Correct known template-matching failures using structural cues.

    - A tall, narrow glyph that matches a bracket is almost certainly an
      integral sign (brackets do not appear in the ground-truth vocabulary).
    - A component whose box spans other glyphs under a full-width top rule is
      a square-root radical.
    """
    total_area = sum(c.area for c in comps) or 1
    for c in comps:
        if c.locked:
            continue
        if c.h > 0.5 * H and c.w / c.h < 0.5 and c.symbol in ("[", "]"):
            c.symbol = "\\int"
            c.conf = 1.0
            c.locked = True
        elif _looks_like_radical(c, comps) or (
            c.area > 0.3 * total_area and c.w > 0.7 * W and c.h > 0.7 * H
        ):
            c.symbol = "\\sqrt"
            c.conf = 1.0
            c.locked = True


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


class Recognizer:
    """Own-code OCR recognizer with optional CNN boost.

    Parameters
    ----------
    cnn : SymbolCNN | None
        Optional trained CNN used to boost classification. If None, only
        template matching is used.
    """

    def __init__(self, cnn: SymbolCNN | None = None):
        self.bank = TemplateBank()
        self.cnn = cnn

    def _classify(self, glyph: Glyph) -> tuple[str, float]:
        name, _style, conf = self.bank.best(glyph)
        if self.cnn is not None:
            cnn_name, cnn_conf = self.cnn.classify(glyph)
            if cnn_conf > conf:
                name, conf = cnn_name, cnn_conf
        return name, conf

    def _recognize_binary(self, binary: np.ndarray) -> str:
        H, W = binary.shape
        min_area = max(MIN_AREA, int(round(0.0006 * H * H)))
        comps = _segment(binary, min_area)
        if not comps:
            return ""
        comps = _merge_equals(comps, H)
        comps = _merge_dots(comps)
        comps = _merge_signs(comps)
        comps = _split_merged(comps, self.bank)
        # A first, shape-only pass gives every glyph a provisional identity so
        # the structural rules below have something to work with; the parser
        # then re-classifies each layout region with its own font metrics.
        for c in comps:
            c.symbol, c.style, c.conf = self.bank.best(c)
        _fix_classification(comps, H, W)
        if self.cnn is not None:
            for c in comps:
                if c.locked:
                    continue
                cnn_name, cnn_conf = self.cnn.classify(c)
                if cnn_conf > c.conf:
                    c.symbol, c.conf = cnn_name, cnn_conf
                    c.locked = True
        return _parse(comps, H, W, self.bank)

    def recognize(self, image_path: str) -> str:
        """Convert a LaTeX equation image into a LaTeX string."""
        binary = preprocess(image_path, height=WORK_HEIGHT)
        return self._recognize_binary(binary)


_SHARED: Recognizer | None = None


def _shared() -> Recognizer:
    """Return the process-wide recognizer, building the bank only once."""
    global _SHARED
    if _SHARED is None:
        _SHARED = Recognizer()
    return _SHARED


def recognize(image_path: str) -> str:
    """Convert a LaTeX equation image into a LaTeX string (template matching)."""
    return _shared().recognize(image_path)


def recognize_with_cnn(image_path: str, cnn: SymbolCNN) -> str:
    """Recognize using template matching boosted by a trained CNN."""
    return Recognizer(cnn=cnn).recognize(image_path)
