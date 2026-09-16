"""Ground-truth dataset generator for the LaTeX OCR benchmark.

Renders a curated set of LaTeX strings to images using **matplotlib mathtext**
(no LaTeX engine is installed). Each sample is saved as a ``{image, latex}``
pair under ``data/``.

The dataset is split into **disjoint** ``train`` and ``test`` subsets using a
fixed deterministic seed. No image appears in both sets. This prevents data
leakage in the benchmark: OWN-001 trains only on ``train``, BENCH-001 evaluates
only on ``test``.

Public API
----------
- ``generate(data_dir, ...)``          : render the baseline corpus.
- ``generate_complex(data_dir, ...)``  : render the 100-expression stress corpus.
- ``load(split="train"|"test", ...)``  : return samples for one split.
- ``train_set`` / ``test_set``         : convenience accessors.
"""

from __future__ import annotations

import io
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Difficulty tiers. Each tier applies a distinct degradation to the rendered
# image so per-tier robustness can be measured.
TIERS = ("clean", "noisy", "low_res", "black_bg", "white_bg")

# Default number of samples per tier.
DEFAULT_PER_TIER = 20

# Default deterministic seed for the train/test split.
DEFAULT_SEED = 42

# Train/test split ratio (fraction of samples that go to train).
TRAIN_RATIO = 0.7

# Curated LaTeX expressions supported by matplotlib mathtext.
CURATED_EXPRESSIONS = [
    r"\frac{a}{b}",
    r"x^2 + y^2 = z^2",
    r"\sqrt{x}",
    r"\int_0^1 x^2 \, dx",
    r"\sum_{i=1}^{n} i",
    r"\alpha + \beta = \gamma",
    r"\pi r^2",
    r"e^{i\pi} + 1 = 0",
    r"\frac{1}{2}",
    r"\sin(x) + \cos(x)",
    r"\log(x)",
    r"\lim_{x \to 0} \frac{\sin x}{x}",
    r"\binom{n}{k}",
    r"\sqrt[3]{x}",
    r"\frac{dy}{dx}",
    r"\int e^{-x^2} \, dx",
    r"\sum_{k=0}^{\infty} \frac{1}{k!}",
    r"\prod_{i=1}^{n} i",
    r"\frac{\partial f}{\partial x}",
    r"\nabla \cdot \mathbf{F}",
    r"\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
    r"\int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi}",
    r"\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}",
    r"\frac{d}{dx} \left( x^2 \right) = 2x",
    r"\hat{x} + \bar{y}",
    r"\vec{v} = \langle a, b, c \rangle",
    r"\frac{1}{1 + e^{-x}}",
    r"\sqrt{a^2 + b^2}",
    r"\int \frac{1}{x} \, dx = \ln|x|",
    r"\lim_{n \to \infty} \left( 1 + \frac{1}{n} \right)^n = e",
]

# A separate stress corpus for the hand-written recognizer.  These are kept
# outside CURATED_EXPRESSIONS so the original benchmark and its checked-in
# manifest remain reproducible.  Every expression uses constructs represented
# by the own-code symbol/layout vocabulary, but the corpus deliberately puts
# them together in longer and more deeply nested combinations.
COMPLEX_EXAMPLES = [
    # Algebra and equations (10)
    ("algebra", r"x^2 + 3x + 2 = 0"),
    ("algebra", r"a^2 + b^2 = c^2"),
    ("algebra", r"2x + 5 = 17"),
    ("algebra", r"p + q = r"),
    ("algebra", r"m^2 - n^2 = (m - n)(m + n)"),
    ("algebra", r"x^3 - 2x + 1 = 0"),
    ("algebra", r"ax + by = c"),
    ("algebra", r"\alpha^2 + \beta^2 = \gamma^2"),
    ("algebra", r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}"),
    ("algebra", r"\frac{a + b}{c + d} = \frac{e}{f}"),
    # Fractions, radicals, and binomials (10)
    ("fractions", r"\frac{x + 1}{x - 1}"),
    ("fractions", r"\frac{1}{1 + \frac{1}{x}}"),
    ("fractions", r"\frac{\frac{a}{b}}{\frac{c}{d}}"),
    ("fractions", r"\sqrt{a^2 + b^2}"),
    ("fractions", r"\sqrt[3]{x^3 + y^3}"),
    ("fractions", r"\sqrt{1 + \sqrt{x}}"),
    ("fractions", r"\frac{x^2 - 1}{x + 1} = x - 1"),
    ("fractions", r"\binom{n}{k} = \frac{n!}{k!(n - k)!}"),
    ("fractions", r"\frac{a}{b} + \frac{c}{d} = \frac{ad + bc}{bd}"),
    ("fractions", r"\frac{-p \pm \sqrt{p^2 - 4qr}}{2q}"),
    # Calculus and differential equations (15)
    ("calculus", r"\frac{dy}{dx} = 3x^2 + 2x + 1"),
    ("calculus", r"\frac{\partial f}{\partial x} + \frac{\partial f}{\partial y} = 0"),
    ("calculus", r"\frac{d}{dx}(x^2 + 1) = 2x"),
    ("calculus", r"\int_0^1 x^2 \, dx"),
    ("calculus", r"\int_{-1}^{1}(x^3 + x) \, dx = 0"),
    ("calculus", r"\int \frac{1}{x} \, dx = \ln|x|"),
    ("calculus", r"\int e^{-x^2} \, dx"),
    ("calculus", r"\int_0^\pi \sin(x) \, dx = 2"),
    ("calculus", r"\lim_{x \to 0} \frac{\sin(x)}{x} = 1"),
    ("calculus", r"\lim_{n \to \infty}(1 + \frac{1}{n})^n = e"),
    ("calculus", r"\sum_{i=1}^{n} i = \frac{n(n + 1)}{2}"),
    ("calculus", r"\prod_{i=1}^{n} i = n!"),
    ("calculus", r"\sum_{k=0}^{\infty}\frac{1}{k!} = e"),
    ("calculus", r"\int_a^b f(x) \, dx = F(b) - F(a)"),
    ("calculus", r"\frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} = 0"),
    # Series and products (10)
    ("series", r"\sum_{n=1}^{\infty}\frac{1}{n^2} = \frac{\pi^2}{6}"),
    ("series", r"\sum_{n=0}^{\infty}x^n = \frac{1}{1 - x}"),
    ("series", r"\sum_{k=0}^{n}\binom{n}{k} = 2^n"),
    ("series", r"\sum_{k=0}^{n}(-1)^k\binom{n}{k} = 0"),
    ("series", r"\prod_{k=1}^{n}(1 + \frac{1}{k}) = n + 1"),
    ("series", r"\sum_{n=1}^{\infty}\frac{(-1)^{n+1}}{n} = \ln(2)"),
    ("series", r"\sum_{n=1}^{\infty}\frac{1}{n(n + 1)} = 1"),
    ("series", r"\frac{1}{1 - x} = 1 + x + x^2 + \cdots"),
    ("series", r"a_n = \frac{1}{n}\sum_{k=1}^{n}x_k"),
    ("series", r"\lim_{n \to \infty}\sum_{k=1}^{n}\frac{1}{k^2} = \frac{\pi^2}{6}"),
    # Trigonometry and inverse functions (10)
    ("trigonometry", r"\sin^2(x) + \cos^2(x) = 1"),
    ("trigonometry", r"\sin(2x) = 2\sin(x)\cos(x)"),
    ("trigonometry", r"\cos(a + b) = \cos(a)\cos(b) - \sin(a)\sin(b)"),
    ("trigonometry", r"\tan(x) = \frac{\sin(x)}{\cos(x)}"),
    ("trigonometry", r"\sin(\pi - x) = \sin(x)"),
    ("trigonometry", r"\cos(2x) = \cos^2(x) - \sin^2(x)"),
    ("trigonometry", r"\arcsin(x) + \arccos(x) = \frac{\pi}{2}"),
    ("trigonometry", r"\frac{d}{dx}\sin(x) = \cos(x)"),
    ("trigonometry", r"\int \cos(x) \, dx = \sin(x)"),
    ("trigonometry", r"\sin(x) + \sin(y) = 2\sin(\frac{x + y}{2})\cos(\frac{x - y}{2})"),
    # Vectors and linear algebra (10)
    ("linear_algebra", r"\vec{v} = \langle a, b, c \rangle"),
    ("linear_algebra", r"\vec{u} \cdot \vec{v} = u_1v_1 + u_2v_2 + u_3v_3"),
    ("linear_algebra", r"\nabla \cdot \mathbf{F} = 0"),
    ("linear_algebra", r"\nabla f = \langle \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \rangle"),
    ("linear_algebra", r"\mathbf{A} + \mathbf{B} = \mathbf{C}"),
    ("linear_algebra", r"\det(\mathbf{A}) = ad - bc"),
    ("linear_algebra", r"\sum_{i=1}^{3}a_i b_i = a_1b_1 + a_2b_2 + a_3b_3"),
    ("linear_algebra", r"\vec{r}(t) = \langle x(t), y(t), z(t) \rangle"),
    ("linear_algebra", r"\frac{d\vec{r}}{dt} = \vec{v}"),
    ("linear_algebra", r"\mathbf{F} = m\mathbf{a}"),
    # Probability and statistics (10)
    ("probability", r"P(A) + P(B) = 1"),
    ("probability", r"P(A \cdot B) = P(A)P(B)"),
    ("probability", r"P(A|B) = \frac{P(A \cdot B)}{P(B)}"),
    ("probability", r"E(X) = \sum_{i=1}^{n}x_i p_i"),
    ("probability", r"\mu = \frac{1}{n}\sum_{i=1}^{n}x_i"),
    ("probability", r"\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2"),
    ("probability", r"z = \frac{x - \mu}{\sigma}"),
    ("probability", r"\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i"),
    ("probability", r"\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x - \mu)^2}{2\sigma^2}}"),
    ("probability", r"\binom{n}{k}p^k(1 - p)^{n-k}"),
    # Physics and applied equations (10)
    ("physics", r"E = mc^2"),
    ("physics", r"F = ma"),
    ("physics", r"p = mv"),
    ("physics", r"V = IR"),
    ("physics", r"P = VI"),
    ("physics", r"\frac{1}{2}mv^2 = mgh"),
    ("physics", r"Q = mc\,dT"),
    ("physics", r"x(t) = x_0 + v_0t + \frac{1}{2}at^2"),
    ("physics", r"v^2 = v_0^2 + 2a(x - x_0)"),
    ("physics", r"r = \frac{v}{f}"),
    # Geometry (10)
    ("geometry", r"A = \pi r^2"),
    ("geometry", r"C = 2\pi r"),
    ("geometry", r"h^2 + r^2 = l^2"),
    ("geometry", r"V = \frac{1}{3}\pi r^2h"),
    ("geometry", r"A = 2\pi r(r + h)"),
    ("geometry", r"d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}"),
    ("geometry", r"\vec{AB} = \langle x_B - x_A, y_B - y_A \rangle"),
    ("geometry", r"\sin(x) = \frac{opposite}{hypotenuse}"),
    ("geometry", r"\cos(x) = \frac{adjacent}{hypotenuse}"),
    ("geometry", r"\tan(x) = \frac{opposite}{adjacent}"),
    # Deeply nested, long expressions (5)
    ("nested", r"\frac{\int_0^1x^2\,dx}{\sum_{n=1}^{\infty}\frac{1}{n^2}} = \frac{2}{\pi^2}"),
    ("nested", r"\lim_{x\to0}\frac{\sin(x) + x^3}{x} = 1"),
    ("nested", r"\frac{d}{dx}\left(\frac{x^2 + 1}{x + 1}\right) = \frac{x^2 + 2x - 1}{(x + 1)^2}"),
    ("nested", r"\sum_{n=1}^{\infty}\frac{1}{n^2} + \int_0^1x\,dx = \frac{\pi^2}{6} + \frac{1}{2}"),
    ("nested", r"\frac{-b + \sqrt{b^2 - 4ac}}{2a} + \frac{-b - \sqrt{b^2 - 4ac}}{2a} = -\frac{b}{a}"),
]

COMPLEX_EXPRESSION_COUNT = 100


class DatasetError(Exception):
    """Raised when the dataset is missing or malformed."""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_latex(
    latex: str,
    fontsize: int = 40,
    dpi: int = 200,
    figsize: tuple[float, float] | None = None,
    text_color: str = "black",
    bg_color: str = "white",
) -> np.ndarray:
    """Render a LaTeX expression to a grayscale numpy array via mathtext."""
    fig = plt.figure(figsize=figsize or (8, 2))
    fig.patch.set_facecolor(bg_color)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(bg_color)
    ax.axis("off")
    ax.text(
        0.5, 0.5, f"${latex}$",
        ha="center", va="center", fontsize=fontsize, color=text_color,
    )

    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.1,
        facecolor=bg_color,
    )
    plt.close(fig)

    buf.seek(0)
    img = Image.open(buf).convert("L")
    return np.array(img)


def _apply_tier(img: np.ndarray, tier: str, rng: np.random.Generator) -> np.ndarray:
    """Apply a difficulty-tier degradation to a rendered grayscale image."""
    if tier == "clean":
        return img

    if tier == "noisy":
        # Add Gaussian noise, then clip back to valid range.
        noise = rng.normal(0, 25, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if tier == "low_res":
        # Render at low resolution, then upscale (blocky, aliased result).
        small = cv2_resize(img, 0.25)
        return cv2_resize(small, 4.0, up=True)

    if tier == "black_bg":
        # White text on a black background (inverted polarity).
        return 255 - img

    if tier == "white_bg":
        # Black text on white background (the default polarity).
        return img

    raise ValueError(f"Unknown tier: {tier!r}")


def cv2_resize(img: np.ndarray, factor: float, up: bool = False) -> np.ndarray:
    """Resize an image by a scale factor using OpenCV."""
    import cv2

    h, w = img.shape
    new_h, new_w = max(1, int(round(h * factor))), max(1, int(round(w * factor)))
    interp = cv2.INTER_CUBIC if up else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(
    data_dir: str,
    per_tier: int = DEFAULT_PER_TIER,
    seed: int = DEFAULT_SEED,
) -> str:
    """Render the full dataset and write a manifest.

    Parameters
    ----------
    data_dir : str
        Directory under which ``images/`` and ``manifest.json`` are written.
    per_tier : int
        Number of samples to generate per difficulty tier.
    seed : int
        Deterministic seed for the train/test split.

    Returns
    -------
    str
        Path to the written manifest file.
    """
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Build the full list of (id, latex, tier) samples.
    samples = []
    sample_id = 0
    for tier in TIERS:
        for latex in CURATED_EXPRESSIONS:
            for _ in range(per_tier):
                samples.append(
                    {"id": sample_id, "latex": latex, "tier": tier}
                )
                sample_id += 1

    # Deterministic disjoint split: shuffle ids with the fixed seed, then cut.
    ids = list(range(len(samples)))
    rng.shuffle(ids)
    n_train = int(round(len(ids) * TRAIN_RATIO))
    train_ids = set(ids[:n_train])
    test_ids = set(ids[n_train:])
    assert train_ids.isdisjoint(test_ids), "train/test split must be disjoint"

    manifest = []
    for sample in samples:
        sid = sample["id"]
        split = "train" if sid in train_ids else "test"
        latex = sample["latex"]
        tier = sample["tier"]

        # Render the base image, then apply the tier degradation.
        base = _render_latex(latex)
        img = _apply_tier(base, tier, rng)

        fname = f"{split}_{tier}_{sid:05d}.png"
        Image.fromarray(img).save(os.path.join(images_dir, fname))

        manifest.append(
            {
                "id": sid,
                "split": split,
                "tier": tier,
                "latex": latex,
                "image": fname,
            }
        )

    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "seed": seed,
                "per_tier": per_tier,
                "tiers": list(TIERS),
                "train_ratio": TRAIN_RATIO,
                "samples": manifest,
            },
            fh, indent=2,
        )

    return manifest_path


def generate_complex(
    data_dir: str,
    seed: int = 2026,
    tiers: tuple[str, ...] = TIERS,
) -> str:
    """Render the 100-expression own-code stress corpus.

    Unlike the baseline corpus, this evaluation set contains one sample for
    each expression and places every sample in the ``test`` split.  The
    expressions are hand-curated to exercise different structural families;
    the difficulty tiers are distributed across the examples so a single run
    measures both long-form recognition and robustness to degradation.  The
    own-code engine builds templates from its symbol bank, not from this set,
    so using all 100 samples as evaluation data does not leak training data.

    Parameters
    ----------
    data_dir : str
        Directory under which ``images/`` and ``manifest.json`` are written.
    seed : int
        Seed used to shuffle the balanced tier assignment and noise.
    tiers : tuple[str, ...]
        Degradation tiers to distribute across the 100 examples.

    Returns
    -------
    str
        Path to the written manifest file.
    """
    if len(COMPLEX_EXAMPLES) != COMPLEX_EXPRESSION_COUNT:
        raise DatasetError(
            "Complex corpus size changed: expected "
            f"{COMPLEX_EXPRESSION_COUNT}, got {len(COMPLEX_EXAMPLES)}"
        )
    if not tiers:
        raise ValueError("tiers must contain at least one difficulty tier")
    unknown = set(tiers) - set(TIERS)
    if unknown:
        raise ValueError(f"Unknown difficulty tiers: {sorted(unknown)}")

    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Repeat and shuffle the requested tiers, making the default assignment
    # exactly balanced: 20 examples in each of the five standard tiers.
    tier_pool = np.resize(np.asarray(tiers, dtype=object), len(COMPLEX_EXAMPLES))
    rng.shuffle(tier_pool)

    manifest = []
    for sample_id, ((category, latex), tier) in enumerate(
        zip(COMPLEX_EXAMPLES, tier_pool)
    ):
        tier = str(tier)
        width = max(8.0, min(18.0, 7.0 + 0.08 * len(latex)))
        base = _render_latex(
            latex,
            fontsize=34,
            figsize=(width, 2.4),
        )
        img = _apply_tier(base, tier, rng)
        fname = f"test_{tier}_{sample_id:03d}.png"
        Image.fromarray(img).save(os.path.join(images_dir, fname))
        manifest.append(
            {
                "id": sample_id,
                "split": "test",
                "tier": tier,
                "category": category,
                "latex": latex,
                "image": fname,
            }
        )

    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "dataset": "complex_100",
                "seed": seed,
                "n_examples": len(COMPLEX_EXAMPLES),
                "tiers": list(tiers),
                "split": "test",
                "samples": manifest,
            },
            fh,
            indent=2,
        )
    return manifest_path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_manifest(data_dir: str) -> dict:
    manifest_path = os.path.join(data_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise DatasetError(
            f"Manifest not found: {manifest_path}. Run generate() first."
        )
    with open(manifest_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load(
    split: str = "train",
    data_dir: str = "data",
) -> list[dict]:
    """Load samples for one split.

    Parameters
    ----------
    split : str
        ``"train"`` or ``"test"``.
    data_dir : str
        Directory containing ``images/`` and ``manifest.json``.

    Returns
    -------
    list[dict]
        Each dict has keys ``id``, ``split``, ``tier``, ``latex``, ``image``
        (relative filename) and ``image_path`` (absolute path).
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    manifest = _read_manifest(data_dir)
    images_dir = os.path.join(data_dir, "images")

    out = []
    for sample in manifest["samples"]:
        if sample["split"] != split:
            continue
        out.append(
            {
                **sample,
                "image_path": os.path.join(images_dir, sample["image"]),
            }
        )
    return out


def train_set(data_dir: str = "data") -> list[dict]:
    """Convenience accessor for the training split."""
    return load("train", data_dir)


def test_set(data_dir: str = "data") -> list[dict]:
    """Convenience accessor for the test split."""
    return load("test", data_dir)
