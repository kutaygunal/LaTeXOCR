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
- ``generate(data_dir, ...)``          : render all samples and write manifest.
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


class DatasetError(Exception):
    """Raised when the dataset is missing or malformed."""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_latex(
    latex: str,
    fontsize: int = 40,
    dpi: int = 200,
    text_color: str = "black",
    bg_color: str = "white",
) -> np.ndarray:
    """Render a LaTeX expression to a grayscale numpy array via mathtext."""
    fig = plt.figure(figsize=(8, 2))
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
