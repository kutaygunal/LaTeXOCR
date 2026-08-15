"""Unit and integration tests for src/dataset.py.

Covers: train/test disjoint, deterministic split, all tiers present,
manifest integrity, load() validation, image files exist, and generation
into a temp dir.
"""

import json
import os
import sys
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset import (  # noqa: E402
    CURATED_EXPRESSIONS,
    DEFAULT_PER_TIER,
    DEFAULT_SEED,
    TIERS,
    TRAIN_RATIO,
    DatasetError,
    generate,
    load,
    test_set as _test_set,
    train_set as _train_set,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _manifest():
    with open(os.path.join(DATA_DIR, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Manifest integrity
# ---------------------------------------------------------------------------


def test_manifest_metadata():
    m = _manifest()
    assert m["seed"] == DEFAULT_SEED
    assert m["per_tier"] == DEFAULT_PER_TIER
    assert m["train_ratio"] == TRAIN_RATIO
    assert list(m["tiers"]) == list(TIERS)


def test_manifest_sample_count():
    m = _manifest()
    expected = len(CURATED_EXPRESSIONS) * len(TIERS) * DEFAULT_PER_TIER
    assert len(m["samples"]) == expected


def test_manifest_all_tiers_present():
    m = _manifest()
    tiers = Counter(s["tier"] for s in m["samples"])
    for t in TIERS:
        assert tiers[t] == len(CURATED_EXPRESSIONS) * DEFAULT_PER_TIER, t


def test_manifest_every_image_file_exists():
    m = _manifest()
    images_dir = os.path.join(DATA_DIR, "images")
    missing = [
        s["image"] for s in m["samples"]
        if not os.path.isfile(os.path.join(images_dir, s["image"]))
    ]
    assert not missing, f"missing image files: {missing[:5]}"


def test_manifest_ids_unique():
    m = _manifest()
    ids = [s["id"] for s in m["samples"]]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Train/test disjointness
# ---------------------------------------------------------------------------


def test_train_test_disjoint():
    train = _train_set(DATA_DIR)
    test = _test_set(DATA_DIR)
    train_ids = {s["id"] for s in train}
    test_ids = {s["id"] for s in test}
    assert train_ids.isdisjoint(test_ids)


def test_train_test_cover_all_samples():
    train = _train_set(DATA_DIR)
    test = _test_set(DATA_DIR)
    all_ids = {s["id"] for s in _manifest()["samples"]}
    train_ids = {s["id"] for s in train}
    test_ids = {s["id"] for s in test}
    assert train_ids | test_ids == all_ids


def test_split_ratio_approximately_07():
    train = _train_set(DATA_DIR)
    test = _test_set(DATA_DIR)
    total = len(train) + len(test)
    ratio = len(train) / total
    assert abs(ratio - TRAIN_RATIO) < 0.01


def test_each_tier_represented_in_both_splits():
    train = _train_set(DATA_DIR)
    test = _test_set(DATA_DIR)
    for t in TIERS:
        assert any(s["tier"] == t for s in train), f"train missing tier {t}"
        assert any(s["tier"] == t for s in test), f"test missing tier {t}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_split_is_deterministic():
    """Regenerating with the same seed yields the same split assignment."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        generate(d, per_tier=2, seed=42)
        m1 = json.load(open(os.path.join(d, "manifest.json")))
        generate(d, per_tier=2, seed=42)
        m2 = json.load(open(os.path.join(d, "manifest.json")))
        s1 = [(s["id"], s["split"]) for s in m1["samples"]]
        s2 = [(s["id"], s["split"]) for s in m2["samples"]]
        assert s1 == s2


def test_different_seed_changes_split():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        generate(d, per_tier=2, seed=42)
        m1 = json.load(open(os.path.join(d, "manifest.json")))
        generate(d, per_tier=2, seed=7)
        m2 = json.load(open(os.path.join(d, "manifest.json")))
        s1 = [(s["id"], s["split"]) for s in m1["samples"]]
        s2 = [(s["id"], s["split"]) for s in m2["samples"]]
        assert s1 != s2


# ---------------------------------------------------------------------------
# load() validation
# ---------------------------------------------------------------------------


def test_load_invalid_split_raises():
    with pytest.raises(ValueError):
        load("validation", DATA_DIR)


def test_load_missing_manifest_raises(tmp_path):
    with pytest.raises(DatasetError):
        load("train", str(tmp_path))


def test_load_returns_image_path():
    train = _train_set(DATA_DIR)
    assert train
    for s in train[:5]:
        assert "image_path" in s
        assert os.path.isfile(s["image_path"])


def test_load_returns_expected_keys():
    train = _train_set(DATA_DIR)
    for s in train[:5]:
        for key in ("id", "split", "tier", "latex", "image", "image_path"):
            assert key in s


# ---------------------------------------------------------------------------
# generate() into temp dir
# ---------------------------------------------------------------------------


def test_generate_small(tmp_path):
    manifest = generate(str(tmp_path), per_tier=1, seed=1)
    assert os.path.isfile(manifest)
    m = json.load(open(manifest))
    assert len(m["samples"]) == len(CURATED_EXPRESSIONS) * len(TIERS) * 1
    # Every image written.
    images_dir = os.path.join(str(tmp_path), "images")
    n_files = len(os.listdir(images_dir))
    assert n_files == len(m["samples"])


def test_generate_disjoint_guarantee(tmp_path):
    generate(str(tmp_path), per_tier=3, seed=99)
    train = _train_set(str(tmp_path))
    test = _test_set(str(tmp_path))
    assert {s["id"] for s in train}.isdisjoint({s["id"] for s in test})


# ---------------------------------------------------------------------------
# Tier degradation (_apply_tier / cv2_resize)
# ---------------------------------------------------------------------------


def _render_base():
    from dataset import _render_latex

    return _render_latex(CURATED_EXPRESSIONS[0])


def test_apply_tier_clean_is_identity():
    from dataset import _apply_tier

    base = _render_base()
    rng = np.random.default_rng(0)
    out = _apply_tier(base, "clean", rng)
    np.testing.assert_array_equal(out, base)


def test_apply_tier_noisy_differs_from_clean():
    from dataset import _apply_tier

    base = _render_base()
    rng = np.random.default_rng(0)
    out = _apply_tier(base, "noisy", rng)
    assert out.shape == base.shape
    assert not np.array_equal(out, base), "noisy tier added no noise"


def test_apply_tier_noisy_is_deterministic_per_seed():
    from dataset import _apply_tier

    base = _render_base()
    a = _apply_tier(base, "noisy", np.random.default_rng(5))
    b = _apply_tier(base, "noisy", np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)


def test_apply_tier_low_res_downscales_intermediate():
    from dataset import _apply_tier, cv2_resize

    base = _render_base()
    rng = np.random.default_rng(0)
    out = _apply_tier(base, "low_res", rng)
    # The intermediate downscale must have reduced resolution.
    small = cv2_resize(base, 0.25)
    assert small.shape[0] < base.shape[0]
    assert small.shape[1] < base.shape[1]
    # The upscaled result differs from the clean base (blocky/aliased).
    assert not np.array_equal(out, base)


def test_apply_tier_black_bg_inverts():
    from dataset import _apply_tier

    base = _render_base()
    rng = np.random.default_rng(0)
    out = _apply_tier(base, "black_bg", rng)
    np.testing.assert_array_equal(out, 255 - base)


def test_apply_tier_white_bg_is_identity():
    from dataset import _apply_tier

    base = _render_base()
    rng = np.random.default_rng(0)
    out = _apply_tier(base, "white_bg", rng)
    np.testing.assert_array_equal(out, base)


def test_apply_tier_unknown_raises():
    from dataset import _apply_tier

    base = _render_base()
    with pytest.raises(ValueError):
        _apply_tier(base, "sepia", np.random.default_rng(0))


def test_cv2_resize_downscale_math():
    from dataset import cv2_resize

    img = np.zeros((100, 200), dtype=np.uint8)
    small = cv2_resize(img, 0.5)
    assert small.shape == (50, 100)


def test_cv2_resize_upscale_math():
    from dataset import cv2_resize

    img = np.zeros((50, 100), dtype=np.uint8)
    big = cv2_resize(img, 2.0, up=True)
    assert big.shape == (100, 200)


def test_cv2_resize_min_size_guard():
    """A tiny scale factor must not produce a zero-size image."""
    from dataset import cv2_resize

    img = np.zeros((10, 20), dtype=np.uint8)
    tiny = cv2_resize(img, 0.01)
    assert tiny.shape[0] >= 1
    assert tiny.shape[1] >= 1


def test_rendered_dataset_image_is_non_blank():
    """mathtext must actually render content (both black and white pixels)."""
    base = _render_base()
    assert 0 in np.unique(base), "rendered image has no black (text) pixels"
    assert 255 in np.unique(base), "rendered image has no white (bg) pixels"
    assert base.size > 0
