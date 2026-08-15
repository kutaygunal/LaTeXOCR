"""End-to-end accuracy guards for the own-code recognizer.

These lock in what the pipeline can read today. They run on a fixed sample of
the **test** split — the recognizer never trains on any split, so this is a
regression guard rather than a leak: the thresholds are set well below the
measured accuracy so ordinary variation does not fail the suite.
"""

import os
import random
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import owncode_recognizer as ocr  # noqa: E402
from dataset import test_set as load_test_split  # noqa: E402
from metrics import levenshtein_similarity, symbol_accuracy  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TIERS = ("clean", "noisy", "low_res", "black_bg", "white_bg")
SAMPLES_PER_TIER = 6


@pytest.fixture(scope="module")
def recognizer():
    return ocr.Recognizer()


@pytest.fixture(scope="module")
def samples():
    by_tier = {tier: [] for tier in TIERS}
    for sample in load_test_split(DATA_DIR):
        by_tier.setdefault(sample["tier"], []).append(sample)
    rng = random.Random(1234)
    chosen = []
    for tier in TIERS:
        pool = by_tier[tier]
        chosen.extend(rng.sample(pool, min(SAMPLES_PER_TIER, len(pool))))
    return chosen


@pytest.fixture(scope="module")
def predictions(recognizer, samples):
    return [(s, recognizer.recognize(s["image_path"])) for s in samples]


def test_symbol_accuracy_meets_threshold(predictions):
    scores = [symbol_accuracy(p, s["latex"]) for s, p in predictions]
    mean = sum(scores) / len(scores)
    assert mean > 0.80, f"symbol accuracy regressed to {mean:.3f}"


def test_similarity_meets_threshold(predictions):
    scores = [levenshtein_similarity(p, s["latex"]) for s, p in predictions]
    mean = sum(scores) / len(scores)
    assert mean > 0.85, f"similarity regressed to {mean:.3f}"


def test_most_samples_are_read_exactly(predictions):
    exact = sum(1 for s, p in predictions if p == s["latex"])
    rate = exact / len(predictions)
    assert rate > 0.55, f"exact-match rate regressed to {rate:.3f}"


def test_recognition_is_fast(recognizer, samples):
    """The template bank is built once, so recognition is milliseconds."""
    recognizer.recognize(samples[0]["image_path"])  # warm any lazy state
    start = time.perf_counter()
    for sample in samples[:10]:
        recognizer.recognize(sample["image_path"])
    mean = (time.perf_counter() - start) / 10
    assert mean < 0.5, f"recognition slowed to {mean:.3f}s per image"


def test_module_level_recognize_reuses_the_shared_bank():
    """`recognize()` must not re-render the symbol library per call."""
    first = ocr._shared()
    ocr.recognize(load_test_split(DATA_DIR)[0]["image_path"])
    assert ocr._shared() is first


@pytest.mark.parametrize(
    "latex",
    [
        r"\frac{a}{b}",
        r"x^2 + y^2 = z^2",
        r"\alpha + \beta = \gamma",
        r"\sqrt{a^2 + b^2}",
        r"\frac{dy}{dx}",
    ],
)
def test_reads_clean_expressions_exactly(recognizer, latex):
    """A handful of clean expressions must come back character for character."""
    sample = next(
        s for s in load_test_split(DATA_DIR)
        if s["latex"] == latex and s["tier"] == "clean"
    )
    assert recognizer.recognize(sample["image_path"]) == latex
