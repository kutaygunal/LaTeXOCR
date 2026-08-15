"""Evaluation metrics for the LaTeX OCR benchmark.

Provides scoring functions used by BENCH-001 to compare the two recognition
approaches (AI-001 and OWN-001) on the held-out test set.

Metrics
-------
- exact match
- normalized Levenshtein similarity
- symbol-level (character) accuracy
- timing (per image)
- per-tier robustness (pass rate across difficulty tiers)
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Return True if the prediction exactly equals the ground truth."""
    return prediction == ground_truth


def levenshtein_distance(a: str, b: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def levenshtein_similarity(prediction: str, ground_truth: str) -> float:
    """Normalized Levenshtein similarity in [0, 1].

    1.0 means identical; 0.0 means completely different. Normalized by the
    length of the longer string.
    """
    if prediction == ground_truth:
        return 1.0
    denom = max(len(prediction), len(ground_truth))
    if denom == 0:
        return 1.0
    return 1.0 - levenshtein_distance(prediction, ground_truth) / denom


def symbol_accuracy(prediction: str, ground_truth: str) -> float:
    """Character-level accuracy in [0, 1].

    Fraction of aligned characters that match, computed over the longer string
    length (mismatches and length differences both count as errors).
    """
    if prediction == ground_truth:
        return 1.0
    n = max(len(prediction), len(ground_truth))
    if n == 0:
        return 1.0
    matches = sum(1 for a, b in zip(prediction, ground_truth) if a == b)
    return matches / n


def evaluate(
    predictions: list[str],
    ground_truths: list[str],
    times: list[float] | None = None,
    tiers: list[str] | None = None,
) -> dict:
    """Compute aggregate metrics over a batch of predictions.

    Parameters
    ----------
    predictions : list[str]
        Recognizer output for each sample.
    ground_truths : list[str]
        Ground-truth LaTeX for each sample.
    times : list[float] | None
        Per-sample inference time in seconds (optional).
    tiers : list[str] | None
        Per-sample difficulty tier (optional). Enables per-tier robustness.

    Returns
    -------
    dict
        Aggregate metrics including exact-match rate, mean similarity,
        mean symbol accuracy, timing stats, and per-tier robustness.
    """
    if not (len(predictions) == len(ground_truths)):
        raise ValueError("predictions and ground_truths must have equal length")
    if times is not None and len(times) != len(predictions):
        raise ValueError("times must match the number of samples")
    if tiers is not None and len(tiers) != len(predictions):
        raise ValueError("tiers must match the number of samples")

    n = len(predictions)
    exact = [exact_match(p, g) for p, g in zip(predictions, ground_truths)]
    sims = [levenshtein_similarity(p, g) for p, g in zip(predictions, ground_truths)]
    syms = [symbol_accuracy(p, g) for p, g in zip(predictions, ground_truths)]

    result: dict = {
        "n": n,
        "exact_match_rate": float(np.mean(exact)) if n else 0.0,
        "mean_levenshtein_similarity": float(np.mean(sims)) if n else 0.0,
        "mean_symbol_accuracy": float(np.mean(syms)) if n else 0.0,
    }

    if times is not None:
        result["timing"] = {
            "mean_seconds": float(np.mean(times)) if n else 0.0,
            "median_seconds": float(np.median(times)) if n else 0.0,
            "total_seconds": float(np.sum(times)) if n else 0.0,
        }

    if tiers is not None:
        result["per_tier"] = _per_tier_robustness(
            predictions, ground_truths, tiers
        )

    return result


def _per_tier_robustness(
    predictions: list[str],
    ground_truths: list[str],
    tiers: list[str],
) -> dict:
    """Compute exact-match rate and mean similarity per difficulty tier."""
    by_tier: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for p, g, t in zip(predictions, ground_truths, tiers):
        by_tier[t].append((p, g))

    out = {}
    for tier, pairs in sorted(by_tier.items()):
        exact = [exact_match(p, g) for p, g in pairs]
        sims = [levenshtein_similarity(p, g) for p, g in pairs]
        out[tier] = {
            "n": len(pairs),
            "exact_match_rate": float(np.mean(exact)),
            "mean_levenshtein_similarity": float(np.mean(sims)),
        }
    return out


def time_recognizer(recognize, image_path: str) -> tuple[str, float]:
    """Run a recognizer on one image and return (prediction, seconds)."""
    start = time.perf_counter()
    prediction = recognize(image_path)
    elapsed = time.perf_counter() - start
    return prediction, elapsed
