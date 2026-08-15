"""Unit and integration tests for src/metrics.py.

Covers: exact match, Levenshtein distance/similarity, symbol accuracy,
timing, per-tier robustness, and error handling.
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from metrics import (  # noqa: E402
    evaluate,
    exact_match,
    levenshtein_distance,
    levenshtein_similarity,
    symbol_accuracy,
    time_recognizer,
)


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------


def test_exact_match_true():
    assert exact_match("x^2", "x^2") is True


def test_exact_match_false():
    assert exact_match("x^2", "x^3") is False


def test_exact_match_empty():
    assert exact_match("", "") is True
    assert exact_match("", "a") is False


# ---------------------------------------------------------------------------
# levenshtein_distance
# ---------------------------------------------------------------------------


def test_levenshtein_identical_zero():
    assert levenshtein_distance("abc", "abc") == 0


def test_levenshtein_single_substitution():
    assert levenshtein_distance("kitten", "sitten") == 1


def test_levenshtein_classic():
    assert levenshtein_distance("kitten", "sitting") == 3


def test_levenshtein_empty():
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("", "abc") == 3
    assert levenshtein_distance("abc", "") == 3


def test_levenshtein_insertion_deletion():
    assert levenshtein_distance("abc", "ab") == 1
    assert levenshtein_distance("ab", "abc") == 1


# ---------------------------------------------------------------------------
# levenshtein_similarity
# ---------------------------------------------------------------------------


def test_similarity_identical_is_one():
    assert levenshtein_similarity("x^2", "x^2") == 1.0


def test_similarity_empty_both_is_one():
    assert levenshtein_similarity("", "") == 1.0


def test_similarity_bounded_01():
    for a, b in [("abc", "xyz"), ("kitten", "sitting"), ("a", "b")]:
        s = levenshtein_similarity(a, b)
        assert 0.0 <= s <= 1.0


def test_similarity_known_value():
    # "abc" vs "abd": distance 1, denom 3 -> 1 - 1/3 = 2/3
    assert levenshtein_similarity("abc", "abd") == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# symbol_accuracy
# ---------------------------------------------------------------------------


def test_symbol_accuracy_identical_is_one():
    assert symbol_accuracy("x^2", "x^2") == 1.0


def test_symbol_accuracy_empty_is_one():
    assert symbol_accuracy("", "") == 1.0


def test_symbol_accuracy_partial():
    # "abc" vs "abd": 2 of 3 match -> 2/3
    assert symbol_accuracy("abc", "abd") == pytest.approx(2 / 3)


def test_symbol_accuracy_length_mismatch_counts_as_error():
    # "abc" vs "abcd": 3 of 4 match -> 0.75
    assert symbol_accuracy("abc", "abcd") == pytest.approx(0.75)


def test_symbol_accuracy_bounded_01():
    for a, b in [("abc", "xyz"), ("", "abc"), ("abc", "")]:
        s = symbol_accuracy(a, b)
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# evaluate() aggregate
# ---------------------------------------------------------------------------


def test_evaluate_basic():
    res = evaluate(["x^2", "x^2", "x^3"], ["x^2", "x^2", "x^2"])
    assert res["n"] == 3
    assert res["exact_match_rate"] == pytest.approx(2 / 3)
    # "x^3" vs "x^2": distance 1, denom 3 -> sim 2/3. Mean = (1+1+2/3)/3 = 8/9.
    assert res["mean_levenshtein_similarity"] == pytest.approx(8 / 9)
    # symbol acc: (3+3+2)/9 = 8/9
    assert res["mean_symbol_accuracy"] == pytest.approx(8 / 9)


def test_evaluate_empty():
    res = evaluate([], [])
    assert res["n"] == 0
    assert res["exact_match_rate"] == 0.0


def test_evaluate_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate(["a"], ["a", "b"])


def test_evaluate_times_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate(["a"], ["a"], times=[0.1, 0.2])


def test_evaluate_tiers_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate(["a"], ["a"], tiers=["clean", "noisy"])


def test_evaluate_timing_stats():
    res = evaluate(["a", "b"], ["a", "b"], times=[0.1, 0.3])
    assert res["timing"]["mean_seconds"] == pytest.approx(0.2)
    assert res["timing"]["median_seconds"] == pytest.approx(0.2)
    assert res["timing"]["total_seconds"] == pytest.approx(0.4)


def test_evaluate_per_tier_robustness():
    res = evaluate(
        ["x", "x", "y", "y"],
        ["x", "x", "x", "y"],
        tiers=["clean", "clean", "noisy", "noisy"],
    )
    pt = res["per_tier"]
    assert set(pt.keys()) == {"clean", "noisy"}
    assert pt["clean"]["n"] == 2
    assert pt["clean"]["exact_match_rate"] == pytest.approx(1.0)
    assert pt["noisy"]["n"] == 2
    assert pt["noisy"]["exact_match_rate"] == pytest.approx(0.5)


def test_evaluate_no_timing_no_tiers():
    res = evaluate(["a"], ["a"])
    assert "timing" not in res
    assert "per_tier" not in res


# ---------------------------------------------------------------------------
# time_recognizer
# ---------------------------------------------------------------------------


def test_time_recognizer_returns_prediction_and_time():
    def fake_recognize(path):
        time.sleep(0.01)
        return "pred"

    pred, elapsed = time_recognizer(fake_recognize, "img.png")
    assert pred == "pred"
    assert elapsed >= 0.01


def test_time_recognizer_propagates_errors():
    def bad(path):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        time_recognizer(bad, "img.png")


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_evaluate_is_thread_safe():
    """evaluate() must return identical results when called concurrently."""
    import threading

    preds = ["x^2", "x^3", "y"] * 20
    truths = ["x^2", "x^2", "y"] * 20
    tiers = ["clean", "noisy", "low_res"] * 20
    times = [0.1] * 60

    expected = evaluate(preds, truths, times=times, tiers=tiers)
    results = []
    errors = []

    def worker():
        try:
            results.append(evaluate(preds, truths, times=times, tiers=tiers))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 8
    for r in results:
        assert r == expected


def test_levenshtein_distance_is_thread_safe():
    import threading

    pairs = [("kitten", "sitting"), ("abc", "abd"), ("", "xyz")]
    expected = [levenshtein_distance(a, b) for a, b in pairs]
    results = []
    errors = []

    def worker():
        try:
            results.append([levenshtein_distance(a, b) for a, b in pairs])
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    for r in results:
        assert r == expected
