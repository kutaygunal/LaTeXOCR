"""Unit and integration tests for src/benchmark.py.

Covers: runs recognizers over test_set only (never train_set), per-tier
metrics, valid results JSON emission, graceful Ollama-unreachable handling,
per-sample failure recording without aborting, --skip-ai / --limit flags, and
metrics integration with the benchmark output.

All recognizers are injected as fakes so no live Ollama / own-code run is
required for the unit tests.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmark import (  # noqa: E402
    BenchmarkError,
    _best_recognizer,
    _build_summary,
    _fastest_recognizer,
    _per_tier_metrics,
    _run_recognizer,
    build_parser,
    main,
    pass_rate,
    run_benchmark,
)
from dataset import test_set as _test_set  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _fake_ai(path):
    return "x^2"


def _fake_own(path):
    return "x^2"


# ---------------------------------------------------------------------------
# pass_rate
# ---------------------------------------------------------------------------


def test_pass_rate_all_pass():
    assert pass_rate(["x^2", "x^2"], ["x^2", "x^2"]) == 1.0


def test_pass_rate_none_pass():
    assert pass_rate(["zzz"], ["x^2"]) == 0.0


def test_pass_rate_empty():
    assert pass_rate([], []) == 0.0


def test_pass_rate_partial():
    # "x^2" vs "x^3": sim = 1 - 1/3 = 0.667 < 0.8 -> fail
    # "x^2" vs "x^2": sim = 1.0 -> pass
    assert pass_rate(["x^2", "x^3"], ["x^2", "x^2"]) == 0.5


# ---------------------------------------------------------------------------
# run_benchmark — structure & JSON emission
# ---------------------------------------------------------------------------


def test_run_benchmark_writes_json(tmp_path):
    results_dir = str(tmp_path / "results")
    payload = run_benchmark(
        data_dir=DATA_DIR,
        results_dir=results_dir,
        results_file="benchmark.json",
        recognizers={"ai": _fake_ai, "owncode": _fake_own},
        limit=10,
    )
    out = os.path.join(results_dir, "benchmark.json")
    assert os.path.isfile(out)
    # Valid JSON, matches returned payload.
    with open(out, encoding="utf-8") as fh:
        disk = json.load(fh)
    assert disk == payload


def test_run_benchmark_payload_structure(tmp_path):
    payload = run_benchmark(
        data_dir=DATA_DIR,
        results_dir=str(tmp_path / "r"),
        recognizers={"ai": _fake_ai, "owncode": _fake_own},
        limit=10,
    )
    assert set(payload.keys()) == {"meta", "results", "summary"}
    assert set(payload["meta"].keys()) >= {
        "timestamp", "data_dir", "n_test_samples", "tiers",
        "pass_threshold", "recognizers",
    }
    assert set(payload["results"].keys()) == {"ai", "owncode"}
    assert set(payload["summary"].keys()) == {
        "best_exact_match", "best_mean_levenshtein_similarity",
        "best_mean_symbol_accuracy", "best_pass_rate", "fastest_mean_seconds",
    }


def test_run_benchmark_limit_respected(tmp_path):
    payload = run_benchmark(
        data_dir=DATA_DIR,
        results_dir=str(tmp_path / "r"),
        recognizers={"ai": _fake_ai},
        limit=5,
    )
    assert payload["meta"]["n_test_samples"] == 5
    assert payload["results"]["ai"]["n"] == 5


def test_run_benchmark_per_tier_metrics(tmp_path):
    # Use the full test set so all 5 tiers are represented.
    payload = run_benchmark(
        data_dir=DATA_DIR,
        results_dir=str(tmp_path / "r"),
        recognizers={"ai": _fake_ai},
        limit=900,
    )
    pt = payload["results"]["ai"]["per_tier"]
    assert set(pt.keys()) == {"black_bg", "clean", "low_res", "noisy", "white_bg"}
    for tier, block in pt.items():
        assert block["n"] > 0
        for key in ("exact_match_rate", "mean_levenshtein_similarity",
                    "mean_symbol_accuracy", "mean_seconds", "pass_rate"):
            assert key in block


def test_run_benchmark_uses_test_set_only(tmp_path):
    """The recognizer must only ever see test_set image paths (no train leak)."""
    test_paths = {s["image_path"] for s in _test_set(DATA_DIR)}
    seen = []

    def fake(path):
        seen.append(path)
        return "x"

    run_benchmark(
        data_dir=DATA_DIR,
        results_dir=str(tmp_path / "r"),
        recognizers={"ai": fake},
        limit=20,
    )
    assert seen, "recognizer was never called"
    for p in seen:
        assert p in test_paths, f"recognizer saw non-test path: {p}"


def test_run_benchmark_missing_dataset_raises(tmp_path):
    with pytest.raises(BenchmarkError):
        run_benchmark(
            data_dir=str(tmp_path / "nope"),
            results_dir=str(tmp_path / "r"),
            recognizers={"ai": _fake_ai},
        )


def test_run_benchmark_zero_limit_raises(tmp_path):
    with pytest.raises(BenchmarkError):
        run_benchmark(
            data_dir=DATA_DIR,
            results_dir=str(tmp_path / "r"),
            recognizers={"ai": _fake_ai},
            limit=0,
        )


# ---------------------------------------------------------------------------
# Graceful handling of unavailable / failing recognizers
# ---------------------------------------------------------------------------


def test_unreachable_recognizer_reported_unavailable(tmp_path):
    def broken(path):
        raise RuntimeError("connection refused")

    payload = run_benchmark(
        data_dir=DATA_DIR,
        results_dir=str(tmp_path / "r"),
        recognizers={"ai": broken, "owncode": _fake_own},
        limit=5,
    )
    ai = payload["results"]["ai"]
    assert ai["available"] is False
    assert ai["n"] == 0
    assert "error" in ai
    # The other recognizer still ran.
    assert payload["results"]["owncode"]["available"] is True


def test_per_sample_failure_recorded_without_abort(tmp_path):
    """A recognizer that fails on some (not all) samples must not abort."""
    calls = {"n": 0}

    def flaky(path):
        calls["n"] += 1
        if calls["n"] % 2 == 0:  # fail every 2nd sample
            raise RuntimeError("boom")
        return "x^2"

    payload = run_benchmark(
        data_dir=DATA_DIR,
        results_dir=str(tmp_path / "r"),
        recognizers={"ai": flaky},
        limit=10,
    )
    block = payload["results"]["ai"]
    assert block["available"] is True
    assert block["n"] == 10
    assert block["error_count"] == 5  # 5 of 10 failed


def test_probe_only_checks_first_sample(tmp_path):
    """Probe failure (first sample) marks unavailable; later failures count."""
    calls = {"n": 0}

    def fails_first(path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("probe fail")
        return "x"

    payload = run_benchmark(
        data_dir=DATA_DIR,
        results_dir=str(tmp_path / "r"),
        recognizers={"ai": fails_first},
        limit=5,
    )
    assert payload["results"]["ai"]["available"] is False


# ---------------------------------------------------------------------------
# _run_recognizer / _per_tier_metrics
# ---------------------------------------------------------------------------


def test_run_recognizer_metrics_integration():
    samples = _test_set(DATA_DIR)[:10]
    result = _run_recognizer(_fake_ai, samples, threshold=0.8, probe=False)
    assert result["available"] is True
    assert result["n"] == 10
    assert "exact_match_rate" in result
    assert "mean_levenshtein_similarity" in result
    assert "mean_symbol_accuracy" in result
    assert "pass_rate" in result
    assert "timing" in result
    assert "per_tier" in result
    assert "error_count" in result


def test_per_tier_metrics_groups_by_tier():
    # Build a sample list covering all 5 tiers (one per tier).
    ts = _test_set(DATA_DIR)
    by_tier = {}
    for s in ts:
        by_tier.setdefault(s["tier"], s)
    samples = [by_tier[t] for t in ("clean", "noisy", "low_res", "black_bg", "white_bg")]
    preds = ["x"] * len(samples)
    times = [0.1] * len(samples)
    pt = _per_tier_metrics(samples, preds, times, threshold=0.8)
    assert set(pt.keys()) == {"black_bg", "clean", "low_res", "noisy", "white_bg"}
    total = sum(b["n"] for b in pt.values())
    assert total == len(samples)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def test_best_recognizer_ignores_unavailable():
    results = {
        "ai": {"available": False, "exact_match_rate": 0.9},
        "owncode": {"available": True, "exact_match_rate": 0.5},
    }
    assert _best_recognizer(results, "exact_match_rate") == "owncode"


def test_best_recognizer_none_when_all_unavailable():
    results = {"ai": {"available": False}}
    assert _best_recognizer(results, "exact_match_rate") is None


def test_fastest_recognizer():
    results = {
        "ai": {"available": True, "timing": {"mean_seconds": 2.0}},
        "owncode": {"available": True, "timing": {"mean_seconds": 0.1}},
    }
    assert _fastest_recognizer(results) == "owncode"


def test_build_summary():
    results = {
        "ai": {"available": True, "exact_match_rate": 0.9,
               "mean_levenshtein_similarity": 0.9,
               "mean_symbol_accuracy": 0.9, "pass_rate": 0.9,
               "timing": {"mean_seconds": 2.0}},
        "owncode": {"available": True, "exact_match_rate": 0.5,
                    "mean_levenshtein_similarity": 0.5,
                    "mean_symbol_accuracy": 0.5, "pass_rate": 0.5,
                    "timing": {"mean_seconds": 0.1}},
    }
    s = _build_summary(results)
    assert s["best_exact_match"] == "ai"
    assert s["best_mean_levenshtein_similarity"] == "ai"
    assert s["best_mean_symbol_accuracy"] == "ai"
    assert s["best_pass_rate"] == "ai"
    assert s["fastest_mean_seconds"] == "owncode"


# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------


def test_cli_skip_ai(tmp_path):
    rc = main([
        "--data-dir", DATA_DIR,
        "--results-dir", str(tmp_path / "r"),
        "--skip-ai",
        "--limit", "5",
    ])
    assert rc == 0
    out = os.path.join(str(tmp_path / "r"), "benchmark.json")
    with open(out, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert set(payload["results"].keys()) == {"owncode"}
    assert payload["meta"]["recognizers"] == ["owncode"]


def test_cli_limit_flag(tmp_path):
    rc = main([
        "--data-dir", DATA_DIR,
        "--results-dir", str(tmp_path / "r"),
        "--skip-ai",
        "--limit", "3",
    ])
    assert rc == 0
    out = os.path.join(str(tmp_path / "r"), "benchmark.json")
    with open(out, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["meta"]["n_test_samples"] == 3


def test_build_parser_has_flags():
    p = build_parser()
    args = p.parse_args(["--skip-ai", "--limit", "7"])
    assert args.skip_ai is True
    assert args.limit == 7
