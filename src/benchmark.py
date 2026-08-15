"""Benchmark harness for the LaTeX OCR project.

Runs BOTH recognizers (the local AI ``qwen3-vl:8b`` and the own-code OCR
pipeline) over the **held-out ``test_set`` ONLY** and produces a
machine-readable results JSON under ``results/``.

Data-leakage guardrail
----------------------
This module evaluates **only** on ``dataset.test_set()``. It never touches
``train_set()``. The own-code recognizer builds its template library / CNN from
rendered symbols (not from the dataset), and the AI recognizer is a pre-trained
model, so evaluating both on the held-out test set is fair and leakage-free.

Public API
----------
- ``run_benchmark(data_dir, results_dir, ...) -> dict`` : run the full
  benchmark and write ``results/benchmark.json``.
- ``main(argv) -> int`` : CLI entry point (``python -m src.benchmark``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# Ensure the package root is importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_recognizer import RecognizerError, recognize as ai_recognize  # noqa: E402
from dataset import DatasetError, test_set  # noqa: E402
from metrics import evaluate, levenshtein_similarity, time_recognizer  # noqa: E402
from owncode_recognizer import recognize as owncode_recognize  # noqa: E402

# Default output filename for the machine-readable results.
DEFAULT_RESULTS_FILE = "benchmark.json"

# A sample "passes" when its normalized Levenshtein similarity is at least this.
DEFAULT_PASS_THRESHOLD = 0.8

# Recognizer registry: name -> callable(image_path) -> str.
RECOGNIZERS = {
    "ai": ai_recognize,
    "owncode": owncode_recognize,
}


class BenchmarkError(Exception):
    """Raised when the benchmark cannot run (e.g. missing dataset)."""


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def pass_rate(
    predictions: list[str],
    ground_truths: list[str],
    threshold: float = DEFAULT_PASS_THRESHOLD,
) -> float:
    """Fraction of samples whose similarity meets or exceeds ``threshold``.

    A more forgiving robustness measure than exact match: a prediction counts
    as a pass if it is close enough to the ground truth.
    """
    if not predictions:
        return 0.0
    hits = sum(
        1
        for p, g in zip(predictions, ground_truths)
        if levenshtein_similarity(p, g) >= threshold
    )
    return hits / len(predictions)


def _per_tier_metrics(
    samples: list[dict],
    predictions: list[str],
    times: list[float],
    threshold: float,
) -> dict:
    """Compute per-tier metrics (exact match, similarity, symbol acc, time,
    pass rate) by grouping samples by their difficulty tier."""
    by_tier: dict[str, dict] = {}
    for sample, pred, elapsed in zip(samples, predictions, times):
        tier = sample["tier"]
        bucket = by_tier.setdefault(
            tier, {"predictions": [], "ground_truths": [], "times": []}
        )
        bucket["predictions"].append(pred)
        bucket["ground_truths"].append(sample["latex"])
        bucket["times"].append(elapsed)

    out: dict[str, dict] = {}
    for tier in sorted(by_tier):
        b = by_tier[tier]
        agg = evaluate(
            b["predictions"],
            b["ground_truths"],
            times=b["times"],
        )
        out[tier] = {
            "n": agg["n"],
            "exact_match_rate": agg["exact_match_rate"],
            "mean_levenshtein_similarity": agg["mean_levenshtein_similarity"],
            "mean_symbol_accuracy": agg["mean_symbol_accuracy"],
            "mean_seconds": agg["timing"]["mean_seconds"],
            "pass_rate": pass_rate(
                b["predictions"], b["ground_truths"], threshold
            ),
        }
    return out


# ---------------------------------------------------------------------------
# Recognizer runner
# ---------------------------------------------------------------------------


def _probe(recognize, sample: dict) -> bool:
    """Return True if the recognizer can process a sample without raising."""
    try:
        recognize(sample["image_path"])
        return True
    except Exception:
        return False


def _run_recognizer(
    recognize,
    samples: list[dict],
    threshold: float,
    probe: bool = True,
) -> dict:
    """Run one recognizer over all samples and return its metrics block.

    If ``probe`` is True and the recognizer fails on the first sample, the
    recognizer is reported as **unavailable** (``available=False``) rather than
    crashing the whole benchmark. This is how an unreachable Ollama is handled
    gracefully.
    """
    if probe and samples and not _probe(recognize, samples[0]):
        return {
            "available": False,
            "n": 0,
            "error": "recognizer unavailable (probe failed)",
        }

    predictions: list[str] = []
    times: list[float] = []
    error_count = 0
    for sample in samples:
        try:
            pred, elapsed = time_recognizer(recognize, sample["image_path"])
        except Exception:
            # A single-sample failure should not abort the benchmark; record an
            # empty prediction and count it as an error.
            pred, elapsed = "", 0.0
            error_count += 1
        predictions.append(pred)
        times.append(elapsed)

    ground_truths = [s["latex"] for s in samples]
    agg = evaluate(predictions, ground_truths, times=times)

    result: dict = {
        "available": True,
        "n": agg["n"],
        "exact_match_rate": agg["exact_match_rate"],
        "mean_levenshtein_similarity": agg["mean_levenshtein_similarity"],
        "mean_symbol_accuracy": agg["mean_symbol_accuracy"],
        "pass_rate": pass_rate(predictions, ground_truths, threshold),
        "timing": agg["timing"],
        "per_tier": _per_tier_metrics(samples, predictions, times, threshold),
        "error_count": error_count,
    }
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _best_recognizer(results: dict, metric: str) -> str | None:
    """Return the recognizer name with the highest value for ``metric``.

    Unavailable recognizers are ignored. Returns None if no recognizer has a
    usable value.
    """
    best_name = None
    best_value = -1.0
    for name, block in results.items():
        if not block.get("available"):
            continue
        value = block.get(metric)
        if value is None:
            continue
        if value > best_value:
            best_value = value
            best_name = name
    return best_name


def _fastest_recognizer(results: dict) -> str | None:
    """Return the recognizer name with the lowest mean inference time."""
    best_name = None
    best_value = float("inf")
    for name, block in results.items():
        if not block.get("available"):
            continue
        timing = block.get("timing") or {}
        value = timing.get("mean_seconds")
        if value is None:
            continue
        if value < best_value:
            best_value = value
            best_name = name
    return best_name


def _build_summary(results: dict) -> dict:
    """Build a concise overall summary comparing the two recognizers."""
    return {
        "best_exact_match": _best_recognizer(results, "exact_match_rate"),
        "best_mean_levenshtein_similarity": _best_recognizer(
            results, "mean_levenshtein_similarity"
        ),
        "best_mean_symbol_accuracy": _best_recognizer(
            results, "mean_symbol_accuracy"
        ),
        "best_pass_rate": _best_recognizer(results, "pass_rate"),
        "fastest_mean_seconds": _fastest_recognizer(results),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_benchmark(
    data_dir: str = "data",
    results_dir: str = "results",
    results_file: str = DEFAULT_RESULTS_FILE,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    recognizers: dict | None = None,
    limit: int | None = None,
) -> dict:
    """Run both recognizers over the held-out test set and write results JSON.

    Parameters
    ----------
    data_dir : str
        Directory containing ``images/`` and ``manifest.json``.
    results_dir : str
        Directory where the results JSON is written.
    results_file : str
        Filename of the results JSON (default ``benchmark.json``).
    pass_threshold : float
        Similarity threshold that defines a "pass" (default 0.8).
    recognizers : dict | None
        Optional override of the recognizer registry (name -> callable). Used
        mainly for testing.
    limit : int | None
        If set, evaluate only the first ``limit`` test samples. Useful for
        quick smoke runs; the full benchmark omits it.

    Returns
    -------
    dict
        The full benchmark result structure (also written to disk).
    """
    try:
        samples = test_set(data_dir)
    except DatasetError as exc:
        raise BenchmarkError(
            f"Could not load test set from {data_dir!r}: {exc}"
        ) from exc
    if not samples:
        raise BenchmarkError(
            f"No test samples found in {data_dir!r}. Run dataset.generate() first."
        )
    if limit is not None:
        samples = samples[: max(0, int(limit))]
        if not samples:
            raise BenchmarkError("limit must be a positive integer")

    registry = recognizers if recognizers is not None else RECOGNIZERS
    results: dict[str, dict] = {}
    for name, recognize in registry.items():
        results[name] = _run_recognizer(recognize, samples, pass_threshold)

    payload = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_dir": data_dir,
            "results_dir": results_dir,
            "n_test_samples": len(samples),
            "tiers": sorted({s["tier"] for s in samples}),
            "pass_threshold": pass_threshold,
            "recognizers": list(registry.keys()),
        },
        "results": results,
        "summary": _build_summary(results),
    }

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, results_file)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="Benchmark the AI and own-code recognizers on the test set.",
    )
    parser.add_argument("--data-dir", default="data", help="dataset directory")
    parser.add_argument(
        "--results-dir", default="results", help="output results directory"
    )
    parser.add_argument(
        "--results-file",
        default=DEFAULT_RESULTS_FILE,
        help="results JSON filename",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=DEFAULT_PASS_THRESHOLD,
        help="similarity threshold that defines a pass",
    )
    parser.add_argument(
        "--skip-ai",
        action="store_true",
        help="skip the AI recognizer (useful when Ollama is down)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N test samples (quick smoke run)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = dict(RECOGNIZERS)
    if args.skip_ai:
        registry.pop("ai", None)
    payload = run_benchmark(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        results_file=args.results_file,
        pass_threshold=args.pass_threshold,
        recognizers=registry,
        limit=args.limit,
    )
    out_path = os.path.join(args.results_dir, args.results_file)
    print(f"Benchmark complete. Results written to {out_path}")
    for name, block in payload["results"].items():
        if block.get("available"):
            print(
                f"  {name}: exact={block['exact_match_rate']:.3f} "
                f"sim={block['mean_levenshtein_similarity']:.3f} "
                f"pass={block['pass_rate']:.3f} "
                f"mean={block['timing']['mean_seconds']:.3f}s"
            )
        else:
            print(f"  {name}: UNAVAILABLE ({block.get('error')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
