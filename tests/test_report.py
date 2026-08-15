"""Unit and integration tests for src/report.py.

Covers: load_results (valid/missing/malformed JSON), build_markdown,
build_html, generate_report (writes both files), and handling of
unavailable / single recognizer payloads.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from report import (  # noqa: E402
    ReportError,
    _available_recognizers,
    _fmt_int,
    _fmt_pct,
    _fmt_sec,
    _html_per_tier,
    _html_table,
    _label,
    _per_tier_section,
    _recommendation,
    _summary_table,
    build_html,
    build_markdown,
    generate_report,
    load_results,
)


def _payload():
    return {
        "meta": {
            "timestamp": "2025-01-01T00:00:00+00:00",
            "data_dir": "data",
            "n_test_samples": 900,
            "tiers": ["clean", "noisy"],
            "pass_threshold": 0.8,
            "recognizers": ["ai", "owncode"],
        },
        "results": {
            "ai": {
                "available": True, "n": 900,
                "exact_match_rate": 0.5, "mean_levenshtein_similarity": 0.7,
                "mean_symbol_accuracy": 0.6, "pass_rate": 0.8,
                "timing": {"mean_seconds": 2.0},
                "per_tier": {
                    "clean": {"n": 450, "exact_match_rate": 0.6,
                              "mean_levenshtein_similarity": 0.8,
                              "mean_symbol_accuracy": 0.7, "pass_rate": 0.9,
                              "mean_seconds": 2.0},
                    "noisy": {"n": 450, "exact_match_rate": 0.4,
                              "mean_levenshtein_similarity": 0.6,
                              "mean_symbol_accuracy": 0.5, "pass_rate": 0.7,
                              "mean_seconds": 2.0},
                },
            },
            "owncode": {
                "available": True, "n": 900,
                "exact_match_rate": 0.3, "mean_levenshtein_similarity": 0.5,
                "mean_symbol_accuracy": 0.4, "pass_rate": 0.6,
                "timing": {"mean_seconds": 0.1},
                "per_tier": {
                    "clean": {"n": 450, "exact_match_rate": 0.4,
                              "mean_levenshtein_similarity": 0.6,
                              "mean_symbol_accuracy": 0.5, "pass_rate": 0.7,
                              "mean_seconds": 0.1},
                    "noisy": {"n": 450, "exact_match_rate": 0.2,
                              "mean_levenshtein_similarity": 0.4,
                              "mean_symbol_accuracy": 0.3, "pass_rate": 0.5,
                              "mean_seconds": 0.1},
                },
            },
        },
        "summary": {
            "best_exact_match": "ai",
            "best_mean_levenshtein_similarity": "ai",
            "best_mean_symbol_accuracy": "ai",
            "best_pass_rate": "ai",
            "fastest_mean_seconds": "owncode",
        },
    }


def _write_results(tmp_path, payload):
    d = str(tmp_path / "results")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "benchmark.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return d


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def test_fmt_pct():
    assert _fmt_pct(0.5) == "50.0%"
    assert _fmt_pct(1.0) == "100.0%"
    assert _fmt_pct(None) == "N/A"


def test_fmt_sec():
    assert _fmt_sec(2.0) == "2.000s"
    assert _fmt_sec(None) == "N/A"


def test_fmt_int():
    assert _fmt_int(900) == "900"
    assert _fmt_int(0) == "0"
    assert _fmt_int(None) == "N/A"


def test_label():
    assert _label("ai") == "AI (qwen3-vl:8b)"
    assert _label("owncode") == "Own-code OCR"
    assert _label("unknown") == "unknown"


def test_available_recognizers():
    p = _payload()
    assert _available_recognizers(p) == ["ai", "owncode"]
    p["results"]["ai"]["available"] = False
    assert _available_recognizers(p) == ["owncode"]


# ---------------------------------------------------------------------------
# _recommendation branches
# ---------------------------------------------------------------------------


def test_recommendation_ai_wins_accuracy():
    p = _payload()  # best_exact_match == "ai"
    rec = _recommendation(p)
    assert "is the recommended approach for accuracy" in rec
    assert "AI (qwen3-vl:8b)" in rec
    assert "Own-code OCR" in rec
    # Speed comparison: owncode faster (0.1s) vs ai (2.0s).
    assert "0.100s" in rec
    assert "2.000s" in rec
    assert "runs fully offline" in rec


def test_recommendation_ai_wins_in_markdown():
    md = build_markdown(_payload())
    assert "is the recommended approach for accuracy" in md


def test_recommendation_owncode_wins_accuracy():
    p = _payload()
    p["summary"]["best_exact_match"] = "owncode"
    p["summary"]["fastest_mean_seconds"] = "owncode"
    rec = _recommendation(p)
    assert "leads on accuracy, while" in rec
    assert "is the fastest" in rec
    assert "Own-code OCR" in rec


def test_recommendation_owncode_wins_in_markdown():
    p = _payload()
    p["summary"]["best_exact_match"] = "owncode"
    p["summary"]["fastest_mean_seconds"] = "owncode"
    md = build_markdown(p)
    assert "leads on accuracy, while" in md


def test_recommendation_no_available_recognizers():
    p = _payload()
    p["results"]["ai"]["available"] = False
    p["results"]["owncode"]["available"] = False
    rec = _recommendation(p)
    assert "no recommendation can be made" in rec
    assert "re-run the benchmark" in rec


def test_recommendation_single_recognizer():
    p = _payload()
    p["results"]["ai"]["available"] = False
    rec = _recommendation(p)
    assert "Only **Own-code OCR** produced usable results." in rec


# ---------------------------------------------------------------------------
# Table / section no-data branches
# ---------------------------------------------------------------------------


def test_summary_table_no_recognizers():
    p = _payload()
    p["results"]["ai"]["available"] = False
    p["results"]["owncode"]["available"] = False
    assert "_No recognizer produced usable results._" in _summary_table(p)


def test_per_tier_section_no_data():
    p = _payload()
    p["results"]["ai"]["available"] = False
    p["results"]["owncode"]["available"] = False
    assert "_No per-tier data available._" in _per_tier_section(p)


def test_per_tier_section_empty_tiers():
    p = _payload()
    p["meta"]["tiers"] = []
    assert "_No per-tier data available._" in _per_tier_section(p)


def test_html_table_no_recognizers():
    p = _payload()
    p["results"]["ai"]["available"] = False
    p["results"]["owncode"]["available"] = False
    assert "No recognizer produced usable results" in _html_table(p)


def test_html_per_tier_empty_tiers():
    p = _payload()
    p["meta"]["tiers"] = []
    assert "No per-tier data available" in _html_per_tier(p)


# ---------------------------------------------------------------------------
# load_results
# ---------------------------------------------------------------------------


def test_load_results_valid(tmp_path):
    d = _write_results(tmp_path, _payload())
    loaded = load_results(d)
    assert loaded == _payload()


def test_load_results_missing_raises(tmp_path):
    with pytest.raises(ReportError):
        load_results(str(tmp_path / "nope"))


def test_load_results_malformed_raises(tmp_path):
    d = str(tmp_path / "results")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "benchmark.json"), "w", encoding="utf-8") as fh:
        fh.write("{ not valid json")
    with pytest.raises(ReportError):
        load_results(d)


# ---------------------------------------------------------------------------
# build_markdown
# ---------------------------------------------------------------------------


def test_build_markdown_structure():
    md = build_markdown(_payload())
    assert "# LaTeX OCR Benchmark Report" in md
    assert "## Summary" in md
    assert "## Recommendation" in md
    assert "## Per-tier robustness" in md
    assert "AI (qwen3-vl:8b)" in md
    assert "Own-code OCR" in md
    assert "50.0%" in md  # ai exact match
    assert "2.000s" in md  # ai mean time
    assert "### Tier: `clean`" in md
    assert "### Tier: `noisy`" in md


def test_build_markdown_single_recognizer():
    p = _payload()
    p["results"]["ai"]["available"] = False
    md = build_markdown(p)
    assert "Only **Own-code OCR** produced usable results." in md
    # The unavailable AI recognizer is reported in the Notes section.
    assert "AI (qwen3-vl:8b)" in md
    assert "unavailable" in md


def test_build_markdown_no_recognizer():
    p = _payload()
    p["results"]["ai"]["available"] = False
    p["results"]["owncode"]["available"] = False
    md = build_markdown(p)
    assert "_No recognizer produced usable results._" in md


def test_build_markdown_notes_for_errors():
    p = _payload()
    p["results"]["owncode"]["error_count"] = 3
    md = build_markdown(p)
    assert "## Notes" in md
    assert "3 sample(s) failed" in md


def test_build_markdown_notes_for_unavailable():
    p = _payload()
    p["results"]["ai"]["available"] = False
    p["results"]["ai"]["error"] = "recognizer unavailable (probe failed)"
    md = build_markdown(p)
    assert "unavailable" in md


# ---------------------------------------------------------------------------
# build_html
# ---------------------------------------------------------------------------


def test_build_html_structure():
    h = build_html(_payload())
    assert "<!DOCTYPE html>" in h
    assert "<html" in h
    assert "<table>" in h
    assert "LaTeX OCR Benchmark Report" in h
    assert "AI (qwen3-vl:8b)" in h
    assert "Own-code OCR" in h
    assert "50.0%" in h
    assert "Tier: <code>clean</code>" in h


def test_build_html_escapes_content():
    p = _payload()
    p["results"]["ai"]["error"] = "<script>alert('x')</script>"
    p["results"]["ai"]["available"] = False
    h = build_html(p)
    assert "<script>" not in h
    assert "&lt;script&gt;" in h


def test_build_html_no_recognizer():
    p = _payload()
    p["results"]["ai"]["available"] = False
    p["results"]["owncode"]["available"] = False
    h = build_html(p)
    assert "No recognizer produced usable results" in h


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


def test_generate_report_writes_both_files(tmp_path):
    d = _write_results(tmp_path, _payload())
    report_dir = str(tmp_path / "report")
    paths = generate_report(results_dir=d, report_dir=report_dir)
    assert os.path.isfile(paths["markdown"])
    assert os.path.isfile(paths["html"])
    assert paths["markdown"].endswith("report.md")
    assert paths["html"].endswith("report.html")
    with open(paths["markdown"], encoding="utf-8") as fh:
        assert "# LaTeX OCR Benchmark Report" in fh.read()
    with open(paths["html"], encoding="utf-8") as fh:
        assert "<!DOCTYPE html>" in fh.read()


def test_generate_report_missing_results_raises(tmp_path):
    with pytest.raises(ReportError):
        generate_report(
            results_dir=str(tmp_path / "nope"),
            report_dir=str(tmp_path / "report"),
        )
