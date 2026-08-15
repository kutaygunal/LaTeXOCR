"""Integration tests for the src/main.py CLI.

Covers: generate-dataset, preprocess, recognize (owncode), benchmark, report,
run-all subcommands, and error handling for bad args / missing files.

The AI recognizer is avoided in these tests (--skip-ai) so no live Ollama call
is required; the owncode recognizer is exercised on a real image.
"""

import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from main import build_parser, main  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")


def _sample_image(tier: str) -> str:
    for fname in sorted(os.listdir(IMAGES_DIR)):
        if f"_{tier}_" in fname:
            return os.path.join(IMAGES_DIR, fname)
    raise FileNotFoundError(f"no image for tier {tier!r}")


# ---------------------------------------------------------------------------
# generate-dataset
# ---------------------------------------------------------------------------


def test_cli_generate_dataset(tmp_path):
    data_dir = str(tmp_path / "data")
    rc = main(["generate-dataset", "--data-dir", data_dir, "--per-tier", "1"])
    assert rc == 0
    assert os.path.isfile(os.path.join(data_dir, "manifest.json"))
    assert os.path.isdir(os.path.join(data_dir, "images"))


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


def test_cli_preprocess(tmp_path):
    out = str(tmp_path / "out.png")
    rc = main(["preprocess", _sample_image("clean"), "--height", "48", "--out", out])
    assert rc == 0
    assert os.path.isfile(out)
    img = cv2.imread(out, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    assert img.shape[0] == 48


def test_cli_preprocess_missing_file_raises(tmp_path):
    with pytest.raises(Exception):
        main(["preprocess", str(tmp_path / "nope.png")])


# ---------------------------------------------------------------------------
# recognize
# ---------------------------------------------------------------------------


def test_cli_recognize_owncode():
    rc = main(["recognize", _sample_image("clean"), "--recognizer", "owncode"])
    assert rc == 0


def test_cli_recognize_default_is_ai():
    # The default recognizer is 'ai'; parser must accept it.
    args = build_parser().parse_args(["recognize", "img.png"])
    assert args.recognizer == "ai"


def test_cli_recognize_invalid_recognizer_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["recognize", "img.png", "--recognizer", "bogus"])


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


def test_cli_benchmark_skip_ai(tmp_path):
    results_dir = str(tmp_path / "results")
    rc = main([
        "benchmark", "--data-dir", DATA_DIR,
        "--results-dir", results_dir, "--skip-ai", "--limit", "3",
    ])
    assert rc == 0
    out = os.path.join(results_dir, "benchmark.json")
    assert os.path.isfile(out)
    with open(out, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert set(payload["results"].keys()) == {"owncode"}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_cli_report(tmp_path):
    results_dir = str(tmp_path / "results")
    report_dir = str(tmp_path / "report")
    # First produce results via benchmark.
    main([
        "benchmark", "--data-dir", DATA_DIR,
        "--results-dir", results_dir, "--skip-ai", "--limit", "3",
    ])
    rc = main([
        "report", "--results-dir", results_dir, "--report-dir", report_dir,
    ])
    assert rc == 0
    assert os.path.isfile(os.path.join(report_dir, "report.md"))
    assert os.path.isfile(os.path.join(report_dir, "report.html"))


def test_cli_report_missing_results_raises(tmp_path):
    with pytest.raises(Exception):
        main([
            "report", "--results-dir", str(tmp_path / "nope"),
            "--report-dir", str(tmp_path / "report"),
        ])


# ---------------------------------------------------------------------------
# run-all
# ---------------------------------------------------------------------------


def test_cli_run_all(tmp_path):
    data_dir = str(tmp_path / "data")
    results_dir = str(tmp_path / "results")
    report_dir = str(tmp_path / "report")
    rc = main([
        "run-all", "--data-dir", data_dir,
        "--results-dir", results_dir, "--report-dir", report_dir,
        "--per-tier", "1", "--skip-ai", "--limit", "3",
    ])
    assert rc == 0
    assert os.path.isfile(os.path.join(data_dir, "manifest.json"))
    assert os.path.isfile(os.path.join(results_dir, "benchmark.json"))
    assert os.path.isfile(os.path.join(report_dir, "report.md"))
    assert os.path.isfile(os.path.join(report_dir, "report.html"))


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


def test_cli_unknown_subcommand_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["bogus-command"])


def test_cli_no_command_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
