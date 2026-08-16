# Contributing to LaTeXOCR

Thanks for your interest in improving LaTeXOCR. This guide explains how to set up the project, run the checks, and submit changes.

## Development setup

Requirements: Python 3.11 or newer, and optionally Ollama with `qwen3-vl:8b` pulled for the AI engine.

```bash
pip install -r requirements.txt
```

## Running the checks

Run the full test suite:

```bash
python -m pytest tests/ -q
```

The suite covers preprocessing (including polarity and deskew regression guards), both recognizers, the benchmark harness, the report generator, and the CLI.

## Making a change

1. Fork the repository and create a feature branch.
2. Add or update tests for your change. New behavior should come with a test that fails before the change and passes after it.
3. Run the full test suite and make sure it passes.
4. If your change affects the benchmark or report, regenerate the sample charts with `python make_charts.py` and update the README numbers where relevant.
5. Open a pull request describing the problem, the change, and the evidence (test output, benchmark numbers).

## Code style

- Keep the code readable and commented where the logic is non-obvious (the own-code OCR pipeline in particular).
- Do not commit generated artifacts: `data/`, `results/`, `report/`, `.pi/`, `devcycle_ws/`, and `__pycache__/` are ignored.
- Do not commit real environment files or secrets.

## Reporting issues

For bug reports, include the input image (if possible), the expected LaTeX, the actual output, and the command you ran. For security issues, use the private reporting path in [SECURITY.md](SECURITY.md) instead of a public issue.
