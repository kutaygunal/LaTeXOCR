"""Report generator for the LaTeX OCR benchmark.

Reads the machine-readable ``results/benchmark.json`` produced by
``benchmark.py`` and turns it into a human-readable report comparing the two
recognition approaches (the local AI ``qwen3-vl:8b`` and the own-code OCR
pipeline). The report covers accuracy, speed, per-tier robustness, and a clear
recommendation.

Two output formats are produced under ``report/``:
- ``report.md``  : a plain-text Markdown report.
- ``report.html``: a self-contained, styled HTML report.

Public API
----------
- ``load_results(results_dir, results_file) -> dict`` : read the results JSON.
- ``build_markdown(payload) -> str``                   : Markdown report text.
- ``build_html(payload) -> str``                        : HTML report text.
- ``generate_report(results_dir, report_dir, ...) -> dict`` : write both files.
- ``main(argv) -> int`` : CLI entry point (``python -m src.report``).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from datetime import datetime

# Ensure the package root is importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Default filenames / directories.
DEFAULT_RESULTS_FILE = "benchmark.json"
DEFAULT_REPORT_DIR = "report"
DEFAULT_MARKDOWN_FILE = "report.md"
DEFAULT_HTML_FILE = "report.html"

# Human-readable labels for the two recognizers.
RECOGNIZER_LABELS = {
    "ai": "AI (qwen3-vl:8b)",
    "owncode": "Own-code OCR",
}

# Metrics shown in the summary table, in display order.
SUMMARY_METRICS = [
    ("exact_match_rate", "Exact match rate"),
    ("mean_levenshtein_similarity", "Mean Levenshtein similarity"),
    ("mean_symbol_accuracy", "Mean symbol accuracy"),
    ("pass_rate", "Pass rate"),
]

# Per-tier metrics shown in the per-tier tables, in display order.
TIER_METRICS = [
    ("exact_match_rate", "Exact match rate"),
    ("mean_levenshtein_similarity", "Mean Levenshtein similarity"),
    ("mean_symbol_accuracy", "Mean symbol accuracy"),
    ("pass_rate", "Pass rate"),
    ("mean_seconds", "Mean time (s)"),
]


class ReportError(Exception):
    """Raised when the report cannot be generated (e.g. missing results)."""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_pct(value: float | None) -> str:
    """Format a fraction in [0, 1] as a percentage string, or 'N/A'."""
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _fmt_sec(value: float | None) -> str:
    """Format a number of seconds, or 'N/A'."""
    if value is None:
        return "N/A"
    return f"{value:.3f}s"


def _fmt_int(value: int | None) -> str:
    """Format an integer, or 'N/A'."""
    if value is None:
        return "N/A"
    return str(value)


def _label(name: str) -> str:
    """Return the human-readable label for a recognizer name."""
    return RECOGNIZER_LABELS.get(name, name)


def _available_recognizers(payload: dict) -> list[str]:
    """Return recognizer names that produced usable results."""
    results = payload.get("results", {})
    return [name for name, block in results.items() if block.get("available")]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_results(
    results_dir: str = "results",
    results_file: str = DEFAULT_RESULTS_FILE,
) -> dict:
    """Read and return the benchmark results JSON.

    Raises ``ReportError`` if the file is missing or malformed.
    """
    path = os.path.join(results_dir, results_file)
    if not os.path.isfile(path):
        raise ReportError(
            f"Results file not found: {path}. Run the benchmark first."
        )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise ReportError(f"Results file is not valid JSON: {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _summary_table(payload: dict) -> str:
    """Build the Markdown summary table comparing the recognizers."""
    results = payload.get("results", {})
    names = _available_recognizers(payload)
    if not names:
        return "_No recognizer produced usable results._"

    header = "| Metric | " + " | ".join(_label(n) for n in names) + " |"
    sep = "|" + "---|" * (len(names) + 1)
    lines = [header, sep]

    for key, title in SUMMARY_METRICS:
        row = [f"**{title}**"]
        for name in names:
            block = results[name]
            row.append(_fmt_pct(block.get(key)))
        lines.append("| " + " | ".join(row) + " |")

    # Timing row (mean seconds).
    row = ["**Mean time (s)**"]
    for name in names:
        timing = results[name].get("timing") or {}
        row.append(_fmt_sec(timing.get("mean_seconds")))
    lines.append("| " + " | ".join(row) + " |")

    # Sample count row.
    row = ["**Samples**"]
    for name in names:
        row.append(_fmt_int(results[name].get("n")))
    lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _per_tier_section(payload: dict) -> str:
    """Build the Markdown per-tier robustness section."""
    results = payload.get("results", {})
    names = _available_recognizers(payload)
    tiers = payload.get("meta", {}).get("tiers", [])
    if not names or not tiers:
        return "_No per-tier data available._"

    blocks = []
    for tier in tiers:
        blocks.append(f"### Tier: `{tier}`")
        header = "| Metric | " + " | ".join(_label(n) for n in names) + " |"
        sep = "|" + "---|" * (len(names) + 1)
        lines = [header, sep]
        for key, title in TIER_METRICS:
            row = [f"**{title}**"]
            for name in names:
                tier_block = results[name].get("per_tier", {}).get(tier, {})
                value = tier_block.get(key)
                row.append(_fmt_pct(value) if key != "mean_seconds" else _fmt_sec(value))
            lines.append("| " + " | ".join(row) + " |")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _recommendation(payload: dict) -> str:
    """Build a clear recommendation paragraph from the summary."""
    summary = payload.get("summary", {})
    results = payload.get("results", {})
    names = _available_recognizers(payload)
    if not names:
        return (
            "No recognizer produced usable results, so no recommendation can "
            "be made. Check that Ollama is running and that the own-code "
            "pipeline is functional, then re-run the benchmark."
        )

    accuracy_winner = summary.get("best_exact_match")
    speed_winner = summary.get("fastest_mean_seconds")

    if len(names) == 1:
        return (
            f"Only **{_label(names[0])}** produced usable results. "
            "The comparison is therefore incomplete; re-run the benchmark with "
            "both recognizers available for a full recommendation."
        )

    if accuracy_winner == "ai":
        return (
            f"**{_label('ai')}** is the recommended approach for accuracy. "
            f"It wins on exact match, similarity, symbol accuracy, and pass "
            f"rate. **{_label('owncode')}** is faster "
            f"(mean {_fmt_sec(results['owncode'].get('timing', {}).get('mean_seconds'))} "
            f"vs {_fmt_sec(results['ai'].get('timing', {}).get('mean_seconds'))}) "
            "and runs fully offline with no external model dependency. "
            "Choose the AI recognizer when correctness matters most; choose "
            "the own-code pipeline when speed, offline operation, or low "
            "resource usage is the priority."
        )

    return (
        f"**{_label(accuracy_winner)}** leads on accuracy, while "
        f"**{_label(speed_winner)}** is the fastest. For the best balance of "
        "accuracy and speed, consider the own-code pipeline if its accuracy "
        "is acceptable for your use case, otherwise prefer the AI recognizer."
    )


def build_markdown(payload: dict) -> str:
    """Build the full Markdown report text from a results payload."""
    meta = payload.get("meta", {})
    results = payload.get("results", {})
    names = _available_recognizers(payload)

    lines = [
        "# LaTeX OCR Benchmark Report",
        "",
        f"- **Generated**: {meta.get('timestamp', 'unknown')}",
        f"- **Test samples**: {meta.get('n_test_samples', 'N/A')}",
        f"- **Tiers**: {', '.join(meta.get('tiers', [])) or 'N/A'}",
        f"- **Pass threshold**: {meta.get('pass_threshold', 'N/A')}",
        f"- **Recognizers**: {', '.join(_label(n) for n in names) or 'none'}",
        "",
        "## Summary",
        "",
        _summary_table(payload),
        "",
        "## Recommendation",
        "",
        _recommendation(payload),
        "",
        "## Per-tier robustness",
        "",
        _per_tier_section(payload),
        "",
    ]

    # Append per-recognizer notes (unavailable / errors).
    notes = []
    for name, block in results.items():
        if not block.get("available"):
            notes.append(
                f"- **{_label(name)}**: unavailable "
                f"({block.get('error', 'unknown reason')})."
            )
        elif block.get("error_count"):
            notes.append(
                f"- **{_label(name)}**: {block['error_count']} sample(s) failed "
                "during recognition."
            )
    if notes:
        lines += ["## Notes", ""] + notes + [""]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    """Escape text for safe embedding in HTML."""
    return html.escape(str(text))


def _html_table(payload: dict, per_tier: bool = False) -> str:
    """Build an HTML table (summary or per-tier) from the payload."""
    results = payload.get("results", {})
    names = _available_recognizers(payload)
    if not names:
        return "<p><em>No recognizer produced usable results.</em></p>"

    metrics = TIER_METRICS if per_tier else SUMMARY_METRICS
    rows = []
    for key, title in metrics:
        cells = [f"<td class='metric'>{_esc(title)}</td>"]
        for name in names:
            block = results[name]
            if per_tier:
                value = block.get("per_tier", {}).get(per_tier, {}).get(key)
            else:
                value = block.get(key)
            if key == "mean_seconds":
                cells.append(f"<td>{_fmt_sec(value)}</td>")
            else:
                cells.append(f"<td>{_fmt_pct(value)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    # Timing + sample count rows for the summary table.
    if not per_tier:
        row = ["<td class='metric'>Mean time (s)</td>"]
        for name in names:
            timing = results[name].get("timing") or {}
            row.append(f"<td>{_fmt_sec(timing.get('mean_seconds'))}</td>")
        rows.append("<tr>" + "".join(row) + "</tr>")

        row = ["<td class='metric'>Samples</td>"]
        for name in names:
            row.append(f"<td>{_fmt_int(results[name].get('n'))}</td>")
        rows.append("<tr>" + "".join(row) + "</tr>")

    header = "".join(f"<th>{_esc(_label(n))}</th>" for n in names)
    return (
        "<table><thead><tr><th>Metric</th>"
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _html_per_tier(payload: dict) -> str:
    """Build the HTML per-tier robustness section."""
    tiers = payload.get("meta", {}).get("tiers", [])
    if not tiers:
        return "<p><em>No per-tier data available.</em></p>"
    sections = []
    for tier in tiers:
        sections.append(
            f"<h3>Tier: <code>{_esc(tier)}</code></h3>"
            + _html_table(payload, per_tier=tier)
        )
    return "\n".join(sections)


def build_html(payload: dict) -> str:
    """Build a self-contained, styled HTML report from a results payload."""
    meta = payload.get("meta", {})
    results = payload.get("results", {})
    names = _available_recognizers(payload)

    notes = []
    for name, block in results.items():
        if not block.get("available"):
            notes.append(
                f"<li><strong>{_esc(_label(name))}</strong>: unavailable "
                f"({_esc(block.get('error', 'unknown reason'))}).</li>"
            )
        elif block.get("error_count"):
            notes.append(
                f"<li><strong>{_esc(_label(name))}</strong>: "
                f"{block['error_count']} sample(s) failed during recognition.</li>"
            )
    notes_html = (
        "<h2>Notes</h2><ul>" + "".join(notes) + "</ul>"
        if notes
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LaTeX OCR Benchmark Report</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         margin: 2rem auto; max-width: 900px; padding: 0 1rem; color: #1a1a1a; }}
  h1 {{ border-bottom: 2px solid #4a6fa5; padding-bottom: .3rem; }}
  h2 {{ color: #2c3e50; margin-top: 2rem; }}
  h3 {{ color: #34495e; }}
  .meta {{ color: #555; font-size: .95rem; }}
  .meta li {{ margin: .15rem 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #d0d7de; padding: .5rem .75rem; text-align: left; }}
  th {{ background: #f0f4f8; }}
  td.metric {{ font-weight: 600; }}
  tr:nth-child(even) td {{ background: #fafbfc; }}
  .recommendation {{ background: #eef4fb; border-left: 4px solid #4a6fa5;
                    padding: .75rem 1rem; border-radius: 4px; }}
  code {{ background: #f0f0f0; padding: .1rem .3rem; border-radius: 3px; }}
</style>
</head>
<body>
<h1>LaTeX OCR Benchmark Report</h1>
<ul class="meta">
  <li><strong>Generated</strong>: {_esc(meta.get('timestamp', 'unknown'))}</li>
  <li><strong>Test samples</strong>: {_esc(meta.get('n_test_samples', 'N/A'))}</li>
  <li><strong>Tiers</strong>: {_esc(', '.join(meta.get('tiers', [])) or 'N/A')}</li>
  <li><strong>Pass threshold</strong>: {_esc(meta.get('pass_threshold', 'N/A'))}</li>
  <li><strong>Recognizers</strong>: {_esc(', '.join(_label(n) for n in names) or 'none')}</li>
</ul>

<h2>Summary</h2>
{_html_table(payload)}

<h2>Recommendation</h2>
<div class="recommendation">{_esc(_recommendation(payload))}</div>

<h2>Per-tier robustness</h2>
{_html_per_tier(payload)}

{notes_html}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(
    results_dir: str = "results",
    report_dir: str = DEFAULT_REPORT_DIR,
    results_file: str = DEFAULT_RESULTS_FILE,
    markdown_file: str = DEFAULT_MARKDOWN_FILE,
    html_file: str = DEFAULT_HTML_FILE,
) -> dict:
    """Read the benchmark results and write Markdown + HTML reports.

    Parameters
    ----------
    results_dir : str
        Directory containing the results JSON.
    report_dir : str
        Directory where the reports are written.
    results_file : str
        Filename of the results JSON (default ``benchmark.json``).
    markdown_file : str
        Filename of the Markdown report (default ``report.md``).
    html_file : str
        Filename of the HTML report (default ``report.html``).

    Returns
    -------
    dict
        Mapping of report kind -> absolute output path.
    """
    payload = load_results(results_dir, results_file)

    os.makedirs(report_dir, exist_ok=True)
    md_path = os.path.join(report_dir, markdown_file)
    html_path = os.path.join(report_dir, html_file)

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(build_markdown(payload))
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(build_html(payload))

    return {
        "markdown": os.path.abspath(md_path),
        "html": os.path.abspath(html_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="report",
        description="Generate a human-readable report from benchmark results.",
    )
    parser.add_argument("--results-dir", default="results", help="results directory")
    parser.add_argument(
        "--results-file",
        default=DEFAULT_RESULTS_FILE,
        help="results JSON filename",
    )
    parser.add_argument(
        "--report-dir", default=DEFAULT_REPORT_DIR, help="output report directory"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = generate_report(
        results_dir=args.results_dir,
        report_dir=args.report_dir,
        results_file=args.results_file,
    )
    print(f"Report written to:")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
