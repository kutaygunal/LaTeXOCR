"""CLI entry point for the LaTeX Image -> LaTeX String converter.

Wires the full pipeline: dataset generation, preprocessing, single-image
recognition, benchmarking, and report generation. A ``run-all`` subcommand
chains the pipeline end-to-end.

Usage
-----
    python -m src.main generate-dataset [--per-tier N] [--seed S] [--data-dir D]
    python -m src.main preprocess IMAGE [--height H] [--out O]
    python -m src.main recognize IMAGE [--recognizer ai|owncode]
    python -m src.main benchmark [--data-dir D] [--results-dir R] [--skip-ai]
    python -m src.main report [--results-dir R] [--report-dir P]
    python -m src.main run-all [--data-dir D] [--results-dir R] [--report-dir P]
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure the package root is importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_recognizer import recognize as ai_recognize  # noqa: E402
from benchmark import run_benchmark  # noqa: E402
from dataset import DatasetError, generate  # noqa: E402
from owncode_recognizer import recognize as owncode_recognize  # noqa: E402
from preprocess import preprocess  # noqa: E402
from report import generate_report  # noqa: E402

# Recognizer registry: name -> callable(image_path) -> str.
RECOGNIZERS = {
    "ai": ai_recognize,
    "owncode": owncode_recognize,
}


def _cmd_generate_dataset(args: argparse.Namespace) -> int:
    manifest = generate(
        data_dir=args.data_dir,
        per_tier=args.per_tier,
        seed=args.seed,
    )
    print(f"Dataset generated. Manifest: {manifest}")
    return 0


def _cmd_preprocess(args: argparse.Namespace) -> int:
    result = preprocess(args.image, height=args.height)
    out = args.out or os.path.splitext(args.image)[0] + "_preprocessed.png"
    import cv2

    cv2.imwrite(out, result)
    print(f"Preprocessed {args.image} -> {out} (shape={result.shape})")
    return 0


def _cmd_recognize(args: argparse.Namespace) -> int:
    recognize = RECOGNIZERS[args.recognizer]
    latex = recognize(args.image)
    print(latex)
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
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


def _cmd_report(args: argparse.Namespace) -> int:
    paths = generate_report(
        results_dir=args.results_dir,
        report_dir=args.report_dir,
        results_file=args.results_file,
    )
    print("Report written to:")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")
    return 0


def _cmd_run_all(args: argparse.Namespace) -> int:
    """Chain the full pipeline: dataset -> benchmark -> report."""
    # 1. Ensure the dataset exists (generate if the manifest is missing).
    manifest = os.path.join(args.data_dir, "manifest.json")
    if not os.path.isfile(manifest):
        print(f"Dataset manifest not found; generating dataset in {args.data_dir}...")
        generate(data_dir=args.data_dir, per_tier=args.per_tier, seed=args.seed)
    else:
        print(f"Dataset already present: {manifest}")

    # 2. Run the benchmark (preprocess + recognize over the test set).
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
    print(f"Benchmark complete. Results written to "
          f"{os.path.join(args.results_dir, args.results_file)}")

    # 3. Generate the report.
    paths = generate_report(
        results_dir=args.results_dir,
        report_dir=args.report_dir,
        results_file=args.results_file,
    )
    print("Report written to:")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latex-ocr",
        description="Convert LaTeX equation images into LaTeX strings.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-dataset", help="Render the ground-truth dataset")
    gen.add_argument("--per-tier", type=int, default=20, help="samples per tier")
    gen.add_argument("--seed", type=int, default=42, help="deterministic split seed")
    gen.add_argument("--data-dir", default="data", help="output data directory")
    gen.set_defaults(func=_cmd_generate_dataset)

    pre = sub.add_parser("preprocess", help="Preprocess a single image")
    pre.add_argument("image", help="path to the input image")
    pre.add_argument("--height", type=int, default=64, help="canonical output height")
    pre.add_argument("--out", default=None, help="output image path")
    pre.set_defaults(func=_cmd_preprocess)

    rec = sub.add_parser("recognize", help="Recognize a single image")
    rec.add_argument("image", help="path to the input image")
    rec.add_argument(
        "--recognizer",
        choices=sorted(RECOGNIZERS),
        default="ai",
        help="recognizer to use (default: ai)",
    )
    rec.set_defaults(func=_cmd_recognize)

    bench = sub.add_parser("benchmark", help="Benchmark both recognizers")
    bench.add_argument("--data-dir", default="data", help="dataset directory")
    bench.add_argument("--results-dir", default="results", help="results directory")
    bench.add_argument(
        "--results-file", default="benchmark.json", help="results JSON filename"
    )
    bench.add_argument(
        "--pass-threshold",
        type=float,
        default=0.8,
        help="similarity threshold that defines a pass",
    )
    bench.add_argument(
        "--skip-ai",
        action="store_true",
        help="skip the AI recognizer (useful when Ollama is down)",
    )
    bench.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N test samples (quick smoke run)",
    )
    bench.set_defaults(func=_cmd_benchmark)

    rep = sub.add_parser("report", help="Generate a report from benchmark results")
    rep.add_argument("--results-dir", default="results", help="results directory")
    rep.add_argument(
        "--results-file", default="benchmark.json", help="results JSON filename"
    )
    rep.add_argument("--report-dir", default="report", help="output report directory")
    rep.set_defaults(func=_cmd_report)

    run = sub.add_parser("run-all", help="Run the full pipeline end-to-end")
    run.add_argument("--data-dir", default="data", help="dataset directory")
    run.add_argument("--results-dir", default="results", help="results directory")
    run.add_argument(
        "--results-file", default="benchmark.json", help="results JSON filename"
    )
    run.add_argument("--report-dir", default="report", help="output report directory")
    run.add_argument("--per-tier", type=int, default=20, help="samples per tier")
    run.add_argument("--seed", type=int, default=42, help="deterministic split seed")
    run.add_argument(
        "--pass-threshold",
        type=float,
        default=0.8,
        help="similarity threshold that defines a pass",
    )
    run.add_argument(
        "--skip-ai",
        action="store_true",
        help="skip the AI recognizer (useful when Ollama is down)",
    )
    run.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N test samples (quick smoke run)",
    )
    run.set_defaults(func=_cmd_run_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
