<div align="center">

# 📐 LaTeXOCR

**Convert images of math equations into real LaTeX, with two competing engines and a built-in benchmark.**

A Python tool that takes a photo or screenshot of a formula and returns editable **LaTeX**. It ships with **two recognition approaches** — a **local AI vision model** (Ollama `qwen3-vl:8b`) and a fully **hand-written OCR pipeline** (OpenCV + font metrics) — and a **benchmark harness** that measures both on the same held-out test set so you can see which to use and when.

On the held-out test set the hand-written pipeline now **beats the vision model on accuracy** (94% vs 91% symbol accuracy) while running **155× faster** (26 ms vs 4.0 s per image).

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-qwen3--vl--8b-1F8ACB?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26+-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> Turn a picture of a formula into `\LaTeX{}` you can edit — and see exactly which engine is best for your input.

</div>

---

## 📖 Overview

**LaTeXOCR** reads an image that contains a rendered math expression and outputs a **valid LaTeX string**. It is built around two very different recognition engines:

1. **Local AI** — a vision-language model (`qwen3-vl:8b`) running locally via **Ollama**. It "sees" the image the way a person would and writes the LaTeX.
2. **Own-code OCR** — a hand-written computer-vision pipeline (no external AI) that segments glyphs, identifies each one by matching it against a bank of rendered templates **scored on shape, proportions, size and baseline position**, and reconstructs the expression structure with a recursive layout parser.

A shared **benchmark harness** runs both engines over the **same held-out test set** — split into five difficulty tiers (`clean`, `noisy`, `low_res`, `black_bg`, `white_bg`) — and produces accuracy, speed, and robustness numbers plus a Markdown/HTML report.

> ⚠️ **Data-integrity by design.** The dataset is split into **disjoint train/test sets**. The own-code pipeline builds its templates by rendering symbols, never by reading the dataset, and its thresholds were tuned against the **train** split only; the benchmark evaluates both engines **only** on the held-out test split — so the comparison is fair and never inflated by leakage.

---

## 🧪 Sample Conversions

Here is the numerator of the **quadratic formula** rendered as an image, and the LaTeX both engines read back from it:

![Quadratic formula](samples/quadratic.png)

```latex
-b \pm \sqrt{b^2 - 4ac}
```

Both engines transcribe it exactly — including the nested superscript inside the radical and the spacing around the infix operators.

Two more expressions, side by side:

| Input image | Ground truth | Local AI | Own-code OCR |
|---|---|---|---|
| ![Fraction](samples/frac.png) | `\frac{a}{b}` | `\frac{a}{b}` ✅ | `\frac{a}{b}` ✅ |
| ![Pythagorean](samples/pythag.png) | `x^2 + y^2 = z^2` | `x^2 + y^2 = z^2` ✅ | `x^2 + y^2 = z^2` ✅ |

> 💡 **Result:** on clean inputs the two engines agree character for character. Where they differ is on **degraded inputs** and on **speed** — see the benchmark below.

---

## 📊 Benchmark

Both engines were evaluated on a **sampled, held-out test set** (12 images per difficulty tier, 60 images total). The charts below show **recognition accuracy** and **inference speed** per tier, plus the aggregate across all tiers:

<p align="center">
  <img src="samples/benchmark_accuracy.png" alt="Recognition accuracy by difficulty tier" width="90%">
</p>

<p align="center">
  <img src="samples/benchmark_speed.png" alt="Inference speed by difficulty tier" width="90%">
</p>

| Tier | AI — symbol acc. | Own-code — symbol acc. | AI — time | Own-code — time |
|---|---|---|---|---|
| `clean` | 94% | **100%** | 3.9s | **25ms** |
| `noisy` | **100%** | 93% | 3.9s | **27ms** |
| `low_res` | **84%** | 81% | 4.9s | **26ms** |
| `black_bg` | 91% | **99%** | 3.6s | **24ms** |
| `white_bg` | 86% | **95%** | 3.6s | **26ms** |
| **Aggregate** | 91% | **94%** | 3.98s | **26ms** |

Run over the **entire** 900-image held-out test set, the own-code engine scores **94.3% symbol accuracy**, **97.0% Levenshtein similarity** and **87.6% exact match**, at **23ms per image**.

> 📌 **Recommendation:** the **own-code engine is the default choice** — higher accuracy than the vision model on aggregate (94% vs 91%), 155× faster, no model to download and nothing to run alongside it. Reach for the **local AI engine** on **inputs the pipeline was not built for** — handwriting, photographs, unusual fonts, or notation outside the symbol library — where a vision model degrades gracefully and a template matcher does not. The AI also stays ahead on the `noisy` and `low_res` tiers.

*Reproduce it:*

```bash
python benchmark_tiers.py 12 results/benchmark_tiers.json   # sample benchmark across all 5 tiers
python -m src.benchmark --limit 6                            # quick smoke run
python -m src.benchmark                                      # full held-out test set
```

---

## ✨ Features

- **Two recognition engines** — local AI vision model + hand-written OCR pipeline
- **`recognize <image>`** — convert a single image to LaTeX with either engine
- **Font-metric classification** — every glyph is scored on shape, proportions, size and baseline position, which is what separates `.` from `\cdot`, `o` from `O`, and an integral sign from a bold `I`
- **Glyph repair** — reassembles the pieces of `=`, `i`, `j`, `!` and `\pm`, and splits apart symbols that were printed touching (a `\sum` and the limit stacked on it)
- **Layout parser** — fractions, radicals with indices, binomials, accents, stacked and side-set limits, and multi-letter function names recovered with a lexicon
- **Idiomatic output** — spaces around infix operators, thin spaces before differentials, tight sub/superscripts: LaTeX that reads the way it would be typed
- **Robust preprocessing** — Otsu binarization, median denoising, speckle removal, projection-profile deskew, auto-crop, and polarity normalization (handles white *and* black backgrounds)
- **Ground-truth dataset generator** — renders a curated set of expressions across **5 difficulty tiers**
- **Fair benchmarking** — disjoint train/test split, per-tier accuracy/speed/robustness, graceful handling when Ollama is offline
- **Automatic report** — Markdown + styled HTML comparing both engines with a clear recommendation
- **Full CLI** — `generate-dataset`, `preprocess`, `recognize`, `benchmark`, `report`, and `run-all`
- **302 passing tests** covering preprocessing, noise and skew handling, both recognizers, layout analysis, the template bank, the benchmark, and the report

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Local AI engine | Ollama `qwen3-vl:8b` (HTTP API) |
| Vision / image processing | OpenCV 4.9+, NumPy 1.26+, Pillow 10+ |
| Symbol rendering | matplotlib mathtext, SymPy |
| Symbol classification | rendered template bank + font metrics (optional PyTorch CNN boost) |
| Dataset / metrics | custom `dataset.py`, `metrics.py` |
| Testing | `pytest` (302 tests) |

---

## 📁 Repository Structure

```text
LaTeXOCR/
├── src/
│   ├── preprocess.py            # Shared image loading + preprocessing
│   ├── dataset.py               # Ground-truth generator (train/test split)
│   ├── metrics.py               # Levenshtein, symbol accuracy, timing, per-tier
│   ├── ai_recognizer.py         # Local AI engine (Ollama qwen3-vl:8b)
│   ├── owncode_recognizer.py    # Own-code OCR engine (segmentation, metrics, layout)
│   ├── symbols.py               # Template bank: symbol variants + font metrics
│   ├── benchmark.py             # Benchmark harness (both engines, test set)
│   ├── report.py                # Markdown + HTML report generator
│   └── main.py                  # CLI entry point (full pipeline)
├── tests/                       # 302 pytest tests
├── samples/                     # Sample images + benchmark chart (used in README)
├── benchmark_tiers.py           # Per-tier benchmark for charting
├── make_charts.py               # Generates samples/benchmark_accuracy.png + benchmark_speed.png
├── requirements.txt             # Runtime dependencies
└── README.md                    # This file
```

---

## 🔨 Installation

### Requirements

| Component | Notes |
|---|---|
| Python | 3.11 or higher |
| Ollama | with `qwen3-vl:8b` pulled (`ollama pull qwen3-vl:8b`) — for the AI engine |

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Recognize a single image

```bash
# Use the local AI engine (default)
python -m src.main recognize path/to/equation.png --recognizer ai

# Use the hand-written OCR engine
python -m src.main recognize path/to/equation.png --recognizer owncode
```

### Generate the ground-truth dataset

```bash
python -m src.main generate-dataset
```

### Run the full benchmark (both engines, held-out test set)

```bash
python -m src.main benchmark
```

### Generate the comparison report

```bash
python -m src.main report        # writes report/report.md and report/report.html
```

### Run everything end-to-end

```bash
python -m src.main run-all
```

### Programmatic use

```python
from ai_recognizer import AIRecognizer
import owncode_recognizer as ocr

latex_ai = AIRecognizer().recognize("samples/quadratic.png")   # -> "-b \pm \sqrt{b^2 - 4ac}"
latex_oc = ocr.recognize("samples/frac.png")                   # -> "\frac{a}{b}"
```

---

## ⚙️ How it works

1. **Preprocess** — the image is loaded, converted to grayscale, binarized (Otsu),
   denoised, deskewed, auto-cropped, and normalized to a canonical height with
   consistent black-on-white polarity. Skew is found by rotating the image to
   maximize the sharpness of its horizontal projection, and a straight image is
   left untouched rather than run through a needless interpolation.
2. **Recognize** — the preprocessed (or raw) image goes to either:
   - **AI engine**: base64-encoded and sent to `qwen3-vl:8b` via Ollama's HTTP API
     with a tuned prompt; the raw output is cleaned into a single LaTeX line.
   - **Own-code engine**: see below.
3. **Benchmark** — both engines run over the held-out **test set**; metrics (exact
   match, Levenshtein similarity, symbol accuracy, timing, per-tier robustness) are
   computed with `metrics.py`.
4. **Report** — the results are turned into a readable Markdown and HTML report
   with a recommendation.

### Inside the own-code engine

The pipeline reads an equation the way a typesetter would write one — glyphs on a
baseline, at a size, in a structure:

1. **Segment** into connected components, then **repair** them: the two bars of an
   `=`, the dot of an `i`, the two halves of a `\pm` are joined back together, and a
   blob that matches nothing is offered to the classifier as two glyphs cut at its
   thinnest row — which is how a `\sum` printed against its own limit comes apart.
2. **Measure the line.** Every template is rendered beside a reference `x`, so the
   bank knows each symbol's height and how far it sits below the baseline, in
   x-heights. The line's x-height and baseline are then estimated jointly, by
   picking the pair that best explains the whole group of glyphs at once.
3. **Classify** each glyph on four cues: overlap with the template, correlation of
   the two stretched to a common square, aspect ratio, and — once the line has been
   measured — size and baseline position. Shape alone cannot tell `.` from `\cdot`;
   their positions on the line can.
4. **Parse the layout** recursively: fraction bars, radicals and their indices,
   binomials, then sub/superscripts. Whether a glyph is a script is settled by
   scoring both readings — on the line, or smaller and raised off it — and taking
   the better one. Each region is re-measured and re-classified at its own scale, so
   a superscript is read as full-size type in its own right.
5. **Emit** LaTeX with the spacing conventions the source would have been written
   with: spaces around infix operators, none inside `\sum_{i=1}^{n}`, a `\,` where
   the typeset gap is too wide to be ordinary letter spacing.

One thing the pipeline deliberately does not guess: mathtext renders `\left( … \right)`
identically to plain parentheses, so nothing in the image distinguishes them and the
engine always emits `(`.

The dataset split is **deterministic** (seed 42, 70/30) and **disjoint**, so the
own-code pipeline's training never sees the benchmark's test images.

---

## 🧪 Running tests

```bash
python -m pytest tests/ -q
```

The suite covers preprocessing (incl. polarity and deskew regression guards), both
recognizers, the benchmark harness, the report generator, and the CLI.

---

## 📄 License

MIT

---

<div align="center">
  Made with ❤️ to turn pictures of math into editable LaTeX
</div>
