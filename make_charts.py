"""Generate benchmark comparison charts from results/benchmark_tiers.json.

Produces two separate, README-sized PNGs:
  samples/benchmark_accuracy.png  - symbol accuracy by tier (+ aggregate)
  samples/benchmark_speed.png     - mean inference time by tier (+ aggregate)
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

with open("results/benchmark_tiers.json") as f:
    data = json.load(f)

tiers = data["tiers"]
per_tier = data["per_tier"]
aggregate = data["aggregate"]

# ---- validated categorical palette (dataviz skill, slots 1 & 2) ----
COLOR_AI = "#2a78d6"
COLOR_OWN = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

LABELS = {"ai": "Local AI (qwen3-vl:8b)", "owncode": "Own-code OCR"}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans",
]
plt.rcParams["axes.edgecolor"] = BASELINE
plt.rcParams["text.color"] = INK_PRIMARY

TIER_LABELS = {
    "clean": "clean", "noisy": "noisy", "low_res": "low res",
    "black_bg": "black bg", "white_bg": "white bg",
}


def build_chart(metric_key, fmt, title, subtitle, ylabel, out_path,
                 ylim_pad_ratio=0.14, better="higher"):
    groups = list(tiers) + ["aggregate"]
    group_labels = [TIER_LABELS[t] for t in tiers] + ["All tiers"]

    def value(rec, group):
        src = aggregate[rec] if group == "aggregate" else per_tier[group][rec]
        return src[metric_key]

    ai_vals = [value("ai", g) for g in groups]
    own_vals = [value("owncode", g) for g in groups]

    n = len(groups)
    gap_before_agg = 0.9  # extra spacing to set the aggregate group apart
    x = np.arange(n, dtype=float)
    x[-1] += gap_before_agg
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.2, 4.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    bars_ai = ax.bar(x - width / 2 - 0.02, ai_vals, width,
                      label=LABELS["ai"], color=COLOR_AI,
                      edgecolor=SURFACE, linewidth=1.2, zorder=3)
    bars_own = ax.bar(x + width / 2 + 0.02, own_vals, width,
                       label=LABELS["owncode"], color=COLOR_OWN,
                       edgecolor=SURFACE, linewidth=1.2, zorder=3)

    top = max(max(ai_vals), max(own_vals))
    ax.set_ylim(0, top * (1 + ylim_pad_ratio))

    for bars, vals in ((bars_ai, ai_vals), (bars_own, own_vals)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.025, fmt(v),
                     ha="center", va="bottom", fontsize=9.5,
                     color=INK_SECONDARY, fontweight="medium", zorder=4)

    # separator between per-tier bars and the aggregate group
    sep_x = (x[-2] + x[-1]) / 2 - width * 0.9
    ax.axvline(sep_x, color=GRIDLINE, lw=1.1, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=10.5, color=INK_SECONDARY)
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.get_xticklabels()[-1].set_color(INK_PRIMARY)

    ax.set_ylabel(ylabel, fontsize=10.5, color=INK_SECONDARY)
    ax.tick_params(axis="y", labelsize=9.5, colors=INK_MUTED, length=0)
    ax.tick_params(axis="x", length=0)

    ax.yaxis.grid(True, color=GRIDLINE, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1.0)

    if metric_key == "mean_symbol_accuracy":
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    # Header stack (title / subtitle / legend) is placed in FIGURE coordinates
    # so it never overlaps the axes content, independent of axes position.
    fig.subplots_adjust(top=0.70, bottom=0.12, left=0.08, right=0.98)

    fig.text(0.015, 0.97, title, ha="left", va="top", fontsize=14.5,
              fontweight="bold", color=INK_PRIMARY)
    fig.text(0.015, 0.885, subtitle, ha="left", va="top", fontsize=10,
              color=INK_MUTED)

    legend = fig.legend(
        handles=[bars_ai, bars_own], labels=[LABELS["ai"], LABELS["owncode"]],
        loc="upper left", bbox_to_anchor=(0.01, 0.80), ncol=2,
        frameon=False, fontsize=10.5, handlelength=1.1, handleheight=1.1,
        columnspacing=1.6, borderaxespad=0,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out_path}")


build_chart(
    metric_key="mean_symbol_accuracy",
    fmt=lambda v: f"{v:.0%}",
    title="Recognition accuracy by difficulty tier",
    subtitle="Symbol accuracy · 12 samples per tier, held-out test set · higher is better",
    ylabel="Symbol accuracy",
    out_path="samples/benchmark_accuracy.png",
)

def format_seconds(value):
    """Label a duration in the unit a reader thinks in at that magnitude."""
    if value < 1.0:
        return f"{value * 1000:.0f}ms"
    return f"{value:.1f}s"


def build_speed_chart(out_path):
    """Compare the two engines' inference time on a logarithmic axis.

    The engines are two orders of magnitude apart, which bars cannot show: drawn
    from zero on a linear axis the faster engine has no visible length, and a
    logarithmic bar would misstate its length as its value. Dots carry position
    without implying length from zero, so the log axis is honest, and the rule
    joining each pair shows the size of the gap directly.
    """
    groups = list(tiers) + ["aggregate"]
    labels = [TIER_LABELS[t] for t in tiers] + ["All tiers"]

    def value(rec, group):
        src = aggregate[rec] if group == "aggregate" else per_tier[group][rec]
        return src["mean_seconds"]

    ai_vals = [value("ai", g) for g in groups]
    own_vals = [value("owncode", g) for g in groups]
    y = np.arange(len(groups), dtype=float)
    y[-1] += 0.45  # set the aggregate row apart (the axis is inverted)

    fig, ax = plt.subplots(figsize=(9.2, 4.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for yi, fast, slow in zip(y, own_vals, ai_vals):
        ax.plot([fast, slow], [yi, yi], color=BASELINE, lw=1.4, zorder=2,
                solid_capstyle="round")
    dots_own = ax.scatter(own_vals, y, s=110, color=COLOR_OWN, zorder=4,
                          edgecolor=SURFACE, linewidth=2)
    dots_ai = ax.scatter(ai_vals, y, s=110, color=COLOR_AI, zorder=4,
                         edgecolor=SURFACE, linewidth=2)

    for yi, v in zip(y, own_vals):
        ax.annotate(format_seconds(v), (v, yi), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9.5,
                    color=INK_SECONDARY, fontweight="medium", zorder=5)
    for yi, v in zip(y, ai_vals):
        ax.annotate(format_seconds(v), (v, yi), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9.5,
                    color=INK_SECONDARY, fontweight="medium", zorder=5)

    ax.set_xscale("log")
    ax.set_xlim(min(own_vals) * 0.45, max(ai_vals) * 2.6)
    ax.set_xticks([0.01, 0.1, 1.0, 10.0])
    ax.xaxis.set_major_formatter(lambda v, _: format_seconds(v))
    ax.xaxis.set_minor_formatter(lambda v, _: "")
    ax.set_ylim(y.min() - 0.7, y.max() + 0.7)
    ax.invert_yaxis()

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5, color=INK_SECONDARY)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.get_yticklabels()[-1].set_color(INK_PRIMARY)
    ax.set_xlabel("Mean time per image (log scale)", fontsize=10.5,
                  color=INK_SECONDARY)
    ax.tick_params(axis="x", labelsize=9.5, colors=INK_MUTED, length=0)
    ax.tick_params(axis="y", length=0)

    ax.xaxis.grid(True, which="major", color=GRIDLINE, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1.0)

    fig.subplots_adjust(top=0.70, bottom=0.14, left=0.12, right=0.97)
    fig.text(0.015, 0.97, "Inference speed by difficulty tier", ha="left",
             va="top", fontsize=14.5, fontweight="bold", color=INK_PRIMARY)
    fig.text(0.015, 0.885,
             "Mean time per image · 12 samples per tier, held-out test set · "
             "further left is faster",
             ha="left", va="top", fontsize=10, color=INK_MUTED)
    legend = fig.legend(
        handles=[dots_ai, dots_own], labels=[LABELS["ai"], LABELS["owncode"]],
        loc="upper left", bbox_to_anchor=(0.01, 0.80), ncol=2, frameon=False,
        fontsize=10.5, handlelength=1.1, borderaxespad=0, columnspacing=1.6,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out_path}")


build_speed_chart("samples/benchmark_speed.png")
