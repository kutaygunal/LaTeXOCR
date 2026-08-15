"""Generate benchmark comparison charts from results/benchmark_tiers.json."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("results/benchmark_tiers.json") as f:
    data = json.load(f)

tiers = data["tiers"]
ai = data["per_tier"]
colors = {"ai": "#4B8BBE", "owncode": "#E8A838"}
labels = {"ai": "Local AI (qwen3-vl:8b)", "owncode": "Own-code OCR"}

def vals(rec, key):
    return [ai[t][rec][key] for t in tiers]

# ---- Figure: accuracy + time ----
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"wspace": 0.3})
fig.patch.set_facecolor("white")

# Panel 1: symbol accuracy by tier
x = np.arange(len(tiers))
width = 0.36
ax = axes[0]
for i, rec in enumerate(["ai", "owncode"]):
    bars = ax.bar(x + (i - 0.5) * width, vals(rec, "mean_symbol_accuracy"),
                  width, label=labels[rec], color=colors[rec])
    for b, v in zip(bars, vals(rec, "mean_symbol_accuracy")):
        ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.0%}",
                ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(tiers)
ax.set_ylabel("Symbol accuracy")
ax.set_title("Recognition accuracy by difficulty tier")
ax.set_ylim(0, 1.12)
ax.legend(frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.axhline(0, color="#ccc", lw=0.8)

# Panel 2: mean time per image
ax = axes[1]
for i, rec in enumerate(["ai", "owncode"]):
    bars = ax.bar(x + (i - 0.5) * width, vals(rec, "mean_seconds"),
                  width, label=labels[rec], color=colors[rec])
    for b, v in zip(bars, vals(rec, "mean_seconds")):
        ax.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.1f}s",
                ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(tiers)
ax.set_ylabel("Mean time per image (s)")
ax.set_title("Inference speed by difficulty tier")
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.axhline(0, color="#ccc", lw=0.8)

fig.suptitle("LaTeXOCR benchmark — 12 samples per tier, held-out test set", y=1.03, fontweight="bold")
fig.savefig("samples/benchmark.png", dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved samples/benchmark.png")
