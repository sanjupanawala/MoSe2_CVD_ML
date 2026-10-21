"""Editable SVG/PDF and 600 dpi PNG comparisons from evaluation JSON files."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(paths, labels, output):
    if len(paths) != len(labels):
        raise ValueError("Supply one label for each metrics file")
    output = Path(output)
    if any(output.with_suffix(suffix).exists() for suffix in [".svg", ".pdf", ".png"]):
        raise FileExistsError("Choose a new figure filename")
    rows = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained")
    colors = ["#657885", "#C78B4B", "#3889A0", "#38875C"]
    positions = np.arange(len(rows))
    axes[0].bar(positions, [r["miou"] for r in rows], color=[colors[i % 4] for i in positions], width=.65)
    axes[0].set(ylim=(0, 1), ylabel="Mean intersection over union")
    for index, label in [(1, "Monolayer"), (2, "Multilayer")]:
        axes[1].bar(positions + (index - 1.5) * .36,
                    [r["coverage_mae_pp"][index] for r in rows], width=.36,
                    label=label, color=["#3489A1", "#D99A56"][index - 1])
    axes[1].set(ylabel="Coverage MAE (percentage points)", ylim=(0, None))
    axes[1].legend(frameon=False)
    for letter, axis in zip("ab", axes):
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.text(-.16, 1.06, letter, transform=axis.transAxes, fontweight="bold", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".svg", ".pdf", ".png"]:
        fig.savefig(output.with_suffix(suffix), dpi=600)
    plt.close(fig)
