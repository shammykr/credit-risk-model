"""Shared Matplotlib/Seaborn styling so notebook and pipeline figures match."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

# Fixed categorical slots: the same entity always gets the same color.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CLASS_COLORS = {0: BLUE, 1: ORANGE}            # repaid, defaulted
CLASS_LABELS = {0: "Repaid", 1: "Defaulted"}
MODEL_COLORS = {"logistic_regression": BLUE, "gradient_boosting": ORANGE}
MODEL_LABELS = {"logistic_regression": "Logistic regression", "gradient_boosting": "Gradient boosting"}
SEQUENTIAL = "Blues"
DIVERGING = sns.diverging_palette(250, 15, s=75, l=50, center="light", as_cmap=True)  # blue <-> red

INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "legend.frameon": False,
        "lines.linewidth": 2,
    })
