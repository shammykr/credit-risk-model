"""Model evaluation: credit-risk metrics and the standard diagnostic plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)

from src.plotting import CLASS_LABELS, MODEL_COLORS, MODEL_LABELS, MUTED


def ks_statistic(y_true, proba) -> float:
    """Kolmogorov-Smirnov: max gap between the score CDFs of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    return float(np.max(tpr - fpr))


def best_f1_threshold(y_true, proba) -> float:
    """Probability cut-off that maximises F1. Call it on out-of-fold training predictions,
    never on the test set, so the reported test metrics stay honest."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, proba)
    return {
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_true, proba),
        "average_precision": average_precision_score(y_true, proba),
        "brier_score": brier_score_loss(y_true, proba),
        "threshold": threshold,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "classification_report": classification_report(
            y_true, pred, target_names=[CLASS_LABELS[0], CLASS_LABELS[1]], output_dict=True),
    }


def metrics_table(results: dict) -> pd.DataFrame:
    cols = ["roc_auc", "gini", "ks", "average_precision", "brier_score", "threshold", "precision", "recall", "f1"]
    return pd.DataFrame({MODEL_LABELS.get(k, k): {c: v[c] for c in cols} for k, v in results.items()}).T


# --- Plots ---------------------------------------------------------------------------------
# `scores` maps model name -> (y_true, predicted probability of default).

def plot_roc(scores: dict, ax=None):
    ax = ax or plt.gca()
    for name, (y, p) in scores.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=MODEL_COLORS.get(name), label=f"{MODEL_LABELS.get(name, name)} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, label="Random")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right")
    return ax


def plot_precision_recall(scores: dict, ax=None):
    ax = ax or plt.gca()
    for name, (y, p) in scores.items():
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, color=MODEL_COLORS.get(name),
                label=f"{MODEL_LABELS.get(name, name)} (AP {average_precision_score(y, p):.3f})")
    base_rate = np.mean(next(iter(scores.values()))[0])
    ax.axhline(base_rate, ls="--", lw=1, color=MUTED, label=f"Base rate ({base_rate:.1%})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="upper right")
    return ax


def plot_confusion(y_true, proba, threshold: float, title: str, ax=None):
    ax = ax or plt.gca()
    cm = confusion_matrix(y_true, (proba >= threshold).astype(int))
    pct = cm / cm.sum(axis=1, keepdims=True)
    labels = np.array([[f"{n:,}\n{p:.1%} of row" for n, p in zip(r_n, r_p)] for r_n, r_p in zip(cm, pct)])
    sns.heatmap(pct, annot=labels, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax,
                linewidths=2, linecolor="white", annot_kws={"fontsize": 11},
                xticklabels=[CLASS_LABELS[0], CLASS_LABELS[1]], yticklabels=[CLASS_LABELS[0], CLASS_LABELS[1]])
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"{title}\nthreshold = {threshold:.3f}")
    ax.tick_params(length=0)
    return ax


def plot_calibration(scores: dict, ax=None, n_bins: int = 10):
    ax = ax or plt.gca()
    for name, (y, p) in scores.items():
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", ms=5, color=MODEL_COLORS.get(name), label=MODEL_LABELS.get(name, name))
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], ls="--", lw=1, color=MUTED, label="Perfect calibration")
    ax.set(xlabel="Mean predicted PD (decile)", ylabel="Observed default rate", title="Calibration")
    ax.legend(loc="upper left")
    return ax


def plot_importance(importance: pd.DataFrame, ax=None):
    """`importance`: index = feature, columns = model names, values = mean AUC drop."""
    ax = ax or plt.gca()
    tidy = (importance.rename(columns=MODEL_LABELS).reset_index(names="feature")
            .melt(id_vars="feature", var_name="model", value_name="auc_drop"))
    sns.barplot(tidy, y="feature", x="auc_drop", hue="model", ax=ax, saturation=1,
                palette={MODEL_LABELS[k]: v for k, v in MODEL_COLORS.items()})
    ax.set(xlabel="Drop in test AUC when the feature is shuffled", ylabel="", title="Permutation importance (raw inputs)")
    ax.legend(title=None, loc="lower right")
    return ax
