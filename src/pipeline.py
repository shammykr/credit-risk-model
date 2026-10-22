"""Train and evaluate the credit-risk models end to end.

    python src/pipeline.py                       # train on data/credit_data.csv
    python src/pipeline.py --generate            # regenerate the synthetic data first
    python src/pipeline.py --data my.csv --test-size 0.25 --seed 7

Steps: load data -> stratified train/test split -> 5-fold CV AUC on train ->
pick each model's decision threshold on out-of-fold train predictions ->
fit on full train -> evaluate once on the held-out test set -> save models,
metrics.json and figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python src/pipeline.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.data import DEFAULT_DATA_PATH, ROOT, generate_synthetic, load_data
from src.evaluation import (
    best_f1_threshold, evaluate, metrics_table, plot_calibration, plot_confusion,
    plot_importance, plot_precision_recall, plot_roc,
)
from src.features import TARGET, FeatureEngineer, build_preprocessor
from src.plotting import MODEL_LABELS, set_style


def build_models(seed: int = 42) -> dict[str, Pipeline]:
    def pipe(model):
        return Pipeline([("features", FeatureEngineer()), ("encode", build_preprocessor()), ("model", model)])

    return {
        "logistic_regression": pipe(LogisticRegression(C=0.5, max_iter=5000)),
        "gradient_boosting": pipe(GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
            min_samples_leaf=50, random_state=seed)),
    }


def split(df: pd.DataFrame, test_size: float, seed: int):
    X, y = df.drop(columns=TARGET), df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def run(data_path: Path, output_dir: Path, models_dir: Path, test_size: float = 0.2, seed: int = 42,
        cv_folds: int = 5, generate: bool = False, n_samples: int = 25_000) -> dict:
    if generate:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic(n_samples, seed).to_csv(data_path, index=False)
        print(f"Generated {n_samples:,} rows -> {data_path}")
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split(df, test_size, seed)
    print(f"Train {len(X_train):,} rows / test {len(X_test):,} rows, default rate {y_train.mean():.2%}")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)
    results, scores, fitted = {}, {}, {}
    for name, model in build_models(seed).items():
        oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        fold_aucs = [roc_auc_score(y_train.iloc[val], oof[val]) for _, val in cv.split(X_train, y_train)]
        threshold = best_f1_threshold(y_train, oof)

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = evaluate(y_test, proba, threshold)
        results[name]["cv_auc_mean"] = float(np.mean(fold_aucs))
        results[name]["cv_auc_std"] = float(np.std(fold_aucs))
        scores[name] = (y_test.to_numpy(), proba)
        fitted[name] = model
        joblib.dump(model, models_dir / f"{name}.joblib")
        print(f"{MODEL_LABELS[name]:<22} CV AUC {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f} | "
              f"test AUC {results[name]['roc_auc']:.4f}")

    importance = pd.DataFrame({
        name: pd.Series(permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                               n_repeats=5, random_state=seed).importances_mean,
                        index=X_test.columns)
        for name, model in fitted.items()
    }).sort_values("logistic_regression", ascending=False)
    for name in results:
        results[name]["permutation_importance"] = importance[name].round(5).to_dict()

    save_figures(scores, results, importance, figures_dir)
    (output_dir / "metrics.json").write_text(json.dumps(
        {"data": str(data_path.relative_to(ROOT) if data_path.is_relative_to(ROOT) else data_path),
         "n_train": len(X_train), "n_test": len(X_test), "seed": seed,
         "test_default_rate": float(y_test.mean()), "models": results}, indent=2))

    print("\nTest-set metrics")
    print(metrics_table(results).round(4).to_string())
    for name, res in results.items():
        print(f"\n{MODEL_LABELS[name]} - classification report (threshold {res['threshold']:.3f})")
        print(pd.DataFrame(res["classification_report"]).T.round(3).to_string())
    print(f"\nSaved models to {models_dir}, metrics and figures to {output_dir}")
    return results


def save_figures(scores: dict, results: dict, importance: pd.DataFrame, figures_dir: Path) -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_roc(scores, axes[0])
    plot_precision_recall(scores, axes[1])
    fig.savefig(figures_dir / "roc_pr_curves.png")

    fig, axes = plt.subplots(1, len(scores), figsize=(6 * len(scores), 4.8))
    for ax, (name, (y, p)) in zip(np.atleast_1d(axes), scores.items()):
        plot_confusion(y, p, results[name]["threshold"], MODEL_LABELS[name], ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrices.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.2]})
    plot_calibration(scores, axes[0])
    plot_importance(importance, axes[1])
    fig.tight_layout()
    fig.savefig(figures_dir / "calibration_importance.png")
    plt.close("all")


def main() -> None:
    matplotlib.use("Agg")  # headless: figures are only written to disk
    parser = argparse.ArgumentParser(description="Train and evaluate credit-risk models.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="CSV with the raw columns")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports", help="metrics.json + figures/")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models", help="fitted .joblib pipelines")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--generate", action="store_true", help="regenerate synthetic data at --data first")
    parser.add_argument("--n-samples", type=int, default=25_000, help="rows to generate with --generate")
    args = parser.parse_args()
    run(args.data, args.output_dir, args.models_dir, args.test_size, args.seed, args.cv_folds, args.generate, args.n_samples)


if __name__ == "__main__":
    main()
