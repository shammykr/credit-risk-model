import json

import numpy as np
import pandas as pd
import pytest

from src.data import COLUMNS, generate_synthetic
from src.evaluation import best_f1_threshold, ks_statistic
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FeatureEngineer, build_preprocessor
from src.pipeline import run
from src.stats import chi_squared_tests, information_value


@pytest.fixture(scope="module")
def df():
    return generate_synthetic(5_000, seed=1)


def test_synthetic_data_shape_and_rates(df):
    assert list(df.columns) == COLUMNS
    assert 0.06 < df["default_flag"].mean() < 0.12
    assert 0.01 < df["annual_income"].isna().mean() < 0.05
    assert df.drop(columns="annual_income").notna().all().all()
    pd.testing.assert_frame_equal(df, generate_synthetic(5_000, seed=1))


def test_feature_engineer_learns_median_from_fit_data_only(df):
    train = df.iloc[:1000]
    fe = FeatureEngineer().fit(train)
    assert fe.income_median_ == train["annual_income"].median()
    out = fe.transform(df)
    assert list(out.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert out[NUMERIC_FEATURES].notna().all().all()
    assert (out.loc[df["annual_income"].isna(), "income_missing"] == 1).all()


def test_bins_cover_every_row(df):
    out = FeatureEngineer().fit(df).transform(df)
    for col in ["credit_score_band", "dti_band", "late_payment_band", "open_accounts_band"]:
        assert "nan" not in set(out[col])


def test_encoder_ignores_unseen_categories(df):
    fe = FeatureEngineer().fit(df)
    enc = build_preprocessor().fit(fe.transform(df))
    new = df.head(3).assign(loan_purpose="crypto")
    assert np.isfinite(enc.transform(fe.transform(new))).all()


def test_chi_squared_detects_signal_not_noise(df):
    rng = np.random.default_rng(0)
    data = FeatureEngineer().fit(df).transform(df).assign(
        default_flag=df["default_flag"].values, noise=rng.choice(list("abc"), len(df)))
    res = chi_squared_tests(data, ["late_payment_band", "noise"], "default_flag").set_index("feature")
    assert res.loc["late_payment_band", "p_value"] < 1e-10
    assert res.loc["noise", "cramers_v"] < 0.05


def test_information_value_ranks_strong_above_weak(df):
    data = FeatureEngineer().fit(df).transform(df).assign(default_flag=df["default_flag"].values)
    strong, _ = information_value(data, "credit_score_band", "default_flag")
    weak, _ = information_value(data, "loan_purpose", "default_flag")
    assert strong > 0.3 > weak


def test_metric_helpers():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert best_f1_threshold(y, np.array([0.1, 0.4, 0.35, 0.9])) == pytest.approx(0.35)


def test_pipeline_end_to_end(tmp_path):
    data_path = tmp_path / "data.csv"
    results = run(data_path, tmp_path / "reports", tmp_path / "models", cv_folds=3,
                  generate=True, n_samples=4_000, seed=3)
    metrics = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert set(metrics["models"]) == {"logistic_regression", "gradient_boosting"}
    for res in results.values():
        assert 0.75 < res["roc_auc"] < 0.95
        assert 0 < res["threshold"] < 1
    assert len(list((tmp_path / "reports" / "figures").glob("*.png"))) == 3
    assert len(list((tmp_path / "models").glob("*.joblib"))) == 2
