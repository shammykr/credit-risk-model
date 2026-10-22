"""Feature engineering: imputation, binning, interaction terms and encoding.

FeatureEngineer is a scikit-learn transformer, so everything it learns (the
median income used for imputation) comes from the training fold only and the
same code runs in the notebook, the CLI pipeline and at scoring time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "default_flag"

# FICO-style score bands.
SCORE_BINS = [-np.inf, 580, 670, 740, 800, np.inf]
SCORE_LABELS = ["Poor (<580)", "Fair (580-669)", "Good (670-739)", "Very good (740-799)", "Exceptional (800+)"]
DTI_BINS = [-np.inf, 0.20, 0.35, 0.50, np.inf]
DTI_LABELS = ["<20%", "20-35%", "35-50%", "50%+"]
LATE_BINS = [-np.inf, 0, 1, 3, np.inf]
LATE_LABELS = ["0", "1", "2-3", "4+"]
ACCOUNT_BINS = [-np.inf, 2, 9, 17, np.inf]
ACCOUNT_LABELS = ["0-2 (thin file)", "3-9", "10-17", "18+"]

NUMERIC_FEATURES = [
    "credit_score", "log_income", "debt_to_income", "num_open_accounts",
    "num_late_payments", "log_loan_amount", "loan_to_income",
    "dti_x_late", "score_x_dti", "income_missing",
]
CATEGORICAL_FEATURES = [
    "loan_purpose", "credit_score_band", "dti_band", "late_payment_band", "open_accounts_band",
]


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Ordered categorical bands for score, DTI, late payments and open accounts."""
    out = df.copy()
    out["credit_score_band"] = pd.cut(out["credit_score"], SCORE_BINS, labels=SCORE_LABELS, right=False)
    out["dti_band"] = pd.cut(out["debt_to_income"], DTI_BINS, labels=DTI_LABELS, right=False)
    out["late_payment_band"] = pd.cut(out["num_late_payments"], LATE_BINS, labels=LATE_LABELS)
    out["open_accounts_band"] = pd.cut(out["num_open_accounts"], ACCOUNT_BINS, labels=ACCOUNT_LABELS)
    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Raw application columns -> engineered feature frame (before encoding)."""

    def fit(self, X: pd.DataFrame, y=None):
        self.income_median_ = float(X["annual_income"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out["income_missing"] = out["annual_income"].isna().astype(int)
        income = out["annual_income"].fillna(self.income_median_)

        out["log_income"] = np.log(income)
        out["log_loan_amount"] = np.log(out["loan_amount"])
        out["loan_to_income"] = out["loan_amount"] / income
        # Interaction terms
        out["dti_x_late"] = out["debt_to_income"] * out["num_late_payments"]
        out["score_x_dti"] = (out["credit_score"] - 690) / 60 * out["debt_to_income"]

        out = add_bins(out)
        for col in CATEGORICAL_FEATURES:
            out[col] = out[col].astype(str)
        return out[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric features and one-hot encode categoricals (including the bins)."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
