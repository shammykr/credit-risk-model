"""Synthetic consumer-loan applications with a realistic default process.

Each applicant has a hidden creditworthiness factor that drives credit score,
income, debt-to-income and late-payment history. Default is then drawn from a
logistic model on the *observed* features plus the part of creditworthiness the
lender cannot see. The model includes a few non-linear effects (thin and very
thick credit files, a loan-to-income threshold, DTI x late-payment interaction)
so feature engineering and tree models have something real to find.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "credit_data.csv"

COLUMNS = [
    "credit_score", "annual_income", "debt_to_income", "num_open_accounts",
    "num_late_payments", "loan_amount", "loan_purpose", "default_flag",
]

PURPOSES = {
    "debt_consolidation": (0.42, 0.00),
    "credit_card": (0.20, -0.10),
    "home_improvement": (0.11, -0.25),
    "auto": (0.09, -0.30),
    "medical": (0.06, 0.35),
    "small_business": (0.05, 0.60),
    "education": (0.04, 0.20),
    "other": (0.03, 0.25),
}  # name: (share of applications, log-odds shift)

TARGET_DEFAULT_RATE = 0.09
INCOME_MISSING_RATE = 0.03


def _default_logit(df: pd.DataFrame, hidden: np.ndarray) -> np.ndarray:
    score = (df["credit_score"] - 690) / 60
    dti = df["debt_to_income"]
    late = df["num_late_payments"].clip(upper=8)
    lti = df["loan_amount"] / df["annual_income"]
    accounts = df["num_open_accounts"]
    purpose = df["loan_purpose"].map({k: v[1] for k, v in PURPOSES.items()})
    return (
        -0.60 * score
        + 0.15 * np.maximum(0, -score) ** 2          # risk accelerates below ~630
        + 2.6 * (dti - 0.30)
        + 0.20 * late
        + 1.8 * dti * late / 4                        # stretched borrowers who already pay late
        + 2.2 * np.maximum(0, lti - 0.35)             # loans large relative to income
        + 0.35 * (accounts <= 2)                      # thin file
        + 0.30 * (accounts >= 18)                     # credit-hungry
        + purpose
        - 1.00 * hidden                               # what the lender cannot observe
    )


def generate_synthetic(n_samples: int = 25_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n_samples)                    # creditworthiness
    hidden = 0.3 * z + 0.95 * rng.normal(size=n_samples)

    income = np.exp(rng.normal(np.log(64_000) + 0.18 * z, 0.55))
    income = np.clip(income, 12_000, 600_000).round(-2)
    loan = np.exp(rng.normal(np.log(14_000) + 0.35 * np.log(income / 64_000), 0.6))
    loan = (np.clip(loan, 1_000, 50_000) / 50).round() * 50

    df = pd.DataFrame({
        "credit_score": np.clip(690 + 52 * z + rng.normal(0, 28, n_samples), 300, 850).round().astype(int),
        "annual_income": income,
        "debt_to_income": np.clip(0.30 - 0.06 * z + rng.normal(0, 0.11, n_samples), 0.0, 0.95).round(3),
        "num_open_accounts": rng.poisson(np.exp(np.log(8) + 0.12 * z + rng.normal(0, 0.45, n_samples))),
        "num_late_payments": rng.poisson(np.exp(-0.6 - 0.9 * z)),
        "loan_amount": loan,
        "loan_purpose": rng.choice(list(PURPOSES), size=n_samples,
                                   p=[v[0] for v in PURPOSES.values()]),
    })

    logit = _default_logit(df, hidden)
    # Pick the intercept that hits the target default rate.
    lo, hi = -10.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(logit + mid)))).mean() > TARGET_DEFAULT_RATE:
            hi = mid
        else:
            lo = mid
    prob = 1 / (1 + np.exp(-(logit + lo)))
    df["default_flag"] = (rng.random(n_samples) < prob).astype(int)

    # Income is self-reported and sometimes missing, as in real application data.
    df.loc[rng.random(n_samples) < INCOME_MISSING_RATE, "annual_income"] = np.nan
    return df[COLUMNS]


def load_data(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df[COLUMNS]
