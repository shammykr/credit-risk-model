"""Statistical checks of feature significance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def chi_squared_tests(df: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    """Chi-squared test of independence between each categorical feature and the target.

    Cramer's V is reported alongside the p-value: with tens of thousands of rows,
    almost any difference is "significant", so effect size is what ranks features.
    """
    rows = []
    for feature in features:
        table = pd.crosstab(df[feature], df[target])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        n = table.to_numpy().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
        rows.append({"feature": feature, "chi2": chi2, "dof": dof, "p_value": p, "cramers_v": cramers_v})
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False, ignore_index=True)


def numeric_significance(df: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    """Point-biserial correlation and Mann-Whitney U test of each numeric feature vs the target."""
    rows = []
    for feature in features:
        data = df[[feature, target]].dropna()
        r, p_r = stats.pointbiserialr(data[target], data[feature])
        bad, good = data.loc[data[target] == 1, feature], data.loc[data[target] == 0, feature]
        _, p_u = stats.mannwhitneyu(bad, good, alternative="two-sided")
        rows.append({"feature": feature, "point_biserial_r": r, "p_value_r": p_r,
                     "mann_whitney_p": p_u, "median_default": bad.median(), "median_repaid": good.median()})
    return (pd.DataFrame(rows)
            .assign(abs_r=lambda d: d["point_biserial_r"].abs())
            .sort_values("abs_r", ascending=False, ignore_index=True)
            .drop(columns="abs_r"))


def information_value(df: pd.DataFrame, feature: str, target: str, eps: float = 0.5) -> tuple[float, pd.DataFrame]:
    """Weight of evidence per category and the feature's information value (IV).

    Rule of thumb: IV < 0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, > 0.3 strong.
    `eps` is added to empty cells so WoE stays finite.
    """
    grouped = df.groupby(feature, observed=True)[target].agg(["count", "sum"])
    bad = grouped["sum"] + eps
    good = grouped["count"] - grouped["sum"] + eps
    dist_bad, dist_good = bad / bad.sum(), good / good.sum()
    woe = np.log(dist_good / dist_bad)
    iv_parts = (dist_good - dist_bad) * woe
    table = pd.DataFrame({
        "count": grouped["count"],
        "default_rate": grouped["sum"] / grouped["count"],
        "woe": woe,
        "iv_contribution": iv_parts,
    })
    return float(iv_parts.sum()), table


def iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "useless"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    return "strong"
