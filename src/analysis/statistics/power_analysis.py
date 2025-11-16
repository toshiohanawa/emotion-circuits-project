"""Power analysis helpers for Phase 7.5."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

try:  # Optional dependency
    from statsmodels.stats.power import TTestIndPower, TTestPower

    HAS_STATSMODELS = True
except Exception:  # pragma: no cover - statsmodels might be unavailable.
    HAS_STATSMODELS = False
    TTestIndPower = None  # type: ignore
    TTestPower = None  # type: ignore


def _paired_power(effect_size: float, n: float, alpha: float) -> float:
    if n <= 1 or effect_size <= 0:
        return float("nan")
    if HAS_STATSMODELS:
        solver = TTestPower()
        return float(solver.power(effect_size=effect_size, nobs=n, alpha=alpha, alternative="two-sided"))
    df = n - 1
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    non_central = effect_size * np.sqrt(n)
    power = 1.0 - stats.nct.cdf(t_crit, df, non_central) + stats.nct.cdf(-t_crit, df, non_central)
    return float(power)


def _independent_power(effect_size: float, n1: float, n2: float, alpha: float) -> float:
    if n1 <= 1 or n2 <= 1 or effect_size <= 0:
        return float("nan")
    ratio = n2 / n1
    if HAS_STATSMODELS:
        solver = TTestIndPower()
        return float(
            solver.power(effect_size=effect_size, nobs1=n1, alpha=alpha, ratio=ratio, alternative="two-sided")
        )
    df = n1 + n2 - 2
    pooled = np.sqrt((n1 * n2) / (n1 + n2))
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    non_central = effect_size * pooled
    power = 1.0 - stats.nct.cdf(t_crit, df, non_central) + stats.nct.cdf(-t_crit, df, non_central)
    return float(power)


def estimate_power(effect_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Append a `power` column to the effect DataFrame."""
    if effect_df.empty:
        return effect_df
    df = effect_df.copy()
    df["power"] = np.nan
    for idx, row in df.iterrows():
        effect = row.get("cohen_d")
        if not np.isfinite(effect) or effect == 0:
            continue
        effect = abs(float(effect))
        if bool(row.get("paired", True)):
            n = row.get("n_effective") or row.get("n_baseline")
            df.loc[idx, "power"] = _paired_power(effect, float(n), alpha)
        else:
            n1 = float(row.get("n_baseline", np.nan))
            n2 = float(row.get("n_patched", np.nan))
            df.loc[idx, "power"] = _independent_power(effect, n1, n2, alpha)
    return df


def required_sample_size(
    effect_size: float,
    target_power: float = 0.8,
    alpha: float = 0.05,
    paired: bool = True,
    ratio: float = 1.0,
) -> Dict[str, float]:
    """Return the minimum sample size (per condition) required to detect an effect."""
    effect = abs(effect_size)
    if effect <= 0:
        return {"n_effective": float("nan")} if paired else {"n_control": float("nan"), "n_treatment": float("nan")}
    if paired:
        if HAS_STATSMODELS:
            solver = TTestPower()
            n = solver.solve_power(effect_size=effect, power=target_power, alpha=alpha, alternative="two-sided")
        else:
            z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
            z_beta = stats.norm.ppf(target_power)
            n = ((z_alpha + z_beta) / effect) ** 2
        return {"n_effective": float(np.ceil(n))}
    ratio = max(ratio, 1e-6)
    if HAS_STATSMODELS:
        solver = TTestIndPower()
        n_control = solver.solve_power(
            effect_size=effect,
            power=target_power,
            alpha=alpha,
            ratio=ratio,
            alternative="two-sided",
        )
    else:
        z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
        z_beta = stats.norm.ppf(target_power)
        n_control = ((z_alpha + z_beta) ** 2) * (1 + 1 / ratio) / (effect**2)
    n_treatment = n_control * ratio
    return {
        "n_control": float(np.ceil(n_control)),
        "n_treatment": float(np.ceil(n_treatment)),
    }


def summarize_power_requirements(
    effect_df: pd.DataFrame,
    effect_targets: Sequence[float] = (0.2, 0.3, 0.5),
    target_power: float = 0.8,
    alpha: float = 0.05,
) -> List[Dict[str, float]]:
    """Summarize required sample sizes for several canonical effect sizes."""
    if effect_df.empty:
        return []
    summary: List[Dict[str, float]] = []
    group_cols = ["profile", "phase", "model_name", "metric_name", "paired"]
    for _, group in effect_df.groupby(group_cols, dropna=False):
        paired = bool(group["paired"].iloc[0])
        ratio = 1.0
        if not paired:
            baseline = group["n_baseline"].astype(float)
            patched = group["n_patched"].astype(float)
            ratios = patched / np.maximum(baseline, 1.0)
            ratio = float(np.nanmedian(ratios)) if np.isfinite(ratios).any() else 1.0
        for target in effect_targets:
            req = required_sample_size(
                target,
                target_power=target_power,
                alpha=alpha,
                paired=paired,
                ratio=ratio,
            )
            record: Dict[str, float] = {
                "profile": group["profile"].iloc[0],
                "phase": group["phase"].iloc[0],
                "model_name": group["model_name"].iloc[0],
                "metric_name": group["metric_name"].iloc[0],
                "paired": paired,
                "target_effect_size": float(target),
                "target_power": target_power,
                "alpha": alpha,
            }
            if paired:
                record["required_n_effective"] = req["n_effective"]
            else:
                record["ratio"] = ratio
                record["required_n_control"] = req["n_control"]
                record["required_n_treatment"] = req["n_treatment"]
            summary.append(record)
    return summary
