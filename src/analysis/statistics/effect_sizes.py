"""Effect size / significance helpers for patching interventions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .data_loading import PatchingRun


@dataclass
class EffectComputationConfig:
    """Parameters for bootstrapping and inference."""

    n_bootstrap: int = 2000
    alpha: float = 0.05
    random_seed: Optional[int] = None


def _bootstrap_ci(
    data: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float,
    statistic: str = "mean",
) -> tuple[float, float]:
    """Return percentile bootstrap CI for the requested statistic."""
    if len(data) == 0:
        return np.nan, np.nan
    if statistic == "mean":
        boot_samples = rng.choice(data, size=(n_bootstrap, len(data)), replace=True)
        stats_arr = boot_samples.mean(axis=1)
    elif statistic == "effect_size":
        boot_samples = rng.choice(data, size=(n_bootstrap, len(data)), replace=True)
        stds = boot_samples.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            stats_arr = np.where(stds > 0, boot_samples.mean(axis=1) / stds, 0.0)
    else:
        raise ValueError(f"Unsupported statistic '{statistic}'")
    lower = np.percentile(stats_arr, 100 * (alpha / 2.0))
    upper = np.percentile(stats_arr, 100 * (1.0 - alpha / 2.0))
    return float(lower), float(upper)


def _bootstrap_unpaired(
    baseline: np.ndarray,
    patched: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Bootstrap CIs for two independent samples."""
    n_base = len(baseline)
    n_patch = len(patched)
    if n_base == 0 or n_patch == 0:
        return (np.nan, np.nan), (np.nan, np.nan)
    diff_stats: List[float] = []
    effect_stats: List[float] = []
    for _ in range(n_bootstrap):
        idx_base = rng.integers(0, n_base, size=n_base)
        idx_patch = rng.integers(0, n_patch, size=n_patch)
        sample_base = baseline[idx_base]
        sample_patch = patched[idx_patch]
        diff_stats.append(float(sample_patch.mean() - sample_base.mean()))
        var_base = sample_base.var(ddof=1)
        var_patch = sample_patch.var(ddof=1)
        pooled = np.sqrt(((n_base - 1) * var_base + (n_patch - 1) * var_patch) / max(n_base + n_patch - 2, 1))
        if pooled > 0:
            effect_stats.append(float((sample_patch.mean() - sample_base.mean()) / pooled))
        else:
            effect_stats.append(0.0)
    low = np.percentile(diff_stats, 100 * (alpha / 2.0))
    high = np.percentile(diff_stats, 100 * (1.0 - alpha / 2.0))
    eff_low = np.percentile(effect_stats, 100 * (alpha / 2.0))
    eff_high = np.percentile(effect_stats, 100 * (1.0 - alpha / 2.0))
    return (float(low), float(high)), (float(eff_low), float(eff_high))


def compute_effect_stats(
    run: PatchingRun,
    config: Optional[EffectComputationConfig] = None,
) -> Dict[str, float]:
    """Compute statistics (paired or independent) for a `PatchingRun`."""
    cfg = config or EffectComputationConfig()
    baseline = np.asarray(run.baseline_values, dtype=float)
    patched = np.asarray(run.patched_values, dtype=float)
    baseline = baseline[~np.isnan(baseline)]
    patched = patched[~np.isnan(patched)]
    n_baseline = baseline.size
    n_patched = patched.size
    if cfg.random_seed is None:
        rng = np.random.default_rng()
    else:
        seed_payload = (
            cfg.random_seed,
            run.profile,
            run.phase,
            run.model_name,
            run.metric_name,
            run.layer,
            run.head,
            run.alpha,
        )
        seed = hash(seed_payload) % (2**63 - 1)
        rng = np.random.default_rng(seed)

    stats_row: Dict[str, float] = {
        "n_baseline": int(n_baseline),
        "n_patched": int(n_patched),
        "mean_baseline": float(baseline.mean()) if n_baseline else np.nan,
        "mean_patched": float(patched.mean()) if n_patched else np.nan,
        "std_baseline": float(baseline.std(ddof=1)) if n_baseline > 1 else np.nan,
        "std_patched": float(patched.std(ddof=1)) if n_patched > 1 else np.nan,
    }

    if run.paired:
        n_effective = min(n_baseline, n_patched)
        stats_row["n_effective"] = int(n_effective)
        if n_effective < 2:
            stats_row.update(
                mean_diff=np.nan,
                cohen_d=np.nan,
                p_value=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                effect_size_ci_lower=np.nan,
                effect_size_ci_upper=np.nan,
            )
            return stats_row
        diffs = patched[:n_effective] - baseline[:n_effective]
        mean_diff = float(diffs.mean())
        std_diff = float(diffs.std(ddof=1))
        stats_row["mean_diff"] = mean_diff
        stats_row["cohen_d"] = float(mean_diff / std_diff) if std_diff > 0 else np.nan
        t_stat, p_value = stats.ttest_rel(patched[:n_effective], baseline[:n_effective])
        stats_row["t_statistic"] = float(t_stat)
        stats_row["p_value"] = float(p_value)
        ci_lower, ci_upper = _bootstrap_ci(
            diffs,
            rng,
            cfg.n_bootstrap,
            cfg.alpha,
            statistic="mean",
        )
        stats_row["ci_lower"] = ci_lower
        stats_row["ci_upper"] = ci_upper
        eff_lower, eff_upper = _bootstrap_ci(
            diffs,
            rng,
            cfg.n_bootstrap,
            cfg.alpha,
            statistic="effect_size",
        )
        stats_row["effect_size_ci_lower"] = eff_lower
        stats_row["effect_size_ci_upper"] = eff_upper
    else:
        if n_baseline < 2 or n_patched < 2:
            stats_row.update(
                mean_diff=np.nan,
                cohen_d=np.nan,
                p_value=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                effect_size_ci_lower=np.nan,
                effect_size_ci_upper=np.nan,
                n_effective=np.nan,
            )
            return stats_row
        mean_diff = float(patched.mean() - baseline.mean())
        var_base = baseline.var(ddof=1)
        var_patch = patched.var(ddof=1)
        pooled = np.sqrt(((n_baseline - 1) * var_base + (n_patched - 1) * var_patch) / (n_baseline + n_patched - 2))
        stats_row["n_effective"] = int(min(n_baseline, n_patched))
        stats_row["mean_diff"] = mean_diff
        stats_row["cohen_d"] = float(mean_diff / pooled) if pooled > 0 else np.nan
        t_stat, p_value = stats.ttest_ind(patched, baseline, equal_var=False)
        stats_row["t_statistic"] = float(t_stat)
        stats_row["p_value"] = float(p_value)
        (ci_lower, ci_upper), (eff_lower, eff_upper) = _bootstrap_unpaired(
            baseline,
            patched,
            rng,
            cfg.n_bootstrap,
            cfg.alpha,
        )
        stats_row["ci_lower"] = ci_lower
        stats_row["ci_upper"] = ci_upper
        stats_row["effect_size_ci_lower"] = eff_lower
        stats_row["effect_size_ci_upper"] = eff_upper
    return stats_row


def aggregate_effects(
    runs: Iterable[PatchingRun],
    config: Optional[EffectComputationConfig] = None,
) -> pd.DataFrame:
    """Compute statistics for a collection of runs and return a DataFrame."""
    cfg = config or EffectComputationConfig()
    rows: List[Dict[str, float]] = []
    for run in runs:
        stats_row = compute_effect_stats(run, cfg)
        stats_row.update(run.metadata_dict())
        rows.append(stats_row)
    if not rows:
        return pd.DataFrame()
    order = [
        "profile",
        "phase",
        "model_name",
        "emotion",
        "layer",
        "head",
        "head_spec",
        "alpha",
        "comparison",
        "metric_name",
        "paired",
        "n_baseline",
        "n_patched",
        "n_effective",
        "mean_baseline",
        "mean_patched",
        "std_baseline",
        "std_patched",
        "mean_diff",
        "cohen_d",
        "ci_lower",
        "ci_upper",
        "effect_size_ci_lower",
        "effect_size_ci_upper",
        "t_statistic",
        "p_value",
    ]
    df = pd.DataFrame(rows)
    for col in order:
        if col not in df.columns:
            df[col] = np.nan
    return df[order]


def _bonferroni_correction(p_values: np.ndarray) -> np.ndarray:
    corrected = p_values * len(p_values)
    corrected = np.minimum(corrected, 1.0)
    return corrected


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = np.empty(n, dtype=float)
    prev = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        val = p_values[order[idx]] * n / rank
        prev = min(prev, val)
        ranked[order[idx]] = min(prev, 1.0)
    return ranked


def apply_multiple_corrections(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply Bonferroni and BH-FDR corrections."""
    if df.empty:
        return df
    group_cols = group_cols or ["profile", "phase", "model_name", "metric_name"]
    df = df.copy()
    df["p_value_bonferroni"] = np.nan
    df["p_value_fdr"] = np.nan
    df["significant_bonferroni"] = False
    df["significant_fdr"] = False
    for _, group in df.groupby(group_cols, dropna=False):
        idx = group.index
        pvals = group["p_value"].astype(float).to_numpy()
        mask = np.isfinite(pvals)
        if not mask.any():
            continue
        valid_idx = idx[mask]
        valid_pvals = pvals[mask]
        bonf = _bonferroni_correction(valid_pvals)
        fdr = _benjamini_hochberg(valid_pvals)
        df.loc[valid_idx, "p_value_bonferroni"] = bonf
        df.loc[valid_idx, "p_value_fdr"] = fdr
        df.loc[valid_idx, "significant_bonferroni"] = bonf < alpha
        df.loc[valid_idx, "significant_fdr"] = fdr < alpha
    return df
