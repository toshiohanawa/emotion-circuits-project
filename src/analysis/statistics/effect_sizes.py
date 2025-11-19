"""Effect size / significance helpers for patching interventions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    # fallback to multiprocessing if joblib is not available
    from multiprocessing import Pool
    from functools import partial

from .data_loading import PatchingRun


@dataclass
class EffectComputationConfig:
    """Parameters for bootstrapping and inference."""

    n_bootstrap: int = 2000
    alpha: float = 0.05
    random_seed: Optional[int] = None
    n_jobs: int = 1  # Number of parallel jobs for bootstrap (1 = sequential)


def _bootstrap_sample_mean(data: np.ndarray, seed: int) -> float:
    """Single bootstrap sample for mean statistic (for parallelization)."""
    rng = np.random.default_rng(seed)
    sample = rng.choice(data, size=len(data), replace=True)
    return float(sample.mean())


def _bootstrap_sample_effect_size(data: np.ndarray, seed: int) -> float:
    """Single bootstrap sample for effect size statistic (for parallelization)."""
    rng = np.random.default_rng(seed)
    sample = rng.choice(data, size=len(data), replace=True)
    std = sample.std(ddof=1)
    if std > 0:
        return float(sample.mean() / std)
    return 0.0


def _bootstrap_ci(
    data: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float,
    statistic: str = "mean",
    n_jobs: int = 1,
) -> tuple[float, float]:
    """
    Return percentile bootstrap CI for the requested statistic.
    
    Args:
        data: Input data array
        rng: Random number generator (for seed generation)
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        statistic: Type of statistic ("mean" or "effect_size")
        n_jobs: Number of parallel jobs (1 = sequential)
    
    Returns:
        (lower, upper) confidence interval bounds
    """
    if len(data) == 0:
        return np.nan, np.nan
    
    # Generate seeds for reproducibility
    seeds = rng.integers(0, 2**31, size=n_bootstrap)
    
    if n_jobs == 1 or not HAS_JOBLIB:
        # Sequential processing
        if statistic == "mean":
            stats_arr = np.array([_bootstrap_sample_mean(data, int(seed)) for seed in seeds])
        elif statistic == "effect_size":
            stats_arr = np.array([_bootstrap_sample_effect_size(data, int(seed)) for seed in seeds])
        else:
            raise ValueError(f"Unsupported statistic '{statistic}'")
    else:
        # Parallel processing with joblib
        if statistic == "mean":
            stats_arr = np.array(
                Parallel(n_jobs=n_jobs)(
                    delayed(_bootstrap_sample_mean)(data, int(seed)) for seed in seeds
                )
            )
        elif statistic == "effect_size":
            stats_arr = np.array(
                Parallel(n_jobs=n_jobs)(
                    delayed(_bootstrap_sample_effect_size)(data, int(seed)) for seed in seeds
                )
            )
        else:
            raise ValueError(f"Unsupported statistic '{statistic}'")
    
    lower = np.percentile(stats_arr, 100 * (alpha / 2.0))
    upper = np.percentile(stats_arr, 100 * (1.0 - alpha / 2.0))
    return float(lower), float(upper)


def _bootstrap_unpaired_sample(
    baseline: np.ndarray,
    patched: np.ndarray,
    seed: int,
) -> Tuple[float, float]:
    """
    Single bootstrap sample for unpaired comparison (for parallelization).
    
    Returns:
        (diff_stat, effect_stat) tuple
    """
    rng = np.random.default_rng(seed)
    n_base = len(baseline)
    n_patch = len(patched)
    
    idx_base = rng.integers(0, n_base, size=n_base)
    idx_patch = rng.integers(0, n_patch, size=n_patch)
    sample_base = baseline[idx_base]
    sample_patch = patched[idx_patch]
    
    diff_stat = float(sample_patch.mean() - sample_base.mean())
    var_base = sample_base.var(ddof=1)
    var_patch = sample_patch.var(ddof=1)
    pooled = np.sqrt(((n_base - 1) * var_base + (n_patch - 1) * var_patch) / max(n_base + n_patch - 2, 1))
    
    if pooled > 0:
        effect_stat = float((sample_patch.mean() - sample_base.mean()) / pooled)
    else:
        effect_stat = 0.0
    
    return diff_stat, effect_stat


def _bootstrap_unpaired(
    baseline: np.ndarray,
    patched: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float,
    n_jobs: int = 1,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Bootstrap CIs for two independent samples.
    
    Args:
        baseline: Baseline sample array
        patched: Patched sample array
        rng: Random number generator (for seed generation)
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        n_jobs: Number of parallel jobs (1 = sequential)
    
    Returns:
        ((diff_lower, diff_upper), (effect_lower, effect_upper)) confidence intervals
    """
    n_base = len(baseline)
    n_patch = len(patched)
    if n_base == 0 or n_patch == 0:
        return (np.nan, np.nan), (np.nan, np.nan)
    
    # Generate seeds for reproducibility
    seeds = rng.integers(0, 2**31, size=n_bootstrap)
    
    if n_jobs == 1 or not HAS_JOBLIB:
        # Sequential processing
        results = [_bootstrap_unpaired_sample(baseline, patched, int(seed)) for seed in seeds]
    else:
        # Parallel processing with joblib
        results = Parallel(n_jobs=n_jobs)(
            delayed(_bootstrap_unpaired_sample)(baseline, patched, int(seed)) for seed in seeds
        )
    
    diff_stats = [r[0] for r in results]
    effect_stats = [r[1] for r in results]
    
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
            n_jobs=cfg.n_jobs,
        )
        stats_row["ci_lower"] = ci_lower
        stats_row["ci_upper"] = ci_upper
        eff_lower, eff_upper = _bootstrap_ci(
            diffs,
            rng,
            cfg.n_bootstrap,
            cfg.alpha,
            statistic="effect_size",
            n_jobs=cfg.n_jobs,
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
            n_jobs=cfg.n_jobs,
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
