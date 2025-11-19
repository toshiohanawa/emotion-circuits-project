"""Subspace dimensionality diagnostics (k-selection)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from src.utils.project_context import ProjectContext


def load_k_sweep_file(result_file: Path, profile: str) -> pd.DataFrame:
    """Load a single k-sweep pickle into a long-form DataFrame."""
    import pickle

    with open(result_file, "rb") as f:
        data = pickle.load(f)
    rows: List[Dict] = []
    model_a = data.get("model1") or data.get("model_a") or "model_a"
    model_b = data.get("model2") or data.get("model_b") or "model_b"
    sweep_results = data.get("sweep_results", {})
    for layer_idx, emotion_dict in sweep_results.items():
        for emotion, k_dict in emotion_dict.items():
            for k, metrics in k_dict.items():
                rows.append(
                    {
                        "profile": profile,
                        "model_a": model_a,
                        "model_b": model_b,
                        "layer": int(layer_idx),
                        "emotion": emotion,
                        "k": int(k),
                        "overlap": float(metrics.get("overlap_cos_squared", np.nan)),
                        "mean_principal_angle": float(metrics.get("mean_principal_angle", np.nan)),
                    }
                )
    return pd.DataFrame(rows)


def collect_k_sweep_results(context: ProjectContext) -> pd.DataFrame:
    """Collect all k-sweep pickles under `results/<profile>/alignment`."""
    alignment_dir = context.results_dir() / "alignment"
    if not alignment_dir.exists():
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for path in sorted(alignment_dir.glob("k_sweep_*.pkl")):
        frames.append(load_k_sweep_file(path, context.profile_name))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _bootstrap_k_sample(overlap: np.ndarray, seed: int) -> float:
    """Single bootstrap sample for k-selection (for parallelization)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(overlap), size=len(overlap))
    return float(overlap[idx].mean())


def summarize_k_selection(
    k_df: pd.DataFrame,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Compute bootstrap summaries for overlap as a function of k.
    
    Args:
        k_df: DataFrame with k-sweep results
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        n_jobs: Number of parallel jobs (1 = sequential)
    
    Returns:
        DataFrame with bootstrap summaries
    """
    if k_df.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(0)
    records: List[Dict] = []
    group_cols = ["profile", "model_a", "model_b", "emotion", "k"]
    for keys, group in k_df.groupby(group_cols, dropna=False):
        overlap = group["overlap"].dropna().to_numpy()
        if overlap.size == 0:
            continue
        mean_overlap = float(overlap.mean())
        std_overlap = float(overlap.std(ddof=1)) if overlap.size > 1 else 0.0
        if overlap.size > 1:
            # Generate seeds for reproducibility
            seeds = rng.integers(0, 2**31, size=n_bootstrap)
            
            if n_jobs == 1 or not HAS_JOBLIB:
                # Sequential processing
                boot_means = np.array([_bootstrap_k_sample(overlap, int(seed)) for seed in seeds])
            else:
                # Parallel processing with joblib
                boot_means = np.array(
                    Parallel(n_jobs=n_jobs)(
                        delayed(_bootstrap_k_sample)(overlap, int(seed)) for seed in seeds
                    )
                )
            ci_lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
            ci_upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
        else:
            ci_lower = ci_upper = mean_overlap
        record = dict(zip(group_cols, keys))
        record.update(
            {
                "n_layers": int(overlap.size),
                "mean_overlap": mean_overlap,
                "std_overlap": std_overlap,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )
        records.append(record)
    return pd.DataFrame(records)
