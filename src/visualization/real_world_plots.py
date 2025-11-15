"""
Visualization and statistical testing for real-world patching results.
Consumes pickle files produced by `src.analysis.real_world_patching`.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

MetricPath = str


def load_results(path: Path) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def flatten_metrics(metrics: Dict, prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for k, v in metrics.items():
        name = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_metrics(v, name))
        else:
            try:
                flat[name] = float(v)
            except (TypeError, ValueError):
                continue
    return flat


def build_dataframe(results: Dict) -> pd.DataFrame:
    rows = []
    prompts = results.get("prompts", [])
    baseline = results["baseline"]
    patched = results["patched"]
    layer = results.get("layer")
    for prompt in prompts:
        base_text = baseline[prompt]["text"]
        base_metrics = flatten_metrics(baseline[prompt]["metrics"])
        for emotion, alpha_dict in patched.items():
            for alpha, outputs in alpha_dict.items():
                entry = outputs.get(prompt, {})
                text = entry.get("text", "")
                mets = flatten_metrics(entry.get("metrics", {}))
                row = {
                    "prompt": prompt,
                    "emotion": emotion,
                    "alpha": alpha,
                    "layer": layer,
                    "patch_text": text,
                    "base_text": base_text,
                }
                for k, v in base_metrics.items():
                    row[f"base/{k}"] = v
                for k, v in mets.items():
                    row[f"patch/{k}"] = v
                rows.append(row)
    return pd.DataFrame(rows)


def paired_stats(df: pd.DataFrame, metric: MetricPath) -> Tuple[float, float, float]:
    """Return (mean_diff, t_stat, p_value) for patch - base."""
    base_col = f"base/{metric}"
    patch_col = f"patch/{metric}"
    if base_col not in df or patch_col not in df:
        return 0.0, 0.0, 1.0
    diffs = df[patch_col] - df[base_col]
    diffs = diffs.dropna()
    if len(diffs) == 0:
        return 0.0, 0.0, 1.0
    t_stat, p_val = stats.ttest_1samp(diffs, popmean=0.0)
    return float(np.mean(diffs)), float(t_stat), float(p_val)


def plot_violin(df: pd.DataFrame, metric: MetricPath, out: Path):
    base_col = f"base/{metric}"
    patch_col = f"patch/{metric}"
    if base_col not in df or patch_col not in df:
        return
    data = [df[base_col].dropna(), df[patch_col].dropna()]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.violinplot(data, showmeans=True, showmedians=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Baseline", "Patched"])
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} (real-world)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved violin: {out}")


def plot_bar_diff(df: pd.DataFrame, metric: MetricPath, out: Path):
    base_col = f"base/{metric}"
    patch_col = f"patch/{metric}"
    if base_col not in df or patch_col not in df:
        return
    diff = (df[patch_col] - df[base_col]).dropna()
    if diff.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([0], [diff.mean()], yerr=[diff.std()], capsize=5)
    ax.set_xticks([0])
    ax.set_xticklabels([metric])
    ax.set_ylabel("Patch - Base")
    ax.set_title(f"Mean difference: {diff.mean():.3f}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved bar diff: {out}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize real-world patching results")
    parser.add_argument("--results", type=str, required=True, help="Pickle from real_world_patching")
    parser.add_argument("--metrics", type=str, nargs="+", default=[
        "sentiment/POSITIVE",
        "politeness/politeness_score",
        "emotions/joy",
        "emotions/anger",
    ])
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save plots/CSV")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(Path(args.results))
    df = build_dataframe(results)
    df.to_csv(out_dir / "real_world_metrics.csv", index=False)
    print(f"Saved flattened metrics CSV: {out_dir / 'real_world_metrics.csv'}")

    # Stats and plots per emotion × metric
    for emotion in df["emotion"].unique():
        emo_df = df[df["emotion"] == emotion]
        for metric in args.metrics:
            mean_diff, t_stat, p_val = paired_stats(emo_df, metric)
            print(f"{emotion} / {metric}: mean_diff={mean_diff:.4f}, t={t_stat:.3f}, p={p_val:.4f}")
            plot_violin(emo_df, metric, out_dir / f"violin_{emotion}_{metric.replace('/','_')}.png")
            plot_bar_diff(emo_df, metric, out_dir / f"bar_{emotion}_{metric.replace('/','_')}.png")


if __name__ == "__main__":
    main()
