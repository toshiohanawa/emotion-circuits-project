"""Visualization helpers for Phase 7.5 statistical outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.project_context import ProjectContext, profile_help_text


def _load_effect_csv(effect_csv: Path) -> pd.DataFrame:
    if not effect_csv.exists():
        raise FileNotFoundError(f"Effect size CSV not found: {effect_csv}")
    df = pd.read_csv(effect_csv)
    numeric_cols = [
        "p_value",
        "p_value_fdr",
        "p_value_bonferroni",
        "cohen_d",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _prepare_output_dir(context: ProjectContext, out_dir: Optional[str]) -> Path:
    if out_dir:
        path = Path(out_dir)
    else:
        path = context.results_dir() / "plots" / "statistics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_pvalue_histograms(df: pd.DataFrame, output_dir: Path, bins: int = 30) -> None:
    """Draw overall and per-metric p-value histograms."""
    if df.empty or "p_value" not in df:
        return
    global_path = output_dir / "pvalue_hist_all.png"
    plt.figure(figsize=(6, 4))
    plt.hist(df["p_value"].dropna(), bins=bins, color="#4C78A8", edgecolor="black")
    plt.axvline(0.05, color="red", linestyle="--", label="0.05")
    plt.xlabel("p-value")
    plt.ylabel("Count")
    plt.title("P-value distribution (all metrics)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(global_path, dpi=200)
    plt.close()

    for metric_name, subset in df.groupby("metric_name"):
        values = subset["p_value"].dropna()
        if values.empty:
            continue
        metric_path = output_dir / f"pvalue_hist_{metric_name.replace('/', '_')}.png"
        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=bins, color="#72B7B2", edgecolor="black")
        plt.axvline(0.05, color="red", linestyle="--")
        plt.xlabel("p-value")
        plt.ylabel("Count")
        plt.title(f"P-value distribution ({metric_name})")
        plt.tight_layout()
        plt.savefig(metric_path, dpi=200)
        plt.close()


def plot_effect_vs_pvalue(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter plot of |effect size| vs p-value."""
    if df.empty or "p_value" not in df or "cohen_d" not in df:
        return
    subset = df[["cohen_d", "p_value", "phase", "metric_name"]].dropna()
    if subset.empty:
        return

    def _scatter(data: pd.DataFrame, label: str, path: Path) -> None:
        plt.figure(figsize=(6, 4))
        plt.scatter(
            data["cohen_d"].abs(),
            data["p_value"],
            c="tab:blue",
            alpha=0.6,
            edgecolor="none",
        )
        plt.axhline(0.05, color="red", linestyle="--", linewidth=1)
        plt.xlabel("|Cohen's d|")
        plt.ylabel("p-value")
        plt.yscale("log")
        plt.title(label)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

    _scatter(subset, "Effect size vs p-value (all)", output_dir / "effect_vs_p_all.png")

    for phase, phase_df in subset.groupby("phase"):
        if phase_df.empty:
            continue
        filename = f"effect_vs_p_{phase}.png"
        _scatter(phase_df, f"Effect size vs p-value ({phase})", output_dir / filename)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot statistics outputs (Phase 7.5)")
    parser.add_argument(
        "--profile",
        type=str,
        default="baseline",
        help=f"Dataset profile ({profile_help_text()})",
    )
    parser.add_argument(
        "--effect-csv",
        type=str,
        default=None,
        help="Path to effect_sizes.csv (default: results/<profile>/statistics/effect_sizes.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for plots (default: results/<profile>/plots/statistics)",
    )
    parser.add_argument("--bins", type=int, default=30, help="Histogram bin count")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    context = ProjectContext(profile_name=args.profile)
    effect_csv = Path(args.effect_csv) if args.effect_csv else context.results_dir() / "statistics" / "effect_sizes.csv"
    output_dir = _prepare_output_dir(context, args.output_dir)

    df = _load_effect_csv(effect_csv)
    plot_pvalue_histograms(df, output_dir, bins=args.bins)
    plot_effect_vs_pvalue(df, output_dir)
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
