"""CLI entrypoint for Phase 7.5 statistical validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.utils.project_context import ProjectContext, profile_help_text
from src.analysis.statistics.data_loading import collect_patching_runs
from src.analysis.statistics.effect_sizes import (
    EffectComputationConfig,
    aggregate_effects,
    apply_multiple_corrections,
)
from src.analysis.statistics.power_analysis import (
    estimate_power,
    summarize_power_requirements,
)
from src.analysis.statistics.k_selection import collect_k_sweep_results, summarize_k_selection


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[save] {path} ({len(df)} rows)")


def run_effect_analysis(args, context: ProjectContext, output_dir: Path) -> pd.DataFrame:
    runs = collect_patching_runs(
        context,
        phase_filter=args.phase_filter,
        head_metrics_device=args.head_metrics_device,
        enable_transformer_metrics=not args.disable_transformer_metrics,
    )
    if not runs:
        print("No patching runs discovered for the requested filter.")
        return pd.DataFrame()
    cfg = EffectComputationConfig(
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        random_seed=args.seed,
    )
    effect_df = aggregate_effects(runs, cfg)
    effect_df = apply_multiple_corrections(
        effect_df,
        group_cols=args.correction_cols or ["profile", "phase", "model_name", "metric_name"],
        alpha=args.alpha,
    )
    effect_path = output_dir / "effect_sizes.csv"
    _save_dataframe(effect_df, effect_path)
    return effect_df


def run_power_analysis(args, output_dir: Path, effect_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if effect_df is None or effect_df.empty:
        effect_path = output_dir / "effect_sizes.csv"
        if not effect_path.exists():
            print("Effect size file not found – run with --mode effect first.")
            return pd.DataFrame()
        effect_df = pd.read_csv(effect_path)
    power_df = estimate_power(effect_df, alpha=args.alpha)
    power_path = output_dir / "power_analysis.csv"
    _save_dataframe(power_df, power_path)
    requirements = summarize_power_requirements(
        power_df,
        effect_targets=tuple(sorted(set(args.effect_targets))),
        target_power=args.power_target,
        alpha=args.alpha,
    )
    summary = {
        "profile": args.profile,
        "alpha": args.alpha,
        "target_power": args.power_target,
        "requirements": requirements,
        "post_hoc_power_csv": str(power_path),
    }
    json_path = output_dir / "power_analysis.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[save] {json_path}")
    return power_df


def run_k_selection(args, context: ProjectContext, output_dir: Path) -> pd.DataFrame:
    k_df = collect_k_sweep_results(context)
    if k_df.empty:
        print("No k-sweep files were found under the alignment directory.")
        return pd.DataFrame()
    raw_path = output_dir / "k_sweep_raw.csv"
    _save_dataframe(k_df, raw_path)
    summary_df = summarize_k_selection(
        k_df,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
    )
    summary_path = output_dir / "k_selection.csv"
    _save_dataframe(summary_df, summary_path)
    return summary_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 7.5 statistical validation pipeline")
    parser.add_argument(
        "--profile",
        type=str,
        default="baseline",
        help=f"Dataset profile ({profile_help_text()})",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["effect", "power", "k", "all"],
        default="all",
        help="Which stage to run.",
    )
    parser.add_argument(
        "--phase-filter",
        type=str,
        default=None,
        help="Comma-separated list of phases (residual,head,random). Default: all.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the statistics output directory (default: results/<profile>/statistics)",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000, help="Number of bootstrap samples.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for tests and CIs.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for bootstrap reproducibility.")
    parser.add_argument(
        "--head-metrics-device",
        type=str,
        default="cpu",
        help="Device to use when recomputing head metrics (cpu/cuda).",
    )
    parser.add_argument(
        "--disable-transformer-metrics",
        action="store_true",
        help="Use heuristic keyword/sentiment metrics instead of transformer classifiers for head patching.",
    )
    parser.add_argument(
        "--correction-cols",
        type=str,
        nargs="*",
        default=None,
        help="Columns used to group p-values before multiple comparison correction.",
    )
    parser.add_argument(
        "--power-target",
        type=float,
        default=0.8,
        help="Desired statistical power for sample-size projections.",
    )
    parser.add_argument(
        "--effect-targets",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.5],
        help="Effect sizes (Cohen's d) to evaluate when projecting sample sizes.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    context = ProjectContext(profile_name=args.profile)
    output_dir = Path(args.output_dir) if args.output_dir else context.results_dir() / "statistics"
    output_dir.mkdir(parents=True, exist_ok=True)

    effect_df: Optional[pd.DataFrame] = None
    if args.mode in {"effect", "all"}:
        effect_df = run_effect_analysis(args, context, output_dir)
    power_df: Optional[pd.DataFrame] = None
    if args.mode in {"power", "all"}:
        power_df = run_power_analysis(args, output_dir, effect_df)
    if args.mode in {"k", "all"}:
        run_k_selection(args, context, output_dir)


if __name__ == "__main__":
    main()
