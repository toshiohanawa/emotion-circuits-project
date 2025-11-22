"""
標準パイプラインにおける Phase 7（統計検証）のエントリポイント。
baseline_smoke（少数サンプル）と baseline（1感情225件前後を想定）の結果を集約し、効果量・p値などを算出する。
"""
from __future__ import annotations

import argparse
import json
import time
import sys
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
from src.utils.timing import record_phase_timing


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[save] {path} ({len(df)} rows)")


def run_effect_analysis(args, context: ProjectContext, output_dir: Path) -> pd.DataFrame:
    start_time = time.time()
    print(f"[Phase 7] パッチング結果を収集中...")
    runs = collect_patching_runs(
        context,
        phase_filter=args.phase_filter,
        head_metrics_device=args.head_metrics_device,
        enable_transformer_metrics=not args.disable_transformer_metrics,
    )
    elapsed = time.time() - start_time
    print(f"[Phase 7] 収集完了: {elapsed:.2f}秒 (実行数: {len(runs)})")
    
    if not runs:
        print("No patching runs discovered for the requested filter.")
        return pd.DataFrame()
    
    cfg = EffectComputationConfig(
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        random_seed=args.seed,
        n_jobs=args.n_jobs,
    )
    
    start_time = time.time()
    if args.n_jobs == 1:
        print(f"[Phase 7] 効果量を計算中... (ブートストラップ回数: {args.n_bootstrap}, 逐次処理)")
    else:
        print(f"[Phase 7] 効果量を計算中... (ブートストラップ回数: {args.n_bootstrap}, 並列処理: {args.n_jobs} jobs)")
    effect_df = aggregate_effects(runs, cfg)
    elapsed = time.time() - start_time
    print(f"[Phase 7] 効果量計算完了: {elapsed:.2f}秒 ({elapsed/60:.2f}分)")
    
    start_time = time.time()
    print(f"[Phase 7] 多重比較補正を適用中...")
    effect_df = apply_multiple_corrections(
        effect_df,
        group_cols=args.correction_cols or ["profile", "phase", "model_name", "metric_name"],
        alpha=args.alpha,
    )
    elapsed = time.time() - start_time
    print(f"[Phase 7] 補正完了: {elapsed:.2f}秒")
    
    effect_path = output_dir / "effect_sizes.csv"
    _save_dataframe(effect_df, effect_path)
    return effect_df


def run_power_analysis(args, output_dir: Path, effect_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if effect_df is None or effect_df.empty:
        effect_path = output_dir / "effect_sizes.csv"
        if not effect_path.exists():
            print("Effect size file not found – run with --mode effect first.")
            return pd.DataFrame()
        start_time = time.time()
        print(f"[Phase 7] 効果量ファイルを読み込み中...")
        effect_df = pd.read_csv(effect_path)
        elapsed = time.time() - start_time
        print(f"[Phase 7] 読み込み完了: {elapsed:.2f}秒")
    
    start_time = time.time()
    print(f"[Phase 7] 検出力分析を実行中...")
    power_df = estimate_power(effect_df, alpha=args.alpha)
    elapsed = time.time() - start_time
    print(f"[Phase 7] 検出力分析完了: {elapsed:.2f}秒 ({elapsed/60:.2f}分)")
    
    power_path = output_dir / "power_analysis.csv"
    _save_dataframe(power_df, power_path)
    
    start_time = time.time()
    print(f"[Phase 7] 必要サンプル数を計算中...")
    requirements = summarize_power_requirements(
        power_df,
        effect_targets=tuple(sorted(set(args.effect_targets))),
        target_power=args.power_target,
        alpha=args.alpha,
    )
    elapsed = time.time() - start_time
    print(f"[Phase 7] 計算完了: {elapsed:.2f}秒")
    
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
    start_time = time.time()
    print(f"[Phase 7] k-sweep結果を収集中...")
    k_df = collect_k_sweep_results(context)
    elapsed = time.time() - start_time
    print(f"[Phase 7] 収集完了: {elapsed:.2f}秒 (行数: {len(k_df)})")
    
    if k_df.empty:
        print("No k-sweep files were found under the alignment directory.")
        return pd.DataFrame()
    
    raw_path = output_dir / "k_sweep_raw.csv"
    _save_dataframe(k_df, raw_path)
    
    start_time = time.time()
    if args.n_jobs == 1:
        print(f"[Phase 7] k選択解析を実行中... (ブートストラップ回数: {args.n_bootstrap}, 逐次処理)")
    else:
        print(f"[Phase 7] k選択解析を実行中... (ブートストラップ回数: {args.n_bootstrap}, 並列処理: {args.n_jobs} jobs)")
    summary_df = summarize_k_selection(
        k_df,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        n_jobs=args.n_jobs,
    )
    elapsed = time.time() - start_time
    print(f"[Phase 7] k選択解析完了: {elapsed:.2f}秒 ({elapsed/60:.2f}分)")
    
    summary_path = output_dir / "k_selection.csv"
    _save_dataframe(summary_df, summary_path)
    return summary_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 7: 統計的厳密性（効果量・有意性・k選択）")
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
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for bootstrap computation (1 = sequential, -1 = all CPUs). Default: 1.",
    )
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
    phase7_start = time.time()
    phase_started = time.perf_counter()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    context = ProjectContext(profile_name=args.profile)
    output_dir = Path(args.output_dir) if args.output_dir else context.results_dir() / "statistics"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Phase 7] 統計解析を開始... (モード: {args.mode})")
    
    effect_df: Optional[pd.DataFrame] = None
    if args.mode in {"effect", "all"}:
        effect_df = run_effect_analysis(args, context, output_dir)
    power_df: Optional[pd.DataFrame] = None
    if args.mode in {"power", "all"}:
        power_df = run_power_analysis(args, output_dir, effect_df)
    if args.mode in {"k", "all"}:
        run_k_selection(args, context, output_dir)
    
    phase7_elapsed = time.time() - phase7_start
    print(f"[Phase 7] 統計解析完了: {phase7_elapsed:.2f}秒 ({phase7_elapsed/60:.2f}分)")

    record_phase_timing(
        context=context,
        phase="phase7_statistics",
        started_at=phase_started,
        model=None,
        device=args.head_metrics_device,
        samples=None,
        metadata={
            "mode": args.mode,
            "phase_filter": args.phase_filter,
            "n_bootstrap": args.n_bootstrap,
            "n_jobs": args.n_jobs,
            "alpha": args.alpha,
            "power_target": args.power_target,
            "effect_targets": args.effect_targets,
            "output_dir": str(output_dir),
        },
        cli_args=argv if argv is not None else sys.argv[1:],
    )


if __name__ == "__main__":
    main()
