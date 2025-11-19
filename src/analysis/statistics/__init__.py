"""Phase 7 統計的厳密性のためのヘルパーモジュール。

masterplan.md の Phase 7「統計的厳密性」に対応する統計計算モジュール群。
効果量、p値、検出力分析、k選択などの計算を提供する。
"""
from .data_loading import PatchingRun, collect_patching_runs
from .effect_sizes import (
    EffectComputationConfig,
    aggregate_effects,
    apply_multiple_corrections,
    compute_effect_stats,
)
from .power_analysis import estimate_power, required_sample_size, summarize_power_requirements
from .k_selection import collect_k_sweep_results, summarize_k_selection

__all__ = [
    "PatchingRun",
    "collect_patching_runs",
    "EffectComputationConfig",
    "aggregate_effects",
    "apply_multiple_corrections",
    "compute_effect_stats",
    "estimate_power",
    "required_sample_size",
    "summarize_power_requirements",
    "collect_k_sweep_results",
    "summarize_k_selection",
]
