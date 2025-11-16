"""Phase 7.5 statistical analysis helpers."""
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
