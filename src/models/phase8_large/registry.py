"""Registry definitions for Phase 8 large models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

LargeModelName = Literal[
    "llama3_8b",
    "gemma2_9b",  # legacy stub
    "qwen2_5_7b",  # legacy stub
    "qwen2_5_14b",  # legacy stub
    "gemma3_12b",
    "qwen3_8b",
]


@dataclass(frozen=True)
class LargeModelSpec:
    name: LargeModelName
    family: str
    hf_model_name: str
    hf_revision: Optional[str]
    n_layers_hint: Optional[int]
    d_model_hint: Optional[int]
    tokenizer_name: Optional[str] = None
    pretty_name: Optional[str] = None


MODEL_REGISTRY: Dict[LargeModelName, LargeModelSpec] = {
    "llama3_8b": LargeModelSpec(
        name="llama3_8b",
        family="llama3",
        hf_model_name="meta-llama/Meta-Llama-3.1-8B",
        hf_revision=None,
        n_layers_hint=32,
        d_model_hint=4096,
        pretty_name="Llama 3.1 8B (meta-llama/Meta-Llama-3.1-8B)",
    ),
    # Stubs for future extensions (fill n_layers/d_model when confirmed)
    "gemma2_9b": LargeModelSpec(
        name="gemma2_9b",
        family="gemma2",
        hf_model_name="google/gemma-2-9b-it",
        hf_revision=None,
        n_layers_hint=None,
        d_model_hint=None,
    ),
    "qwen2_5_7b": LargeModelSpec(
        name="qwen2_5_7b",
        family="qwen2_5",
        hf_model_name="Qwen/Qwen2.5-7B",
        hf_revision=None,
        n_layers_hint=None,
        d_model_hint=None,
    ),
    "qwen2_5_14b": LargeModelSpec(
        name="qwen2_5_14b",
        family="qwen2_5",
        hf_model_name="Qwen/Qwen2.5-14B",
        hf_revision=None,
        n_layers_hint=None,
        d_model_hint=None,
    ),
    "gemma3_12b": LargeModelSpec(
        name="gemma3_12b",
        family="gemma3",
        hf_model_name="google/gemma-3-12b-it",
        hf_revision=None,
        n_layers_hint=48,
        d_model_hint=3072,
        pretty_name="Gemma 3 12B (google/gemma-3-12b-it)",
    ),
    "qwen3_8b": LargeModelSpec(
        name="qwen3_8b",
        family="qwen3",
        hf_model_name="Qwen/Qwen3-8B-Base",
        hf_revision=None,
        n_layers_hint=36,
        d_model_hint=None,
        pretty_name="Qwen 3 8B (Qwen/Qwen3-8B-Base)",
    ),
}


def get_spec(name: str) -> LargeModelSpec:
    try:
        return MODEL_REGISTRY[name]  # type: ignore[index]
    except KeyError as exc:
        raise KeyError(f"Unknown large model '{name}'. Available: {list(MODEL_REGISTRY.keys())}") from exc
