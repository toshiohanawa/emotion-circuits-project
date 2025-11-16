"""Registry definitions for Phase 8 large models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

LargeModelName = Literal[
    "llama3_8b",
    "gemma2_9b",
    "qwen2_5_7b",
    "qwen2_5_14b",
]


@dataclass(frozen=True)
class LargeModelSpec:
    name: LargeModelName
    family: str
    hf_model_name: str
    hf_revision: Optional[str]
    n_layers: int
    d_model: int
    tokenizer_name: Optional[str] = None


MODEL_REGISTRY: Dict[LargeModelName, LargeModelSpec] = {
    "llama3_8b": LargeModelSpec(
        name="llama3_8b",
        family="llama3",
        hf_model_name="meta-llama/Meta-Llama-3.1-8B",
        hf_revision=None,
        n_layers=32,
        d_model=4096,
    ),
    # Stubs for future extensions (fill n_layers/d_model when confirmed)
    "gemma2_9b": LargeModelSpec(
        name="gemma2_9b",
        family="gemma2",
        hf_model_name="google/gemma-2-9b-it",
        hf_revision=None,
        n_layers=-1,
        d_model=-1,
    ),
    "qwen2_5_7b": LargeModelSpec(
        name="qwen2_5_7b",
        family="qwen2_5",
        hf_model_name="Qwen/Qwen2.5-7B",
        hf_revision=None,
        n_layers=-1,
        d_model=-1,
    ),
    "qwen2_5_14b": LargeModelSpec(
        name="qwen2_5_14b",
        family="qwen2_5",
        hf_model_name="Qwen/Qwen2.5-14B",
        hf_revision=None,
        n_layers=-1,
        d_model=-1,
    ),
}
