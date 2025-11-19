"""
小型〜中規模モデルの統一レジストリ。

小型モデル: TransformerLensベース (HookedTransformer)
中規模モデル: HuggingFace (LargeHFModel)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    family: str
    n_layers_hint: int | None
    d_model_hint: int | None
    pretty_name: str
    is_large: bool = False  # LargeHFModel を使う場合 True
    hf_revision: str | None = None
    tokenizer_name: str | None = None

    # Phase8 LargeHFModel 互換アクセサ
    @property
    def hf_model_name(self) -> str:
        return self.hf_id


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # 小型
    "gpt2": ModelSpec(
        name="gpt2",
        hf_id="gpt2",
        family="gpt2",
        n_layers_hint=12,
        d_model_hint=768,
        pretty_name="GPT-2 small",
    ),
    "gpt2_small": ModelSpec(
        name="gpt2_small",
        hf_id="gpt2",
        family="gpt2",
        n_layers_hint=12,
        d_model_hint=768,
        pretty_name="GPT-2 small",
    ),
    "pythia-160m": ModelSpec(
        name="pythia-160m",
        hf_id="EleutherAI/pythia-160m",
        family="pythia",
        n_layers_hint=12,
        d_model_hint=768,
        pretty_name="Pythia 160M",
    ),
    "gpt-neo-125m": ModelSpec(
        name="gpt-neo-125m",
        hf_id="EleutherAI/gpt-neo-125M",
        family="gpt-neo",
        n_layers_hint=12,
        d_model_hint=768,
        pretty_name="GPT-Neo 125M",
    ),
    # 中規模（LargeHFModel）
    "llama3_8b": ModelSpec(
        name="llama3_8b",
        hf_id="meta-llama/Meta-Llama-3.1-8B",
        family="llama3",
        n_layers_hint=32,
        d_model_hint=4096,
        pretty_name="Llama 3.1 8B",
        is_large=True,
    ),
    "gemma3_12b": ModelSpec(
        name="gemma3_12b",
        hf_id="google/gemma-3-12b-it",
        family="gemma3",
        n_layers_hint=48,
        d_model_hint=3072,
        pretty_name="Gemma 3 12B",
        is_large=True,
    ),
    "qwen3_8b": ModelSpec(
        name="qwen3_8b",
        hf_id="Qwen/Qwen3-8B-Base",
        family="qwen3",
        n_layers_hint=36,
        d_model_hint=None,
        pretty_name="Qwen 3 8B",
        is_large=True,
    ),
}


def list_model_names() -> List[str]:
    return sorted(MODEL_REGISTRY.keys())


def list_small_models() -> List[str]:
    return sorted([m for m, spec in MODEL_REGISTRY.items() if not spec.is_large])


def list_large_models() -> List[str]:
    return sorted([m for m, spec in MODEL_REGISTRY.items() if spec.is_large])


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"未知のモデル名です: {name}. 利用可能: {list_model_names()}") from exc
