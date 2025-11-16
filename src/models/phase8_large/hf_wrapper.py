"""Minimal HuggingFace wrapper for Phase 8 large models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .registry import LargeModelSpec


@dataclass
class ActivationBatch:
    prompts: List[str]
    token_ids: torch.Tensor  # (batch, seq)
    layer_activations: Dict[int, torch.Tensor]  # layer -> (batch, seq, d_model)


HookPosition = Literal["resid_pre", "resid_post"]


class LargeHFModel:
    """
    Thin wrapper around a HF causal LM that exposes:
    - encode(prompts) -> token_ids
    - get_resid_activations(prompts, layers, hook_pos) -> ActivationBatch
    """

    def __init__(
        self,
        spec: LargeModelSpec,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.spec = spec
        self.device = device
        self.dtype = dtype

        model_name = spec.hf_model_name
        tok_name = spec.tokenizer_name or model_name

        self.tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            # Decoder-only 系は pad が無いことが多いので eos を使う
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # decoder-only は左詰めパディングに統一
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=spec.hf_revision,
            torch_dtype=self.dtype,
            device_map="auto" if device.startswith("cuda") else None,
        )
        # For MPS and CPU, explicitly move model to device
        if not device.startswith("cuda"):
            self.model = self.model.to(device)
        self.model.eval()
        cfg = AutoConfig.from_pretrained(model_name, revision=spec.hf_revision)
        self.n_layers = getattr(cfg, "num_hidden_layers", None)

    def encode(self, prompts: List[str]) -> torch.Tensor:
        """Tokenize prompts into token_ids (batch, seq) on the target device."""
        enc = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        return input_ids

    def _select_hidden(self, hidden_states, layer_idx: int, hook_pos: HookPosition) -> torch.Tensor:
        """
        HuggingFace hidden_states: tuple length = n_layers + 1
        index 0: embedding output (pre layer 0)
        index i: output after layer i-1
        """
        if hook_pos == "resid_pre":
            # pre-layer: use hidden before block (embedding for layer 0)
            idx = layer_idx
        else:  # resid_post
            idx = layer_idx + 1
        return hidden_states[idx]

    def get_resid_activations(
        self,
        prompts: List[str],
        layers: Sequence[int],
        hook_pos: HookPosition = "resid_post",
        max_new_tokens: int | None = None,
    ) -> ActivationBatch:
        """
        Run the model on prompts and capture residual-like activations at specified layers.
        """
        input_ids = self.encode(prompts)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = outputs.hidden_states  # tuple len = n_layers + 1
        layer_acts: Dict[int, torch.Tensor] = {}
        for layer_idx in layers:
            layer_tensor = self._select_hidden(hidden_states, layer_idx, hook_pos)
            # Ensure full precision on CPU if needed
            layer_acts[layer_idx] = layer_tensor.detach().to("cpu")

        return ActivationBatch(
            prompts=prompts,
            token_ids=input_ids.to("cpu"),
            layer_activations=layer_acts,
        )


def load_large_model(spec: LargeModelSpec, device: str = "cuda", dtype: torch.dtype = torch.float16) -> LargeHFModel:
    """Factory for LargeHFModel."""
    return LargeHFModel(spec, device=device, dtype=dtype)
