"""
OV/QK circuit helpers using TransformerLens (new backend with use_attn_result=True).

This module is intentionally limited to measurement utilities:
- Loading HookedTransformer with attn result enabled
- Extracting Q/K routing (patterns, Q, K)
- Computing OV contributions
- Projecting OV vectors onto emotion directions

All experiment/ablation logic lives in ``circuit_experiments.py``.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformer_lens import HookedTransformer


def load_model_and_tokenizer(
    model_name: str,
    device: Optional[str] = None,
    use_attn_result: bool = True,
) -> HookedTransformer:
    """
    Load HookedTransformer with the new backend flags required for OV/QK hooks.
    """
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not use_attn_result:
        print("Warning: OV/QK analysis requires use_attn_result=True. Overriding to True.")
        use_attn_result = True
    print(f"Loading model: {model_name} (use_attn_result={use_attn_result})")
    model = HookedTransformer.from_pretrained(
        model_name,
        device=resolved_device,
    )
    # Set use_attn_result after loading (not a from_pretrained parameter)
    model.cfg.use_attn_result = use_attn_result
    model.eval()
    print(f"✓ Model loaded on {resolved_device}")
    return model


def _trim_sequences(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """Trim tensors (..., seq, seq) to a common min length for stacking."""
    if not tensors:
        return []
    min_len = min(t.shape[-1] for t in tensors)
    return [t[..., :min_len, :min_len].clone() for t in tensors]


def compute_qk_routing(
    model: HookedTransformer,
    prompts: Sequence[str],
    layers: Optional[Sequence[int]] = None,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Compute mean Q/K routing (pattern, q, k) over prompts.
    """
    target_layers = list(layers) if layers is not None else list(range(model.cfg.n_layers))
    qk_cache: Dict[int, Dict[str, List[torch.Tensor]]] = {
        layer: {"pattern": [], "q": [], "k": []} for layer in target_layers
    }

    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        names = []
        for layer in target_layers:
            base = f"blocks.{layer}.attn."
            names.extend([base + "hook_pattern", base + "hook_q", base + "hook_k"])

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                remove_batch_dim=False,
                names_filter=lambda name: name in names,
            )

        for layer in target_layers:
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"].detach().cpu()  # [1, head, seq, seq]
            q = cache[f"blocks.{layer}.attn.hook_q"].detach().cpu()  # [1, seq, head, d_head]
            k = cache[f"blocks.{layer}.attn.hook_k"].detach().cpu()  # [1, seq, head, d_head]
            qk_cache[layer]["pattern"].append(pattern)
            qk_cache[layer]["q"].append(q)
            qk_cache[layer]["k"].append(k)

    aggregated: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer, data in qk_cache.items():
        trimmed_patterns = _trim_sequences(data["pattern"])
        # For q and k: shape is [1, seq, head, d_head], need to trim to min seq length
        if data["q"]:
            min_seq_len = min(t.shape[1] for t in data["q"])  # seq dimension
            trimmed_q = [t[:, :min_seq_len, :, :].clone() for t in data["q"]]
        else:
            trimmed_q = []
        if data["k"]:
            min_seq_len_k = min(t.shape[1] for t in data["k"])  # seq dimension
            trimmed_k = [t[:, :min_seq_len_k, :, :].clone() for t in data["k"]]
        else:
            trimmed_k = []

        aggregated[layer] = {
            "pattern": torch.stack(trimmed_patterns).mean(dim=0).squeeze(0) if trimmed_patterns else torch.empty(0),
            "q": torch.stack(trimmed_q).mean(dim=0).squeeze(0) if trimmed_q else torch.empty(0),
            "k": torch.stack(trimmed_k).mean(dim=0).squeeze(0) if trimmed_k else torch.empty(0),
        }
    return aggregated


def _extract_W_O(block) -> torch.Tensor:
    """
    Return W_O reshaped per head for the new backend.

    TransformerLens stores W_O with shape [n_heads, d_head, d_model] (already correct)
    or [d_model, d_head * n_heads] (needs reshaping).
    We return [n_heads, d_head, d_model] so that head_out[d_head] @ W_O[h] -> [d_model].
    """
    W_O = block.attn.W_O
    weight = W_O.weight if hasattr(W_O, "weight") else W_O
    weight = weight.detach()
    n_heads = block.attn.cfg.n_heads
    d_head = block.attn.cfg.d_head
    d_model = block.attn.cfg.d_model
    
    # Check if already in [n_heads, d_head, d_model] format
    if len(weight.shape) == 3 and weight.shape == (n_heads, d_head, d_model):
        return weight
    # Otherwise, assume [d_model, d_head * n_heads] and reshape
    elif len(weight.shape) == 2:
    reshaped = weight.view(d_model, n_heads, d_head).permute(1, 2, 0).contiguous()
    return reshaped
    else:
        # Already in correct format or unexpected shape, return as-is
        return weight


def compute_ov_contributions(
    model: HookedTransformer,
    prompts: Sequence[str],
    layers: Optional[Sequence[int]] = None,
    position: int = -1,
) -> Dict[int, torch.Tensor]:
    """
    Compute mean OV contributions per head at the specified position.
    Returns dict[layer] -> tensor [n_heads, d_model] on CPU.
    """
    target_layers = list(layers) if layers is not None else list(range(model.cfg.n_layers))
    layer_outputs: Dict[int, List[torch.Tensor]] = {layer: [] for layer in target_layers}

    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        names = []
        for layer in target_layers:
            base = f"blocks.{layer}.attn."
            names.extend([base + "hook_pattern", base + "hook_v"])
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                remove_batch_dim=False,
                names_filter=lambda name: name in names,
            )

        for layer in target_layers:
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"]  # [1, head, seq, seq]
            v = cache[f"blocks.{layer}.attn.hook_v"]  # [1, seq, head, d_head]
            v_heads = v.permute(0, 2, 1, 3)  # [1, head, seq, d_head]
            head_out = torch.einsum("bhqp,bhpd->bhqd", pattern, v_heads)  # [1, head, seq, d_head]

            pos_idx = position
            if pos_idx < 0 or pos_idx >= head_out.shape[2]:
                pos_idx = head_out.shape[2] - 1

            head_vecs = head_out[0, :, pos_idx, :]  # [head, d_head]
            W_O = _extract_W_O(model.blocks[layer])  # [head, d_head, d_model]

            ov_list: List[torch.Tensor] = []
            for head_idx in range(head_vecs.shape[0]):
                head_matrix = W_O[head_idx]  # [d_head, d_model]
                ov_vec = head_vecs[head_idx] @ head_matrix  # [d_model]
                ov_list.append(ov_vec)
            layer_outputs[layer].append(torch.stack(ov_list, dim=0).detach().cpu())

    aggregated: Dict[int, torch.Tensor] = {}
    for layer, tensors in layer_outputs.items():
        aggregated[layer] = torch.stack(tensors).mean(dim=0) if tensors else torch.empty(0)
    return aggregated


def project_ov_onto_emotion(
    ov_contribs: torch.Tensor,
    emotion_vector: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Project per-head OV contributions onto an emotion vector.
    Returns a dict with raw dot products and cosine similarities (per head).
    """
    if ov_contribs.numel() == 0:
        return {"dot": np.array([]), "cos": np.array([])}
    vec = torch.tensor(emotion_vector, device=ov_contribs.device, dtype=ov_contribs.dtype)
    dot = torch.matmul(ov_contribs, vec)  # [head]
    denom = ov_contribs.norm(dim=1) * torch.norm(vec)
    cos = dot / torch.clamp(denom, min=1e-8)
    return {"dot": dot.cpu().numpy(), "cos": cos.cpu().numpy()}


__all__ = [
    "load_model_and_tokenizer",
    "compute_qk_routing",
    "compute_ov_contributions",
    "project_ov_onto_emotion",
]
