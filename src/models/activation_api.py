"""
小型〜中規模モデルの活性抽出を共通化するAPI。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from transformer_lens import HookedTransformer

from src.models.model_registry import ModelSpec
from src.models.phase8_large.hf_wrapper import LargeHFModel, load_large_model
from src.utils.device import get_default_device_str


@dataclass
class ActivationBatch:
    model_name: str
    prompts: List[str]
    labels: List[str]
    layer_indices: List[int]
    token_ids: torch.Tensor  # [batch, seq]
    token_strings: List[List[str]]
    resid_pre: torch.Tensor | None  # [batch, layers, seq, d_model]
    resid_post: torch.Tensor | None  # [batch, layers, seq, d_model]

    metadata: Dict[str, str] | None = None


def _capture_small(
    spec: ModelSpec,
    prompts: Sequence[str],
    labels: Sequence[str],
    layers: Sequence[int],
    device: str,
    hook_pos: str = "both",
    batch_size: int = 16,
) -> ActivationBatch:
    """
    小型モデルの活性をバッチ処理で抽出。
    
    Args:
        batch_size: バッチサイズ（デフォルト: 16）
    """
    model = HookedTransformer.from_pretrained(spec.hf_id, device=device)
    model.eval()
    resid_pre_list: List[torch.Tensor] = []
    resid_post_list: List[torch.Tensor] = []
    token_ids: List[torch.Tensor] = []
    token_strings: List[List[str]] = []

    eos_id = getattr(model.tokenizer, "eos_token_id", 0) or 0
    
    # Hook名を事前に決定
    names: List[str] = []
    if hook_pos in ("both", "pre"):
        names.extend([f"blocks.{l}.hook_resid_pre" for l in layers])
    if hook_pos in ("both", "post"):
        names.extend([f"blocks.{l}.hook_resid_post" for l in layers])
    
    # バッチごとに処理
    total = len(prompts)
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_prompts = prompts[batch_start:batch_end]
        batch_labels = labels[batch_start:batch_end]
        
        # バッチ内の各プロンプトをトークン化
        batch_tokens_list = [model.to_tokens(p) for p in batch_prompts]
        batch_token_strings = [model.to_str_tokens(p) for p in batch_prompts]
        
        # 最大長に合わせてパディング
        max_len = max(t.shape[1] for t in batch_tokens_list)
        padded_tokens = []
        for t in batch_tokens_list:
            pad_len = max_len - t.shape[1]
            if pad_len > 0:
                pad = torch.full((t.shape[0], pad_len), eos_id, dtype=t.dtype, device=t.device)
                t = torch.cat([t, pad], dim=1)
            padded_tokens.append(t)
        
        batch_tokens = torch.cat(padded_tokens, dim=0)  # [batch, seq]
        
        # バッチでforward & cache取得
        _, cache = model.run_with_cache(batch_tokens, names_filter=names)
        
        # 各サンプルの活性を抽出
        batch_size_actual = batch_tokens.shape[0]
        for i in range(batch_size_actual):
            # 元のトークン長を取得（パディング前）
            orig_len = batch_tokens_list[i].shape[1]
            token_ids.append(batch_tokens_list[i])
            token_strings.append(batch_token_strings[i])
            
            if hook_pos in ("both", "pre"):
                # cache[f"blocks.{l}.hook_resid_pre"] shape: [batch, seq, d_model]
                pre_stack = torch.stack([
                    cache[f"blocks.{l}.hook_resid_pre"][i, :orig_len, :].to("cpu") 
                    for l in layers
                ], dim=0)  # [layers, seq, d_model]
                resid_pre_list.append(pre_stack)
            
            if hook_pos in ("both", "post"):
                post_stack = torch.stack([
                    cache[f"blocks.{l}.hook_resid_post"][i, :orig_len, :].to("cpu") 
                    for l in layers
                ], dim=0)  # [layers, seq, d_model]
                resid_post_list.append(post_stack)
        
        # 進捗表示
        if batch_end % 50 == 0 or batch_end == total:
            print(f"  [活性抽出] 進捗: {batch_end}/{total}")

    def _pad_ids(tensors: List[torch.Tensor]) -> torch.Tensor:
        max_seq = max(t.shape[1] for t in tensors)
        padded = []
        for t in tensors:
            if t.shape[1] == max_seq:
                padded.append(t.to("cpu"))
                continue
            pad_len = max_seq - t.shape[1]
            pad = torch.full((t.shape[0], pad_len), eos_id, dtype=t.dtype, device="cpu")
            padded.append(torch.cat([t.to("cpu"), pad], dim=1))
        return torch.cat(padded, dim=0)

    def _pad_stack(tensors: List[torch.Tensor]) -> torch.Tensor:
        # tensors: [layers, seq, d_model] for each sample
        max_seq = max(t.shape[1] for t in tensors)
        padded = []
        for t in tensors:
            if t.shape[1] == max_seq:
                padded.append(t)
                continue
            pad_len = max_seq - t.shape[1]
            pad = torch.zeros((t.shape[0], pad_len, t.shape[2]), dtype=t.dtype)
            padded.append(torch.cat([t, pad], dim=1))
        return torch.stack(padded, dim=0)

    resid_pre = _pad_stack(resid_pre_list) if resid_pre_list else None
    resid_post = _pad_stack(resid_post_list) if resid_post_list else None
    token_ids_padded = _pad_ids(token_ids)
    return ActivationBatch(
        model_name=spec.name,
        prompts=list(prompts),
        labels=list(labels),
        layer_indices=list(layers),
        token_ids=token_ids_padded.to("cpu"),
        token_strings=token_strings,
        resid_pre=resid_pre,
        resid_post=resid_post,
        metadata={"source": "transformer_lens"},
    )


def _capture_large(
    spec: ModelSpec,
    prompts: Sequence[str],
    labels: Sequence[str],
    layers: Sequence[int],
    device: str,
    hook_pos: str = "resid_post",
    batch_size: int = 16,
) -> ActivationBatch:
    lf_model: LargeHFModel = load_large_model(spec=spec, device=device, dtype=torch.float16 if device.startswith("cuda") else torch.float32)
    batch = lf_model.get_resid_activations(list(prompts), layers=layers, hook_pos="resid_post" if hook_pos == "resid_post" else "resid_pre")
    token_strings = []
    for ids in batch.token_ids:
        tokens = lf_model.tokenizer.convert_ids_to_tokens(ids.tolist())
        token_strings.append(tokens)
    # Largeモデルはresid_postのみ
    resid_post = torch.stack([batch.layer_activations[l] for l in layers], dim=0)  # [layers, batch, seq, d_model]
    resid_post = resid_post.permute(1, 0, 2, 3).to("cpu")  # [batch, layers, seq, d_model]
    return ActivationBatch(
        model_name=spec.name,
        prompts=list(prompts),
        labels=list(labels),
        layer_indices=list(layers),
        token_ids=batch.token_ids,
        token_strings=token_strings,
        resid_pre=None,
        resid_post=resid_post,
        metadata={"source": "hf_causal_lm"},
    )


def get_activations(
    spec: ModelSpec,
    prompts: Sequence[str],
    labels: Sequence[str],
    layers: Sequence[int],
    device: str = "cpu",
    hook_pos: str = "both",
    batch_size: int = 16,
) -> ActivationBatch:
    """
    指定モデル・プロンプト集合に対して residual 活性を取得する。
    
    Args:
        spec: モデル仕様
        prompts: プロンプトのリスト
        labels: ラベルのリスト
        layers: 取得する層のリスト
        device: 計算デバイス
        hook_pos: 取得する位置（"both", "resid_pre", "resid_post"）
        batch_size: バッチサイズ（デフォルト: 16）
    
    Returns:
        ActivationBatch オブジェクト
    """
    if spec.is_large:
        return _capture_large(spec, prompts, labels, layers, device=device, hook_pos="resid_post", batch_size=batch_size)
    return _capture_small(spec, prompts, labels, layers, device=device, hook_pos=hook_pos, batch_size=batch_size)


def save_activation_batch(batch: ActivationBatch, out_path: Path) -> None:
    import pickle

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": batch.model_name,
        "prompts": batch.prompts,
        "labels": batch.labels,
        "layer_indices": batch.layer_indices,
        "token_ids": batch.token_ids,
        "token_strings": batch.token_strings,
        "resid_pre": batch.resid_pre,
        "resid_post": batch.resid_post,
        "metadata": batch.metadata or {},
    }
    with out_path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"✓ 活性を保存: {out_path}")
