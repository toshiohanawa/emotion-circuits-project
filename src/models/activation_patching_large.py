"""
LargeHFModel を使った残差パッチング（Phase5向け）の簡易実装。
現状は decoder-only (Llama 系など) を前提に、block list を探索して resid_pre に前置フックを挿入する。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.model_registry import ModelSpec
from src.utils.device import get_default_device_str


class LargeActivationPatcher:
    """
    HFモデルでresid_preへのパッチを行う簡易パッチャ。
    """
    
    def __init__(self, spec: ModelSpec, device: Optional[str] = None):
        self.spec = spec
        self.device = device or get_default_device_str()
        
        print(f"Loading HF model: {spec.hf_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        
        torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        # device_mapはCUDAのみ自動、MPS/CPUは明示移動
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.hf_model_name,
            revision=spec.hf_revision,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device.startswith("cuda") else None,
        )
        if not self.device.startswith("cuda"):
            self.model = self.model.to(self.device)
        self.model.eval()
        
        # block list を推定（Llama系: model.layers, GPT系: transformer.h）
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.blocks = list(self.model.model.layers)
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            self.blocks = list(self.model.transformer.h)
        else:
            raise ValueError("このモデル構造では block list を特定できません。")
        
        self.generation_config = {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": None,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        print(f"✓ HF model loaded on {self.device}")
    
    def _build_alpha_schedule(
        self,
        base_alpha: float,
        max_steps: int,
        alpha_schedule: Optional[Sequence[float]] = None,
        decay_rate: Optional[float] = None,
    ) -> List[float]:
        schedule: List[float] = []
        for step in range(max_steps):
            if alpha_schedule:
                if step < len(alpha_schedule):
                    schedule.append(float(alpha_schedule[step]))
                else:
                    schedule.append(float(alpha_schedule[-1]))
            elif decay_rate is not None:
                schedule.append(float(base_alpha * (decay_rate ** step)))
            else:
                schedule.append(float(base_alpha))
        return schedule
    
    def _resolve_patch_positions(
        self,
        seq_len: int,
        prompt_len: int,
        patch_window: Optional[int],
        patch_positions: Optional[Sequence[int]],
        patch_new_tokens_only: bool,
    ) -> List[int]:
        positions: List[int] = []
        if patch_positions:
            for pos in patch_positions:
                idx = pos if pos >= 0 else seq_len + pos
                positions.append(idx)
        else:
            window = patch_window or 1
            start = max(prompt_len if patch_new_tokens_only else seq_len - window, 0)
            end = seq_len
            positions.extend(list(range(start, end)))
        return positions
    
    def _encode_batch(self, prompts: List[str]):
        enc = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        prompt_lens = attention_mask.sum(dim=1).tolist()
        return input_ids, attention_mask, prompt_lens
    
    def _decode_batch(self, generated_ids: torch.Tensor, prompt_lens: List[int]) -> List[str]:
        texts = []
        for i, plen in enumerate(prompt_lens):
            new_tokens = generated_ids[i, plen:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            texts.append(text)
        return texts
    
    def _generate_text_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 20,
        batch_size: int = 4,
    ) -> List[str]:
        outputs: List[str] = []
        for idx in range(0, len(prompts), batch_size):
            batch_prompts = prompts[idx:idx+batch_size]
            input_ids, attention_mask, prompt_lens = self._encode_batch(batch_prompts)
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    **self.generation_config,
                )
            texts = self._decode_batch(generated, prompt_lens)
            outputs.extend([(p + " " + t).strip() for p, t in zip(batch_prompts, texts)])
        return outputs
    
    def generate_with_patching_batch(
        self,
        prompts: List[str],
        emotion_vector: torch.Tensor,
        layer_idx: int,
        alpha: float = 1.0,
        max_new_tokens: int = 20,
        patch_window: Optional[int] = None,
        patch_positions: Optional[Sequence[int]] = None,
        alpha_schedule: Optional[Sequence[float]] = None,
        alpha_decay_rate: Optional[float] = None,
        patch_new_tokens_only: bool = False,
        batch_size: int = 4,
    ) -> List[str]:
        if layer_idx < 0 or layer_idx >= len(self.blocks):
            raise ValueError(f"layer_idx {layer_idx} は範囲外です (0-{len(self.blocks)-1})")
        
        schedule = self._build_alpha_schedule(
            base_alpha=alpha,
            max_steps=max_new_tokens,
            alpha_schedule=alpha_schedule,
            decay_rate=alpha_decay_rate,
        )
        
        outputs: List[str] = []
        
        for idx in range(0, len(prompts), batch_size):
            batch_prompts = prompts[idx:idx+batch_size]
            input_ids, attention_mask, prompt_lens = self._encode_batch(batch_prompts)
            batch_size_actual = input_ids.shape[0]
            model_dtype = next(self.model.parameters()).dtype
            patch_vec = torch.tensor(emotion_vector, device=self.device, dtype=model_dtype)
            
            def hook_fn(module, args):
                hidden_states = args[0]
                # hidden_states: [batch, seq, hidden]
                seq_len = hidden_states.shape[1]
                hs = hidden_states.clone()
                limit = min(batch_size_actual, hs.shape[0])
                for b in range(limit):
                    plen = prompt_lens[b]
                    generated_len = max(seq_len - plen, 0)
                    step = min(generated_len, max_new_tokens - 1)
                    alpha_value = schedule[step] if step < len(schedule) else 0.0
                    if alpha_value == 0.0:
                        continue
                    indices = self._resolve_patch_positions(
                        seq_len,
                        plen,
                        patch_window,
                        patch_positions,
                        patch_new_tokens_only,
                    )
                    for pos in indices:
                        if 0 <= pos < hs.shape[1]:
                            hs[b, pos, :] += alpha_value * patch_vec
                return (hs,) + tuple(args[1:])
            
            handle = self.blocks[layer_idx].register_forward_pre_hook(hook_fn)
            try:
                with torch.no_grad():
                    generated = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        **self.generation_config,
                    )
            finally:
                handle.remove()
            
            texts = self._decode_batch(generated, prompt_lens)
            outputs.extend([(p + " " + t).strip() for p, t in zip(batch_prompts, texts)])
        
        return outputs
