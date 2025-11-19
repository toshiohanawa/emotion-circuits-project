"""
LargeHFModel で head アブレーションを行う簡易ヘルパー。
decoder-only (Llama 系など) を想定し、self_attn 出力を head 次元でゼロ化する前提。
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.models.model_registry import ModelSpec
from src.utils.device import get_default_device_str


class LargeHeadAblator:
    def __init__(self, spec: ModelSpec, device: str | None = None):
        if not spec.is_large:
            raise ValueError("LargeHeadAblator は大モデル専用です。")
        self.spec = spec
        self.device = device or get_default_device_str()
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        model_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.hf_model_name,
            revision=spec.hf_revision,
            torch_dtype=model_dtype,
            device_map="auto" if self.device.startswith("cuda") else None,
        )
        if not self.device.startswith("cuda"):
            self.model = self.model.to(self.device)
        self.model.eval()

        cfg = AutoConfig.from_pretrained(spec.hf_model_name, revision=spec.hf_revision)
        self.n_heads = getattr(cfg, "num_attention_heads", None)
        if self.n_heads is None:
            raise ValueError("num_attention_heads を取得できませんでした。")

        # block list を特定
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

    def _encode(self, prompts: List[str]):
        enc = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        prompt_lens = attention_mask.sum(dim=1).tolist()
        return input_ids, attention_mask, prompt_lens

    def _decode(self, generated_ids: torch.Tensor, prompt_lens: List[int]) -> List[str]:
        out: List[str] = []
        for i, plen in enumerate(prompt_lens):
            new_tokens = generated_ids[i, plen:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            out.append(text)
        return out

    def generate(
        self,
        prompts: List[str],
        ablate_heads: List[Tuple[int, int]] | None = None,
        max_new_tokens: int = 20,
        batch_size: int = 4,
    ) -> List[str]:
        """
        ablate_heads: [(layer_idx, head_idx), ...] をゼロ化。None のときはベースライン。
        """
        outputs: List[str] = []
        # ヘッドを層ごとにグループ化してフック数を削減
        heads_by_layer: Dict[int, List[int]] = {}
        if ablate_heads:
            for layer_idx, head_idx in ablate_heads:
                heads_by_layer.setdefault(layer_idx, []).append(head_idx)

        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            input_ids, attention_mask, prompt_lens = self._encode(batch_prompts)

            handles = []
            if ablate_heads:
                for layer_idx, head_list in heads_by_layer.items():
                    if layer_idx < 0 or layer_idx >= len(self.blocks):
                        continue
                    def hook_fn(module, _input, output, head_list=head_list):
                        # output: tuple or Tensor
                        attn_out = output[0] if isinstance(output, tuple) else output
                        if attn_out.dim() != 3:
                            return output
                        b, t, hidden = attn_out.shape
                        head_dim = hidden // self.n_heads
                        if head_dim * self.n_heads != hidden:
                            return output
                        attn_out = attn_out.view(b, t, self.n_heads, head_dim)
                        for h in head_list:
                            if 0 <= h < self.n_heads:
                                attn_out[:, :, h, :] = 0
                        attn_out = attn_out.view(b, t, hidden)
                        if isinstance(output, tuple):
                            return (attn_out,) + tuple(output[1:])
                        return attn_out
                    handles.append(self.blocks[layer_idx].self_attn.register_forward_hook(hook_fn))

            try:
                with torch.no_grad():
                    generated = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        **self.generation_config,
                    )
            finally:
                for h in handles:
                    h.remove()

            texts = self._decode(generated, prompt_lens)
            outputs.extend([(p + " " + t).strip() for p, t in zip(batch_prompts, texts)])

        return outputs
