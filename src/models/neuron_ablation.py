"""
Neuron-level ablation for MLP units.
zeros out specified neurons in hook_mlp_out during generation and compares metrics.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm

from src.analysis.sentiment_eval import SentimentEvaluator


def parse_neuron_spec(spec: str) -> List[Tuple[int, int]]:
    """
    Parse a spec like "3:10,12;5:7" -> [(3,10),(3,12),(5,7)]
    """
    neurons: List[Tuple[int, int]] = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        layer_str, idx_str = part.split(":", 1)
        try:
            layer = int(layer_str)
        except ValueError:
            continue
        for n_str in idx_str.split(","):
            n_str = n_str.strip()
            if not n_str:
                continue
            try:
                neurons.append((layer, int(n_str)))
            except ValueError:
                continue
    return neurons


@dataclass
class GenerationConfig:
    max_new_tokens: int = 30
    do_sample: bool = False
    temperature: float = 1.0
    top_p: Optional[float] = None
    stop_at_eos: bool = True
    return_type: str = "tokens"


class NeuronAblator:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model: {model_name}")
        self.model = HookedTransformer.from_pretrained(model_name, device=self.device)
        self.model.eval()
        self.gen_cfg = GenerationConfig()
        self.metric_evaluator = SentimentEvaluator(
            model_name,
            device=self.device,
            load_generation_model=False,
            enable_transformer_metrics=True,
        )
        print(f"✓ Model loaded on {self.device}")

    def _generate(
        self,
        prompt: str,
        neuron_mask: Dict[int, List[int]] | None = None,
    ) -> str:
        tokens = self.model.to_tokens(prompt)

        def ablate_mlp(activation, hook):
            if neuron_mask is None:
                return activation
            layer_idx = int(hook.name.split(".")[1])
            if layer_idx in neuron_mask and neuron_mask[layer_idx]:
                activation = activation.clone()
                activation[..., neuron_mask[layer_idx]] = 0.0
            return activation

        handles = []
        if neuron_mask:
            for layer_idx in neuron_mask.keys():
                hook_name = f"blocks.{layer_idx}.hook_mlp_out"
                handle = self.model.add_hook(hook_name, ablate_mlp)
                handles.append((hook_name, handle))
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    tokens,
                    max_new_tokens=self.gen_cfg.max_new_tokens,
                    do_sample=self.gen_cfg.do_sample,
                    temperature=self.gen_cfg.temperature,
                    top_p=self.gen_cfg.top_p,
                    stop_at_eos=self.gen_cfg.stop_at_eos,
                    return_type=self.gen_cfg.return_type,
                )
        finally:
            for hook_name, handle in handles:
                if hook_name in self.model.hook_dict:
                    hook_point = self.model.hook_dict[hook_name]
                    hook_point.fwd_hooks = []

        new_tokens = generated[:, tokens.shape[1]:]
        decoded = self.model.tokenizer.decode(new_tokens[0].tolist(), skip_special_tokens=True)
        return (prompt + " " + decoded).strip()

    def run_ablation(
        self,
        prompts: List[str],
        neuron_list: List[Tuple[int, int]],
        max_new_tokens: int = 30,
    ) -> Dict:
        self.gen_cfg.max_new_tokens = max_new_tokens
        neuron_mask: Dict[int, List[int]] = {}
        for layer, idx in neuron_list:
            neuron_mask.setdefault(layer, []).append(idx)

        results = {
            "model": self.model_name,
            "neurons": neuron_list,
            "max_new_tokens": max_new_tokens,
            "prompts": prompts,
            "baseline": {},
            "ablated": {},
        }

        print(f"Generating baseline for {len(prompts)} prompts...")
        for prompt in tqdm(prompts, desc="Baseline"):
            text = self._generate(prompt, neuron_mask=None)
            metrics = self.metric_evaluator.evaluate_text_metrics(text)
            results["baseline"][prompt] = {"text": text, "metrics": metrics}

        print("Generating with neuron ablation...")
        for prompt in tqdm(prompts, desc="Ablated"):
            text = self._generate(prompt, neuron_mask=neuron_mask)
            metrics = self.metric_evaluator.evaluate_text_metrics(text)
            results["ablated"][prompt] = {"text": text, "metrics": metrics}

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Neuron-level ablation for MLP units.")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--prompts-file", type=str, required=True, help="JSON with {'prompts': [...]}")
    parser.add_argument("--neurons", type=str, required=True, help="Neuron spec e.g., '3:10,12;5:7'")
    parser.add_argument("--max-new-tokens", type=int, default=30, help="Tokens to generate")
    parser.add_argument("--output", type=str, required=True, help="Output pickle path")

    args = parser.parse_args()

    with open(args.prompts_file, "r") as f:
        data = json.load(f)
        prompts = data.get("prompts", [])

    neuron_list = parse_neuron_spec(args.neurons)
    ablator = NeuronAblator(args.model)
    results = ablator.run_ablation(
        prompts=prompts,
        neuron_list=neuron_list,
        max_new_tokens=args.max_new_tokens,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved ablation results to: {out_path}")


if __name__ == "__main__":
    main()
