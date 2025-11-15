"""
Cross-model patching using alignment linear maps.

Procedure:
1) Load linear maps per layer from model_alignment results (pickled dict with 'linear_maps').
2) Load source emotion vectors and transfer them to the target model space via W @ vec.
3) Run activation patching on the target model using transferred vectors and collect transformer-based metrics.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import numpy as np

from src.models.activation_patching import ActivationPatcher
from src.analysis.sentiment_eval import SentimentEvaluator


def load_linear_maps(alignment_path: Path) -> Dict[int, np.ndarray]:
    with open(alignment_path, "rb") as f:
        data = pickle.load(f)
    return {int(k): np.array(v) for k, v in data.get("linear_maps", {}).items()}


def load_vectors(path: Path) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["emotion_vectors"]


def transfer_vectors(
    src_vectors: Dict[str, np.ndarray],
    linear_maps: Dict[int, np.ndarray],
) -> Dict[str, np.ndarray]:
    transferred = {}
    for emo, vec in src_vectors.items():
        mapped_layers = []
        for layer_idx, layer_vec in enumerate(vec):
            if layer_idx in linear_maps:
                W = linear_maps[layer_idx]
                mapped_layers.append((W @ layer_vec).astype(np.float32))
            else:
                mapped_layers.append(layer_vec)
        transferred[emo] = np.stack(mapped_layers, axis=0)
    return transferred


def run_cross_model_patching(
    target_model: str,
    transferred_vectors: Dict[str, np.ndarray],
    prompts: List[str],
    layer: int,
    alpha: float,
    max_new_tokens: int = 30,
) -> Dict:
    patcher = ActivationPatcher(target_model)
    evaluator = SentimentEvaluator(target_model, load_generation_model=False, enable_transformer_metrics=True)
    results = {
        "model": target_model,
        "layer": layer,
        "alpha": alpha,
        "prompts": prompts,
        "patched": {},
    }
    for emo, vec in transferred_vectors.items():
        layer_vec = vec[layer]
        outputs = {}
        for prompt in tqdm(prompts, desc=f"{emo} patching"):
            text = patcher.generate_with_patching(
                prompt,
                layer_vec,
                layer_idx=layer,
                alpha=alpha,
                max_new_tokens=max_new_tokens,
            )
            metrics = evaluator.evaluate_text_metrics(text)
            outputs[prompt] = {"text": text, "metrics": metrics}
        results["patched"][emo] = outputs
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cross-model patching using alignment maps.")
    parser.add_argument("--alignment", type=str, required=True, help="Pickle from model_alignment (contains linear_maps)")
    parser.add_argument("--src-vectors", type=str, required=True, help="Emotion vectors from source model")
    parser.add_argument("--target-model", type=str, required=True, help="Target HF model id")
    parser.add_argument("--prompts-file", type=str, required=True, help="Prompts JSON with {'prompts': [...]}")
    parser.add_argument("--layer", type=int, default=6, help="Layer index to patch")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha for patching")
    parser.add_argument("--max-new-tokens", type=int, default=30, help="Tokens to generate")
    parser.add_argument("--output", type=str, required=True, help="Output pickle path")

    args = parser.parse_args()

    with open(args.prompts_file, "r") as f:
        prompts_data = json.load(f)
        prompts = prompts_data.get("prompts", [])

    linear_maps = load_linear_maps(Path(args.alignment))
    src_vectors = load_vectors(Path(args.src_vectors))
    transferred = transfer_vectors(src_vectors, linear_maps)

    results = run_cross_model_patching(
        target_model=args.target_model,
        transferred_vectors=transferred,
        prompts=prompts,
        layer=args.layer,
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved cross-model patching results to: {out_path}")


if __name__ == "__main__":
    main()
