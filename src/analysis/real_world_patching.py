"""
Real-world activation patching evaluation.
Uses multi-token patching and transformer-based metrics on real-world prompts.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from src.models.activation_patching import ActivationPatcher
from src.analysis.sentiment_eval import SentimentEvaluator


def load_prompts(prompts_file: Path) -> List[str]:
    with open(prompts_file, "r") as f:
        data = json.load(f)
    return data.get("prompts", [])


def evaluate_real_world(
    model_name: str,
    vectors_file: Path,
    prompts_file: Path,
    layer: int,
    alpha_values: List[float],
    max_new_tokens: int,
    patch_window: int | None = None,
):
    patcher = ActivationPatcher(model_name)
    emotion_vectors = patcher.load_emotion_vectors(vectors_file)
    prompts = load_prompts(prompts_file)
    metric_eval = SentimentEvaluator(model_name, load_generation_model=False, enable_transformer_metrics=True)

    results = {
        "model": model_name,
        "layer": layer,
        "alpha_values": alpha_values,
        "prompts": prompts,
        "baseline": {},
        "patched": {},
    }

    print(f"Generating baseline for {len(prompts)} prompts...")
    for prompt in tqdm(prompts, desc="Baseline"):
        text = patcher._generate_text(prompt, max_new_tokens=max_new_tokens)  # type: ignore[attr-defined]
        metrics = metric_eval.evaluate_text_metrics(text)
        results["baseline"][prompt] = {"text": text, "metrics": metrics}

    for emotion_label, vec in emotion_vectors.items():
        results["patched"][emotion_label] = {}
        layer_vec = vec[layer]
        for alpha in alpha_values:
            alpha_outputs = {}
            for prompt in tqdm(prompts, desc=f"{emotion_label} α={alpha}"):
                text = patcher.generate_with_patching(
                    prompt,
                    layer_vec,
                    layer_idx=layer,
                    alpha=alpha,
                    max_new_tokens=max_new_tokens,
                    patch_window=patch_window,
                )
                metrics = metric_eval.evaluate_text_metrics(text)
                alpha_outputs[prompt] = {"text": text, "metrics": metrics}
            results["patched"][emotion_label][alpha] = alpha_outputs

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate real-world patching with multi-token generation.")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--vectors_file", type=str, required=True, help="Emotion vectors file (pickle)")
    parser.add_argument("--prompts_file", type=str, default="data/real_world_samples.json", help="Real-world prompts JSON")
    parser.add_argument("--layer", type=int, default=6, help="Layer index to patch")
    parser.add_argument("--alpha", type=float, nargs="+", default=[-1.0, 0.0, 1.0], help="Alpha values to test")
    parser.add_argument("--max-new-tokens", type=int, default=30, help="Number of tokens to generate")
    parser.add_argument("--patch-window", type=int, default=None, help="Patch window size from sequence end")
    parser.add_argument("--output", type=str, required=True, help="Output path (pickle)")

    args = parser.parse_args()

    results = evaluate_real_world(
        model_name=args.model,
        vectors_file=Path(args.vectors_file),
        prompts_file=Path(args.prompts_file),
        layer=args.layer,
        alpha_values=args.alpha,
        max_new_tokens=args.max_new_tokens,
        patch_window=args.patch_window,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
