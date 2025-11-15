"""
Simple neuron saliency analysis for emotion vectors.
For each layer, compute the alignment between emotion vectors and MLP output matrix (W_out),
and rank neurons by absolute contribution.
"""
from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from transformer_lens import HookedTransformer


def load_emotion_vectors(path: Path) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["emotion_vectors"]


def neuron_contribution_scores(model: HookedTransformer, emotion_vec: np.ndarray, layer_idx: int) -> np.ndarray:
    """
    Compute contribution per neuron using |W_out^T @ emotion_vec| heuristic.
    """
    block = model.blocks[layer_idx]
    # mlp.W_out: [d_model, d_mlp]
    W_out = block.mlp.W_out.weight.detach().cpu().numpy()  # [d_model, d_mlp]
    vec = emotion_vec  # [d_model]
    contrib = np.abs(W_out.T @ vec)  # [d_mlp]
    return contrib


def analyze(model_name: str, vectors_path: Path, top_k: int = 20) -> pd.DataFrame:
    model = HookedTransformer.from_pretrained(model_name)
    model.eval()
    emotion_vectors = load_emotion_vectors(vectors_path)
    rows = []
    for emotion, vec in emotion_vectors.items():
        for layer_idx in range(vec.shape[0]):
            contrib = neuron_contribution_scores(model, vec[layer_idx], layer_idx)
            top_indices = np.argsort(contrib)[::-1][:top_k]
            for rank, idx in enumerate(top_indices):
                rows.append({
                    "emotion": emotion,
                    "layer": layer_idx,
                    "neuron": int(idx),
                    "score": float(contrib[idx]),
                    "rank": rank + 1,
                })
    return pd.DataFrame(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Neuron saliency ranking from emotion vectors.")
    parser.add_argument("--model", type=str, required=True, help="Model name (HF id)")
    parser.add_argument("--vectors", type=str, required=True, help="Emotion vectors pickle")
    parser.add_argument("--top-k", type=int, default=20, help="Top neurons per layer/emotion")
    parser.add_argument("--output", type=str, required=True, help="CSV output path")

    args = parser.parse_args()
    df = analyze(args.model, Path(args.vectors), top_k=args.top_k)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved neuron saliency CSV: {out_path}")


if __name__ == "__main__":
    main()
