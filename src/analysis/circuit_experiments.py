"""
End-to-end OV/QK circuit experiments and ablations.

Pipeline:
1) Load GPT-2 via TransformerLens new backend (use_attn_result=True)
2) Extract QK routing and OV contributions on emotion prompts
3) Project OV onto emotion vectors to identify influential heads
4) Run OV ablation, QK routing patching, and optional combined neuron+head experiments
5) Compute baseline vs patched deltas for transformer-based metrics
6) Log everything to MLflow and save pkl/csv/png/MD/JSON artifacts
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import mlflow
from src.analysis.circuit_ov_qk import (
    compute_ov_contributions,
    compute_qk_routing,
    load_model_and_tokenizer,
    project_ov_onto_emotion,
)
from src.analysis.sentiment_eval import SentimentEvaluator
from src.utils import mlflow_utils as mlfu
from src.analysis import circuit_report

GEN_CONFIG = {
    "do_sample": False,
    "temperature": 1.0,
    "top_p": None,
    "stop_at_eos": True,
    "return_type": "tokens",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_emotion_vectors(path: Path) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["emotion_vectors"]


def flatten_metrics(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for k, v in d.items():
        name = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_metrics(v, name))
        elif isinstance(v, (int, float)):
            flat[name] = float(v)
    return flat


def mean_nested_metrics(metrics_list: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for item in metrics_list:
        for k, v in flatten_metrics(item).items():
            sums[k] = sums.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    return {k: (sums[k] / counts[k]) for k in sums if counts[k] > 0}


def delta_metrics(baseline: Sequence[Dict[str, Any]], patched: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    base_mean = mean_nested_metrics(baseline)
    pat_mean = mean_nested_metrics(patched)
    keys = set(base_mean.keys()) | set(pat_mean.keys())
    return {k: pat_mean.get(k, 0.0) - base_mean.get(k, 0.0) for k in keys}


def _decode_continuation(model, prompt_tokens: torch.Tensor, generated: torch.Tensor) -> str:
    new_tokens = generated[:, prompt_tokens.shape[1]:]
    text = model.tokenizer.decode(new_tokens[0].tolist(), skip_special_tokens=True)
    return text.strip()


def _generate_with_hooks(
    model,
    prompt: str,
    hook_fns: List[Tuple[str, callable]],
    max_new_tokens: int,
) -> str:
    tokens = model.to_tokens(prompt)
    handles = []
    for hook_name, fn in hook_fns:
        handle = model.add_hook(hook_name, fn)
        handles.append((hook_name, handle))
    try:
        with torch.no_grad():
            generated = model.generate(tokens, max_new_tokens=max_new_tokens, **GEN_CONFIG)
    finally:
        for hook_name, handle in handles:
            if hook_name in model.hook_dict:
                model.hook_dict[hook_name].fwd_hooks = []
    continuation = _decode_continuation(model, tokens, generated)
    return (prompt + " " + continuation).strip()


# ---------------------------------------------------------------------------
# Projection helpers and plots
# ---------------------------------------------------------------------------
def build_projection_table(
    ov_contribs: Dict[int, Any],
    emotion_vectors: Dict[str, np.ndarray],
    layer: int,
) -> pd.DataFrame:
    rows = []
    for emotion, vec in emotion_vectors.items():
        if layer not in ov_contribs or ov_contribs[layer] is None:
            continue
        contrib_tensor = ov_contribs[layer]
        if not isinstance(contrib_tensor, torch.Tensor):
            contrib_tensor = torch.tensor(contrib_tensor)
        head_scores = project_ov_onto_emotion(contrib_tensor, vec[layer])
        for head_idx, (dot, cos) in enumerate(zip(head_scores["dot"], head_scores["cos"])):
            rows.append(
                {
                    "emotion": emotion,
                    "layer": layer,
                    "head": head_idx,
                    "dot": float(dot),
                    "cos": float(cos),
                }
            )
    return pd.DataFrame(rows)


def plot_projection_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    pivot = df.pivot(index="head", columns="emotion", values="cos")
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot.values, aspect="auto", cmap="coolwarm")
    plt.colorbar(label="cosine (OV · emotion)")
    plt.xlabel("Emotion")
    plt.ylabel("Head")
    plt.xticks(ticks=range(len(pivot.columns)), labels=pivot.columns, rotation=45, ha="right")
    plt.yticks(ticks=range(len(pivot.index)), labels=pivot.index)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_qk_heatmap(routing: Dict[int, Dict[str, torch.Tensor]], layer: int, head: int, output_path: Path) -> None:
    if layer not in routing or routing[layer]["pattern"].numel() == 0:
        return
    attn = routing[layer]["pattern"][head].numpy()
    plt.figure(figsize=(6, 5))
    plt.imshow(attn, cmap="viridis")
    plt.colorbar(label="Attention weight")
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.title(f"Layer {layer} Head {head} QK routing")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def run_head_ov_ablation_experiment(
    model,
    evaluator: SentimentEvaluator,
    prompts: Sequence[str],
    heads: Sequence[Tuple[int, int]],
    max_new_tokens: int = 30,
) -> Dict:
    def ablate_hook(activation, hook):
        layer_idx = int(hook.name.split(".")[1])
        target_heads = [h for h in heads if h[0] == layer_idx]
        if not target_heads:
            return activation
        act = activation.clone()
        for _, head_idx in target_heads:
            if act.ndim == 4 and act.shape[1] == model.cfg.n_heads:
                act[0, head_idx, :, :] = 0.0
            elif act.ndim == 4 and act.shape[2] == model.cfg.n_heads:
                act[0, :, head_idx, :] = 0.0
        return act

    baseline, ablated = {}, {}
    for prompt in tqdm(prompts, desc="OV ablation baseline"):
        text = _generate_with_hooks(model, prompt, [], max_new_tokens)
        baseline[prompt] = {"text": text, "metrics": evaluator.evaluate_text_metrics(text)}
    hook_layers = sorted({l for l, _ in heads})
    hook_fns = [(f"blocks.{layer}.attn.hook_result", ablate_hook) for layer in hook_layers]
    for prompt in tqdm(prompts, desc="OV ablated"):
        text = _generate_with_hooks(model, prompt, hook_fns, max_new_tokens)
        ablated[prompt] = {"text": text, "metrics": evaluator.evaluate_text_metrics(text)}

    return {
        "baseline": baseline,
        "ablated": ablated,
        "delta": delta_metrics([v["metrics"] for v in baseline.values()], [v["metrics"] for v in ablated.values()]),
    }


def patch_qk_routing(
    model,
    evaluator: SentimentEvaluator,
    prompts: Sequence[str],
    routing_templates: Dict[int, torch.Tensor],
    heads: Sequence[Tuple[int, int]],
    max_new_tokens: int = 30,
) -> Dict:
    def pattern_hook(activation, hook):
        layer_idx = int(hook.name.split(".")[1])
        if layer_idx not in routing_templates or routing_templates[layer_idx].numel() == 0:
            return activation
        act = activation.clone()
        for patch_layer, head_idx in heads:
            if patch_layer != layer_idx or head_idx >= routing_templates[layer_idx].shape[0]:
                continue
            template = routing_templates[layer_idx][head_idx]  # [seq, seq] from compute_qk_routing
            # Get current sequence lengths from activation: [batch, head, query_pos, key_pos]
            act_query_len = act.shape[-2]  # query_pos dimension
            act_key_len = act.shape[-1]   # key_pos dimension
            template_len = template.shape[-1]  # template is square [seq, seq]
            # Use minimum of template size and activation size
            patch_query_len = min(template_len, act_query_len)
            patch_key_len = min(template_len, act_key_len)
            # Only patch if template is large enough
            if patch_query_len > 0 and patch_key_len > 0:
                act[0, head_idx, :patch_query_len, :patch_key_len] = template[:patch_query_len, :patch_key_len].to(act.device)
        return act

    baseline, patched = {}, {}
    for prompt in tqdm(prompts, desc="QK baseline"):
        text = _generate_with_hooks(model, prompt, [], max_new_tokens)
        baseline[prompt] = {"text": text, "metrics": evaluator.evaluate_text_metrics(text)}
    hook_layers = sorted(routing_templates.keys())
    hook_fns = [(f"blocks.{layer}.attn.hook_pattern", pattern_hook) for layer in hook_layers]
    for prompt in tqdm(prompts, desc="QK patched"):
        text = _generate_with_hooks(model, prompt, hook_fns, max_new_tokens)
        patched[prompt] = {"text": text, "metrics": evaluator.evaluate_text_metrics(text)}

    return {
        "baseline": baseline,
        "patched": patched,
        "delta": delta_metrics([v["metrics"] for v in baseline.values()], [v["metrics"] for v in patched.values()]),
    }


def run_neuron_and_head_combined_experiment(
    model,
    evaluator: SentimentEvaluator,
    prompts: Sequence[str],
    neurons: Sequence[Tuple[int, int]],
    heads: Sequence[Tuple[int, int]],
    max_new_tokens: int = 30,
) -> Dict:
    neuron_mask: Dict[int, List[int]] = {}
    for layer, idx in neurons:
        neuron_mask.setdefault(layer, []).append(idx)

    def mlp_hook(activation, hook):
        layer_idx = int(hook.name.split(".")[1])
        if layer_idx not in neuron_mask:
            return activation
        act = activation.clone()
        act[..., neuron_mask[layer_idx]] = 0.0
        return act

    def ov_hook(activation, hook):
        layer_idx = int(hook.name.split(".")[1])
        target_heads = [h for h in heads if h[0] == layer_idx]
        if not target_heads:
            return activation
        act = activation.clone()
        for _, head_idx in target_heads:
            if act.ndim == 4 and act.shape[1] == model.cfg.n_heads:
                act[0, head_idx, :, :] = 0.0
            elif act.ndim == 4 and act.shape[2] == model.cfg.n_heads:
                act[0, :, head_idx, :] = 0.0
        return act

    baseline, ablated = {}, {}
    for prompt in tqdm(prompts, desc="Combined baseline"):
        text = _generate_with_hooks(model, prompt, [], max_new_tokens)
        baseline[prompt] = {"text": text, "metrics": evaluator.evaluate_text_metrics(text)}

    hook_fns: List[Tuple[str, callable]] = []
    for layer in sorted(neuron_mask.keys()):
        hook_fns.append((f"blocks.{layer}.hook_mlp_out", mlp_hook))
    for layer in sorted({layer for layer, _ in heads}):
        hook_fns.append((f"blocks.{layer}.attn.hook_result", ov_hook))

    for prompt in tqdm(prompts, desc="Combined ablated"):
        text = _generate_with_hooks(model, prompt, hook_fns, max_new_tokens)
        ablated[prompt] = {"text": text, "metrics": evaluator.evaluate_text_metrics(text)}

    return {
        "baseline": baseline,
        "ablated": ablated,
        "delta": delta_metrics([v["metrics"] for v in baseline.values()], [v["metrics"] for v in ablated.values()]),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def _resolve_prompts(prompts_path: Path) -> List[str]:
    with open(prompts_path, "r") as f:
        data = json.load(f)
    for key in ["prompts", "neutral_prompts", "samples"]:
        if key in data:
            return data[key]
    raise ValueError(f"Unrecognized prompt file format: {prompts_path}")


def log_text_artifact(output_dir: Path, name: str, data: Dict[str, Dict[str, Any]]) -> Path:
    path = output_dir / f"{name}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    mlfu.log_artifact_file(str(path))
    return path


def run_circuit_pipeline(
    model_name: str,
    prompts_path: Path,
    emotion_vectors_path: Path,
    layer: int,
    heads: Sequence[Tuple[int, int]],
    neurons: Sequence[Tuple[int, int]],
    max_new_tokens: int,
    output_dir: Path,
) -> Dict[str, Any]:
    prompts = _resolve_prompts(prompts_path)
    model = load_model_and_tokenizer(model_name, use_attn_result=True)
    metric_eval = SentimentEvaluator(
        model_name,
        device=str(model.cfg.device),
        load_generation_model=False,
        enable_transformer_metrics=True,
    )

    print("Computing QK routing caches...")
    routing = compute_qk_routing(model, prompts, layers=[layer])
    print("Computing OV contributions...")
    ov_contribs = compute_ov_contributions(model, prompts, layers=[layer])
    emotion_vectors = load_emotion_vectors(emotion_vectors_path)
    projection_df = build_projection_table({layer: ov_contribs.get(layer, torch.empty(0))}, emotion_vectors, layer)
    projection_per_head = {
        int(row["head"]): float(row["cos"])
        for _, row in projection_df.groupby("head")["cos"].mean().reset_index().iterrows()
    }

    print("Running OV head ablation...")
    ov_ablation = run_head_ov_ablation_experiment(
        model,
        evaluator=metric_eval,
        prompts=prompts,
        heads=heads,
        max_new_tokens=max_new_tokens,
    )

    print("Running QK routing patches...")
    qk_patch = patch_qk_routing(
        model,
        evaluator=metric_eval,
        prompts=prompts,
        routing_templates={layer: routing.get(layer, {}).get("pattern", torch.empty(0))},
        heads=heads,
        max_new_tokens=max_new_tokens,
    )

    combined = None
    if neurons:
        print("Running combined neuron+head experiment...")
        combined = run_neuron_and_head_combined_experiment(
            model,
            evaluator=metric_eval,
            prompts=prompts,
            neurons=neurons,
            heads=heads,
            max_new_tokens=max_new_tokens,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / "ov_qk_results.pkl"
    results = {
        "model": model_name,
        "layer": layer,
        "heads": list(heads),
        "neurons": list(neurons),
        "prompts": prompts,
        "routing": routing,
        "ov_contribs": {layer: ov_contribs.get(layer)},
        "projection_df": projection_df,
        "experiments": {
            "ov_ablation": ov_ablation,
            "qk_patch": qk_patch,
            "combined": combined,
        },
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    csv_path = output_dir / "ov_head_projections.csv"
    projection_df.to_csv(csv_path, index=False)
    png_path = output_dir / "ov_head_projections.png"
    plot_projection_heatmap(projection_df, png_path)

    # QK routing plot (first requested head)
    if heads:
        plot_qk_heatmap(routing, layer=heads[0][0], head=heads[0][1], output_path=output_dir / "qk_routing.png")

    md_path, json_path = circuit_report.export_summary(results, output_dir)

    # MLflow logging
    mlfu.auto_experiment_from_repo()
    with mlflow.start_run(run_name="ov_qk_circuit"):
        # Text artifacts (must be logged inside MLflow run context)
        log_text_artifact(output_dir, "ov_ablation_texts", {**ov_ablation["baseline"], **ov_ablation["ablated"]})
        log_text_artifact(output_dir, "qk_patch_texts", {**qk_patch["baseline"], **qk_patch["patched"]})
        if combined:
            log_text_artifact(output_dir, "combined_texts", {**combined["baseline"], **combined["ablated"]})
        mlfu.log_params_dict(
            {
                "model": model_name,
                "layer": layer,
                "heads": heads,
                "neurons": neurons,
                "max_new_tokens": max_new_tokens,
                "prompts_path": str(prompts_path),
                "emotion_vectors": str(emotion_vectors_path),
            }
        )
        routing_pattern = None
        if layer in routing and isinstance(routing[layer].get("pattern"), torch.Tensor):
            routing_pattern = routing[layer]["pattern"]
        mlfu.log_nested_metrics(
            {
                "circuit": {
                    "ov": {
                        "projection_mean": projection_df["cos"].mean() if not projection_df.empty else 0.0,
                        "projection_per_head": projection_per_head,
                        "delta": ov_ablation["delta"],
                    },
                    "qk": {
                        "delta": qk_patch["delta"],
                        "routing_mean": float(routing_pattern.mean().item()) if routing_pattern is not None and routing_pattern.numel() > 0 else 0.0,
                    },
                }
            }
        )
        if combined:
            mlfu.log_nested_metrics({"circuit": {"ablation": {"delta": combined["delta"]}}})

        mlfu.log_artifact_file(str(pkl_path))
        mlfu.log_artifact_file(str(csv_path))
        mlfu.log_artifact_file(str(png_path))
        if heads:
            mlfu.log_artifact_file(str(output_dir / "qk_routing.png"))
        mlfu.log_artifact_file(str(md_path))
        mlfu.log_artifact_file(str(json_path))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_heads(heads_str: str) -> List[Tuple[int, int]]:
    heads: List[Tuple[int, int]] = []
    for spec in heads_str.split(","):
        if ":" not in spec:
            continue
        try:
            layer_s, head_s = spec.split(":")
            heads.append((int(layer_s), int(head_s)))
        except ValueError:
            continue
    return heads


def parse_neurons(neurons_str: Optional[str]) -> List[Tuple[int, int]]:
    if not neurons_str:
        return []
    neurons: List[Tuple[int, int]] = []
    for part in neurons_str.split(";"):
        if ":" not in part:
            continue
        layer_s, idx_s = part.split(":")
        for n_s in idx_s.split(","):
            try:
                neurons.append((int(layer_s), int(n_s)))
            except ValueError:
                continue
    return neurons


def main():
    parser = argparse.ArgumentParser(description="Run OV/QK circuit pipeline end-to-end.")
    parser.add_argument("--model", type=str, required=True, help="HF model id (e.g., gpt2)")
    parser.add_argument("--prompts", type=str, required=True, help="JSON with prompts list")
    parser.add_argument("--emotion-vectors", type=str, required=True, help="Pickle with emotion_vectors")
    parser.add_argument("--layer", type=int, default=6, help="Layer index to analyze")
    parser.add_argument("--heads", type=str, default="6:0", help='Comma-separated heads, e.g., "6:0,7:3"')
    parser.add_argument("--neurons", type=str, default=None, help='Optional neuron spec "6:10,12;7:8"')
    parser.add_argument("--max-new-tokens", type=int, default=30, help="Generation length")
    parser.add_argument("--output", type=str, default="results/ov_qk", help="Output directory for artifacts")

    args = parser.parse_args()
    heads = parse_heads(args.heads)
    neurons = parse_neurons(args.neurons)

    run_circuit_pipeline(
        model_name=args.model,
        prompts_path=Path(args.prompts),
        emotion_vectors_path=Path(args.emotion_vectors),
        layer=args.layer,
        heads=heads,
        neurons=neurons,
        max_new_tokens=args.max_new_tokens,
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
