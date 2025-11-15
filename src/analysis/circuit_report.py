"""
Reporting utilities for OV/QK circuit experiments.

Produces:
- QK routing heatmaps (token→token) per layer/head
- Head-importance heatmap (OV projection cosine)
- Circuit summaries (Markdown + JSON) combining OV/QK stats and metric deltas
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch


def load_results(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_qk_routing_map(routing: Dict[int, Dict[str, torch.Tensor]], layer: int, head: int, output_path: Path) -> None:
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


def plot_head_importance_heatmap(projection_df: pd.DataFrame, output_path: Path) -> None:
    if projection_df.empty:
        return
    pivot = projection_df.pivot(index="head", columns="emotion", values="cos")
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot.values, aspect="auto", cmap="coolwarm")
    plt.colorbar(label="Head importance (cosine)")
    plt.xlabel("Emotion")
    plt.ylabel("Head")
    plt.xticks(ticks=range(len(pivot.columns)), labels=pivot.columns, rotation=45, ha="right")
    plt.yticks(ticks=range(len(pivot.index)), labels=pivot.index)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def summarize_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    exps = results.get("experiments", {})
    def _delta(name: str) -> Dict[str, float]:
        exp = exps.get(name)
        if exp and "delta" in exp:
            return exp["delta"]
        return {}

    ov_delta = _delta("ov_ablation")
    qk_delta = _delta("qk_patch")
    combined_delta = _delta("combined")

    proj_df = results.get("projection_df")
    proj_summary = {}
    if isinstance(proj_df, pd.DataFrame) and not proj_df.empty:
        proj_summary = {
            "cos_mean": float(proj_df["cos"].mean()),
            "cos_max": float(proj_df["cos"].max()),
            "cos_min": float(proj_df["cos"].min()),
        }
    return {
        "ov_projection": proj_summary,
        "delta": {
            "ov": ov_delta,
            "qk": qk_delta,
            "combined": combined_delta,
        },
    }


def render_markdown(summary: Dict[str, Any], output_path: Path) -> None:
    lines = ["# Circuit Summary", ""]
    proj = summary.get("ov_projection", {})
    if proj:
        lines.append("## OV Projection")
        lines.append(f"- cos_mean: {proj.get('cos_mean', 0):.4f}")
        lines.append(f"- cos_max: {proj.get('cos_max', 0):.4f}")
        lines.append(f"- cos_min: {proj.get('cos_min', 0):.4f}")
        lines.append("")
    delta = summary.get("delta", {})
    lines.append("## Δ Metrics")
    for name, vals in delta.items():
        lines.append(f"### {name}")
        for k, v in vals.items():
            lines.append(f"- {k}: {v:.4f}")
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def export_summary(results: Dict[str, Any], output_dir: Path) -> Tuple[Path, Path]:
    summary = summarize_metrics(results)
    md_path = output_dir / "circuit_summary.md"
    json_path = output_dir / "circuit_summary.json"
    render_markdown(summary, md_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return md_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Summarize OV/QK circuit experiment results.")
    parser.add_argument("--results", type=str, required=True, help="Pickle produced by circuit_experiments.py")
    parser.add_argument("--output", type=str, default="results/ov_qk/report", help="Output directory")
    parser.add_argument("--head", type=str, default=None, help="Optional single head plot e.g., '6:0'")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(Path(args.results))
    projection = results.get("projection_df")
    if isinstance(projection, pd.DataFrame):
        plot_head_importance_heatmap(projection, output_dir / "head_importance.png")

    if args.head:
        if ":" in args.head:
            layer_s, head_s = args.head.split(":")
            plot_qk_routing_map(
                results.get("routing", {}),
                layer=int(layer_s),
                head=int(head_s),
                output_path=output_dir / "qk_routing.png",
            )

    md_path, json_path = export_summary(results, output_dir)
    print(f"Saved circuit summary: {md_path}, {json_path}")


if __name__ == "__main__":
    main()
