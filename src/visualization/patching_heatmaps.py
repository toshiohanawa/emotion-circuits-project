"""
Activation Patching Sweep結果の可視化（ヒートマップ + バイオリン）。
新しいTransformerベースのメトリクス構造（ネスト辞書）に対応。
"""
from __future__ import annotations

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def load_sweep_results(results_file: Path) -> Dict:
    """スイープ実験の結果を読み込む"""
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    return results


def _get_metric_value(metrics: Dict, metric_path: str) -> float:
    """
    ネストしたメトリクス辞書から、'/''区切りのパスで値を取得する。
    見つからない場合は0.0を返す。
    """
    parts = metric_path.split("/")
    cur = metrics
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return 0.0
        cur = cur[p]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return 0.0


def create_heatmap(
    aggregated_metrics: Dict,
    emotion_label: str,
    metric_path: str,
    layers: List[int],
    alpha_values: List[float],
    output_path: Path,
    title: Optional[str] = None
):
    """
    ヒートマップを作成
    """
    heatmap_data = np.zeros((len(layers), len(alpha_values)))
    
    for i, layer_idx in enumerate(layers):
        for j, alpha in enumerate(alpha_values):
            if alpha in aggregated_metrics[emotion_label].get(layer_idx, {}):
                metrics = aggregated_metrics[emotion_label][layer_idx][alpha]
                value = _get_metric_value(metrics, metric_path)
                heatmap_data[i, j] = value
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(alpha_values)))
    ax.set_xticklabels([f"{α:.1f}" for α in alpha_values])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"Layer {L}" for L in layers])
    ax.set_xlabel('Alpha (α)')
    ax.set_ylabel('Layer')
    plt.colorbar(im, ax=ax, label=metric_path)
    
    for i_layer in range(len(layers)):
        for j_alpha in range(len(alpha_values)):
            ax.text(j_alpha, i_layer, f'{heatmap_data[i_layer, j_alpha]:.2f}',
                    ha="center", va="center",
                    color="white" if heatmap_data[i_layer, j_alpha] > heatmap_data.max()/2 else "black")
    
    if title is None:
        title = f"{emotion_label.title()} - {metric_path}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to: {output_path}")


def create_violin(
    sweep_results: Dict,
    emotion_label: str,
    metric_path: str,
    layers: Sequence[int],
    alpha_values: Sequence[float],
    output_path: Path,
):
    """
    ラベル付きバイオリンプロットを作成（各αについて一つのバイオリン）。
    """
    fig, axes = plt.subplots(len(layers), 1, figsize=(10, 4 * len(layers)), sharex=True)
    if len(layers) == 1:
        axes = [axes]
    for ax, layer_idx in zip(axes, layers):
        data = []
        labels = []
        for alpha in alpha_values:
            prompt_metrics = sweep_results[emotion_label].get(layer_idx, {}).get(alpha, {}).get('metrics', {})
            values = [_get_metric_value(m, metric_path) for m in prompt_metrics.values() if m]
            if not values:
                continue
            data.append(values)
            labels.append(f"α={alpha}")
        if not data:
            ax.set_title(f"Layer {layer_idx}: no data")
            continue
        ax.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(metric_path)
        ax.set_title(f"{emotion_label} — Layer {layer_idx}")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved violin plot to: {output_path}")


def create_all_visuals(
    results_file: Path,
    output_dir: Path,
    metric_paths: Sequence[str],
    make_violin: bool = True,
):
    """
    全てのヒートマップ・バイオリンを作成
    """
    results = load_sweep_results(results_file)
    aggregated = results['aggregated_metrics']
    sweep_results = results['sweep_results']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layers = results['layers']
    alpha_values = results['alpha_values']
    
    for emotion_label in results['emotions']:
        for metric_path in metric_paths:
            safe_metric = metric_path.replace("/", "_")
            heatmap_path = output_dir / f"heatmap_{emotion_label}_{safe_metric}.png"
            create_heatmap(
                aggregated,
                emotion_label,
                metric_path,
                layers,
                alpha_values,
                heatmap_path,
                title=f"{emotion_label.title()} Patching - {metric_path}"
            )
            if make_violin:
                violin_path = output_dir / f"violin_{emotion_label}_{safe_metric}.png"
                create_violin(
                    sweep_results,
                    emotion_label,
                    metric_path,
                    layers,
                    alpha_values,
                    violin_path,
                )


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create heatmaps/violin plots from sweep results")
    parser.add_argument("--results_file", type=str, required=True, help="Sweep results file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for plots")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=[
            "sentiment/POSITIVE",
            "politeness/politeness_score",
            "emotions/joy",
            "emotions/anger",
            "emotions/sadness",
        ],
        help="Metric paths to plot (nested paths separated by '/')",
    )
    parser.add_argument("--no-violin", action="store_true", help="Disable violin plot generation")
    
    args = parser.parse_args()
    
    create_all_visuals(
        Path(args.results_file),
        Path(args.output_dir),
        metric_paths=args.metrics,
        make_violin=not args.no_violin,
    )
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
