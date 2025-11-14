"""
Activation Patching Sweep結果のヒートマップ可視化
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


def load_sweep_results(results_file: Path) -> Dict:
    """スイープ実験の結果を読み込む"""
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    return results


def create_heatmap(
    aggregated_metrics: Dict,
    emotion_label: str,
    metric_name: str,
    layers: List[int],
    alpha_values: List[float],
    output_path: Path,
    title: Optional[str] = None
):
    """
    ヒートマップを作成
    
    Args:
        aggregated_metrics: 集計されたメトリクス
        emotion_label: 感情ラベル
        metric_name: メトリクス名（'gratitude', 'anger', 'apology', 'politeness', 'sentiment'）
        layers: 層のリスト
        alpha_values: α値のリスト
        output_path: 出力パス
        title: タイトル（Noneの場合は自動生成）
    """
    # データを準備
    heatmap_data = np.zeros((len(layers), len(alpha_values)))
    
    for i, layer_idx in enumerate(layers):
        for j, alpha in enumerate(alpha_values):
            if alpha in aggregated_metrics[emotion_label][layer_idx]:
                metrics = aggregated_metrics[emotion_label][layer_idx][alpha]
                
                if metric_name in ['gratitude', 'anger', 'apology']:
                    value = metrics['emotion_keywords'][metric_name]
                elif metric_name == 'politeness':
                    value = metrics['politeness']
                elif metric_name == 'sentiment':
                    value = metrics['sentiment']
                else:
                    value = 0.0
                
                heatmap_data[i, j] = value
    
    # ヒートマップを作成
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # matplotlibでヒートマップを作成
    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(alpha_values)))
    ax.set_xticklabels([f"{α:.1f}" for α in alpha_values])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"Layer {L}" for L in layers])
    ax.set_xlabel('Alpha (α)')
    ax.set_ylabel('Layer')
    plt.colorbar(im, ax=ax, label=metric_name.title())
    
    # 数値を表示
    for i in range(len(layers)):
        for j in range(len(alpha_values)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                         ha="center", va="center", color="white" if heatmap_data[i, j] > heatmap_data.max()/2 else "black")
    
    if title is None:
        title = f"{emotion_label.title()} - {metric_name.title()} Score"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap to: {output_path}")


def create_all_heatmaps(results_file: Path, output_dir: Path):
    """
    全てのヒートマップを作成
    
    Args:
        results_file: スイープ実験の結果ファイル
        output_dir: 出力ディレクトリ
    """
    results = load_sweep_results(results_file)
    aggregated = results['aggregated_metrics']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layers = results['layers']
    alpha_values = results['alpha_values']
    
    # 各感情×各メトリクスでヒートマップを作成
    for emotion_label in results['emotions']:
        # 感情キーワードのヒートマップ
        for keyword in ['gratitude', 'anger', 'apology']:
            output_path = output_dir / f"heatmap_{emotion_label}_{keyword}.png"
            create_heatmap(
                aggregated,
                emotion_label,
                keyword,
                layers,
                alpha_values,
                output_path,
                title=f"{emotion_label.title()} Patching - {keyword.title()} Keywords"
            )
        
        # 丁寧さとsentimentのヒートマップ
        for metric in ['politeness', 'sentiment']:
            output_path = output_dir / f"heatmap_{emotion_label}_{metric}.png"
            create_heatmap(
                aggregated,
                emotion_label,
                metric,
                layers,
                alpha_values,
                output_path,
                title=f"{emotion_label.title()} Patching - {metric.title()} Score"
            )


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create heatmaps from sweep results")
    parser.add_argument("--results_file", type=str, required=True, help="Sweep results file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for heatmaps")
    
    args = parser.parse_args()
    
    create_all_heatmaps(Path(args.results_file), Path(args.output_dir))
    print(f"\nAll heatmaps saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

