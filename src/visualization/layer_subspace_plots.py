"""
層ごとのサブスペース可視化
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


def plot_k_sweep_results(
    results: Dict,
    output_path: Path,
    layers: Optional[List[int]] = None,
    emotions: Optional[List[str]] = None
):
    """
    k-sweep結果を可視化（k値ごとのoverlap変化）
    
    Args:
        results: k-sweep実験の結果
        output_path: 出力ファイルパス
        layers: 対象層のリスト（Noneの場合は全層）
        emotions: 対象感情のリスト（Noneの場合は全感情）
    """
    if layers is None:
        layers = results['layers']
    if emotions is None:
        emotions = results['emotions']
    
    k_values = results['k_values']
    
    n_emotions = len(emotions)
    fig, axes = plt.subplots(1, n_emotions, figsize=(5 * n_emotions, 5))
    
    if n_emotions == 1:
        axes = [axes]
    
    for emotion_idx, emotion_label in enumerate(emotions):
        ax = axes[emotion_idx]
        
        # 各層のk値ごとのoverlapを集計
        for layer_idx in layers:
            if layer_idx in results['sweep_results']:
                if emotion_label in results['sweep_results'][layer_idx]:
                    overlaps = []
                    for k in k_values:
                        if k in results['sweep_results'][layer_idx][emotion_label]:
                            overlap = results['sweep_results'][layer_idx][emotion_label][k]['overlap_cos_squared']
                            overlaps.append(overlap)
                        else:
                            overlaps.append(0.0)
                    
                    ax.plot(k_values, overlaps, marker='o', label=f'Layer {layer_idx}', alpha=0.7)
        
        ax.set_xlabel('k (PCA components)')
        ax.set_ylabel('Overlap (cos²)')
        ax.set_title(f'{emotion_label.capitalize()} Subspace Overlap')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"K-sweep plot saved to: {output_path}")


def plot_k_sweep_heatmap(
    results: Dict,
    output_path: Path,
    layer_idx: int,
    emotions: Optional[List[str]] = None
):
    """
    k-sweep結果のヒートマップ（層×k値）
    
    Args:
        results: k-sweep実験の結果
        output_path: 出力ファイルパス
        layer_idx: 対象層インデックス
        emotions: 対象感情のリスト（Noneの場合は全感情）
    """
    if emotions is None:
        emotions = results['emotions']
    
    k_values = results['k_values']
    
    # 各感情のk値ごとのoverlapを集計
    overlap_matrix = []
    
    for emotion_label in emotions:
        row = []
        for k in k_values:
            if layer_idx in results['sweep_results']:
                if emotion_label in results['sweep_results'][layer_idx]:
                    if k in results['sweep_results'][layer_idx][emotion_label]:
                        overlap = results['sweep_results'][layer_idx][emotion_label][k]['overlap_cos_squared']
                        row.append(overlap)
                    else:
                        row.append(0.0)
                else:
                    row.append(0.0)
            else:
                row.append(0.0)
        overlap_matrix.append(row)
    
    overlap_matrix = np.array(overlap_matrix)
    
    # ヒートマップを描画
    fig, ax = plt.subplots(figsize=(max(6, len(k_values) * 0.8), max(4, len(emotions) * 0.8)))
    
    im = ax.imshow(overlap_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    
    # カラーバー
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Overlap (cos²)', rotation=270, labelpad=20)
    
    # 軸ラベル
    ax.set_xticks(np.arange(len(k_values)))
    ax.set_yticks(np.arange(len(emotions)))
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_yticklabels([e.capitalize() for e in emotions])
    
    # 値を表示
    for i in range(len(emotions)):
        for j in range(len(k_values)):
            text = ax.text(j, i, f'{overlap_matrix[i, j]:.3f}',
                          ha="center", va="center", color="white" if overlap_matrix[i, j] < 0.5 else "black", fontsize=8)
    
    ax.set_xlabel('k (PCA components)')
    ax.set_ylabel('Emotion')
    ax.set_title(f'Subspace Overlap Heatmap (Layer {layer_idx})')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"K-sweep heatmap saved to: {output_path}")


def plot_k_sweep_by_layer(
    results: Dict,
    output_path: Path,
    emotion_label: str,
    layers: Optional[List[int]] = None
):
    """
    特定感情のk-sweep結果を層ごとに可視化
    
    Args:
        results: k-sweep実験の結果
        output_path: 出力ファイルパス
        emotion_label: 対象感情ラベル
        layers: 対象層のリスト（Noneの場合は全層）
    """
    if layers is None:
        layers = results['layers']
    
    k_values = results['k_values']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for layer_idx in layers:
        if layer_idx in results['sweep_results']:
            if emotion_label in results['sweep_results'][layer_idx]:
                overlaps = []
                for k in k_values:
                    if k in results['sweep_results'][layer_idx][emotion_label]:
                        overlap = results['sweep_results'][layer_idx][emotion_label][k]['overlap_cos_squared']
                        overlaps.append(overlap)
                    else:
                        overlaps.append(0.0)
                
                ax.plot(k_values, overlaps, marker='o', label=f'Layer {layer_idx}', linewidth=2, markersize=6)
    
    ax.set_xlabel('k (PCA components)')
    ax.set_ylabel('Overlap (cos²)')
    ax.set_title(f'{emotion_label.capitalize()} Subspace Overlap by Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"K-sweep by layer plot saved to: {output_path}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize k-sweep results")
    parser.add_argument("--results_file", type=str, required=True, help="K-sweep results file (JSON or pickle)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--layers", type=int, nargs='+', default=None, help="Layer indices (default: all)")
    parser.add_argument("--emotions", type=str, nargs='+', default=None, help="Emotion labels (default: all)")
    
    args = parser.parse_args()
    
    # 結果を読み込み
    results_path = Path(args.results_file)
    if results_path.suffix == '.json':
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        import pickle
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 各種プロットを生成
    plot_k_sweep_results(
        results,
        output_dir / "k_sweep_overlap.png",
        layers=args.layers,
        emotions=args.emotions
    )
    
    # 各層のヒートマップ（最初の3層のみ）
    if args.layers:
        layers_to_plot = args.layers[:3]
    else:
        layers_to_plot = results['layers'][:3]
    
    for layer_idx in layers_to_plot:
        plot_k_sweep_heatmap(
            results,
            output_dir / f"k_sweep_heatmap_layer_{layer_idx}.png",
            layer_idx,
            emotions=args.emotions
        )
    
    # 各感情の層ごとのプロット
    if args.emotions:
        emotions_to_plot = args.emotions
    else:
        emotions_to_plot = results['emotions']
    
    for emotion_label in emotions_to_plot:
        plot_k_sweep_by_layer(
            results,
            output_dir / f"k_sweep_{emotion_label}_by_layer.png",
            emotion_label,
            layers=args.layers
        )


if __name__ == "__main__":
    main()

