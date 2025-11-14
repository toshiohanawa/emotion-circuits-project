"""
アライメント結果の可視化
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import pickle


def plot_alignment_improvement(
    results: Dict,
    output_path: Path,
    layers: Optional[List[int]] = None,
    emotions: Optional[List[str]] = None
):
    """
    アライメント前後のoverlap改善を可視化
    
    Args:
        results: アライメント実験の結果
        output_path: 出力ファイルパス
        layers: 対象層のリスト（Noneの場合は全層）
        emotions: 対象感情のリスト（Noneの場合は全感情）
    """
    if layers is None:
        layers = results['layers']
    if emotions is None:
        emotions = results['emotions']
    
    n_emotions = len(emotions)
    fig, axes = plt.subplots(1, n_emotions, figsize=(5 * n_emotions, 5))
    
    if n_emotions == 1:
        axes = [axes]
    
    for emotion_idx, emotion_label in enumerate(emotions):
        ax = axes[emotion_idx]
        
        before_overlaps = []
        after_overlaps = []
        layer_labels = []
        
        for layer_idx in layers:
            if layer_idx in results['alignment_results']:
                if emotion_label in results['alignment_results'][layer_idx]:
                    result = results['alignment_results'][layer_idx][emotion_label]
                    before_overlaps.append(result['before']['overlap_cos_squared'])
                    after_overlaps.append(result['after']['overlap_cos_squared'])
                    layer_labels.append(f"L{layer_idx}")
        
        x = np.arange(len(layer_labels))
        width = 0.35
        
        ax.bar(x - width/2, before_overlaps, width, label='Before alignment', alpha=0.7)
        ax.bar(x + width/2, after_overlaps, width, label='After alignment', alpha=0.7)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Overlap (cos²)')
        ax.set_title(f'{emotion_label.capitalize()} Subspace Alignment')
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Alignment improvement plot saved to: {output_path}")


def plot_improvement_heatmap(
    results: Dict,
    output_path: Path,
    layers: Optional[List[int]] = None,
    emotions: Optional[List[str]] = None
):
    """
    アライメント改善のヒートマップを描画
    
    Args:
        results: アライメント実験の結果
        output_path: 出力ファイルパス
        layers: 対象層のリスト（Noneの場合は全層）
        emotions: 対象感情のリスト（Noneの場合は全感情）
    """
    if layers is None:
        layers = results['layers']
    if emotions is None:
        emotions = results['emotions']
    
    # 改善値を集計
    improvement_matrix = []
    
    for emotion_label in emotions:
        row = []
        for layer_idx in layers:
            if layer_idx in results['alignment_results']:
                if emotion_label in results['alignment_results'][layer_idx]:
                    improvement = results['alignment_results'][layer_idx][emotion_label]['improvement']['overlap_cos_squared']
                    row.append(improvement)
                else:
                    row.append(0.0)
            else:
                row.append(0.0)
        improvement_matrix.append(row)
    
    improvement_matrix = np.array(improvement_matrix)
    
    # ヒートマップを描画
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.8), max(4, len(emotions) * 0.8)))
    
    im = ax.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)
    
    # カラーバー
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Overlap Improvement (cos²)', rotation=270, labelpad=20)
    
    # 軸ラベル
    ax.set_xticks(np.arange(len(layers)))
    ax.set_yticks(np.arange(len(emotions)))
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_yticklabels([e.capitalize() for e in emotions])
    
    # 値を表示
    for i in range(len(emotions)):
        for j in range(len(layers)):
            text = ax.text(j, i, f'{improvement_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Emotion')
    ax.set_title('Alignment Improvement Heatmap')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Improvement heatmap saved to: {output_path}")


def plot_layer_alignment_comparison(
    results: Dict,
    output_path: Path,
    layer_idx: int,
    emotions: Optional[List[str]] = None
):
    """
    特定層でのアライメント前後の比較を可視化
    
    Args:
        results: アライメント実験の結果
        output_path: 出力ファイルパス
        layer_idx: 対象層インデックス
        emotions: 対象感情のリスト（Noneの場合は全感情）
    """
    if emotions is None:
        emotions = results['emotions']
    
    if layer_idx not in results['alignment_results']:
        print(f"Warning: Layer {layer_idx} not found in results")
        return
    
    fig, axes = plt.subplots(1, len(emotions), figsize=(5 * len(emotions), 5))
    
    if len(emotions) == 1:
        axes = [axes]
    
    for emotion_idx, emotion_label in enumerate(emotions):
        ax = axes[emotion_idx]
        
        if emotion_label in results['alignment_results'][layer_idx]:
            result = results['alignment_results'][layer_idx][emotion_label]
            
            before = result['before']
            after = result['after']
            
            metrics = ['overlap_cos_squared', 'overlap_principal_angles']
            x = np.arange(len(metrics))
            width = 0.35
            
            before_values = [before[m] for m in metrics]
            after_values = [after[m] for m in metrics]
            
            ax.bar(x - width/2, before_values, width, label='Before', alpha=0.7)
            ax.bar(x + width/2, after_values, width, label='After', alpha=0.7)
            
            ax.set_xlabel('Metric')
            ax.set_ylabel('Value')
            ax.set_title(f'Layer {layer_idx}: {emotion_label.capitalize()}')
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Layer alignment comparison plot saved to: {output_path}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize alignment results")
    parser.add_argument("--results_file", type=str, required=True, help="Alignment results file (pickle)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--layers", type=int, nargs='+', default=None, help="Layer indices (default: all)")
    parser.add_argument("--emotions", type=str, nargs='+', default=None, help="Emotion labels (default: all)")
    
    args = parser.parse_args()
    
    # 結果を読み込み
    with open(args.results_file, 'rb') as f:
        results = pickle.load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 各種プロットを生成
    plot_alignment_improvement(
        results,
        output_dir / "alignment_improvement.png",
        layers=args.layers,
        emotions=args.emotions
    )
    
    plot_improvement_heatmap(
        results,
        output_dir / "improvement_heatmap.png",
        layers=args.layers,
        emotions=args.emotions
    )
    
    # 各層の比較プロット（最初の3層のみ）
    if args.layers:
        layers_to_plot = args.layers[:3]
    else:
        layers_to_plot = results['layers'][:3]
    
    for layer_idx in layers_to_plot:
        plot_layer_alignment_comparison(
            results,
            output_dir / f"layer_{layer_idx}_comparison.png",
            layer_idx,
            emotions=args.emotions
        )


if __name__ == "__main__":
    main()

