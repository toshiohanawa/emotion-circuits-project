"""
Head解析結果の可視化
head_screening・head_ablation・head_patchingの結果を可視化
"""
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


def plot_head_reaction_heatmap(
    head_scores_file: Path,
    output_path: Path,
    emotion: Optional[str] = None
):
    """
    Head反応度heatmapを描画
    
    Args:
        head_scores_file: head_screening結果のJSONファイル
        output_path: 出力ファイルパス
        emotion: 特定の感情のみを表示（Noneの場合は全感情）
    """
    with open(head_scores_file, 'r') as f:
        data = json.load(f)
    
    n_layers = data['layers']
    n_heads = data['heads_per_layer']
    
    # フィルタリング
    scores = data['scores']
    if emotion:
        scores = [s for s in scores if s.get('emotion') == emotion]
    
    # Heatmapデータを構築
    heatmap_data = np.zeros((n_layers, n_heads))
    
    for score in scores:
        layer_idx = score['layer']
        head_idx = score['head']
        delta_attn = score['delta_attn']
        heatmap_data[layer_idx, head_idx] = delta_attn
    
    # ヒートマップを描画
    fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.5), max(6, n_layers * 0.5)))
    
    im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=-np.max(np.abs(heatmap_data)), vmax=np.max(np.abs(heatmap_data)))
    
    # カラーバー
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Δ Attention (emotion vs neutral)', rotation=270, labelpad=20)
    
    # 軸ラベル
    ax.set_xticks(np.arange(n_heads))
    ax.set_yticks(np.arange(n_layers))
    ax.set_xticklabels([f"H{i}" for i in range(n_heads)])
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
    
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Layer Index')
    
    emotion_label = emotion.capitalize() if emotion else "All Emotions"
    ax.set_title(f'Head Reaction Heatmap ({emotion_label})')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Head reaction heatmap saved to: {output_path}")


def plot_ablation_sentiment_comparison(
    ablation_file: Path,
    output_path: Path
):
    """
    Ablationによるsentiment変化を可視化
    
    Args:
        ablation_file: head_ablation結果のpklファイル
        output_path: 出力ファイルパス
    """
    with open(ablation_file, 'rb') as f:
        data = pickle.load(f)
    
    baseline_sentiment = data['baseline_metrics']['sentiment']['mean']
    ablation_sentiment = data['ablation_metrics']['sentiment']['mean']
    
    baseline_std = data['baseline_metrics']['sentiment']['std']
    ablation_std = data['ablation_metrics']['sentiment']['std']
    
    # 棒グラフを描画
    fig, ax = plt.subplots(figsize=(6, 5))
    
    x = np.arange(2)
    means = [baseline_sentiment, ablation_sentiment]
    stds = [baseline_std, ablation_std]
    labels = ['Baseline', 'Ablation']
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=['skyblue', 'lightcoral'])
    
    ax.set_ylabel('Sentiment Score (POSITIVE)')
    ax.set_title('Sentiment Comparison: Baseline vs Ablation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 値を表示
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Ablation sentiment comparison saved to: {output_path}")


def plot_patching_keyword_comparison(
    patching_file: Path,
    output_path: Path
):
    """
    Patchingによる感情キーワード増減を可視化
    
    Args:
        patching_file: head_patching結果のpklファイル
        output_path: 出力ファイルパス
    """
    with open(patching_file, 'rb') as f:
        data = pickle.load(f)
    
    baseline_keywords = data['baseline_metrics']['keyword_counts']
    patched_keywords = data['patched_metrics']['keyword_counts']
    
    emotions = ['gratitude', 'anger', 'apology']
    baseline_totals = [baseline_keywords[emotion]['total'] for emotion in emotions]
    patched_totals = [patched_keywords[emotion]['total'] for emotion in emotions]
    
    # 棒グラフを描画
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(emotions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_totals, width, label='Baseline', alpha=0.7, color='skyblue')
    bars2 = ax.bar(x + width/2, patched_totals, width, label='Patched', alpha=0.7, color='lightcoral')
    
    ax.set_ylabel('Keyword Count')
    ax.set_title('Emotion Keyword Comparison: Baseline vs Patched')
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in emotions])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 値を表示
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Patching keyword comparison saved to: {output_path}")


def plot_top_heads(
    head_scores_file: Path,
    output_path: Path,
    top_n: int = 20,
    emotion: Optional[str] = None
):
    """
    上位headを可視化
    
    Args:
        head_scores_file: head_screening結果のJSONファイル
        output_path: 出力ファイルパス
        top_n: 表示する上位head数
        emotion: 特定の感情のみを表示（Noneの場合は全感情）
    """
    with open(head_scores_file, 'r') as f:
        data = json.load(f)
    
    scores = data['scores']
    if emotion:
        scores = [s for s in scores if s.get('emotion') == emotion]
    
    # 絶対値でソート
    sorted_scores = sorted(scores, key=lambda x: abs(x['delta_attn']), reverse=True)[:top_n]
    
    # データを準備
    labels = [f"L{s['layer']}H{s['head']}" for s in sorted_scores]
    delta_attns = [s['delta_attn'] for s in sorted_scores]
    colors = ['red' if d > 0 else 'blue' for d in delta_attns]
    
    # 棒グラフを描画
    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.5), 6))
    
    bars = ax.barh(range(len(labels)), delta_attns, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Δ Attention (emotion vs neutral)')
    ax.set_title(f'Top {top_n} Heads by |Δ Attention|')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 値を表示
    for i, (bar, delta) in enumerate(zip(bars, delta_attns)):
        ax.text(delta + (0.01 if delta > 0 else -0.01), i, f'{delta:.4f}',
               ha='left' if delta > 0 else 'right', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Top heads plot saved to: {output_path}")


def main():
    """メイン関数"""
    import argparse
    from src.config.project_profiles import list_profiles
    from src.utils.project_context import ProjectContext, profile_help_text
    
    parser = argparse.ArgumentParser(description="Visualize head analysis results")
    parser.add_argument("--profile", type=str, choices=list_profiles(), default="baseline",
                        help=f"Dataset profile ({profile_help_text()})")
    parser.add_argument("--head-scores", type=str, default=None, help="Head scores file (JSON, overrides profile default)")
    parser.add_argument("--ablation-file", type=str, default=None, help="Head ablation results file (pkl)")
    parser.add_argument("--patching-file", type=str, default=None, help="Head patching results file (pkl)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (overrides profile default)")
    parser.add_argument("--emotion", type=str, default=None, choices=['gratitude', 'anger', 'apology'], help="Filter by emotion")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top heads to display")
    
    args = parser.parse_args()
    
    # ProjectContextを使用してパスを解決
    context = ProjectContext(profile_name=args.profile)
    results_dir = context.results_dir()
    
    # 出力ディレクトリを解決
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / "plots" / "heads"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Head scoresファイルのデフォルトパスを設定
    if not args.head_scores:
        # デフォルト: results/{profile}/alignment/head_scores_{model}.json
        # モデル名は推測できないので、ユーザーに指定してもらう必要がある
        pass
    
    # Head scoresの可視化
    if args.head_scores:
        head_scores_file = Path(args.head_scores)
        if head_scores_file.exists():
            # Heatmap
            emotion_suffix = f"_{args.emotion}" if args.emotion else ""
            plot_head_reaction_heatmap(
                head_scores_file,
                output_dir / f"head_reaction_heatmap{emotion_suffix}.png",
                emotion=args.emotion
            )
            
            # Top heads
            plot_top_heads(
                head_scores_file,
                output_dir / f"top_heads{emotion_suffix}.png",
                top_n=args.top_n,
                emotion=args.emotion
            )
        else:
            print(f"Warning: Head scores file not found: {head_scores_file}")
    
    # Ablation結果の可視化
    if args.ablation_file:
        ablation_file = Path(args.ablation_file)
        if ablation_file.exists():
            plot_ablation_sentiment_comparison(
                ablation_file,
                output_dir / "ablation_sentiment_comparison.png"
            )
        else:
            print(f"Warning: Ablation file not found: {ablation_file}")
    
    # Patching結果の可視化
    if args.patching_file:
        patching_file = Path(args.patching_file)
        if patching_file.exists():
            plot_patching_keyword_comparison(
                patching_file,
                output_dir / "patching_keyword_comparison.png"
            )
        else:
            print(f"Warning: Patching file not found: {patching_file}")
    
    if not any([args.head_scores, args.ablation_file, args.patching_file]):
        print("Warning: No input files specified. Please provide at least one of --head-scores, --ablation-file, or --patching-file")


if __name__ == "__main__":
    main()

