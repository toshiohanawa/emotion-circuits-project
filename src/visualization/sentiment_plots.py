"""
Sentiment評価結果の可視化モジュール
"""
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


def load_evaluation_results(results_file: Path) -> List[Dict]:
    """評価結果を読み込む"""
    suffix = results_file.suffix
    
    if suffix == '.json':
        with open(results_file, 'r') as f:
            return json.load(f)
    elif suffix == '.pkl':
        with open(results_file, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def plot_emotion_keyword_frequencies(
    results: List[Dict],
    output_path: Path,
    title: Optional[str] = None
):
    """
    感情キーワード頻度の可視化
    
    Args:
        results: 評価結果のリスト
        output_path: 出力パス
        title: タイトル
    """
    # データを集計
    gratitude_counts = []
    anger_counts = []
    apology_counts = []
    
    for result in results:
        if 'emotion_keywords' in result:
            gratitude_counts.append(result['emotion_keywords'].get('gratitude', 0))
            anger_counts.append(result['emotion_keywords'].get('anger', 0))
            apology_counts.append(result['emotion_keywords'].get('apology', 0))
    
    # プロット
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    emotions = ['Gratitude', 'Anger', 'Apology']
    counts = [gratitude_counts, anger_counts, apology_counts]
    
    for ax, emotion, count_list in zip(axes, emotions, counts):
        ax.hist(count_list, bins=range(max(count_list) + 2), edgecolor='black', alpha=0.7)
        ax.set_xlabel('Keyword Count')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{emotion} Keywords')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved emotion keyword frequencies plot to: {output_path}")


def plot_sentiment_scores(
    results: List[Dict],
    output_path: Path,
    title: Optional[str] = None
):
    """
    Sentimentスコアの可視化
    
    Args:
        results: 評価結果のリスト
        output_path: 出力パス
        title: タイトル
    """
    # データを集計
    positive_scores = []
    negative_scores = []
    
    for result in results:
        if result.get('sentiment') is not None:
            sentiment = result['sentiment']
            if 'POSITIVE' in sentiment:
                positive_scores.append(sentiment['POSITIVE'])
            if 'NEGATIVE' in sentiment:
                negative_scores.append(sentiment['NEGATIVE'])
    
    if not positive_scores:
        print("No sentiment scores found in results")
        return
    
    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(positive_scores, bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[0].set_xlabel('POSITIVE Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('POSITIVE Sentiment Distribution')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(negative_scores, bins=20, edgecolor='black', alpha=0.7, color='red')
    axes[1].set_xlabel('NEGATIVE Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('NEGATIVE Sentiment Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved sentiment scores plot to: {output_path}")


def plot_sentiment_comparison(
    results: List[Dict],
    output_path: Path,
    title: Optional[str] = None
):
    """
    Sentimentスコアの比較プロット（POSITIVE vs NEGATIVE）
    
    Args:
        results: 評価結果のリスト
        output_path: 出力パス
        title: タイトル
    """
    # データを集計
    positive_scores = []
    negative_scores = []
    
    for result in results:
        if result.get('sentiment') is not None:
            sentiment = result['sentiment']
            if 'POSITIVE' in sentiment and 'NEGATIVE' in sentiment:
                positive_scores.append(sentiment['POSITIVE'])
                negative_scores.append(sentiment['NEGATIVE'])
    
    if not positive_scores:
        print("No sentiment scores found in results")
        return
    
    # プロット
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(positive_scores, negative_scores, alpha=0.6)
    ax.set_xlabel('POSITIVE Score')
    ax.set_ylabel('NEGATIVE Score')
    ax.set_title('Sentiment Score Comparison' if title is None else title)
    ax.grid(True, alpha=0.3)
    
    # 対角線を追加
    ax.plot([0, 1], [1, 0], 'r--', alpha=0.5, label='Equal probability')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved sentiment comparison plot to: {output_path}")


def create_all_sentiment_plots(
    results_file: Path,
    output_dir: Path
):
    """
    全てのsentiment可視化を作成
    
    Args:
        results_file: 評価結果ファイル
        output_dir: 出力ディレクトリ
    """
    results = load_evaluation_results(results_file)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 感情キーワード頻度のプロット
    plot_emotion_keyword_frequencies(
        results,
        output_dir / "emotion_keyword_frequencies.png",
        title="Emotion Keyword Frequencies"
    )
    
    # Sentimentスコアのプロット
    plot_sentiment_scores(
        results,
        output_dir / "sentiment_scores.png",
        title="Sentiment Score Distribution"
    )
    
    # Sentiment比較プロット
    plot_sentiment_comparison(
        results,
        output_dir / "sentiment_comparison.png",
        title="Sentiment Score Comparison"
    )


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create sentiment visualization plots")
    parser.add_argument("--results_file", type=str, required=True, help="Evaluation results file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for plots")
    
    args = parser.parse_args()
    
    create_all_sentiment_plots(Path(args.results_file), Path(args.output_dir))
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

