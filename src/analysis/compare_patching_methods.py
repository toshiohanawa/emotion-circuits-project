"""
文末ベースと感情語トークンベースのActivation Patching効果を比較
"""
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
from src.models.activation_patching_sweep import ActivationPatchingSweep


def load_vectors(vectors_file: Path) -> Dict[str, np.ndarray]:
    """感情方向ベクトルを読み込む"""
    with open(vectors_file, 'rb') as f:
        data = pickle.load(f)
    return data['emotion_vectors']


def compare_methods(
    model_name: str,
    sentence_end_vectors_file: Path,
    token_based_vectors_file: Path,
    prompts_file: Path,
    output_dir: Path,
    layers: List[int] = [3, 5, 7, 9, 11],
    alpha_values: List[float] = [-2, -1, -0.5, 0, 0.5, 1, 2]
):
    """
    文末ベースと感情語トークンベースのpatching効果を比較
    
    Args:
        model_name: モデル名
        sentence_end_vectors_file: 文末ベースのベクトルファイル
        token_based_vectors_file: 感情語トークンベースのベクトルファイル
        prompts_file: プロンプトファイル
        output_dir: 出力ディレクトリ
        layers: 層のリスト
        alpha_values: α値のリスト
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # パッチャーを作成
    patcher = ActivationPatchingSweep(model_name)
    
    # ベクトルを読み込み
    sentence_end_vectors = load_vectors(sentence_end_vectors_file)
    token_based_vectors = load_vectors(token_based_vectors_file)
    
    print(f"Sentence-end vectors: {list(sentence_end_vectors.keys())}")
    print(f"Token-based vectors: {list(token_based_vectors.keys())}")
    
    # プロンプトを読み込み
    with open(prompts_file, 'r') as f:
        prompts_data = json.load(f)
        prompts = prompts_data.get('prompts', [])
    
    print(f"Using {len(prompts)} prompts")
    
    # 各方法でスイープ実験を実行
    print("\n" + "="*80)
    print("Running sweep with SENTENCE-END vectors")
    print("="*80)
    sentence_end_results = patcher.run_sweep(
        prompts,
        sentence_end_vectors,
        layers=layers,
        alpha_values=alpha_values
    )
    sentence_end_aggregated, sentence_end_delta = patcher.aggregate_metrics(sentence_end_results)
    
    print("\n" + "="*80)
    print("Running sweep with TOKEN-BASED vectors")
    print("="*80)
    token_based_results = patcher.run_sweep(
        prompts,
        token_based_vectors,
        layers=layers,
        alpha_values=alpha_values
    )
    token_based_aggregated, token_based_delta = patcher.aggregate_metrics(token_based_results)
    
    # 結果を保存
    comparison_results = {
        'model': model_name,
        'sentence_end': {
            'results': sentence_end_results,
            'aggregated': sentence_end_aggregated,
            'delta': sentence_end_delta,
        },
        'token_based': {
            'results': token_based_results,
            'aggregated': token_based_aggregated,
            'delta': token_based_delta,
        }
    }
    
    results_file = output_dir / f"{model_name.replace('/', '_')}_comparison.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(comparison_results, f)
    
    print(f"\nComparison results saved to: {results_file}")
    
    # 比較プロットを作成
    create_comparison_plots(
        sentence_end_aggregated,
        token_based_aggregated,
        layers,
        alpha_values,
        output_dir,
        model_name
    )
    
    return comparison_results


def create_comparison_plots(
    sentence_end_aggregated: Dict,
    token_based_aggregated: Dict,
    layers: List[int],
    alpha_values: List[float],
    output_dir: Path,
    model_name: str
):
    """
    比較プロットを作成
    
    Args:
        sentence_end_aggregated: 文末ベースの集計結果
        token_based_aggregated: 感情語トークンベースの集計結果
        layers: 層のリスト
        alpha_values: α値のリスト
        output_dir: 出力ディレクトリ
        model_name: モデル名
    """
    emotions = list(sentence_end_aggregated.keys())
    
    for emotion_label in emotions:
        # 各メトリクスで比較
        for metric_name in ['gratitude', 'anger', 'apology', 'politeness', 'sentiment']:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # 文末ベース
            data_se = np.zeros((len(layers), len(alpha_values)))
            for i, layer_idx in enumerate(layers):
                for j, alpha in enumerate(alpha_values):
                    if alpha in sentence_end_aggregated[emotion_label][layer_idx]:
                        metrics = sentence_end_aggregated[emotion_label][layer_idx][alpha]
                        if metric_name in ['gratitude', 'anger', 'apology']:
                            value = metrics['emotion_keywords'][metric_name]
                        elif metric_name == 'politeness':
                            value = metrics['politeness']
                        elif metric_name == 'sentiment':
                            value = metrics['sentiment']
                        else:
                            value = 0.0
                        data_se[i, j] = value
            
            im1 = axes[0].imshow(data_se, cmap='viridis', aspect='auto')
            axes[0].set_xticks(range(len(alpha_values)))
            axes[0].set_xticklabels([f"{α:.1f}" for α in alpha_values])
            axes[0].set_yticks(range(len(layers)))
            axes[0].set_yticklabels([f"Layer {L}" for L in layers])
            axes[0].set_xlabel('Alpha (α)')
            axes[0].set_ylabel('Layer')
            axes[0].set_title(f'Sentence-End Vectors\n{emotion_label.title()} - {metric_name.title()}')
            plt.colorbar(im1, ax=axes[0])
            
            # 感情語トークンベース
            data_tb = np.zeros((len(layers), len(alpha_values)))
            for i, layer_idx in enumerate(layers):
                for j, alpha in enumerate(alpha_values):
                    if alpha in token_based_aggregated[emotion_label][layer_idx]:
                        metrics = token_based_aggregated[emotion_label][layer_idx][alpha]
                        if metric_name in ['gratitude', 'anger', 'apology']:
                            value = metrics['emotion_keywords'][metric_name]
                        elif metric_name == 'politeness':
                            value = metrics['politeness']
                        elif metric_name == 'sentiment':
                            value = metrics['sentiment']
                        else:
                            value = 0.0
                        data_tb[i, j] = value
            
            im2 = axes[1].imshow(data_tb, cmap='viridis', aspect='auto')
            axes[1].set_xticks(range(len(alpha_values)))
            axes[1].set_xticklabels([f"{α:.1f}" for α in alpha_values])
            axes[1].set_yticks(range(len(layers)))
            axes[1].set_yticklabels([f"Layer {L}" for L in layers])
            axes[1].set_xlabel('Alpha (α)')
            axes[1].set_ylabel('Layer')
            axes[1].set_title(f'Token-Based Vectors\n{emotion_label.title()} - {metric_name.title()}')
            plt.colorbar(im2, ax=axes[1])
            
            plt.tight_layout()
            output_path = output_dir / f"comparison_{emotion_label}_{metric_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved comparison plot to: {output_path}")
        
        # 効果の強さを比較（最大値の比較）
        print(f"\n{emotion_label.upper()} - Effect Strength Comparison:")
        for metric_name in ['gratitude', 'anger', 'apology', 'politeness', 'sentiment']:
            max_se = 0.0
            max_tb = 0.0
            
            for layer_idx in layers:
                for alpha in alpha_values:
                    if alpha in sentence_end_aggregated[emotion_label][layer_idx]:
                        metrics = sentence_end_aggregated[emotion_label][layer_idx][alpha]
                        if metric_name in ['gratitude', 'anger', 'apology']:
                            value = metrics['emotion_keywords'][metric_name]
                        elif metric_name == 'politeness':
                            value = metrics['politeness']
                        elif metric_name == 'sentiment':
                            value = metrics['sentiment']
                        max_se = max(max_se, abs(value))
                    
                    if alpha in token_based_aggregated[emotion_label][layer_idx]:
                        metrics = token_based_aggregated[emotion_label][layer_idx][alpha]
                        if metric_name in ['gratitude', 'anger', 'apology']:
                            value = metrics['emotion_keywords'][metric_name]
                        elif metric_name == 'politeness':
                            value = metrics['politeness']
                        elif metric_name == 'sentiment':
                            value = metrics['sentiment']
                        max_tb = max(max_tb, abs(value))
            
            print(f"  {metric_name}: Sentence-End={max_se:.3f}, Token-Based={max_tb:.3f}, Ratio={max_tb/max_se:.2f}x" if max_se > 0 else f"  {metric_name}: Sentence-End={max_se:.3f}, Token-Based={max_tb:.3f}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare sentence-end vs token-based patching")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--sentence_end_vectors", type=str, required=True, help="Sentence-end vectors file")
    parser.add_argument("--token_based_vectors", type=str, required=True, help="Token-based vectors file")
    parser.add_argument("--prompts_file", type=str, default="data/neutral_prompts.json", help="Prompts file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--layers", type=int, nargs='+', default=[3, 5, 7, 9, 11], help="Layer indices")
    parser.add_argument("--alpha", type=float, nargs='+', default=[-2, -1, -0.5, 0, 0.5, 1, 2], help="Alpha values")
    
    args = parser.parse_args()
    
    compare_methods(
        args.model,
        Path(args.sentence_end_vectors),
        Path(args.token_based_vectors),
        Path(args.prompts_file),
        Path(args.output_dir),
        layers=args.layers,
        alpha_values=args.alpha
    )


if __name__ == "__main__":
    main()
