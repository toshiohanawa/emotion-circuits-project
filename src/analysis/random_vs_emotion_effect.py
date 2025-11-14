"""
Random vs Emotion Patching Effect Comparison
ランダム対照ベクトルと感情方向ベクトルのパッチング効果を比較
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
from scipy import stats


def load_random_control_results(results_file: Path) -> Dict:
    """ランダム対照実験の結果を読み込む"""
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    return results


def aggregate_random_metrics(results: Dict) -> Dict:
    """
    ランダムベクトルのメトリクスを集計
    
    Args:
        results: ランダム対照実験の結果
        
    Returns:
        集計されたメトリクス
    """
    aggregated = {}
    
    for emotion_label in results['random_results'].keys():
        aggregated[emotion_label] = {}
        
        for layer_idx in results['layers']:
            aggregated[emotion_label][layer_idx] = {}
            
            # ランダムベクトルのメトリクスを集計
            random_metrics_list = []
            for rand_idx in range(results['num_random']):
                for alpha in results['alpha_values']:
                    if alpha in results['random_results'][emotion_label][layer_idx][rand_idx]:
                        for item in results['random_results'][emotion_label][layer_idx][rand_idx][alpha]:
                            if 'metrics' in item:
                                random_metrics_list.append(item['metrics'])
            
            # 感情方向ベクトルのメトリクスを集計
            emotion_metrics_list = []
            for alpha in results['alpha_values']:
                if alpha in results['emotion_results'][emotion_label][layer_idx]:
                    for item in results['emotion_results'][emotion_label][layer_idx][alpha]:
                        if 'metrics' in item:
                            emotion_metrics_list.append(item['metrics'])
            
            # 統計を計算
            if random_metrics_list and emotion_metrics_list:
                # 各メトリクスタイプで比較
                metric_types = ['emotion_keywords', 'politeness', 'sentiment']
                
                for metric_type in metric_types:
                    if metric_type == 'emotion_keywords':
                        # 感情キーワードは辞書形式
                        for emotion_key in ['gratitude', 'anger', 'apology']:
                            random_values = [m.get(metric_type, {}).get(emotion_key, 0) for m in random_metrics_list]
                            emotion_values = [m.get(metric_type, {}).get(emotion_key, 0) for m in emotion_metrics_list]
                            
                            if random_values and emotion_values:
                                random_mean = np.mean(random_values)
                                emotion_mean = np.mean(emotion_values)
                                random_std = np.std(random_values)
                                emotion_std = np.std(emotion_values)
                                
                                # t-test
                                t_stat, p_value = stats.ttest_ind(random_values, emotion_values)
                                
                                # Cohen's d
                                pooled_std = np.sqrt((random_std**2 + emotion_std**2) / 2)
                                cohens_d = (emotion_mean - random_mean) / pooled_std if pooled_std > 0 else 0
                                
                                # 95% CI
                                random_ci = 1.96 * random_std / np.sqrt(len(random_values))
                                emotion_ci = 1.96 * emotion_std / np.sqrt(len(emotion_values))
                                
                                aggregated[emotion_label][layer_idx][f'{metric_type}_{emotion_key}'] = {
                                    'random_mean': random_mean,
                                    'random_std': random_std,
                                    'random_ci_lower': random_mean - random_ci,
                                    'random_ci_upper': random_mean + random_ci,
                                    'emotion_mean': emotion_mean,
                                    'emotion_std': emotion_std,
                                    'emotion_ci_lower': emotion_mean - emotion_ci,
                                    'emotion_ci_upper': emotion_mean + emotion_ci,
                                    'effect_size': cohens_d,
                                    'p_value': p_value,
                                    't_statistic': t_stat
                                }
                    else:
                        # politeness, sentimentは数値
                        random_values = [m.get(metric_type, 0) for m in random_metrics_list]
                        emotion_values = [m.get(metric_type, 0) for m in emotion_metrics_list]
                        
                        if random_values and emotion_values:
                            random_mean = np.mean(random_values)
                            emotion_mean = np.mean(emotion_values)
                            random_std = np.std(random_values)
                            emotion_std = np.std(emotion_values)
                            
                            # t-test
                            t_stat, p_value = stats.ttest_ind(random_values, emotion_values)
                            
                            # Cohen's d
                            pooled_std = np.sqrt((random_std**2 + emotion_std**2) / 2)
                            cohens_d = (emotion_mean - random_mean) / pooled_std if pooled_std > 0 else 0
                            
                            # 95% CI
                            random_ci = 1.96 * random_std / np.sqrt(len(random_values))
                            emotion_ci = 1.96 * emotion_std / np.sqrt(len(emotion_values))
                            
                            aggregated[emotion_label][layer_idx][metric_type] = {
                                'random_mean': random_mean,
                                'random_std': random_std,
                                'random_ci_lower': random_mean - random_ci,
                                'random_ci_upper': random_mean + random_ci,
                                'emotion_mean': emotion_mean,
                                'emotion_std': emotion_std,
                                'emotion_ci_lower': emotion_mean - emotion_ci,
                                'emotion_ci_upper': emotion_mean + emotion_ci,
                                'effect_size': cohens_d,
                                'p_value': p_value,
                                't_statistic': t_stat
                            }
    
    return aggregated


def create_comparison_plots(aggregated: Dict, output_dir: Path):
    """
    比較プロットを作成
    
    Args:
        aggregated: 集計されたメトリクス
        output_dir: 出力ディレクトリ
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for emotion_label in aggregated.keys():
        for layer_idx in aggregated[emotion_label].keys():
            layer_data = aggregated[emotion_label][layer_idx]
            
            # 各メトリクスでプロット
            for metric_key, metric_data in layer_data.items():
                if isinstance(metric_data, dict) and 'random_mean' in metric_data:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    # バーとエラーバーをプロット
                    x_pos = [0, 1]
                    means = [metric_data['random_mean'], metric_data['emotion_mean']]
                    ci_lower = [metric_data['random_ci_lower'], metric_data['emotion_ci_lower']]
                    ci_upper = [metric_data['random_ci_upper'], metric_data['emotion_ci_upper']]
                    errors = [[m - l for m, l in zip(means, ci_lower)], 
                             [u - m for m, u in zip(means, ci_upper)]]
                    
                    ax.bar(x_pos, means, yerr=errors, capsize=5, alpha=0.7, 
                          color=['red', 'blue'], label=['Random', 'Emotion'])
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(['Random Control', 'Emotion Vector'])
                    ax.set_ylabel(metric_key.replace('_', ' ').title())
                    ax.set_title(f'{emotion_label.title()} - Layer {layer_idx} - {metric_key}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # 統計情報を表示
                    textstr = f"Effect size (Cohen's d): {metric_data['effect_size']:.3f}\n"
                    textstr += f"p-value: {metric_data['p_value']:.4f}\n"
                    textstr += f"t-statistic: {metric_data['t_statistic']:.3f}"
                    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    plt.tight_layout()
                    output_file = output_dir / f'comparison_{emotion_label}_layer{layer_idx}_{metric_key}.png'
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Saved: {output_file}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare random vs emotion patching effects")
    parser.add_argument("--results_file", type=str, required=True, help="Random control results file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for plots")
    
    args = parser.parse_args()
    
    # 結果を読み込み
    results = load_random_control_results(Path(args.results_file))
    
    # メトリクスを集計
    aggregated = aggregate_random_metrics(results)
    
    # プロットを作成
    create_comparison_plots(aggregated, Path(args.output_dir))
    
    # サマリーを表示
    print("\n" + "="*80)
    print("Random vs Emotion Effect Comparison Summary")
    print("="*80)
    
    for emotion_label in aggregated.keys():
        print(f"\n{emotion_label.upper()}:")
        for layer_idx in aggregated[emotion_label].keys():
            print(f"  Layer {layer_idx}:")
            for metric_key, metric_data in aggregated[emotion_label][layer_idx].items():
                if isinstance(metric_data, dict) and 'random_mean' in metric_data:
                    print(f"    {metric_key}:")
                    print(f"      Random: {metric_data['random_mean']:.3f} ± {metric_data['random_std']:.3f}")
                    print(f"      Emotion: {metric_data['emotion_mean']:.3f} ± {metric_data['emotion_std']:.3f}")
                    print(f"      Effect size: {metric_data['effect_size']:.3f}")
                    print(f"      p-value: {metric_data['p_value']:.4f}")


if __name__ == "__main__":
    main()

