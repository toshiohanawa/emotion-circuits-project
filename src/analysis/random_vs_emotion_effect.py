"""
Random vs Emotion Patching Effect Comparison
ランダム対照ベクトルと感情方向ベクトルのパッチング効果を比較し、
統計量と可視化（効果量ヒートマップ / バイオリンプロット）を生成する。
"""
from __future__ import annotations

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict


def load_random_control_results(results_file: Path) -> Dict:
    """ランダム対照実験の結果を読み込む"""
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    return results


def _flatten_metric_dict(metric: Dict, prefix: str = "") -> List[Tuple[str, float]]:
    """ネストしたメトリクス辞書をフラット化し、(名前, 値)のリストを返す。"""
    items: List[Tuple[str, float]] = []
    for key, value in metric.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(_flatten_metric_dict(value, name))
        else:
            try:
                items.append((name, float(value)))
            except (TypeError, ValueError):
                continue
    return items


def _collect_metric_distributions(metrics_list: List[Dict]) -> Dict[str, List[float]]:
    """各メトリクス名に対し、全サンプルの値リストを収集。"""
    buckets: Dict[str, List[float]] = defaultdict(list)
    for metric in metrics_list:
        for name, value in _flatten_metric_dict(metric):
            buckets[name].append(value)
    return buckets


def _bootstrap_ci(data: np.ndarray, func, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    """単純ブートストラップによる信頼区間。"""
    boot_stats = []
    n = len(data)
    for _ in range(n_boot):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(func(sample))
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def _permutation_pvalue(random_vals: List[float], emotion_vals: List[float], n_perm: int = 1000) -> float:
    """平均差についてのパーミュテーション検定p値。"""
    combined = np.array(random_vals + emotion_vals)
    n_random = len(random_vals)
    observed = abs(np.mean(emotion_vals) - np.mean(random_vals))
    count = 0
    for _ in range(n_perm):
        np.random.shuffle(combined)
        diff = abs(np.mean(combined[:n_random]) - np.mean(combined[n_random:]))
        if diff >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def _compute_stats(random_vals: List[float], emotion_vals: List[float], n_perm: int = 1000, n_boot: int = 1000) -> Dict[str, float]:
    """ランダム/感情の分布から統計量を計算。"""
    random_arr = np.array(random_vals, dtype=float)
    emotion_arr = np.array(emotion_vals, dtype=float)
    random_mean = float(np.mean(random_arr))
    emotion_mean = float(np.mean(emotion_arr))
    random_std = float(np.std(random_arr))
    emotion_std = float(np.std(emotion_arr))
    t_stat, p_value = stats.ttest_ind(random_arr, emotion_arr, equal_var=False)
    pooled_std = np.sqrt((random_std**2 + emotion_std**2) / 2) if (random_std + emotion_std) > 0 else 0.0
    cohens_d = (emotion_mean - random_mean) / pooled_std if pooled_std > 0 else 0.0
    random_ci = 1.96 * random_std / np.sqrt(len(random_arr)) if len(random_arr) > 0 else 0.0
    emotion_ci = 1.96 * emotion_std / np.sqrt(len(emotion_arr)) if len(emotion_arr) > 0 else 0.0
    # ブートストラップCI（平均差と効果量）
    mean_diff_boot = _bootstrap_ci(emotion_arr - random_arr.mean(), np.mean, n_boot=n_boot)
    cohens_d_boot = _bootstrap_ci((emotion_arr - random_mean) / pooled_std if pooled_std > 0 else emotion_arr * 0, np.mean, n_boot=n_boot) if pooled_std > 0 else (0.0, 0.0)
    p_perm = _permutation_pvalue(random_vals, emotion_vals, n_perm=n_perm)

    return {
        'random_mean': random_mean,
        'random_std': random_std,
        'random_ci_lower': random_mean - random_ci,
        'random_ci_upper': random_mean + random_ci,
        'emotion_mean': emotion_mean,
        'emotion_std': emotion_std,
        'emotion_ci_lower': emotion_mean - emotion_ci,
        'emotion_ci_upper': emotion_mean + emotion_ci,
        'effect_size': cohens_d,
        'p_value': float(p_value),
        't_statistic': float(t_stat),
        'p_perm': float(p_perm),
        'mean_diff_boot_lower': mean_diff_boot[0],
        'mean_diff_boot_upper': mean_diff_boot[1],
        'effect_size_boot_lower': cohens_d_boot[0] if isinstance(cohens_d_boot, tuple) else 0.0,
        'effect_size_boot_upper': cohens_d_boot[1] if isinstance(cohens_d_boot, tuple) else 0.0,
    }


def aggregate_random_metrics(results: Dict, n_perm: int = 1000, n_boot: int = 1000) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    ランダムベクトルのメトリクスを集計し、効果量データフレームと分布データフレームを返す。
    """
    aggregated: Dict = {}
    effect_rows: List[Dict] = []
    dist_rows: List[Dict] = []
    
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
                            if 'metrics' in item and item['metrics']:
                                random_metrics_list.append(item['metrics'])
            
            # 感情方向ベクトルのメトリクスを集計
            emotion_metrics_list = []
            for alpha in results['alpha_values']:
                if alpha in results['emotion_results'][emotion_label][layer_idx]:
                    for item in results['emotion_results'][emotion_label][layer_idx][alpha]:
                        if 'metrics' in item and item['metrics']:
                            emotion_metrics_list.append(item['metrics'])
            
            if not random_metrics_list or not emotion_metrics_list:
                continue

            random_bucket = _collect_metric_distributions(random_metrics_list)
            emotion_bucket = _collect_metric_distributions(emotion_metrics_list)
            common_metrics = sorted(set(random_bucket.keys()) & set(emotion_bucket.keys()))

            for metric_name in common_metrics:
                r_vals = random_bucket[metric_name]
                e_vals = emotion_bucket[metric_name]
                if not r_vals or not e_vals:
                    continue
                stats_dict = _compute_stats(r_vals, e_vals, n_perm=n_perm, n_boot=n_boot)
                aggregated[emotion_label][layer_idx][metric_name] = stats_dict
                effect_rows.append({
                    "emotion": emotion_label,
                    "layer": layer_idx,
                    "metric": metric_name,
                    **stats_dict,
                })
                # 分布データ（バイオリン用）
                for val in r_vals:
                    dist_rows.append({
                        "emotion": emotion_label,
                        "layer": layer_idx,
                        "metric": metric_name,
                        "group": "random",
                        "value": val,
                    })
                for val in e_vals:
                    dist_rows.append({
                        "emotion": emotion_label,
                        "layer": layer_idx,
                        "metric": metric_name,
                        "group": "emotion",
                        "value": val,
                    })
    
    effect_df = pd.DataFrame(effect_rows)
    dist_df = pd.DataFrame(dist_rows)
    return aggregated, effect_df, dist_df


def create_heatmaps(effect_df: pd.DataFrame, output_dir: Path):
    """
    効果量（Cohen's d）のヒートマップを作成（emotionごと）。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if effect_df.empty:
        print("No data for heatmaps.")
        return
    for emotion in effect_df['emotion'].unique():
        df = effect_df[effect_df['emotion'] == emotion]
        pivot = df.pivot(index="layer", columns="metric", values="effect_size")
        fig, ax = plt.subplots(figsize=(max(6, 0.8 * pivot.shape[1]), 6))
        cax = ax.imshow(pivot.values, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"Cohen's d (emotion vs random) — {emotion}")
        fig.colorbar(cax, ax=ax, label="Cohen's d")
        plt.tight_layout()
        out_file = output_dir / f"heatmap_effect_size_{emotion}.png"
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved heatmap: {out_file}")


def create_violin_plots(dist_df: pd.DataFrame, output_dir: Path):
    """
    各emotion×metricについて、random vs emotionの分布をバイオリンプロットで描画。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dist_df.empty:
        print("No data for violin plots.")
        return
    for (emotion, metric) in dist_df.groupby(['emotion', 'metric']).groups.keys():
        subset = dist_df[(dist_df['emotion'] == emotion) & (dist_df['metric'] == metric)]
        if subset.empty:
            continue
        data_random = subset[subset['group'] == 'random']['value']
        data_emotion = subset[subset['group'] == 'emotion']['value']
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.violinplot(
            [data_random, data_emotion],
            showmeans=True,
            showextrema=True,
            showmedians=True,
        )
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Random', 'Emotion'])
        ax.set_ylabel(metric)
        ax.set_title(f"{emotion} — {metric} (random vs emotion)")
        ax.grid(True, alpha=0.3)
        out_file = output_dir / f"violin_{emotion}_{metric.replace('/', '_')}.png"
        plt.tight_layout()
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved violin: {out_file}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare random vs emotion patching effects")
    parser.add_argument("--results_file", type=str, required=True, help="Random control results file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for plots")
    parser.add_argument("--output_csv", type=str, default=None, help="Save summary CSV (effect sizes)")
    parser.add_argument("--distributions_csv", type=str, default=None, help="Save distributions CSV (for plotting/debug)")
    parser.add_argument("--permutations", type=int, default=1000, help="Number of permutations for p_perm")
    parser.add_argument("--bootstraps", type=int, default=1000, help="Number of bootstrap samples for CI")
    
    args = parser.parse_args()
    
    # 結果を読み込み
    results = load_random_control_results(Path(args.results_file))
    
    # メトリクスを集計
    aggregated, effect_df, dist_df = aggregate_random_metrics(results, n_perm=args.permutations, n_boot=args.bootstraps)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV出力
    if args.output_csv:
        effect_df.to_csv(args.output_csv, index=False)
        print(f"Saved effect size CSV: {args.output_csv}")
    if args.distributions_csv:
        dist_df.to_csv(args.distributions_csv, index=False)
        print(f"Saved distributions CSV: {args.distributions_csv}")

    # プロットを作成
    create_heatmaps(effect_df, output_dir)
    create_violin_plots(dist_df, output_dir)
    
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
