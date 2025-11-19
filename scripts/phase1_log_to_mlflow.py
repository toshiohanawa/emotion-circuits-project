"""
Phase 1: データセット概要をMLflowに記録する補助スクリプト。
現行の標準パイプライン（baseline≈225/感情、baseline_smokeは少数）に対応。
"""
import argparse
import json
from pathlib import Path
import mlflow

from src.config.project_profiles import list_profiles
from src.data.validate_dataset import validate_dataset
from src.utils.mlflow_utils import auto_experiment_from_repo, log_params_dict, log_metrics_dict, log_artifact_file
from src.utils.project_context import ProjectContext, profile_help_text


def main():
    """データセット情報をMLflowに記録"""
    parser = argparse.ArgumentParser(description="Phase1 データセット統計をMLflowに記録（現行パイプライン用）")
    parser.add_argument("--profile", type=str, choices=list_profiles(), default="baseline",
                        help=f"Dataset profile ({profile_help_text()})")
    args = parser.parse_args()
    
    context = ProjectContext(args.profile)
    dataset_path = context.dataset_path()
    stats_root = context.results_dir() / "evaluation"
    
    # MLflow実験を設定
    experiment_name = auto_experiment_from_repo()
    
    # データセットを検証
    stats, dataset = validate_dataset(dataset_path)
    
    # MLflow runを開始
    with mlflow.start_run(run_name="phase1_data_creation"):
        # パラメータを記録
        params = {
            "phase": "phase1",
            "dataset_file": str(dataset_path),
            "total_samples": stats['total_samples'],
        }
        log_params_dict(params)
        
        # メトリクスを記録
        metrics = {
            "total_samples": stats['total_samples'],
            "avg_text_length": stats['text_length']['average'],
            "avg_word_count": stats['word_count']['average'],
        }
        
        # 各感情カテゴリのサンプル数を記録
        for emotion, count in stats['emotion_distribution'].items():
            metrics[f"samples_{emotion}"] = count
        
        log_metrics_dict(metrics)
        
        # アーティファクトを記録（エラーが発生しても続行）
        try:
            log_artifact_file(str(dataset_path), "dataset")
        except Exception as e:
            print(f"Warning: Could not log dataset artifact: {e}")
        
        # 統計情報をJSONで保存して記録
        stats_file = stats_root / "phase1_stats.json"
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        try:
            log_artifact_file(str(stats_file), "statistics")
        except Exception as e:
            print(f"Warning: Could not log statistics artifact: {e}")
        
        print("✓ Phase 1 data logged to MLflow")


if __name__ == "__main__":
    main()
