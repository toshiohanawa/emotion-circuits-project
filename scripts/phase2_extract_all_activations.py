"""
Phase 2: 全モデル×全感情カテゴリで内部活性を抽出
MLflowに記録しながら実行。データセットはプロファイルで指定する。
"""
import argparse
import time
import subprocess
from pathlib import Path
import mlflow

from src.config.project_profiles import list_profiles
from src.utils.mlflow_utils import auto_experiment_from_repo, log_params_dict, log_metrics_dict
from src.utils.project_context import ProjectContext, profile_help_text

# モデルリスト
MODELS = [
    "gpt2",
    "EleutherAI/pythia-160m",
    "EleutherAI/gpt-neo-125M"
]

# 感情カテゴリリスト
EMOTIONS = ["gratitude", "anger", "apology", "neutral"]

# モデル名の短縮形（ディレクトリ名用）
MODEL_SHORT_NAMES = {
    "gpt2": "gpt2",
    "EleutherAI/pythia-160m": "pythia-160m",
    "EleutherAI/gpt-neo-125M": "gpt-neo-125m"
}


def extract_activations_for_model_emotion(
    model: str,
    emotion: str,
    dataset_path: Path,
    output_root: Path,
) -> dict:
    """
    特定のモデル×感情カテゴリで活性を抽出
    
    Returns:
        実行結果の辞書（処理時間、ファイルサイズなど）
    """
    model_short = MODEL_SHORT_NAMES[model]
    output_dir = output_root / model_short
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # 活性抽出を実行
    import sys
    python_exec = sys.executable
    cmd = [
        python_exec, "-m", "src.models.extract_activations",
        "--model", model,
        "--dataset", str(dataset_path),
        "--output", str(output_dir),
        "--emotion", emotion
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path.cwd()
    )
    
    elapsed_time = time.time() - start_time
    
    # 出力ファイルのサイズを取得
    output_file = output_dir / f"activations_{emotion}.pkl"
    file_size = output_file.stat().st_size if output_file.exists() else 0
    
    return {
        "success": result.returncode == 0,
        "elapsed_time": elapsed_time,
        "file_size": file_size,
        "stdout": result.stdout,
        "stderr": result.stderr
    }


def main():
    """全モデル×全感情カテゴリで活性抽出を実行"""
    parser = argparse.ArgumentParser(description="Phase 2 activation extraction sweep")
    parser.add_argument("--profile", type=str, choices=list_profiles(), default="baseline",
                        help=f"Dataset profile to use ({profile_help_text()})")
    args = parser.parse_args()
    
    context = ProjectContext(args.profile)
    dataset_path = context.dataset_path()
    output_root = context.results_dir() / "activations"
    
    # MLflow実験を設定
    experiment_name = auto_experiment_from_repo()
    
    total_runs = len(MODELS) * len(EMOTIONS)
    run_count = 0
    
    print(f"Starting Phase 2: Extracting activations for {total_runs} model×emotion combinations")
    print(f"  - Dataset: {dataset_path}")
    print(f"  - Results root: {output_root}")
    print("=" * 80)
    
    for model in MODELS:
        model_short = MODEL_SHORT_NAMES[model]
        print(f"\n{'='*80}")
        print(f"Model: {model} ({model_short})")
        print(f"{'='*80}")
        
        for emotion in EMOTIONS:
            run_count += 1
            run_name = f"phase2_{model_short}_{emotion}"
            
            print(f"\n[{run_count}/{total_runs}] Processing: {model_short} × {emotion}")
            
            with mlflow.start_run(run_name=run_name, nested=True):
                # パラメータを記録
                params = {
                    "phase": "phase2",
                    "model": model,
                    "model_short": model_short,
                    "emotion": emotion,
                    "dataset": str(dataset_path)
                }
                log_params_dict(params)
                
                # 活性抽出を実行
                result = extract_activations_for_model_emotion(
                    model,
                    emotion,
                    dataset_path,
                    output_root,
                )
                
                # メトリクスを記録
                metrics = {
                    "elapsed_time_seconds": result["elapsed_time"],
                    "file_size_bytes": result["file_size"],
                    "file_size_mb": result["file_size"] / (1024 * 1024),
                    "success": 1 if result["success"] else 0
                }
                log_metrics_dict(metrics)
                
                if result["success"]:
                    print(f"  ✓ Success: {result['elapsed_time']:.2f}s, {result['file_size']/(1024*1024):.2f}MB")
                else:
                    print(f"  ✗ Failed: {result['stderr']}")
                    print(f"  Error output: {result['stderr'][:200]}")
    
    print(f"\n{'='*80}")
    print(f"Phase 2 completed: {run_count}/{total_runs} runs")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
