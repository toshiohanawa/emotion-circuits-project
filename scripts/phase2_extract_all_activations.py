"""
Phase 2: 現行パイプラインの活性抽出をモデルごとに一括実行する補助スクリプト。
`run_phase2_activations` をモデル単位で呼び出し、プロファイル（baseline/baseline_smoke）を指定して実行する。
"""
import argparse
import time
import subprocess
from pathlib import Path
import mlflow

from src.config.project_profiles import list_profiles
from src.utils.mlflow_utils import (
    auto_experiment_from_repo,
    log_params_dict,
    log_metrics_dict,
    log_artifact_file,
)
from src.utils.model_utils import sanitize_model_name
from src.utils.project_context import ProjectContext, profile_help_text
from src.models.model_registry import list_model_names

def run_phase2_for_model(model: str, profile: str, layers: str | None) -> dict:
    """指定モデルで run_phase2_activations を実行し、ログを返す。"""
    import sys

    python_exec = sys.executable
    cmd = [
        python_exec,
        "-m",
        "src.analysis.run_phase2_activations",
        "--profile",
        profile,
        "--model",
        model,
    ]
    if layers:
        cmd += ["--layers"] + layers.split()
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
    elapsed_time = time.time() - start_time
    return {
        "success": result.returncode == 0,
        "elapsed_time": elapsed_time,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    """run_phase2_activations をモデルごとに呼び出すバッチスクリプト"""
    parser = argparse.ArgumentParser(description="Phase2活性抽出をモデルごとに一括実行（現行パイプライン用）")
    parser.add_argument("--profile", type=str, choices=list_profiles(), default="baseline",
                        help=f"使用するプロファイル ({profile_help_text()})")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="実行するモデル（未指定ならmodel_registryの全モデル）")
    parser.add_argument("--layers", type=str, default=None,
                        help="層指定（例: \"0 3 6 9 11\"。未指定ならデフォルト）")
    args = parser.parse_args()
    
    context = ProjectContext(args.profile)
    logs_root = context.results_dir() / "logs" / "phase2"
    logs_root.mkdir(parents=True, exist_ok=True)
    models = args.models or list_model_names()

    auto_experiment_from_repo()

    total_runs = len(models)
    run_count = 0
    print(f"Phase2 batch start: profile={args.profile}, models={models}")
    for model in models:
        run_count += 1
        run_name = f"phase2_{sanitize_model_name(model)}"
        with mlflow.start_run(run_name=run_name, nested=True):
            params = {
                "phase": "phase2",
                "model": model,
                "profile": args.profile,
                "layers": args.layers or "default",
            }
            log_params_dict(params)
            result = run_phase2_for_model(model, args.profile, args.layers)
            log_file = logs_root / f"{sanitize_model_name(model)}.log"
            with open(log_file, "w") as fh:
                fh.write("=== STDOUT ===\n")
                fh.write(result["stdout"])
                fh.write("\n\n=== STDERR ===\n")
                fh.write(result["stderr"])
            log_artifact_file(str(log_file), artifact_path="phase2")
            log_metrics_dict(
                {
                    "elapsed_time_seconds": result["elapsed_time"],
                    "success": 1 if result["success"] else 0,
                }
            )
            if result["success"]:
                print(f"[{run_count}/{total_runs}] {model}: ✓ ({result['elapsed_time']:.2f}s)")
            else:
                print(f"[{run_count}/{total_runs}] {model}: ✗ {result['stderr'][:200]}")

    print(f"Phase2 batch finished: {run_count}/{total_runs}")


if __name__ == "__main__":
    main()
