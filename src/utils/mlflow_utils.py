"""
MLflowユーティリティ
実験追跡とロギングの共通機能を提供
"""
import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import mlflow
import mlflow.pytorch


def get_repo_name() -> str:
    """
    Gitリポジトリ名を取得
    
    Returns:
        リポジトリ名（例: "emotion-circuits-project"）
    """
    try:
        # GitリモートURLから取得を試みる
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=True
        )
        url = result.stdout.strip()
        # URLからリポジトリ名を抽出
        if url.endswith(".git"):
            url = url[:-4]
        repo_name = url.split("/")[-1]
        return repo_name
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Gitが使えない場合、カレントディレクトリ名を使用
        cwd = Path.cwd()
        return cwd.name


def _default_local_tracking_uri() -> str:
    """
    Return the local file:// tracking URI, creating ``mlruns`` on demand.
    """
    local_dir = Path.cwd() / "mlruns"
    local_dir.mkdir(parents=True, exist_ok=True)
    return f"file://{local_dir}"


def resolve_tracking_uri() -> str:
    """
    Resolve the tracking URI following this precedence:

    1. ``MLFLOW_TRACKING_URI`` environment variable (user override)
    2. ``file://`` store inside the repository (no MLflow server required)
    """
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    return _default_local_tracking_uri()


def auto_experiment_from_repo() -> str:
    """
    GitHubリポジトリ名から自動的に実験名を設定
    
    Returns:
        設定された実験名
    """
    repo_name = get_repo_name()
    experiment_name = repo_name
    
    # MLflow tracking URIを設定
    tracking_uri = resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    
    # 実験を作成または取得
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✓ Created new experiment: {experiment_name} (ID: {experiment_id})")
        else:
            print(f"✓ Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
    except Exception as e:
        print(f"Warning: Could not set experiment on '{tracking_uri}': {e}")
        # 最初にリモートURIを試し、失敗したらローカルfile://にフォールバック
        fallback_uri = _default_local_tracking_uri()
        if fallback_uri != tracking_uri:
            print(f"  Falling back to local MLflow store at {fallback_uri}")
            mlflow.set_tracking_uri(fallback_uri)
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"✓ Created new local experiment: {experiment_name} (ID: {experiment_id})")
            else:
                print(f"✓ Using existing local experiment: {experiment_name} (ID: {experiment.experiment_id})")
        else:
            print("  Using default MLflow experiment")
            experiment_name = "Default"
    
    mlflow.set_experiment(experiment_name)
    return experiment_name


def log_params_dict(params: Dict[str, Any]) -> None:
    """
    パラメータの辞書を一括でログに記録
    
    Args:
        params: パラメータの辞書
    """
    for key, value in params.items():
        # 値がNoneの場合はスキップ
        if value is not None:
            # リストやタプルの場合は文字列に変換
            if isinstance(value, (list, tuple)):
                mlflow.log_param(key, str(value))
            else:
                mlflow.log_param(key, value)


def log_metrics_dict(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    メトリクスの辞書を一括でログに記録
    
    Args:
        metrics: メトリクスの辞書
        step: ステップ番号（オプション）
    """
    for key, value in metrics.items():
        if value is not None and isinstance(value, (int, float)):
            mlflow.log_metric(key, value, step=step)


def log_nested_metrics(metrics: Dict[str, Any], prefix: str = "", step: Optional[int] = None) -> None:
    """
    ネストした辞書をフラット化してMLflowに記録
    """
    def _flatten(d: Dict[str, Any], pre: str) -> Dict[str, float]:
        flat: Dict[str, float] = {}
        for k, v in d.items():
            name = f"{pre}/{k}" if pre else k
            if isinstance(v, dict):
                flat.update(_flatten(v, name))
            elif isinstance(v, (int, float)):
                flat[name] = float(v)
        return flat
    flat = _flatten(metrics, prefix)
    for k, v in flat.items():
        mlflow.log_metric(k, v, step=step)


def log_metrics_prefixed(prefix: str, metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Flatten a metrics dict under a prefix and log to MLflow.

    This is useful when metrics have already been averaged and only need a
    namespacing step (e.g., ``ov_ablation/baseline``).
    """
    for key, value in metrics.items():
        if value is None or not isinstance(value, (int, float)):
            continue
        name = f"{prefix}/{key}" if prefix else key
        mlflow.log_metric(name, float(value), step=step)


def set_run_tags(tags: Dict[str, Any]) -> None:
    """MLflowタグを設定"""
    if not tags:
        return
    mlflow.set_tags({k: v for k, v in tags.items() if v is not None})


def log_artifact_file(file_path: str, artifact_path: Optional[str] = None) -> None:
    """
    ファイルをアーティファクトとしてログに記録
    
    Args:
        file_path: ログに記録するファイルのパス
        artifact_path: MLflow内でのアーティファクトパス（オプション）
    """
    if os.path.exists(file_path):
        mlflow.log_artifact(file_path, artifact_path=artifact_path)
    else:
        print(f"Warning: File not found: {file_path}")


def log_artifact_dir(dir_path: str, artifact_path: Optional[str] = None) -> None:
    """
    ディレクトリをアーティファクトとしてログに記録
    
    Args:
        dir_path: ログに記録するディレクトリのパス
        artifact_path: MLflow内でのアーティファクトパス（オプション）
    """
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        mlflow.log_artifacts(dir_path, artifact_path=artifact_path)
    else:
        print(f"Warning: Directory not found: {dir_path}")
