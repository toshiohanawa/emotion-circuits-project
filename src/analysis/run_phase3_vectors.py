"""
標準パイプラインにおける Phase 3（感情ベクトル・サブスペース構築）のスクリプト。
baseline_smoke（少数サンプル）と baseline（1感情225件前後を想定）に対応し、
Phase2で保存した活性から感情ベクトルとサブスペースを計算して保存する。
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA

from src.config.project_profiles import EMOTION_LABELS, list_profiles
from src.models.model_registry import get_model_spec
from src.utils.device import get_default_device
from src.utils.project_context import ProjectContext, profile_help_text


def _load_activation_file(path: Path) -> Dict:
    with path.open("rb") as f:
        return pickle.load(f)


def _select_resid(data: Dict) -> np.ndarray:
    if data.get("resid_post") is not None:
        return np.array(data["resid_post"])
    if data.get("resid_pre") is not None:
        return np.array(data["resid_pre"])
    raise ValueError("resid_pre/resid_post のいずれも存在しません。")


def _pca_torch(data: np.ndarray, n_components: int, device: torch.device) -> Dict[str, np.ndarray]:
    """
    torch.svdを使ったPCA計算（GPU/MPS対応）。
    
    Args:
        data: [n_samples, n_features] のデータ
        n_components: PCA次元数
        device: 計算デバイス
    
    Returns:
        components, explained_variance, explained_variance_ratio
    """
    data_tensor = torch.from_numpy(data).to(device, dtype=torch.float32)
    # 平均を引く
    mean = data_tensor.mean(dim=0, keepdim=True)
    centered = data_tensor - mean
    
    # SVD
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    
    # 主成分（最初のn_components個）
    n_comp = min(n_components, Vh.shape[0])
    components = Vh[:n_comp, :].cpu().numpy()  # [n_components, n_features]
    
    # 説明分散
    explained_variance = (S[:n_comp] ** 2 / (data.shape[0] - 1)).cpu().numpy()
    total_var = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_var if total_var > 0 else explained_variance
    
    return {
        "components": components,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
    }


def compute_vectors_and_subspaces(
    resid: np.ndarray,
    labels: Sequence[str],
    layer_indices: Sequence[int],
    n_components: int,
    use_torch: bool = True,
    device: torch.device | None = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, Dict[int, Dict[str, np.ndarray]]]]:
    """
    Args:
        resid: [batch, layer, seq, d_model]
        labels: 各サンプルの感情ラベル
    """
    label_to_rows: Dict[str, List[int]] = {emo: [] for emo in EMOTION_LABELS}
    for idx, lab in enumerate(labels):
        if lab in label_to_rows:
            label_to_rows[lab].append(idx)

    neutral_idx = label_to_rows.get("neutral", [])
    if not neutral_idx:
        raise ValueError("neutral サンプルが必要です。")

    vectors: Dict[str, np.ndarray] = {}
    subspaces: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {"neutral": {}}

    # デバイスを取得（torch使用時）
    if device is None:
        device = get_default_device() if use_torch else torch.device("cpu")

    # neutralサブスペース（アライメントで使用）
    for layer_pos, layer in enumerate(layer_indices):
        neu_slice = resid[neutral_idx, layer_pos]  # [n_samples, seq, d_model]
        neu_vecs = neu_slice.mean(axis=1)  # [n_samples, d_model]
        comp_dim = min(n_components, neu_vecs.shape[0], neu_vecs.shape[1])
        
        if use_torch:
            pca_result = _pca_torch(neu_vecs, comp_dim, device)
            subspaces["neutral"][layer] = pca_result
        else:
            pca = PCA(n_components=comp_dim)
            comps = pca.fit(neu_vecs)
            subspaces["neutral"][layer] = {
                "components": comps.components_,
                "explained_variance": comps.explained_variance_,
                "explained_variance_ratio": comps.explained_variance_ratio_,
            }

    for emo in ("gratitude", "anger", "apology"):
        emo_idx = label_to_rows.get(emo, [])
        pair_len = min(len(emo_idx), len(neutral_idx))
        if pair_len == 0:
            continue
        diffs_per_layer: List[np.ndarray] = []
        subspaces[emo] = {}
        for layer_pos, layer in enumerate(layer_indices):
            emo_slice = resid[emo_idx[:pair_len], layer_pos]
            neu_slice = resid[neutral_idx[:pair_len], layer_pos]
            diff = emo_slice - neu_slice  # [pair_len, seq, d_model]
            # トークン平均（トリガートークンを特定せず簡易平均）
            diff_vecs = diff.mean(axis=1)  # [pair_len, d_model]
            diffs_per_layer.append(diff_vecs)

            comp_dim = min(n_components, diff_vecs.shape[0], diff_vecs.shape[1])
            
            if use_torch:
                pca_result = _pca_torch(diff_vecs, comp_dim, device)
                subspaces[emo][layer] = pca_result
            else:
                pca = PCA(n_components=comp_dim)
                comps = pca.fit(diff_vecs)
                subspaces[emo][layer] = {
                    "components": comps.components_,
                    "explained_variance": comps.explained_variance_,
                    "explained_variance_ratio": comps.explained_variance_ratio_,
                }
        # 層間平均を計算（ベクトル化）
        stacked = np.stack([arr.mean(axis=0) for arr in diffs_per_layer], axis=0)
        vectors[emo] = stacked

    return vectors, subspaces


def main():
    parser = argparse.ArgumentParser(description="Phase3: 感情ベクトルとPCAサブスペースを計算する。")
    parser.add_argument("--profile", type=str, choices=list_profiles(), default="baseline", help=profile_help_text())
    parser.add_argument("--model", type=str, required=True, help="モデル名（activationsファイル名と一致）")
    parser.add_argument("--activations", type=str, default=None, help="activationsファイルパス（未指定ならプロファイル既定パスを検索）")
    parser.add_argument("--n-components", type=int, default=8, help="PCA次元")
    parser.add_argument("--use-torch", action="store_true", default=True, help="torchベースのPCA計算を使用（デフォルト: True）")
    parser.add_argument("--no-use-torch", dest="use_torch", action="store_false", help="sklearnベースのPCA計算を使用（後方互換性）")
    parser.add_argument("--device", type=str, default=None, help="計算デバイス（未指定なら自動選択: mps/cuda/cpu）")
    args = parser.parse_args()

    ctx = ProjectContext(args.profile)
    spec = get_model_spec(args.model)
    act_path = Path(args.activations) if args.activations else ctx.results_dir() / "activations" / f"{spec.name}.pkl"
    if not act_path.exists():
        raise FileNotFoundError(f"活性ファイルが見つかりません: {act_path}")

    start_time = time.time()
    print(f"[Phase 3] 活性ファイルを読み込み中...")
    data = _load_activation_file(act_path)
    resid = _select_resid(data)
    labels = data["labels"]
    layer_indices = data["layer_indices"]
    elapsed = time.time() - start_time
    print(f"[Phase 3] 読み込み完了: {elapsed:.2f}秒")

    # デバイス設定
    if args.device:
        compute_device = torch.device(args.device)
    elif args.use_torch:
        compute_device = get_default_device()
    else:
        compute_device = torch.device("cpu")

    backend_name = "torch" if args.use_torch else "sklearn"
    print(f"[Phase 3] 計算バックエンド: {backend_name}, デバイス: {compute_device}")

    start_time = time.time()
    print(f"[Phase 3] 感情ベクトルとサブスペースを計算中...")
    vectors, subspaces = compute_vectors_and_subspaces(
        resid,
        labels,
        layer_indices,
        n_components=args.n_components,
        use_torch=args.use_torch,
        device=compute_device,
    )
    elapsed = time.time() - start_time
    print(f"[Phase 3] 計算完了: {elapsed:.2f}秒 ({elapsed/60:.2f}分)")

    out_vec_dir = ctx.results_dir() / "emotion_vectors"
    out_sub_dir = ctx.results_dir() / "emotion_subspaces"
    out_vec_dir.mkdir(parents=True, exist_ok=True)
    out_sub_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print(f"[Phase 3] 結果を保存中...")
    with (out_vec_dir / f"{spec.name}_vectors_token_based.pkl").open("wb") as f:
        pickle.dump(
            {
                "emotion_vectors": vectors,
                "metadata": {
                    "model_name": spec.name,
                    "layer_indices": layer_indices,
                    "position_strategy": "mean_over_tokens",
                },
            },
            f,
        )
    with (out_sub_dir / f"{spec.name}_subspaces.pkl").open("wb") as f:
        pickle.dump(
            {
                "subspaces": subspaces,
                "metadata": {
                    "model_name": spec.name,
                    "layer_indices": layer_indices,
                    "n_components": args.n_components,
                },
            },
            f,
        )
    elapsed = time.time() - start_time
    print(f"[Phase 3] 保存完了: {elapsed:.2f}秒")
    print("✓ Phase3 完了: ベクトルとサブスペースを保存しました。")


if __name__ == "__main__":
    main()
