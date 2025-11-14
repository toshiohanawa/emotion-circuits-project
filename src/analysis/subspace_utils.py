"""
サブスペース解析ユーティリティ（共通基盤）
PCA、主角度、overlap計算、アライメント機能を共通化
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union


def load_layer_residuals(
    activations_file: Path,
    layer_idx: int,
    position: str = "last"
) -> np.ndarray:
    """
    活性データファイルから特定層のresidualを読み込む
    
    Args:
        activations_file: 活性データファイルのパス
        layer_idx: 層インデックス
        position: 使用する位置（"last": 文末, "all": 全位置を平均）
        
    Returns:
        Residualデータ [n_samples, d_model]
    """
    with open(activations_file, 'rb') as f:
        activations = pickle.load(f)
    
    data = activations['residual_stream']  # [n_layers, n_samples]
    
    if layer_idx >= len(data):
        raise ValueError(f"Layer {layer_idx} not found in activations")
    
    layer_samples = []
    
    # 各サンプルの活性を集める
    for sample_activations in data[layer_idx]:
        if position == "last":
            # 文末の活性を使用
            layer_samples.append(sample_activations[-1])
        elif position == "all":
            # 全位置の活性を使用（平均）
            layer_samples.append(np.mean(sample_activations, axis=0))
        else:
            layer_samples.append(sample_activations[-1])
    
    return np.array(layer_samples)  # [n_samples, d_model]


def compute_pca_subspace(
    data: np.ndarray,
    n_components: int,
    center: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCAでサブスペースを計算
    
    Args:
        data: データ行列 [n_samples, d_model]
        n_components: 主成分数
        center: 平均を引いて中心化するか
        
    Returns:
        (subspace_vectors, explained_variance_ratio)
        - subspace_vectors: [n_components, d_model] 主成分ベクトル（正規化済み）
        - explained_variance_ratio: [n_components] 説明分散比
    """
    if center:
        data = data - np.mean(data, axis=0)
    
    # SVDでPCAを実行
    n_comp = min(n_components, data.shape[0], data.shape[1])
    U, s, Vt = np.linalg.svd(data, full_matrices=False)
    
    # 主成分ベクトル [n_components, d_model]
    components = Vt[:n_comp]  # 最初のn_comp成分
    
    # 説明分散比を計算
    total_var = np.sum(s ** 2)
    explained_var_ratio = (s[:n_comp] ** 2) / total_var if total_var > 0 else np.zeros(n_comp)
    
    return components, explained_var_ratio


def compute_principal_angles(
    subspace1: np.ndarray,
    subspace2: np.ndarray
) -> np.ndarray:
    """
    2つのサブスペース間のprincipal anglesを計算
    
    Args:
        subspace1: サブスペース1 [n_components1, d_model]（正規化済みの主成分ベクトル）
        subspace2: サブスペース2 [n_components2, d_model]（正規化済みの主成分ベクトル）
        
    Returns:
        Principal angles (radians) [min(n_components1, n_components2)]
    """
    # Gram行列を計算
    G = subspace1 @ subspace2.T  # [n_components1, n_components2]
    
    # SVDでprincipal anglesを計算
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    
    # Principal angles (cosine of angles)
    cosines = np.clip(s, -1.0, 1.0)
    angles = np.arccos(cosines)  # [min(n_components1, n_components2)]
    
    return angles


def compute_subspace_overlap(
    subspace1: np.ndarray,
    subspace2: np.ndarray,
    method: str = "cos_squared"
) -> float:
    """
    サブスペース間のoverlapを計算
    
    Args:
        subspace1: サブスペース1 [n_components1, d_model]
        subspace2: サブスペース2 [n_components2, d_model]
        method: 計算方法（"cos_squared": cos^2の平均, "principal_angles": principal anglesのcosine平均）
        
    Returns:
        Overlap score (0-1)
    """
    if method == "cos_squared":
        # Gram行列を計算
        G = subspace1 @ subspace2.T  # [n_components1, n_components2]
        
        # cos^2の平均を計算
        cos_squared = np.clip(G ** 2, 0.0, 1.0)
        overlap = np.mean(cos_squared)
        
    elif method == "principal_angles":
        # Principal anglesを計算
        angles = compute_principal_angles(subspace1, subspace2)
        cosines = np.cos(angles)
        # 最初の数成分の平均を取る
        n_use = min(5, len(cosines))
        overlap = np.mean(cosines[:n_use])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(overlap)


def procrustes_alignment(
    X: np.ndarray,
    Y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Procrustesアライメント（直交行列での回転アライメント）
    
    XとYを最も近づける直交行列Rを求める: Y ≈ RX
    
    Args:
        X: ソース行列 [n_samples, d] または [n_components, d]
        Y: ターゲット行列 [n_samples, d] または [n_components, d]
        
    Returns:
        (aligned_X, rotation_matrix)
        - aligned_X: アライメント後のX [same shape as X]
        - rotation_matrix: 回転行列R [d, d]
    """
    # 中心化
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    # SVDで最適な回転行列を計算
    H = X_centered.T @ Y_centered  # [d, d]
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    
    # 回転行列
    R = Vt.T @ U.T  # [d, d]
    
    # アライメント後のX
    aligned_X = X_centered @ R + np.mean(Y, axis=0)
    
    return aligned_X, R


def cca_alignment(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Canonical Correlation Analysis (CCA)
    
    XとYの間の最大相関を求める正規化された方向を見つける
    
    Args:
        X: データ行列1 [n_samples, d1]
        Y: データ行列2 [n_samples, d2]
        n_components: 使用する成分数（Noneの場合はmin(d1, d2)）
        
    Returns:
        (X_canonical, Y_canonical, canonical_correlations)
        - X_canonical: Xの正準変量 [n_samples, n_components]
        - Y_canonical: Yの正準変量 [n_samples, n_components]
        - canonical_correlations: 正準相関係数 [n_components]
    """
    n_samples, d1 = X.shape
    _, d2 = Y.shape
    
    if n_components is None:
        n_components = min(d1, d2)
    
    # 中心化
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    # 共分散行列を計算
    Cxx = X_centered.T @ X_centered / (n_samples - 1)
    Cyy = Y_centered.T @ Y_centered / (n_samples - 1)
    Cxy = X_centered.T @ Y_centered / (n_samples - 1)
    
    # CxxとCyyの逆行列を計算（正則化を追加）
    reg = 1e-6
    Cxx_inv = np.linalg.inv(Cxx + reg * np.eye(d1))
    Cyy_inv = np.linalg.inv(Cyy + reg * np.eye(d2))
    
    # CCAの一般化固有値問題を解く
    # Cxx^(-1) Cxy Cyy^(-1) Cyx の固有値分解
    M = Cxx_inv @ Cxy @ Cyy_inv @ Cxy.T
    
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    
    # 降順にソート
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 正準相関係数
    canonical_correlations = np.sqrt(np.clip(eigenvalues, 0.0, 1.0))[:n_components]
    
    # 正準変量を計算
    X_canonical = X_centered @ eigenvectors[:, :n_components]
    Y_canonical = Y_centered @ (Cyy_inv @ Cxy.T @ eigenvectors[:, :n_components])
    
    # 正規化
    X_canonical = X_canonical / np.linalg.norm(X_canonical, axis=0, keepdims=True)
    Y_canonical = Y_canonical / np.linalg.norm(Y_canonical, axis=0, keepdims=True)
    
    return X_canonical, Y_canonical, canonical_correlations


def batch_load_residuals(
    activations_dir: Path,
    emotion_labels: List[str],
    layer_idx: int,
    position: str = "last"
) -> Dict[str, np.ndarray]:
    """
    複数の感情カテゴリのresidualを一括で読み込む
    
    Args:
        activations_dir: 活性データディレクトリ
        emotion_labels: 感情ラベルのリスト
        layer_idx: 層インデックス
        position: 使用する位置
        
    Returns:
        感情ラベルをキーとするresidualデータの辞書
    """
    residuals = {}
    
    for emotion_label in emotion_labels:
        activations_file = activations_dir / f"activations_{emotion_label}.pkl"
        
        if not activations_file.exists():
            print(f"Warning: {activations_file} not found, skipping...")
            continue
        
        residuals[emotion_label] = load_layer_residuals(
            activations_file,
            layer_idx,
            position
        )
    
    return residuals


def compute_subspace_from_residuals(
    residuals: np.ndarray,
    n_components: int,
    center: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Residualデータからサブスペースを計算
    
    Args:
        residuals: Residualデータ [n_samples, d_model]
        n_components: PCAの主成分数
        center: 平均を引いて中心化するか
        
    Returns:
        (subspace_vectors, explained_variance_ratio)
    """
    return compute_pca_subspace(residuals, n_components, center)


def compare_subspaces(
    subspace1: np.ndarray,
    subspace2: np.ndarray,
    compute_angles: bool = True,
    compute_overlap: bool = True
) -> Dict[str, Union[float, np.ndarray]]:
    """
    2つのサブスペースを比較
    
    Args:
        subspace1: サブスペース1 [n_components1, d_model]
        subspace2: サブスペース2 [n_components2, d_model]
        compute_angles: Principal anglesを計算するか
        compute_overlap: Overlapを計算するか
        
    Returns:
        比較結果の辞書
    """
    results = {}
    
    if compute_angles:
        angles = compute_principal_angles(subspace1, subspace2)
        results['principal_angles'] = angles
        results['mean_principal_angle'] = np.mean(angles)
    
    if compute_overlap:
        overlap_cos_sq = compute_subspace_overlap(subspace1, subspace2, method="cos_squared")
        overlap_principal = compute_subspace_overlap(subspace1, subspace2, method="principal_angles")
        results['overlap_cos_squared'] = overlap_cos_sq
        results['overlap_principal_angles'] = overlap_principal
    
    return results


def align_and_compare(
    subspace1: np.ndarray,
    subspace2: np.ndarray,
    alignment_method: str = "procrustes"
) -> Dict[str, Union[float, np.ndarray, np.ndarray]]:
    """
    サブスペースをアライメントしてから比較
    
    Args:
        subspace1: サブスペース1 [n_components1, d_model]
        subspace2: サブスペース2 [n_components2, d_model]
        alignment_method: アライメント方法（"procrustes" または "cca"）
        
    Returns:
        アライメント結果と比較結果の辞書
    """
    # アライメント前の比較
    before_comparison = compare_subspaces(subspace1, subspace2)
    
    # アライメント
    if alignment_method == "procrustes":
        aligned_subspace1, rotation_matrix = procrustes_alignment(subspace1, subspace2)
        alignment_info = {'rotation_matrix': rotation_matrix}
    elif alignment_method == "cca":
        # CCAは異なる次元のデータに対して使用
        # ここでは同じ次元を仮定
        X_canonical, Y_canonical, correlations = cca_alignment(subspace1, subspace2)
        aligned_subspace1 = X_canonical
        alignment_info = {'canonical_correlations': correlations}
    else:
        raise ValueError(f"Unknown alignment method: {alignment_method}")
    
    # アライメント後の比較
    after_comparison = compare_subspaces(aligned_subspace1, subspace2)
    
    return {
        'before': before_comparison,
        'after': after_comparison,
        'aligned_subspace1': aligned_subspace1,
        'alignment_info': alignment_info,
        'improvement': {
            'overlap_cos_squared': after_comparison['overlap_cos_squared'] - before_comparison['overlap_cos_squared'],
            'overlap_principal_angles': after_comparison['overlap_principal_angles'] - before_comparison['overlap_principal_angles']
        }
    }

