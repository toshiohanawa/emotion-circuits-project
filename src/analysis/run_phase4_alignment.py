"""
標準パイプラインにおける Phase 4（サブスペースアライメント）のスクリプト。

baseline_smoke（少数サンプル）と baseline（1感情225件前後を想定）に対し、
サブスペースのモデル間アライメントを計算して保存する。

GPU/MPS対応: torch ベースの実装により、GPU/MPS で高速化可能。
"""
from __future__ import annotations

import argparse
import pickle
import time
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.utils.device import get_default_device
from src.utils.project_context import ProjectContext, profile_help_text
from src.utils.timing import record_phase_timing


def _load_subspace(path: Path) -> Dict:
    with path.open("rb") as f:
        return pickle.load(f)


def _orthonormalize_torch(mat: torch.Tensor) -> torch.Tensor:
    """
    torch ベースの正規直交化（QR分解）。
    
    Args:
        mat: [k, d] のテンソル（k次元の主成分ベクトル）
    
    Returns:
        [d, k] の正規直交基底
    """
    if mat.device.type == "mps":
        # MPS は torch.linalg.qr 未サポートのため CPU でQRを計算し、結果だけMPSへ戻す
        q_cpu, _ = torch.linalg.qr(mat.T.cpu())
        return q_cpu.to(mat.device, dtype=mat.dtype)  # [d, k]
    q, _ = torch.linalg.qr(mat.T)
    return q  # [d, k]


def _orthonormalize(mat: np.ndarray) -> np.ndarray:
    """
    numpy ベースの正規直交化（後方互換性のため残す）。
    
    注意: torch ベースの実装を使用する場合は _orthonormalize_torch を推奨。
    """
    q, _ = np.linalg.qr(mat.T)
    return q  # [d, k]


def _subspace_overlap_torch(u: torch.Tensor, v: torch.Tensor, k: int, device: torch.device) -> float:
    """
    torch ベースのサブスペース overlap 計算（GPU/MPS対応）。
    
    Args:
        u: [k, d] の主成分ベクトル（最初のk次元を使用）
        v: [k, d] の主成分ベクトル（最初のk次元を使用）
        k: 使用する次元数
        device: 計算デバイス
    
    Returns:
        overlap 値（0-1の範囲）
    """
    u_k = _orthonormalize_torch(u[:k])
    v_k = _orthonormalize_torch(v[:k])
    # trace((u_k^T @ v_k) @ (v_k^T @ u_k)) を計算
    overlap = torch.trace((u_k.T @ v_k) @ (v_k.T @ u_k))
    return float(overlap.cpu().item())


def _subspace_overlap(u: np.ndarray, v: np.ndarray, k: int) -> float:
    """
    numpy ベースのサブスペース overlap 計算（後方互換性のため残す）。
    """
    u_k = _orthonormalize(u[:k])
    v_k = _orthonormalize(v[:k])
    return float(np.trace((u_k.T @ v_k) @ (v_k.T @ u_k)))


def _procrustes_torch(src: torch.Tensor, tgt: torch.Tensor, k: int, device: torch.device) -> torch.Tensor:
    """
    torch ベースの Procrustes 分析（最小二乗で線形写像を学習）。
    
    GPU/MPS対応により、大規模データセットでの計算が高速化される。
    
    Args:
        src: [k_max, d_src] のソース主成分ベクトル（k_maxは最大次元数）
        tgt: [k_max, d_tgt] のターゲット主成分ベクトル
        k: 使用する次元数
        device: 計算デバイス
    
    Returns:
        [k, k] の線形写像行列（subspace coords）
    
    Note:
        異なるd_modelを持つモデル間のアライメントでは、PCA成分をk次元に射影してから
        Procrustes分析を適用する。標準的なProcrustes分析は同じ次元空間を前提とするため、
        異なるd_modelの場合は、PCA成分をk次元に射影（src[:k]とtgt[:k]を使用）してから
        内積行列を計算してアライメントする。
    """
    # MPSはfloat64をサポートしていないため、デバイスに応じてdtypeを選択
    if device.type == "mps":
        dtype = torch.float32
    else:
        dtype = torch.float64
    
    # PCA成分の最初のk次元を使用（形状: [k, d_model]）
    src_k = src[:k].to(device, dtype=dtype)  # [k, d_src]
    tgt_k = tgt[:k].to(device, dtype=dtype)  # [k, d_tgt]
    
    # 異なるd_modelを持つモデル間のアライメントでは、PCA成分をk次元空間に射影
    # 標準的なProcrustes分析は同じ次元空間を前提とするため、異なるd_modelの場合は
    # PCA成分をk次元に射影してから、内積行列を計算してアライメントする
    # ここでは、src_k @ src_k.T と tgt_k @ tgt_k.T のアライメントを計算
    src_cov = src_k @ src_k.T  # [k, k] - ソースの共分散行列
    tgt_cov = tgt_k @ tgt_k.T  # [k, k] - ターゲットの共分散行列
    
    # Procrustes分析: src_covをtgt_covにアライメントする線形写像を求める
    # W @ src_cov @ W.T ≈ tgt_cov となるWを求める
    # これは、src_cov @ W.T ≈ tgt_cov @ W.T と等価
    # より正確には、src_k @ W.T ≈ tgt_k となるWを求める
    # しかし、異なるd_modelの場合、直接的なアライメントは不可能
    # 代わりに、PCA成分の内積行列をアライメントする
    
    # 標準的なProcrustes分析: src_k.T @ W ≈ tgt_k.T となるWを求める
    # しかし、src_k.Tは[d_src, k]、tgt_k.Tは[d_tgt, k]で次元が異なる
    # そのため、PCA成分をk次元に射影してからアライメントする
    # ここでは、src_kとtgt_kのk×kの内積行列を使用
    
    if device.type == "mps":
        # MPSは torch.linalg.lstsq 未サポート。擬似逆を用いた最小二乗解で回避。
        # src_cov @ W ≈ tgt_cov となるWを求める
        w = torch.linalg.pinv(src_cov) @ tgt_cov
    else:
        # torch.linalg.lstsq で最小二乗解を計算
        # src_cov @ W ≈ tgt_cov となるWを求める
        w, _, _, _ = torch.linalg.lstsq(src_cov, tgt_cov, rcond=None)
    return w  # [k, k] in subspace coords


def _procrustes(src: np.ndarray, tgt: np.ndarray, k: int) -> np.ndarray:
    """
    numpy ベースの Procrustes 分析（後方互換性のため残す）。
    
    注意: torch ベースの実装を使用する場合は _procrustes_torch を推奨。
    """
    src_k = src[:k].T  # [d, k]
    tgt_k = tgt[:k].T
    w, _, _, _ = np.linalg.lstsq(src_k, tgt_k, rcond=None)
    return w  # [k, k] in subspace coords


def main():
    parser = argparse.ArgumentParser(description="Phase 4: モデル間アライメントを計算する。")
    parser.add_argument("--profile", type=str, default="baseline", help=profile_help_text())
    parser.add_argument("--model-a", type=str, required=True, help="サブスペースファイルのモデルA")
    parser.add_argument("--model-b", type=str, required=True, help="サブスペースファイルのモデルB")
    parser.add_argument("--k-max", type=int, default=8, help="最大k次元")
    parser.add_argument("--subspace-dir", type=str, default=None, help="サブスペース格納ディレクトリ（未指定なら profile 結果パス）")
    parser.add_argument("--device", type=str, default=None, help="計算デバイス（未指定なら自動選択: mps/cuda/cpu）")
    parser.add_argument(
        "--use-torch",
        action="store_true",
        default=True,
        help="torch ベースの実装を使用（GPU/MPS加速、デフォルト: True）",
    )
    parser.add_argument(
        "--no-use-torch",
        dest="use_torch",
        action="store_false",
        help="numpy ベースの実装を使用（後方互換性）",
    )
    args = parser.parse_args()

    ctx = ProjectContext(args.profile)
    base_dir = Path(args.subspace_dir) if args.subspace_dir else ctx.results_dir() / "emotion_subspaces"
    path_a = base_dir / f"{args.model_a}_subspaces.pkl"
    path_b = base_dir / f"{args.model_b}_subspaces.pkl"
    if not path_a.exists() or not path_b.exists():
        raise FileNotFoundError("サブスペースファイルが見つかりません。")

    phase_started = time.perf_counter()
    start_time = time.time()
    print(f"[Phase 4] サブスペースファイルを読み込み中...")
    data_a = _load_subspace(path_a)["subspaces"]
    data_b = _load_subspace(path_b)["subspaces"]
    layers = sorted(set(data_a.get("neutral", {}).keys()) & set(data_b.get("neutral", {}).keys()))
    elapsed = time.time() - start_time
    print(f"[Phase 4] 読み込み完了: {elapsed:.2f}秒 (層数: {len(layers)})")

    # デバイス設定
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_default_device()
    
    use_torch = args.use_torch
    
    if use_torch:
        print(f"[Phase 4] 計算デバイス: {device} (torch ベース実装)")
    else:
        print(f"[Phase 4] 計算デバイス: CPU (numpy ベース実装)")
    
    start_time = time.time()
    print(f"[Phase 4] アライメント計算中... (層数: {len(layers)}, k_max: {args.k_max})")
    results: List[Dict] = []
    total_combinations = len(layers) * args.k_max * 3  # 層数 × k値 × 感情数
    processed = 0
    
    for layer in layers:
        neutral_a = data_a["neutral"][layer]["components"]
        neutral_b = data_b["neutral"][layer]["components"]
        
        for k in range(1, args.k_max + 1):
            if neutral_a.shape[0] < k or neutral_b.shape[0] < k:
                continue
            
            if use_torch:
                # torch ベースの実装（GPU/MPS対応）
                # MPSはfloat64をサポートしていないため、デバイスに応じてdtypeを選択
                if device.type == "mps":
                    dtype = torch.float32
                else:
                    dtype = torch.float64
                neutral_a_torch = torch.from_numpy(neutral_a).to(device, dtype=dtype)
                neutral_b_torch = torch.from_numpy(neutral_b).to(device, dtype=dtype)
                w_torch = _procrustes_torch(neutral_a_torch, neutral_b_torch, k, device)
                w = w_torch.cpu().numpy()
            else:
                # numpy ベースの実装（後方互換性）
                w = _procrustes(neutral_a, neutral_b, k)
            
            for emotion in ("gratitude", "anger", "apology"):
                if emotion not in data_a or emotion not in data_b:
                    continue
                comp_a = data_a[emotion][layer]["components"]
                comp_b = data_b[emotion][layer]["components"]
                if comp_a.shape[0] < k or comp_b.shape[0] < k:
                    continue
                
                if use_torch:
                    # torch ベースの実装
                    # MPSはfloat64をサポートしていないため、デバイスに応じてdtypeを選択
                    if device.type == "mps":
                        dtype = torch.float32
                    else:
                        dtype = torch.float64
                    comp_a_torch = torch.from_numpy(comp_a).to(device, dtype=dtype)
                    comp_b_torch = torch.from_numpy(comp_b).to(device, dtype=dtype)
                    overlap_before = _subspace_overlap_torch(comp_a_torch, comp_b_torch, k, device)
                    
                    # map A->B
                    comp_a_k_torch = comp_a_torch[:k]
                    mapped_torch = w_torch @ comp_a_k_torch
                    overlap_after = _subspace_overlap_torch(mapped_torch, comp_b_torch, k, device)
                else:
                    # numpy ベースの実装
                    overlap_before = _subspace_overlap(comp_a, comp_b, k)
                    # map A->B
                    mapped = (w @ comp_a[:k]).astype(np.float64)
                    overlap_after = _subspace_overlap(mapped, comp_b, k)
                
                results.append(
                    {
                        "layer": layer,
                        "k": k,
                        "emotion": emotion,
                        "overlap_before": overlap_before,
                        "overlap_after": overlap_after,
                    }
                )
                processed += 1
                if processed % 10 == 0:
                    print(f"  [Phase 4] 進捗: {processed}/{total_combinations} 組み合わせ処理済み")
    
    elapsed = time.time() - start_time
    print(f"[Phase 4] 計算完了: {elapsed:.2f}秒 ({elapsed/60:.2f}分) (結果数: {len(results)})")

    out_dir = ctx.results_dir() / "alignment"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model_a}_vs_{args.model_b}_token_based_full.pkl"
    
    start_time = time.time()
    print(f"[Phase 4] 結果を保存中...")
    with out_path.open("wb") as f:
        pickle.dump({"results": results, "metadata": {"profile": args.profile, "model_a": args.model_a, "model_b": args.model_b}}, f)
    elapsed = time.time() - start_time
    print(f"[Phase 4] 保存完了: {elapsed:.2f}秒")

    record_phase_timing(
        context=ctx,
        phase="phase4",
        started_at=phase_started,
        model=f"{args.model_a}_vs_{args.model_b}",
        device=str(device) if use_torch else "cpu",
        samples=len(layers),
        metadata={
            "k_max": args.k_max,
            "use_torch": use_torch,
            "layer_count": len(layers),
            "result_count": len(results),
            "output_path": str(out_path),
        },
        cli_args=sys.argv[1:],
    )
    print(f"✓ Phase4 完了: {out_path}")


if __name__ == "__main__":
    main()
