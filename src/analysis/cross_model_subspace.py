"""
モデル間での感情サブスペースoverlap分析
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from src.analysis.emotion_subspace import EmotionSubspaceAnalyzer


def load_model_subspaces(model_names: List[str], subspaces_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    複数モデルの感情サブスペースを読み込む
    
    Args:
        model_names: モデル名のリスト
        subspaces_dir: サブスペースファイルが保存されているディレクトリ
        
    Returns:
        モデル名をキーとする感情サブスペースの辞書
    """
    model_subspaces = {}
    
    for model_name in model_names:
        subspaces_file = subspaces_dir / f"{model_name}_subspaces.pkl"
        
        if not subspaces_file.exists():
            print(f"Warning: {subspaces_file} not found, skipping...")
            continue
        
        with open(subspaces_file, 'rb') as f:
            data = pickle.load(f)
        
        model_subspaces[model_name] = data['subspaces']
    
    return model_subspaces


def compute_subspace_overlap(
    subspace1: np.ndarray,
    subspace2: np.ndarray
) -> np.ndarray:
    """
    サブスペース間のoverlapを計算
    
    Args:
        subspace1: サブスペース1 [n_layers, n_components, d_model]
        subspace2: サブスペース2 [n_layers, n_components, d_model]
        
    Returns:
        層ごとのoverlap [n_layers]
    """
    n_layers = subspace1.shape[0]
    overlaps = []
    
    for layer_idx in range(n_layers):
        U1 = subspace1[layer_idx]  # [n_components1, d_model]
        U2 = subspace2[layer_idx]  # [n_components2, d_model]
        
        # Gram行列を計算
        G = U1 @ U2.T  # [n_components1, n_components2]
        
        # SVDでprincipal anglesを計算
        U, s, Vt = np.linalg.svd(G, full_matrices=False)
        
        # Principal angles (cosine of angles)
        cosines = np.clip(s, -1.0, 1.0)
        
        # 最初の数成分の平均を取る
        n_use = min(5, len(cosines))
        overlap = np.mean(cosines[:n_use])
        overlaps.append(overlap)
    
    return np.array(overlaps)


def create_subspace_overlap_table(
    model_subspaces: Dict[str, Dict[str, np.ndarray]],
    emotion_labels: List[str] = ["gratitude", "anger", "apology"]
) -> pd.DataFrame:
    """
    モデル間のサブスペースoverlapを表形式で作成
    
    Args:
        model_subspaces: モデル名をキーとする感情サブスペースの辞書
        emotion_labels: 感情ラベルのリスト
        
    Returns:
        Overlapテーブル（DataFrame）
    """
    models = list(model_subspaces.keys())
    n_models = len(models)
    
    results = []
    
    for emotion_label in emotion_labels:
        for i in range(n_models):
            for j in range(i+1, n_models):
                model1 = models[i]
                model2 = models[j]
                
                subspace1 = model_subspaces[model1][emotion_label]
                subspace2 = model_subspaces[model2][emotion_label]
                
                overlaps = compute_subspace_overlap(subspace1, subspace2)
                avg_overlap = np.mean(overlaps)
                std_overlap = np.std(overlaps)
                
                results.append({
                    'emotion': emotion_label,
                    'model1': model1,
                    'model2': model2,
                    'avg_overlap': avg_overlap,
                    'std_overlap': std_overlap,
                    'min_overlap': np.min(overlaps),
                    'max_overlap': np.max(overlaps)
                })
    
    df = pd.DataFrame(results)
    return df


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-model emotion subspace overlap analysis")
    parser.add_argument("--subspaces_dir", type=str, default="results/emotion_subspaces", help="Subspaces directory")
    parser.add_argument("--models", type=str, nargs='+', default=["gpt2", "pythia-160m", "gpt-neo-125m"], help="Model names")
    parser.add_argument("--output_table", type=str, default="results/cross_model_subspace_overlap.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    # モデルサブスペースを読み込み
    print("Loading model subspaces...")
    model_subspaces = load_model_subspaces(args.models, Path(args.subspaces_dir))
    
    print(f"Loaded subspaces for {len(model_subspaces)} models: {list(model_subspaces.keys())}")
    
    # Overlapテーブルを作成
    print("\nComputing cross-model subspace overlaps...")
    overlap_table = create_subspace_overlap_table(model_subspaces)
    
    # テーブルを保存
    overlap_table.to_csv(args.output_table, index=False)
    print(f"\nSubspace overlap table saved to: {args.output_table}")
    
    # テーブルを表示
    print("\nCross-Model Subspace Overlap Table:")
    print("=" * 80)
    print(overlap_table.to_string(index=False))
    print("=" * 80)
    
    # ランダムベースラインとの比較（参考値）
    print("\nNote: Random baseline overlap is typically around 0.0-0.1")
    print("Higher values indicate shared subspace structure across models.")


if __name__ == "__main__":
    main()

