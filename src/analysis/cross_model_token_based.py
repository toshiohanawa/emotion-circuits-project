"""
モデル間での感情語トークンベースの感情方向類似度分析
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from src.visualization.emotion_plots import EmotionVisualizer


def load_model_vectors(model_names: List[str], vectors_dir: Path, suffix: str = "_token_based") -> Dict[str, Dict[str, np.ndarray]]:
    """
    複数モデルの感情ベクトルを読み込む
    
    Args:
        model_names: モデル名のリスト
        vectors_dir: ベクトルファイルが保存されているディレクトリ
        suffix: ファイル名のサフィックス
        
    Returns:
        モデル名をキーとする感情ベクトルの辞書
    """
    model_vectors = {}
    
    for model_name in model_names:
        vectors_file = vectors_dir / f"{model_name}_vectors{suffix}.pkl"
        
        if not vectors_file.exists():
            print(f"Warning: {vectors_file} not found, skipping...")
            continue
        
        with open(vectors_file, 'rb') as f:
            data = pickle.load(f)
        
        model_vectors[model_name] = data['emotion_vectors']
    
    return model_vectors


def compute_cross_model_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray
) -> np.ndarray:
    """
    2つのモデルの感情ベクトル間のcosine類似度を計算（層ごと）
    
    Args:
        vec1: モデル1のベクトル [n_layers, d_model]
        vec2: モデル2のベクトル [n_layers, d_model]
        
    Returns:
        層ごとのcosine類似度 [n_layers]
    """
    similarities = []
    
    for layer_idx in range(vec1.shape[0]):
        v1 = vec1[layer_idx]
        v2 = vec2[layer_idx]
        
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (norm1 * norm2)
        
        similarities.append(similarity)
    
    return np.array(similarities)


def create_similarity_table(
    model_vectors: Dict[str, Dict[str, np.ndarray]],
    emotion_labels: List[str] = ["gratitude", "anger", "apology"]
) -> pd.DataFrame:
    """
    モデル間の類似度を表形式で作成
    
    Args:
        model_vectors: モデル名をキーとする感情ベクトルの辞書
        emotion_labels: 感情ラベルのリスト
        
    Returns:
        類似度テーブル（DataFrame）
    """
    models = list(model_vectors.keys())
    n_models = len(models)
    
    results = []
    
    for emotion_label in emotion_labels:
        for i in range(n_models):
            for j in range(i+1, n_models):
                model1 = models[i]
                model2 = models[j]
                
                vec1 = model_vectors[model1][emotion_label]
                vec2 = model_vectors[model2][emotion_label]
                
                similarities = compute_cross_model_similarity(vec1, vec2)
                avg_similarity = np.mean(similarities)
                std_similarity = np.std(similarities)
                
                results.append({
                    'emotion': emotion_label,
                    'model1': model1,
                    'model2': model2,
                    'avg_similarity': avg_similarity,
                    'std_similarity': std_similarity,
                    'min_similarity': np.min(similarities),
                    'max_similarity': np.max(similarities)
                })
    
    df = pd.DataFrame(results)
    return df


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-model emotion vector analysis (token-based)")
    parser.add_argument("--vectors_dir", type=str, default="results/emotion_vectors", help="Vectors directory")
    parser.add_argument("--models", type=str, nargs='+', default=["gpt2", "pythia-160m", "gpt-neo-125m"], help="Model names")
    parser.add_argument("--output_dir", type=str, default="results/plots/cross_model_token_based", help="Output directory")
    parser.add_argument("--output_table", type=str, default="results/cross_model_similarity_token_based.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    # モデルベクトルを読み込み
    print("Loading model vectors (token-based)...")
    model_vectors = load_model_vectors(args.models, Path(args.vectors_dir), suffix="_token_based")
    
    print(f"Loaded vectors for {len(model_vectors)} models: {list(model_vectors.keys())}")
    
    # 類似度テーブルを作成
    print("\nComputing cross-model similarities...")
    similarity_table = create_similarity_table(model_vectors)
    
    # テーブルを保存
    similarity_table.to_csv(args.output_table, index=False)
    print(f"\nSimilarity table saved to: {args.output_table}")
    
    # テーブルを表示
    print("\nCross-Model Similarity Table (Token-Based):")
    print("=" * 80)
    print(similarity_table.to_string(index=False))
    print("=" * 80)
    
    # 可視化
    print("\nCreating visualizations...")
    visualizer = EmotionVisualizer(args.output_dir)
    
    emotion_labels = ["gratitude", "anger", "apology"]
    
    for emotion_label in emotion_labels:
        visualizer.plot_cross_model_similarity(
            model_vectors,
            emotion_label,
            save_path=visualizer.output_dir / f"cross_model_{emotion_label}_token_based.png"
        )
    
    print(f"\nAll plots saved to: {visualizer.output_dir}")


if __name__ == "__main__":
    main()

