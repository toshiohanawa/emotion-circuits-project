"""
感情方向ベクトルの抽出と分析モジュール
emotion_vec[layer] = mean(resid_emotion[layer]) - mean(resid_neutral[layer])
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch


class EmotionVectorExtractor:
    """感情方向ベクトルを抽出するクラス"""
    
    def __init__(self, activations_dir: Path):
        """
        初期化
        
        Args:
            activations_dir: 活性データが保存されているディレクトリ
        """
        self.activations_dir = Path(activations_dir)
        self.emotion_vectors = {}
        self.metadata = {}
    
    def load_activations(self, emotion_label: str) -> Dict:
        """
        活性データを読み込む
        
        Args:
            emotion_label: 感情ラベル（gratitude, anger, apology, neutral）
            
        Returns:
            活性データの辞書
        """
        file_path = self.activations_dir / f"activations_{emotion_label}.pkl"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Activation file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            activations = pickle.load(f)
        
        return activations
    
    def compute_emotion_vector(
        self,
        emotion_label: str,
        neutral_label: str = "neutral",
        use_mlp: bool = False
    ) -> np.ndarray:
        """
        感情方向ベクトルを計算
        
        Args:
            emotion_label: 感情ラベル（gratitude, anger, apology）
            neutral_label: 中立ラベル（デフォルト: "neutral"）
            use_mlp: MLP出力を使用するか（Falseの場合はresidual stream）
            
        Returns:
            感情方向ベクトル [n_layers, d_model]
        """
        # 活性データを読み込み
        emotion_activations = self.load_activations(emotion_label)
        neutral_activations = self.load_activations(neutral_label)
        
        # メタデータを保存
        if not self.metadata:
            self.metadata = emotion_activations['metadata']
        
        # Residual streamまたはMLP出力を選択
        key = 'mlp_output' if use_mlp else 'residual_stream'
        
        emotion_data = emotion_activations[key]  # [n_layers, n_samples]
        neutral_data = neutral_activations[key]   # [n_layers, n_samples]
        
        n_layers = len(emotion_data)
        emotion_vectors = []
        
        for layer_idx in range(n_layers):
            # 各層のサンプルを平均
            # emotion_data[layer_idx]は [n_samples] のリストで、各要素は [pos, d_model]
            # 最後の位置（文末）の活性を使用
            emotion_samples = np.array([sample[-1] for sample in emotion_data[layer_idx]])  # [n_samples, d_model]
            neutral_samples = np.array([sample[-1] for sample in neutral_data[layer_idx]])  # [n_samples, d_model]
            
            # 平均を計算
            emotion_mean = np.mean(emotion_samples, axis=0)  # [d_model]
            neutral_mean = np.mean(neutral_samples, axis=0)  # [d_model]
            
            # 感情方向ベクトル = 感情の平均 - 中立の平均
            emotion_vec = emotion_mean - neutral_mean  # [d_model]
            emotion_vectors.append(emotion_vec)
        
        emotion_vector = np.array(emotion_vectors)  # [n_layers, d_model]
        
        # キャッシュに保存
        cache_key = f"{emotion_label}_{'mlp' if use_mlp else 'resid'}"
        self.emotion_vectors[cache_key] = emotion_vector
        
        return emotion_vector
    
    def compute_all_emotion_vectors(
        self,
        emotion_labels: List[str] = ["gratitude", "anger", "apology"],
        use_mlp: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        全感情カテゴリの方向ベクトルを計算
        
        Args:
            emotion_labels: 感情ラベルのリスト
            use_mlp: MLP出力を使用するか
            
        Returns:
            感情ラベルをキーとする方向ベクトルの辞書
        """
        all_vectors = {}
        
        for emotion_label in emotion_labels:
            vector = self.compute_emotion_vector(emotion_label, use_mlp=use_mlp)
            all_vectors[emotion_label] = vector
        
        return all_vectors
    
    def compute_layer_norms(
        self,
        emotion_vector: np.ndarray,
        norm_type: str = "l2"
    ) -> np.ndarray:
        """
        層ごとのノルムを計算
        
        Args:
            emotion_vector: 感情方向ベクトル [n_layers, d_model]
            norm_type: ノルムの種類（"l1" または "l2"）
            
        Returns:
            層ごとのノルム [n_layers]
        """
        if norm_type == "l1":
            norms = np.linalg.norm(emotion_vector, ord=1, axis=1)
        elif norm_type == "l2":
            norms = np.linalg.norm(emotion_vector, ord=2, axis=1)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
        
        return norms
    
    def compute_cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> np.ndarray:
        """
        2つのベクトル間のcosine類似度を計算（層ごと）
        
        Args:
            vec1: ベクトル1 [n_layers, d_model]
            vec2: ベクトル2 [n_layers, d_model]
            
        Returns:
            層ごとのcosine類似度 [n_layers]
        """
        # 層ごとにcosine類似度を計算
        similarities = []
        
        for layer_idx in range(vec1.shape[0]):
            v1 = vec1[layer_idx]
            v2 = vec2[layer_idx]
            
            # Cosine類似度
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm1 * norm2)
            
            similarities.append(similarity)
        
        return np.array(similarities)
    
    def compute_emotion_distances(
        self,
        emotion_vectors: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        感情間の距離（cosine類似度）を計算
        
        Args:
            emotion_vectors: 感情ラベルをキーとする方向ベクトルの辞書
            
        Returns:
            感情ペアをキーとする類似度の辞書
        """
        distances = {}
        
        emotions = list(emotion_vectors.keys())
        
        for i, emotion1 in enumerate(emotions):
            for emotion2 in emotions[i+1:]:
                vec1 = emotion_vectors[emotion1]
                vec2 = emotion_vectors[emotion2]
                
                similarities = self.compute_cosine_similarity(vec1, vec2)
                distances[f"{emotion1}_vs_{emotion2}"] = similarities
        
        return distances


def main():
    """メイン関数（テスト用）"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract emotion vectors")
    parser.add_argument("--activations_dir", type=str, required=True, help="Activations directory")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--use-mlp", action="store_true", help="Use MLP output instead of residual stream")
    
    args = parser.parse_args()
    
    # Extractorを作成
    extractor = EmotionVectorExtractor(args.activations_dir)
    
    # 全感情の方向ベクトルを計算
    emotion_vectors = extractor.compute_all_emotion_vectors(use_mlp=args.use_mlp)
    
    print(f"Computed emotion vectors for {len(emotion_vectors)} emotions")
    for emotion, vector in emotion_vectors.items():
        print(f"  - {emotion}: shape {vector.shape}")
    
    # 感情間の距離を計算
    distances = extractor.compute_emotion_distances(emotion_vectors)
    print(f"\nComputed distances for {len(distances)} emotion pairs")
    for pair, similarities in distances.items():
        avg_sim = np.mean(similarities)
        print(f"  - {pair}: average similarity = {avg_sim:.4f}")
    
    # 保存
    if args.output:
        output_data = {
            'emotion_vectors': emotion_vectors,
            'emotion_distances': distances,
            'metadata': extractor.metadata
        }
        
        with open(args.output, 'wb') as f:
            pickle.dump(output_data, f)
        
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()

