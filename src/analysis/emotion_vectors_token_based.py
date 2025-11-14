"""
感情語トークンベースの感情方向ベクトル抽出
文末ではなく、感情語そのもののトークン位置で感情が最も濃く表現されている可能性を検証
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set
import re


# 各感情カテゴリの代表トークン
EMOTION_TOKENS = {
    "gratitude": {
        "thank", "thanks", "thanked", "thanking",
        "grateful", "gratitude", "gratefully",
        "appreciate", "appreciated", "appreciating", "appreciation",
        "gratitude"
    },
    "anger": {
        "angry", "anger", "angrily",
        "frustrated", "frustrating", "frustration",
        "terrible", "terribly",
        "annoyed", "annoying", "annoyance",
        "upset", "upsetting",
        "mad", "maddening",
        "furious", "furiously",
        "irritated", "irritating", "irritation"
    },
    "apology": {
        "sorry", "sorrier", "sorriest",
        "apologize", "apologized", "apologizing", "apology", "apologies",
        "regret", "regretted", "regretting", "regretful",
        "apologetic", "apologetically"
    }
}


class TokenBasedEmotionVectorExtractor:
    """感情語トークンベースの感情方向ベクトル抽出クラス"""
    
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
    
    def find_emotion_token_positions(
        self,
        token_strings: List[str],
        emotion_label: str
    ) -> List[int]:
        """
        感情語トークンの位置を特定
        
        Args:
            token_strings: トークン文字列のリスト
            emotion_label: 感情ラベル
            
        Returns:
            感情語トークンの位置のリスト（複数見つかる場合は全て返す）
        """
        if emotion_label == "neutral":
            # 中立の場合は最後のトークンを使用
            return [len(token_strings) - 1]
        
        if emotion_label not in EMOTION_TOKENS:
            # 感情語が定義されていない場合は最後のトークンを使用
            return [len(token_strings) - 1]
        
        emotion_tokens = EMOTION_TOKENS[emotion_label]
        positions = []
        
        for idx, token_str in enumerate(token_strings):
            # トークン文字列を正規化（小文字、句読点除去）
            normalized = token_str.lower().strip('.,!?;:')
            if normalized in emotion_tokens:
                positions.append(idx)
        
        # 感情語が見つからない場合は最後のトークンを使用
        if not positions:
            positions = [len(token_strings) - 1]
        
        return positions
    
    def compute_emotion_vector_from_tokens(
        self,
        emotion_label: str,
        neutral_label: str = "neutral",
        use_mlp: bool = False,
        position_strategy: str = "first"
    ) -> np.ndarray:
        """
        感情語トークン位置のresidual streamから感情方向ベクトルを計算
        
        Args:
            emotion_label: 感情ラベル（gratitude, anger, apology）
            neutral_label: 中立ラベル（デフォルト: "neutral"）
            use_mlp: MLP出力を使用するか（Falseの場合はresidual stream）
            position_strategy: 位置選択戦略（"first": 最初の感情語, "last": 最後の感情語, "all": 全ての平均）
            
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
        emotion_token_strings = emotion_activations['token_strings']
        neutral_token_strings = neutral_activations['token_strings']
        
        n_layers = len(emotion_data)
        emotion_vectors = []
        
        for layer_idx in range(n_layers):
            # 各サンプルの感情語位置の活性を集める
            emotion_samples_at_tokens = []
            neutral_samples_at_tokens = []
            
            # 感情カテゴリのサンプルを処理
            for sample_idx, sample_activations in enumerate(emotion_data[layer_idx]):
                token_strings = emotion_token_strings[sample_idx]
                positions = self.find_emotion_token_positions(token_strings, emotion_label)
                
                # 位置選択戦略に応じて活性を選択
                if position_strategy == "first":
                    pos = positions[0]
                elif position_strategy == "last":
                    pos = positions[-1]
                elif position_strategy == "all":
                    # 全ての位置の平均
                    selected_activations = [sample_activations[p] for p in positions]
                    emotion_samples_at_tokens.append(np.mean(selected_activations, axis=0))
                    continue
                else:
                    pos = positions[0]
                
                if pos < len(sample_activations):
                    emotion_samples_at_tokens.append(sample_activations[pos])
            
            # 中立カテゴリのサンプルを処理（最後のトークンを使用）
            for sample_idx, sample_activations in enumerate(neutral_data[layer_idx]):
                token_strings = neutral_token_strings[sample_idx]
                pos = len(token_strings) - 1
                if pos < len(sample_activations):
                    neutral_samples_at_tokens.append(sample_activations[pos])
            
            # 平均を計算
            if emotion_samples_at_tokens:
                emotion_mean = np.mean(emotion_samples_at_tokens, axis=0)  # [d_model]
            else:
                emotion_mean = np.zeros(neutral_samples_at_tokens[0].shape)
            
            if neutral_samples_at_tokens:
                neutral_mean = np.mean(neutral_samples_at_tokens, axis=0)  # [d_model]
            else:
                neutral_mean = np.zeros(emotion_mean.shape)
            
            # 感情方向ベクトル = 感情の平均 - 中立の平均
            emotion_vec = emotion_mean - neutral_mean  # [d_model]
            emotion_vectors.append(emotion_vec)
        
        emotion_vector = np.array(emotion_vectors)  # [n_layers, d_model]
        
        return emotion_vector
    
    def compute_all_emotion_vectors(
        self,
        emotion_labels: List[str] = ["gratitude", "anger", "apology"],
        use_mlp: bool = False,
        position_strategy: str = "first"
    ) -> Dict[str, np.ndarray]:
        """
        全感情カテゴリの方向ベクトルを計算
        
        Args:
            emotion_labels: 感情ラベルのリスト
            use_mlp: MLP出力を使用するか
            position_strategy: 位置選択戦略
            
        Returns:
            感情ラベルをキーとする方向ベクトルの辞書
        """
        all_vectors = {}
        
        for emotion_label in emotion_labels:
            vector = self.compute_emotion_vector_from_tokens(
                emotion_label,
                use_mlp=use_mlp,
                position_strategy=position_strategy
            )
            all_vectors[emotion_label] = vector
        
        return all_vectors
    
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
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract emotion vectors from emotion tokens")
    parser.add_argument("--activations_dir", type=str, required=True, help="Activations directory")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--use-mlp", action="store_true", help="Use MLP output instead of residual stream")
    parser.add_argument("--position-strategy", type=str, default="first", choices=["first", "last", "all"], help="Position selection strategy")
    
    args = parser.parse_args()
    
    # Extractorを作成
    extractor = TokenBasedEmotionVectorExtractor(args.activations_dir)
    
    # 全感情の方向ベクトルを計算
    emotion_vectors = extractor.compute_all_emotion_vectors(
        use_mlp=args.use_mlp,
        position_strategy=args.position_strategy
    )
    
    print(f"Computed emotion vectors (token-based) for {len(emotion_vectors)} emotions")
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
            'metadata': extractor.metadata,
            'position_strategy': args.position_strategy,
            'use_mlp': args.use_mlp
        }
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)
        
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()

