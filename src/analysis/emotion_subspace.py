"""
感情サブスペース分析
感情表現は1次元の方向ではなく、数次元のサブスペースに広がっている可能性を検証
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class EmotionSubspaceAnalyzer:
    """感情サブスペース分析クラス"""
    
    def __init__(self, activations_dir: Path):
        """
        初期化
        
        Args:
            activations_dir: 活性データが保存されているディレクトリ
        """
        self.activations_dir = Path(activations_dir)
        self.subspaces = {}
        self.metadata = {}
    
    def load_activations(self, emotion_label: str) -> Dict:
        """
        活性データを読み込む
        
        Args:
            emotion_label: 感情ラベル
            
        Returns:
            活性データの辞書
        """
        file_path = self.activations_dir / f"activations_{emotion_label}.pkl"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Activation file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            activations = pickle.load(f)
        
        return activations
    
    def compute_emotion_subspace(
        self,
        emotion_label: str,
        n_components: int = 10,
        use_mlp: bool = False,
        position: str = "last"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        感情サブスペースをPCAで計算
        
        Args:
            emotion_label: 感情ラベル
            n_components: PCAの主成分数
            use_mlp: MLP出力を使用するか
            position: 使用する位置（"last": 文末, "all": 全位置）
            
        Returns:
            (subspace_vectors, explained_variance_ratio)
            - subspace_vectors: [n_layers, n_components, d_model]
            - explained_variance_ratio: [n_layers, n_components]
        """
        activations = self.load_activations(emotion_label)
        
        if not self.metadata:
            self.metadata = activations['metadata']
        
        key = 'mlp_output' if use_mlp else 'residual_stream'
        data = activations[key]  # [n_layers, n_samples]
        
        n_layers = len(data)
        subspace_vectors_list = []
        explained_variance_list = []
        
        for layer_idx in range(n_layers):
            layer_samples = []
            
            # 各サンプルの活性を集める
            for sample_activations in data[layer_idx]:
                if position == "last":
                    # 文末の活性を使用
                    layer_samples.append(sample_activations[-1])
                elif position == "all":
                    # 全位置の活性を使用（フラット化）
                    layer_samples.append(sample_activations.flatten())
                else:
                    layer_samples.append(sample_activations[-1])
            
            # データ行列を作成 [n_samples, d_model]
            X = np.array(layer_samples)
            
            # 平均を引いて中心化
            X_centered = X - np.mean(X, axis=0)
            
            # SVDでPCAを実行
            n_comp = min(n_components, X.shape[0], X.shape[1])
            U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
            
            # 主成分ベクトル [n_components, d_model]
            components = Vt[:n_comp]  # 最初のn_comp成分
            
            # 説明分散比を計算
            total_var = np.sum(s ** 2)
            explained_var_ratio = (s[:n_comp] ** 2) / total_var if total_var > 0 else np.zeros(n_comp)
            
            subspace_vectors_list.append(components)
            explained_variance_list.append(explained_var_ratio)
        
        subspace_vectors = np.array(subspace_vectors_list)  # [n_layers, n_components, d_model]
        explained_variance = np.array(explained_variance_list)  # [n_layers, n_components]
        
        return subspace_vectors, explained_variance
    
    def compute_principal_angles(
        self,
        subspace1: np.ndarray,
        subspace2: np.ndarray
    ) -> np.ndarray:
        """
        2つのサブスペース間のprincipal anglesを計算（層ごと）
        
        Args:
            subspace1: サブスペース1 [n_layers, n_components, d_model]
            subspace2: サブスペース2 [n_layers, n_components, d_model]
            
        Returns:
            層ごとのprincipal angles [n_layers, n_components]
        """
        n_layers = subspace1.shape[0]
        n_components = min(subspace1.shape[1], subspace2.shape[1])
        
        principal_angles_list = []
        
        for layer_idx in range(n_layers):
            U1 = subspace1[layer_idx]  # [n_components1, d_model]
            U2 = subspace2[layer_idx]  # [n_components2, d_model]
            
            # Gram行列を計算
            G = U1 @ U2.T  # [n_components1, n_components2]
            
            # SVDでprincipal anglesを計算
            U, s, Vt = np.linalg.svd(G, full_matrices=False)
            
            # Principal angles (cosine of angles)
            cosines = np.clip(s, -1.0, 1.0)
            angles = np.arccos(cosines)  # [min(n_components1, n_components2)]
            
            # n_components分だけ返す
            principal_angles_list.append(angles[:n_components])
        
        return np.array(principal_angles_list)  # [n_layers, n_components]
    
    def compute_subspace_overlap(
        self,
        subspace1: np.ndarray,
        subspace2: np.ndarray
    ) -> np.ndarray:
        """
        サブスペース間のoverlap（平均cosine）を計算
        
        Args:
            subspace1: サブスペース1 [n_layers, n_components, d_model]
            subspace2: サブスペース2 [n_layers, n_components, d_model]
            
        Returns:
            層ごとのoverlap [n_layers]
        """
        principal_angles = self.compute_principal_angles(subspace1, subspace2)
        
        # Principal anglesのcosineの平均を計算
        overlaps = []
        for layer_idx in range(principal_angles.shape[0]):
            angles = principal_angles[layer_idx]
            cosines = np.cos(angles)
            # 最初の数成分の平均を取る
            n_use = min(5, len(cosines))
            overlap = np.mean(cosines[:n_use])
            overlaps.append(overlap)
        
        return np.array(overlaps)


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze emotion subspaces")
    parser.add_argument("--activations_dir", type=str, required=True, help="Activations directory")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--n-components", type=int, default=10, help="Number of PCA components")
    parser.add_argument("--use-mlp", action="store_true", help="Use MLP output")
    parser.add_argument("--position", type=str, default="last", choices=["last", "all"], help="Position to use")
    
    args = parser.parse_args()
    
    analyzer = EmotionSubspaceAnalyzer(args.activations_dir)
    
    # 各感情のサブスペースを計算
    emotion_labels = ["gratitude", "anger", "apology", "neutral"]
    subspaces = {}
    explained_variances = {}
    
    for emotion_label in emotion_labels:
        subspace, explained_var = analyzer.compute_emotion_subspace(
            emotion_label,
            n_components=args.n_components,
            use_mlp=args.use_mlp,
            position=args.position
        )
        subspaces[emotion_label] = subspace
        explained_variances[emotion_label] = explained_var
        
        avg_explained = np.mean(explained_var[:, :5])  # 上位5成分の平均
        print(f"{emotion_label}: subspace shape {subspace.shape}, avg explained variance (top 5) = {avg_explained:.4f}")
    
    # 感情間のサブスペースoverlapを計算
    print("\nComputing subspace overlaps...")
    overlaps = {}
    
    emotions = ["gratitude", "anger", "apology"]
    for i, emotion1 in enumerate(emotions):
        for emotion2 in emotions[i+1:]:
            overlap = analyzer.compute_subspace_overlap(
                subspaces[emotion1],
                subspaces[emotion2]
            )
            overlaps[f"{emotion1}_vs_{emotion2}"] = overlap
            avg_overlap = np.mean(overlap)
            print(f"{emotion1} vs {emotion2}: average overlap = {avg_overlap:.4f}")
    
    # 保存
    if args.output:
        output_data = {
            'subspaces': subspaces,
            'explained_variances': explained_variances,
            'overlaps': overlaps,
            'metadata': analyzer.metadata,
            'n_components': args.n_components
        }
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)
        
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()

