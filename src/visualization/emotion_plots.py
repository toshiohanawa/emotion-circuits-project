"""
感情方向ベクトルの可視化モジュール
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from typing import Dict, List, Optional

# フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


class EmotionVisualizer:
    """感情方向ベクトルの可視化クラス"""
    
    def __init__(self, output_dir: Path):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_layer_norms(
        self,
        emotion_vectors: Dict[str, np.ndarray],
        norm_type: str = "l2",
        save_path: Optional[Path] = None
    ):
        """
        層ごとの感情方向ベクトルのノルムをプロット
        
        Args:
            emotion_vectors: 感情ラベルをキーとする方向ベクトルの辞書
            norm_type: ノルムの種類（"l1" または "l2"）
            save_path: 保存パス（Noneの場合は表示のみ）
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for emotion_label, vector in emotion_vectors.items():
            if norm_type == "l1":
                norms = np.linalg.norm(vector, ord=1, axis=1)
            else:
                norms = np.linalg.norm(vector, ord=2, axis=1)
            
            layers = np.arange(len(norms))
            ax.plot(layers, norms, marker='o', label=emotion_label, linewidth=2, markersize=6)
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel(f'{norm_type.upper()} Norm', fontsize=12)
        ax.set_title(f'Emotion Vector Strength by Layer ({norm_type.upper()} Norm)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_emotion_distances(
        self,
        emotion_distances: Dict[str, np.ndarray],
        save_path: Optional[Path] = None
    ):
        """
        感情間の距離（cosine類似度）をプロット
        
        Args:
            emotion_distances: 感情ペアをキーとする類似度の辞書
            save_path: 保存パス
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for pair, similarities in emotion_distances.items():
            layers = np.arange(len(similarities))
            emotion1, emotion2 = pair.split('_vs_')
            label = f"{emotion1} vs {emotion2}"
            ax.plot(layers, similarities, marker='o', label=label, linewidth=2, markersize=6)
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title('Emotion Distance by Layer (Cosine Similarity)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_cross_model_similarity(
        self,
        model_vectors: Dict[str, Dict[str, np.ndarray]],
        emotion_label: str,
        save_path: Optional[Path] = None
    ):
        """
        モデル間での感情方向ベクトルの類似度をプロット
        
        Args:
            model_vectors: モデル名をキーとする感情ベクトルの辞書
            emotion_label: 感情ラベル
            save_path: 保存パス
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(model_vectors.keys())
        n_models = len(models)
        
        # 全てのモデルペアの類似度を計算
        similarities_data = []
        labels = []
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                model1 = models[i]
                model2 = models[j]
                
                vec1 = model_vectors[model1][emotion_label]
                vec2 = model_vectors[model2][emotion_label]
                
                # Cosine類似度を計算
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
                
                similarities_data.append(similarities)
                labels.append(f"{model1} vs {model2}")
        
        # プロット
        layers = np.arange(len(similarities_data[0]))
        for similarities, label in zip(similarities_data, labels):
            ax.plot(layers, similarities, marker='o', label=label, linewidth=2, markersize=6)
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title(f'Cross-Model Similarity: {emotion_label.capitalize()}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_emotion_similarity_heatmap(
        self,
        emotion_distances: Dict[str, np.ndarray],
        layer_idx: Optional[int] = None,
        save_path: Optional[Path] = None
    ):
        """
        感情間の類似度をヒートマップで表示
        
        Args:
            emotion_distances: 感情ペアをキーとする類似度の辞書
            layer_idx: 層インデックス（Noneの場合は平均）
            save_path: 保存パス
        """
        # 感情ラベルを抽出
        emotions = set()
        for pair in emotion_distances.keys():
            emotion1, emotion2 = pair.split('_vs_')
            emotions.add(emotion1)
            emotions.add(emotion2)
        
        emotions = sorted(list(emotions))
        n_emotions = len(emotions)
        
        # 類似度行列を作成
        similarity_matrix = np.eye(n_emotions)  # 対角成分は1
        
        for pair, similarities in emotion_distances.items():
            emotion1, emotion2 = pair.split('_vs_')
            idx1 = emotions.index(emotion1)
            idx2 = emotions.index(emotion2)
            
            if layer_idx is not None:
                similarity = similarities[layer_idx]
            else:
                similarity = np.mean(similarities)
            
            similarity_matrix[idx1, idx2] = similarity
            similarity_matrix[idx2, idx1] = similarity
        
        # ヒートマップをプロット
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(similarity_matrix, cmap='coolwarm', vmin=0, vmax=1)
        
        # カラーバー
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity', fontsize=12)
        
        # ラベルを設定
        ax.set_xticks(np.arange(n_emotions))
        ax.set_yticks(np.arange(n_emotions))
        ax.set_xticklabels(emotions, fontsize=10)
        ax.set_yticklabels(emotions, fontsize=10)
        
        # 値を表示
        for i in range(n_emotions):
            for j in range(n_emotions):
                text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        layer_title = f"Layer {layer_idx}" if layer_idx is not None else "Average"
        ax.set_title(f'Emotion Similarity Matrix ({layer_title})', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """メイン関数（テスト用）"""
    import argparse
    from src.config.project_profiles import list_profiles
    from src.utils.project_context import ProjectContext, profile_help_text
    
    parser = argparse.ArgumentParser(description="Visualize emotion vectors")
    parser.add_argument("--vectors_file", type=str, required=True, help="Emotion vectors pickle file")
    parser.add_argument("--profile", type=str, choices=list_profiles(), default="baseline",
                        help=f"Dataset profile ({profile_help_text()})")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides profile default)")
    
    args = parser.parse_args()
    
    # ProjectContextを使用してパスを解決
    context = ProjectContext(profile_name=args.profile)
    results_dir = context.results_dir()
    
    # デフォルト値を設定（指定されていない場合）
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # データを読み込み
    with open(args.vectors_file, 'rb') as f:
        data = pickle.load(f)
    
    emotion_vectors = data['emotion_vectors']
    emotion_distances = data['emotion_distances']
    
    # Visualizerを作成
    visualizer = EmotionVisualizer(str(output_dir))
    
    # プロットを作成
    model_name = Path(args.vectors_file).stem.replace('_vectors', '')
    
    # 層ごとのノルム
    visualizer.plot_layer_norms(
        emotion_vectors,
        norm_type="l2",
        save_path=visualizer.output_dir / f"{model_name}_layer_norms_l2.png"
    )
    
    # 感情間の距離
    visualizer.plot_emotion_distances(
        emotion_distances,
        save_path=visualizer.output_dir / f"{model_name}_emotion_distances.png"
    )
    
    # 類似度ヒートマップ（平均）
    visualizer.plot_emotion_similarity_heatmap(
        emotion_distances,
        layer_idx=None,
        save_path=visualizer.output_dir / f"{model_name}_similarity_heatmap_avg.png"
    )
    
    print(f"\nAll plots saved to: {visualizer.output_dir}")


if __name__ == "__main__":
    main()
