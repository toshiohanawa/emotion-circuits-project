"""
モデル間アライメント: Neutral空間で線形写像を学習し、Emotionで検証
"""
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from transformer_lens import HookedTransformer
from tqdm import tqdm

from src.analysis.subspace_utils import (
    load_layer_residuals,
    compute_subspace_from_residuals,
    compare_subspaces,
    align_and_compare
)


class ModelAlignment:
    """モデル間の線形写像を学習し、感情サブスペースのアライメントを検証"""
    
    def __init__(self, model1_name: str, model2_name: str, device: Optional[str] = None):
        """
        初期化
        
        Args:
            model1_name: ソースモデル名（例: "gpt2"）
            model2_name: ターゲットモデル名（例: "EleutherAI/pythia-160m"）
            device: 使用するデバイス
        """
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading models: {model1_name} and {model2_name}")
        self.model1 = HookedTransformer.from_pretrained(model1_name, device=self.device)
        self.model1.eval()
        self.model2 = HookedTransformer.from_pretrained(model2_name, device=self.device)
        self.model2.eval()
        
        print(f"✓ Models loaded on {self.device}")
    
    def extract_neutral_residuals(
        self,
        prompts: List[str],
        activations_dir: Optional[Path] = None,
        max_samples: Optional[int] = None
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        中立プロンプトからresidualを抽出
        
        Args:
            prompts: 中立プロンプトのリスト
            activations_dir: 活性データディレクトリ（Noneの場合は新規抽出）
            max_samples: 最大サンプル数（Noneの場合は全て）
            
        Returns:
            (model1_residuals, model2_residuals)
            - model1_residuals: {layer_idx: residuals [n_samples, d_model]}
            - model2_residuals: {layer_idx: residuals [n_samples, d_model]}
        """
        if max_samples:
            prompts = prompts[:max_samples]
        
        n_layers = self.model1.cfg.n_layers
        model1_residuals = {i: [] for i in range(n_layers)}
        model2_residuals = {i: [] for i in range(n_layers)}
        
        def save_residual1(activation, hook):
            """Model1のresidualを保存"""
            layer_idx = int(hook.name.split('.')[1])
            model1_residuals[layer_idx].append(activation[0, -1, :].detach().cpu().numpy())
            return activation
        
        def save_residual2(activation, hook):
            """Model2のresidualを保存"""
            layer_idx = int(hook.name.split('.')[1])
            model2_residuals[layer_idx].append(activation[0, -1, :].detach().cpu().numpy())
            return activation
        
        # Hookを登録
        hook_names1 = [f"blocks.{i}.hook_resid_pre" for i in range(n_layers)]
        hook_names2 = [f"blocks.{i}.hook_resid_pre" for i in range(n_layers)]
        
        for hook_name in hook_names1:
            self.model1.add_hook(hook_name, save_residual1)
        for hook_name in hook_names2:
            self.model2.add_hook(hook_name, save_residual2)
        
        # 推論実行
        with torch.no_grad():
            for prompt in tqdm(prompts, desc="Extracting neutral residuals"):
                tokens1 = self.model1.to_tokens(prompt)
                tokens2 = self.model2.to_tokens(prompt)
                
                _ = self.model1(tokens1)
                _ = self.model2(tokens2)
        
        # Hookを削除
        for hook_name in hook_names1:
            if hook_name in self.model1.hook_dict:
                hook_point = self.model1.hook_dict[hook_name]
                hook_point.fwd_hooks = []
        for hook_name in hook_names2:
            if hook_name in self.model2.hook_dict:
                hook_point = self.model2.hook_dict[hook_name]
                hook_point.fwd_hooks = []
        
        # リストをnumpy配列に変換
        model1_residuals_array = {
            layer_idx: np.array(residuals) for layer_idx, residuals in model1_residuals.items()
        }
        model2_residuals_array = {
            layer_idx: np.array(residuals) for layer_idx, residuals in model2_residuals.items()
        }
        
        return model1_residuals_array, model2_residuals_array
    
    def learn_linear_mapping(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> np.ndarray:
        """
        線形写像Wを学習: Y ≈ WX（最小二乗）
        
        Args:
            X: ソースresidual [n_samples, d1]
            Y: ターゲットresidual [n_samples, d2]
            
        Returns:
            線形写像W [d2, d1]
        """
        # 最小二乗解: W = Y^T X (X^T X)^(-1)
        XTX = X.T @ X
        XTX_inv = np.linalg.pinv(XTX)  # 疑似逆行列を使用
        W = Y.T @ X @ XTX_inv
        
        return W
    
    def align_emotion_subspaces(
        self,
        model1_subspace: np.ndarray,
        model2_subspace: np.ndarray,
        linear_map: np.ndarray
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        感情サブスペースを線形写像で変換して比較
        
        Args:
            model1_subspace: Model1の感情サブスペース [n_components, d1]
            model2_subspace: Model2の感情サブスペース [n_components, d2]
            linear_map: 線形写像W [d2, d1]
            
        Returns:
            比較結果の辞書
        """
        # Model1のサブスペースをModel2の空間に写像
        mapped_subspace = (linear_map @ model1_subspace.T).T  # [n_components, d2]
        
        # 写像前の比較
        before_comparison = compare_subspaces(model1_subspace, model2_subspace)
        
        # 写像後の比較
        after_comparison = compare_subspaces(mapped_subspace, model2_subspace)
        
        return {
            'before': before_comparison,
            'after': after_comparison,
            'mapped_subspace': mapped_subspace,
            'improvement': {
                'overlap_cos_squared': after_comparison['overlap_cos_squared'] - before_comparison['overlap_cos_squared'],
                'overlap_principal_angles': after_comparison['overlap_principal_angles'] - before_comparison['overlap_principal_angles']
            }
        }
    
    def run_alignment_experiment(
        self,
        neutral_prompts: List[str],
        model1_activations_dir: Path,
        model2_activations_dir: Path,
        emotion_labels: List[str] = ["gratitude", "anger", "apology"],
        n_components: int = 10,
        layers: Optional[List[int]] = None
    ) -> Dict:
        """
        アライメント実験を実行
        
        Args:
            neutral_prompts: 中立プロンプトのリスト
            model1_activations_dir: Model1の活性データディレクトリ
            model2_activations_dir: Model2の活性データディレクトリ
            emotion_labels: 感情ラベルのリスト
            n_components: PCAの主成分数
            layers: 対象層のリスト（Noneの場合は全層）
            
        Returns:
            実験結果の辞書
        """
        n_layers = self.model1.cfg.n_layers
        if layers is None:
            layers = list(range(n_layers))
        
        results = {
            'model1': self.model1_name,
            'model2': self.model2_name,
            'n_components': n_components,
            'layers': layers,
            'emotions': emotion_labels,
            'linear_maps': {},
            'alignment_results': {}
        }
        
        # Neutral residualを抽出（または読み込み）
        print("Extracting/loading neutral residuals...")
        model1_neutral_residuals, model2_neutral_residuals = self.extract_neutral_residuals(
            neutral_prompts,
            max_samples=len(neutral_prompts)
        )
        
        # 各層で線形写像を学習
        print("Learning linear mappings...")
        for layer_idx in tqdm(layers, desc="Layers"):
            X = model1_neutral_residuals[layer_idx]  # [n_samples, d1]
            Y = model2_neutral_residuals[layer_idx]  # [n_samples, d2]
            
            # 線形写像を学習
            W = self.learn_linear_mapping(X, Y)
            
            # 保存
            results['linear_maps'][layer_idx] = W
            
            # 各感情のサブスペースを読み込んで比較
            results['alignment_results'][layer_idx] = {}
            
            for emotion_label in emotion_labels:
                # Model1の感情サブスペースを読み込み
                model1_activations_file = model1_activations_dir / f"activations_{emotion_label}.pkl"
                model2_activations_file = model2_activations_dir / f"activations_{emotion_label}.pkl"
                
                if not model1_activations_file.exists() or not model2_activations_file.exists():
                    print(f"Warning: Activation files not found for {emotion_label}, skipping...")
                    continue
                
                # Residualを読み込み
                model1_residuals = load_layer_residuals(model1_activations_file, layer_idx, position="last")
                model2_residuals = load_layer_residuals(model2_activations_file, layer_idx, position="last")
                
                # サブスペースを計算
                model1_subspace, _ = compute_subspace_from_residuals(model1_residuals, n_components)
                model2_subspace, _ = compute_subspace_from_residuals(model2_residuals, n_components)
                
                # アライメント実験
                alignment_result = self.align_emotion_subspaces(
                    model1_subspace,
                    model2_subspace,
                    W
                )
                
                results['alignment_results'][layer_idx][emotion_label] = alignment_result
        
        return results


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model alignment experiment")
    parser.add_argument("--model1", type=str, required=True, help="Source model name")
    parser.add_argument("--model2", type=str, required=True, help="Target model name")
    parser.add_argument("--neutral_prompts_file", type=str, required=True, help="Neutral prompts file (JSON)")
    parser.add_argument("--model1_activations_dir", type=str, required=True, help="Model1 activations directory")
    parser.add_argument("--model2_activations_dir", type=str, required=True, help="Model2 activations directory")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--n-components", type=int, default=10, help="Number of PCA components")
    parser.add_argument("--layers", type=int, nargs='+', default=None, help="Layer indices (default: all)")
    parser.add_argument("--emotions", type=str, nargs='+', default=["gratitude", "anger", "apology"], help="Emotion labels")
    
    args = parser.parse_args()
    
    # プロンプトを読み込み
    with open(args.neutral_prompts_file, 'r') as f:
        prompts_data = json.load(f)
        neutral_prompts = prompts_data.get('prompts', [])
    
    print(f"Using {len(neutral_prompts)} neutral prompts")
    
    # アライメント実験を実行
    aligner = ModelAlignment(args.model1, args.model2)
    
    results = aligner.run_alignment_experiment(
        neutral_prompts,
        Path(args.model1_activations_dir),
        Path(args.model2_activations_dir),
        emotion_labels=args.emotions,
        n_components=args.n_components,
        layers=args.layers
    )
    
    # 結果を保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Pickle形式で保存
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    # 線形写像も個別に保存
    linear_maps_dir = output_path.parent / "linear_maps"
    linear_maps_dir.mkdir(parents=True, exist_ok=True)
    
    for layer_idx, W in results['linear_maps'].items():
        map_file = linear_maps_dir / f"layer_{layer_idx}.pt"
        torch.save(torch.tensor(W), map_file)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Linear maps saved to: {linear_maps_dir}")
    
    # サマリーを表示
    print("\n" + "=" * 80)
    print("Alignment Experiment Summary")
    print("=" * 80)
    print(f"Model1: {args.model1}")
    print(f"Model2: {args.model2}")
    print(f"Layers: {results['layers']}")
    print(f"Emotions: {results['emotions']}")
    
    # 各層・各感情の改善を表示
    for layer_idx in results['layers']:
        print(f"\nLayer {layer_idx}:")
        for emotion_label in args.emotions:
            if emotion_label in results['alignment_results'][layer_idx]:
                result = results['alignment_results'][layer_idx][emotion_label]
                improvement = result['improvement']
                print(f"  {emotion_label}:")
                print(f"    Overlap (cos^2) improvement: {improvement['overlap_cos_squared']:+.4f}")
                print(f"    Overlap (principal angles) improvement: {improvement['overlap_principal_angles']:+.4f}")


if __name__ == "__main__":
    main()

