"""
サブスペースアライメント: CCA / Procrustesアライメント
PCA後のサブスペースに対し、CCAまたはProcrustesアライメントを適用
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from src.analysis.subspace_utils import (
    load_layer_residuals,
    compute_subspace_from_residuals,
    procrustes_alignment,
    cca_alignment,
    compare_subspaces
)


class SubspaceAlignment:
    """サブスペースアライメントクラス"""
    
    def __init__(self, activations_dir: Path):
        """
        初期化
        
        Args:
            activations_dir: 活性データディレクトリ
        """
        self.activations_dir = Path(activations_dir)
    
    def align_subspaces(
        self,
        model1_name: str,
        model2_name: str,
        emotion_labels: List[str] = ["gratitude", "anger", "apology"],
        n_components: int = 10,
        alignment_method: str = "procrustes",
        layers: Optional[List[int]] = None
    ) -> Dict:
        """
        サブスペースアライメント実験を実行
        
        Args:
            model1_name: Model1名（ファイル名のプレフィックス）
            model2_name: Model2名（ファイル名のプレフィックス）
            emotion_labels: 感情ラベルのリスト
            n_components: PCAの主成分数
            alignment_method: アライメント方法（"procrustes" または "cca"）
            layers: 対象層のリスト（Noneの場合は全層を推測）
            
        Returns:
            実験結果の辞書
        """
        # 層数を推測
        # ディレクトリ構造: activations_dir/{model_name}/activations_{emotion}.pkl
        sample_file = self.activations_dir / model1_name / f"activations_{emotion_labels[0]}.pkl"
        if not sample_file.exists():
            sample_file = self.activations_dir / f"{model1_name}_activations_{emotion_labels[0]}.pkl"
        if not sample_file.exists():
            sample_file = self.activations_dir / f"activations_{emotion_labels[0]}.pkl"
        
        if sample_file.exists():
            with open(sample_file, 'rb') as f:
                sample_data = pickle.load(f)
            n_layers = len(sample_data['residual_stream'])
        else:
            raise FileNotFoundError(f"Cannot find sample activation file to determine number of layers: {sample_file}")
        
        if layers is None:
            layers = list(range(n_layers))
        
        results = {
            'model1': model1_name,
            'model2': model2_name,
            'n_components': n_components,
            'alignment_method': alignment_method,
            'emotions': emotion_labels,
            'layers': layers,
            'alignment_results': {}
        }
        
        # 各層・各感情でアライメント実験
        for layer_idx in layers:
            results['alignment_results'][layer_idx] = {}
            
            for emotion_label in emotion_labels:
                # Model1とModel2の活性ファイルを探す
                # ディレクトリ構造: activations_dir/{model_name}/activations_{emotion}.pkl
                model1_file = self.activations_dir / model1_name / f"activations_{emotion_label}.pkl"
                model2_file = self.activations_dir / model2_name / f"activations_{emotion_label}.pkl"
                
                # ファイル名のパターンが異なる場合を試す
                if not model1_file.exists():
                    model1_file = self.activations_dir / f"{model1_name}_activations_{emotion_label}.pkl"
                if not model2_file.exists():
                    model2_file = self.activations_dir / f"{model2_name}_activations_{emotion_label}.pkl"
                if not model1_file.exists():
                    model1_file = self.activations_dir / f"activations_{emotion_label}.pkl"
                if not model2_file.exists():
                    model2_file = self.activations_dir / f"activations_{emotion_label}.pkl"
                
                if not model1_file.exists() or not model2_file.exists():
                    print(f"Warning: Activation files not found for {emotion_label}, skipping...")
                    continue
                
                # Residualを読み込み
                model1_residuals = load_layer_residuals(model1_file, layer_idx, position="last")
                model2_residuals = load_layer_residuals(model2_file, layer_idx, position="last")
                
                # サブスペースを計算
                model1_subspace, _ = compute_subspace_from_residuals(model1_residuals, n_components)
                model2_subspace, _ = compute_subspace_from_residuals(model2_residuals, n_components)
                
                # アライメント前の比較
                before_comparison = compare_subspaces(model1_subspace, model2_subspace)
                
                # アライメント
                if alignment_method == "procrustes":
                    aligned_subspace1, rotation_matrix = procrustes_alignment(model1_subspace, model2_subspace)
                    alignment_info = {'rotation_matrix': rotation_matrix}
                elif alignment_method == "cca":
                    # CCAは異なる次元のデータに対して使用
                    # ここでは同じ次元を仮定
                    X_canonical, Y_canonical, correlations = cca_alignment(model1_subspace, model2_subspace)
                    aligned_subspace1 = X_canonical
                    alignment_info = {'canonical_correlations': correlations}
                else:
                    raise ValueError(f"Unknown alignment method: {alignment_method}")
                
                # アライメント後の比較
                after_comparison = compare_subspaces(aligned_subspace1, model2_subspace)
                
                results['alignment_results'][layer_idx][emotion_label] = {
                    'before': before_comparison,
                    'after': after_comparison,
                    'aligned_subspace1': aligned_subspace1,
                    'alignment_info': alignment_info,
                    'improvement': {
                        'overlap_cos_squared': after_comparison['overlap_cos_squared'] - before_comparison['overlap_cos_squared'],
                        'overlap_principal_angles': after_comparison['overlap_principal_angles'] - before_comparison['overlap_principal_angles']
                    }
                }
        
        return results


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Subspace alignment experiment")
    parser.add_argument("--activations_dir", type=str, required=True, help="Activations directory")
    parser.add_argument("--model1", type=str, required=True, help="Model1 name (file prefix)")
    parser.add_argument("--model2", type=str, required=True, help="Model2 name (file prefix)")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--n-components", type=int, default=10, help="Number of PCA components")
    parser.add_argument("--alignment-method", type=str, default="procrustes", choices=["procrustes", "cca"], help="Alignment method")
    parser.add_argument("--layers", type=int, nargs='+', default=None, help="Layer indices (default: all)")
    parser.add_argument("--emotions", type=str, nargs='+', default=["gratitude", "anger", "apology"], help="Emotion labels")
    
    args = parser.parse_args()
    
    # アライメント実験を実行
    aligner = SubspaceAlignment(Path(args.activations_dir))
    
    results = aligner.align_subspaces(
        args.model1,
        args.model2,
        emotion_labels=args.emotions,
        n_components=args.n_components,
        alignment_method=args.alignment_method,
        layers=args.layers
    )
    
    # 結果を保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Pickle形式で保存
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to: {output_path}")
    
    # サマリーを表示
    print("\n" + "=" * 80)
    print("Subspace Alignment Summary")
    print("=" * 80)
    print(f"Model1: {args.model1}")
    print(f"Model2: {args.model2}")
    print(f"Alignment method: {args.alignment_method}")
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

