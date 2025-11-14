"""
サブスペース次元kのスイープ実験
各感情ごとにPCAの次元数kを変えてoverlapを測定
"""
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from src.analysis.subspace_utils import (
    load_layer_residuals,
    compute_subspace_from_residuals,
    compare_subspaces
)


class SubspaceKSweep:
    """サブスペース次元kのスイープ実験クラス"""
    
    def __init__(self, activations_dir: Path):
        """
        初期化
        
        Args:
            activations_dir: 活性データディレクトリ
        """
        self.activations_dir = Path(activations_dir)
    
    def compute_k_sweep(
        self,
        model1_name: str,
        model2_name: str,
        emotion_labels: List[str] = ["gratitude", "anger", "apology"],
        k_values: List[int] = [2, 5, 10, 20],
        layers: Optional[List[int]] = None
    ) -> Dict:
        """
        k-sweep実験を実行
        
        Args:
            model1_name: Model1名（ファイル名のプレフィックス）
            model2_name: Model2名（ファイル名のプレフィックス）
            emotion_labels: 感情ラベルのリスト
            k_values: k値のリスト
            layers: 対象層のリスト（Noneの場合は全層を推測）
            
        Returns:
            実験結果の辞書
        """
        # 層数を推測（最初のファイルから）
        # ディレクトリ構造: activations_dir/{model_name}/activations_{emotion}.pkl
        sample_file = self.activations_dir / model1_name / f"activations_{emotion_labels[0]}.pkl"
        if not sample_file.exists():
            # 別のパターンも試す
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
            'k_values': k_values,
            'emotions': emotion_labels,
            'layers': layers,
            'sweep_results': {}
        }
        
        # 各層・各感情・各k値で実験
        for layer_idx in tqdm(layers, desc="Layers"):
            results['sweep_results'][layer_idx] = {}
            
            for emotion_label in emotion_labels:
                results['sweep_results'][layer_idx][emotion_label] = {}
                
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
                
                # 各k値でサブスペースを計算して比較
                for k in k_values:
                    # サブスペースを計算
                    model1_subspace, _ = compute_subspace_from_residuals(model1_residuals, k)
                    model2_subspace, _ = compute_subspace_from_residuals(model2_residuals, k)
                    
                    # 比較
                    comparison = compare_subspaces(model1_subspace, model2_subspace)
                    
                    results['sweep_results'][layer_idx][emotion_label][k] = {
                        'overlap_cos_squared': comparison['overlap_cos_squared'],
                        'overlap_principal_angles': comparison['overlap_principal_angles'],
                        'mean_principal_angle': comparison.get('mean_principal_angle', None)
                    }
        
        return results
    
    def aggregate_results(self, results: Dict) -> Dict:
        """
        結果を集計
        
        Args:
            results: sweep実験の結果
            
        Returns:
            集計結果の辞書
        """
        aggregated = {}
        
        for layer_idx in results['layers']:
            aggregated[layer_idx] = {}
            
            for emotion_label in results['emotions']:
                aggregated[layer_idx][emotion_label] = {}
                
                for k in results['k_values']:
                    if k in results['sweep_results'][layer_idx][emotion_label]:
                        data = results['sweep_results'][layer_idx][emotion_label][k]
                        aggregated[layer_idx][emotion_label][k] = data
        
        return aggregated


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Subspace k-sweep experiment")
    parser.add_argument("--activations_dir", type=str, required=True, help="Activations directory")
    parser.add_argument("--model1", type=str, required=True, help="Model1 name (file prefix)")
    parser.add_argument("--model2", type=str, required=True, help="Model2 name (file prefix)")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--k-values", type=int, nargs='+', default=[2, 5, 10, 20], help="k values to sweep")
    parser.add_argument("--layers", type=int, nargs='+', default=None, help="Layer indices (default: all)")
    parser.add_argument("--emotions", type=str, nargs='+', default=["gratitude", "anger", "apology"], help="Emotion labels")
    
    args = parser.parse_args()
    
    # k-sweep実験を実行
    sweeper = SubspaceKSweep(Path(args.activations_dir))
    
    results = sweeper.compute_k_sweep(
        args.model1,
        args.model2,
        emotion_labels=args.emotions,
        k_values=args.k_values,
        layers=args.layers
    )
    
    # 結果を保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # JSON形式で保存（NumPy配列をリストに変換）
    def convert_to_json_serializable(obj):
        """NumPy配列やその他の型をJSONシリアライズ可能な形式に変換"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)
    
    json_results = convert_to_json_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Pickle形式でも保存
    with open(output_path.with_suffix('.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to:")
    print(f"  JSON: {output_path}")
    print(f"  Pickle: {output_path.with_suffix('.pkl')}")
    
    # サマリーを表示
    print("\n" + "=" * 80)
    print("K-Sweep Summary")
    print("=" * 80)
    print(f"Model1: {args.model1}")
    print(f"Model2: {args.model2}")
    print(f"K values: {args.k_values}")
    print(f"Layers: {results['layers']}")
    
    # 各層・各感情のk値ごとのoverlapを表示
    for layer_idx in results['layers'][:3]:  # 最初の3層のみ表示
        print(f"\nLayer {layer_idx}:")
        for emotion_label in args.emotions:
            if emotion_label in results['sweep_results'][layer_idx]:
                print(f"  {emotion_label}:")
                for k in args.k_values:
                    if k in results['sweep_results'][layer_idx][emotion_label]:
                        overlap = results['sweep_results'][layer_idx][emotion_label][k]['overlap_cos_squared']
                        print(f"    k={k:2d}: overlap={overlap:.4f}")


if __name__ == "__main__":
    main()

