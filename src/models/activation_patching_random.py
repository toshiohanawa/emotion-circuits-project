"""
Activation Patching with Random Control Vectors
感情方向ベクトルと同じL2ノルムを持つランダムベクトルを生成し、パッチング実験を実行
"""
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import mlflow

from src.models.activation_patching_sweep import ActivationPatchingSweep


class RandomControlPatcher(ActivationPatchingSweep):
    """ランダム対照ベクトルでパッチングを実行するクラス"""
    
    def generate_random_vector(self, reference_vector: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        参照ベクトルと同じL2ノルムを持つランダムベクトルを生成
        
        Args:
            reference_vector: 参照ベクトル [d_model]
            seed: ランダムシード（Noneの場合は現在時刻を使用）
            
        Returns:
            同じL2ノルムを持つランダムベクトル [d_model]
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 参照ベクトルのL2ノルムを計算
        reference_norm = np.linalg.norm(reference_vector)
        
        # ランダムベクトルを生成（標準正規分布）
        random_vector = np.random.randn(*reference_vector.shape)
        
        # L2ノルムを正規化して参照ベクトルと同じにする
        random_norm = np.linalg.norm(random_vector)
        if random_norm > 0:
            random_vector = random_vector * (reference_norm / random_norm)
        
        return random_vector
    
    def run_random_control_sweep(
        self,
        prompts: List[str],
        emotion_vectors: Dict[str, np.ndarray],
        layers: List[int],
        alpha_values: List[float],
        num_random: int = 100,
        mlflow_run: bool = False,
    ) -> Dict:
        """
        ランダム対照ベクトルでスイープ実験を実行
        
        Args:
            prompts: 入力プロンプトのリスト
            emotion_vectors: 感情ラベルをキーとする方向ベクトル [n_layers, d_model]
            layers: パッチを適用する層のリスト
            alpha_values: パッチの強度のリスト
            num_random: 生成するランダムベクトルの数
            
        Returns:
            実験結果の辞書
        """
        results = {
            'model': self.model_name,
            'prompts': prompts,
            'layers': layers,
            'alpha_values': alpha_values,
            'num_random': num_random,
            'random_results': {},
            'emotion_results': {}
        }
        if mlflow_run:
            mlflow.log_params({
                "phase": "random_control",
                "model": self.model_name,
                "num_random": num_random,
                "layers": layers,
                "alpha_values": alpha_values,
                "prompts": len(prompts),
            })
        
        # 各感情方向でランダム対照実験
        for emotion_label, emotion_vec in emotion_vectors.items():
            print(f"\n{'='*80}")
            print(f"Processing {emotion_label.upper()}")
            print(f"{'='*80}")
            
            results['random_results'][emotion_label] = {}
            results['emotion_results'][emotion_label] = {}
            
            for layer_idx in layers:
                layer_vector = emotion_vec[layer_idx]  # [d_model]
                reference_norm = np.linalg.norm(layer_vector)
                
                results['random_results'][emotion_label][layer_idx] = {}
                results['emotion_results'][emotion_label][layer_idx] = {}
                
                # ランダムベクトルでスイープ
                print(f"\nLayer {layer_idx}: Random control (norm={reference_norm:.4f})")
                for rand_idx in range(num_random):
                    random_vector = self.generate_random_vector(layer_vector, seed=rand_idx)
                    results['random_results'][emotion_label][layer_idx][rand_idx] = {}
                    
                    for alpha in tqdm(alpha_values, desc=f"Random {rand_idx+1}/{num_random}"):
                        sweep_results = []
                        for prompt in prompts:
                            try:
                                generated = self.generate_with_patching(
                                    prompt,
                                    random_vector,
                                    layer_idx,
                                    alpha,
                                    max_new_tokens=20
                                )
                                metrics = self.metric_evaluator.evaluate_text_metrics(generated)
                                sweep_results.append({
                                    'prompt': prompt,
                                    'generated': generated,
                                    'metrics': metrics
                                })
                            except Exception as e:
                                print(f"Error: {e}")
                                sweep_results.append({
                                    'prompt': prompt,
                                    'generated': f"ERROR: {str(e)}",
                                    'metrics': {}
                                })
                        
                        results['random_results'][emotion_label][layer_idx][rand_idx][alpha] = sweep_results
                
                # 感情方向ベクトルでスイープ（比較用）
                print(f"\nLayer {layer_idx}: Emotion vector")
                for alpha in tqdm(alpha_values, desc="Emotion"):
                    sweep_results = []
                    for prompt in prompts:
                        try:
                            generated = self.generate_with_patching(
                                prompt,
                                layer_vector,
                                layer_idx,
                                alpha,
                                max_new_tokens=20
                            )
                            metrics = self.metric_evaluator.evaluate_text_metrics(generated)
                            sweep_results.append({
                                'prompt': prompt,
                                'generated': generated,
                                'metrics': metrics
                            })
                        except Exception as e:
                            print(f"Error: {e}")
                            sweep_results.append({
                                'prompt': prompt,
                                'generated': f"ERROR: {str(e)}",
                                'metrics': {}
                            })
                    
                    results['emotion_results'][emotion_label][layer_idx][alpha] = sweep_results
        
        return results


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Activation patching with random control")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--vectors_file", type=str, required=True, help="Emotion vectors file")
    parser.add_argument("--prompts_file", type=str, default="data/neutral_prompts.json", help="Prompts file (JSON)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--layers", type=int, nargs='+', default=[3, 5, 7, 9, 11], help="Layer indices")
    parser.add_argument("--alpha", type=float, nargs='+', default=[-2, -1, -0.5, 0, 0.5, 1, 2], help="Alpha values")
    parser.add_argument("--num_random", type=int, default=100, help="Number of random vectors")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging for this run")
    
    args = parser.parse_args()
    
    # パッチャーを作成
    patcher = RandomControlPatcher(args.model)
    
    # 感情ベクトルを読み込み
    emotion_vectors = patcher.load_emotion_vectors(Path(args.vectors_file))
    print(f"Loaded emotion vectors: {list(emotion_vectors.keys())}")
    
    # プロンプトを読み込み
    with open(args.prompts_file, 'r') as f:
        prompts_data = json.load(f)
        prompts = prompts_data.get('prompts', [])
    
    print(f"Using {len(prompts)} prompts")
    
    # ランダム対照スイープ実験を実行
    run_ctx = mlflow.start_run() if args.mlflow else None
    try:
        results = patcher.run_random_control_sweep(
            prompts,
            emotion_vectors,
            layers=args.layers,
            alpha_values=args.alpha,
            num_random=args.num_random,
            mlflow_run=args.mlflow,
        )
    finally:
        if run_ctx:
            mlflow.end_run()
    
    # 結果を保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 各感情×各層×各ランダムベクトルで保存
    for emotion_label in emotion_vectors.keys():
        for layer_idx in args.layers:
            for rand_idx in range(args.num_random):
                for alpha in args.alpha:
                    output_file = output_dir / f"{args.model.replace('/', '-')}_{emotion_label}_alpha{alpha}_rand{rand_idx}.pkl"
                    # 簡易版：全結果を1ファイルに保存
                    break
                break
            break
        break
    
    # 全結果を1ファイルに保存
    output_file = output_dir / f"{args.model.replace('/', '-')}_random_control.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
