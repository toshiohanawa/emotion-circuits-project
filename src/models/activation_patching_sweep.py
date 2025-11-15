"""
Activation Patching Sweep Experiment
層×αのスイープ実験を実行し、感情トーンへの影響を評価
"""
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from transformer_lens import HookedTransformer
from tqdm import tqdm
import mlflow

from src.analysis.sentiment_eval import SentimentEvaluator


class ActivationPatchingSweep:
    """層×αのスイープ実験を実行するクラス"""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        初期化
        
        Args:
            model_name: モデル名
            device: 使用するデバイス
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model: {model_name}")
        self.model = HookedTransformer.from_pretrained(model_name, device=self.device)
        self.model.eval()
        # Deterministic generation settings – we rely on repeatability when
        # comparing alpha sweeps.
        self.generation_config = {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": None,
            "stop_at_eos": True,
            "return_type": "tokens",
        }
        
        self.metric_evaluator = SentimentEvaluator(
            model_name=model_name,
            device=self.device,
            load_generation_model=False,
            enable_transformer_metrics=True,
        )
        print(f"✓ Model loaded on {self.device}")
    
    def load_emotion_vectors(self, vectors_file: Path) -> Dict[str, np.ndarray]:
        """
        感情方向ベクトルを読み込む
        
        Args:
            vectors_file: ベクトルファイルのパス
            
        Returns:
            感情ラベルをキーとする方向ベクトルの辞書 [n_layers, d_model]
        """
        with open(vectors_file, 'rb') as f:
            data = pickle.load(f)
        
        return data['emotion_vectors']

    def _decode_new_tokens(self, all_tokens: torch.Tensor, prompt_length: int) -> str:
        """
        Decode only the newly generated continuations (excluding the prompt).
        """
        new_tokens = all_tokens[:, prompt_length:]
        if new_tokens.shape[1] == 0:
            return ""
        token_ids = new_tokens[0].tolist()
        decoded = self.model.tokenizer.decode(token_ids, skip_special_tokens=True)
        return decoded.strip()

    def _generate_tokens(
        self,
        tokens: torch.Tensor,
        hook_name: Optional[str] = None,
        hook_fn: Optional[Callable] = None,
        max_new_tokens: int = 20,
    ) -> torch.Tensor:
        """
        Helper that optionally attaches a hook, runs generation, and returns tokens.
        """
        handles = []
        if hook_name and hook_fn:
            handle = self.model.add_hook(hook_name, hook_fn)
            handles.append((hook_name, handle))
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    tokens,
                    max_new_tokens=max_new_tokens,
                    **self.generation_config,
                )
        finally:
            for hook_name, handle in handles:
                if hook_name in self.model.hook_dict:
                    hook_point = self.model.hook_dict[hook_name]
                    hook_point.fwd_hooks = []
        return generated

    def _generate_text(
        self,
        prompt: str,
        hook_name: Optional[str] = None,
        hook_fn: Optional[Callable] = None,
        max_new_tokens: int = 20,
    ) -> str:
        """
        Generate text optionally under a hooked residual modification.
        """
        tokens = self.model.to_tokens(prompt)
        generated = self._generate_tokens(
            tokens,
            hook_name=hook_name,
            hook_fn=hook_fn,
            max_new_tokens=max_new_tokens,
        )
        return self._decode_new_tokens(generated, tokens.shape[1])
    
    def generate_with_patching(
        self,
        prompt: str,
        emotion_vector: np.ndarray,
        layer_idx: int,
        alpha: float = 1.0,
        max_new_tokens: int = 20
    ) -> str:
        """
        パッチを適用してテキストを生成
        
        Args:
            prompt: 入力プロンプト
            emotion_vector: 感情方向ベクトル [d_model]（特定の層のベクトル）
            layer_idx: パッチを適用する層
            alpha: パッチの強度
            max_new_tokens: 生成する最大トークン数
            
        Returns:
            生成されたテキスト
        """
        tokens = self.model.to_tokens(prompt)
        prompt_length = tokens.shape[1]
        target_position = max(prompt_length - 1, 0)
        patch_vector = torch.tensor(
            emotion_vector,
            device=self.device,
            dtype=self.model.cfg.dtype,
        )

        def patch_hook(activation, hook):
            """Residual streamをパッチするhook"""
            if activation.shape[1] == 0:
                return activation
            activation = activation.clone()
            pos = min(target_position, activation.shape[1] - 1)
            activation[0, pos, :] += alpha * patch_vector.to(activation.dtype)
            return activation

        generated = self._generate_tokens(
            tokens,
            hook_name=f"blocks.{layer_idx}.hook_resid_pre",
            hook_fn=patch_hook,
            max_new_tokens=max_new_tokens,
        )
        return self._decode_new_tokens(generated, prompt_length)
    
    def _mean_nested_metrics(self, metric_list: List[Dict]) -> Dict:
        """Average nested metric dictionaries recursively."""
        if not metric_list:
            return {}
        keys = set().union(*(metric.keys() for metric in metric_list))
        aggregated: Dict[str, Dict] = {}
        for key in keys:
            values = [metric[key] for metric in metric_list if key in metric]
            if not values:
                continue
            if isinstance(values[0], dict):
                aggregated[key] = self._mean_nested_metrics(values)
            else:
                aggregated[key] = float(np.mean(values))
        return aggregated

    def _subtract_metric_dicts(self, metrics: Dict, baseline: Dict) -> Dict:
        """Compute nested difference between two metric dicts."""
        result: Dict[str, Dict] = {}
        for key, value in metrics.items():
            base_value = baseline.get(key)
            if isinstance(value, dict) and isinstance(base_value, dict):
                result[key] = self._subtract_metric_dicts(value, base_value)
            elif base_value is not None:
                result[key] = float(value - base_value)
            else:
                result[key] = float(value) if not isinstance(value, dict) else value
        return result

    def _flatten_metrics(self, metrics: Dict, prefix: str = "") -> Dict[str, float]:
        """Flatten nested dictionaries for MLflow logging."""
        flattened: Dict[str, float] = {}
        for key, value in metrics.items():
            metric_name = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(self._flatten_metrics(value, metric_name))
            else:
                flattened[metric_name] = float(value)
        return flattened

    def _log_metrics_to_mlflow(self, prefix: str, metrics: Dict, step: Optional[int] = None) -> None:
        if not mlflow.active_run():
            return
        flattened = self._flatten_metrics(metrics, prefix)
        for name, val in flattened.items():
            mlflow.log_metric(name, val, step=step)
    
    def run_sweep(
        self,
        prompts: List[str],
        emotion_vectors: Dict[str, np.ndarray],
        layers: List[int] = [3, 5, 7, 9, 11],
        alpha_values: List[float] = [-2, -1, -0.5, 0, 0.5, 1, 2]
    ) -> Dict:
        """
        層×αのスイープ実験を実行
        
        Args:
            prompts: 入力プロンプトのリスト
            emotion_vectors: 感情ラベルをキーとする方向ベクトル [n_layers, d_model]
            layers: パッチを適用する層のリスト
            alpha_values: パッチの強度のリスト
            
        Returns:
            スイープ実験の結果
        """
        results = {
            'model': self.model_name,
            'prompts': prompts,
            'layers': layers,
            'alpha_values': alpha_values,
            'emotions': list(emotion_vectors.keys()),
            'sweep_results': {}
        }
        
        # Baseline（パッチなし）の生成と評価
        print("Generating baseline outputs...")
        baseline_outputs = {}
        baseline_metrics = {}
        
        baseline_metric_values = []
        for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Baseline")):
            generated_text = self._generate_text(
                prompt,
                max_new_tokens=20,
            )
            baseline_outputs[prompt] = generated_text
            metrics = self.metric_evaluator.evaluate_text_metrics(generated_text)
            baseline_metrics[prompt] = metrics
            baseline_metric_values.append(baseline_metrics[prompt])
            self._log_metrics_to_mlflow(f"baseline/prompt_{prompt_idx}", metrics, step=prompt_idx)
        
        results['baseline'] = {
            'outputs': baseline_outputs,
            'metrics': baseline_metrics
        }
        results['baseline_summary'] = self._mean_nested_metrics(baseline_metric_values)
        
        # 各感情方向でスイープ実験
        for emotion_label, emotion_vec in emotion_vectors.items():
            print(f"\n{'='*80}")
            print(f"Running sweep for {emotion_label.upper()}")
            print(f"{'='*80}")
            
            results['sweep_results'][emotion_label] = {}
            
            for layer_idx in layers:
                print(f"\nLayer {layer_idx}:")
                layer_vector = emotion_vec[layer_idx]  # [d_model]
                
                results['sweep_results'][emotion_label][layer_idx] = {}
                
                for alpha_idx, alpha in enumerate(alpha_values):
                    print(f"  α={alpha:4.1f}...", end=" ", flush=True)
                    
                    layer_alpha_results = {
                        'outputs': {},
                        'metrics': {}
                    }
                    
                    for prompt_idx, prompt in enumerate(prompts):
                        try:
                            generated_text = self.generate_with_patching(
                                prompt,
                                layer_vector,
                                layer_idx,
                                alpha,
                                max_new_tokens=20
                            )
                            
                            layer_alpha_results['outputs'][prompt] = generated_text
                            metrics = self.metric_evaluator.evaluate_text_metrics(generated_text)
                            layer_alpha_results['metrics'][prompt] = metrics
                            log_prefix = f"{emotion_label}/layer_{layer_idx}/alpha_{alpha}/prompt_{prompt_idx}"
                            self._log_metrics_to_mlflow(log_prefix, metrics, step=alpha_idx)
                        except Exception as e:
                            print(f"\n    Error with prompt '{prompt}': {e}")
                            layer_alpha_results['outputs'][prompt] = f"ERROR: {str(e)}"
                            layer_alpha_results['metrics'][prompt] = {}
                    
                    results['sweep_results'][emotion_label][layer_idx][alpha] = layer_alpha_results
                    print("✓")
        
        return results
    
    def aggregate_metrics(self, results: Dict) -> Tuple[Dict, Dict]:
        """
        メトリクスを集計してヒートマップ用とベースライン差分用のデータを作成
        """
        aggregated = {}
        delta_metrics = {}
        baseline_summary = results.get('baseline_summary') or {}
        
        for emotion_label in results['emotions']:
            aggregated[emotion_label] = {}
            delta_metrics[emotion_label] = {}
            
            for layer_idx in results['layers']:
                aggregated[emotion_label][layer_idx] = {}
                delta_metrics[emotion_label][layer_idx] = {}
                
                for alpha in results['alpha_values']:
                    sweep_layer = results['sweep_results'][emotion_label].get(layer_idx, {})
                    if alpha not in sweep_layer:
                        continue
                    
                    metrics_list = [m for m in sweep_layer[alpha]['metrics'].values() if m]
                    avg_metrics = self._mean_nested_metrics(metrics_list)
                    aggregated[emotion_label][layer_idx][alpha] = avg_metrics
                    delta_metrics[emotion_label][layer_idx][alpha] = self._subtract_metric_dicts(
                        avg_metrics,
                        baseline_summary
                    )
        
        return aggregated, delta_metrics


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Activation patching sweep experiment")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--vectors_file", type=str, required=True, help="Emotion vectors file")
    parser.add_argument("--prompts_file", type=str, default="data/neutral_prompts.json", help="Prompts file (JSON)")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--layers", type=int, nargs='+', default=[3, 5, 7, 9, 11], help="Layer indices")
    parser.add_argument("--alpha", type=float, nargs='+', default=[-2, -1, -0.5, 0, 0.5, 1, 2], help="Alpha values")
    
    args = parser.parse_args()
    
    # パッチャーを作成
    patcher = ActivationPatchingSweep(args.model)
    
    # 感情ベクトルを読み込み
    emotion_vectors = patcher.load_emotion_vectors(Path(args.vectors_file))
    print(f"Loaded emotion vectors: {list(emotion_vectors.keys())}")
    
    # プロンプトを読み込み
    with open(args.prompts_file, 'r') as f:
        prompts_data = json.load(f)
        prompts = prompts_data.get('prompts', [])
    
    print(f"Using {len(prompts)} prompts")
    
    # スイープ実験を実行
    results = patcher.run_sweep(
        prompts,
        emotion_vectors,
        layers=args.layers,
        alpha_values=args.alpha
    )
    
    # メトリクスを集計
    aggregated, delta_metrics = patcher.aggregate_metrics(results)
    results['aggregated_metrics'] = aggregated
    results['delta_metrics'] = delta_metrics
    
    # 結果を保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
