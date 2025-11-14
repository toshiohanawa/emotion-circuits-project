"""
Activation Patching Sweep Experiment
層×αのスイープ実験を実行し、感情トーンへの影響を評価
"""
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformer_lens import HookedTransformer
from tqdm import tqdm
import re


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
        original_length = tokens.shape[1]
        
        patched_activations = {}
        
        def patch_hook(activation, hook):
            """Residual streamをパッチするhook"""
            if hook.name == f"blocks.{layer_idx}.hook_resid_pre":
                activation = activation.clone()
                # 最後の位置にパッチ
                activation[0, -1, :] += alpha * torch.tensor(emotion_vector, device=self.device, dtype=activation.dtype)
            return activation
        
        # Hookを登録
        hook_name = f"blocks.{layer_idx}.hook_resid_pre"
        hook_handle = self.model.add_hook(hook_name, patch_hook)
        
        # 推論実行
        with torch.no_grad():
            logits = self.model(tokens)
        
        # Hookを削除
        if hook_name in self.model.hook_dict:
            hook_point = self.model.hook_dict[hook_name]
            hook_point.fwd_hooks = []
        
        # 生成されたテキストを取得
        generated_tokens = self.model.to_str_tokens(logits.argmax(dim=-1)[0])
        generated_text = ' '.join(generated_tokens[original_length:])
        
        return generated_text
    
    def count_emotion_keywords(self, text: str) -> Dict[str, int]:
        """
        感情キーワードの頻度をカウント
        
        Args:
            text: テキスト
            
        Returns:
            各感情カテゴリのキーワード出現回数
        """
        text_lower = text.lower()
        
        gratitude_keywords = [
            "thank", "thanks", "grateful", "gratitude", "appreciate", 
            "appreciation", "thankful", "blessed"
        ]
        anger_keywords = [
            "angry", "anger", "frustrated", "frustration", "terrible", 
            "annoyed", "annoyance", "upset", "mad", "furious", "irritated"
        ]
        apology_keywords = [
            "sorry", "apologize", "apology", "apologies", "regret", 
            "regretful", "apologetic"
        ]
        
        counts = {
            'gratitude': sum(1 for kw in gratitude_keywords if kw in text_lower),
            'anger': sum(1 for kw in anger_keywords if kw in text_lower),
            'apology': sum(1 for kw in apology_keywords if kw in text_lower)
        }
        
        return counts
    
    def calculate_politeness_score(self, text: str) -> float:
        """
        丁寧さスコアを計算（proxy指標）
        
        Args:
            text: テキスト
            
        Returns:
            丁寧さスコア（0-1）
        """
        text_lower = text.lower()
        
        politeness_markers = [
            "please", "kindly", "thank you", "thanks", "sorry", 
            "appreciate", "grateful", "would", "could", "may"
        ]
        
        # マーカーの出現回数をカウント
        count = sum(1 for marker in politeness_markers if marker in text_lower)
        
        # 正規化（最大10回で1.0）
        score = min(count / 10.0, 1.0)
        
        return score
    
    def calculate_sentiment_score(self, text: str) -> float:
        """
        簡易的なsentiment scoreを計算（ポジ/ネガ）
        
        Args:
            text: テキスト
            
        Returns:
            sentiment score（-1から1、ポジティブが正）
        """
        text_lower = text.lower()
        
        positive_words = [
            "good", "great", "excellent", "wonderful", "amazing", "fantastic",
            "happy", "pleased", "delighted", "satisfied", "positive", "nice"
        ]
        negative_words = [
            "bad", "terrible", "awful", "horrible", "disappointed", "frustrated",
            "angry", "upset", "negative", "unhappy", "sad", "worried"
        ]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        # 正規化（最大10回で±1）
        if pos_count + neg_count == 0:
            return 0.0
        
        score = (pos_count - neg_count) / max(pos_count + neg_count, 10.0)
        return np.clip(score, -1.0, 1.0)
    
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
        
        for prompt in tqdm(prompts, desc="Baseline"):
            tokens = self.model.to_tokens(prompt)
            with torch.no_grad():
                logits = self.model(tokens)
            generated_tokens = self.model.to_str_tokens(logits.argmax(dim=-1)[0])
            generated_text = ' '.join(generated_tokens[tokens.shape[1]:])
            
            baseline_outputs[prompt] = generated_text
            baseline_metrics[prompt] = {
                'emotion_keywords': self.count_emotion_keywords(generated_text),
                'politeness': self.calculate_politeness_score(generated_text),
                'sentiment': self.calculate_sentiment_score(generated_text)
            }
        
        results['baseline'] = {
            'outputs': baseline_outputs,
            'metrics': baseline_metrics
        }
        
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
                
                for alpha in alpha_values:
                    print(f"  α={alpha:4.1f}...", end=" ", flush=True)
                    
                    layer_alpha_results = {
                        'outputs': {},
                        'metrics': {}
                    }
                    
                    for prompt in prompts:
                        try:
                            generated_text = self.generate_with_patching(
                                prompt,
                                layer_vector,
                                layer_idx,
                                alpha,
                                max_new_tokens=20
                            )
                            
                            layer_alpha_results['outputs'][prompt] = generated_text
                            layer_alpha_results['metrics'][prompt] = {
                                'emotion_keywords': self.count_emotion_keywords(generated_text),
                                'politeness': self.calculate_politeness_score(generated_text),
                                'sentiment': self.calculate_sentiment_score(generated_text)
                            }
                        except Exception as e:
                            print(f"\n    Error with prompt '{prompt}': {e}")
                            layer_alpha_results['outputs'][prompt] = f"ERROR: {str(e)}"
                            layer_alpha_results['metrics'][prompt] = {
                                'emotion_keywords': {'gratitude': 0, 'anger': 0, 'apology': 0},
                                'politeness': 0.0,
                                'sentiment': 0.0
                            }
                    
                    results['sweep_results'][emotion_label][layer_idx][alpha] = layer_alpha_results
                    print("✓")
        
        return results
    
    def aggregate_metrics(self, results: Dict) -> Dict:
        """
        メトリクスを集計してヒートマップ用のデータを作成
        
        Args:
            results: スイープ実験の結果
            
        Returns:
            集計されたメトリクス
        """
        aggregated = {}
        
        for emotion_label in results['emotions']:
            aggregated[emotion_label] = {}
            
            for layer_idx in results['layers']:
                aggregated[emotion_label][layer_idx] = {}
                
                for alpha in results['alpha_values']:
                    if alpha not in results['sweep_results'][emotion_label][layer_idx]:
                        continue
                    
                    metrics_list = list(results['sweep_results'][emotion_label][layer_idx][alpha]['metrics'].values())
                    
                    # 平均を計算
                    avg_metrics = {
                        'emotion_keywords': {
                            'gratitude': np.mean([m['emotion_keywords']['gratitude'] for m in metrics_list]),
                            'anger': np.mean([m['emotion_keywords']['anger'] for m in metrics_list]),
                            'apology': np.mean([m['emotion_keywords']['apology'] for m in metrics_list])
                        },
                        'politeness': np.mean([m['politeness'] for m in metrics_list]),
                        'sentiment': np.mean([m['sentiment'] for m in metrics_list])
                    }
                    
                    aggregated[emotion_label][layer_idx][alpha] = avg_metrics
        
        return aggregated


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
    aggregated = patcher.aggregate_metrics(results)
    results['aggregated_metrics'] = aggregated
    
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

