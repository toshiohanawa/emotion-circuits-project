"""
Head ablation実験
指定headの出力をゼロアウトして生成させ、感情トーンやsentimentがどう変わるかを見る
"""
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformer_lens import HookedTransformer
from tqdm import tqdm

from src.analysis.sentiment_eval import SentimentEvaluator


class HeadAblator:
    """Head ablationを実行するクラス"""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        初期化
        
        Args:
            model_name: モデル名（例: "gpt2"）
            device: 使用するデバイス
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model: {model_name}")
        self.model = HookedTransformer.from_pretrained(model_name, device=self.device)
        self.model.eval()
        
        print(f"✓ Model loaded on {self.device}")
        
        # Sentiment評価器を初期化
        self.sentiment_evaluator = SentimentEvaluator(model_name, device=device)
    
    def parse_head_spec(self, head_spec: str) -> List[Tuple[int, int]]:
        """
        Head指定文字列をパース
        
        Args:
            head_spec: "layer:head"のカンマ区切り（例: "3:5,7:2"）
            
        Returns:
            [(layer_idx, head_idx), ...] のリスト
        """
        heads = []
        for spec in head_spec.split(','):
            spec = spec.strip()
            if ':' in spec:
                layer_str, head_str = spec.split(':')
                try:
                    layer_idx = int(layer_str.strip())
                    head_idx = int(head_str.strip())
                    heads.append((layer_idx, head_idx))
                except ValueError:
                    print(f"Warning: Invalid head spec '{spec}', skipping...")
        return heads
    
    def generate_with_ablation(
        self,
        prompt: str,
        ablated_heads: List[Tuple[int, int]],
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> str:
        """
        Head ablationを適用して生成
        
        Args:
            prompt: 入力プロンプト
            ablated_heads: ゼロアウトするheadのリスト [(layer_idx, head_idx), ...]
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度
            top_p: nucleus samplingのパラメータ
            
        Returns:
            生成されたテキスト
        """
        tokens = self.model.to_tokens(prompt)
        generated_tokens = tokens.clone()
        
        # Ablation用のhook関数を定義
        def ablation_hook(activation, hook):
            """指定headの出力をゼロアウト"""
            layer_idx = int(hook.name.split('.')[1])
            
            # この層でablationするheadがあるかチェック
            for abl_layer, abl_head in ablated_heads:
                if abl_layer == layer_idx:
                    # activation shape: [batch, pos, head, d_head] または [batch, head, pos, d_head]
                    # TransformerLensの形式に合わせて処理
                    if len(activation.shape) == 4:
                        # [batch, pos, head, d_head] の場合
                        activation = activation.clone()
                        activation[:, :, abl_head, :] = 0.0
                    elif len(activation.shape) == 4 and activation.shape[1] == self.model.cfg.n_heads:
                        # [batch, head, pos, d_head] の場合
                        activation = activation.clone()
                        activation[:, abl_head, :, :] = 0.0
            
            return activation
        
        # Hookを登録
        hook_handles = []
        for layer_idx, head_idx in ablated_heads:
            hook_name = f"blocks.{layer_idx}.attn.hook_result"
            handle = self.model.add_hook(hook_name, ablation_hook)
            hook_handles.append((hook_name, handle))
        
        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    # 現在のトークンでlogitsを取得
                    logits = self.model(generated_tokens)
                    
                    # 最後のトークンのlogitsを使用
                    next_token_logits = logits[0, -1, :] / temperature
                    
                    # Top-p sampling
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Top-pでフィルタ
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                    
                    # サンプリング
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # 生成されたトークンを追加
                    generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)
        
        finally:
            # Hookを削除
            for hook_name, handle in hook_handles:
                if hook_name in self.model.hook_dict:
                    hook_point = self.model.hook_dict[hook_name]
                    hook_point.fwd_hooks = []
        
        # トークンをテキストに変換
        generated_text = self.model.to_str_tokens(generated_tokens[0])
        full_text = ' '.join(generated_text)
        
        return full_text
    
    def generate_baseline(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> str:
        """
        ベースライン生成（ablationなし）
        
        Args:
            prompt: 入力プロンプト
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度
            top_p: nucleus samplingのパラメータ
            
        Returns:
            生成されたテキスト
        """
        return self.sentiment_evaluator.generate_long_text(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
    
    def evaluate_texts(
        self,
        texts: List[str]
    ) -> Dict:
        """
        テキストを評価（感情キーワード頻度とsentimentスコア）
        
        Args:
            texts: テキストのリスト
            
        Returns:
            評価結果の辞書
        """
        all_keyword_counts = {
            'gratitude': [],
            'anger': [],
            'apology': []
        }
        all_sentiment_scores = []
        
        for text in texts:
            # 感情キーワード頻度
            keyword_counts = self.sentiment_evaluator.count_emotion_keywords(text)
            for emotion in ['gratitude', 'anger', 'apology']:
                all_keyword_counts[emotion].append(keyword_counts[emotion])
            
            # Sentimentスコア
            sentiment = self.sentiment_evaluator.calculate_sentiment_score(text)
            if sentiment:
                all_sentiment_scores.append(sentiment.get('POSITIVE', 0.0))
        
        metrics = {
            'keyword_counts': {
                emotion: {
                    'mean': float(np.mean(counts)),
                    'std': float(np.std(counts)),
                    'total': int(np.sum(counts))
                }
                for emotion, counts in all_keyword_counts.items()
            },
            'sentiment': {
                'mean': float(np.mean(all_sentiment_scores)) if all_sentiment_scores else 0.0,
                'std': float(np.std(all_sentiment_scores)) if all_sentiment_scores else 0.0
            }
        }
        
        return metrics
    
    def run_ablation_experiment(
        self,
        prompts: List[str],
        ablated_heads: List[Tuple[int, int]],
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> Dict:
        """
        Ablation実験を実行
        
        Args:
            prompts: プロンプトのリスト
            ablated_heads: ゼロアウトするheadのリスト
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度
            top_p: nucleus samplingのパラメータ
            
        Returns:
            実験結果の辞書
        """
        baseline_texts = []
        ablation_texts = []
        
        print(f"Running ablation experiment with {len(ablated_heads)} heads...")
        print(f"Ablated heads: {ablated_heads}")
        
        for prompt in tqdm(prompts, desc="Generating texts"):
            # ベースライン生成
            baseline_text = self.generate_baseline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            baseline_texts.append(baseline_text)
            
            # Ablation生成
            ablation_text = self.generate_with_ablation(
                prompt,
                ablated_heads,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            ablation_texts.append(ablation_text)
        
        # 評価
        baseline_metrics = self.evaluate_texts(baseline_texts)
        ablation_metrics = self.evaluate_texts(ablation_texts)
        
        results = {
            "model": self.model_name,
            "heads": ablated_heads,
            "prompts": prompts,
            "baseline_texts": baseline_texts,
            "ablation_texts": ablation_texts,
            "baseline_metrics": baseline_metrics,
            "ablation_metrics": ablation_metrics
        }
        
        return results


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Head ablation experiment")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--head-spec", type=str, required=True, help="Head specification (e.g., '3:5,7:2')")
    parser.add_argument("--prompts-file", type=str, required=True, help="Prompts file (JSON)")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--max-tokens", type=int, default=30, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    
    args = parser.parse_args()
    
    # Head指定をパース
    ablator = HeadAblator(args.model, device=args.device)
    ablated_heads = ablator.parse_head_spec(args.head_spec)
    
    if not ablated_heads:
        raise ValueError(f"No valid heads found in spec: {args.head_spec}")
    
    # プロンプトを読み込み
    prompts_file = Path(args.prompts_file)
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    with open(prompts_file, 'r') as f:
        prompts_data = json.load(f)
        prompts = prompts_data.get('prompts', [])
    
    print(f"Using {len(prompts)} prompts")
    
    # Ablation実験を実行
    results = ablator.run_ablation_experiment(
        prompts,
        ablated_heads,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # 結果を保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")
    
    # サマリーを表示
    print("\nBaseline metrics:")
    print(f"  Sentiment mean: {results['baseline_metrics']['sentiment']['mean']:.4f}")
    print(f"  Gratitude keywords: {results['baseline_metrics']['keyword_counts']['gratitude']['total']}")
    print(f"  Anger keywords: {results['baseline_metrics']['keyword_counts']['anger']['total']}")
    print(f"  Apology keywords: {results['baseline_metrics']['keyword_counts']['apology']['total']}")
    
    print("\nAblation metrics:")
    print(f"  Sentiment mean: {results['ablation_metrics']['sentiment']['mean']:.4f}")
    print(f"  Gratitude keywords: {results['ablation_metrics']['keyword_counts']['gratitude']['total']}")
    print(f"  Anger keywords: {results['ablation_metrics']['keyword_counts']['anger']['total']}")
    print(f"  Apology keywords: {results['ablation_metrics']['keyword_counts']['apology']['total']}")
    
    print("\nDifference:")
    sentiment_diff = results['ablation_metrics']['sentiment']['mean'] - results['baseline_metrics']['sentiment']['mean']
    print(f"  Sentiment change: {sentiment_diff:+.4f}")


if __name__ == "__main__":
    main()

