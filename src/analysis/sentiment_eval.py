"""
長文生成とSentiment評価モジュール
30トークン程度の長文生成をサポートし、HuggingFaceのsentimentモデルで評価
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


class SentimentEvaluator:
    """長文生成とsentiment評価を行うクラス"""
    
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
        
        # Sentimentモデルをロード（利用可能な場合）
        self.sentiment_model = None
        self.sentiment_tokenizer = None
        self._load_sentiment_model()
    
    def _load_sentiment_model(self):
        """HuggingFaceのsentimentモデルをロード"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # distilbert-base-uncased-finetuned-sst-2-english を試す
            try:
                model_id = "distilbert-base-uncased-finetuned-sst-2-english"
                print(f"Loading sentiment model: {model_id}")
                self.sentiment_tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_id)
                self.sentiment_model.eval()
                self.sentiment_model.to(self.device)
                print(f"✓ Sentiment model loaded")
            except Exception as e:
                print(f"Warning: Could not load sentiment model: {e}")
                print("Continuing without sentiment model...")
        except ImportError:
            print("Warning: transformers library not available. Sentiment evaluation will be limited.")
    
    def generate_long_text(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> str:
        """
        長文を生成
        
        Args:
            prompt: 入力プロンプト
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度
            top_p: nucleus samplingのパラメータ
            
        Returns:
            生成されたテキスト（プロンプト含む）
        """
        tokens = self.model.to_tokens(prompt)
        original_length = tokens.shape[1]
        
        generated_tokens = tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 現在のトークン列で推論
                logits = self.model(generated_tokens)
                
                # 最後のトークンのlogitsを取得
                next_token_logits = logits[0, -1, :] / temperature
                
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 累積確率がtop_pを超える最初のインデックス
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
        
        # トークンをテキストに変換
        generated_text = self.model.to_str_tokens(generated_tokens[0])
        full_text = ' '.join(generated_text)
        
        return full_text
    
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
            "thank", "thanks", "thanked", "thanking",
            "grateful", "gratitude", "gratefully",
            "appreciate", "appreciated", "appreciating", "appreciation",
            "thankful", "blessed"
        ]
        anger_keywords = [
            "angry", "anger", "angrily",
            "frustrated", "frustrating", "frustration",
            "terrible", "terribly",
            "annoyed", "annoying", "annoyance",
            "upset", "upsetting",
            "mad", "maddening",
            "furious", "furiously",
            "irritated", "irritating", "irritation"
        ]
        apology_keywords = [
            "sorry", "sorrier", "sorriest",
            "apologize", "apologized", "apologizing", "apology", "apologies",
            "regret", "regretted", "regretting", "regretful",
            "apologetic", "apologetically"
        ]
        
        counts = {
            'gratitude': sum(1 for kw in gratitude_keywords if kw in text_lower),
            'anger': sum(1 for kw in anger_keywords if kw in text_lower),
            'apology': sum(1 for kw in apology_keywords if kw in text_lower)
        }
        
        return counts
    
    def calculate_sentiment_score(self, text: str) -> Optional[Dict[str, float]]:
        """
        HuggingFaceのsentimentモデルでsentimentスコアを計算
        
        Args:
            text: テキスト
            
        Returns:
            sentimentスコアの辞書（モデルが利用可能な場合）、None（利用不可の場合）
        """
        if self.sentiment_model is None or self.sentiment_tokenizer is None:
            return None
        
        try:
            # テキストをトークン化
            inputs = self.sentiment_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 推論
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            # スコアを取得（通常はNEGATIVEとPOSITIVEの2クラス）
            scores = probs[0].cpu().numpy()
            
            # ラベルを取得
            if hasattr(self.sentiment_model.config, 'id2label'):
                labels = self.sentiment_model.config.id2label
                result = {labels[i]: float(scores[i]) for i in range(len(scores))}
            else:
                # デフォルトラベル
                result = {
                    'NEGATIVE': float(scores[0]),
                    'POSITIVE': float(scores[1]) if len(scores) > 1 else 0.0
                }
            
            return result
        except Exception as e:
            print(f"Error calculating sentiment score: {e}")
            return None
    
    def evaluate_text(self, text: str) -> Dict:
        """
        テキストを評価
        
        Args:
            text: 評価するテキスト
            
        Returns:
            評価結果の辞書
        """
        result = {
            'text': text,
            'emotion_keywords': self.count_emotion_keywords(text),
            'sentiment': self.calculate_sentiment_score(text)
        }
        
        return result
    
    def evaluate_generation(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> Dict:
        """
        プロンプトから生成して評価
        
        Args:
            prompt: 入力プロンプト
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度
            top_p: nucleus samplingのパラメータ
            
        Returns:
            評価結果の辞書
        """
        # 生成
        generated_text = self.generate_long_text(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # 評価
        evaluation = self.evaluate_text(generated_text)
        evaluation['prompt'] = prompt
        evaluation['max_new_tokens'] = max_new_tokens
        
        return evaluation
    
    def batch_evaluate(
        self,
        prompts: List[str],
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> List[Dict]:
        """
        複数のプロンプトをバッチで評価
        
        Args:
            prompts: プロンプトのリスト
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度
            top_p: nucleus samplingのパラメータ
            
        Returns:
            評価結果のリスト
        """
        results = []
        
        for prompt in tqdm(prompts, desc="Evaluating generations"):
            try:
                evaluation = self.evaluate_generation(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                results.append(evaluation)
            except Exception as e:
                print(f"Error evaluating prompt '{prompt}': {e}")
                results.append({
                    'prompt': prompt,
                    'error': str(e)
                })
        
        return results


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate text generation with sentiment analysis")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--prompts_file", type=str, required=True, help="Prompts file (JSON)")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--max_tokens", type=int, default=30, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    
    args = parser.parse_args()
    
    # プロンプトを読み込み
    with open(args.prompts_file, 'r') as f:
        prompts_data = json.load(f)
        prompts = prompts_data.get('prompts', [])
    
    print(f"Using {len(prompts)} prompts")
    
    # 評価器を作成
    evaluator = SentimentEvaluator(args.model)
    
    # バッチ評価を実行
    results = evaluator.batch_evaluate(
        prompts,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # 結果を保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # JSON形式で保存
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Pickle形式でも保存
    with open(output_path.with_suffix('.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to:")
    print(f"  JSON: {output_path.with_suffix('.json')}")
    print(f"  Pickle: {output_path.with_suffix('.pkl')}")
    
    # サマリーを表示
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    
    total_gratitude = sum(r.get('emotion_keywords', {}).get('gratitude', 0) for r in results if 'emotion_keywords' in r)
    total_anger = sum(r.get('emotion_keywords', {}).get('anger', 0) for r in results if 'emotion_keywords' in r)
    total_apology = sum(r.get('emotion_keywords', {}).get('apology', 0) for r in results if 'emotion_keywords' in r)
    
    print(f"Total emotion keywords found:")
    print(f"  Gratitude: {total_gratitude}")
    print(f"  Anger: {total_anger}")
    print(f"  Apology: {total_apology}")
    
    # Sentimentスコアの平均を計算
    sentiment_scores = [r.get('sentiment') for r in results if r.get('sentiment') is not None]
    if sentiment_scores:
        print(f"\nSentiment analysis available for {len(sentiment_scores)} texts")
        if 'POSITIVE' in sentiment_scores[0]:
            avg_positive = np.mean([s.get('POSITIVE', 0.0) for s in sentiment_scores])
            avg_negative = np.mean([s.get('NEGATIVE', 0.0) for s in sentiment_scores])
            print(f"  Average POSITIVE score: {avg_positive:.3f}")
            print(f"  Average NEGATIVE score: {avg_negative:.3f}")


if __name__ == "__main__":
    main()

