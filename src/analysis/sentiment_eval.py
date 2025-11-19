"""
長文生成とTransformerベースのSentiment/Politeness/Emotion評価モジュール
"""
from __future__ import annotations

import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from transformer_lens import HookedTransformer

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None
    AutoModelForSequenceClassification = None


class TransformerSequenceClassifier:
    """Utility wrapper around HuggingFace sequence classifiers."""

    def __init__(self, model_id: str, device: str):
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise ImportError("transformers is required for TransformerSequenceClassifier")
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval()
        self.model.to(device)
        self.id2label = getattr(self.model.config, "id2label", None)

    def predict_proba(self, text: str) -> Dict[str, float]:
        """Return probability distribution over labels for a single text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        label_map = self.id2label or {i: str(i) for i in range(len(probs))}
        return {label_map[i]: float(probs[i]) for i in range(len(probs))}
    
    def predict_proba_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
        """
        複数テキストをバッチで処理して確率分布を返す。
        
        Args:
            texts: 評価するテキストのリスト
            batch_size: バッチサイズ
        
        Returns:
            各テキストの確率分布のリスト
        """
        results: List[Dict[str, float]] = []
        label_map = self.id2label or {}
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            
            for prob in probs:
                if label_map:
                    result = {label_map.get(j, str(j)): float(prob[j]) for j in range(len(prob))}
                else:
                    result = {str(j): float(prob[j]) for j in range(len(prob))}
                results.append(result)
        
        return results


class SentimentEvaluator:
    """長文生成とtransformerベースの属性評価を行うクラス"""
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        load_generation_model: bool = True,
        enable_transformer_metrics: bool = True,
    ):
        """
        Args:
            model_name: HookedTransformerで利用するモデル名
            device: CUDA/CPU設定
            load_generation_model: Trueの場合はHookedTransformerをロード
            enable_transformer_metrics: Trueの場合はsentiment/politeness/emotionモデルをロード
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[HookedTransformer] = None
        self.enable_transformer_metrics = enable_transformer_metrics
        self._sentiment_classifier: Optional[TransformerSequenceClassifier] = None
        self._politeness_classifier: Optional[TransformerSequenceClassifier] = None
        self._emotion_classifier: Optional[TransformerSequenceClassifier] = None

        if load_generation_model:
            self._load_generation_model()
        elif enable_transformer_metrics:
            print(f"✓ Transformer metrics enabled on {self.device} (generation model skipped)")
    
    def _load_generation_model(self) -> None:
        """HookedTransformerのロードを分離（必要なときのみ使用）。"""
        if self.model is not None:
            return
        print(f"Loading model: {self.model_name}")
        self.model = HookedTransformer.from_pretrained(self.model_name, device=self.device)
        self.model.eval()
        print(f"✓ Model loaded on {self.device}")
    
    def _ensure_sentiment_classifier(self) -> Optional[TransformerSequenceClassifier]:
        if not self.enable_transformer_metrics:
            return None
        if self._sentiment_classifier is None:
            try:
                model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
                print(f"Loading sentiment classifier: {model_id}")
                self._sentiment_classifier = TransformerSequenceClassifier(model_id, self.device)
                print("✓ Sentiment classifier ready")
            except Exception as exc:
                print(f"Warning: Sentiment classifier unavailable ({exc}); falling back to heuristics.")
        return self._sentiment_classifier

    def _ensure_politeness_classifier(self) -> Optional[TransformerSequenceClassifier]:
        if not self.enable_transformer_metrics:
            return None
        if self._politeness_classifier is None:
            try:
                model_id = "michellejieli/Stanford_politeness_roberta"
                print(f"Loading politeness classifier: {model_id}")
                self._politeness_classifier = TransformerSequenceClassifier(model_id, self.device)
                print("✓ Politeness classifier ready")
            except Exception as exc:
                print(f"Warning: Politeness classifier unavailable ({exc}); falling back to heuristics.")
        return self._politeness_classifier

    def _ensure_emotion_classifier(self) -> Optional[TransformerSequenceClassifier]:
        if not self.enable_transformer_metrics:
            return None
        if self._emotion_classifier is None:
            try:
                model_id = "bhadresh-savani/roberta-base-go-emotions"
                print(f"Loading GoEmotions classifier: {model_id}")
                self._emotion_classifier = TransformerSequenceClassifier(model_id, self.device)
                print("✓ GoEmotions classifier ready")
            except Exception as exc:
                print(f"Warning: Emotion classifier unavailable ({exc}); falling back to token heuristics.")
        return self._emotion_classifier
    
    def generate_long_text(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> str:
        """
        長文を生成
        """
        if self.model is None:
            raise ValueError("Generation model is disabled for this SentimentEvaluator instance.")
        tokens = self.model.to_tokens(prompt)
        generated_tokens = tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.model(generated_tokens)
                next_token_logits = logits[0, -1, :] / temperature
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)
        
        generated_text = self.model.to_str_tokens(generated_tokens[0])
        return ' '.join(generated_text)
    
    def count_emotion_keywords(self, text: str) -> Dict[str, int]:
        """
        感情キーワードの頻度をカウント（フォールバック用）
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
    
    def calculate_politeness_score(self, text: str) -> float:
        """
        ヒューリスティックな丁寧さスコア（フォールバック用）
        """
        text_lower = text.lower()
        politeness_markers = [
            "please", "kindly", "thank you", "thanks", "sorry", 
            "appreciate", "grateful", "would", "could", "may"
        ]
        count = sum(1 for marker in politeness_markers if marker in text_lower)
        score = min(count / 10.0, 1.0)
        return score
    
    def calculate_sentiment_score(self, text: str) -> Optional[Dict[str, float]]:
        """
        Sentimentスコアを計算。Transformerモデルが利用可能な場合はそちらを優先。
        """
        classifier = self._ensure_sentiment_classifier()
        if classifier:
            return classifier.predict_proba(text)
        
        # フォールバック: 旧ヒューリスティック
        text_lower = text.lower()
        positive_words = [
            "good", "great", "excellent", "wonderful", "amazing", "fantastic",
            "happy", "pleased", "delighted", "satisfied", "positive", "nice"
        ]
        negative_words = [
            "bad", "horrible", "awful", "terrible", "worst", "angry", "upset",
            "frustrated", "negative", "unhappy", "sad", "worried"
        ]
        pos = sum(1 for word in positive_words if word in text_lower)
        neg = sum(1 for word in negative_words if word in text_lower)
        total = max(pos + neg, 1)
        return {"POSITIVE": pos / total, "NEGATIVE": neg / total}
    
    def evaluate_text_metrics(self, text: str) -> Dict[str, Dict[str, float]]:
        """
        Transformerベースの指標（sentiment / politeness / emotions）を一括で算出。
        対応するモデルが利用できない場合はヒューリスティックにフォールバック。
        """
        metrics: Dict[str, Dict[str, float]] = {}

        sentiment = self._ensure_sentiment_classifier()
        if sentiment:
            metrics['sentiment'] = sentiment.predict_proba(text)
        else:
            metrics['sentiment'] = self.calculate_sentiment_score(text) or {}

        politeness = self._ensure_politeness_classifier()
        if politeness:
            metrics['politeness'] = politeness.predict_proba(text)
        else:
            metrics['politeness'] = {'politeness_score': float(self.calculate_politeness_score(text))}

        emotion_classifier = self._ensure_emotion_classifier()
        if emotion_classifier:
            metrics['emotions'] = emotion_classifier.predict_proba(text)
        else:
            metrics['emotions'] = {
                emotion: float(count)
                for emotion, count in self.count_emotion_keywords(text).items()
            }

        return metrics
    
    def evaluate_text(self, text: str) -> Dict[str, Any]:
        """
        互換性維持用: 旧形式の評価結果を返す。
        """
        return {
            'text': text,
            'emotion_keywords': self.count_emotion_keywords(text),
            'sentiment': self.calculate_sentiment_score(text),
            'metrics': self.evaluate_text_metrics(text),
        }
    
    def evaluate_generation(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """
        プロンプトから生成して評価
        """
        generated_text = self.generate_long_text(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
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
    ) -> List[Dict[str, Any]]:
        """
        複数のプロンプトをバッチで評価
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            try:
                evaluation = self.evaluate_generation(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                results.append(evaluation)
            except Exception as exc:
                print(f"Error evaluating prompt '{prompt}': {exc}")
                results.append({
                    'prompt': prompt,
                    'error': str(exc)
                })
            if (i + 1) % 10 == 0 or (i + 1) == len(prompts):
                print(f"  評価進捗: {i+1}/{len(prompts)}")
        
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
    
    with open(args.prompts_file, 'r') as f:
        prompts_data = json.load(f)
        prompts = prompts_data.get('prompts', [])
    
    print(f"Using {len(prompts)} prompts")
    
    evaluator = SentimentEvaluator(args.model)
    
    results = evaluator.batch_evaluate(
        prompts,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(output_path.with_suffix('.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to:")
    print(f"  JSON: {output_path.with_suffix('.json')}")
    print(f"  Pickle: {output_path.with_suffix('.pkl')}")
    
    total_gratitude = sum(r.get('emotion_keywords', {}).get('gratitude', 0) for r in results if 'emotion_keywords' in r)
    total_anger = sum(r.get('emotion_keywords', {}).get('anger', 0) for r in results if 'emotion_keywords' in r)
    total_apology = sum(r.get('emotion_keywords', {}).get('apology', 0) for r in results if 'emotion_keywords' in r)
    
    print(f"\nTotal emotion keywords found:")
    print(f"  Gratitude: {total_gratitude}")
    print(f"  Anger: {total_anger}")
    print(f"  Apology: {total_apology}")
