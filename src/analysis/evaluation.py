"""
Transformerベースの共通評価モジュール。

全フェーズ（5/6/7など）で同じ評価器を再利用することを目的とし、
sentiment / politeness / GoEmotions を一貫したインターフェースで提供する。

デフォルトの公開HFモデル:
- sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest（3値: NEG/NEU/POS）
- politeness: NOVA-vision-language/polite_bert（4クラス: polite〜not polite）
- goemotions: SamLowe/roberta-base-go_emotions（28クラスのmulti-label）
すべて公開モデルで、認証不要で取得できるものを採用している。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from src.analysis.sentiment_eval import TransformerSequenceClassifier
from src.utils.device import get_default_device_str


EVAL_MODEL_IDS = {
    # 公開かつ安定して利用可能なモデルのみ指定する
    "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "politeness": "NOVA-vision-language/polite_bert",
    "goemotions": "SamLowe/roberta-base-go_emotions",
}


@dataclass
class EvaluatorPool:
    """評価モデルをキャッシュし、複数回のバッチ評価を高速化する。"""

    device: str = field(default_factory=get_default_device_str)
    _cache: Dict[str, Optional[TransformerSequenceClassifier]] = field(default_factory=dict, init=False)

    def get(self, metric: str) -> Optional[TransformerSequenceClassifier]:
        if metric not in EVAL_MODEL_IDS:
            raise ValueError(f"未知のメトリクス: {metric}")
        if metric in self._cache:
            return self._cache[metric]
        model_id = EVAL_MODEL_IDS[metric]
        try:
            print(f"✓ {metric} 評価器を読み込み中: {model_id} ({self.device})")
            clf = TransformerSequenceClassifier(model_id, self.device)
            self._cache[metric] = clf
            print(f"✓ {metric} 評価器準備完了")
            return clf
        except Exception as exc:  # pragma: no cover - remote model依存
            print(f"⚠ {metric} 評価器のロードに失敗しました: {exc}")
            self._cache[metric] = None  # 次回以降もスキップ
            return None


class TextEvaluator:
    """
    バッチ対応のテキスト評価器。
    
    複数のテキストを一括で評価し、各メトリクスごとに np.ndarray を返す。
    評価モデルは1回だけロードされ、キャッシュされる。
    """
    
    def __init__(self, device: Optional[str] = None, metrics: Optional[Iterable[str]] = None):
        """
        Args:
            device: 使用デバイス（未指定なら自動選択: MPS > CUDA > CPU）
            metrics: 評価対象メトリクス（デフォルト: sentiment, politeness, goemotions）
        """
        self.device = device or get_default_device_str()
        self.metrics = list(metrics) if metrics else ["sentiment", "politeness", "goemotions"]
        self._pool = EvaluatorPool(device=self.device)
        # 初期化時にすべての評価器をロード
        for metric in self.metrics:
            self._pool.get(metric)
    
    def evaluate_batch(self, texts: List[str], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        複数テキストをバッチで評価する。

        Args:
            texts: 評価するテキストのリスト
            batch_size: 各評価器の内部バッチサイズ

        Returns:
            メトリクス名をキーとし、各テキストのスコアを値とする辞書。
            例:
            {
                "sentiment/POSITIVE": np.array([0.8, 0.6, ...]),
                "sentiment/NEGATIVE": np.array([0.1, 0.3, ...]),
                "politeness/polite": np.array([0.9, 0.7, ...]),
                ...
            }
        """
        results: Dict[str, List[float]] = {}

        for metric in self.metrics:
            clf = self._pool.get(metric)
            if clf is None:
                continue

            # バッチ推論を実行
            batch_results = clf.predict_proba_batch(texts, batch_size=batch_size)

            # 各ラベルのスコアを抽出
            for result_dict in batch_results:
                for label, score in result_dict.items():
                    key = f"{metric}/{label}"
                    if key not in results:
                        results[key] = []
                    results[key].append(score)

        # List[float] を np.ndarray に変換
        return {key: np.array(values) for key, values in results.items()}

    def evaluate_batch_as_dicts(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        複数テキストをバッチで評価し、テキストごとの辞書リストで返す。

        後方互換性のため、evaluate_text_batch() と同じ形式で返す。

        Args:
            texts: 評価するテキストのリスト
            batch_size: 各評価器の内部バッチサイズ

        Returns:
            各テキストのメトリクス辞書リスト
            例:
            [
                {
                    "sentiment": {"POSITIVE": 0.8, "NEGATIVE": 0.1, "neutral": 0.1},
                    "politeness": {"polite": 0.9, ...},
                    "goemotions": {"joy": 0.6, ...},
                },
                ...
            ]
        """
        n_texts = len(texts)
        outputs: List[Dict[str, Any]] = [{} for _ in range(n_texts)]

        for metric in self.metrics:
            clf = self._pool.get(metric)
            if clf is None:
                continue

            # バッチ推論を実行
            batch_results = clf.predict_proba_batch(texts, batch_size=batch_size)

            # 各テキストの結果を格納
            for i, result_dict in enumerate(batch_results):
                outputs[i][metric] = result_dict

        return outputs


def evaluate_text_batch(
    texts: Sequence[str],
    metrics: Iterable[str] = ("sentiment", "politeness", "goemotions"),
    device: Optional[str] = None,
    pool: Optional[EvaluatorPool] = None,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    複数テキストに対して sentiment / politeness / GoEmotions を一括評価する。

    後方互換性のための関数。内部では TextEvaluator.evaluate_batch_as_dicts() を使用。

    Args:
        texts: 評価するテキスト配列
        metrics: 評価対象メトリクス名
        device: 使用デバイス（未指定なら自動判定）
        pool: 既存のEvaluatorPoolインスタンス（廃止予定、無視される）
        batch_size: 各評価器の内部バッチサイズ

    Returns:
        各テキストのメトリクス辞書リスト
    """
    # poolパラメータは後方互換性のために残すが、TextEvaluatorを使用
    evaluator = TextEvaluator(device=device, metrics=metrics)
    return evaluator.evaluate_batch_as_dicts(list(texts), batch_size=batch_size)
