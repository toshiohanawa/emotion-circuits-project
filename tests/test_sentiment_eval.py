"""
Tests for the revamped SentimentEvaluator.
"""
from src.analysis.sentiment_eval import SentimentEvaluator


def test_evaluator_metrics_without_generation_model():
    """Transformer metrics can fall back to heuristics when disabled."""
    evaluator = SentimentEvaluator(
        "gpt2",
        load_generation_model=False,
        enable_transformer_metrics=False,
    )
    metrics = evaluator.evaluate_text_metrics("Thank you for your help.")
    assert 'sentiment' in metrics
    assert 'politeness' in metrics
    assert 'emotions' in metrics


def test_generate_requires_model():
    """Calling generate_long_text without a loaded model should raise."""
    evaluator = SentimentEvaluator(
        "gpt2",
        load_generation_model=False,
        enable_transformer_metrics=False,
    )
    raised = False
    try:
        _ = evaluator.generate_long_text("Hello there.")
    except ValueError:
        raised = True
    assert raised
