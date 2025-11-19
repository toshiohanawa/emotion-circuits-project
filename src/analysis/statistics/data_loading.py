"""Utilities for loading patching experiment outputs into analysis-friendly formats."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import logging
import pickle
import json

import numpy as np

from src.utils.project_context import ProjectContext

try:  # Optional heavy dependency – only used when available.
    from src.analysis.sentiment_eval import SentimentEvaluator
except Exception:  # pragma: no cover - sentiment evaluator is optional.
    SentimentEvaluator = None


PhaseName = str
VALID_PHASES: Set[PhaseName] = {"residual", "head", "random", "head_screening"}


@dataclass
class PatchingRun:
    """In-memory representation of a single statistical comparison."""

    profile: str
    phase: PhaseName
    model_name: str
    metric_name: str
    baseline_values: List[float]
    patched_values: List[float]
    emotion: Optional[str] = None
    layer: Optional[int] = None
    head: Optional[int] = None
    alpha: Optional[float] = None
    head_spec: Optional[str] = None
    comparison: str = "patched_vs_baseline"
    paired: bool = True
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def metadata_dict(self) -> Dict[str, Any]:
        """Return the metadata payload used when converting to rows."""
        base = {
            "profile": self.profile,
            "phase": self.phase,
            "model_name": self.model_name,
            "metric_name": self.metric_name,
            "emotion": self.emotion,
            "layer": self.layer,
            "head": self.head,
            "alpha": self.alpha,
            "head_spec": self.head_spec,
            "comparison": self.comparison,
            "paired": self.paired,
        }
        base.update(self.extra_metadata)
        return base


def _extract_positive_probability(scores: Any) -> Optional[float]:
    """Return the POSITIVE probability from a classifier output dict."""
    if scores is None:
        return None
    if isinstance(scores, (int, float, np.floating)):
        return float(scores)
    if not isinstance(scores, dict):
        return None
    for key in ("positive", "POSITIVE", "label_2", "LABEL_2", "pos"):
        if key in scores:
            try:
                return float(scores[key])
            except (TypeError, ValueError):
                continue
    # Try a best-effort fallback (highest probability treated as positive)
    try:
        return float(max(scores.values(), key=float))
    except (ValueError, TypeError):
        return None


def _flatten_metric_dict(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """Flatten nested metric dictionaries into dot-delimited scalar entries."""
    flat: Dict[str, float] = {}
    for key, value in metrics.items():
        if key == "sentiment" and isinstance(value, dict):
            prob = _extract_positive_probability(value)
            if prob is not None:
                name = f"{prefix}sentiment" if prefix else "sentiment"
                flat[name] = float(prob)
            continue
        if key == "politeness" and isinstance(value, dict):
            # Most politeness metrics store a single score.
            score = value.get("politeness_score")
            if score is not None:
                name = f"{prefix}politeness" if prefix else "politeness"
                flat[name] = float(score)
                continue
        if isinstance(value, dict):
            nested_prefix = f"{prefix}{key}." if prefix else f"{key}."
            flat.update(_flatten_metric_dict(value, nested_prefix))
            continue
        name = f"{prefix}{key}" if prefix else key
        try:
            flat[name] = float(value)
        except (TypeError, ValueError):
            continue
    return flat


def _normalize_phase_filter(phase_filter: Optional[Sequence[str] | str]) -> Set[PhaseName]:
    """Standardize CLI/user filters into a phase set."""
    if phase_filter is None:
        return set(VALID_PHASES)
    if isinstance(phase_filter, str):
        tokens = [p.strip() for p in phase_filter.split(",") if p.strip()]
    else:
        tokens = [p.strip() for p in phase_filter if p.strip()]
    if not tokens or any(token in ("*", "all") for token in tokens):
        return set(VALID_PHASES)
    selected = {token for token in tokens if token in VALID_PHASES}
    return selected


# -----------------------------------------------------------------------------
# Head patching metric helpers
# -----------------------------------------------------------------------------
_GRATITUDE_KEYWORDS = [
    "thank",
    "thanks",
    "thanked",
    "thanking",
    "grateful",
    "gratitude",
    "gratefully",
    "appreciate",
    "appreciated",
    "appreciating",
    "appreciation",
    "thankful",
    "blessed",
]
_ANGER_KEYWORDS = [
    "angry",
    "anger",
    "angrily",
    "frustrated",
    "frustrating",
    "frustration",
    "terrible",
    "terribly",
    "annoyed",
    "annoying",
    "annoyance",
    "upset",
    "upsetting",
    "mad",
    "maddening",
    "furious",
    "furiously",
    "irritated",
    "irritating",
    "irritation",
]
_APOLOGY_KEYWORDS = [
    "sorry",
    "sorrier",
    "sorriest",
    "apologize",
    "apologized",
    "apologizing",
    "apology",
    "apologies",
    "regret",
    "regretted",
    "regretting",
    "regretful",
    "apologetic",
    "apologetically",
]
_POSITIVE_WORDS = [
    "good",
    "great",
    "excellent",
    "wonderful",
    "amazing",
    "fantastic",
    "happy",
    "pleased",
    "delighted",
    "satisfied",
    "positive",
    "nice",
]
_NEGATIVE_WORDS = [
    "bad",
    "horrible",
    "awful",
    "terrible",
    "worst",
    "angry",
    "upset",
    "frustrated",
    "negative",
    "unhappy",
    "sad",
    "worried",
]
_POLITENESS_MARKERS = [
    "please",
    "kindly",
    "thank you",
    "thanks",
    "sorry",
    "appreciate",
    "grateful",
    "would",
    "could",
    "may",
]


def _heuristic_keyword_counts(text: str) -> Dict[str, int]:
    """Fallback keyword counter when the SentimentEvaluator is unavailable."""
    lower = text.lower()
    return {
        "gratitude": sum(1 for kw in _GRATITUDE_KEYWORDS if kw in lower),
        "anger": sum(1 for kw in _ANGER_KEYWORDS if kw in lower),
        "apology": sum(1 for kw in _APOLOGY_KEYWORDS if kw in lower),
    }


def _heuristic_sentiment_score(text: str) -> Dict[str, float]:
    """Return POSITIVE/NEGATIVE ratios using the classic fallback heuristic."""
    lower = text.lower()
    pos = sum(1 for word in _POSITIVE_WORDS if word in lower)
    neg = sum(1 for word in _NEGATIVE_WORDS if word in lower)
    total = max(pos + neg, 1)
    return {"POSITIVE": pos / total, "NEGATIVE": neg / total}


def _heuristic_politeness_score(text: str) -> float:
    lower = text.lower()
    count = sum(1 for marker in _POLITENESS_MARKERS if marker in lower)
    return min(count / 10.0, 1.0)


_HEAD_EXTRACTOR_CACHE: Dict[Tuple[str, str, bool], "_HeadMetricExtractor"] = {}


class _HeadMetricExtractor:
    """Compute per-text metrics for head patching runs."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        enable_transformer_metrics: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device or "cpu"
        self.enable_transformer_metrics = enable_transformer_metrics
        self._evaluator: Optional[SentimentEvaluator] = None

    def _get_evaluator(self) -> Optional[SentimentEvaluator]:
        if SentimentEvaluator is None:
            return None
        if self._evaluator is None:
            self._evaluator = SentimentEvaluator(
                model_name=self.model_name,
                device=self.device,
                load_generation_model=False,
                enable_transformer_metrics=self.enable_transformer_metrics,
            )
        return self._evaluator

    def evaluate_text(self, text: str) -> Dict[str, float]:
        evaluator = self._get_evaluator()
        if evaluator:
            keyword_counts = evaluator.count_emotion_keywords(text)
            sentiment_scores = evaluator.calculate_sentiment_score(text) or {}
            politeness_score = evaluator.calculate_politeness_score(text)
        else:
            keyword_counts = _heuristic_keyword_counts(text)
            sentiment_scores = _heuristic_sentiment_score(text)
            politeness_score = _heuristic_politeness_score(text)

        metrics: Dict[str, float] = {
            f"emotion_keywords.{emotion}": float(value)
            for emotion, value in keyword_counts.items()
        }
        sentiment_value = _extract_positive_probability(sentiment_scores)
        if sentiment_value is not None:
            metrics["sentiment"] = float(sentiment_value)
        metrics["politeness"] = float(politeness_score)
        return metrics

    def evaluate_many(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        return [self.evaluate_text(text) for text in texts]


def _get_head_metric_extractor(
    model_name: str,
    device: Optional[str],
    enable_transformer_metrics: bool,
) -> _HeadMetricExtractor:
    key = (model_name, (device or "cpu"), bool(enable_transformer_metrics))
    if key not in _HEAD_EXTRACTOR_CACHE:
        _HEAD_EXTRACTOR_CACHE[key] = _HeadMetricExtractor(
            model_name=model_name,
            device=device,
            enable_transformer_metrics=enable_transformer_metrics,
        )
    return _HEAD_EXTRACTOR_CACHE[key]


# -----------------------------------------------------------------------------
# Residual patching loader
# -----------------------------------------------------------------------------
def _collect_prompt_metrics(metric_tree: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    """Flatten metrics per prompt."""
    flattened: Dict[str, Dict[str, float]] = {}
    for prompt, metrics in metric_tree.items():
        if not metrics:
            continue
        flattened[prompt] = _flatten_metric_dict(metrics)
    return flattened


def _paired_metric_lists(
    prompts: Sequence[str],
    baseline_metrics: Dict[str, Dict[str, float]],
    patched_metrics: Dict[str, Dict[str, float]],
    metric_name: str,
) -> Tuple[List[float], List[float]]:
    base_vals: List[float] = []
    patched_vals: List[float] = []
    for prompt in prompts:
        base_metric = baseline_metrics.get(prompt, {}).get(metric_name)
        patched_metric = patched_metrics.get(prompt, {}).get(metric_name)
        if base_metric is None or patched_metric is None:
            continue
        base_vals.append(float(base_metric))
        patched_vals.append(float(patched_metric))
    return base_vals, patched_vals


def load_residual_patching_runs(result_file: Path, profile: str) -> List[PatchingRun]:
    """Load residual activation patching sweeps into PatchingRun entries."""
    with open(result_file, "rb") as f:
        data = pickle.load(f)
    model_name = data.get("model", "unknown")
    stored_prompts: List[str] = data.get("prompts", [])
    prompts: List[str] = stored_prompts or list(data.get("baseline", {}).get("metrics", {}).keys())
    baseline_metrics = _collect_prompt_metrics(data.get("baseline", {}).get("metrics", {}))
    runs: List[PatchingRun] = []
    for emotion_label, layers in data.get("sweep_results", {}).items():
        for layer_idx, alpha_dict in layers.items():
            for alpha, alpha_results in alpha_dict.items():
                patched_metrics = _collect_prompt_metrics(alpha_results.get("metrics", {}))
                metric_sources = list(patched_metrics.values()) + list(baseline_metrics.values())
                metric_names = sorted(
                    set().union(*[metrics.keys() for metrics in metric_sources] or set())
                )
                for metric_name in metric_names:
                    base_vals, patched_vals = _paired_metric_lists(
                        prompts,
                        baseline_metrics,
                        patched_metrics,
                        metric_name,
                    )
                    if len(base_vals) < 2 or len(patched_vals) < 2:
                        continue
                    runs.append(
                        PatchingRun(
                            profile=profile,
                            phase="residual",
                            model_name=model_name,
                            metric_name=metric_name,
                            baseline_values=base_vals,
                            patched_values=patched_vals,
                            emotion=emotion_label,
                            layer=int(layer_idx),
                            head=None,
                            alpha=float(alpha),
                            head_spec=None,
                            comparison="patched_vs_neutral",
                            paired=True,
                            extra_metadata={"prompt_count": len(prompts)},
                        )
                    )
    return runs


# -----------------------------------------------------------------------------
# Head patching loader
# -----------------------------------------------------------------------------
def _parse_emotion_from_name(path: Path) -> Optional[str]:
    stem = path.stem.lower()
    for emotion in ("gratitude", "anger", "apology", "neutral"):
        if emotion in stem:
            return emotion
    return None


def load_head_patching_runs(
    result_file: Path,
    profile: str,
    device: Optional[str] = "cpu",
    enable_transformer_metrics: bool = True,
) -> List[PatchingRun]:
    """Convert head patching experiment pickle into PatchingRun entries."""
    with open(result_file, "rb") as f:
        data = pickle.load(f)
    model_name = data.get("model", "unknown")
    neutral_texts: List[str] = data.get("baseline_texts", [])
    patched_texts: List[str] = data.get("patched_texts", [])
    if not neutral_texts or not patched_texts:
        logging.warning("Skipping %s (missing texts)", result_file)
        return []
    if len(neutral_texts) != len(patched_texts):
        logging.warning("Skipping %s (baseline/patched count mismatch)", result_file)
        return []

    evaluator = _get_head_metric_extractor(model_name, device, enable_transformer_metrics)
    baseline_metrics = evaluator.evaluate_many(neutral_texts)
    patched_metrics = evaluator.evaluate_many(patched_texts)

    metric_names = sorted(
        set().union(*[metrics.keys() for metrics in baseline_metrics + patched_metrics])
    )
    head_specs: List[Tuple[int, int]] = data.get("heads", [])
    head_spec_str = ",".join(f"{layer}:{head}" for layer, head in head_specs) or None
    layer = head_specs[0][0] if len(head_specs) == 1 else None
    head_index = head_specs[0][1] if len(head_specs) == 1 else None
    emotion_label = _parse_emotion_from_name(result_file)
    runs: List[PatchingRun] = []
    for metric_name in metric_names:
        base_vals: List[float] = []
        patched_vals: List[float] = []
        for base_metric, patched_metric in zip(baseline_metrics, patched_metrics):
            if metric_name not in base_metric or metric_name not in patched_metric:
                continue
            base_vals.append(float(base_metric[metric_name]))
            patched_vals.append(float(patched_metric[metric_name]))
        if len(base_vals) < 2:
            continue
        runs.append(
            PatchingRun(
                profile=profile,
                phase="head",
                model_name=model_name,
                metric_name=metric_name,
                baseline_values=base_vals,
                patched_values=patched_vals,
                emotion=emotion_label,
                layer=layer,
                head=head_index,
                head_spec=head_spec_str,
                comparison="patched_vs_neutral",
                paired=True,
            )
        )
    return runs


# -----------------------------------------------------------------------------
# Random baseline loader
# -----------------------------------------------------------------------------
def _flatten_metric_entries(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, float]]:
    flattened: List[Dict[str, float]] = []
    for entry in entries:
        metrics = entry.get("metrics")
        if not metrics:
            continue
        flattened.append(_flatten_metric_dict(metrics))
    return flattened


def load_random_patching_runs(result_file: Path, profile: str) -> List[PatchingRun]:
    """Load random vs emotion control experiments."""
    with open(result_file, "rb") as f:
        data = pickle.load(f)
    model_name = data.get("model", "unknown")
    runs: List[PatchingRun] = []
    for emotion_label, layer_dict in data.get("random_results", {}).items():
        for layer_idx, random_runs in layer_dict.items():
            for alpha in data.get("alpha_values", []):
                random_metric_entries: List[Dict[str, float]] = []
                for rand_idx, alpha_bucket in random_runs.items():
                    entries = alpha_bucket.get(alpha, [])
                    random_metric_entries.extend(_flatten_metric_entries(entries))
                patched_entries = (
                    data.get("emotion_results", {})
                    .get(emotion_label, {})
                    .get(layer_idx, {})
                    .get(alpha, [])
                )
                patched_metric_entries = _flatten_metric_entries(patched_entries)
                if not random_metric_entries or not patched_metric_entries:
                    continue
                metric_names = sorted(
                    set().union(
                        *[metrics.keys() for metrics in random_metric_entries + patched_metric_entries]
                    )
                )
                for metric_name in metric_names:
                    random_vals = [
                        float(metrics[metric_name])
                        for metrics in random_metric_entries
                        if metric_name in metrics
                    ]
                    patched_vals = [
                        float(metrics[metric_name])
                        for metrics in patched_metric_entries
                        if metric_name in metrics
                    ]
                    if len(random_vals) < 2 or len(patched_vals) < 2:
                        continue
                    runs.append(
                        PatchingRun(
                            profile=profile,
                            phase="random",
                            model_name=model_name,
                            metric_name=metric_name,
                            baseline_values=random_vals,
                            patched_values=patched_vals,
                            emotion=emotion_label,
                            layer=int(layer_idx),
                            alpha=float(alpha),
                            comparison="emotion_vs_random",
                            paired=False,
                            extra_metadata={"num_random": data.get("num_random")},
                        )
                    )
    return runs


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def collect_patching_runs(
    context: ProjectContext,
    phase_filter: Optional[Sequence[str] | str] = None,
    head_metrics_device: Optional[str] = "cpu",
    enable_transformer_metrics: bool = True,
) -> List[PatchingRun]:
    """Collect all available patching runs under the given project context."""
    phases = _normalize_phase_filter(phase_filter)
    runs: List[PatchingRun] = []
    results_dir = context.results_dir()
    if "residual" in phases:
        sweep_dirs = [
            results_dir / "patching",
            results_dir / "patching" / "residual",
        ]
        for sweep_pattern in sweep_dirs:
            if sweep_pattern.exists():
                for path in sorted(sweep_pattern.glob("*sweep*.pkl")):
                    runs.extend(load_residual_patching_runs(path, context.profile_name))
    if "head" in phases:
        head_dir = results_dir / "patching" / "head_patching"
        if head_dir.exists():
            for path in sorted(head_dir.glob("*.pkl")):
                runs.extend(
                    load_head_patching_runs(
                        path,
                        context.profile_name,
                        device=head_metrics_device,
                        enable_transformer_metrics=enable_transformer_metrics,
                    )
                )
    if "random" in phases:
        random_dir = results_dir / "patching_random"
        if random_dir.exists():
            for path in sorted(random_dir.glob("*.pkl")):
                runs.extend(load_random_patching_runs(path, context.profile_name))
    if "head_screening" in phases:
        screening_dir = results_dir / "screening"
        if screening_dir.exists():
            for path in sorted(screening_dir.glob("head_scores_*.json")):
                runs.extend(load_head_screening_runs(path, context.profile_name))
    return runs


# -----------------------------------------------------------------------------
# Head screening loader
# -----------------------------------------------------------------------------
def _flatten_nested_metrics(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """ネストされた辞書をフラット化（例: {"sentiment": {"positive": 0.96}} -> {"sentiment.positive": 0.96}）"""
    result = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_nested_metrics(v, key))
        else:
            result[key] = float(v)
    return result


def load_head_screening_runs(result_file: Path, profile: str) -> List[PatchingRun]:
    """Convert head screening JSON (delta per head) into PatchingRun entries."""
    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    model_name = data.get("model", "unknown")
    head_samples = data.get("head_samples", [])
    runs: List[PatchingRun] = []
    for entry in head_samples:
        layer = entry.get("layer")
        head = entry.get("head")
        samples = entry.get("samples", [])
        baseline_metrics: Dict[str, Dict[str, float]] = {}
        patched_metrics: Dict[str, Dict[str, float]] = {}
        prompts: List[str] = []
        for sample in samples:
            prompt = sample.get("prompt")
            if prompt is None:
                continue
            prompts.append(prompt)
            # ネストされた辞書をフラット化
            baseline_metrics[prompt] = _flatten_nested_metrics(sample.get("baseline", {}))
            patched_metrics[prompt] = _flatten_nested_metrics(sample.get("patched", {}))
        metric_names = set()
        for m in list(baseline_metrics.values()) + list(patched_metrics.values()):
            metric_names.update(m.keys())
        for metric_name in sorted(metric_names):
            base_vals, patched_vals = _paired_metric_lists(
                prompts, baseline_metrics, patched_metrics, metric_name
            )
            if len(base_vals) < 2 or len(patched_vals) < 2:
                continue
            runs.append(
                PatchingRun(
                    profile=profile,
                    phase="head_screening",
                    model_name=model_name,
                    metric_name=metric_name,
                    baseline_values=base_vals,
                    patched_values=patched_vals,
                    layer=layer,
                    head=head,
                    comparison="screened_vs_baseline",
                    paired=True,
                    extra_metadata={"source": "head_screening"},
                )
            )
    return runs
