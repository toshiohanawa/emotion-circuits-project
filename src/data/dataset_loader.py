"""
Phase 1 以降で共通利用するデータセットローダ。

想定フォーマット:
- JSONL: {"text": ..., "emotion": ...}
- CSV   : text, emotion/label 列
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from src.config.project_profiles import EMOTION_LABELS
from src.utils.project_context import ProjectContext


def _load_csv(path: Path, allowed: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    allowed_set = set(allowed)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("emotion") or row.get("label") or "").strip()
            if not text or not label:
                continue
            if label not in allowed_set:
                continue
            rows.append({"text": text, "emotion": label, "lang": row.get("lang", "en")})
    return rows


def _load_jsonl(path: Path, allowed: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    allowed_set = set(allowed)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            text = (data.get("text") or "").strip()
            label = (data.get("emotion") or data.get("label") or "").strip()
            if not text or not label:
                continue
            if label not in allowed_set:
                continue
            rows.append({"text": text, "emotion": label, "lang": data.get("lang", "en")})
    return rows


def load_dataset(path: Path, emotions: Iterable[str] | None = None) -> List[Dict[str, str]]:
    """
    データセットを読み込み、text/emotionを返す。
    """
    allowed = tuple(emotions or EMOTION_LABELS)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path, allowed)
    if suffix in (".jsonl", ".json"):
        return _load_jsonl(path, allowed)
    raise ValueError(f"未対応のデータセット形式: {path}")


def load_dataset_for_profile(profile: str, data_dir: str | Path = "data") -> List[Dict[str, str]]:
    """
    プロファイル設定からデータセットパスを解決して読み込む。
    """
    ctx = ProjectContext(profile, data_dir=data_dir)
    return load_dataset(ctx.dataset_path(), emotions=ctx.emotions)
