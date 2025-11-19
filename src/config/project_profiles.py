"""Central definitions for dataset profiles and result layouts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

EMOTION_LABELS: Sequence[str] = ("gratitude", "anger", "apology", "neutral")


@dataclass(frozen=True)
class DatasetProfile:
    """Metadata describing how prompts, datasets, and results are organized."""

    name: str
    dataset_file: str
    prompt_suffix: str = ""
    results_subdir: str | None = None
    description: str = ""

    def dataset_path(self, data_dir: Path | str = Path("data")) -> Path:
        """Return the dataset JSONL path resolved inside ``data_dir``."""
        base = Path(data_dir)
        return base / self.dataset_file

    def prompt_file(self, emotion: str, data_dir: Path | str = Path("data")) -> Path:
        """Return the prompt JSON file for a specific emotion."""
        if emotion not in EMOTION_LABELS:
            raise ValueError(f"Unknown emotion: {emotion}")
        base = Path(data_dir)
        suffix = self.prompt_suffix
        return base / f"{emotion}_prompts{suffix}.json"

    def prompt_files(
        self,
        data_dir: Path | str = Path("data"),
        emotions: Iterable[str] | None = None,
    ) -> Dict[str, Path]:
        """Return a mapping of emotion -> prompt file."""
        keys = list(emotions) if emotions else list(EMOTION_LABELS)
        return {emotion: self.prompt_file(emotion, data_dir) for emotion in keys}

    def results_dir(self, results_root: Path | str = Path("results")) -> Path:
        """Return the root directory for experiment artifacts."""
        base = Path(results_root)
        subdir = self.results_subdir or self.name
        return base / subdir


DATASET_PROFILES: Mapping[str, DatasetProfile] = {
    "baseline": DatasetProfile(
        name="baseline",
        dataset_file="emotion_dataset.jsonl",
        prompt_suffix="",
        results_subdir="baseline",
        description=(
            "標準プロファイル（感情ごとに225サンプル前後を目標）。"
            "旧版では70件/感情だったが、統計的検出力向上のため約225件に刷新。"
        ),
    ),
    "baseline_smoke": DatasetProfile(
        name="baseline_smoke",
        dataset_file="emotion_dataset_smoke.jsonl",
        prompt_suffix="",
        results_subdir="baseline_smoke",
        description="スモークテスト用の極小プロファイル（各感情3〜5サンプル程度、配線確認用）。",
    ),
}


def list_profiles() -> List[str]:
    """Return the available dataset profile names."""
    return sorted(DATASET_PROFILES.keys())


def get_profile(name: str) -> DatasetProfile:
    """Fetch a dataset profile by name."""
    try:
        return DATASET_PROFILES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset profile '{name}'. Available: {list_profiles()}") from exc
