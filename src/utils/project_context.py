"""Helpers for working with predefined dataset profiles."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

from src.config.project_profiles import (
    DATASET_PROFILES,
    DatasetProfile,
    EMOTION_LABELS,
    get_profile,
    list_profiles,
)


def profile_choices() -> Dict[str, DatasetProfile]:
    """Return the name -> profile mapping."""
    return dict(DATASET_PROFILES)


@dataclass(frozen=True)
class ContextPaths:
    """Resolved absolute and relative paths derived from a `ProjectContext`."""

    dataset_file: Path
    dataset_file_absolute: Path
    dataset_file_relative: Path
    results_root: Path
    results_root_absolute: Path
    results_root_relative: Path


@dataclass
class ProjectContext:
    """Convenience wrapper for resolving dataset/data/result paths."""

    profile_name: str = "baseline"
    data_dir: Path | str = Path("data")
    results_root: Path | str = Path("results")

    def __post_init__(self) -> None:
        self.profile = get_profile(self.profile_name)
        self.data_dir = Path(self.data_dir)
        self.results_root = Path(self.results_root)

    @property
    def emotions(self):
        return tuple(EMOTION_LABELS)

    def dataset_path(self) -> Path:
        return self.profile.dataset_path(self.data_dir)

    def prompt_file(self, emotion: str) -> Path:
        return self.profile.prompt_file(emotion, self.data_dir)

    def prompt_files(self, emotions: Iterable[str] | None = None) -> Dict[str, Path]:
        return self.profile.prompt_files(self.data_dir, emotions)

    def results_dir(self) -> Path:
        return self.profile.results_dir(self.results_root)

    def dataset_path_absolute(self) -> Path:
        return self.dataset_path().resolve()

    def results_dir_absolute(self) -> Path:
        return self.results_dir().resolve()

    def _relative_to_cwd(self, path: Path) -> Path:
        cwd = Path.cwd()
        try:
            return path.resolve().relative_to(cwd)
        except ValueError:
            return path

    def dataset_path_relative(self) -> Path:
        return self._relative_to_cwd(self.dataset_path())

    def results_dir_relative(self) -> Path:
        return self._relative_to_cwd(self.results_dir())

    def ensure_results_dir(self) -> Path:
        path = self.results_dir()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def paths(self) -> ContextPaths:
        dataset = self.dataset_path()
        results = self.results_dir()
        return ContextPaths(
            dataset_file=dataset,
            dataset_file_absolute=dataset.resolve(),
            dataset_file_relative=self.dataset_path_relative(),
            results_root=results,
            results_root_absolute=results.resolve(),
            results_root_relative=self.results_dir_relative(),
        )


def profile_help_text() -> str:
    """Return a CLI-friendly help string listing available profiles."""
    pairs = [f"{name}: {profile.description}" for name, profile in DATASET_PROFILES.items()]
    return " | ".join(pairs)
