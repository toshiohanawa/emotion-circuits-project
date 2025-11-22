"""
Utilities for recording per-phase runtimes and execution metadata.

Each CLI calls :func:`record_phase_timing` once it completes, which appends a
JSON line under ``results/<profile>/timing/phase_timings.jsonl``.  Entries
capture the phase, profile, model, device, number of samples, CLI arguments,
and arbitrary metadata (layers, heads, sampling設定など) so that downstream
analysisやレポート生成から容易に参照できる。
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from src.utils.project_context import ProjectContext


def _sanitize_value(value: Any) -> Any:
    """JSONに保存できる形へ変換する。"""
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_value(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def _sanitize_metadata(metadata: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not metadata:
        return {}
    return {str(key): _sanitize_value(value) for key, value in metadata.items()}


@dataclass
class PhaseTimingEntry:
    phase: str
    profile: str
    elapsed_seconds: float
    model: str | None = None
    device: str | None = None
    samples: int | None = None
    cli_args: Iterable[str] | None = None
    metadata: Mapping[str, Any] | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "phase": self.phase,
            "profile": self.profile,
            "model": self.model,
            "device": self.device,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "samples": self.samples,
            "metadata": _sanitize_metadata(self.metadata),
        }
        if self.cli_args is not None:
            payload["cli_args"] = list(self.cli_args)
        return payload


def record_phase_timing(
    *,
    context: ProjectContext,
    phase: str,
    started_at: float,
    model: str | None,
    device: str | None,
    samples: int | None,
    metadata: Mapping[str, Any] | None = None,
    cli_args: Iterable[str] | None = None,
) -> Path:
    """
    Append a phase timing entry under ``results/<profile>/timing``.

    Args:
        context: Project context bound to the active profile.
        phase: フェーズ識別子（例: ``phase2``）
        started_at: ``time.perf_counter()`` で取得した開始時刻
        model: 対象モデル名（Phase7などモデル非依存フェーズでは ``None``）
        device: 実行デバイス文字列
        samples: 入力サンプル数
        metadata: 任意の追加メタデータ（層/ヘッド/コマンド設定など）
        cli_args: CLIへ渡した引数（通常 ``sys.argv[1:]``）
    Returns:
        追記したJSONLファイルパス
    """
    elapsed = max(time.perf_counter() - started_at, 0.0)
    timing_dir = context.results_dir() / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)
    entry = PhaseTimingEntry(
        phase=phase,
        profile=context.profile_name,
        elapsed_seconds=elapsed,
        model=model,
        device=device,
        samples=samples,
        cli_args=cli_args if cli_args is not None else sys.argv[1:],
        metadata=metadata,
    )
    out_path = timing_dir / "phase_timings.jsonl"
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry.as_dict(), ensure_ascii=False) + "\n")
    return out_path
