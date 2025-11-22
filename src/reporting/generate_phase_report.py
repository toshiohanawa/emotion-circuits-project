"""
Utility CLI for generating per-phase Markdown reports populated with timing metadata.

Example:
    python -m src.reporting.generate_phase_report \
      --phase phase2 \
      --profile baseline_smoke \
      --model gpt2_small \
      --output docs/report/phase2_baseline_smoke_gpt2_small.md
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.project_profiles import list_profiles
from src.utils.project_context import ProjectContext, profile_help_text


PHASE_TEMPLATE_MAP: Dict[str, str] = {
    "phase1_dataset": "docs/report_template/phase1_dataset_template.md",
    "phase2": "docs/report_template/phase2_activations_template.md",
    "phase3": "docs/report_template/phase3_vectors_template.md",
    "phase4": "docs/report_template/phase4_alignment_template.md",
    "phase5": "docs/report_template/phase5_residual_patching_template.md",
    "phase6_head_patching": "docs/report_template/phase6_head_template.md",
    "phase6_head_screening": "docs/report_template/phase6_head_template.md",
    "phase7_statistics": "docs/report_template/phase7_statistics_template.md",
}


@dataclass
class PhaseTiming:
    raw: Dict[str, Any]

    @property
    def timestamp(self) -> datetime:
        return datetime.fromisoformat(self.raw["timestamp"])

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.raw.get("metadata") or {}


def _load_timing_entries(timing_path: Path) -> List[PhaseTiming]:
    if not timing_path.exists():
        raise FileNotFoundError(f"Timing log not found: {timing_path}")
    entries: List[PhaseTiming] = []
    with timing_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entries.append(PhaseTiming(json.loads(line)))
    return entries


def _select_entry(
    entries: List[PhaseTiming],
    *,
    phase: str,
    profile: str,
    model: Optional[str],
    run_id: Optional[str],
) -> PhaseTiming:
    filtered = [
        entry
        for entry in entries
        if entry.raw.get("phase") == phase and entry.raw.get("profile") == profile
    ]
    if model:
        filtered = [entry for entry in filtered if entry.raw.get("model") == model]
    if not filtered:
        raise ValueError("Matching timing entry not found. Re-run the phase or relax filters.")
    if run_id:
        for entry in filtered:
            if entry.raw.get("run_id") == run_id:
                return entry
        raise ValueError(f"Timing run_id '{run_id}' not found for requested filters.")
    filtered.sort(key=lambda e: e.timestamp)
    return filtered[-1]


def _load_template(phase: str, explicit_path: Optional[str]) -> str:
    if explicit_path:
        template_path = Path(explicit_path)
    else:
        template_rel = PHASE_TEMPLATE_MAP.get(phase)
        if not template_rel:
            raise ValueError(f"No default template registered for phase '{phase}'.")
        template_path = Path(template_rel)
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def _format_metadata(metadata: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for key in sorted(metadata):
        value = metadata[key]
        pretty = json.dumps(value, ensure_ascii=False)
        lines.append(f"  - {key}: {pretty}")
    return lines


def build_summary_block(entry: PhaseTiming, timing_path: Path, profile: str) -> str:
    elapsed_min = entry.raw.get("elapsed_seconds") / 60 if entry.raw.get("elapsed_seconds") else None
    lines = [
        "## 自動生成サマリ（LLMレビュー用）",
        f"- 実行ID: `{entry.raw.get('run_id')}`",
        f"- プロファイル: `{profile}`",
        f"- フェーズ: `{entry.raw.get('phase')}`",
        f"- モデル: `{entry.raw.get('model') or 'N/A'}`",
        f"- デバイス: `{entry.raw.get('device') or 'auto'}`",
        f"- サンプル数: {entry.raw.get('samples') if entry.raw.get('samples') is not None else 'N/A'}",
        f"- 計測時間: {entry.raw.get('elapsed_seconds')} 秒"
        + (f" ({elapsed_min:.2f} 分)" if elapsed_min is not None else ""),
        f"- 実行日時: {entry.raw.get('timestamp')}",
        f"- タイミングログ: `{timing_path}`",
    ]
    cli_args = entry.raw.get("cli_args")
    if cli_args:
        lines.append(f"- 実行コマンド: `python {' '.join(str(arg) for arg in cli_args)}`")
    metadata_lines = _format_metadata(entry.metadata)
    if metadata_lines:
        lines.append("- メタデータ:")
        lines.extend(metadata_lines)
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a per-phase Markdown report that embeds timing metadata.")
    parser.add_argument("--phase", type=str, choices=sorted(PHASE_TEMPLATE_MAP.keys()), required=True, help="対象フェーズ")
    parser.add_argument("--profile", type=str, choices=list_profiles(), default="baseline", help=profile_help_text())
    parser.add_argument("--model", type=str, default=None, help="モデル名（Phase7などモデル非依存の場合は省略）")
    parser.add_argument("--output", type=str, required=True, help="生成するMarkdownパス")
    parser.add_argument("--template", type=str, default=None, help="テンプレートを明示する場合はファイルパスを指定")
    parser.add_argument("--timing-run-id", type=str, default=None, help="特定の run_id を選択する場合に指定")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    ctx = ProjectContext(args.profile)
    timing_path = ctx.results_dir() / "timing" / "phase_timings.jsonl"
    entries = _load_timing_entries(timing_path)
    entry = _select_entry(
        entries,
        phase=args.phase,
        profile=args.profile,
        model=args.model,
        run_id=args.timing_run_id,
    )
    template_text = _load_template(args.phase, args.template)
    summary_block = build_summary_block(entry, timing_path, args.profile)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = template_text.rstrip() + "\n\n---\n" + summary_block + "\n"
    output_path.write_text(output_text, encoding="utf-8")
    print(f"[report] Generated {output_path} from timing entry {entry.raw.get('run_id')}")


if __name__ == "__main__":
    main()
