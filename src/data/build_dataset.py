"""
標準化データセットの構築スクリプト。

主な目的:
- ユーザーが用意するCSV/JSONLをプロファイル基準のJSONLに変換する
- 既存のプロンプトJSON (emotion_prompts*.json) からの旧来生成も互換維持

想定する入力:
- CSV: 列に `text`, `label` (または `emotion`) を含む
- JSONL: 各行に {"text": ..., "emotion": ...}

規模の目安:
- baseline: 感情4種 × 各225サンプル前後（合計約900行）を想定（サンプルはユーザーが準備）
- baseline_smoke: 各感情3〜5件程度の配線確認用

CLI例:
    python -m src.data.build_dataset \\
      --profile baseline \\
      --input data/emotion_dataset.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.config.project_profiles import EMOTION_LABELS, list_profiles
from src.utils.project_context import ProjectContext, profile_help_text


# ---------------------------------------------------------------------------#
# データ構造
# ---------------------------------------------------------------------------#
@dataclass
class Sample:
    text: str
    emotion: str
    lang: str = "en"


# ---------------------------------------------------------------------------#
# 入力ローダ
# ---------------------------------------------------------------------------#
def _load_from_csv(
    path: Path,
    text_field: str = "text",
    label_field: str = "label",
    allowed_emotions: Sequence[str] | None = None,
) -> List[Sample]:
    samples: List[Sample] = []
    allowed = set(allowed_emotions or EMOTION_LABELS)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get(text_field) or "").strip()
            label = (row.get(label_field) or row.get("emotion") or "").strip()
            if not text or not label:
                continue
            if allowed and label not in allowed:
                continue
            samples.append(Sample(text=text, emotion=label))
    return samples


def _load_from_jsonl(
    path: Path,
    allowed_emotions: Sequence[str] | None = None,
) -> List[Sample]:
    allowed = set(allowed_emotions or EMOTION_LABELS)
    samples: List[Sample] = []
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
            if allowed and label not in allowed:
                continue
            samples.append(Sample(text=text, emotion=label))
    return samples


def _load_from_prompt_json(path: Path, label: str) -> List[Sample]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    prompts: Iterable[str] = payload.get("prompts", [])
    return [Sample(text=p, emotion=label) for p in prompts]


# ---------------------------------------------------------------------------#
# メインロジック
# ---------------------------------------------------------------------------#
def build_dataset(
    input_path: Optional[Path],
    output_path: Path,
    profile: Optional[str],
    text_field: str = "text",
    label_field: str = "label",
    prefer_prompts: bool = False,
    allowed_emotions: Sequence[str] | None = None,
) -> Dict:
    """
    CSV/JSONL/プロンプトJSONから標準JSONLを生成。

    Args:
        input_path: 入力ファイル（csv/json/jsonl）。Noneの場合、プロンプトJSONを探索。
        output_path: 出力先 (JSONL)
        profile: プロファイル名（ログ出力用）
        text_field: CSVの本文列名
        label_field: CSVのラベル列名
        prefer_prompts: Trueの場合、input_pathが無くてもプロンプトJSON探索を優先
        allowed_emotions: 使用する感情ラベル
    """
    allowed = tuple(allowed_emotions or EMOTION_LABELS)
    samples: List[Sample] = []

    if input_path:
        suffix = input_path.suffix.lower()
        if suffix in {".csv"}:
            samples = _load_from_csv(input_path, text_field=text_field, label_field=label_field, allowed_emotions=allowed)
        elif suffix in {".jsonl"}:
            samples = _load_from_jsonl(input_path, allowed_emotions=allowed)
        elif suffix in {".json"}:
            # 単一JSONは旧来のpromptファイルとは限らないが、prompts配列があれば利用
            data = json.loads(input_path.read_text(encoding="utf-8"))
            if "prompts" in data and len(allowed) == 1:
                label = allowed[0]
                samples = [Sample(text=t, emotion=label) for t in data["prompts"]]
            else:
                raise ValueError("JSON入力はprompts配列を持つ単一感情ファイルのみをサポートします。")
        else:
            raise ValueError(f"未対応の入力拡張子です: {suffix}")
    else:
        if not prefer_prompts:
            raise ValueError("input_pathが指定されていません。CSV/JSONLを --input で渡してください。")
        # レガシー: emotion_prompts*.json から構築
        ctx = ProjectContext(profile or "baseline")
        for emotion in allowed:
            prompt_file = ctx.prompt_file(emotion)
            if not prompt_file.exists():
                print(f"⚠ プロンプトファイルが見つかりません: {prompt_file}")
                continue
            samples.extend(_load_from_prompt_json(prompt_file, emotion))

    if not samples:
        raise ValueError("有効なサンプルが1件も読み込まれませんでした。入力を確認してください。")

    # 書き出し
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"text": s.text, "emotion": s.emotion, "lang": s.lang}, ensure_ascii=False) + "\n")

    counts: Dict[str, int] = {emo: 0 for emo in allowed}
    lengths: List[int] = []
    for s in samples:
        counts[s.emotion] = counts.get(s.emotion, 0) + 1
        lengths.append(len(s.text))

    stats = {
        "total_samples": len(samples),
        "emotion_counts": counts,
        "text_length": {
            "avg": sum(lengths) / len(lengths),
            "min": min(lengths),
            "max": max(lengths),
        },
        "profile": profile or "custom",
        "input": str(input_path) if input_path else "prompt_json",
        "output": str(output_path),
    }

    print("✓ データセットを書き出しました")
    print(f"  - 出力: {output_path}")
    print(f"  - プロファイル: {profile or 'custom'}")
    for emo, cnt in counts.items():
        print(f"    * {emo}: {cnt}")
    print(f"  - 文字数: 平均 {stats['text_length']['avg']:.1f} / 最小 {stats['text_length']['min']} / 最大 {stats['text_length']['max']}")
    return stats


# ---------------------------------------------------------------------------#
# CLI
# ---------------------------------------------------------------------------#
def main():
    parser = argparse.ArgumentParser(
        description="CSV/JSONLを標準化した感情データセット(JSONL)に変換します。",
    )
    parser.add_argument("--profile", type=str, choices=list_profiles(), default="baseline", help=profile_help_text())
    parser.add_argument("--input", type=str, default=None, help="入力ファイル (csv/json/jsonl)。指定しない場合は旧来のemotion_prompts*.jsonを探索。")
    parser.add_argument("--output", type=str, default=None, help="出力JSONLパス。未指定ならプロファイル既定パスに保存。")
    parser.add_argument("--text-field", type=str, default="text", help="CSVにおける本文列名")
    parser.add_argument("--label-field", type=str, default="label", help="CSVにおけるラベル列名（emotion列があればそちら優先）")
    parser.add_argument("--prefer-prompts", action="store_true", help="inputが無い場合に旧来のemotion_prompts*.jsonを使用")
    parser.add_argument("--emotions", type=str, nargs="*", default=list(EMOTION_LABELS), help="使用する感情ラベル（フィルタ用）")

    args = parser.parse_args()
    context = ProjectContext(args.profile)
    output_path = Path(args.output) if args.output else context.dataset_path()
    input_path = Path(args.input) if args.input else None

    build_dataset(
        input_path=input_path,
        output_path=output_path,
        profile=args.profile,
        text_field=args.text_field,
        label_field=args.label_field,
        prefer_prompts=args.prefer_prompts,
        allowed_emotions=args.emotions,
    )


if __name__ == "__main__":
    main()
