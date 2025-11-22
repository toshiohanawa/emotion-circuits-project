"""
このプロジェクトの標準パイプラインにおける Phase 2（活性抽出）のスクリプト。
baseline_smoke（少数サンプル）と baseline（1感情225件前後を想定）に対応し、
model_registry / activation_api を用いて resid_pre/resid_post を一貫フォーマットで保存する。
旧バージョンのスクリプトは整理済みで、本スクリプトが正式なルート。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Sequence
import sys

from src.config.project_profiles import EMOTION_LABELS, list_profiles
from src.data.dataset_loader import load_dataset_for_profile
from src.models.activation_api import get_activations, save_activation_batch
from src.models.model_registry import get_model_spec, list_large_models, list_model_names, list_small_models
from src.utils.project_context import ProjectContext, profile_help_text
from src.utils.timing import record_phase_timing


def _select_prompts_per_emotion(rows: List[Dict[str, str]], max_per_emotion: int | None) -> tuple[List[str], List[str]]:
    grouped: Dict[str, List[str]] = {emo: [] for emo in EMOTION_LABELS}
    label_grouped: Dict[str, List[str]] = {emo: [] for emo in EMOTION_LABELS}
    for row in rows:
        emo = row["emotion"]
        if emo not in grouped:
            continue
        bucket = grouped[emo]
        if max_per_emotion is None or len(bucket) < max_per_emotion:
            bucket.append(row["text"])
            label_grouped[emo].append(emo)
    merged: List[str] = []
    merged_labels: List[str] = []
    for emo in EMOTION_LABELS:
        merged.extend(grouped[emo])
        merged_labels.extend(label_grouped[emo])
    return merged, merged_labels


def main():
    parser = argparse.ArgumentParser(description="Phase2: データセット上で活性(residual)を抽出する。")
    parser.add_argument("--profile", type=str, choices=list_profiles(), default="baseline", help=profile_help_text())
    parser.add_argument("--model", type=str, choices=list_model_names(), required=True, help="対象モデル名")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="取得する層番号 (指定なしで全層ヒントを使用)")
    parser.add_argument("--device", type=str, default=None, help="cuda/mps/cpu など")
    parser.add_argument("--hook-pos", type=str, choices=["both", "resid_pre", "resid_post"], default="both", help="取得する位置（大型モデルはresid_postのみ）")
    parser.add_argument("--max-samples-per-emotion", type=int, default=None, help="感情ごとの最大サンプル数（小規模テストに便利）")
    parser.add_argument("--batch-size", type=int, default=16, help="活性抽出のバッチサイズ（デフォルト: 16）")
    args = parser.parse_args()

    spec = get_model_spec(args.model)
    ctx = ProjectContext(args.profile)
    rows = load_dataset_for_profile(args.profile)
    prompts, labels = _select_prompts_per_emotion(rows, args.max_samples_per_emotion)

    if not prompts:
        raise ValueError("プロンプトが読み込めませんでした。データセットを確認してください。")

    layers: Sequence[int]
    if args.layers:
        layers = args.layers
    else:
        if spec.n_layers_hint is None:
            raise ValueError("層数が不明なモデルです。--layers を指定してください。")
        layers = list(range(spec.n_layers_hint))

    from src.utils.device import get_default_device_str

    device = args.device or get_default_device_str()

    print(f"✓ プロファイル: {args.profile} / モデル: {spec.pretty_name} / サンプル: {len(prompts)}")
    print(f"  - 層: {list(layers)}")
    
    phase_started = time.perf_counter()
    start_time = time.time()
    print(f"[Phase 2] 活性抽出を開始...")
    batch = get_activations(spec, prompts=prompts, labels=labels, layers=layers, device=device, hook_pos=args.hook_pos, batch_size=args.batch_size)
    elapsed = time.time() - start_time
    print(f"[Phase 2] 活性抽出完了: {elapsed:.2f}秒 ({elapsed/60:.2f}分)")

    out_dir = ctx.results_dir() / "activations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.name}.pkl"
    
    start_time = time.time()
    print(f"[Phase 2] 結果を保存中...")
    save_activation_batch(batch, out_path)
    elapsed = time.time() - start_time
    print(f"[Phase 2] 保存完了: {elapsed:.2f}秒")

    record_phase_timing(
        context=ctx,
        phase="phase2",
        started_at=phase_started,
        model=spec.name,
        device=device,
        samples=len(prompts),
        metadata={
            "layers": list(layers),
            "hook_pos": args.hook_pos,
            "batch_size": args.batch_size,
            "max_samples_per_emotion": args.max_samples_per_emotion,
            "output_path": str(out_path),
        },
        cli_args=sys.argv[1:],
    )

    print("✓ Phase2 完了")


if __name__ == "__main__":
    main()
