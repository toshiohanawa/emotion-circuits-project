"""
現行パイプライン用の簡易コンシステンシチェック。
- データセット行数の目安（baseline≈900行=225/感情×4、baseline_smokeは少数）
- 主要CLIが存在するかの確認
結果アーティファクトは再生成前提のため存在チェックは行わない。
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.config.project_profiles import EMOTION_LABELS, get_profile, list_profiles


def check_dataset_sizes() -> bool:
    ok = True
    for profile_name in list_profiles():
        profile = get_profile(profile_name)
        path = profile.dataset_path()
        if not path.exists():
            print(f"[WARN] データセットが見つかりません: {path}")
            ok = False
            continue
        count = sum(1 for _ in path.open())
        if profile_name == "baseline":
            target = 225 * len(EMOTION_LABELS)
            if count < target:
                print(f"[WARN] baseline の行数が目標より少ないかもしれません: {count} 行 (目安 {target} 行)")
                ok = False
        elif profile_name == "baseline_smoke":
            if count > 50:
                print(f"[WARN] baseline_smoke の行数が多すぎます: {count} 行（少数での配線確認用を想定）")
                ok = False
    return ok


def check_cli_presence() -> bool:
    """主要CLIファイルが存在するかを確認（中身の検証は行わない）。"""
    required = [
        "src/analysis/run_phase2_activations.py",
        "src/analysis/run_phase3_vectors.py",
        "src/analysis/run_phase4_alignment.py",
        "src/analysis/run_phase5_residual_patching.py",
        "src/analysis/run_phase6_head_patching.py",
        "src/analysis/run_phase6_head_screening.py",
        "src/analysis/run_phase7_statistics.py",
    ]
    ok = True
    for rel in required:
        if not Path(rel).exists():
            print(f"[ERR] CLI が見つかりません: {rel}")
            ok = False
    return ok


def main():
    checks = [
        ("dataset_sizes", check_dataset_sizes),
        ("cli_presence", check_cli_presence),
    ]
    success = True
    for name, fn in checks:
        if not fn():
            success = False
    if success:
        print("✓ コンシステンシチェック通過")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
