#!/bin/bash
# 包括的多モデル実験の全Phaseを順次実行するスクリプト
# 使用方法: bash scripts/run_comprehensive_experiment.sh

set -e  # エラー時に停止

PROFILE="baseline"
DEVICE="cuda"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=========================================="
echo "包括的多モデル実験を開始します"
echo "プロファイル: $PROFILE"
echo "デバイス: $DEVICE"
echo "=========================================="

# Phase 1: データセット構築
echo ""
echo "[Phase 1] データセット構築を開始..."
python3 -m src.data.build_dataset --profile "$PROFILE" --input data/emotion_dataset.jsonl
echo "[Phase 1] 完了"

# Phase 2: 活性抽出（全モデル）
echo ""
echo "[Phase 2] 活性抽出を開始（全モデル）..."
bash scripts/run_phase2_all_models.sh "$PROFILE" "$DEVICE"
echo "[Phase 2] 完了"

# Phase 3: 感情ベクトル計算（全モデル）
echo ""
echo "[Phase 3] 感情ベクトル計算を開始（全モデル）..."
bash scripts/run_phase3_all_models.sh "$PROFILE" "$DEVICE"
echo "[Phase 3] 完了"

# Phase 4: モデル間アライメント（9組み合わせ）
echo ""
echo "[Phase 4] モデル間アライメントを開始（9組み合わせ）..."
bash scripts/run_phase4_all_combinations.sh "$PROFILE" "$DEVICE"
echo "[Phase 4] 完了"

# Phase 5: 残差パッチング（全モデル）
echo ""
echo "[Phase 5] 残差パッチングを開始（全モデル）..."
bash scripts/run_phase5_all_models.sh "$PROFILE" "$DEVICE"
echo "[Phase 5] 完了"

# Phase 6a: Head Screening（全モデル）
echo ""
echo "[Phase 6a] Head Screeningを開始（全モデル）..."
bash scripts/run_phase6a_all_models.sh "$PROFILE" "$DEVICE"
echo "[Phase 6a] 完了"

# Phase 6b: Head Patching（全モデル）
echo ""
echo "[Phase 6b] Head Patchingを開始（全モデル）..."
bash scripts/run_phase6b_all_models.sh "$PROFILE" "$DEVICE"
echo "[Phase 6b] 完了"

# Phase 7: 統計解析（全モデル統合）
echo ""
echo "[Phase 7] 統計解析を開始（全モデル統合）..."
bash scripts/run_phase7_all.sh "$PROFILE"
echo "[Phase 7] 完了"

echo ""
echo "=========================================="
echo "包括的多モデル実験が完了しました"
echo "=========================================="

