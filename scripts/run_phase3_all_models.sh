#!/bin/bash
# Phase3: 全モデルで感情ベクトル計算を実行
# 使用方法: bash scripts/run_phase3_all_models.sh <profile> <device>

set -e

PROFILE="${1:-baseline}"
DEVICE="${2:-cuda}"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "[Phase 3] 全モデル6つの感情ベクトル計算を開始..."

MODELS=("gpt2_small" "pythia-160m" "gpt-neo-125m" "llama3_8b" "gemma3_12b" "qwen3_8b")

for model in "${MODELS[@]}"; do
    echo "  - $model"
    python3 -m src.analysis.run_phase3_vectors --profile "$PROFILE" --model "$model" --n-components 8 --use-torch --device "$DEVICE"
done

echo "[Phase 3] 全モデルの感情ベクトル計算が完了しました"

