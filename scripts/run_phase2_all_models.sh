#!/bin/bash
# Phase2: 全モデルで活性抽出を実行
# 使用方法: bash scripts/run_phase2_all_models.sh <profile> <device>

set -e

PROFILE="${1:-baseline}"
DEVICE="${2:-cuda}"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "[Phase 2] 小型モデル3つの活性抽出を開始..."

# 小型モデル（各12層）
echo "  - gpt2_small (層0-11)"
python3 -m src.analysis.run_phase2_activations --profile "$PROFILE" --model gpt2_small --device "$DEVICE" --batch-size 16

echo "  - pythia-160m (層0-11)"
python3 -m src.analysis.run_phase2_activations --profile "$PROFILE" --model pythia-160m --device "$DEVICE" --batch-size 16

echo "  - gpt-neo-125m (層0-11)"
python3 -m src.analysis.run_phase2_activations --profile "$PROFILE" --model gpt-neo-125m --device "$DEVICE" --batch-size 16

echo "[Phase 2] 中規模モデル3つの活性抽出を開始..."

# 中規模モデル
echo "  - llama3_8b (層0-31)"
python3 -m src.analysis.run_phase2_activations --profile "$PROFILE" --model llama3_8b --device "$DEVICE" --batch-size 8

echo "  - gemma3_12b (層0-47) - スキップ（メモリ不足のため、後で個別実行）"
# gemma3_12bは非常に大きく、GPUメモリが不足する可能性があるため、個別に実行することを推奨
# python3 -m src.analysis.run_phase2_activations --profile "$PROFILE" --model gemma3_12b --device "$DEVICE" --batch-size 1

echo "  - qwen3_8b (層0-35)"
python3 -m src.analysis.run_phase2_activations --profile "$PROFILE" --model qwen3_8b --device "$DEVICE" --batch-size 8

echo "[Phase 2] 全モデルの活性抽出が完了しました"

