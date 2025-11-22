#!/bin/bash
# Phase6a: 全モデルでHead Screeningを実行（層分割対応）
# 使用方法: bash scripts/run_phase6a_all_models.sh <profile> <device>

set -e

PROFILE="${1:-baseline}"
DEVICE="${2:-cuda}"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "[Phase 6a] 全モデルでHead Screeningを開始..."

# 小型モデル（全層を一度に実行）
echo "  - 小型モデル3つ（全層）"
echo "    gpt2_small (層0-11)"
python3 -m src.analysis.run_phase6_head_screening \
    --profile "$PROFILE" \
    --model gpt2_small \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 \
    --device "$DEVICE" \
    --batch-size 16

echo "    pythia-160m (層0-11)"
python3 -m src.analysis.run_phase6_head_screening \
    --profile "$PROFILE" \
    --model pythia-160m \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 \
    --device "$DEVICE" \
    --batch-size 16

echo "    gpt-neo-125m (層0-11)"
python3 -m src.analysis.run_phase6_head_screening \
    --profile "$PROFILE" \
    --model gpt-neo-125m \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 \
    --device "$DEVICE" \
    --batch-size 16

# 中規模モデル（層を分割して実行）
echo "  - 中規模モデル3つ（層分割実行）"
echo "    llama3_8b (層0-31, 16層ずつ分割)"
python3 -m src.analysis.run_phase6_head_screening \
    --profile "$PROFILE" \
    --model llama3_8b \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
    --device "$DEVICE" \
    --batch-size 4

python3 -m src.analysis.run_phase6_head_screening \
    --profile "$PROFILE" \
    --model llama3_8b \
    --layers 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
    --device "$DEVICE" \
    --batch-size 4

echo "    gemma3_12b (層0-47, 16層ずつ分割)"
python3 -m src.analysis.run_phase6_head_screening \
    --profile "$PROFILE" \
    --model gemma3_12b \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
    --device "$DEVICE" \
    --batch-size 4

python3 -m src.analysis.run_phase6_head_screening \
    --profile "$PROFILE" \
    --model gemma3_12b \
    --layers 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
    --device "$DEVICE" \
    --batch-size 4

python3 -m src.analysis.run_phase6_head_screening \
    --profile "$PROFILE" \
    --model gemma3_12b \
    --layers 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 \
    --device "$DEVICE" \
    --batch-size 4

echo "    qwen3_8b (層0-35, 18層ずつ分割)"
python3 -m src.analysis.run_phase6_head_screening \
    --profile "$PROFILE" \
    --model qwen3_8b \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 \
    --device "$DEVICE" \
    --batch-size 4

python3 -m src.analysis.run_phase6_head_screening \
    --profile "$PROFILE" \
    --model qwen3_8b \
    --layers 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 \
    --device "$DEVICE" \
    --batch-size 4

echo "[Phase 6a] 全モデルのHead Screeningが完了しました"

