#!/bin/bash
# Phase7: 全モデル統合統計解析を実行
# 使用方法: bash scripts/run_phase7_all.sh <profile> [n_jobs]

set -e

PROFILE="${1:-baseline}"
N_JOBS="${2:-8}"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "[Phase 7] 全モデル統合統計解析を開始..."

echo "  - effect mode"
python3 -m src.analysis.run_phase7_statistics \
    --profile "$PROFILE" \
    --mode effect \
    --phase-filter residual,head \
    --n-jobs "$N_JOBS"

echo "  - power mode"
python3 -m src.analysis.run_phase7_statistics \
    --profile "$PROFILE" \
    --mode power \
    --phase-filter residual,head \
    --n-jobs "$N_JOBS"

echo "  - k mode"
python3 -m src.analysis.run_phase7_statistics \
    --profile "$PROFILE" \
    --mode k \
    --phase-filter residual,head

echo "[Phase 7] 全モデル統合統計解析が完了しました"

