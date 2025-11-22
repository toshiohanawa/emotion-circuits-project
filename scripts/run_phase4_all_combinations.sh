#!/bin/bash
# Phase4: 全モデル間アライメント（9組み合わせ）を実行
# 使用方法: bash scripts/run_phase4_all_combinations.sh <profile> <device>

set -e

PROFILE="${1:-baseline}"
DEVICE="${2:-cuda}"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "[Phase 4] 全モデル間アライメントを開始..."
echo "  ⚠ 注意: 異なるd_modelを持つモデル間のアライメントは現在サポートされていません"
echo "  Phase4はスキップし、Phase5以降を実行します"
echo "[Phase 4] スキップ（後で個別に対応）"

