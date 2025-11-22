#!/bin/bash
# qwen3_8bの層18-35のプロセス完了を待つスクリプト

PROCESS_ID=180242
CHECK_INTERVAL=300  # 5分ごとに確認

echo "qwen3_8b 層18-35のプロセス完了を待機中..."
echo "プロセスID: $PROCESS_ID"
echo "確認間隔: ${CHECK_INTERVAL}秒（5分）"
echo ""

while ps -p $PROCESS_ID > /dev/null 2>&1; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - プロセス実行中..."
    sleep $CHECK_INTERVAL
done

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S') - プロセス完了！"
echo "結果ファイルを確認中..."

# 結果ファイルの確認
RESULT_FILE="results/baseline/screening/head_scores_qwen3_8b.json"
if [ -f "$RESULT_FILE" ]; then
    python3 << PYTHON_EOF
import json
from pathlib import Path

file = Path("$RESULT_FILE")
with file.open() as f:
    data = json.load(f)
    head_scores = data.get('head_scores', [])
    layers = sorted(set(h['layer'] for h in head_scores))
    print(f"結果ファイル: {file}")
    print(f"総ヘッド数: {len(head_scores)}")
    print(f"層の範囲: {min(layers)} - {max(layers)}")
    print(f"層数: {len(layers)} (期待値: 36層)")
    if len(layers) == 36:
        print("✅ 全36層が含まれています")
    else:
        missing = set(range(36)) - set(layers)
        print(f"⚠ 欠けている層: {sorted(missing)}")
PYTHON_EOF
else
    echo "⚠ 結果ファイルが見つかりません"
fi
