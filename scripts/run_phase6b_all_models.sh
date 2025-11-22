#!/bin/bash
# Phase6b: 全モデルでHead Patchingを実行
# Head Screeningの結果から上位N個のヘッドを選択して実行
# 使用方法: bash scripts/run_phase6b_all_models.sh <profile> <device> [top_n_heads]

set -e

PROFILE="${1:-baseline}"
DEVICE="${2:-cuda}"
TOP_N="${3:-10}"  # デフォルトで上位10個のヘッドを選択

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "[Phase 6b] 全モデルでHead Patchingを開始（上位${TOP_N}個のヘッド）..."

# Head Screeningの結果から上位N個のヘッドを抽出するPythonスクリプトを実行
MODELS=("gpt2_small" "pythia-160m" "gpt-neo-125m" "llama3_8b" "gemma3_12b" "qwen3_8b")

for model in "${MODELS[@]}"; do
    echo "  - $model"
    
    # Head Screeningの結果ファイルを確認
    SCREENING_FILE="results/${PROFILE}/screening/head_scores_${model}.json"
    
    if [ ! -f "$SCREENING_FILE" ]; then
        echo "    ⚠ 警告: $SCREENING_FILE が見つかりません。スキップします。"
        continue
    fi
    
    # Pythonスクリプトで上位N個のヘッドを抽出
    HEADS=$(python3 <<EOF
import json
from pathlib import Path

screening_file = Path("$SCREENING_FILE")
if not screening_file.exists():
    print("")
    exit(1)

with screening_file.open() as f:
    data = json.load(f)

# head_scoresから上位N個を選択
head_scores = data.get("head_scores", [])
if not head_scores:
    print("")
    exit(1)

# delta_meanの絶対値でソート
sorted_heads = sorted(head_scores, key=lambda x: abs(x.get("delta_mean", 0)), reverse=True)
top_n = int("$TOP_N")
top_heads = sorted_heads[:top_n]

# "layer:head"形式で出力
head_list = [f"{h['layer']}:{h['head']}" for h in top_heads]
print(" ".join(head_list))
EOF
)
    
    if [ -z "$HEADS" ]; then
        echo "    ⚠ 警告: ヘッドを抽出できませんでした。スキップします。"
        continue
    fi
    
    echo "    選択されたヘッド: $HEADS"
    
    # Head Patchingを実行
    python3 -m src.analysis.run_phase6_head_patching \
        --profile "$PROFILE" \
        --model "$model" \
        --heads $HEADS \
        --device "$DEVICE" \
        --batch-size 16
done

echo "[Phase 6b] 全モデルのHead Patchingが完了しました"

