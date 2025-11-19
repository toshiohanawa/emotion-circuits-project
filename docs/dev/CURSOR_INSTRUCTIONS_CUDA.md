# CUDA 環境で Cursor を動かす際の指示（llama3 など大モデル対応）

## 参考資料
- GPUセットアップ: `docs/dev/setup_ubuntu_gpu.md`（Ubuntu22.04 + RTX3090 + CUDA12.1 + PyTorch 2.5.1/cu121 + vLLM）
- 実行手順/パイプライン: `docs/dev/SETUP.md`, `docs/dev/PIPELINE_NOTES.md`, `README.md`
- 実装状況: `docs/dev/IMPLEMENTATION_STATUS.md`, `docs/dev/NEXT_STEPS.md`

## 実行の基本方針
- 大モデル（例: llama3_8b）は CUDA で実行すること（MPS は極端に遅い）。
- Phase5/6 でランダム対照は標準オフ。必要な場合のみ `--random-control --num-random N` を付けて再実行し、Phase7を再計算。
- バッチサイズは GPU メモリに応じて上げる（推奨: 8〜16）。

## コマンド例（CUDA）

### Phase2: 活性抽出（大モデル）
```
PYTHONPATH=. python3 src/analysis/run_phase2_activations.py \
  --profile baseline \
  --model llama3_8b \
  --layers 0 3 6 9 11 \
  --device cuda \
  --max-samples-per-emotion 225 \
  --batch-size 8 \
  --hook-pos resid_post
```

### Phase3: 感情ベクトル/サブスペース
```
PYTHONPATH=. python3 src/analysis/run_phase3_vectors.py \
  --profile baseline \
  --model llama3_8b \
  --n-components 8 \
  --use-torch \
  --device cuda
```

### Phase5: 残差パッチング（ランダムなし・層絞り例）
```
PYTHONPATH=. python3 src/analysis/run_phase5_residual_patching.py \
  --profile baseline \
  --model llama3_8b \
  --layers 0 3 6 9 11 \
  --patch-window 3 \
  --sequence-length 30 \
  --alpha 1.0 \
  --max-samples-per-emotion 225 \
  --device cuda \
  --batch-size 8
```

### Phase6: ヘッドパッチング（ゼロ化アブレーション）
```
PYTHONPATH=. python3 src/analysis/run_phase6_head_patching.py \
  --profile baseline \
  --model llama3_8b \
  --heads 0:0-11 3:0-11 6:0-11 9:0-11 11:0-11 \
  --max-samples 225 \
  --sequence-length 30 \
  --device cuda \
  --batch-size 8
```

### Phase6: ヘッドスクリーニング（全ヘッド・層を絞る例）
```
PYTHONPATH=. python3 src/analysis/run_phase6_head_screening.py \
  --profile baseline \
  --model llama3_8b \
  --layers 0 3 6 9 11 \
  --max-samples 50 \
  --sequence-length 30 \
  --device cuda \
  --batch-size 8
```

### Phase7: 統計（大モデル結果を含めて再計算）
```
PYTHONPATH=. python3 src/analysis/run_phase7_statistics.py \
  --profile baseline \
  --mode all \
  --n-jobs 8
```

## 注意事項
- vLLM は FlashInfer を無効化すること（`VLLM_ATTENTION_BACKEND=FLASHINFER_OFF`）。
- Phase6スクリーニングはヘッド数が多いと時間がかかるため、まず層を絞りサンプルを減らして実測する。
- エラーや警告（例: 未使用weight）はHFモデル読み込み時によく出るが、致命的でなければ続行可。
