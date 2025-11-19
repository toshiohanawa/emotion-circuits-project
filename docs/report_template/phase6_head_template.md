# Phase 6 — ヘッドパッチング / スクリーニング

## 🎯 目的
- 指定ヘッドの出力をゼロ化して因果効果を測る（head ablation/patching）。
- 各ヘッドを網羅的にアブレートし、評価指標の変化から重要度をスクリーニングする。

## 📦 生成物
- `results/<profile>/patching/head_patching/<model>_head_ablation.pkl`（run_phase6_head_patching）
- `results/<profile>/screening/head_scores_<model>.json`（run_phase6_head_screening）
- 本レポート（例: `docs/phase6_report.md`）

## 🚀 実行コマンド例
```bash
# ヘッドパッチング（例: 複数層の全ヘッド）
python -m src.analysis.run_phase6_head_patching \
  --profile baseline \
  --model gpt2_small \
  --heads 0:0-11 3:0-11 6:0-11 9:0-11 11:0-11 \
  --max-samples 225 \
  --sequence-length 30 \
  --device mps \
  --batch-size 8

# ヘッドスクリーニング（全層）
python -m src.analysis.run_phase6_head_screening \
  --profile baseline \
  --model gpt2_small \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 \
  --max-samples 225 \
  --sequence-length 30 \
  --device mps \
  --batch-size 8
```
※ baseline_smoke で少数サンプル実行してから本番を推奨。
※ `--device mps` でApple Silicon加速、`--batch-size` でメモリに応じた並列処理が可能。

## 📄 レポート項目
1. 実行設定
   - プロファイル / モデル / 層・ヘッド指定 / サンプル数 / sequence_length
2. スクリーニング結果
   - head_scores_<model>.json から、メトリクス別 delta_mean/delta_std 上位ヘッドを表や図で示す。
3. パッチング結果
   - ablation 前後のメトリクス比較（sentiment/politeness/goemotionsなど）
   - 代表的な生成文の差分
4. 次のアクション
   - 追加で試すヘッド、組み合わせ、他モデルへの展開
