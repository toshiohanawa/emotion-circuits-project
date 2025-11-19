# Phase 2 — 活性抽出（run_phase2_activations）

## 🎯 目的
- baseline/baseline_smoke データセット上で residual stream（必要に応じて resid_pre/resid_post）を抽出し、後続フェーズで再利用できる形式で保存する。

## 📦 生成物
- `results/<profile>/activations/<model>.pkl`
- 本レポート（例: `docs/phase2_report.md`）

## 🚀 実行コマンド例
```bash
python -m src.analysis.run_phase2_activations \
  --profile baseline \
  --model gpt2_small \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 \
  --device mps \
  --batch-size 32 \
  --max-samples-per-emotion 225
```
※ 配線確認なら baseline_smoke と少数サンプルで実行。
※ `--device mps` でApple Silicon加速、`--batch-size` でメモリに応じた並列処理が可能。

## 📄 レポート項目
1. 実行設定
   - プロファイル / モデル / 層 / サンプル数 / デバイス / resid_pre or resid_post
2. 抽出したテンソルの形状
   - resid_pre/resid_post の shape（batch×layer×seq×d_model）
   - token_ids/token_strings の保存有無
3. 実行コスト
   - 所要時間、メモリ使用のメモ
4. 次のアクション
   - Phase3 以降で利用するファイルパスを記載
