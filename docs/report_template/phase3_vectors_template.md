# Phase 3 — 感情ベクトル・サブスペース構築（run_phase3_vectors）

## 🎯 目的
- Phase2で抽出した活性から、感情ごとの差分ベクトルを計算し、層別に平均ベクトルとPCAサブスペースを得る。

## 📦 生成物
- `results/<profile>/emotion_vectors/<model>_vectors_token_based.pkl`
- `results/<profile>/emotion_subspaces/<model>_subspaces.pkl`
- 本レポート（例: `docs/phase3_report.md`）

## 🚀 実行コマンド例
```bash
python -m src.analysis.run_phase3_vectors \
  --profile baseline \
  --model gpt2_small \
  --n-components 8 \
  --use-torch \
  --device mps
```
※ 配線確認なら baseline_smoke で少数サンプル実行。
※ `--use-torch --device mps` でGPU/MPS加速PCAを使用（従来のsklearn版は `--no-use-torch`）。

## 📄 レポート項目
1. 実行設定
   - プロファイル / モデル / n_components / 入力活性ファイル
2. 感情ベクトルの概要
   - 層ごとのノルム、感情間cos類似など
3. サブスペース（PCA）の概要
   - explained_variance_ratio、主成分の解釈メモ
4. 次のアクション
   - Phase4 アライメント/Phase5 パッチングで利用するファイルパス
