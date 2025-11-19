# Phase 4 — サブスペースアライメント

## 🎯 目的
- モデル間で感情サブスペースを比較し、線形写像（例: Procrustes）で overlap を改善できるか検証する。
- k 次元サブスペースごとの overlap をレイヤー別に整理し、共通構造の有無を確認する。

## 📦 生成物
- `results/<profile>/alignment/<modelA>_vs_<modelB>_token_based_full.pkl`
- レイヤー×k 別の overlap 集計（必要に応じて図表化）
- 本レポート（例: `docs/phase4_alignment_report.md`）

## 🚀 実行コマンド例
```bash
python -m src.analysis.run_phase4_alignment \
  --profile baseline \
  --model-a gpt2_small \
  --model-b gpt2_small \
  --k-max 8
```
※ baseline_smoke で配線確認する場合は `--profile baseline_smoke` を指定。

## 📄 レポート項目
1. 実行設定
   - プロファイル: [baseline / baseline_smoke]
   - モデルA/モデルB: [名前]
   - 層リスト: [例: 0-11]
   - k の範囲: [例: 1–8]
2. overlap 結果（before/after）
   - レイヤー別・k別の overlap_before / overlap_after の表またはヒートマップ
   - 改善が大きい層/k をコメント
3. 解釈
   - 共通サブスペースが強い層やkが存在するか
   - self-alignment と cross-model の違い
4. 次のアクション
   - k 値や層の絞り込み
   - 大型モデルへの適用計画
