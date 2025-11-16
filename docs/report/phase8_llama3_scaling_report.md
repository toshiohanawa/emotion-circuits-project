# Phase 8: 中規模モデル（Llama3 8B）の感情サブスペース整合性解析
_Last updated: 2025-11-16_

## 🎯 目的

中規模モデル（Llama3 8B）とGPT-2 small間の感情サブスペースの整合性を解析し、
線形写像によるアライメント前後のoverlapを比較しました。

## 📦 生成物

- `phase8_llama3_scaling_report.md` - 本レポート

## 🚀 実行コマンド

```bash
python -m src.analysis.summarize_phase8_llama3 \
  --profile baseline \
  --alignment-path results/baseline/alignment/gpt2_vs_llama3_8b_token_based_full.pkl \
  --output-csv results/baseline/statistics/phase8_llama3_alignment_summary.csv \
  --write-report docs/report/phase8_llama3_scaling_report.md
```

## 📄 レポート項目

### 1. 実験設定

- **モデル**: GPT-2 small (124M) (source), Llama3 8B (Meta-Llama-3.1-8B) (target)
- **プロファイル**: baseline
- **データ**: 感情プロンプト × 4感情（gratitude / anger / apology / neutral）
- **手法概要**: token-based emotion vector, 多サンプルPCA (k=8), neutralサブスペースからの線形写像, GPT-2 vs Llama3 のsubspace overlap (before/after)
- **対象層**: 0–11 (12層)

### 2. ハイパーパラメータ

- **PCA次元数 (k)**: 8
- **対象層**: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
- **max-samples-per-emotion**: N/A（メタデータに記録されていません）

### 3. 結果概要

| Layer | Mean overlap (before) | Mean overlap (after) | Δ (after - before) |
|-------|-----------------------|----------------------|--------------------|
| 0 | 0.000216 | 0.132543 | 0.132327 |
| 1 | 0.000197 | 0.517283 | 0.517086 |
| 2 | 0.000243 | 0.485675 | 0.485432 |
| 3 | 0.000207 | 0.390680 | 0.390473 |
| 4 | 0.000206 | 0.349699 | 0.349494 |
| 5 | 0.000225 | 0.383109 | 0.382884 |
| 6 | 0.000224 | 0.362916 | 0.362692 |
| 7 | 0.000209 | 0.409811 | 0.409603 |
| 8 | 0.000222 | 0.451717 | 0.451495 |
| 9 | 0.000208 | 0.533897 | 0.533689 |
| 10 | 0.000234 | 0.600778 | 0.600544 |
| 11 | 0.000253 | 0.703222 | 0.702969 |

### 4. 簡単な考察

- **改善が最も大きい層**: Layer 11 (Δ = 0.702969)
- **改善が最も小さい層**: Layer 0 (Δ = 0.132327)
- **平均overlap (before)**: 0.000220
- **平均overlap (after)**: 0.443444
- **平均改善幅**: 0.443224

線形写像によるアライメント後、すべての層でoverlapが改善しているかどうかを確認してください。
中間層で特に改善が大きい場合、感情表現の抽象化が進んでいる可能性があります。

小型モデル（Phase 6）との定性的な比較:
- Phase 6ではGPT-2とPythia-160M間の比較を行いましたが、本Phase 8ではより大きなモデル（Llama3 8B）との比較を行っています。
- モデルサイズが大きくなることで、感情サブスペースの表現がどのように変化するかを観察できます。

### 5. 今後のステップ

- **Gemma2 / Qwen への展開**: 他の大規模モデルでも同様の解析を実施し、モデル間の一般性を検証
- **サンプル数やkの増加**: より多くのサンプルやPCA次元数で再実験し、結果の頑健性を確認
- **Phase 5 / 7.5 の統計パイプラインへの組み込み**: 効果量や検出力分析を実施し、統計的有意性を検証
- **複数モデル間の比較**: GPT-2、Pythia-160M、Llama3 8Bの3モデル間での比較解析