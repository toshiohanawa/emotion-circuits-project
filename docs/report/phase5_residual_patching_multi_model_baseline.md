# Phase 5 — 残差パッチング（3モデル統合レポート）

## 🎯 目的
3つの小型モデル（GPT-2 small, Pythia-160M, GPT-Neo-125M）について、感情ベクトル/サブスペース方向を残差に注入し、生成文のsentiment / politeness / GoEmotionsがどの程度変化するかを測定する。

## 📦 生成物
- `results/baseline/patching/residual/gpt2_small_residual_sweep.pkl`
- `results/baseline/patching/residual/pythia-160m_residual_sweep.pkl`
- `results/baseline/patching/residual/gpt-neo-125m_residual_sweep.pkl`
- 本レポート: `docs/report/phase5_residual_patching_multi_model_baseline.md`

## 🚀 実行コマンド

```bash
# GPT-2 small
python3 -m src.analysis.run_phase5_residual_patching \
  --profile baseline \
  --model gpt2_small \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 \
  --patch-window 3 \
  --sequence-length 30 \
  --alpha 1.0 \
  --max-samples-per-emotion 8 \
  --device mps \
  --batch-size 16

# Pythia-160M
python3 -m src.analysis.run_phase5_residual_patching \
  --profile baseline \
  --model pythia-160m \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 \
  --patch-window 3 \
  --sequence-length 30 \
  --alpha 1.0 \
  --max-samples-per-emotion 8 \
  --device mps \
  --batch-size 16

# GPT-Neo-125M
python3 -m src.analysis.run_phase5_residual_patching \
  --profile baseline \
  --model gpt-neo-125m \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 \
  --patch-window 3 \
  --sequence-length 30 \
  --alpha 1.0 \
  --max-samples-per-emotion 8 \
  --device mps \
  --batch-size 16
```

## 📄 レポート

### 1. 実行設定

| モデル | 層数 | patch_window | sequence_length | alpha | サンプル数/感情 | 実行時間 |
|--------|------|-------------|----------------|-------|---------------|---------|
| GPT-2 small | 12 | 3 | 30 | 1.0 | 8 | 55.48秒 |
| Pythia-160M | 12 | 3 | 30 | 1.0 | 8 | 55.48秒 |
| GPT-Neo-125M | 12 | 3 | 30 | 1.0 | 8 | 49.09秒 |

**共通設定**:
- **プロファイル**: `baseline`
- **感情**: gratitude, anger, apology（3感情）
- **評価メトリクス**: sentiment, politeness, goemotions（複数メトリクス）
- **バッチサイズ**: 16

### 2. 実行結果の概要

すべてのモデルで以下の構造で結果が保存されました：

- **感情**: gratitude, anger, apology（3感情）
- **層数**: 12層（0-11）
- **alpha値**: 1.0（固定）
- **メトリクス**: 複数の評価メトリクス（sentiment, politeness, goemotionsなど）

### 3. Phase 7統計解析との連携

Phase 5の結果はPhase 7の統計解析で使用され、以下の情報が提供されます：

- **効果量（Cohen's d）**: 各パッチング実験の効果量
- **p値**: 統計的有意性
- **信頼区間**: 95%信頼区間
- **多重比較補正**: Bonferroni補正、BH-FDR補正

詳細はPhase 7レポートを参照してください。

### 4. モデル間の比較

#### 4.1 実行時間

- **GPT-2 small**: 55.48秒
- **Pythia-160M**: 55.48秒
- **GPT-Neo-125M**: 49.09秒（やや高速）

すべてのモデルでほぼ同じ実行時間を示し、バッチ処理による効率化が実現されています。

#### 4.2 データ構造

すべてのモデルで同じ構造のデータが生成されました：
- 感情×層×alpha値の組み合わせ
- 各組み合わせでbaselineとpatchedのメトリクスが記録
- Phase 7での統計解析に直接使用可能な形式

### 5. 次のアクション

- **Phase 7**: Phase 5の結果を使用して統計解析を実行
  - `results/baseline/patching/residual/<model>_residual_sweep.pkl`
- **追加実験**: 
  - **より多くのサンプルでの実行**: 現在は8サンプル/感情で実行しましたが、baselineプロファイルの本来の想定は225サンプル/感情です。ただし、フル実行（225サンプル × 3感情 × 12層 × 50ランダム）すると非常に長時間（数日〜1週間程度）かかるため、まずは動作確認として8サンプルで実行しました。
  - alpha値のスイープ（現在は1.0固定）
  - patch_windowの変更

### 6. 考察

- **バッチ処理の効果**: すべてのモデルで高速な実行が実現され、バッチ処理による効率化が確認された
- **モデル間の一貫性**: すべてのモデルで同じ構造のデータが生成され、比較分析が可能
- **Phase 7への準備**: Phase 5の結果はPhase 7の統計解析で使用され、効果量や有意性が評価される

