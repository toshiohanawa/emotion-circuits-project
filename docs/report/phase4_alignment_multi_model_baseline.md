# Phase 4 — サブスペースアライメント（3モデル統合レポート）

## 🎯 目的
GPT-2 smallを基準として、Pythia-160MとGPT-Neo-125Mとの間で感情サブスペースを比較し、線形写像（Procrustes）でoverlapを改善できるか検証する。k次元サブスペースごとのoverlapをレイヤー別に整理し、共通構造の有無を確認する。

## 📦 生成物
- `results/baseline/alignment/gpt2_small_vs_pythia-160m_token_based_full.pkl`
- `results/baseline/alignment/gpt2_small_vs_gpt-neo-125m_token_based_full.pkl`
- 本レポート: `docs/report/phase4_alignment_multi_model_baseline.md`

## 🚀 実行コマンド

```bash
# GPT-2 vs Pythia-160M
python3 -m src.analysis.run_phase4_alignment \
  --profile baseline \
  --model-a gpt2_small \
  --model-b pythia-160m \
  --k-max 8 \
  --use-torch \
  --device mps

# GPT-2 vs GPT-Neo-125M
python3 -m src.analysis.run_phase4_alignment \
  --profile baseline \
  --model-a gpt2_small \
  --model-b gpt-neo-125m \
  --k-max 8 \
  --use-torch \
  --device mps
```

## 📄 レポート

### 1. 実行設定

| 比較ペア | モデルA | モデルB | kの範囲 | 実行時間 |
|---------|--------|--------|---------|---------|
| GPT-2 vs Pythia | GPT-2 small | Pythia-160M | 1-8 | 0.84秒 |
| GPT-2 vs GPT-Neo | GPT-2 small | GPT-Neo-125M | 1-8 | 0.82秒 |

**共通設定**:
- **プロファイル**: `baseline`
- **層リスト**: 0-11（全12層）
- **kの範囲**: 1-8（最大8次元）
- **計算方法**: Procrustes分析（torch.linalg.pinv使用、MPS対応）

### 2. Overlap結果（before/after）

#### 2.1 GPT-2 vs Pythia-160M（k=8）

**全体統計**:
- **総結果数**: 288（12層 × 3感情 × 8 k値）
- **overlap_before平均**: 0.1355（範囲: 0.0831-0.2287）
- **overlap_after平均**: 0.1355（範囲: 0.0831-0.2287）
- **改善量平均**: -0.0000（実質的に改善なし）

**層別overlap（k=8）**:

| 層 | overlap_before | overlap_after | 改善量 |
|---|---------------|--------------|--------|
| 0 | 0.1491 | 0.1491 | 0.0000 |
| 6 | 0.1290 | 0.1290 | -0.0000 |
| 11 | 0.1202 | 0.1202 | -0.0000 |

**観察**:
- overlap値は0.12-0.15の範囲で、比較的低い
- Procrustes写像による改善は実質的にゼロ
- 層による差異は小さい（Layer 0がやや高い）

#### 2.2 GPT-2 vs GPT-Neo-125M（k=8）

**全体統計**:
- **総結果数**: 288（12層 × 3感情 × 8 k値）
- **overlap_before平均**: 0.0886（範囲: 0.0625-0.1201）
- **overlap_after平均**: 0.0886（範囲: 0.0625-0.1201）
- **改善量平均**: -0.0000（実質的に改善なし）

**層別overlap（k=8）**:

| 層 | overlap_before | overlap_after | 改善量 |
|---|---------------|--------------|--------|
| 0 | 0.0902 | 0.0902 | 0.0000 |
| 6 | 0.0924 | 0.0924 | 0.0000 |
| 11 | 0.0784 | 0.0784 | -0.0000 |

**観察**:
- overlap値は0.08-0.09の範囲で、GPT-2 vs Pythiaよりさらに低い
- Procrustes写像による改善は実質的にゼロ
- Layer 11で最も低いoverlap（0.0784）

### 3. モデル間の比較

#### 3.1 Overlap値の比較

| 比較ペア | overlap平均（k=8） | 範囲 |
|---------|------------------|------|
| GPT-2 vs Pythia-160M | 0.1355 | 0.0831-0.2287 |
| GPT-2 vs GPT-Neo-125M | 0.0886 | 0.0625-0.1201 |

**観察**:
- **GPT-2 vs Pythia**: より高いoverlap（0.1355）
- **GPT-2 vs GPT-Neo**: より低いoverlap（0.0886）
- 両方とも比較的低いoverlap値（0.1以下）を示す

#### 3.2 Procrustes写像の効果

- **改善量**: 両方の比較で実質的にゼロ
- **解釈**: 
  - 線形写像ではoverlapを改善できない
  - モデル間のサブスペース構造が大きく異なる可能性
  - または、neutralサンプルベースの写像では感情サブスペースのアライメントが不十分

### 4. 解釈

#### 4.1 共通サブスペース構造の有無

- **低いoverlap値**: すべての比較でoverlap値が0.1以下と低く、共通サブスペース構造は限定的
- **モデル間の違い**: GPT-2 vs Pythia（0.1355）とGPT-2 vs GPT-Neo（0.0886）で異なるoverlap値
- **線形アライメントの限界**: Procrustes写像による改善がゼロであり、単純な線形写像では不十分

#### 4.2 層による差異

- **GPT-2 vs Pythia**: Layer 0でやや高いoverlap（0.1491）
- **GPT-2 vs GPT-Neo**: Layer 6でやや高いoverlap（0.0924）
- **深層での差異**: Layer 11で両方とも低いoverlapを示す傾向

### 5. 次のアクション

- **Phase 7（k選択）**: k値ごとのoverlapを詳細に分析し、最適なk値を決定
- **非線形アライメント**: 線形写像では不十分なため、非線形アライメント手法の検討
- **感情別アライメント**: neutralベースではなく、感情別のアライメントを試す
- **中規模モデルへの展開**: 小型モデル間での共通構造が限定的なため、中規模モデルとの比較も検討

### 6. 考察

- **モデル間の構造の違い**: 3つの小型モデル間で感情サブスペースの共通構造は限定的（overlap < 0.15）
- **アーキテクチャの影響**: GPT-2, Pythia, GPT-Neoは異なるアーキテクチャを持つため、内部表現の構造も異なる可能性
- **研究への示唆**: masterplan.mdのQ2（Cross-model shared structure）については、小型モデル間では共通構造が弱い可能性がある

