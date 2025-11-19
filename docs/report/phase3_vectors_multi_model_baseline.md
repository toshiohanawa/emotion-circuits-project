# Phase 3 — 感情ベクトル・サブスペース構築（3モデル統合レポート）

## 🎯 目的
Phase2で抽出した活性から、3つの小型モデル（GPT-2 small, Pythia-160M, GPT-Neo-125M）について、感情ごとの差分ベクトルを計算し、層別に平均ベクトルとPCAサブスペースを得る。

## 📦 生成物
- `results/baseline/emotion_vectors/gpt2_small_vectors_token_based.pkl`
- `results/baseline/emotion_vectors/pythia-160m_vectors_token_based.pkl`
- `results/baseline/emotion_vectors/gpt-neo-125m_vectors_token_based.pkl`
- `results/baseline/emotion_subspaces/gpt2_small_subspaces.pkl`
- `results/baseline/emotion_subspaces/pythia-160m_subspaces.pkl`
- `results/baseline/emotion_subspaces/gpt-neo-125m_subspaces.pkl`
- 本レポート: `docs/report/phase3_vectors_multi_model_baseline.md`

## 🚀 実行コマンド

```bash
# GPT-2 small
python3 -m src.analysis.run_phase3_vectors \
  --profile baseline \
  --model gpt2_small \
  --n-components 8 \
  --use-torch \
  --device mps

# Pythia-160M
python3 -m src.analysis.run_phase3_vectors \
  --profile baseline \
  --model pythia-160m \
  --n-components 8 \
  --use-torch \
  --device mps

# GPT-Neo-125M
python3 -m src.analysis.run_phase3_vectors \
  --profile baseline \
  --model gpt-neo-125m \
  --n-components 8 \
  --use-torch \
  --device mps
```

## 📄 レポート

### 1. 実行設定

| モデル | n_components | 計算バックエンド | 実行時間 |
|--------|-------------|----------------|---------|
| GPT-2 small | 8 | torch (MPS) | 0.50秒 |
| Pythia-160M | 8 | torch (MPS) | 0.41秒 |
| GPT-Neo-125M | 8 | torch (MPS) | 0.41秒 |

**共通設定**:
- **プロファイル**: `baseline`
- **n_components**: 8（PCA次元数）
- **計算バックエンド**: torch（MPS加速）
- **入力活性ファイル**: `results/baseline/activations/<model>.pkl`

### 2. 感情ベクトルの概要

#### 2.1 感情ベクトルのノルム（全層平均）

| モデル | gratitude | anger | apology |
|--------|----------|-------|---------|
| GPT-2 small | 52.03 | 56.40 | 84.56 |
| Pythia-160M | 10.97 | 14.82 | 20.01 |
| GPT-Neo-125M | 262.68 | 320.61 | 621.51 |

**観察**:
- **GPT-2 small**: 中程度のノルム（52-85）。apologyが最も大きい（84.56）。
- **Pythia-160M**: 最も小さいノルム（11-20）。感情間の差が比較的小さい。
- **GPT-Neo-125M**: 最も大きいノルム（263-622）。apologyが特に大きい（621.51）。
- すべてのモデルで **apology > anger > gratitude** の順にノルムが大きい。

#### 2.2 モデル間の比較

- **ノルムのスケール**: モデル間でノルムのスケールが大きく異なる（GPT-Neoが最大、Pythiaが最小）
- **感情間の相対関係**: すべてのモデルで同じ順序（apology > anger > gratitude）を示す
- **スケールの違い**: これはモデルの内部表現のスケールの違いであり、Phase 4のアライメントで正規化される可能性がある

### 3. サブスペース（PCA）の概要

#### 3.1 主成分の説明分散比（PC1、Layer 0, 6, 11）

| モデル | Layer 0 | Layer 6 | Layer 11 |
|--------|---------|---------|----------|
| GPT-2 small | 66.95% | 26.75% | 63.34% |
| Pythia-160M | 29.06% | 97.80% | 94.35% |
| GPT-Neo-125M | 97.88% | 97.96% | 35.48% |

**観察**:
- **GPT-2 small**: Layer 0と11で高い説明分散（66.95%, 63.34%）、Layer 6で低い（26.75%）。層による変動が大きい。
- **Pythia-160M**: Layer 6と11で非常に高い説明分散（97.80%, 94.35%）、Layer 0で低い（29.06%）。深層で一方向的な構造。
- **GPT-Neo-125M**: Layer 0と6で非常に高い説明分散（97.88%, 97.96%）、Layer 11で低い（35.48%）。浅層で一方向的な構造。

#### 3.2 モデル間の構造の違い

- **GPT-2 small**: 層による説明分散の変動が大きく、より複雑な構造を示す
- **Pythia-160M**: 深層（Layer 6-11）で高い説明分散、一方向的な構造が強い
- **GPT-Neo-125M**: 浅層（Layer 0-6）で高い説明分散、深層で分散が増加

### 4. 次のアクション

- **Phase 4**: 各モデルのサブスペースファイルを使用してモデル間アライメントを計算
  - `results/baseline/emotion_subspaces/gpt2_small_subspaces.pkl`
  - `results/baseline/emotion_subspaces/pythia-160m_subspaces.pkl`
  - `results/baseline/emotion_subspaces/gpt-neo-125m_subspaces.pkl`
- **Phase 5**: 各モデルの感情ベクトルファイルを使用して残差パッチング
  - `results/baseline/emotion_vectors/gpt2_small_vectors_token_based.pkl`
  - `results/baseline/emotion_vectors/pythia-160m_vectors_token_based.pkl`
  - `results/baseline/emotion_vectors/gpt-neo-125m_vectors_token_based.pkl`

### 5. 考察

- **モデル間の構造の違い**: 3モデルでPCAの説明分散比のパターンが大きく異なり、モデルごとに異なる感情表現構造を持つことが示唆される
- **感情間の一貫性**: すべてのモデルで同じ感情順序（apology > anger > gratitude）を示し、感情間の相対的な関係は共通している可能性がある
- **Phase 4への示唆**: モデル間の構造の違いは大きいが、線形アライメントで共通構造を発見できる可能性がある

