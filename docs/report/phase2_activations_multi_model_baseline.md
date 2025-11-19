# Phase 2 — 活性抽出（3モデル統合レポート）

## 🎯 目的
baselineデータセット上で3つの小型モデル（GPT-2 small, Pythia-160M, GPT-Neo-125M）のresidual streamを抽出し、後続フェーズで再利用できる形式で保存する。

## 📦 生成物
- `results/baseline/activations/gpt2_small.pkl`
- `results/baseline/activations/pythia-160m.pkl`
- `results/baseline/activations/gpt-neo-125m.pkl`
- 本レポート: `docs/report/phase2_activations_multi_model_baseline.md`

## 🚀 実行コマンド

```bash
# GPT-2 small
python3 -m src.analysis.run_phase2_activations \
  --profile baseline \
  --model gpt2_small \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 \
  --device mps \
  --max-samples-per-emotion 225 \
  --batch-size 32

# Pythia-160M
python3 -m src.analysis.run_phase2_activations \
  --profile baseline \
  --model pythia-160m \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 \
  --device mps \
  --max-samples-per-emotion 225 \
  --batch-size 32

# GPT-Neo-125M
python3 -m src.analysis.run_phase2_activations \
  --profile baseline \
  --model gpt-neo-125m \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 \
  --device mps \
  --max-samples-per-emotion 225 \
  --batch-size 32
```

## 📄 レポート

### 1. 実行設定

| モデル | パラメータ数 | 層数 | サンプル数 | 実行時間 | デバイス |
|--------|------------|------|-----------|---------|---------|
| GPT-2 small | 124M | 12 | 900 (225×4感情) | 7.15秒 | MPS |
| Pythia-160M | 160M | 12 | 900 (225×4感情) | 7.07秒 | MPS |
| GPT-Neo-125M | 125M | 12 | 900 (225×4感情) | 8.48秒 | MPS |

**共通設定**:
- **プロファイル**: `baseline`（感情4種 × 各225サンプル）
- **層**: 0-11（全12層）
- **活性位置**: `resid_pre` と `resid_post` の両方を抽出
- **バッチサイズ**: 32

### 2. 抽出したテンソルの形状

すべてのモデルで同じ形状のテンソルが抽出されました：

| モデル | resid_pre形状 | resid_post形状 | ファイルサイズ |
|--------|--------------|---------------|---------------|
| GPT-2 small | [900, 12, 13, 768] | [900, 12, 13, 768] | 823MB |
| Pythia-160M | [900, 12, 13, 768] | [900, 12, 13, 768] | 823MB |
| GPT-Neo-125M | [900, 12, 13, 768] | [900, 12, 13, 768] | 823MB |

**データ構造**:
- **batch**: 900サンプル（感情4種 × 各225サンプル）
- **layers**: 12層（0-11）
- **sequence**: 最大13トークン（パディング済み）
- **d_model**: 768次元（すべてのモデルで共通）

### 2.1 感情別サンプル数

すべてのモデルで同じ分布：

| 感情 | サンプル数 |
|------|-----------|
| neutral | 225 |
| anger | 225 |
| apology | 225 |
| gratitude | 225 |
| **合計** | **900** |

### 3. 実行コスト

| モデル | 実行時間 | メモリ使用（ファイルサイズ） |
|--------|---------|------------------------|
| GPT-2 small | 7.15秒 | 823MB |
| Pythia-160M | 7.07秒 | 823MB |
| GPT-Neo-125M | 8.48秒 | 823MB |
| **合計** | **22.70秒** | **2.47GB** |

**観察**:
- すべてのモデルでほぼ同じ実行時間（7-8秒）
- ファイルサイズも同一（823MB）
- MPS環境でのバッチ処理により高速化を実現

### 4. 次のアクション

- **Phase 3**: 各モデルの活性ファイルを使用して感情ベクトルとサブスペースを計算
  - `results/baseline/activations/gpt2_small.pkl`
  - `results/baseline/activations/pythia-160m.pkl`
  - `results/baseline/activations/gpt-neo-125m.pkl`
- **Phase 4以降**: 同じ活性ファイルを参照可能

### 5. モデル間の比較

すべてのモデルで同じデータセット（900サンプル、感情4種×225件）を使用し、同じ形状のテンソルが抽出されました。これにより、Phase 4でのモデル間アライメントが可能になります。

