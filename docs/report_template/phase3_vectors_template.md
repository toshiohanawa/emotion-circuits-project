# Phase 3 — Emotion Vector Construction

## 🎯 目的

- sentence-end vectors（文章末の residual）
- token-based vectors（emotion token ベース）
- 各感情の平均ベクトルを計算

## 📦 生成物

- `emotion_vectors.pkl` (sentence-end)
- `emotion_vectors_token_based.pkl` (token-based)
- `docs/phase3_report.md`

## 🚀 実行コマンド例

```bash
python -m src.analysis.emotion_vectors --model gpt2
python -m src.analysis.emotion_vectors_token_based --model gpt2
```

## 📄 レポート項目

### 1. sentence-end / token-based 手法の比較

#### Sentence-end手法
- 説明: [文章末のresidual streamを使用]
- メリット: [メリットの説明]
- デメリット: [デメリットの説明]

#### Token-based手法
- 説明: [感情語トークンのresidual streamを使用]
- メリット: [メリットの説明]
- デメリット: [デメリットの説明]

#### 比較結果
| 手法 | 安定性 | 解釈可能性 | 計算コスト |
|------|--------|----------|-----------|
| Sentence-end | [評価] | [評価]   | [評価]    |
| Token-based  | [評価] | [評価]   | [評価]    |

### 2. 感情間の cosine 類似（表）

#### Sentence-end手法

| 感情1 | 感情2 | Cosine類似度 |
|------|------|------------|
| Gratitude | Anger | [値] |
| Gratitude | Apology | [値] |
| Anger | Apology | [値] |

#### Token-based手法

| 感情1 | 感情2 | Cosine類似度 |
|------|------|------------|
| Gratitude | Anger | [値] |
| Gratitude | Apology | [値] |
| Anger | Apology | [値] |

### 3. 層ごとのベクトル強度

#### L1 Norm
| 層 | Gratitude | Anger | Apology |
|----|----------|-------|---------|
| 0  | [値]     | [値]  | [値]    |
| 3  | [値]     | [値]  | [値]    |
| 6  | [値]     | [値]  | [値]    |
| 9  | [値]     | [値]  | [値]    |
| 11 | [値]     | [値]  | [値]    |

#### L2 Norm
| 層 | Gratitude | Anger | Apology |
|----|----------|-------|---------|
| 0  | [値]     | [値]  | [値]    |
| 3  | [値]     | [値]  | [値]    |
| 6  | [値]     | [値]  | [値]    |
| 9  | [値]     | [値]  | [値]    |
| 11 | [値]     | [値]  | [値]    |

### 4. トークン数の影響

#### 感情語トークン数
| 感情カテゴリ | 平均トークン数 | 最小 | 最大 |
|------------|-------------|------|------|
| Gratitude  | [数]        | [数] | [数] |
| Anger      | [数]        | [数] | [数] |
| Apology    | [数]        | [数] | [数] |

#### トークン数とベクトル安定性
- [トークン数がベクトル安定性に与える影響の考察]

### 5. モデル間比較

#### モデル間の感情ベクトル類似度
| モデル1 | モデル2 | Gratitude | Anger | Apology |
|---------|---------|----------|-------|---------|
| gpt2    | pythia-160m | [値] | [値] | [値] |
| gpt2    | gpt-neo-125M | [値] | [値] | [値] |
| pythia-160m | gpt-neo-125M | [値] | [値] | [値] |

### 6. 考察

#### 手法の選択
- [どの手法を採用するか、その理由]

#### 発見
- [重要な発見]

#### 課題
- [発見された課題]

#### 次のフェーズへの準備
- [Phase 4以降で使用するベクトルの選択理由]

## 📝 備考

[その他の注意事項やメモ]

