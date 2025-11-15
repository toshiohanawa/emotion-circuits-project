# Phase 8 — Epic: QK/OV Circuit Analysis

## 🎯 目的

- QK routing（attention flow）
- OV projection（value → residual 書き込み）
- head-level / neuron-level 因果実験
- circuit summary の生成（md/json）

## 📦 生成物

- `ov_qk_results.pkl`
- `ov_head_projections.csv/png`
- `qk_routing.png`
- `circuit_summary.md/json`
- `docs/phase8_report.md`

## 🚀 実行コマンド例

```bash
python -m src.analysis.circuit_experiments \
  --model gpt2 \
  --prompts data/neutral_prompts.json \
  --emotion-vectors results/baseline/emotion_vectors/gpt2_vectors_token_based.pkl \
  --layer 6 \
  --heads "6:0,6:1" \
  --neurons "6:10,12" \
  --max-new-tokens 30 \
  --output results/baseline/circuits/ov_qk_results

python -m src.analysis.circuit_report \
  --results results/baseline/circuits/ov_qk_results/ov_qk_results.pkl \
  --output results/baseline/circuits/report
```

## 📄 レポート項目

### 1. OV Projection 結果（cosine / per-head）

#### OV投影のcosine類似度

| Head | Gratitude | Anger | Apology | 平均 |
|------|----------|-------|---------|------|
| 0    | [値]     | [値]  | [値]    | [値] |
| 1    | [値]     | [値]  | [値]    | [値] |
| 2    | [値]     | [値]  | [値]    | [値] |
| ... | ... | ... | ... | ... |

#### 統計
- cos_mean: [値]
- cos_max: [値]
- cos_min: [値]

#### Dot Product（生の内積）

| Head | Gratitude | Anger | Apology |
|------|----------|-------|---------|
| 0    | [値]     | [値]  | [値]    |
| 1    | [値]     | [値]  | [値]    |
| ... | ... | ... | ... |

### 2. QK Routing（attention flow）

#### QK Routingパターン
- [QK routingヒートマップの説明]
- [主要なattentionパターンの特徴]

#### 感情語トークンへのattention
- [感情語トークン位置へのattention weight]

#### Routingパターンの比較
- Gratitude: [パターンの特徴]
- Anger: [パターンの特徴]
- Apology: [パターンの特徴]

### 3. OV Ablation（Δemotion / Δsentiment / Δpoliteness）

#### OV Ablation設定
- 対象Heads: [層:Headのリスト]
- プロンプト数: [数]
- max_new_tokens: [数]

#### Baseline vs OV Ablation

| メトリクス | Baseline | OV Ablation | Δ |
|-----------|---------|-------------|---|
| Sentiment (POSITIVE) | [値] | [値] | [値] |
| Sentiment (NEGATIVE) | [値] | [値] | [値] |
| Politeness | [値] | [値] | [値] |
| Gratitude keywords | [値] | [値] | [値] |
| Anger keywords | [値] | [値] | [値] |
| Apology keywords | [値] | [値] | [値] |

#### GoEmotions変化

| Emotion | Baseline | OV Ablation | Δ |
|---------|---------|-------------|---|
| joy | [値] | [値] | [値] |
| anger | [値] | [値] | [値] |
| ... | ... | ... | ... |

### 4. QK Patching（Δmetrics）

#### QK Patching設定
- Routing template: [感情カテゴリ]
- 対象Heads: [層:Headのリスト]

#### Baseline vs QK Patching

| メトリクス | Baseline | QK Patching | Δ |
|-----------|---------|-------------|---|
| Sentiment (POSITIVE) | [値] | [値] | [値] |
| Sentiment (NEGATIVE) | [値] | [値] | [値] |
| Politeness | [値] | [値] | [値] |
| Gratitude keywords | [値] | [値] | [値] |
| Anger keywords | [値] | [値] | [値] |
| Apology keywords | [値] | [値] | [値] |

#### Routing Mean
- Baseline: [値]
- QK Patching: [値]

### 5. Neuron×Head Combined

#### Combined Ablation設定
- 対象Neurons: [層:ニューロンインデックスのリスト]
- 対象Heads: [層:Headのリスト]

#### Baseline vs Combined Ablation

| メトリクス | Baseline | Combined Ablation | Δ |
|-----------|---------|-------------------|---|
| Sentiment (POSITIVE) | [値] | [値] | [値] |
| Sentiment (NEGATIVE) | [値] | [値] | [値] |
| Politeness | [値] | [値] | [値] |

#### 個別効果 vs 統合効果
- Neuron only: [Δ値]
- Head only: [Δ値]
- Combined: [Δ値]
- 相乗効果: [Yes/No]

### 6. Circuit Summary

#### 主要な発見
- [主要な発見1]
- [主要な発見2]
- [主要な発見3]

#### Head重要性ランキング（OV投影ベース）

| ランク | 層:Head | OV投影スコア | Ablation効果 |
|--------|---------|------------|-------------|
| 1      | [層:Head] | [値]      | [Δ値]       |
| 2      | [層:Head] | [値]      | [Δ値]       |
| 3      | [層:Head] | [値]      | [Δ値]       |

#### QK Routingの特徴
- [QK routingの主要な特徴]

### 7. まとめ（Emotion Circuit は存在するか？）

#### 証拠の統合
- OV投影: [証拠の説明]
- QK routing: [証拠の説明]
- Ablation効果: [証拠の説明]
- Patching効果: [証拠の説明]

#### 結論
- [Emotion Circuitの存在に関する結論]
- [主要なHead/Neuronの特定]
- [回路の構造の理解]

#### 限界と今後の課題
- [発見された限界]
- [今後の研究課題]

### 8. 可視化結果

#### OV投影ヒートマップ
- [ファイルパス]
- [図の説明]

#### QK Routingヒートマップ
- [ファイルパス]
- [図の説明]

#### Head重要性ヒートマップ
- [ファイルパス]
- [図の説明]

## 📝 備考

[その他の注意事項やメモ]

