# Phase 6 — Head Screening & Ablation

## 🎯 目的

- 各 head の attention パターン（Δattention）
- ablation で Δemotion を計測
- head importance ranking を作成

## 📦 生成物

- `head_scores.json`
- `head_ablation.pkl`
- `docs/phase6_report.md`

## 🚀 実行コマンド例

```bash
python -m src.analysis.head_screening \
  --model gpt2 \
  --profile baseline \
  --output results/baseline/alignment/head_scores_gpt2.json

python -m src.models.head_ablation \
  --model gpt2 \
  --head-spec "3:5,7:2" \
  --prompts-file data/gratitude_prompts.json \
  --output results/baseline/patching/head_ablation_gpt2_gratitude.pkl
```

## 📄 レポート項目

### 1. Head Screening結果

#### 感情語トークンへの反応度（Δattention）

| 層 | Head | Gratitude | Anger | Apology | 平均 |
|----|------|----------|-------|---------|------|
| 0  | 0    | [値]     | [値]  | [値]    | [値] |
| 0  | 1    | [値]     | [値]  | [値]    | [値] |
| ... | ... | ... | ... | ... | ... |

#### Top-N Heads（感情別）

##### Gratitude
| ランク | 層:Head | Δattention | サンプル数 |
|--------|---------|-----------|-----------|
| 1      | [層:Head] | [値]     | [数]      |
| 2      | [層:Head] | [値]     | [数]      |
| 3      | [層:Head] | [値]     | [数]      |

##### Anger
| ランク | 層:Head | Δattention | サンプル数 |
|--------|---------|-----------|-----------|
| 1      | [層:Head] | [値]     | [数]      |
| 2      | [層:Head] | [値]     | [数]      |
| 3      | [層:Head] | [値]     | [数]      |

##### Apology
| ランク | 層:Head | Δattention | サンプル数 |
|--------|---------|-----------|-----------|
| 1      | [層:Head] | [値]     | [数]      |
| 2      | [層:Head] | [値]     | [数]      |
| 3      | [層:Head] | [値]     | [数]      |

### 2. Head Ablation結果

#### Ablation設定
- 対象Heads: [層:Headのリスト]
- プロンプト数: [数]
- max_new_tokens: [数]

#### Baseline vs Ablation比較

| メトリクス | Baseline | Ablation | Δ |
|-----------|---------|----------|---|
| Sentiment (POSITIVE) | [値] | [値] | [値] |
| Sentiment (NEGATIVE) | [値] | [値] | [値] |
| Politeness | [値] | [値] | [値] |
| Gratitude keywords | [値] | [値] | [値] |
| Anger keywords | [値] | [値] | [値] |
| Apology keywords | [値] | [値] | [値] |

### 3. Head重要性ランキング

#### 総合ランキング（全感情平均）

| ランク | 層:Head | 総合スコア | Gratitude | Anger | Apology |
|--------|---------|-----------|----------|-------|---------|
| 1      | [層:Head] | [値]     | [値]     | [値]  | [値]    |
| 2      | [層:Head] | [値]     | [値]     | [値]  | [値]    |
| 3      | [層:Head] | [値]     | [値]     | [値]  | [値]    |

### 4. 層ごとの特徴

#### 層別統計
- Layer 0: [特徴]
- Layer 3: [特徴]
- Layer 6: [特徴]
- Layer 9: [特徴]
- Layer 11: [特徴]

### 5. 可視化結果

#### Head反応度ヒートマップ
- [ファイルパス]

#### Ablation効果の可視化
- [ファイルパス]

### 6. 考察

#### 重要なHeadの特定
- [どのHeadが感情回路に重要か]

#### 層依存性
- [層による違い]

#### Ablation効果の解釈
- [Ablationによる変化の意味]

#### 次のフェーズへの準備
- [Phase 7でのHead Patching実験への示唆]

## 📝 備考

[その他の注意事項やメモ]

