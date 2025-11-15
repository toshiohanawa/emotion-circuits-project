# Phase 6 — Head Screening & Ablation

## 🎯 目的

- 各 head の attention パターン（Δattention）を測定
- ablation で Δemotion を計測
- head importance ranking を作成

## 📦 生成物

- `results/baseline/alignment/head_scores_gpt2.json` ✅
- `results/baseline/patching/head_ablation/gpt2_gratitude_1_10.pkl` ✅
- `results/baseline/plots/heads/*.png` ✅
- `docs/report/phase6_head_screening_report.md` ✅

## 🚀 実行コマンド例

```bash
python -m src.analysis.head_screening \
  --model gpt2 \
  --profile baseline \
  --output results/baseline/alignment/head_scores_gpt2.json

python -m src.models.head_ablation \
  --model gpt2 \
  --profile baseline \
  --head-spec "1:10" \
  --emotion gratitude \
  --max-tokens 15 \
  --output results/baseline/patching/head_ablation/gpt2_gratitude_1_10.pkl

python -m src.visualization.head_plots \
  --profile baseline \
  --head-scores results/baseline/alignment/head_scores_gpt2.json \
  --ablation-file results/baseline/patching/head_ablation/gpt2_gratitude_1_10.pkl \
  --output-dir results/baseline/plots/heads \
  --top-n 20
```

## 📄 レポート項目

### 1. Head Screening結果

#### 感情語トークンへの反応度（Δattention）

**Gratitude Top-10 Heads**:

| ランク | 層:Head | Δattention | Emotion Mean | Neutral Mean | サンプル数 |
|--------|---------|-----------|--------------|--------------|-----------|
| 1      | 1:10    | 0.340434  | 0.572721     | 0.232287     | 70/70     |
| 2      | 3:2     | 0.242236  | 0.322768     | 0.080532     | 70/70     |
| 3      | 1:11    | 0.228443  | 0.464736     | 0.236293     | 70/70     |
| 4      | 0:3     | 0.212430  | 0.696900     | 0.484470     | 70/70     |
| 5      | 11:8    | 0.176943  | 0.636841     | 0.459898     | 70/70     |
| 6      | 0:5     | 0.171414  | 0.612541     | 0.441128     | 70/70     |
| 7      | 3:6     | 0.103138  | 0.196405     | 0.093267     | 70/70     |
| 8      | 1:3     | 0.093397  | 0.222555     | 0.129158     | 70/70     |
| 9      | 0:4     | 0.087875  | 0.405507     | 0.317632     | 70/70     |
| 10     | 0:6     | 0.064671  | 0.078693     | 0.014022     | 70/70     |

**重要な発見**: Layer 1 Head 10がgratitude感情に最も強く反応（Δattention: 0.340434）

#### Top-N Heads（感情別）

詳細なランキングは`head_scores_gpt2.json`に保存されており、各感情（gratitude, anger, apology）ごとにTop-10が確認可能。

### 2. Head Ablation結果

#### Ablation設定
- **対象Heads**: Layer 1 Head 10（gratitudeに最も強く反応するhead）
- **プロンプト数**: 70（gratitude_prompts.json）
- **max_new_tokens**: 15

#### Baseline vs Ablation比較

| メトリクス | Baseline | Ablation | Δ |
|-----------|---------|----------|---|
| Sentiment (mean) | 0.0000 | 0.0000 | 0.0000 |
| Gratitude keywords (mean) | 1.1143 | 1.0857 | -0.0286 |
| Gratitude keywords (total) | 78 | 76 | -2 |
| Anger keywords (mean) | 0.0429 | 0.0429 | 0.0000 |
| Apology keywords (mean) | 0.0143 | 0.0143 | 0.0000 |

**考察**: Layer 1 Head 10のablationにより、gratitudeキーワードがわずかに減少（-2）。ただし、sentimentスコアには大きな変化が見られない。

### 3. Head重要性ランキング

#### 総合ランキング（全感情平均）
- 詳細なランキングは`head_scores_gpt2.json`に保存
- Layer 1 Head 10がgratitudeで最も高いスコア（0.340434）

### 4. 層ごとの特徴

#### 層別統計
- **Layer 0**: 複数のheadが感情語トークンに反応（0:3, 0:4, 0:5, 0:6）
- **Layer 1**: Head 10とHead 11が特に強い反応を示す
- **Layer 3**: Head 2とHead 6が感情語トークンに反応
- **Layer 11**: Head 8が感情語トークンに反応

**考察**: 浅い層（0-1）と深い層（11）で感情語トークンへの反応が強い。

### 5. 可視化結果

#### Head反応度ヒートマップ
- `results/baseline/plots/heads/head_reaction_heatmap.png`

#### Ablation効果の可視化
- `results/baseline/plots/heads/ablation_comparison.png`

### 6. 考察

#### 重要なHeadの特定
- **Layer 1 Head 10**: gratitude感情に最も強く反応（Δattention: 0.340434）
- **Layer 3 Head 2**: gratitude感情に2番目に強く反応（Δattention: 0.242236）
- **Layer 1 Head 11**: gratitude感情に3番目に強く反応（Δattention: 0.228443）

#### 層依存性
- 浅い層（0-1）で感情語トークンへの反応が強い
- 深い層（11）でも一部のheadが感情語トークンに反応

#### Ablation効果の解釈
- Layer 1 Head 10のablationにより、gratitudeキーワードがわずかに減少
- ただし、sentimentスコアには大きな変化が見られない
- より多くのheadを同時にablationすることで、より大きな効果が期待される

#### 次のフェーズへの準備
- Phase 7では、重要なhead（Layer 1 Head 10など）をpatchingして、感情方向への影響を検証
- 複数のheadを同時にpatchingすることで、より強い効果が期待される

## 📝 備考

- Head scoresは`results/baseline/alignment/head_scores_gpt2.json`に保存されている
- Head ablation結果は`results/baseline/patching/head_ablation/`に保存されている
- 可視化結果は`results/baseline/plots/heads/`に保存されている
- Layer 1 Head 10がgratitude感情に最も強く反応することが確認された

