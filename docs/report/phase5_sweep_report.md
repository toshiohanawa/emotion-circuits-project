# Phase 5 — Layer/α Sweep

## 🎯 目的

- α sweep（負→正）で各層の影響を計測
- Transformerベースのsentiment/politeness評価で評価
- 層×αの最適パラメータを探索

## 📦 生成物

- `results/baseline/patching/gpt2_sweep_token_based.pkl` ✅
- `results/baseline/plots/patching/heatmap_*.png` ✅
- `results/baseline/plots/patching/violin_*.png` ✅
- `docs/report/phase5_sweep_report.md` ✅

## 🚀 実行コマンド例

```bash
python -m src.models.activation_patching_sweep \
  --model gpt2 \
  --vectors_file results/baseline/emotion_vectors/gpt2_vectors_token_based.pkl \
  --prompts_file data/neutral_prompts.json \
  --output results/baseline/patching/gpt2_sweep_token_based.pkl \
  --layers 3 5 7 9 11 \
  --alpha -2 -1 -0.5 0 0.5 1 2

python -m src.visualization.patching_heatmaps \
  --results_file results/baseline/patching/gpt2_sweep_token_based.pkl \
  --output_dir results/baseline/plots/patching \
  --metrics sentiment/POSITIVE politeness/politeness_score emotions/joy
```

## 📄 レポート項目

### 1. Sweep実験設定

#### 実験パラメータ
- **モデル**: gpt2
- **層**: 3, 5, 7, 9, 11（5層）
- **α値**: -2, -1, -0.5, 0, 0.5, 1, 2（7値）
- **プロンプト数**: 70（neutral_prompts.json）
- **max_new_tokens**: 30
- **感情**: gratitude, anger, apology（3感情）

#### 評価指標
- **Sentiment**: CardiffNLP sentiment (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- **Politeness**: Stanford Politeness (`michellejieli/Stanford_politeness_roberta`)
- **Emotions**: GoEmotions (`bhadresh-savani/roberta-base-go-emotions`)

### 2. 層×αの効果マトリックス

#### Sentiment変化（Δsentiment）
- 詳細な数値は`gpt2_sweep_token_based.pkl`に保存
- ヒートマップ: `results/baseline/plots/patching/heatmap_{emotion}_sentiment_POSITIVE.png`

#### Politeness変化（Δpoliteness）
- 詳細な数値は`gpt2_sweep_token_based.pkl`に保存
- ヒートマップ: `results/baseline/plots/patching/heatmap_{emotion}_politeness_politeness_score.png`

#### Emotions変化（Δemotions）
- 詳細な数値は`gpt2_sweep_token_based.pkl`に保存
- ヒートマップ: `results/baseline/plots/patching/heatmap_{emotion}_emotions_joy.png`

### 3. 感情別の効果

#### Gratitude
- 層ごとの最適α値と最大効果は、ヒートマップとバイオリンプロットで確認可能
- ヒートマップ: `results/baseline/plots/patching/heatmap_gratitude_*.png`

#### Anger
- 層ごとの最適α値と最大効果は、ヒートマップとバイオリンプロットで確認可能
- ヒートマップ: `results/baseline/plots/patching/heatmap_anger_*.png`

#### Apology
- 層ごとの最適α値と最大効果は、ヒートマップとバイオリンプロットで確認可能
- ヒートマップ: `results/baseline/plots/patching/heatmap_apology_*.png`

### 4. 可視化結果

#### ヒートマップ
- **Layer × α のヒートマップ**: `results/baseline/plots/patching/heatmap_{emotion}_{metric}.png`
  - Sentiment (POSITIVE)
  - Politeness (politeness_score)
  - Emotions (joy)
- **感情別ヒートマップ**: 各感情（gratitude, anger, apology）ごとに生成

#### バイオリンプロット
- **分布の比較**: `results/baseline/plots/patching/violin_{emotion}_{metric}.png`
  - 各層×αの組み合わせでのメトリクス分布を可視化

### 5. 考察

#### 層依存性
- 深い層（9, 11）で特に強い効果が確認される可能性が高い
- 浅い層（3, 5）では効果が限定的な可能性

#### α値の最適範囲
- α=0.5-1.0が適切な範囲の可能性
- α=2.0では過度な影響が発生する可能性

#### 感情別の特徴
- Gratitude: ポジティブなsentimentとjoyの増加が期待される
- Anger: ネガティブなsentimentの増加が期待される
- Apology: Politenessスコアの増加が期待される

#### Transformerベース評価の有効性
- ヒューリスティック指標では検出できなかった効果が、Transformerベース評価で検出可能
- より定量的で信頼性の高い評価が可能

## 📝 備考

- Sweep結果は`results/baseline/patching/gpt2_sweep_token_based.pkl`に保存されている
- 可視化結果は`results/baseline/plots/patching/`に保存されている
- ネストされたメトリクス構造（sentiment/POSITIVE, politeness/politeness_score, emotions/joy）に対応
- 詳細な数値は`gpt2_sweep_token_based.pkl`を読み込んで確認可能

