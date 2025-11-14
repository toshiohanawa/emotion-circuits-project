# フェーズ5レポート: 因果性をちゃんと殴る（Activation Patching完全版）

## 実行日時
2024年11月14日

## 概要
フェーズ5では、「この方向 / サブスペースは本当に感情を担っている」と言い切るための実験を実施しました。層×αのスイープ実験と、文末ベースと感情語トークンベースの因果力比較を行いました。

## 実験設定

### モデル
- **GPT-2 small (124M)**: 古典的構造の基準モデル

### プロンプト
- **中立プロンプト**: 50文（`data/neutral_prompts.json`）
- 例: "The weather today is", "I need to check", "Can you help me with" など

### 実験パラメータ
- **層**: {3, 5, 7, 9, 11}
- **α値**: {-2, -1, -0.5, 0, 0.5, 1, 2}
- **感情方向ベクトル**:
  - 文末ベース: `results/baseline/emotion_vectors/gpt2_vectors.pkl`
  - 感情語トークンベース: `results/baseline/emotion_vectors/gpt2_vectors_token_based.pkl`

### 評価指標
1. **感情キーワード頻度**: 感謝語・怒り語・謝罪語の出現回数
2. **丁寧さスコア**: "please", "kindly", "thank you", "sorry"などの出現頻度（0-1）
3. **Sentimentスコア**: ポジティブ/ネガティブ語の出現頻度（-1から1）

## 実装内容

### 5-1: 層×αのスイープ実験

**実装ファイル**:
- `src/models/activation_patching_sweep.py`: スイープ実験の実行
- `src/visualization/patching_heatmaps.py`: ヒートマップ生成

**実行コマンド**:
```bash
# 感情語トークンベース
python src/models/activation_patching_sweep.py \
  --model gpt2 \
  --vectors_file results/baseline/emotion_vectors/gpt2_vectors_token_based.pkl \
  --prompts_file data/neutral_prompts.json \
  --output results/baseline/patching/gpt2_sweep_token_based.pkl \
  --layers 3 5 7 9 11 \
  --alpha -2 -1 -0.5 0 0.5 1 2

# 文末ベース
python src/models/activation_patching_sweep.py \
  --model gpt2 \
  --vectors_file results/baseline/emotion_vectors/gpt2_vectors.pkl \
  --prompts_file data/neutral_prompts.json \
  --output results/baseline/patching/gpt2_sweep_sentence_end.pkl \
  --layers 3 5 7 9 11 \
  --alpha -2 -1 -0.5 0 0.5 1 2
```

**生成されたヒートマップ**:
- `results/baseline/plots/heatmaps_token_based/`: 感情語トークンベースのヒートマップ（15ファイル）
- `results/baseline/plots/heatmaps_sentence_end/`: 文末ベースのヒートマップ（15ファイル）
- 各感情（gratitude, anger, apology）×各メトリクス（gratitude, anger, apology, politeness, sentiment）

### 5-2: 文末 vs 感情語ベクトルの「因果力」比較

**実装ファイル**:
- `src/analysis/compare_patching_methods.py`: 比較実験の実行

**実行コマンド**:
```bash
python src/analysis/compare_patching_methods.py \
  --model gpt2 \
  --sentence_end_vectors results/baseline/emotion_vectors/gpt2_vectors.pkl \
  --token_based_vectors results/baseline/emotion_vectors/gpt2_vectors_token_based.pkl \
  --prompts_file data/neutral_prompts.json \
  --output_dir results/baseline/patching/comparison \
  --layers 3 5 7 9 11 \
  --alpha -2 -1 -0.5 0 0.5 1 2
```

**生成された比較プロット**:
- `results/baseline/patching/comparison/comparison_*.png`: 各感情×各メトリクスの比較プロット（15ファイル）

## 結果と考察

### スイープ実験の結果

実験は正常に完了し、以下のデータが生成されました：

1. **スイープ結果ファイル**:
   - `results/baseline/patching/gpt2_sweep_token_based.pkl`: 感情語トークンベースの全結果
   - `results/baseline/patching/gpt2_sweep_sentence_end.pkl`: 文末ベースの全結果

2. **集計メトリクス**:
   - 各層×各α値×各感情で、50プロンプトの平均メトリクスを計算
   - 感情キーワード頻度、丁寧さスコア、sentimentスコアを記録

### 比較実験の結果

比較実験も正常に完了し、以下のデータが生成されました：

1. **比較結果ファイル**:
   - `results/baseline/patching/comparison/gpt2_comparison.pkl`: 両方法の比較結果

2. **効果の強さの比較**:
   - 各感情×各メトリクスで、文末ベースと感情語トークンベースの最大効果を比較
   - 現時点では、両方法とも効果が限定的であることが観察されました

### 重要な発見

1. **実験フレームワークの確立**:
   - 層×αのスイープ実験のフレームワークが確立されました
   - 評価指標（感情キーワード、丁寧さ、sentiment）の自動計算が可能になりました
   - ヒートマップによる可視化により、層とαの組み合わせの効果を一目で確認できるようになりました

2. **Activation Patchingの課題**:
   - 簡易的なActivation Patchingでは、生成テキストに明確な感情的変化を観察することは困難でした
   - これは、以下の要因が考えられます：
     - 単一トークンの予測では感情のニュアンスを捉えるには不十分
     - パッチングの層や強度の調整が必要
     - 評価方法の改善が必要（より長いテキスト生成、感情分類器の導入など）

3. **方法論の比較**:
   - 文末ベースと感情語トークンベースの両方で実験を実施
   - 現時点では、どちらの方法も明確な優位性を示すには至っていませんが、比較のフレームワークは確立されました

## 生成されたファイル

### データファイル
- `results/baseline/patching/gpt2_sweep_token_based.pkl`: 感情語トークンベースのスイープ結果
- `results/baseline/patching/gpt2_sweep_sentence_end.pkl`: 文末ベースのスイープ結果
- `results/baseline/patching/comparison/gpt2_comparison.pkl`: 比較実験の結果

### 可視化ファイル
- `results/baseline/plots/heatmaps_token_based/*.png`: 感情語トークンベースのヒートマップ（15ファイル）
- `results/baseline/plots/heatmaps_sentence_end/*.png`: 文末ベースのヒートマップ（15ファイル）
- `results/baseline/patching/comparison/comparison_*.png`: 比較プロット（15ファイル）

## 今後の改善案

1. **Activation Patchingの改善**:
   - より長いテキスト生成（複数トークン）での評価
   - 感情分類器の導入による自動評価
   - パッチングの層や強度の最適化

2. **評価指標の改善**:
   - より洗練された感情キーワードリスト
   - 文脈を考慮した感情分析
   - 人間評価との比較

3. **実験設定の改善**:
   - より多様なプロンプトでの検証
   - 異なるモデルでの比較
   - より細かいα値のスイープ

## 結論

フェーズ5では、Activation Patchingの完全版実験を実施し、層×αのスイープ実験と、文末ベースと感情語トークンベースの比較実験を完了しました。実験フレームワークは確立されましたが、Activation Patchingの効果を明確に観察するには、評価方法や実験設定のさらなる改善が必要であることが明らかになりました。

この結果は、感情回路の因果的検証が複雑な課題であることを示しており、今後の研究において、より洗練された方法論の開発が重要であることを示唆しています。

