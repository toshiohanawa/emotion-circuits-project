# フェーズ3: 感情方向ベクトルの抽出・可視化 レポート

## 実行日時
2024年11月14日

## 概要
フェーズ2で抽出した内部活性データを使用して、感情方向ベクトルを抽出し、モデル内・モデル間での分析と可視化を行いました。

## 対象モデル

1. **GPT-2 small (124M)**
2. **Pythia-160M**
3. **GPT-Neo-125M**

## 感情方向ベクトルの定義

各層ごとに以下の式で感情方向ベクトルを計算しました：

```
emotion_vec[layer] = mean(resid_emotion[layer]) - mean(resid_neutral[layer])
```

- `resid_emotion[layer]`: 感情カテゴリ（gratitude, anger, apology）の各サンプルのresidual stream
- `resid_neutral[layer]`: 中立カテゴリの各サンプルのresidual stream
- 各サンプルは文末位置（最後のトークン）の活性を使用

## 結果サマリー

### 1. モデル内での感情間の類似度

#### GPT-2 small
- **Gratitude vs Anger**: 平均類似度 0.5479
- **Gratitude vs Apology**: 平均類似度 0.7136
- **Anger vs Apology**: 平均類似度 0.7726

**解釈**: 
- GratitudeとAngerは比較的異なる方向（類似度0.55）
- GratitudeとApologyはより類似（類似度0.71）
- AngerとApologyは最も類似（類似度0.77）

#### Pythia-160M
- **Gratitude vs Anger**: 平均類似度 0.9861
- **Gratitude vs Apology**: 平均類似度 0.9885
- **Anger vs Apology**: 平均類似度 0.9911

**解釈**: 
- Pythia-160Mでは、全ての感情ペアが非常に高い類似度（0.98以上）を示す
- これは、Pythia-160Mでは感情方向がほぼ同じ方向を向いている可能性を示唆

#### GPT-Neo-125M
- **Gratitude vs Anger**: 平均類似度 0.6487
- **Gratitude vs Apology**: 平均類似度 0.7796
- **Anger vs Apology**: 平均類似度 0.8334

**解釈**: 
- GPT-Neo-125MはGPT-2と同様の傾向を示すが、AngerとApologyの類似度がより高い（0.83）

### 2. モデル間での感情方向の類似度

#### Gratitude（感謝）
- **GPT-2 vs Pythia-160M**: 平均類似度 -0.0001（ほぼ0）
- **GPT-2 vs GPT-Neo-125M**: 平均類似度 0.0020（ほぼ0）
- **Pythia-160M vs GPT-Neo-125M**: 平均類似度 -0.0115（ほぼ0）

#### Anger（怒り）
- **GPT-2 vs Pythia-160M**: 平均類似度 0.0029（ほぼ0）
- **GPT-2 vs GPT-Neo-125M**: 平均類似度 -0.0129（ほぼ0）
- **Pythia-160M vs GPT-Neo-125M**: 平均類似度 -0.0118（ほぼ0）

#### Apology（謝罪）
- **GPT-2 vs Pythia-160M**: 平均類似度 0.0070（ほぼ0）
- **GPT-2 vs GPT-Neo-125M**: 平均類似度 -0.0019（ほぼ0）
- **Pythia-160M vs GPT-Neo-125M**: 平均類似度 -0.0166（ほぼ0）

**重要な発見**: 
- **モデル間での感情方向の類似度はほぼ0**（-0.02から0.01の範囲）
- これは、異なるモデル間で感情方向が共通していないことを示唆
- 各モデルは独自の感情表現方法を持っている可能性

### 3. 層ごとの感情方向強度（L2 Norm）

各モデルで層ごとの感情方向ベクトルの強度を分析しました。詳細は可視化結果を参照してください。

**一般的な傾向**:
- 初期層（0-3層）では感情方向が弱い
- 中間層（4-8層）で感情方向が強くなる
- 後期層（9-11層）では強度が安定または減少

## 可視化結果

以下の可視化ファイルが生成されました：

### モデル別の可視化
各モデルごとに以下のプロットが作成されました（`results/baseline/plots/{model}/`）：

1. **層ごとのL2ノルム**: `{model}_layer_norms_l2.png`
   - 各感情カテゴリの方向ベクトルの強度を層ごとに表示

2. **感情間の距離**: `{model}_emotion_distances.png`
   - 感情ペア間のcosine類似度を層ごとに表示

3. **類似度ヒートマップ**: `{model}_similarity_heatmap_avg.png`
   - 感情間の類似度をマトリックス形式で表示

### モデル間比較の可視化
`results/baseline/plots/cross_model/` に以下のプロットが作成されました：

1. **Cross-model similarity (gratitude)**: `cross_model_gratitude.png`
2. **Cross-model similarity (anger)**: `cross_model_anger.png`
3. **Cross-model similarity (apology)**: `cross_model_apology.png`

## データファイル

### 感情方向ベクトル
- `results/baseline/emotion_vectors/gpt2_vectors.pkl`
- `results/baseline/emotion_vectors/pythia-160m_vectors.pkl`
- `results/baseline/emotion_vectors/gpt-neo-125m_vectors.pkl`

各ファイルには以下が含まれます：
- `emotion_vectors`: 感情ラベルをキーとする方向ベクトル [n_layers, d_model]
- `emotion_distances`: 感情ペア間の類似度
- `metadata`: モデル情報

### モデル間類似度テーブル
- `results/baseline/cross_model_similarity.csv`: モデル間の類似度を表形式で保存

## 技術的な実装詳細

### 使用ライブラリ
- **NumPy**: 数値計算とベクトル操作
- **Matplotlib**: 可視化
- **Pandas**: データテーブルの作成と保存

### 主要な関数
1. **EmotionVectorExtractor.compute_emotion_vector()**: 感情方向ベクトルの計算
2. **EmotionVectorExtractor.compute_cosine_similarity()**: Cosine類似度の計算
3. **EmotionVisualizer.plot_layer_norms()**: 層ごとのノルムの可視化
4. **EmotionVisualizer.plot_emotion_distances()**: 感情間の距離の可視化
5. **EmotionVisualizer.plot_cross_model_similarity()**: モデル間類似度の可視化

## 重要な発見と考察

### 1. モデル内での感情の区別
- **GPT-2**: 感情間の区別が明確（特にGratitudeとAnger）
- **Pythia-160M**: 感情間の区別が不明確（全ての感情が類似）
- **GPT-Neo-125M**: GPT-2と同様の傾向だが、AngerとApologyがより類似

### 2. モデル間での感情方向の非共通性
- **重要な発見**: モデル間での感情方向の類似度がほぼ0
- これは、各モデルが独自の感情表現方法を持っていることを示唆
- 感情の「普遍的な方向」は存在しない可能性

### 3. 層ごとの感情方向の強度
- 初期層では感情方向が弱く、中間層で強くなる傾向
- これは、感情の処理が中間層で行われることを示唆

## 次のステップ（フェーズ4）

フェーズ4では、Activation Patchingを使用して感情方向の操作が出力に与える因果的影響を検証します：

1. **Activation Patching実装**
   - 中立文に対する推論時にresidual streamを改変
   - `patched = resid + α * emotion_vec`

2. **実験設定**
   - baseline（操作なし）
   - 感謝方向を加算/抑制
   - 怒り方向を加算/抑制
   - 謝罪方向を加算/抑制

3. **評価**
   - 出力テキストの変化を記録
   - 文末の丁寧さ、感謝語の出現、トーンの変化を評価

## 課題と今後の拡張

### 課題
1. **モデル間での感情方向の非共通性**: 予想外の発見。より詳細な分析が必要
2. **Pythia-160Mでの感情の区別が不明確**: モデルの特性か、データの問題か要検討

### 今後の拡張案
1. **より多くのモデルでの検証**: 異なるアーキテクチャでの検証
2. **位置ごとの分析**: 文末だけでなく、文中の各位置での感情方向の分析
3. **Attention重みの分析**: 特定のattention headが感情を担っているかの検証
4. **より大規模なデータセット**: サンプル数を増やして安定性を検証

## 結論

フェーズ3は成功裏に完了しました。主要な発見は以下の通りです：

1. **モデル内での感情の区別**: GPT-2とGPT-Neo-125Mでは感情間の区別が明確だが、Pythia-160Mでは不明確

2. **モデル間での感情方向の非共通性**: 異なるモデル間で感情方向が共通していない（類似度ほぼ0）

3. **層ごとの感情方向の強度**: 中間層で感情方向が強くなる傾向

これらの発見は、感情表現がモデル固有である可能性を示唆しており、フェーズ4でのActivation Patching実験で因果的検証を行う価値があります。

