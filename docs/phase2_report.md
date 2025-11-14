# フェーズ2: 内部活性抽出 レポート

## 実行日時
2024年11月14日

## 概要
軽量LLMモデル（GPT-2 small、Pythia-160M、GPT-Neo-125M）を用いて、感情プロンプトデータセットから内部活性（residual stream、MLP出力）を抽出しました。

## 対象モデル

### 1. GPT-2 small (124M)
- **モデル名**: `gpt2`
- **アーキテクチャ**: GPT-2（古典的構造）
- **層数**: 12層
- **隠れ層サイズ**: 768
- **語彙サイズ**: 50,257

### 2. Pythia-160M
- **モデル名**: `EleutherAI/pythia-160m`
- **アーキテクチャ**: GPT-Neo系
- **層数**: 12層
- **隠れ層サイズ**: 768
- **語彙サイズ**: 50,304

### 3. GPT-Neo-125M
- **モデル名**: `EleutherAI/gpt-neo-125M`
- **アーキテクチャ**: GPT-Neo系（Pythiaと類似だが、より軽量）
- **層数**: 12層
- **隠れ層サイズ**: 768
- **語彙サイズ**: 50,257

## データセット

- **総サンプル数**: 200サンプル（英語のみ）
- **感情カテゴリ**: 
  - Gratitude（感謝）: 50サンプル
  - Anger（怒り）: 50サンプル
  - Apology（謝罪）: 50サンプル
  - Neutral（中立）: 50サンプル

## 抽出された活性

### GPT-2 small
以下のファイルが `results/activations/gpt2/` に保存されました：

- `activations_gratitude.pkl` - 感謝感情の活性（50サンプル）
- `activations_anger.pkl` - 怒り感情の活性（50サンプル）
- `activations_apology.pkl` - 謝罪感情の活性（50サンプル）
- `activations_neutral.pkl` - 中立感情の活性（50サンプル）

各ファイルには以下が含まれます：
- **Residual stream**: 12層分（各層は50サンプルのリスト）
- **MLP出力**: 12層分（各層は50サンプルのリスト）
- **メタデータ**: モデル情報、層数、隠れ層サイズなど

### Pythia-160M
以下のファイルが `results/activations/pythia-160m/` に保存されました：

- `activations_gratitude.pkl` - 感謝感情の活性（50サンプル）
- `activations_anger.pkl` - 怒り感情の活性（50サンプル）
- `activations_apology.pkl` - 謝罪感情の活性（50サンプル）
- `activations_neutral.pkl` - 中立感情の活性（50サンプル）

各ファイルには以下が含まれます：
- **Residual stream**: 12層分（各層は50サンプルのリスト）
- **MLP出力**: 12層分（各層は50サンプルのリスト）
- **メタデータ**: モデル情報、層数、隠れ層サイズなど

### GPT-Neo-125M
以下のファイルが `results/activations/gpt-neo-125m/` に保存されました：

- `activations_gratitude.pkl` - 感謝感情の活性（50サンプル）
- `activations_anger.pkl` - 怒り感情の活性（50サンプル）
- `activations_apology.pkl` - 謝罪感情の活性（50サンプル）
- `activations_neutral.pkl` - 中立感情の活性（50サンプル）

各ファイルには以下が含まれます：
- **Residual stream**: 12層分（各層は50サンプルのリスト）
- **MLP出力**: 12層分（各層は50サンプルのリスト）
- **メタデータ**: モデル情報、層数、隠れ層サイズなど

## データ構造

各活性ファイル（`.pkl`）の構造：

```python
{
    'residual_stream': [
        [sample1_layer0, sample2_layer0, ..., sample50_layer0],  # 層0
        [sample1_layer1, sample2_layer1, ..., sample50_layer1],  # 層1
        ...
        [sample1_layer11, sample2_layer11, ..., sample50_layer11]  # 層11
    ],
    'mlp_output': [
        [sample1_layer0, sample2_layer0, ..., sample50_layer0],  # 層0
        ...
    ],
    'tokens': [token_tensor1, token_tensor2, ..., token_tensor50],
    'token_strings': [token_strings1, token_strings2, ..., token_strings50],
    'labels': ['emotion1', 'emotion2', ..., 'emotion50'],
    'metadata': {
        'model_name': 'gpt2' or 'EleutherAI/pythia-160m',
        'n_layers': 12,
        'd_model': 768,
        'd_vocab': 50257 or 50304,
        'n_samples': 50,
        'emotion_label': 'gratitude' | 'anger' | 'apology' | 'neutral',
        'save_residual_stream': True,
        'save_mlp_output': True,
        'save_attention': False
    }
}
```

各サンプルの活性形状：
- **Residual stream**: `[pos, d_model]` - 位置ごとの隠れ状態
- **MLP出力**: `[pos, d_model]` - 位置ごとのMLP出力

## 処理時間

### GPT-2 small
- 各感情カテゴリ: 約2秒（50サンプル）
- 処理速度: 約25サンプル/秒
- 総処理時間: 約8秒（4カテゴリ）

### Pythia-160M
- 各感情カテゴリ: 約2秒（50サンプル）
- 処理速度: 約24-25サンプル/秒
- 総処理時間: 約8秒（4カテゴリ）

### GPT-Neo-125M
- 各感情カテゴリ: 約2秒（50サンプル）
- 処理速度: 約25-27サンプル/秒
- 総処理時間: 約8秒（4カテゴリ）

## 技術的な実装詳細

### 使用ライブラリ
- **TransformerLens**: モデルの読み込みとhook機能
- **PyTorch**: テンソル操作
- **NumPy**: 数値計算とデータ保存

### Hook実装
- `blocks.{layer_idx}.hook_resid_pre`: 各層のresidual stream（層への入力前）
- `blocks.{layer_idx}.hook_mlp_out`: 各層のMLP出力

### データ保存形式
- **形式**: Pickle（`.pkl`）
- **理由**: NumPy配列のリストを効率的に保存可能

## 検証結果

### 単体テスト
- `tests/test_extract_activations.py`: すべて成功
  - ActivationExtractorの初期化
  - 単一テキストの活性抽出
  - 複数テキストの活性抽出
  - データセット処理

### 結合テスト
- GPT-2 smallでgratitudeカテゴリの活性抽出: 成功
- 全感情カテゴリの活性抽出: 成功
- Pythia-160Mでの同様の処理: 成功

## 次のステップ（フェーズ3）

フェーズ3では、抽出された活性を使用して以下を実施します：

1. **感情方向ベクトルの抽出**
   - `emotion_vec[layer] = mean(resid_emotion[layer]) - mean(resid_neutral[layer])`
   - 各層ごとに感情方向を計算

2. **層ごとの強度分析**
   - L1 norm、L2 normの計算
   - cos-simによる安定性評価
   - 層ごとの強度を可視化

3. **モデル内での感情距離分析**
   - 感謝 vs 怒り、感謝 vs 謝罪、怒り vs 謝罪のcos-sim計算
   - 2D/3D plotで可視化

4. **モデル間での感情方向類似度**
   - 3モデル間（GPT-2 vs Pythia-160M、GPT-2 vs GPT-Neo-125M、Pythia-160M vs GPT-Neo-125M）の感情方向ベクトルのcos-sim計算
   - 感情カテゴリごとの比較表作成

## 課題と今後の拡張

### 課題
1. **Attention重みの保存**: メモリ使用量が多いため未実装
   - 必要に応じて後から実装可能

### 今後の拡張案
1. **Llama系モデルの追加**: Gemma-2BやLlama-3.2-1Bなど（認証が必要）
2. **Attention重みの分析**: 特定のattention headの感情への寄与度分析
3. **バッチ処理の最適化**: より効率的な活性抽出の実装
4. **データ圧縮**: 活性データの圧縮によるストレージ効率化

## 結論

フェーズ2は成功裏に完了しました。GPT-2 small、Pythia-160M、GPT-Neo-125Mの3モデルで、全感情カテゴリ（gratitude, anger, apology, neutral）の内部活性を正常に抽出できました。

**総合計:**
- **3モデル** × **4感情カテゴリ** = **12ファイル**
- **合計600サンプル**の活性データ（各モデル200サンプル）

抽出されたデータは、フェーズ3での感情方向ベクトル分析の準備が整っています。特に、GPT-2系とGPT-Neo系の2つの異なるアーキテクチャファミリー間での感情表現の共通性を検証できるようになりました。

