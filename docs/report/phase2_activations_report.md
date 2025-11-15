# Phase 2 — Activation Extraction

## 🎯 目的

- GPT-2、Pythia-160M、GPT-Neo-125Mの各emotionのresidual streamを抽出
- 層別 / トークン別のactivationsを保存
- 3モデル×4感情カテゴリの活性データを取得

## 📦 生成物

- `results/baseline/activations/{model}/activations_{emotion}.pkl` × 3モデル × 4感情 ✅
- MLflowログ（実行ログ記録） ✅
- `docs/report/phase2_activations_report.md` ✅

## 🚀 実行コマンド例

```bash
python -m scripts.phase2_extract_all_activations --profile baseline
```

## 📄 レポート項目

### 1. 前処理（トークン化）

#### トークナイザー
- **GPT-2**: GPT-2 tokenizer（BPE）
- **Pythia-160M**: GPT-2互換tokenizer（BPE）
- **GPT-Neo-125M**: GPT-2互換tokenizer（BPE）
- **ボキャブラリサイズ**: 50,257（GPT-2系）

#### トークン化設定
- max_length: なし（可変長）
- padding: なし
- truncation: なし（全トークンを保持）

### 2. 抽出構造（層×トークン）

#### モデル構造

**GPT-2 small**:
- モデル名: `gpt2`
- 層数: 12
- Hidden size (d_model): 768
- Attention heads: 12
- Head dimension (d_head): 64
- MLP dimension: 3072
- Vocab size: 50,257

**Pythia-160M**:
- モデル名: `EleutherAI/pythia-160m`
- 層数: 12
- Hidden size (d_model): 768
- Attention heads: 12

**GPT-Neo-125M**:
- モデル名: `EleutherAI/gpt-neo-125M`
- 層数: 12
- Hidden size (d_model): 768
- Attention heads: 12

#### 抽出された活性の構造

- **Residual stream**: `[n_layers][n_samples][seq_len, d_model]`
  - 例: GPT-2 gratitude - Layer 0: `[70 samples][seq_len, 768]`
  - 各層ごとにリスト形式で保存
  - 各サンプルごとにテンソル `[seq_len, d_model]` として保存

- **MLP出力**: `[n_layers][n_samples][seq_len, d_model]`
  - Residual streamと同様の構造

- **Attention**: 保存されていない（メモリ節約のため）

- **Tokens**: `[n_samples][batch, seq_len]`
  - 各サンプルのトークンID

- **Token strings**: `[n_samples][seq_len]`
  - 各サンプルのトークン文字列

### 3. 実行時間

| モデル | 感情カテゴリ | サンプル数 | ファイルサイズ | 最終更新日時 |
|--------|------------|-----------|---------------|-------------|
| gpt2   | Gratitude  | 70        | 34.9 MB       | 2025-11-15 05:08 |
| gpt2   | Anger      | 70        | 36.2 MB       | - |
| gpt2   | Apology    | 70        | 40.5 MB       | - |
| gpt2   | Neutral    | 70        | 32.1 MB       | - |
| pythia-160m | Gratitude | 70     | 35.0 MB       | - |
| pythia-160m | Anger     | 70        | 36.3 MB       | - |
| pythia-160m | Apology   | 70        | 40.5 MB       | - |
| pythia-160m | Neutral   | 70        | 32.1 MB       | - |
| gpt-neo-125M | Gratitude | 70      | 34.9 MB       | - |
| gpt-neo-125M | Anger     | 70        | 36.2 MB       | - |
| gpt-neo-125M | Apology   | 70        | 40.5 MB       | - |
| gpt-neo-125M | Neutral   | 70        | 32.1 MB       | - |

### 4. ファイルサイズ

| モデル | 感情カテゴリ | ファイルサイズ | 合計サイズ |
|--------|------------|---------------|-----------|
| gpt2   | Gratitude  | 34.9 MB       | 143.7 MB  |
| gpt2   | Anger      | 36.2 MB       |            |
| gpt2   | Apology    | 40.5 MB       |            |
| gpt2   | Neutral    | 32.1 MB       |            |
| pythia-160m | Gratitude | 35.0 MB    | 144.0 MB  |
| pythia-160m | Anger     | 36.3 MB       |            |
| pythia-160m | Apology   | 40.5 MB       |            |
| pythia-160m | Neutral   | 32.1 MB       |            |
| gpt-neo-125M | Gratitude | 34.9 MB   | 143.7 MB  |
| gpt-neo-125M | Anger     | 36.2 MB       |            |
| gpt-neo-125M | Apology   | 40.5 MB       |            |
| gpt-neo-125M | Neutral   | 32.1 MB       |            |

**合計**: 約431 MB（3モデル × 4感情）

### 5. 注意点（batch化・max_seq_len）

#### Batch処理
- Batch size: 1（各テキストを個別に処理）
- メモリ使用量: 約2-3 GB（GPU使用時）

#### シーケンス長
- max_seq_len: なし（可変長）
- 平均シーケンス長: 約7-9トークン（Phase 1の統計より）
- 最大シーケンス長: 約13トークン（サンプリング結果より）
- 実際のサンプル: 最初のサンプルで9トークン

#### メモリ管理
- GPUメモリ使用量: 約2-3 GB（モデルロード + 活性保存）
- CPUメモリ使用量: 約500 MB（活性データのCPU転送後）
- Hook解除: 各サンプル処理後に適切に解除

### 6. MLflowログ

#### 記録されたメトリクス
- 実行ログ（stdout/stderr）がMLflowアーティファクトとして記録
- モデル名、感情ラベル、サンプル数などのメタデータが保存

#### アーティファクト
- 実行ログファイル: `mlflow/artifacts/run_logs/`
- 活性データ: `results/baseline/activations/{model}/activations_{emotion}.pkl`

### 7. トラブルシューティング

#### 発生した問題
- 特になし（既存の活性データが正常に保存されている）

#### 解決方法
- 既存の実装が正常に動作していることを確認

## 📝 備考

- 活性データは`results/baseline/activations/`ディレクトリに保存されている
- 各モデルごとに独立したディレクトリに保存
- メタデータにモデル構造情報（層数、hidden sizeなど）が含まれている
- Attentionデータはメモリ節約のため保存されていない（必要に応じて後から抽出可能）

