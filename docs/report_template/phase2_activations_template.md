# Phase 2 — Activation Extraction

## 🎯 目的

- GPT-2 の各 emotion の residual stream を抽出
- 層別 / トークン別の activations を保存

## 📦 生成物

- `results/phase2/activations_{emotion}.pkl` × models
- MLflowログ
- `docs/phase2_report.md`

## 🚀 実行コマンド例

```bash
python -m scripts.phase2_extract_all_activations --model gpt2
python -m scripts.phase2_extract_all_activations --model EleutherAI/pythia-160m
python -m scripts.phase2_extract_all_activations --model EleutherAI/gpt-neo-125M
```

## 📄 レポート項目

### 1. 前処理（トークン化）

#### トークナイザー
- モデル: [gpt2 / pythia-160m / gpt-neo-125M]
- ボキャブラリサイズ: [数]

#### トークン化設定
- max_length: [数]
- padding: [left / right]
- truncation: [Yes/No]

### 2. 抽出構造（層×トークン）

#### モデル構造
- モデル名: [gpt2 / pythia-160m / gpt-neo-125M]
- 層数: [数]
- Hidden size: [数]
- Attention heads: [数]

#### 抽出された活性の構造
- Residual stream: [shape]
- MLP出力: [shape]
- Attention: [shape] (オプション)

### 3. 実行時間

| モデル | 感情カテゴリ | サンプル数 | 実行時間 | 1サンプルあたり |
|--------|------------|-----------|---------|----------------|
| gpt2   | Gratitude  | [数]      | [時間]  | [時間]         |
| gpt2   | Anger      | [数]      | [時間]  | [時間]         |
| gpt2   | Apology    | [数]      | [時間]  | [時間]         |
| gpt2   | Neutral    | [数]      | [時間]  | [時間]         |

### 4. ファイルサイズ

| モデル | 感情カテゴリ | ファイルサイズ | 圧縮率 |
|--------|------------|---------------|--------|
| gpt2   | Gratitude  | [MB]          | [%]    |
| gpt2   | Anger      | [MB]          | [%]    |
| gpt2   | Apology    | [MB]          | [%]    |
| gpt2   | Neutral    | [MB]          | [%]    |

### 5. 注意点（batch化・max_seq_len）

#### Batch処理
- Batch size: [数]
- メモリ使用量: [GB]

#### シーケンス長
- max_seq_len: [数]
- 平均シーケンス長: [数]
- 最大シーケンス長: [数]

#### メモリ管理
- GPUメモリ使用量: [GB]
- CPUメモリ使用量: [GB]

### 6. MLflowログ

#### 記録されたメトリクス
- [メトリクス名]: [値]

#### アーティファクト
- [アーティファクト名]: [パス]

### 7. トラブルシューティング

#### 発生した問題
- [問題の説明]

#### 解決方法
- [解決手順]

## 📝 備考

[その他の注意事項やメモ]

