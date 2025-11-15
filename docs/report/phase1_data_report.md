# Phase 1 — Dataset Construction

## 🎯 目的

- 感情データセット（emotion / neutral）を生成
- 英語フィルタリング / バランス調整
- 基本統計の取得

## 📦 生成物

- `data/emotion_dataset.jsonl` ✅
- `data/emotion_dataset_extended.jsonl` ✅
- `results/phase1/phase1_stats.json` ✅
- `docs/report/phase1_data_report.md` ✅

## 🚀 実行コマンド例

```bash
python -m src.data.build_dataset --profile baseline
python -m src.data.build_dataset --profile extended
python -m src.data.validate_dataset data/emotion_dataset.jsonl
```

## 📄 レポート項目

### 1. 使用したプロンプトテンプレート

#### 感情カテゴリ
- **Gratitude（感謝）**: "Thank you very much for your help.", "I really appreciate your assistance.", "I'm so grateful for your support." など
- **Anger（怒り）**: "I'm quite frustrated with this situation.", "I'm very disappointed with this outcome.", "This is extremely frustrating." など
- **Apology（謝罪）**: "I sincerely apologize for the inconvenience.", "I deeply regret the mistake I made.", "I'm truly sorry for what happened." など
- **Neutral（中立）**: "What is the weather like today?", "Can you tell me the time?", "How does this work?" など

#### データソース
- 手動で作成した英語プロンプト
- 各感情カテゴリごとに独立したJSONファイルとして保存（`data/{emotion}_prompts.json`）
- Baseline版とExtended版の2種類を用意

### 2. データ構築ステップ

1. **個別プロンプトファイルの作成**: `src/data/create_individual_prompt_files.py`を使用して各感情カテゴリのJSONファイルを作成
2. **JSONLデータセットの構築**: `src/data/build_dataset.py`を使用してプロファイル（baseline/extended）ごとにJSONL形式のデータセットを構築
3. **データ検証**: `src/data/validate_dataset.py`を使用してデータセットの整合性を確認

### 3. 最終サンプル数（感情別）

| 感情カテゴリ | Baseline | Extended | 合計 |
|------------|----------|----------|------|
| Gratitude  | 70       | 100      | 170  |
| Anger      | 70       | 100      | 170  |
| Apology    | 70       | 100      | 170  |
| Neutral    | 70       | 100      | 170  |
| **合計**   | 280      | 400      | 680  |

### 4. 統計（文字数 / token数 / 言語比率）

#### 文字数統計（Baseline）
- 平均文字数: 27.7
- 最小文字数: 6
- 最大文字数: 50
- 標準偏差: 8.7

#### 文字数統計（Extended）
- 平均文字数: 33.0
- 最小文字数: 6
- 最大文字数: 71
- 標準偏差: 11.8

#### Token数統計（Baseline、サンプリング）
- 平均token数: 7.1
- 最小token数: 3
- 最大token数: 13
- 標準偏差: 1.9

#### Token数統計（Extended、サンプリング）
- 平均token数: 7.1
- 最小token数: 3
- 最大token数: 13
- 標準偏差: 1.9

#### 言語比率
- 英語: 100%
- その他: 0%

### 5. サンプル例

#### Gratitude
```
Thank you very much for your help.
I really appreciate your assistance.
I'm so grateful for your support.
```

#### Anger
```
I'm quite frustrated with this situation.
I'm very disappointed with this outcome.
This is extremely frustrating.
```

#### Apology
```
I sincerely apologize for the inconvenience.
I deeply regret the mistake I made.
I'm truly sorry for what happened.
```

#### Neutral
```
What is the weather like today?
Can you tell me the time?
How does this work?
```

### 6. 考察 / 課題

#### データ品質
- すべてのプロンプトが英語で統一されている
- 各感情カテゴリが明確に区別されている
- BaselineとExtendedでバランスが取れている

#### バランス
- Baseline: 各感情カテゴリ70サンプル（25%ずつ）で完全にバランスが取れている
- Extended: 各感情カテゴリ100サンプル（25%ずつ）で完全にバランスが取れている

#### 課題
- Token数が比較的短い（平均7.1トークン）ため、長文での感情表現の検証には限界がある可能性
- Extendedデータセットでも文字数のばらつきが大きい（std: 11.8）

#### 今後の改善案
- より長い文脈での感情表現を含むデータセットの追加
- 多様な文体（フォーマル/カジュアル）のバランス調整

## 📝 備考

- データセットは`data/`ディレクトリに保存されている
- プロファイル（baseline/extended）を使用することで、一貫したデータセット管理が可能
- 統計情報は`results/phase1/phase1_stats.json`に保存されている

