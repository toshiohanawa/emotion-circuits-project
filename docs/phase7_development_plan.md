# Phase 7: Head/Unitレベル解析 - 開発計画

## 現在の実装状況（2024年11月15日時点）

### ✅ 実装済みモジュール

1. **`src/analysis/head_screening.py`** ✅ **完了**
   - 実装状況: 実装済み・修正済み・動作確認済み
   - プロファイル対応: ✅ 対応済み（`--profile`オプション）
   - 実行結果: `results/baseline/alignment/head_scores_gpt2.json` (90KB, 432スコア)
   - 状態: **Phase 7-1完了**

2. **`src/models/head_ablation.py`** ✅ **実装済み（未テスト）**
   - 実装状況: コード実装済み
   - プロファイル対応: ❌ 未対応
   - CLI動作確認: ✅ `--help`で正常表示
   - 実行結果: なし（まだ実行していない）
   - 状態: **Phase 7-2実装済み・テスト待ち**

3. **`src/models/head_patching.py`** ✅ **実装済み（未テスト）**
   - 実装状況: コード実装済み
   - プロファイル対応: ❌ 未対応
   - CLI動作確認: ✅ `--help`で正常表示
   - 実行結果: なし（まだ実行していない）
   - 状態: **Phase 7-3実装済み・テスト待ち**

4. **`src/visualization/head_plots.py`** ✅ **実装済み（未テスト）**
   - 実装状況: コード実装済み
   - プロファイル対応: ❌ 未対応
   - CLI動作確認: ✅ `--help`で正常表示
   - 実行結果: なし（まだ実行していない）
   - 状態: **Phase 7-4実装済み・テスト待ち**

## Phase 7の開発計画

### ステップ1: プロファイル対応の追加（優先度: 高）

**目的**: 既存のhead_ablation, head_patching, head_plotsにプロファイル対応を追加し、一貫したワークフローを実現

#### 1-1. `head_ablation.py`のプロファイル対応

**変更内容**:
- `--profile`オプションを追加
- `ProjectContext`を使用してプロンプトファイルを自動解決
- 出力パスをプロファイルベースに変更

**変更後のCLI**:
```bash
python -m src.models.head_ablation \
  --model gpt2 \
  --profile baseline \
  --head-spec "3:5,7:2" \
  --prompts-file data/gratitude_prompts.json \
  --output results/baseline/patching/head_ablation/gpt2_gratitude.pkl
```

#### 1-2. `head_patching.py`のプロファイル対応

**変更内容**:
- `--profile`オプションを追加
- `ProjectContext`を使用してプロンプトファイルを自動解決
- 出力パスをプロファイルベースに変更

**変更後のCLI**:
```bash
python -m src.models.head_patching \
  --model gpt2 \
  --profile baseline \
  --head-spec "3:5,7:2" \
  --neutral-prompts data/neutral_prompts.json \
  --emotion-prompts data/gratitude_prompts.json \
  --output results/baseline/patching/head_patching/gpt2_gratitude.pkl
```

#### 1-3. `head_plots.py`のプロファイル対応

**変更内容**:
- `--profile`オプションを追加
- デフォルトの入力パスをプロファイルベースに変更

**変更後のCLI**:
```bash
python -m src.visualization.head_plots \
  --profile baseline \
  --head-scores results/baseline/alignment/head_scores_gpt2.json \
  --output-dir results/baseline/plots/heads
```

### ステップ2: Head Screening結果の活用（優先度: 高）

**目的**: Head screeningで特定された重要なheadを使用して、ablation/patching実験を実行

#### 2-1. Head screening結果の確認

**実行済み**:
- GPT-2でhead screening完了
- 上位head: Layer 0 (Head 0, 1, 4, 7), Layer 3 (Head 5) など

**次のステップ**:
- 上位headを特定（例: Top 5-10 head）
- これらのheadを使用してablation/patching実験を実行

#### 2-2. 重要なheadの特定

**スクリプト例**:
```python
# head_scores_gpt2.jsonから上位headを抽出
import json
data = json.load(open('results/baseline/alignment/head_scores_gpt2.json'))
sorted_scores = sorted(data['scores'], key=lambda x: abs(x['delta_attn']), reverse=True)
top_heads = sorted_scores[:10]
# 例: [(0, 0), (0, 1), (3, 5), ...]
```

### ステップ3: Head Ablation実験の実行（優先度: 中）

**目的**: 重要なheadをゼロアウトして、感情トーンやsentimentの変化を測定

#### 3-1. 実験設計

**対象head**:
- Head screeningで特定された上位head（例: Layer 0 Head 0, Layer 3 Head 5）
- 複数のheadを組み合わせて実験

**評価指標**:
- Sentimentスコアの変化
- 感情キーワード頻度の変化
- 生成テキストの質的変化

#### 3-2. 実行計画

```bash
# 例1: Layer 0 Head 0をablation
python -m src.models.head_ablation \
  --model gpt2 \
  --profile baseline \
  --head-spec "0:0" \
  --prompts-file data/gratitude_prompts.json \
  --output results/baseline/patching/head_ablation/gpt2_gratitude_layer0_head0.pkl

# 例2: 複数headをablation
python -m src.models.head_ablation \
  --model gpt2 \
  --profile baseline \
  --head-spec "0:0,0:1,3:5" \
  --prompts-file data/gratitude_prompts.json \
  --output results/baseline/patching/head_ablation/gpt2_gratitude_multiple.pkl
```

### ステップ4: Head Patching実験の実行（優先度: 中）

**目的**: 感情文のhead出力を中立文に移植して、感情トーンの変化を測定

#### 4-1. 実験設計

**対象head**:
- Head screeningで特定された上位head
- 感情カテゴリごとに実験（gratitude, anger, apology）

#### 4-2. 実行計画

```bash
# Gratitude head patching
python -m src.models.head_patching \
  --model gpt2 \
  --profile baseline \
  --head-spec "0:0,3:5" \
  --neutral-prompts data/neutral_prompts.json \
  --emotion-prompts data/gratitude_prompts.json \
  --output results/baseline/patching/head_patching/gpt2_gratitude.pkl

# Anger head patching
python -m src.models.head_patching \
  --model gpt2 \
  --profile baseline \
  --head-spec "0:0,3:5" \
  --neutral-prompts data/neutral_prompts.json \
  --emotion-prompts data/anger_prompts.json \
  --output results/baseline/patching/head_patching/gpt2_anger.pkl
```

### ステップ5: 可視化と結果分析（優先度: 中）

**目的**: Head解析結果を可視化して、重要な発見をまとめる

#### 5-1. Head screening結果の可視化

```bash
python -m src.visualization.head_plots \
  --profile baseline \
  --head-scores results/baseline/alignment/head_scores_gpt2.json \
  --output-dir results/baseline/plots/heads \
  --top-n 20
```

#### 5-2. Ablation/Patching結果の可視化

```bash
python -m src.visualization.head_plots \
  --profile baseline \
  --ablation-file results/baseline/patching/head_ablation/gpt2_gratitude_layer0_head0.pkl \
  --patching-file results/baseline/patching/head_patching/gpt2_gratitude.pkl \
  --output-dir results/baseline/plots/heads
```

### ステップ6: 複数モデルでの実行（優先度: 低）

**目的**: GPT-2だけでなく、Pythia-160M、GPT-Neo-125Mでもhead解析を実行

#### 6-1. 実行計画

```bash
# Pythia-160M
python -m src.analysis.head_screening --model EleutherAI/pythia-160m --profile baseline --output results/baseline/alignment/head_scores_pythia-160m.json

# GPT-Neo-125M
python -m src.analysis.head_screening --model EleutherAI/gpt-neo-125M --profile baseline --output results/baseline/alignment/head_scores_gpt-neo-125m.json
```

## 実装優先順位

### 最優先（即座に実施）

1. ✅ **Head screeningの修正完了** - 完了済み
2. **プロファイル対応の追加** - head_ablation, head_patching, head_plots
3. **Head screening結果の可視化** - 基本的な可視化を実行

### 優先度: 高（1週間以内）

4. **Head ablation実験の実行** - GPT-2で上位headをablation
5. **Head patching実験の実行** - GPT-2で上位headをpatching
6. **結果の可視化と分析** - ablation/patching結果の可視化

### 優先度: 中（2週間以内）

7. **複数モデルでのhead screening** - Pythia-160M, GPT-Neo-125M
8. **モデル間比較** - 異なるモデルでのhead反応度の比較
9. **統合レポート作成** - Phase 7の結果をまとめたレポート

### 優先度: 低（オプション）

10. **Unitレベル解析** - MLP unitの解析（Phase 7の拡張）
11. **Head回路の可視化** - Attentionパターンの詳細可視化

## 期待される成果物

### データファイル

- `results/baseline/alignment/head_scores_{model}.json` - Head screening結果
- `results/baseline/patching/head_ablation/{model}_{emotion}_{heads}.pkl` - Ablation実験結果
- `results/baseline/patching/head_patching/{model}_{emotion}.pkl` - Patching実験結果

### 可視化ファイル

- `results/baseline/plots/heads/head_reaction_heatmap.png` - Head反応度heatmap
- `results/baseline/plots/heads/top_heads.png` - 上位headの棒グラフ
- `results/baseline/plots/heads/ablation_sentiment_comparison.png` - Ablationによるsentiment変化
- `results/baseline/plots/heads/patching_keyword_comparison.png` - Patchingによるキーワード変化

### レポート

- `docs/report/phase7_head_analysis_report.md` - Phase 7の統合レポート

## 次のアクション

1. **即座に実施**: プロファイル対応の追加（head_ablation, head_patching, head_plots）
2. **次に実施**: Head screening結果の可視化
3. **その後**: Head ablation/patching実験の実行

## 注意事項

- Head ablation/patching実験は時間がかかる可能性がある（各プロンプトで生成が必要）
- メモリ使用量に注意（hookの適切な解除が必要）
- 結果ファイルは大きくなる可能性がある（生成テキストを含む）

