# Emotion Circuits Project

軽量LLMを用いた「感情表現回路」探索プロジェクト

## プロジェクト概要

大規模言語モデル（LLM）は、人間のような「感謝」「怒り」「謝罪」などの感情・社会的表現を自然に生成します。しかし、これらの感情がモデル内部でどのように表現されているのか（どの層・どの方向・どの回路が担当しているか）については未解明です。

本プロジェクトでは、軽量なオープンソースLLMを複数モデル使用し、感情の内部表現を解析し、モデル間で共通性があるかを検証する研究を行います。

**核心的な研究クエスチョン**: 「異なるLLMの中に、"座標系は違うが本質的には同じ" 感情表現の部分空間は存在するのか？そしてそれを因果的に操作できるのか？」

## 対象モデル

1. **GPT-2 small (124M)** - 古典的構造の基準モデルとして必須
2. **Pythia-160M** - GPT-Neo系アーキテクチャ、訓練データが透明
3. **GPT-Neo-125M** - GPT-Neo系（TinyLlamaの代替）

## 実装状況

### 完了したフェーズ

- ✅ **フェーズ0**: 環境構築
- ✅ **フェーズ1**: 感情プロンプトデータ作成（英語のみ）
- ✅ **フェーズ2**: 内部活性の抽出
- ✅ **フェーズ3**: 感情方向ベクトルの抽出・可視化
- ✅ **フェーズ3.5**: 感情語トークンベースのベクトル抽出とサブスペース解析
- ✅ **フェーズ4ライト**: 簡易Activation Patching
- ✅ **フェーズ5**: 層×αのスイープ実験と因果力比較
- ✅ **フェーズ6**: サブスペース構造とアライメント（線形写像学習、k-sweep、Procrustesアライメント）
- ✅ **フェーズ7**: Head/Unitレベル解析（head screening、ablation、patching）

### 実装済みモジュール

#### データ処理
- `src/data/create_emotion_dataset.py` - 感情プロンプトデータセット作成
- `src/data/create_individual_prompt_files.py` - 個別プロンプトJSONファイル作成
- `src/data/build_dataset.py` - プロンプトJSONからJSONLデータセット構築
- `src/data/validate_dataset.py` - データセット検証

#### モデル操作
- `src/models/extract_activations.py` - 内部活性抽出
- `src/models/activation_patching.py` - 簡易Activation Patching
- `src/models/activation_patching_sweep.py` - 層×αスイープ実験
- `src/models/activation_patching_iterative.py` - Iterative Patching
- `src/models/activation_patching_swap.py` - Swap Patching
- `src/models/activation_patching_random.py` - ランダム対照Patching実験
- `src/models/head_ablation.py` - Head ablation実験
- `src/models/head_patching.py` - Head patching実験

#### 分析
- `src/analysis/emotion_vectors.py` - 感情方向ベクトル抽出（文末ベース）
- `src/analysis/emotion_vectors_token_based.py` - 感情方向ベクトル抽出（トークンベース）
- `src/analysis/emotion_subspace.py` - 感情サブスペース解析（PCA）
- `src/analysis/cross_model_analysis.py` - モデル間比較
- `src/analysis/cross_model_token_based.py` - トークンベースモデル間比較
- `src/analysis/cross_model_subspace.py` - サブスペースモデル間比較
- `src/analysis/subspace_utils.py` - サブスペース解析ユーティリティ（PCA、主角度、overlap、アライメント）
- `src/analysis/model_alignment.py` - Neutral空間での線形写像学習
- `src/analysis/subspace_k_sweep.py` - サブスペース次元kのスイープ実験
- `src/analysis/subspace_alignment.py` - Procrustes/CCAアライメント
- `src/analysis/compare_patching_methods.py` - Patching手法の比較
- `src/analysis/random_vs_emotion_effect.py` - ランダム対照 vs 感情ベクトルの効果比較
- `src/analysis/sentiment_eval.py` - Sentiment評価
- `src/analysis/head_screening.py` - Headスクリーニング

#### 可視化
- `src/visualization/emotion_plots.py` - 感情ベクトル可視化
- `src/visualization/patching_heatmaps.py` - Patchingヒートマップ
- `src/visualization/sentiment_plots.py` - Sentiment可視化
- `src/visualization/alignment_plots.py` - アライメント結果可視化
- `src/visualization/layer_subspace_plots.py` - 層ごとサブスペース可視化
- `src/visualization/head_plots.py` - Head解析結果可視化

#### ユーティリティ
- `src/utils/mlflow_utils.py` - MLflow実験追跡ユーティリティ
- `src/utils/hf_hooks.py` - HuggingFaceモデル用フック（Phase 8用）

## セットアップ

### 前提条件

- Python 3.9以上（3.11推奨）
- [uv](https://github.com/astral-sh/uv) パッケージマネージャー（推奨）
- CUDA対応GPU（推奨、CPUでも動作可能）
- 8GB以上のメモリ（GPU使用時は16GB以上推奨）

### クイックスタート

1. **リポジトリをクローン**
```bash
git clone <repository-url>
cd emotion-circuits-project
```

2. **仮想環境を作成（uvを使用）**
```bash
uv venv
source .venv/bin/activate  # Linux/Mac
# または
.venv\Scripts\activate  # Windows
```

3. **依存関係をインストール**
```bash
# uvを使用する場合（推奨）
uv pip install -e .

# または通常のpipを使用する場合
pip install -r requirements.txt
pip install -e .
```

4. **動作確認**
```bash
python -m src.models.extract_activations --help
```

**詳細なセットアップ手順は [`SETUP.md`](SETUP.md) を参照してください。**

## ディレクトリ構造

```
emotion-circuits-project/
├── data/                      # データセット
│   ├── neutral_prompts.json   # 中立プロンプト（70文）
│   ├── gratitude_prompts.json # 感謝プロンプト（70文）
│   ├── anger_prompts.json     # 怒りプロンプト（70文）
│   ├── apology_prompts.json   # 謝罪プロンプト（70文）
│   ├── *_prompts_extended.json # 拡張版プロンプト（各100文）
│   ├── emotion_dataset.jsonl   # Baselineデータセット（280サンプル）
│   ├── emotion_dataset_extended.jsonl # Extendedデータセット（400サンプル）
│   └── real_world_samples.json # 実世界テキストサンプル（35文）
├── docs/                      # ドキュメント
│   ├── proposal/              # 企画書
│   ├── implementation_plan.md # 実装計画（詳細）
│   ├── report/                # Phaseレポート（統合版）
│   │   ├── phase0_setup_report.md
│   │   ├── phase1_data_report.md
│   │   ├── phase2_activations_report.md
│   │   ├── phase3_vectors_report.md
│   │   ├── phase3.5_subspace_report.md
│   │   ├── phase4_patching_report.md
│   │   ├── phase5_sweep_report.md
│   │   └── phase6_alignment_report.md
│   └── archive/               # 過去のレポート（アーカイブ）
├── notebooks/                 # Jupyterノートブック（Gitに追跡）
├── results/                   # 実験結果（データセットプロファイルごと）
│   ├── baseline/              # Baselineデータセットの成果物
│   │   ├── activations/       # 抽出した活性データ（*.pklはGit除外）
│   │   ├── emotion_vectors/   # 感情方向ベクトル（*.pklはGit除外）
│   │   ├── emotion_subspaces/ # サブスペースデータ（*.pklはGit除外）
│   │   ├── patching/          # Patching実験結果（*.pklはGit除外）
│   │   ├── alignment/         # アライメント実験結果（*.pkl, *.ptはGit除外）
│   │   ├── plots/             # 可視化結果（Gitに追跡）
│   │   └── cross_model_*.csv  # モデル間比較結果（CSV）
│   └── extended/              # Extendedデータセットの成果物
│       ├── activations/       # 拡張版活性データ
│       ├── emotion_vectors/   # 拡張版ベクトル
│       ├── emotion_subspaces/ # 拡張版サブスペース
│       ├── patching/          # 拡張版Patching結果
│       ├── patching_random/   # ランダム対照実験結果
│       ├── alignment/         # 拡張版アライメント結果
│       └── plots/             # 拡張版可視化結果
├── src/                       # ソースコード
│   ├── data/                  # データ処理
│   ├── models/                 # モデル操作
│   ├── analysis/              # 分析モジュール
│   ├── visualization/         # 可視化モジュール
│   └── utils/                 # ユーティリティ
├── tests/                     # テストコード
├── pyproject.toml             # プロジェクト設定・依存関係
└── README.md                  # このファイル
```

## 使用方法

### 1. データセットの準備

感情プロンプトデータセットを作成：

```bash
# 個別プロンプトファイルの作成
python -m src.data.create_individual_prompt_files --data_dir data

# Baselineデータセットの構築
python -m src.data.build_dataset --profile baseline

# Extendedデータセットの構築（オプション）
python -m src.data.build_dataset --profile extended
```

### 2. 内部活性の抽出

以下の例はBaselineプロファイル（`results/baseline`）を想定しています。Extendedプロファイルで実行する場合は、`baseline`を`extended`に置き換えてください。

モデルから内部活性を抽出：

```bash
# GPT-2 small
python -m src.models.extract_activations \
  --model gpt2 \
  --dataset data/emotion_dataset.jsonl \
  --output results/baseline/activations/gpt2/

# Pythia-160M
python -m src.models.extract_activations \
  --model EleutherAI/pythia-160m \
  --dataset data/emotion_dataset.jsonl \
  --output results/baseline/activations/pythia-160m/
```

### 3. 感情方向ベクトルの抽出

```bash
# 文末ベース
python -m src.analysis.emotion_vectors \
  --activations_dir results/baseline/activations/gpt2 \
  --output results/baseline/emotion_vectors/gpt2_vectors.pkl

# トークンベース
python -m src.analysis.emotion_vectors_token_based \
  --activations_dir results/baseline/activations/gpt2 \
  --output results/baseline/emotion_vectors/gpt2_vectors_token_based.pkl
```

### 4. サブスペース解析

```bash
# 感情サブスペースの計算
python -m src.analysis.emotion_subspace \
  --activations_dir results/baseline/activations/gpt2 \
  --output results/baseline/subspaces/gpt2_subspaces.pkl \
  --n-components 10

# k-sweep実験
python -m src.analysis.subspace_k_sweep \
  --activations_dir results/baseline/activations \
  --model1 gpt2 \
  --model2 pythia-160m \
  --output results/baseline/alignment/k_sweep_gpt2_pythia.json \
  --k-values 2 5 10 20 \
  --layers 3 5 7 9 11
```

> ℹ️ **プロファイルベースのワークフロー**: 以下のスクリプトは `--profile` 引数で `baseline` / `extended` を切り替えられます（省略時はbaseline）:
> - `src.data.build_dataset`: データセット構築
> - `src.analysis.cross_model_analysis`: モデル間ベクトル類似度分析
> - `src.analysis.cross_model_token_based`: トークンベースモデル間分析
> - `src.analysis.cross_model_subspace`: モデル間サブスペースoverlap分析
> - `src.analysis.head_screening`: Headスクリーニング
> - `src.visualization.emotion_plots`: 感情ベクトル可視化
> - `scripts/phase1_log_to_mlflow.py`: MLflowログ記録
> - `scripts/phase2_extract_all_activations.py`: 活性抽出スイープ

### 5. モデル間アライメント

```bash
# Neutral空間での線形写像学習
python -m src.analysis.model_alignment \
  --model1 gpt2 \
  --model2 EleutherAI/pythia-160m \
  --neutral_prompts_file data/neutral_prompts.json \
  --model1_activations_dir results/baseline/activations/gpt2 \
  --model2_activations_dir results/baseline/activations/pythia-160m \
  --output results/baseline/alignment/model_alignment_gpt2_pythia.pkl \
  --n-components 10 \
  --layers 3 5 7 9 11
```

### 6. Activation Patching実験

```bash
# 層×αスイープ実験
python -m src.models.activation_patching_sweep \
  --model gpt2 \
  --vectors_file results/baseline/emotion_vectors/gpt2_vectors_token_based.pkl \
  --prompts_file data/neutral_prompts.json \
  --output results/baseline/patching/gpt2_sweep.pkl \
  --layers 3 5 7 9 11 \
  --alpha -2 -1 -0.5 0 0.5 1 2
```

### 7. Head解析

```bash
# Headスクリーニング（プロファイルを使用）
python -m src.analysis.head_screening \
  --model gpt2 \
  --profile baseline \
  --output results/baseline/alignment/head_scores_gpt2.json

# Head ablation
python -m src.models.head_ablation \
  --model gpt2 \
  --head-spec "3:5,7:2" \
  --prompts-file data/gratitude_prompts.json \
  --output results/baseline/patching/head_ablation_gpt2_gratitude.pkl

# Head patching
python -m src.models.head_patching \
  --model gpt2 \
  --head-spec "3:5,7:2" \
  --neutral-prompts data/neutral_prompts.json \
  --emotion-prompts data/gratitude_prompts.json \
  --output results/baseline/patching/head_patching_gpt2_gratitude.pkl
```

### 8. 可視化

```bash
# Head解析結果の可視化
python -m src.visualization.head_plots \
  --head-scores results/baseline/alignment/head_scores_gpt2.json \
  --ablation-file results/baseline/patching/head_ablation_gpt2_gratitude.pkl \
  --patching-file results/baseline/patching/head_patching_gpt2_gratitude.pkl \
  --output-dir results/baseline/plots/heads

# アライメント結果の可視化
python -m src.visualization.alignment_plots \
  --results_file results/baseline/alignment/model_alignment_gpt2_pythia.pkl \
  --output_dir results/baseline/plots/alignment
```

## 主要な発見

### Phase 3.5の核心的な発見

1. **Token-basedベクトルの有効性**: 感情語トークンベースのベクトル抽出により、Pythia-160Mの「全部0.99」現象が大幅に解消され、より明確な感情表現が抽出可能

2. **サブスペースレベルでの共通性**: モデル間のサブスペースoverlapが0.13-0.15（ランダムベースライン0.0-0.1より高い）を示し、モデル間で共有されたサブスペース構造が存在することを示唆

### Phase 6の核心的な発見

1. **低次元での共有構造**: k-sweep実験により、k=2でoverlapが最も高く（0.002〜0.005）、コアな共有因子が低次元に存在することを示唆

2. **線形写像による大幅な改善**: Neutral空間で学習した線形写像により、感情サブスペースのoverlapが**0.001から0.99**まで大幅に改善。これは「座標系は違うが本質的には同じ構造」という仮説を強く支持

3. **層依存性**: 深い層（9, 11）で特にアライメント効果が大きい。Layer 3では、cos²改善が+0.99に到達

4. **Extendedデータセットでの検証**: Extendedデータセット（400サンプル）でも同様のパターンが確認され、発見の頑健性が示された

詳細は`docs/report/phase6_alignment_report.md`を参照してください。

## データファイル

### プロンプトデータ

**Baselineデータセット**:
- `data/neutral_prompts.json`: 中立プロンプト70文
- `data/gratitude_prompts.json`: 感謝プロンプト70文
- `data/anger_prompts.json`: 怒りプロンプト70文
- `data/apology_prompts.json`: 謝罪プロンプト70文
- `data/emotion_dataset.jsonl`: Baseline統合データセット（280サンプル）

**Extendedデータセット**:
- `data/*_prompts_extended.json`: 各感情カテゴリ100文（拡張版）
- `data/emotion_dataset_extended.jsonl`: Extended統合データセット（400サンプル）

**実世界データ**:
- `data/real_world_samples.json`: 実世界テキストサンプル35文（SNS、レビュー、メール）

### 結果データ

実験結果は`results/`以下に保存されます：

- **活性データ**: `results/baseline/activations/{model_name}/activations_{emotion}.pkl`（各ファイル20-30MB）
- **感情ベクトル**: `results/baseline/emotion_vectors/{model}_vectors.pkl`（各ファイル約100KB）
- **サブスペース**: `results/baseline/emotion_subspaces/{model}_subspaces.pkl`（各ファイル約1.5MB）
- **アライメント結果**: `results/baseline/alignment/*.pkl`, `results/baseline/alignment/*.json`（数MB〜数十MB）
- **Patching結果**: `results/baseline/patching/*.pkl`（数百KB〜数MB）
- **可視化**: `results/baseline/plots/**/*.png`（Gitに追跡されます）

**注意**: `.gitignore`により、大きなデータファイル（`.pkl`, `.pt`）はGitに追跡されません。実験結果はローカルに保存され、必要に応じて手動でバックアップしてください。

## トラブルシューティング

### GPUメモリ不足

- バッチサイズを小さくする
- CPUモードで実行（`--device cpu`）
- モデルサイズを小さくする

### TransformerLensのモデルロードエラー

- モデル名を確認（`gpt2`, `EleutherAI/pythia-160m`など）
- インターネット接続を確認（初回ダウンロード時）
- HuggingFaceの認証が必要な場合（Llamaなど）は`huggingface-cli login`

### ファイルが見つからないエラー

- `results/`ディレクトリが存在することを確認
- 必要な親ディレクトリを作成（スクリプトが自動作成する場合もあります）

### Gitリポジトリの管理

**大きなファイルについて**:
- 実験結果ファイル（`.pkl`, `.pt`）は`.gitignore`によりGitに追跡されません
- これにより、リポジトリのサイズを小さく保ちます
- 実験結果はローカルに保存され、必要に応じて手動でバックアップしてください

**既にGitに追跡されている大きなファイルを削除する場合**:
```bash
git rm --cached results/**/*.pkl results/**/*.pt 2>/dev/null || true
git commit -m "Remove large data files from Git tracking"
```

詳細は[`SETUP.md`](SETUP.md)の「Gitリポジトリの管理」セクションを参照してください。

## プロジェクト統計

- **実装済みPythonモジュール**: 35ファイル以上
- **ドキュメント**: Phase 0-6の統合レポート（`docs/report/`配下）
- **データファイル**: Baseline（280サンプル）+ Extended（400サンプル）+ 実世界（35サンプル）
- **実装完了フェーズ**: Phase 0-6（統合実行完了）
- **MLflow統合**: 全フェーズで実験追跡対応

## プロジェクトフェーズ

詳細な実装計画は `docs/implementation_plan.md` を参照してください。

- ✅ **フェーズ0**: 環境構築
- ✅ **フェーズ1**: 感情プロンプトデータ作成
- ✅ **フェーズ2**: 内部活性の抽出
- ✅ **フェーズ3**: 感情方向ベクトルの抽出・可視化
- ✅ **フェーズ3.5**: 感情語トークンベースの再検証
- ✅ **フェーズ4ライト**: 簡易Activation Patching
- ✅ **フェーズ5**: 層×αスイープ実験と因果力比較
- ✅ **フェーズ6**: サブスペース構造とアライメント
- ✅ **フェーズ7**: Head/Unitレベル解析
- 🔄 **フェーズ8**: モデルサイズと普遍性（部分的に実装済み）

## 研究クエスチョン（RQ）

- **RQ1**: 感謝・怒り・謝罪などの感情は、層ごとに安定した方向ベクトルとして現れるか？
- **RQ2**: 異なるLLM間で、同じ感情方向はどれくらい似ているか？（サブスペースレベル）
- **RQ3**: 感情方向を操作（増幅・抑制）すると、出力トーンは変化するか？
- **RQ4**: 特定のattention headやMLP unitが感情を強く担っているか？
- **RQ5**: 異なるLLM間で、Neutral空間での線形写像により感情サブスペースはアライメントできるか？

## ライセンス

[ライセンス情報を追加]

## 参考文献

- TransformerLens: https://github.com/neelnanda-io/TransformerLens
- HuggingFace Transformers: https://huggingface.co/docs/transformers

## 貢献

[貢献ガイドラインを追加]

## 連絡先

[連絡先情報を追加]
