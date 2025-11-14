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
- `src/data/validate_dataset.py` - データセット検証

#### モデル操作
- `src/models/extract_activations.py` - 内部活性抽出
- `src/models/activation_patching.py` - 簡易Activation Patching
- `src/models/activation_patching_sweep.py` - 層×αスイープ実験
- `src/models/activation_patching_iterative.py` - Iterative Patching
- `src/models/activation_patching_swap.py` - Swap Patching
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
│   ├── neutral_prompts.json   # 中立プロンプト（50文）
│   ├── gratitude_prompts.json # 感謝プロンプト
│   ├── anger_prompts.json     # 怒りプロンプト（作成が必要）
│   └── apology_prompts.json   # 謝罪プロンプト（作成が必要）
├── docs/                      # ドキュメント
│   ├── proposal/              # 企画書
│   ├── implementation_plan.md # 実装計画（詳細）
│   ├── phase2_report.md       # Phase 2レポート
│   ├── phase3_report.md       # Phase 3レポート
│   ├── phase3.5_and_4light_report.md  # Phase 3.5/4 Lightレポート
│   ├── phase5_report.md       # Phase 5レポート
│   └── phase6_expansion_report.md     # Phase 6レポート
├── notebooks/                 # Jupyterノートブック
├── results/                   # 実験結果
│   ├── activations/           # 抽出した活性データ
│   │   ├── gpt2/
│   │   ├── pythia-160m/
│   │   └── gpt-neo-125m/
│   ├── emotion_vectors/       # 感情方向ベクトル
│   ├── subspaces/             # サブスペースデータ
│   ├── patching/              # Patching実験結果
│   ├── alignment/             # アライメント実験結果
│   │   ├── linear_maps/       # 線形写像
│   │   └── *.json, *.pkl     # 実験結果
│   ├── evaluation/            # 評価結果
│   └── plots/                 # 可視化結果
│       ├── cross_model/
│       ├── k_sweep/
│       ├── alignment/
│       └── heads/
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

感情プロンプトデータセットを作成（既に`data/`に一部存在）：

```bash
python -m src.data.create_emotion_dataset --output data/emotion_dataset.jsonl
```

### 2. 内部活性の抽出

モデルから内部活性を抽出：

```bash
# GPT-2 small
python -m src.models.extract_activations \
  --model gpt2 \
  --dataset data/emotion_dataset.jsonl \
  --output results/activations/gpt2/

# Pythia-160M
python -m src.models.extract_activations \
  --model EleutherAI/pythia-160m \
  --dataset data/emotion_dataset.jsonl \
  --output results/activations/pythia-160m/
```

### 3. 感情方向ベクトルの抽出

```bash
# 文末ベース
python -m src.analysis.emotion_vectors \
  --activations_dir results/activations/gpt2 \
  --output results/emotion_vectors/gpt2_vectors.pkl

# トークンベース
python -m src.analysis.emotion_vectors_token_based \
  --activations_dir results/activations/gpt2 \
  --output results/emotion_vectors/gpt2_vectors_token_based.pkl
```

### 4. サブスペース解析

```bash
# 感情サブスペースの計算
python -m src.analysis.emotion_subspace \
  --activations_dir results/activations/gpt2 \
  --output results/subspaces/gpt2_subspaces.pkl \
  --n-components 10

# k-sweep実験
python -m src.analysis.subspace_k_sweep \
  --activations_dir results/activations \
  --model1 gpt2 \
  --model2 pythia-160m \
  --output results/alignment/k_sweep_gpt2_pythia.json \
  --k-values 2 5 10 20 \
  --layers 3 5 7 9 11
```

### 5. モデル間アライメント

```bash
# Neutral空間での線形写像学習
python -m src.analysis.model_alignment \
  --model1 gpt2 \
  --model2 EleutherAI/pythia-160m \
  --neutral_prompts_file data/neutral_prompts.json \
  --model1_activations_dir results/activations/gpt2 \
  --model2_activations_dir results/activations/pythia-160m \
  --output results/alignment/model_alignment_gpt2_pythia.pkl \
  --n-components 10 \
  --layers 3 5 7 9 11
```

### 6. Activation Patching実験

```bash
# 層×αスイープ実験
python -m src.models.activation_patching_sweep \
  --model gpt2 \
  --vectors_file results/emotion_vectors/gpt2_vectors_token_based.pkl \
  --prompts_file data/neutral_prompts.json \
  --output results/patching/gpt2_sweep.pkl \
  --layers 3 5 7 9 11 \
  --alpha -2 -1 -0.5 0 0.5 1 2
```

### 7. Head解析

```bash
# Headスクリーニング
python -m src.analysis.head_screening \
  --model gpt2 \
  --output results/alignment/head_scores_gpt2.json

# Head ablation
python -m src.models.head_ablation \
  --model gpt2 \
  --head-spec "3:5,7:2" \
  --prompts-file data/gratitude_prompts.json \
  --output results/patching/head_ablation_gpt2_gratitude.pkl

# Head patching
python -m src.models.head_patching \
  --model gpt2 \
  --head-spec "3:5,7:2" \
  --neutral-prompts data/neutral_prompts.json \
  --emotion-prompts data/gratitude_prompts.json \
  --output results/patching/head_patching_gpt2_gratitude.pkl
```

### 8. 可視化

```bash
# Head解析結果の可視化
python -m src.visualization.head_plots \
  --head-scores results/alignment/head_scores_gpt2.json \
  --ablation-file results/patching/head_ablation_gpt2_gratitude.pkl \
  --patching-file results/patching/head_patching_gpt2_gratitude.pkl \
  --output-dir results/plots/heads

# アライメント結果の可視化
python -m src.visualization.alignment_plots \
  --results_file results/alignment/model_alignment_gpt2_pythia.pkl \
  --output_dir results/plots/alignment
```

## 主要な発見

### Phase 6の核心的な発見

1. **低次元での共有構造**: k-sweep実験により、k=2でoverlapが最も高く（0.002〜0.005）、コアな共有因子が低次元に存在することを示唆

2. **線形写像による大幅な改善**: Neutral空間で学習した線形写像により、感情サブスペースのoverlapが**0.53〜0.99**まで大幅に改善。これは「座標系は違うが本質的には同じ構造」という仮説を強く支持

3. **層依存性**: 深い層（9, 11）で特にアライメント効果が大きい。Layer 9では、AngerとApologyでoverlapが1.0（完全一致）に到達

詳細は`docs/phase6_expansion_report.md`を参照してください。

## データファイル

### プロンプトデータ

- `data/neutral_prompts.json`: 中立プロンプト50文
- `data/gratitude_prompts.json`: 感謝プロンプト（10文）
- `data/anger_prompts.json`: 怒りプロンプト（作成が必要）
- `data/apology_prompts.json`: 謝罪プロンプト（作成が必要）

### 結果データ

実験結果は`results/`以下に保存されます：

- **活性データ**: `results/activations/{model_name}/activations_{emotion}.pkl`
- **感情ベクトル**: `results/emotion_vectors/{model}_vectors.pkl`
- **サブスペース**: `results/subspaces/{model}_subspaces.pkl`
- **アライメント結果**: `results/alignment/*.pkl`, `results/alignment/*.json`
- **Patching結果**: `results/patching/*.pkl`
- **可視化**: `results/plots/**/*.png`

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

## プロジェクト統計

- **実装済みPythonモジュール**: 31ファイル
- **ドキュメント**: 8レポート（Phase 2-6の詳細レポート含む）
- **データファイル**: 2プロンプトファイル（neutral, gratitude）
- **実装完了フェーズ**: Phase 0-7（Phase 8は部分的）

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
