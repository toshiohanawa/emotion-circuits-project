# セットアップガイド

このドキュメントでは、プロジェクトを別のマシンでセットアップして実行するための詳細な手順を説明します。

## システム要件

- **OS**: Linux, macOS, Windows (WSL推奨)
- **Python**: 3.9以上（3.11推奨）
- **GPU**: CUDA対応GPU（推奨、CPUでも動作可能）
- **メモリ**:
  - 小型モデル（Phase 1-7）: 8GB以上（GPU使用時は16GB以上推奨）
  - 中規模モデル（Phase 8）: 32GB以上（GPU使用時は48GB以上推奨、または `device_map="auto"` による自動分割）
- **ディスク**: 10GB以上の空き容量（モデルダウンロード用、Phase 8では50GB以上推奨）

## セットアップ手順

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd emotion-circuits-project
```

### 2. uvのインストール

`uv`は高速なPythonパッケージマネージャーです。インストール方法：

```bash
# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# または pip経由
pip install uv
```

### 3. 仮想環境の作成と有効化

```bash
# 仮想環境を作成
uv venv

# 有効化
source .venv/bin/activate  # Linux/Mac
# または
.venv\Scripts\activate  # Windows
```

### 4. 依存関係のインストール

```bash
# プロジェクトをインストール（依存関係も自動インストール）
uv pip install -e .

# または、通常のpipを使用する場合
pip install -e .
```

### 5. Jupyterカーネルの設定（オプション）

```bash
python -m ipykernel install --user --name emotion-circuits --display-name "Python (emotion-circuits)"
```

### 6. 動作確認

```bash
# モジュールが正しくインポートできるか確認
python -c "from src.models.activation_api import get_activations; print('OK')"

# ヘルプを表示してCLIが動作するか確認
python -m src.analysis.run_phase2_activations --help
```

## データの準備

### プロンプトデータ

プロジェクトには以下のプロンプトデータが含まれています：

- `data/neutral_prompts.json`: 中立プロンプト70文 ✅
- `data/gratitude_prompts.json`: 感謝プロンプト70文 ✅
- `data/anger_prompts.json`: 怒りプロンプト70文 ✅
- `data/apology_prompts.json`: 謝罪プロンプト70文 ✅
- `data/real_world_samples.json`: 実世界テキストサンプル35文 ✅

プロンプトデータは、`src/data/create_individual_prompt_files.py`を使用して作成できます。データセットは`src/data/build_dataset.py`を使用してJSONL形式に変換できます。

> ⚠️ **注意**: 現行パイプラインでは `baseline` プロファイル（約225サンプル/感情）を標準とし、`baseline_smoke` を配線確認用として使用します。旧 `extended` プロファイルは廃止されました。

### 実験結果データ

`results/`ディレクトリには実験結果が保存されます。初回実行時は空です。

**ディレクトリ構造**:
- `results/baseline/`: 標準データセットの結果（約225サンプル/感情）
- `results/baseline_smoke/`: 配線確認用の最小データセット結果（各感情3-5件）

**重要な注意事項**:
- 実験結果ファイル（`.pkl`, `.pt`など）は非常に大きくなる可能性があります（数MB〜数百MB）
- `.gitignore`により、これらの大きなファイルはGitに追跡されません
- 実験結果はローカルに保存され、必要に応じて手動でバックアップしてください
- `results/{profile}/plots/`配下の可視化結果（`.png`など）はGitに追跡されます

### MLflow統合

プロジェクトはMLflowを使用して実験追跡を行います：

- **Tracking URI**: `http://localhost:5001`（デフォルト）
- **実験名**: リポジトリ名から自動設定（`auto_experiment_from_repo()`）
- **ログ内容**: パラメータ、メトリクス、アーティファクト

MLflowサーバーが起動していることを確認してください。詳細は`src/utils/mlflow_utils.py`を参照してください。

## 実行例

### データセットの作成

```bash
# 1. 個別プロンプトファイルの作成
python -m src.data.create_individual_prompt_files --data_dir data

# 2. Baselineデータセットの構築（標準プロファイル、約225サンプル/感情）
python -m src.data.build_dataset --profile baseline

# 3. Smoke testデータセットの構築（配線確認用、各感情3-5件）
python -m src.data.build_dataset --profile baseline_smoke

# 4. データセットの検証
python -m src.data.validate_dataset data/emotion_dataset.jsonl
```

> ℹ️ **プロファイルベースのワークフロー**: プロジェクトでは、データセットプロファイル（`baseline`/`baseline_smoke`）を使用して、プロンプトファイル、データセット、結果ディレクトリを自動的に解決します。多くのスクリプトは `--profile` 引数を受け取り、適切なパスを自動的に設定します。`baseline` は統計的に有意な分析用、`baseline_smoke` は配線確認用です。

### 最小限の実行例（GPT-2 smallのみ）

```bash
# 1. 内部活性を抽出（標準パイプライン Phase 2）
python -m src.analysis.run_phase2_activations \
  --profile baseline \
  --model gpt2_small \
  --layers 0 3 6 9 11 \
  --device mps \
  --batch-size 16

# 2. 感情方向ベクトルを抽出（標準パイプライン Phase 3）
python -m src.analysis.run_phase3_vectors \
  --profile baseline \
  --model gpt2_small \
  --n-components 8 \
  --use-torch \
  --device mps

# 3. Activation Patching実験（標準パイプライン Phase 5）
python -m src.analysis.run_phase5_residual_patching \
  --profile baseline \
  --model gpt2_small \
  --layers 0 3 6 9 11 \
  --alpha 1.0 \
  --max-samples-per-emotion 8 \
  --device mps \
  --batch-size 16
```

### 標準パイプラインでの実行例（Phase 2-7）

```bash
# Phase 2: 活性抽出（バッチ処理対応）
python -m src.analysis.run_phase2_activations \
  --profile baseline \
  --model gpt2_small \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 \
  --device mps \
  --batch-size 32

# Phase 3: 感情ベクトル構築（torch/sklearn切り替え可能）
python -m src.analysis.run_phase3_vectors \
  --profile baseline \
  --model gpt2_small \
  --n-components 8 \
  --use-torch \
  --device mps

# Phase 4: 自己アライメント（torch/numpy切り替え可能、CUDA推奨）
python -m src.analysis.run_phase4_alignment \
  --profile baseline \
  --model-a gpt2_small \
  --model-b gpt2_small \
  --k-max 8 \
  --use-torch \
  --device cuda

# Phase 5: 残差パッチング（バッチ処理対応）
## 小型（HookedTransformer）
python -m src.analysis.run_phase5_residual_patching \
  --profile baseline \
  --model gpt2_small \
  --layers 0 3 6 9 11 \
  --alpha 1.0 \
  --max-samples-per-emotion 8 \
  --device mps \
  --batch-size 16

## 大型（LargeHFModel, 例: llama3_8b）CUDA推奨
python -m src.analysis.run_phase5_residual_patching \
  --profile baseline \
  --model llama3_8b \
  --layers 0 3 6 9 11 \
  --alpha 1.0 \
  --max-samples-per-emotion 50 \
  --device cuda \
  --batch-size 8

# Phase 6: ヘッドスクリーニング（バッチ処理対応）
## 小型（HookedTransformer）
python -m src.analysis.run_phase6_head_screening \
  --profile baseline \
  --model gpt2_small \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 \
  --max-samples 8 \
  --device mps \
  --batch-size 8

## 大型（LargeHFModel, 例: llama3_8b）※CUDA推奨（MPSは非常に遅い）
python -m src.analysis.run_phase6_head_screening \
  --profile baseline \
  --model llama3_8b \
  --layers 0 3 6 9 11 \
  --max-samples 50 \
  --device cuda \
  --batch-size 8

# Phase 7: 統計分析（並列処理対応）
python -m src.analysis.run_phase7_statistics \
  --profile baseline \
  --mode effect \
  --n-jobs 4
```

### Smoke testでの配線確認

```bash
# 最小限のデータで配線確認（各感情3-5件）
python -m src.analysis.run_phase2_activations \
  --profile baseline_smoke \
  --model gpt2_small \
  --layers 0 5 11
```

## トラブルシューティング

### 問題1: `ModuleNotFoundError`

**原因**: 依存関係がインストールされていない

**解決策**:
```bash
uv pip install -e .
# または
pip install -e .
```

### 問題2: CUDA/GPU/MPS関連のエラー

**原因**: CUDAがインストールされていない、またはPyTorchがCPU版

**解決策**:
- CPUモードで実行: `--device cpu`を追加
- Apple Silicon（M1/M2/M3/M4）の場合: `--device mps`を使用
- CUDA版PyTorchをインストール: https://pytorch.org/get-started/locally/
- デバイスの自動選択: `--device`を省略すると自動的に最適なデバイスを選択

### 問題3: TransformerLensのモデルロードエラー

**原因**: モデル名が間違っている、またはインターネット接続の問題

**解決策**:
- モデル名を確認: `gpt2`, `EleutherAI/pythia-160m`, `EleutherAI/gpt-neo-125M`
- インターネット接続を確認
- HuggingFaceの認証が必要な場合: `huggingface-cli login`

### 問題4: メモリ不足エラー

**原因**: GPU/CPUメモリが不足

**解決策**:
- バッチサイズを小さくする（`--batch-size 4` などCLI引数で調整可能）
- CPUモードで実行（`--device cpu`）
- より小さなモデルを使用
- サンプル数を減らす（`--max-samples-per-emotion 4` など）

### 問題5: ファイルが見つからないエラー

**原因**: 必要なディレクトリが存在しない

**解決策**:
```bash
# 必要なディレクトリを作成（プロファイルごとに用意）
mkdir -p results/{baseline,baseline_smoke}/{activations,emotion_vectors,subspaces,patching,alignment,evaluation,plots,statistics}
```

## Gitリポジトリの管理

### `.gitignore`について

プロジェクトの`.gitignore`は以下のファイル/ディレクトリを除外します：

- **大きなデータファイル**: `results/**/*.pkl`, `results/**/*.pt`（実験結果）
- **モデルファイル**: `/models/`（ダウンロードしたモデル）
- **仮想環境**: `.venv/`, `venv/`
- **キャッシュ**: `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`
- **ML/AIフレームワーク**: `wandb/`, `mlruns/`, `*.pt`, `*.onnx`
- **IDE設定**: `.vscode/`, `.idea/`, `.cursor/`

**追跡されるもの**:
- ソースコード（`src/`配下）
- ドキュメント（`docs/`配下）
- 設定ファイル（`pyproject.toml`, `requirements.txt`など）
- プロンプトデータ（`data/`配下のJSONファイル）
- 可視化結果（`results/{profile}/plots/`配下の`.png`など）

### 既にGitに追跡されている大きなファイルの削除

もし過去に大きなファイルがGitに追加されてしまった場合、以下のコマンドでGitの追跡から削除できます（ファイル自体は残ります）：

```bash
# すべての大きなファイルをGitの追跡から削除
git rm --cached results/**/*.pkl results/**/*.pt 2>/dev/null || true

# 変更をコミット
git commit -m "Remove large data files from Git tracking"
```

## 環境変数（オプション）

必要に応じて環境変数を設定：

```bash
# HuggingFaceキャッシュディレクトリ
export HF_HOME=/path/to/huggingface_cache

# CUDAデバイス指定
export CUDA_VISIBLE_DEVICES=0

# トランスフォーマーのキャッシュ
export TRANSFORMERS_CACHE=/path/to/transformers_cache

# HuggingFaceトークン（Phase 8 の中規模モデル用）
export HF_TOKEN=your_token_here
```

## Phase 8: 中規模モデルのセットアップ

> ⚠️ **注意**: Phase 8の標準パイプラインスクリプト（`run_phase8_*.py`）は現在開発中です。以下は参考情報として記載しています。中規模モデルの活性抽出には`src/models/activation_api.py`の`get_activations()`が対応しています。

Phase 8 では、Llama3 8B / Gemma3 12B / Qwen3 8B などの中規模モデルを使用します。これらのモデルにアクセスするには、追加の設定が必要です。

### HuggingFace トークンの設定

1. HuggingFace アカウントを作成: https://huggingface.co/join
2. アクセストークンを生成: https://huggingface.co/settings/tokens
3. トークンを設定:

```bash
# 方法1: 環境変数として設定
export HF_TOKEN=your_token_here

# 方法2: huggingface-cli を使用
huggingface-cli login

# 方法3: Pythonから直接設定
from huggingface_hub import login
login(token="your_token_here")
```

### ゲートモデルへのアクセス申請

一部のモデル（特に Llama3）は、HuggingFace でアクセス申請が必要です：

1. **Llama3 8B** (`meta-llama/Meta-Llama-3.1-8B`):
   - https://huggingface.co/meta-llama/Meta-Llama-3.1-8B にアクセス
   - "Request access" をクリックして利用規約に同意
   - 承認後（通常数分〜数時間）、モデルをダウンロード可能

2. **Gemma3 12B** (`google/gemma-3-12b-it`):
   - https://huggingface.co/google/gemma-3-12b-it にアクセス
   - Google の利用規約に同意

3. **Qwen3 8B** (`Qwen/Qwen3-8B-Base`):
   - 通常はアクセス制限なし（公開モデル）

### Phase 8 パイプラインの実行

```bash
# 単一モデルでの実行
python3 -m src.analysis.run_phase8_pipeline \
  --target llama3_8b \
  --profile baseline \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 \
  --k 8 \
  --n-samples 50

# デバイス指定（GPU メモリが不足する場合）
python3 -m src.analysis.run_phase8_pipeline \
  --target llama3_8b \
  --profile baseline \
  --device auto  # device_map="auto" で自動分割

# CPU で実行（非推奨、非常に遅い）
python3 -m src.analysis.run_phase8_pipeline \
  --target qwen3_8b \
  --profile baseline \
  --device cpu
```

### トラブルシューティング（Phase 8）

#### 問題: `OSError: You need to agree to the terms of use`

**原因**: ゲートモデルへのアクセス申請が未完了

**解決策**:
- HuggingFace の該当モデルページでアクセス申請を行う
- 申請が承認されるまで待つ（通常数分〜数時間）
- `huggingface-cli whoami` で正しいアカウントにログインしていることを確認

#### 問題: `OutOfMemoryError`

**原因**: GPU/CPU メモリ不足

**解決策**:
```bash
# device_map="auto" を使用して自動分割
python3 -m src.analysis.run_phase8_pipeline --device auto

# または、より小さなモデルを使用
python3 -m src.analysis.run_phase8_pipeline --target qwen3_8b

# サンプル数を減らす
python3 -m src.analysis.run_phase8_pipeline --n-samples 20
```

#### 問題: 数値的不安定性（特に Gemma3）

**原因**: モデルアーキテクチャの特性

**解決策**:
- 他のモデル（Llama3、Qwen3）を優先的に使用
- PCA 次元 `--k` を小さくする（例: `--k 4`）
- 結果の解釈時にこの制限を考慮する

## 次のステップ

1. `docs/implementation_plan.md`を読んでプロジェクトの全体像を理解
2. `masterplan.md`で各Phaseの目標と成果物を確認
3. `baseline_smoke`プロファイルで配線確認（各感情3-5件、数分で完了）
4. `baseline`プロファイルで本格的な分析（1感情225件前後を想定）
5. Phase 7の統計分析で結果の厳密性を検証
6. `docs/report_template/`のテンプレートを参考にレポートを作成
7. MLflowで実験結果を追跡・比較

## サポート

問題が発生した場合は、以下を確認してください：

1. Pythonバージョン: `python --version`（3.9以上）
2. 依存関係のインストール状況: `pip list`
3. GPUの利用可能性: `python -c "import torch; print(torch.cuda.is_available())"`
4. ディスク容量: `df -h`（Linux/Mac）または`dir`（Windows）
