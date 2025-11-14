# セットアップガイド

このドキュメントでは、プロジェクトを別のマシンでセットアップして実行するための詳細な手順を説明します。

## システム要件

- **OS**: Linux, macOS, Windows (WSL推奨)
- **Python**: 3.9以上（3.11推奨）
- **GPU**: CUDA対応GPU（推奨、CPUでも動作可能）
- **メモリ**: 8GB以上（GPU使用時は16GB以上推奨）
- **ディスク**: 10GB以上の空き容量（モデルダウンロード用）

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
python -c "from src.models.extract_activations import ActivationExtractor; print('OK')"

# ヘルプを表示してCLIが動作するか確認
python -m src.models.extract_activations --help
```

## データの準備

### プロンプトデータ

プロジェクトには以下のプロンプトデータが含まれています：

- `data/neutral_prompts.json`: 中立プロンプト50文 ✅
- `data/gratitude_prompts.json`: 感謝プロンプト10文 ✅
- `data/anger_prompts.json`: 怒りプロンプト（作成が必要）
- `data/apology_prompts.json`: 謝罪プロンプト（作成が必要）

不足しているプロンプトデータは、`src/data/create_emotion_dataset.py`を使用して作成できます。

### 実験結果データ

`results/`ディレクトリには実験結果が保存されます。初回実行時は空です。

**重要な注意事項**:
- 実験結果ファイル（`.pkl`, `.pt`など）は非常に大きくなる可能性があります（数MB〜数十MB）
- `.gitignore`により、これらの大きなファイルはGitに追跡されません
- 実験結果はローカルに保存され、必要に応じて手動でバックアップしてください
- `results/plots/`配下の可視化結果（`.png`など）はGitに追跡されます

## 実行例

### 最小限の実行例（GPT-2 smallのみ）

```bash
# 1. 内部活性を抽出
python -m src.models.extract_activations \
  --model gpt2 \
  --dataset data/emotion_dataset.jsonl \
  --output results/activations/gpt2/

# 2. 感情方向ベクトルを抽出
python -m src.analysis.emotion_vectors_token_based \
  --activations_dir results/activations/gpt2 \
  --output results/emotion_vectors/gpt2_vectors_token_based.pkl

# 3. Activation Patching実験
python -m src.models.activation_patching_sweep \
  --model gpt2 \
  --vectors_file results/emotion_vectors/gpt2_vectors_token_based.pkl \
  --prompts_file data/neutral_prompts.json \
  --output results/patching/gpt2_sweep.pkl \
  --layers 3 5 7 9 11 \
  --alpha -2 -1 -0.5 0 0.5 1 2
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

### 問題2: CUDA/GPU関連のエラー

**原因**: CUDAがインストールされていない、またはPyTorchがCPU版

**解決策**:
- CPUモードで実行: `--device cpu`を追加
- CUDA版PyTorchをインストール: https://pytorch.org/get-started/locally/

### 問題3: TransformerLensのモデルロードエラー

**原因**: モデル名が間違っている、またはインターネット接続の問題

**解決策**:
- モデル名を確認: `gpt2`, `EleutherAI/pythia-160m`, `EleutherAI/gpt-neo-125M`
- インターネット接続を確認
- HuggingFaceの認証が必要な場合: `huggingface-cli login`

### 問題4: メモリ不足エラー

**原因**: GPU/CPUメモリが不足

**解決策**:
- バッチサイズを小さくする（コード内で調整）
- CPUモードで実行
- より小さなモデルを使用

### 問題5: ファイルが見つからないエラー

**原因**: 必要なディレクトリが存在しない

**解決策**:
```bash
# 必要なディレクトリを作成
mkdir -p results/{activations,emotion_vectors,subspaces,patching,alignment,evaluation,plots}
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
- 可視化結果（`results/plots/`配下の`.png`など）

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
```

## 次のステップ

1. `docs/implementation_plan.md`を読んでプロジェクトの全体像を理解
2. `docs/phase6_expansion_report.md`でPhase 6の結果を確認
3. 小さな実験から始める（GPT-2 smallのみ）
4. 結果を確認してから、より大きな実験に進む

## サポート

問題が発生した場合は、以下を確認してください：

1. Pythonバージョン: `python --version`（3.9以上）
2. 依存関係のインストール状況: `pip list`
3. GPUの利用可能性: `python -c "import torch; print(torch.cuda.is_available())"`
4. ディスク容量: `df -h`（Linux/Mac）または`dir`（Windows）

