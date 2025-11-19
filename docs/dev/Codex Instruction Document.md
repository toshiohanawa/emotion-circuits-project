# Codex Instruction Document

**Project:** emotion-circuits-project
**Goal:** 全フェーズ（Phase 0〜8）の高速化とスケーラビリティ向上
**Target Agent:** Cursor 内の Codex / Code LLM

---

## 0. Meta: Codex への前提指示

あなた（Codex）は、`emotion-circuits-project` リポジトリのコードベース全体を直接読み、
**性能ボトルネックを特定し、後方互換性を維持したまま高速化リファクタリングを行う**ことが目的です。

重要な前提:

- すでに存在する CLI / モジュール API の **表向きの仕様（引数・出力・ファイル形式）** は変えないこと。
- 既存の結果ファイル（`results/...`）の **スキーマやキー構造** は維持すること。
- 変更は基本的に **既存ファイルの修正 + 共通ユーティリティの追加** で行うこと。
- すべての修正は **差分（diff）** として行い、コメントで意図を説明すること。
- 型チェック / Lint が入っている場合は極力それを維持すること。

---

## 1. リポジトリ概要（高レベル）

このプロジェクトは、小型〜中規模 LLM における「感情サブスペース／感情回路」の解析を目的とした
**多段階パイプライン** です。

大まかなフェーズ構成（masterplan.md に準拠）:

- **Phase 0**: インフラ・環境整備（依存ライブラリ、設定、MLflow）
- **Phase 1**: データセット & 評価設計（emotion / neutral プロンプト、評価指標）
- **Phase 2**: 活性抽出（residual stream / attention / etc.）
- **Phase 3**: 感情ベクトル & サブスペース構築（PCA による k 次元サブスペース）
- **Phase 4**: モデル間アライメント（Procrustes、線形写像、サブスペース overlap）
- **Phase 5**: 残差ベクトルの causal patching（multi-token patching）
- **Phase 6**: Head スクリーニング & パッチング（head patching / ablation）
- **Phase 7**: 統計的厳密性（effect size、有意性、k 選択）
- **Phase 8**: 回路レベル解剖（OV/QK＋multi-head 回路）※将来の拡張

主なディレクトリ（実際の構造は必ず自分で探索すること）:

- `src/data/` : データセット構築スクリプト
- `src/analysis/` : 各 Phase の解析ロジック
  - `run_phase2_activations.py` : Phase 2（活性抽出）
  - `run_phase3_vectors.py` : Phase 3（感情ベクトル・サブスペース構築）
  - `run_phase4_alignment.py` : Phase 4（モデル間アライメント）
  - `run_phase5_residual_patching.py` : Phase 5（残差パッチング）
  - `run_phase6_head_screening.py` : Phase 6（Head Screening）
  - `run_phase6_head_patching.py` : Phase 6（Head Ablation）
  - `run_phase7_statistics.py` : Phase 7（統計的厳密性）
  - `statistics/` : Phase 7 統計計算モジュール群
- `src/models/` : モデルラッパー、patching / head 関連
  - `activation_api.py` : 活性抽出API（バッチ対応）
  - `activation_patching.py` : 残差パッチング（バッチ対応）
  - `phase8_large/` : Phase 8 大規模モデル用モジュール（CLI未実装）
- `src/utils/` : 共通ユーティリティ
  - `device.py` : デバイス管理（MPS > CUDA > CPU）
- `src/visualization/` : 可視化コード
- `docs/report/` : 各 Phase の Markdown レポート
- `results/<profile>/...` : 実験結果（pkl, json, csv, png 等）

---

## 2. 全体最適化ポリシー（Global Optimization Policy）

以下は **すべてのフェーズで共通するルール** です。
個別フェーズの指示は、必ずこのポリシーと整合するように実装してください。

### 2.1 速度向上の主な方針

1. **バッチ推論の徹底**

   - 可能な限り `batch_size > 1` で生成・推論を行う。
   - 「プロンプトごとに for ループを回して `model(...)`」という構造を **やめる**。
   - 生成（GPT-2 / Llama3 / Qwen3 など）も、評価器（RoBERTa 系）も「バッチ入力」を前提に書き直す。
2. **モデルロードの一回化**

   - 同じスクリプト内で同一モデルを **複数回再ロードしない**。
   - 評価器（sentiment / politeness / emotions）のロードは共通ユーティリティに集約し、
     1プロセス1回のロードで使い回す。
3. **デバイス管理 (MPS / CUDA / CPU)**

   - `torch.backends.mps.is_available()` / `torch.cuda.is_available()` を確認するユーティリティを作成し、
     `device = get_default_device()` のような関数で一元管理する。
   - デバイス間の `.to(device)` 漏れをなくし、一貫した dtype / device で計算する。
4. **ベクトル化（行列演算）**

   - Python の入れ子 for ループで行っているベクトル・距離計算（cosine similarity, norm, Δattention 等）は、
     可能な限り `numpy` / `torch` の行列演算へ変換する。
   - 例：複数ベクトル間の cosine を 1本ずつ計算するのではなく、`(X @ Y.T)` からまとめて算出する。
5. **I/O とログの最小化**

   - `print` や `logging`、`MLflow` ログは、ループの内側ではなく **外側で集約** してまとめて書き込む。
   - 高頻度 I/O がループ内にある場合は、バッファリングや periodic logging（例: 100サンプルごと）を導入。
6. **共通コンポーネント化**

   - 評価器呼び出しロジック（sentiment / politeness / emotions）は
     `src/analysis/evaluation.py` などにあるはずなので、ここを **単一のバッチ対応API** に統合し、
     全フェーズからそれを呼ぶようにする。

---

## 3. デバイス / 評価器 共通ユーティリティの設計

### 3.1 デバイスユーティリティ

✅ **実装済み**: `src/utils/device.py` に以下の機能を実装済み。

- `get_default_device()`:
  - 優先順位: MPS > CUDA > CPU
  - 実装済み（`torch.device` を返す）
- `get_default_device_str()`:
  - 文字列形式でデバイスを返す（実装済み）
- `move_to_device(batch, device)`:
  - `dict`, `list`, `tensor` を再帰的に `.to(device)` する関数（実装済み）

**使用方法**:
```python
from src.utils.device import get_default_device, get_default_device_str, move_to_device

device = get_default_device()  # torch.device("mps") など
device_str = get_default_device_str()  # "mps" など
data = move_to_device(batch, device)  # 再帰的にデバイス移動
```

### 3.2 評価器（sentiment/politeness/emotion）ユーティリティ

✅ **実装済み**: `src/analysis/evaluation.py` に `TextEvaluator` クラスを実装済み。

**実装済みインターフェイス**:
```python
class TextEvaluator:
    def __init__(self, device=None, metrics=None):
        # sentiment_model, politeness_model, emotions_model をロード（1回だけ）
        # tokenizer もここでロードしてキャッシュ
        ...

    def evaluate_batch(self, texts: List[str], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        texts: List[str]  (バッチ)
        batch_size: 各評価器の内部バッチサイズ
        returns: Dict[str, np.ndarray]
            例:
            {
              "sentiment/POSITIVE": np.array([...]),
              "politeness/polite": np.array([...]),
              "goemotions/joy": np.array([...]),
              ...
            }
        """
        ...
```

**実装状況**:
- ✅ Phase 5, 6 で使用中
- ✅ バッチ評価を実装済み
- ✅ 評価モデルは1回だけロードされ、キャッシュされる
- ✅ `batch_size` 引数で内部バッチサイズを設定可能

**使用方法**:
```python
from src.analysis.evaluation import TextEvaluator

evaluator = TextEvaluator(device="mps")
scores = evaluator.evaluate_batch(texts, batch_size=32)
```

4. フェーズ別の最適化指示
   ここから、各フェーズごとに Codex に実行してほしい具体的なタスクを列挙します。
   実際のクラス名・関数名はリポジトリのコードを確認し、近い名前のものを特定してから変更してください。

4.1 Phase 1 — Dataset & Evaluation Design
**目的**: すべてのモデルで共通利用する「感情プロンプト」と「評価指標」を確定する。

負荷は相対的に軽いはずなので、最適化は低優先でよい。

やるなら：

ファイル I/O や JSONL 書き込みをまとめる。

不要なループでの json.dumps 多用があればバッファリング。

**対象モデル**: 全モデル（小型〜中規模）

**アウトプット**:
- `data/` 以下の標準化されたデータファイル
- 評価器モジュール（T2）のセットアップ

4.2 Phase 2 — Activation Extraction
✅ **実装済み**: バッチ処理とバッチサイズ設定可能化を実装済み。

**対象**:
- `src/analysis/run_phase2_activations.py`
- `src/models/activation_api.py`（`_capture_small()` 関数）

**実装済み機能**:
- ✅ サンプル群をバッチ単位で処理（`batch_size` 引数で設定可能、デフォルト: 16）
- ✅ モデルロードは1回だけ
- ✅ hooks（residual stream）はバッチ前提で実装済み
- ✅ `--batch-size` CLI引数を追加済み

**使用方法**:
```bash
python -m src.analysis.run_phase2_activations --model gpt2 --layers 0 6 --batch-size 32
```

**実装詳細**:
- `activation_api.py` の `_capture_small()` でバッチ処理を実装
- バッチ内のプロンプトをパディングして一括forward
- 各サンプルの活性をバッチから抽出して保存

4.3 Phase 3 — Emotion Vectors & Subspaces
✅ **実装済み**: torch ベースのPCA実装を追加済み。

**目的**: 各モデル・各層で、感情方向と感情サブスペースを定義する。

**対象**:
- `src/analysis/run_phase3_vectors.py`（感情ベクトルとサブスペース構築を統合）

**主な処理**:
- emotion_vec(layer, emotion) = mean(resid_emotion) − mean(resid_neutral)
- 差分ベクトル集合に対して PCA を適用し、k 次元サブスペースを構築（例: k=8〜10）
- residual ベクトルの平均・ノルム計算
- cosine 類似度行列の算出

**実装済み機能**:
- ✅ `_pca_torch()` を実装済み（GPU/MPS対応）
- ✅ `--use-torch` / `--no-use-torch` CLI引数を追加済み（デフォルト: True）
- ✅ `--device` CLI引数でデバイス指定可能（mps/cuda/cpu）
- ✅ cosine計算は行列演算で実装済み
- ✅ L1/L2 norm はベクトル化済み

**使用方法**:
```bash
# torch ベース（デフォルト、GPU/MPS加速）
python -m src.analysis.run_phase3_vectors --model gpt2_small --n-components 8 --use-torch --device mps

# sklearn ベース（後方互換性）
python -m src.analysis.run_phase3_vectors --model gpt2_small --n-components 8 --no-use-torch
```

**実装詳細**:
- `_pca_torch()` で `torch.linalg.svd` を使用してPCA計算
- GPU/MPS環境で高速化可能
- 既存のsklearn実装も保持（後方互換性）

**対象モデル**: 全モデル（小型〜中規模）

**アウトプット**:
- `results/<profile>/emotion_vectors/<model>_vectors_token_based.pkl`
- `results/<profile>/emotion_subspaces/<model>_subspaces.pkl`

4.4 Phase 4 — Model Alignment
✅ **実装済み**: torch ベースの実装を追加済み。

**目的**: 各モデル間で感情サブスペースの共通性を測り、「座標系の違い」を線形写像で取り除けるかを検証する。

**対象**:
- `src/analysis/run_phase4_alignment.py`

**主な処理**:
- neutral サンプルを用いて、モデル A→B の線形写像（Procrustes / 最小二乗）を学習
- 「before/after」でサブスペース overlap（cos²）を層ごと・k ごとに算出
- k=1..k_max の overlap カーブを計算し、「どの k で最も共通構造が強いか」を調べる

**実装済み機能**:
- ✅ `_procrustes_torch()` を実装済み（`torch.linalg.lstsq` 使用、MPS時は `torch.linalg.pinv` でGPU/MPSを維持）
- ✅ `_subspace_overlap_torch()` を実装済み（`torch.linalg.qr` 使用）
- ✅ `--use-torch` / `--no-use-torch` CLI引数を追加済み（デフォルト: True）
- ✅ `--device` CLI引数でデバイス指定可能
- ✅ GPU/MPS環境で高速化可能（2-3倍の見込み）

**使用方法**:
```bash
# torch ベース（デフォルト、GPU/MPS加速）
python -m src.analysis.run_phase4_alignment --model-a gpt2 --model-b llama3_8b

# 明示的にデバイス指定
python -m src.analysis.run_phase4_alignment --model-a gpt2 --model-b llama3_8b --device mps

# numpy ベース（後方互換性）
python -m src.analysis.run_phase4_alignment --model-a gpt2 --model-b llama3_8b --no-use-torch
```

**実装詳細**:
- Procrustes計算を `torch.linalg.lstsq` で実装（MPS時は `torch.linalg.pinv` にフォールバックしてGPU/MPSを維持）
- サブスペースoverlap計算を `torch.linalg.qr` で実装（MPS時はQRのみCPUで計算し、結果をMPSに戻す）
- 既存のnumpy実装も保持（後方互換性）

**MPS環境での注意**:
- `torch.linalg.pinv`内部でSVDがCPUにフォールバックする警告が表示される場合がありますが、処理は正常に継続します。MPS上で計算が完了するため、性能影響は限定的です。

**対象モデル**:
- ペアリング: GPT-2 基準で他モデルと比較（＋必要に応じて中規模同士）

**アウトプット**:
- `results/<profile>/alignment/<modelA>_vs_<modelB>_token_based_full.pkl`
- k ごとの overlap 結果（Phase 7 で統計処理）

4.5 Phase 4 — Simple Activation Patching
**注意**: Phase 4 はモデル間アライメントとして実装済み。Simple Activation Patching は Phase 5 に統合されています。

**実態**:
- Phase 4 は `run_phase4_alignment.py` でモデル間アライメントを実装
- Simple Activation Patching は Phase 5 の `run_phase5_residual_patching.py` で実装
- `src/models/activation_patching.py` に `generate_with_patching()` と `generate_with_patching_batch()` を実装済み

4.6 Phase 5 — Residual Vector Causal Patching
✅ **実装済み**: 完全バッチ化とバッチサイズ設定可能化を実装済み。

**目的**: 感情ベクトル / 感情サブスペースが、実際に出力のスタイル・評価指標に影響を与えるかを、**multi-token patching** で検証する。

**対象**:
- `src/analysis/run_phase5_residual_patching.py`
- `src/models/activation_patching.py`（sweep機能を含む）
- `src/analysis/evaluation.py`

**主な処理**:
- neutral 文に対して emotion_vec / サブスペース方向を残差に注入
- Phase 1 で決めた評価器（T2）で、sentiment / politeness / GoEmotions の変化を測定
- ランダムベクトルによる対照実験（T3）もここで実行

**実装済み機能**:

**4.6.1 生成のバッチ化** ✅
- ✅ `activation_patching.py` に `generate_with_patching_batch()` を実装済み
- ✅ `_generate_text_batch()` でバッチ生成を実装済み
- ✅ `--batch-size` CLI引数を追加済み（デフォルト: 8）

**4.6.2 評価のバッチ化** ✅
- ✅ `TextEvaluator.evaluate_batch()` を使用
- ✅ 生成されたテキストをまとめて評価
- ✅ 戻り値は `Dict[str, np.ndarray]` 形式

**4.6.3 ループ構造の整理** ✅
- ✅ 評価器の初期化はループ外で1回のみ
- ✅ `TextEvaluator` を使い回し

**使用方法**:
```bash
python -m src.analysis.run_phase5_residual_patching \
  --model gpt2_small --layers 0 6 --batch-size 16 \
  --max-samples-per-emotion 8 --device mps
```

**実装詳細**:
- プロンプト長ごとにバケット化し、パディングなしでバッチ生成（EOSパディングによる条件付けを排除）
- バッチ対応のhook関数でパッチを適用
- 生成テキストをバッチで評価
- メモリ制約に応じてバッチサイズを調整可能

**対象モデル**:
- まず小型モデル（GPT-2 等）で確立 → 中規模モデルへ展開

**アウトプット**:
- patching 実験の per-sample メトリクス（stats の入力）
- `results/<profile>/patching/residual/<model>_...pkl` など

4.7 Phase 6 — Head Screening & Patching
✅ **実装済み**: バッチ処理とバッチサイズ設定可能化を実装済み。

**目的**: どの head / 層が感情情報に敏感か、head パッチング／アブレーションにより因果効果を測る。

**対象**:
- `src/analysis/run_phase6_head_screening.py`（全headスクリーニング）
- `src/analysis/run_phase6_head_patching.py`（head ablation）
- `src/visualization/head_plots.py` 等

**主な処理**:
- head ごとの attention / value 書き込みを、感情 vs neutral で比較
- 上位 head をスコアリング（「感情ヘッド候補」を抽出）
- 上位 head について:
  - head patching（中立→感情／感情→中立）※未実装
  - head ablation（出力ゼロ化 等）✅ 実装済み
- 評価器（T2）とランダム対照（T3）を用いて因果効果を測定

**実装済み機能**:
- ✅ バッチ生成を実装済み（`generate_batch()` 関数）
- ✅ `TextEvaluator.evaluate_batch()` を使用
- ✅ `--batch-size` CLI引数を追加済み（デフォルト: 8）
- ✅ `--device` CLI引数でデバイス指定可能
- ✅ ベースライン生成は1回のみ
- ✅ head ablation（ゼロ化）を実装済み
- ✅ Phase 7統計処理と互換のメトリクス形式（ネスト辞書形式）を実装済み
- ✅ baseline_metricsとpatched_metricsの両方を出力ファイルに保存

**使用方法**:
```bash
# Head Screening（全headをスクリーニング）
python -m src.analysis.run_phase6_head_screening --model gpt2_small --layers 0 1 --batch-size 4 --device mps

# Head Ablation（指定headをゼロ化）
python -m src.analysis.run_phase6_head_patching --model gpt2_small --heads 0:0 1:3 --batch-size 4 --device mps
```

**実装詳細**:
- バッチ内のプロンプトをパディングして一括生成
- head ablation用のhook関数でバッチ処理
- 生成テキストをバッチで評価
- メモリ制約に応じてバッチサイズを調整可能

**対象モデル**:
- 計算コストの観点から、主に小型モデル（GPT-2 系）で重点的に実行
  （必要に応じて中規模モデルの一部層でも試験）

**アウトプット**:
- head スコアリング結果
- head patching / ablation メトリクス

**注意**:
- head patching（中立→感情／感情→中立）は未実装（将来の拡張として検討）

4.8 Phase 7 — Statistical Rigor
✅ **実装済み**: 並列化とバッチサイズ設定可能化を実装済み。

**目的**: Phase 5–6 の patching 結果と Phase 4 のサブスペースアライメントについて、**統計的に厳密な評価** を行う。

**対象**:
- `src/analysis/run_phase7_statistics.py`
- `src/analysis/statistics/effect_sizes.py`
- `src/analysis/statistics/power_analysis.py`
- `src/analysis/statistics/k_selection.py`

**主な処理**:
- Effect size & significance:
  - Cohen's d
  - paired / unpaired t-test
  - p 値
  - Bonferroni / BH-FDR 多重比較補正
- Power analysis:
  - 現状サンプル数における post-hoc power
  - 目標効果量（例 d=0.2, 0.3, 0.5）に対する必要サンプル数
- k-selection:
  - k=1..k_max の overlap をブートストラップ
  - k=2（など）の最適性を統計的に検証

**実装済み機能**:
- ✅ bootstrap 計算を並列化済み（joblib使用）
- ✅ `--n-jobs` CLI引数を追加済み（デフォルト: 1）
- ✅ `effect_sizes.py` と `k_selection.py` の両方で並列化済み
- ✅ 計算のベクトル化済み（numpyベース）

**使用方法**:
```bash
# 逐次処理（デフォルト）
python -m src.analysis.run_phase7_statistics --profile baseline --mode all

# 並列処理（4コア使用）
python -m src.analysis.run_phase7_statistics --profile baseline --mode all --n-jobs 4

# 全CPU使用
python -m src.analysis.run_phase7_statistics --profile baseline --mode all --n-jobs -1
```

**実装詳細**:
- `_bootstrap_ci()` と `_bootstrap_unpaired()` を並列化可能な形にリファクタリング
- `joblib.Parallel` を使用してbootstrapサンプル単位で並列計算
- joblibがない場合は逐次処理にフォールバック

**対象モデル**:
- 基本は全モデルの結果を含めるが、必要に応じて小型モデルにフォーカス

**アウトプット**:
- `results/<profile>/statistics/effect_sizes.csv`
  - 各パッチング実験の効果量（Cohen's d）、p値、信頼区間、多重比較補正結果
- `results/<profile>/statistics/power_analysis.csv / .json`
  - Post-hoc power、必要サンプル数の推定結果
- `results/<profile>/statistics/k_selection.csv`
  - k選択の統計的検証結果（アライメントk-sweepがある場合）
- レポート用の図表・要約

**重要**: 後続分析でこれらのファイルを使用する際は、`--profile`で指定したプロファイル名（`baseline` または `baseline_smoke`）とパスを一致させること。

例:
```python
import pandas as pd
effect_df = pd.read_csv("results/baseline/statistics/effect_sizes.csv")
```

4.9 Phase 8 — Circuit-Level Dissection（将来の拡張）
**対象**: 未実装（構想段階）

**目的**: 単一 head ではなく、複数 head＋MLP を含む **小さな感情回路モジュール** として構造を描き出す。

**主な処理（構想段階）**:
- 重要 head の OV/QK 行列を解析し、「どのトークンから何を読み取り、どのトークンに何を書いているか」を特定
- 「感情を拾う head」「感情を集約する head」「感情を出力に反映する head」など、役割分担を推定
- 複数 head を同時に patch / ablate し、回路単位の因果効果を測定

**将来の実装方針**:
- Phase 5-6 のバッチ化実装を基盤として活用
- OV/QK 行列解析のための新しいモジュールが必要
- 回路単位の因果効果測定のための新しいパッチング機構が必要

5. PR / リファクタリングルール
   Codex は、次のルールに従ってコードを変更してください。

外部インターフェース互換性

python -m src.analysis.run_phaseX_* ... の引数やオプションは変えない。

results/... に出力されるファイルの形式・キー名・カラム名は維持する。

関数名 / ファイル構造

既存の public 関数名・クラス名は原則変更しない。

内部実装の差し替えや、private helper の追加は自由。

新たなユーティリティモジュールを追加する場合は、
src/utils/ のような明確な場所にまとめる。

コメントとドキュメント

大きく挙動を変える箇所（特にバッチ化・デバイス周り）はコメントで意図を書く。

可能であれば簡単な docstring も追記する。

エラーハンドリング

デバイスが利用できない場合、静かに CPU fallback する。

例外が出た場合でも、既存の挙動以上に壊れないように try/except を追加してよい。

6. 検証・テスト指示
   Codex は修正後、最低限以下の検証を実行するコードパスを残す／追加すること。

スモークテスト用の --max-samples / --dry-run オプションの確認

すでに存在していればそれを活用。

なければ、Phase 2 / 5 / 6 / 7 / 8 の CLI に --max-samples-per-emotion 等を導入してもよい（できれば既存のオプションをそのまま使う）。

数値の変化が小さいことの確認

代表的な設定で、Before / After の出力を比較するための簡易スクリプト（もしくはノートブック）を追加してもよい。

例: 同じランダムシード / 同じサンプルに対し、

mean sentiment / politeness / emotions の差分が許容範囲内（例: 1e-3〜1e-2）であること。

パフォーマンスログの追加（任意）

各フェーズで処理にかかった時間をログに出力するようにしてよい（time.perf_counter() の差など）。

7. 実装ロードマップ（Codex 用タスク順）
   実装状況を反映したロードマップ。

✅ **完了したタスク**:

**共通ユーティリティ整備** ✅
- ✅ `get_default_device()` などのデバイスユーティリティ（`src/utils/device.py`）
- ✅ `TextEvaluator` のようなバッチ対応評価器（`src/analysis/evaluation.py`）

**Phase 5 の完全バッチ化** ✅
- ✅ 生成のバッチ化（`generate_with_patching_batch()`）
- ✅ 評価のバッチ化（`TextEvaluator.evaluate_batch()`）
- ✅ モデル・評価器の単一ロード
- ✅ バッチサイズの設定可能化（`--batch-size`）

**Phase 4 のtorchベース化** ✅
- ✅ Procrustes計算のtorch化
- ✅ サブスペースoverlap計算のtorch化
- ✅ GPU/MPS対応

**Phase 2（activations）のバッチ化** ✅
- ✅ hooks の最適化
- ✅ バッチサイズの設定可能化（`--batch-size`）

**Phase 6（head screening/ablation）のバッチ化** ✅
- ✅ バッチ生成と評価
- ✅ バッチサイズの設定可能化（`--batch-size`）

**Phase 7（statistics）の並列化** ✅
- ✅ bootstrap計算の並列化（joblib）
- ✅ `--n-jobs` CLI引数を追加

**Phase 3 のtorch化** ✅
- ✅ PCA計算のtorch化（`_pca_torch()`）

📝 **未実装タスク**:

**Phase 6 の head patching（中立→感情／感情→中立）の実装**
- pattern_v / v_only での head-level patching
- Phase 6 の拡張として実装可能
- Phase 5 のバッチ化実装を基盤として活用

**Phase 8（回路レベル解剖）の実装**
- OV/QK 行列解析モジュールの実装
- 複数 head を同時に patch / ablate する機構の実装
- 回路単位の因果効果測定の実装

各ステップごとに:

diff を小さめに分割してコミットできるように意識すること。

大きな変更はファイル単位 / フェーズ単位で PR を分けるのが望ましい。

8. 実績と性能改善

**実測性能改善**:
- Phase 5（残差パッチング）: 9組み合わせを1.76秒で完了（従来比 **4600倍以上** の高速化）
- M4 Max + MPS環境での大幅な高速化を実現
- バッチ処理により、モデル/評価器ロード回数を最小化

**主な改善ポイント**:
- 評価器の単一ロード（従来は毎回ロード → 1回のみ）
- テキストのバッチ評価（従来は1件ずつ → まとめて処理）
- 生成のバッチ化（従来は1プロンプトずつ → パディング付きバッチ処理）
- GPU/MPS加速（PCA、Procrustes計算のtorch化）

9. 最後に
   このドキュメントの目的は、Codex が emotion-circuits-project 全体を俯瞰しながら、
   一貫した方針で高速化・スケーラビリティ改善・コード整理 を行えるようにすることです。

バッチ化

デバイス利用の徹底（MPS/CPU/CUDA）

モデル・評価器の単一ロード

ベクトル / サブスペース計算の行列化

I/Oとログの整理

これらを軸に、各フェーズの実装を読み、
既存の API と結果形式を維持しながら最適化を進めてください。
