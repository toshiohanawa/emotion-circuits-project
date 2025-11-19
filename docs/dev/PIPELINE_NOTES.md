# パイプラインノート（現行標準）

## 現行パイプラインの位置づけ
- このドキュメントに記載された構成が唯一の正式パイプライン。旧Phase0-8のスクリプト/結果は削除済みで、必要ならGit履歴から取得する。
- コア構成要素:
  - プロファイル/データ: `src/config/project_profiles.py`, `src/data/build_dataset.py`
  - モデル/活性API: `src/models/model_registry.py`, `src/models/activation_api.py`
  - 評価スタック: `src/analysis/evaluation.py`（公開HFモデル3種: sentiment/politeness/goemotions）
  - Phase2–6 CLIs: `run_phase2_activations.py`, `run_phase3_vectors.py`, `run_phase4_alignment.py`, `run_phase5_residual_patching.py`, `run_phase6_head_patching.py`, `run_phase6_head_screening.py`
  - 統計: `run_phase7_statistics.py` + `src/analysis/statistics/*`（phase=head_screeningもサポート）
-> baseline / baseline_smoke の2プロファイルをこの構成で扱う。

このリポジトリはフェーズ1〜7を統一的に再実行するための構成に更新済みです。旧パイロット結果は破棄を前提とし、以下のCLIでクリーンにリランしてください。

## 主なCLI

- フェーズ1（データセット変換）  
  `python -m src.data.build_dataset --profile baseline --input data/emotions_baseline.csv`
  - スモーク用プロファイル: `python -m src.data.build_dataset --profile baseline_smoke --input data/emotion_dataset_smoke.jsonl`

- フェーズ2（活性抽出・全モデル共通）  
  `python -m src.analysis.run_phase2_activations --profile baseline --model gpt2 --layers 0 1 2 --max-samples-per-emotion 8`
  - スモーク例: `python -m src.analysis.run_phase2_activations --profile baseline_smoke --model gpt2_small --layers 0 6 --device mps --max-samples-per-emotion 3`

- フェーズ3（感情ベクトル & サブスペース）  
  `python -m src.analysis.run_phase3_vectors --profile baseline --model gpt2 --n-components 8`
  - スモーク例: `python -m src.analysis.run_phase3_vectors --profile baseline_smoke --model gpt2_small --n-components 3`

- フェーズ4（モデル間アライメント）  
  `python -m src.analysis.run_phase4_alignment --profile baseline --model-a gpt2 --model-b llama3_8b --k-max 8`
  - スモーク例（自己アライメント）: `python -m src.analysis.run_phase4_alignment --profile baseline_smoke --model-a gpt2_small --model-b gpt2_small --k-max 3`

- フェーズ5（multi-token残差パッチング）  
  `python -m src.analysis.run_phase5_residual_patching --profile baseline_smoke --model gpt2_small --layers 0 6 --patch-window 3 --sequence-length 20 --alpha 0.8 --max-samples-per-emotion 3 --device mps --random-control --num-random 2`

- フェーズ6（簡易headアブレーション）  
  `python -m src.analysis.run_phase6_head_patching --profile baseline_smoke --model gpt2_small --heads 0:0 --max-samples 3 --sequence-length 20 --device mps`

- フェーズ6（ヘッドスクリーニング）  
  `python -m src.analysis.run_phase6_head_screening --profile baseline_smoke --model gpt2_small --layers 0 1 --max-samples 3 --sequence-length 20 --device mps`

- 評価器（全フェーズ共通）  
  `from src.analysis.evaluation import evaluate_text_batch`

## モデルレジストリ

- 小型: gpt2 / pythia-160m / gpt-neo-125m（TransformerLens）
  - gpt2_small は gpt2 のエイリアス（スモーク用コマンドを簡潔化）
- 中規模: llama3_8b / gemma3_12b / qwen3_8b（HF LargeHFModelラッパ）

## 評価モデル (公開HF)
- sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- politeness: `NOVA-vision-language/polite_bert`
- goemotions: `SamLowe/roberta-base-go_emotions`

## ベースラインデータセット
- 入力: `data/emotion_dataset.jsonl`（感情4種 × 各225サンプル前後を目標。旧版の70件は廃止）
- プロファイル: baseline（出力は `results/baseline/` 配下）
- 生成例: `python -m src.data.build_dataset --profile baseline --input data/emotion_dataset.jsonl`

## baseline プロファイル (gpt2_small, 225サンプル/感情) フル実験レシピ

本セクションは、GPT-2 small（124M）を用いたbaselineプロファイルの完全な実験手順を記載する。
各Phaseのコマンド、出力先、次のPhaseへの依存関係を明記する。

**前提条件:**
- プロファイル: `baseline`（感情4種 × 各225サンプル前後を想定）
- モデル: `gpt2_small`（GPT-2 small, 124M, 12層）
- デバイス: `mps`（デフォルト、M4 MacなどMetal対応環境で推奨。動作しない場合は`cpu`にフォールバック）
- 結果保存先: `results/baseline/...`

**実行順序と依存関係:**
1. Dataset build → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6-a → Phase 6-b → Phase 7

### 1) Dataset build（baseline, 225サンプル/感情）

```bash
python3 -m src.data.build_dataset \
  --profile baseline \
  --input data/emotion_dataset.jsonl
```

- **目的**: 入力JSONLを標準化されたデータセットに変換
- **出力**: `data/emotion_dataset.jsonl`（プロファイル既定パス、既に存在する場合はスキップ可能）
- **次のPhase**: Phase 2で使用

### 2) Phase 2: activations（全層0-11、225サンプル/感情）

```bash
python3 -m src.analysis.run_phase2_activations \
  --profile baseline \
  --model gpt2_small \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 \
  --device mps \
  --max-samples-per-emotion 225
```

- **目的**: データセット上でresidual stream（resid_pre/resid_post）を抽出
- **出力**: `results/baseline/activations/gpt2_small.pkl`
- **次のPhase**: Phase 3で使用
- **実行時間目安**: 225サンプル/感情 × 4感情 × 12層 = 約10,800サンプルの活性抽出（数時間〜半日程度）

### 3) Phase 3: emotion vectors + token-based directions（PCA components）

```bash
python3 -m src.analysis.run_phase3_vectors \
  --profile baseline \
  --model gpt2_small \
  --n-components 8
```

- **目的**: Phase2で抽出した活性から感情ごとの差分ベクトルとPCAサブスペースを計算
- **出力**: 
  - `results/baseline/emotion_vectors/gpt2_small_vectors_token_based.pkl`
  - `results/baseline/emotion_subspaces/gpt2_small_subspaces.pkl`
- **次のPhase**: Phase 4（アライメント）とPhase 5（パッチング）で使用
- **実行時間目安**: 数分〜数十分

### 4) Phase 4: subspaces & alignment（self-alignment gpt2_small vs gpt2_small）

```bash
python3 -m src.analysis.run_phase4_alignment \
  --profile baseline \
  --model-a gpt2_small \
  --model-b gpt2_small \
  --k-max 8
```

- **目的**: サブスペースのモデル間アライメントを計算（自己アライメントで検証）
- **出力**: `results/baseline/alignment/gpt2_small_vs_gpt2_small_token_based_full.pkl`
- **次のPhase**: Phase 7（k選択）で使用
- **実行時間目安**: 数分

### 5) Phase 5: residual patching（multi-token, with random controls）

```bash
python3 -m src.analysis.run_phase5_residual_patching \
  --profile baseline \
  --model gpt2_small \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 \
  --patch-window 3 \
  --sequence-length 30 \
  --alpha 0.8 \
  --max-samples-per-emotion 225 \
  --device mps \
  --random-control \
  --num-random 50
```

- **目的**: 感情ベクトル/サブスペース方向を残差に注入し、生成文のsentiment/politeness/GoEmotions変化を測定。ランダム方向との比較で因果効果の特異性を検証。
- **出力**: 
  - `results/baseline/patching/residual/gpt2_small_residual_sweep.pkl`
  - `results/baseline/patching_random/gpt2_small_random_sweep.pkl`（--random-control実行時）
- **次のPhase**: Phase 7（統計解析）で使用
- **実行時間目安**: 225サンプル × 3感情（gratitude/anger/apology）× 12層 × 50ランダム = 非常に長時間（数日〜1週間程度の可能性）

### 6-a) Phase 6: head screening（全層ヘッドのスコアリング）

```bash
python3 -m src.analysis.run_phase6_head_screening \
  --profile baseline \
  --model gpt2_small \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 \
  --max-samples 225 \
  --sequence-length 30 \
  --device mps
```

- **目的**: 各ヘッドを網羅的にアブレートし、評価指標の変化から重要度をスクリーニング
- **出力**: `results/baseline/screening/head_scores_gpt2_small.json`
- **次のPhase**: Phase 6-b（重要ヘッドのパッチング）とPhase 7（統計解析）で使用
- **実行時間目安**: 12層 × 12ヘッド × 225サンプル = 非常に長時間（数日程度の可能性）

### 6-b) Phase 6: head patching / ablation

```bash
python3 -m src.analysis.run_phase6_head_patching \
  --profile baseline \
  --model gpt2_small \
  --heads 0:0-11 1:0-11 2:0-11 3:0-11 4:0-11 5:0-11 6:0-11 7:0-11 8:0-11 9:0-11 10:0-11 11:0-11 \
  --max-samples 225 \
  --sequence-length 30 \
  --device mps
```

- **目的**: 指定ヘッドの出力をゼロ化して因果効果を測定（全層全ヘッドを対象）
- **出力**: `results/baseline/patching/head_patching/gpt2_small_head_ablation.pkl`
- **次のPhase**: Phase 7（統計解析）で使用
- **実行時間目安**: 12層 × 12ヘッド × 225サンプル = 非常に長時間（数日程度の可能性）
- **注意**: スクリーニング結果に基づいて重要ヘッドのみに絞ることも可能

### 7) Phase 7: 統計解析（effect / power / k-selection）

```bash
# 効果量解析
python3 -m src.analysis.run_phase7_statistics \
  --profile baseline \
  --mode effect

# 検出力解析
python3 -m src.analysis.run_phase7_statistics \
  --profile baseline \
  --mode power \
  --effect-targets 0.2 0.5 \
  --power-target 0.8

# k選択解析（Phase 4のアライメント結果を使用）
python3 -m src.analysis.run_phase7_statistics \
  --profile baseline \
  --mode k
```

- **目的**: Phase5/6の結果を統合し、効果量・p値・信頼区間を算出。検出力分析やk選択を整理。
- **出力**: 
  - `results/baseline/statistics/effect_sizes.csv`
  - `results/baseline/statistics/power_analysis.csv` / `power_analysis.json`
  - `results/baseline/statistics/k_selection.csv`
- **実行時間目安**: 各モードで数分〜数十分

### 注意事項

- **デバイス**: デフォルトで`mps`を使用（M4 MacなどMetal対応環境で推奨）。`--device mps`が動作しない場合は`cpu`にフォールバック
- **サンプル数**: 225サンプル/感情は縮小しない（時間がかかっても実行）
- **ランダム本数**: `--num-random 50`は統計的検証に必要な最小本数として設定
- **メモリ不足**: メモリ不足などの問題が発生した場合は、エラーを記録して部分実行を提案（例: 層範囲を分割）
- **実行時間**: Phase 5とPhase 6は非常に長時間になる可能性がある。必要に応じて層やヘッドを絞ることも検討可能

### 旧版のフル実験用コマンド（参考）

以下は過去に記載されていた短縮版や例示用のコマンド。上記のフル実験レシピが正式版。

## 注意

- 入力データはユーザーが用意するCSV/JSONLを使用し、本コードは合成データを生成しません。
- 評価関連のログ・docstringはすべて日本語に統一しています。
- 過去の実験結果はリファクタ時に削除済み。必要ならGit履歴から取得してください。

## スモークテスト状況（2025-11-16）

- プロファイル `baseline_smoke` を追加（各感情5件程度の最小セット、目的は配線確認）  
  - 入力: `data/emotion_dataset_smoke.jsonl`（既存emotion_datasetから抽出）  
  - 生成: `python -m src.data.build_dataset --profile baseline_smoke --input data/emotion_dataset_smoke.jsonl`
- GPT-2 small を用いたPhase2〜4の軽量走行を実施（MPS）
  - Phase2: `python -m src.analysis.run_phase2_activations --profile baseline_smoke --model gpt2_small --layers 0 6 --device mps --max-samples-per-emotion 3`  
    - 出力: `results/baseline_smoke/activations/gpt2_small.pkl`
  - Phase3: `python -m src.analysis.run_phase3_vectors --profile baseline_smoke --model gpt2_small --n-components 3`  
    - 出力: `results/baseline_smoke/emotion_vectors/gpt2_small_vectors_token_based.pkl`  
            `results/baseline_smoke/emotion_subspaces/gpt2_small_subspaces.pkl`
  - Phase4: `python -m src.analysis.run_phase4_alignment --profile baseline_smoke --model-a gpt2_small --model-b gpt2_small --k-max 3`  
    - 出力: `results/baseline_smoke/alignment/gpt2_small_vs_gpt2_small_token_based_full.pkl`
- Phase5: `python -m src.analysis.run_phase5_residual_patching --profile baseline_smoke --model gpt2_small --layers 0 6 --patch-window 3 --sequence-length 20 --alpha 0.8 --max-samples-per-emotion 3 --device mps --random-control --num-random 2`  
  - 出力: `results/baseline_smoke/patching/residual/gpt2_small_residual_sweep.pkl`、`results/baseline_smoke/patching_random/gpt2_small_random_sweep.pkl`  
  - 評価器: sentiment/politeness/goemotions すべてロード成功
- Phase6: `python -m src.analysis.run_phase6_head_patching --profile baseline_smoke --model gpt2_small --heads 0:0 --max-samples 3 --sequence-length 20 --device mps`  
  - 出力: `results/baseline_smoke/patching/head_patching/gpt2_small_head_ablation.pkl`
- Phase6 スクリーニング: `python -m src.analysis.run_phase6_head_screening --profile baseline_smoke --model gpt2_small --layers 0 1 --max-samples 3 --sequence-length 20 --device mps`  
  - 出力: `results/baseline_smoke/screening/head_scores_gpt2_small.json`
- Phase7: `python -m src.analysis.run_phase7_statistics --profile baseline_smoke --mode effect`  
  - 出力: `results/baseline_smoke/statistics/effect_sizes.csv`

## ベースライン（本番プロファイル）実行ログ（現状: 短縮サンプルで配線確認）
- Phase2（短縮実行）: `python -m src.analysis.run_phase2_activations --profile baseline --model gpt2_small --layers 0 6 11 --device mps --max-samples-per-emotion 70`  
  - 出力: `results/baseline/activations/gpt2_small.pkl`
- Phase3: `python -m src.analysis.run_phase3_vectors --profile baseline --model gpt2_small --n-components 8`  
  - 出力: `results/baseline/emotion_vectors/gpt2_small_vectors_token_based.pkl` 等
- Phase4: `python -m src.analysis.run_phase4_alignment --profile baseline --model-a gpt2_small --model-b gpt2_small --k-max 8`  
  - 出力: `results/baseline/alignment/gpt2_small_vs_gpt2_small_token_based_full.pkl`
- Phase5（短縮版・各感情5件、random=1で実行）:  
  `python -m src.analysis.run_phase5_residual_patching --profile baseline --model gpt2_small --layers 0 6 11 --patch-window 3 --sequence-length 30 --alpha 0.8 --max-samples-per-emotion 5 --device mps --random-control --num-random 1`  
  - 出力: `results/baseline/patching/residual/gpt2_small_residual_sweep.pkl`, `results/baseline/patching_random/gpt2_small_random_sweep.pkl`  
  - 備考: プロンプト長ごとにバケット化してパディングせずに生成（Phase5バグ修正: EOSパディングによる条件付けを回避）。セッション時間制限のため短縮実行。フル設定は下記を参照。
- Phase6（短縮版ヘッドアブレーション）:  
  `python -m src.analysis.run_phase6_head_patching --profile baseline --model gpt2_small --heads 0:0 1:0 6:0 11:0 --max-samples 5 --sequence-length 30 --device mps`  
  - 出力: `results/baseline/patching/head_patching/gpt2_small_head_ablation.pkl`
- Phase7: `python -m src.analysis.run_phase7_statistics --profile baseline --mode effect`  
  - 出力: `results/baseline/statistics/effect_sizes.csv`

## NVIDIA/CUDA 環境用メモ（llama3等の大モデルを高速に回す場合）
1. セットアップ: `docs/dev/setup_ubuntu_gpu.md` に手順あり（Ubuntu 22.04 + RTX3090 + CUDA12.1 + PyTorch 2.5.1/cu121 + vLLM）。FlashInferは無効化（環境変数 `VLLM_ATTENTION_BACKEND=FLASHINFER_OFF`）。
2. 推奨実行: 大モデルは CUDA で実行（MPS は大幅に遅い）。
   - Phase2 (例) `python -m src.analysis.run_phase2_activations --profile baseline --model llama3_8b --layers 0 3 6 9 11 --device cuda --max-samples-per-emotion 50 --batch-size 8 --hook-pos resid_post`
   - Phase3 (例) `python -m src.analysis.run_phase3_vectors --profile baseline --model llama3_8b --use-torch --device cuda`
   - Phase5 (例) `python -m src.analysis.run_phase5_residual_patching --profile baseline --model llama3_8b --layers 0 3 6 9 11 --batch-size 8 --device cuda --max-samples-per-emotion 50`
   - Phase6 Patching/Screening (例) `--device cuda --batch-size 8`（ヘッド数や層を絞る）
3. ランダム対照は標準オフ。必要なら Phase5 で `--random-control --num-random N` を付け、完了後に Phase7 を再計算。
4. MPS はヘッドスクリーニングで極端に遅くなるため、CUDA 環境が望ましい。

### ベースラインの「フル」実験用コマンド（推奨; 長時間想定・未実行）
- Phase2（全層×225サンプル想定）  
  `python -m src.analysis.run_phase2_activations --profile baseline --model gpt2_small --layers 0 1 2 3 4 5 6 7 8 9 10 11 --device mps --max-samples-per-emotion 225`
- Phase3  
  `python -m src.analysis.run_phase3_vectors --profile baseline --model gpt2_small --n-components 8`
- Phase4（自己アライメント）  
  `python -m src.analysis.run_phase4_alignment --profile baseline --model-a gpt2_small --model-b gpt2_small --k-max 8`
- Phase5（残差パッチング＋ランダム50本）  
  `python -m src.analysis.run_phase5_residual_patching --profile baseline --model gpt2_small --layers 0 3 6 9 11 --patch-window 3 --sequence-length 30 --alpha 0.8 --max-samples-per-emotion 225 --device mps --random-control --num-random 50`
- Phase6（ヘッドアブレーション例: 前半層/後半層は全head、中央層は数本に絞るなど）  
  例: `python -m src.analysis.run_phase6_head_patching --profile baseline --model gpt2_small --heads 0:0-11 1:0-11 2:0-11 3:0-11 8:0-11 9:0-11 10:0-11 11:0-11 4:0-3 5:0-3 6:0-3 7:0-3 --max-samples 225 --sequence-length 30 --device mps`
- Phase6（ヘッドスクリーニング 全層×全head）  
  `python -m src.analysis.run_phase6_head_screening --profile baseline --model gpt2_small --layers 0 1 2 3 4 5 6 7 8 9 10 11 --max-samples 225 --sequence-length 30 --device mps`
- Phase7: `python -m src.analysis.run_phase7_statistics --profile baseline --mode effect`  
  - head_screening の結果も phase=head_screening として effect_sizes.csv に統合される（data_loading拡張済み）。

### ヘッドスクリーニングの使い方
- 例: `python -m src.analysis.run_phase6_head_screening --profile baseline_smoke --model gpt2_small --layers 0 1 --max-samples 3 --sequence-length 20 --device mps`  
  - 出力ファイル: `results/<profile>/screening/head_scores_<model>.json`  
  - 各headについて metricごとの delta_mean / delta_std / n_samples を記録  
  - headパッチング候補選定や統計処理前のランキングに利用可能  
  - 実行コストは (層数×ヘッド数×サンプル数) に比例するため、必要に応じて層やサンプルを間引く
  - 統計統合: run_phase7_statistics の phase_filter に含めれば effect_sizes.csv に phase=head_screening として出力される（baseline_smokeで確認済み）。
