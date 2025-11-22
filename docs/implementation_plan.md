# 実装計画（現行パイプライン）

本計画は Emotion Circuits Project の baseline/baseline_smoke プロファイルを対象に、現行スクリプト群で再現可能な実装順序と依存関係を定義する。

## 文書の位置づけと依存関係
- **上流**: 研究ビジョンとフェーズ仕様は `docs/proposal/masterplan.md`、`docs/proposal/emotion_circuits_project.md` を参照。README の「コア構成」「ドキュメントマップ」がこの文書の前提をまとめる。
- **横方向**: ランブック/ログは `docs/dev/PIPELINE_NOTES.md` が管理。実験状況・TODOは `docs/dev/IMPLEMENTATION_STATUS.md`, `docs/dev/NEXT_STEPS.md` に記載する。
- **下流**: 本計画に準拠して実行された結果は `docs/report/*.md`（Phase別レポート, テンプレは `docs/report_template/`）と `results/<profile>/...` に反映する。

依存の解決順は **README → masterplan → 本実装計画 → PIPELINE_NOTES → レポート**。仕様変更時はこの順で整合性を取り、`docs/archive/*` は参照のみとする。

## プロファイルとデータ
| プロファイル | 入力 | 目標サンプル | 目的 |
| --- | --- | --- | --- |
| `baseline` | `data/emotion_dataset.jsonl` | 感情4種 × 225件前後 | 正式な統計検証用フルラン |
| `baseline_smoke` | `data/emotion_dataset_smoke.jsonl` | 各感情3–5件 | 配線・評価器確認用スモーク |

`python -m src.data.build_dataset --profile <profile> --input <path>` で標準化ファイルを生成。`src/config/project_profiles.py` が各Phase CLIにパス/閾値を提供する。

## モデルと評価
- **モデルレジストリ**: `src/models/model_registry.py`
  - 小型（TransformerLens）: `gpt2_small`, `pythia-160m`, `gpt-neo-125m`
  - 中規模（LargeHFModelラッパ）: `llama3_8b`, `gemma3_12b`, `qwen3_8b`
- **活性API**: `src/models/activation_api.py`
- **評価スタック**: `src/analysis/evaluation.py`（sentiment/politeness/goemotions のHFモデル）

## フェーズ依存関係サマリ

| Phase | スクリプト | 主な入力 | 主な出力 | 次に使用する処理 |
| --- | --- | --- | --- | --- |
| Dataset build | `src/data/build_dataset.py` | 生CSV/JSONL, `src/config/project_profiles.py` | `data/emotion_dataset*.jsonl` | Phase2, Phase6 screening |
| Phase2 | `run_phase2_activations.py` | Dataset build結果, モデルレジストリ | `results/<profile>/activations/<model>.pkl` | Phase3 |
| Phase3 | `run_phase3_vectors.py` | Phase2出力 | `emotion_vectors/*.pkl`, `emotion_subspaces/*.pkl` | Phase4, Phase5 |
| Phase4 | `run_phase4_alignment.py` | Phase3出力 | `alignment/<modelA>_vs_<modelB>_*.pkl` | Phase7 (kモード) |
| Phase5 | `run_phase5_residual_patching.py` | Phase3出力, 評価器 | `patching/residual/*.pkl`, 任意で `patching_random/*.pkl` | Phase7 (effect/power) |
| Phase6a | `run_phase6_head_screening.py` | Dataset build結果, 評価器 | `screening/head_scores_<model>.json` | Phase6b候補選定, Phase7 |
| Phase6b | `run_phase6_head_patching.py` | 評価器, 任意でPhase6a結果 | `patching/head_patching/<model>_head_ablation.pkl` | Phase7 |
| Phase7 | `run_phase7_statistics.py` | Phase5/6出力, Phase4(k) | `statistics/effect_sizes.csv`, `power_analysis.*`, `k_selection.csv` | レポート/解析 |

## フェーズ別CLI（標準ルート）
- **Phase2**: `python -m src.analysis.run_phase2_activations --profile <profile> --model <model> --layers ... --max-samples-per-emotion ... --device <device> --batch-size <size>`
- **Phase3**: `python -m src.analysis.run_phase3_vectors --profile <profile> --model <model> --n-components 8 --use-torch --device <device>`
- **Phase4**: `python -m src.analysis.run_phase4_alignment --profile <profile> --model-a <model> --model-b <model> --k-max 8 --use-torch --device <device>`
- **Phase5**: `python -m src.analysis.run_phase5_residual_patching --profile <profile> --model <model> --layers ... --patch-window ... --sequence-length ... --alpha ... [--random-control --num-random ...] --device <device> --batch-size <size>`
- **Phase6 (head patching)**: `python -m src.analysis.run_phase6_head_patching --profile <profile> --model <model> --heads ... --max-samples ... --sequence-length ... --device <device> --batch-size <size>`
- **Phase6 (head screening)**: `python -m src.analysis.run_phase6_head_screening --profile <profile> --model <model> --layers ... --max-samples ... --sequence-length ... --device <device> --batch-size <size>`
- **Phase7**: `python -m src.analysis.run_phase7_statistics --profile <profile> --mode all --n-jobs <jobs>`（`--mode effect|power|k` で個別実行も可能）

## 包括的多モデル実験の進捗状況

### 完了したPhase
- ✅ Phase 1: データセット構築（baselineプロファイル、900サンプル）
- ✅ Phase 2: 活性抽出
  - 小型モデル3つ: gpt2_small, pythia-160m, gpt-neo-125m（各12層）
  - 中規模モデル2つ: llama3_8b（32層）, qwen3_8b（36層）
  - ⚠ gemma3_12bはメモリ不足のためスキップ（48層、後で個別対応）
- ✅ Phase 3: 感情ベクトル計算（全モデル5つ）

### 進行中・未完了のPhase
- ⏸ Phase 4: モデル間アライメント
  - ⚠ 異なるd_modelを持つモデル間のアライメントは現在サポートされていないためスキップ
  - 後で個別に対応が必要
- ✅ Phase 5: 残差パッチング
  - ✅ 完了: gpt2_small, pythia-160m, gpt-neo-125m（全12層）
  - ✅ 完了: llama3_8b（層0-31、層分割実行）、qwen3_8b（層0-35、層分割実行）
  - ⚠ スキップ: gemma3_12b（メモリ不足）
- ⏳ Phase 6: Head Screening & Patching（未実行、Phase5完了後）
- ⏳ Phase 7: 統計解析（未実行、Phase5/6完了後）

### 実行スクリプト
- `scripts/run_comprehensive_experiment.sh`: 全Phaseを順次実行
- `scripts/run_phase2_all_models.sh`: Phase2を全モデルで実行
- `scripts/run_phase3_all_models.sh`: Phase3を全モデルで実行
- `scripts/run_phase4_all_combinations.sh`: Phase4を全組み合わせで実行（現在スキップ）
- `scripts/run_phase5_all_models.sh`: Phase5を全モデルで実行（層分割対応）
- `scripts/run_phase6a_all_models.sh`: Phase6aを全モデルで実行（層分割対応）
- `scripts/run_phase6b_all_models.sh`: Phase6bを全モデルで実行
- `scripts/run_phase7_all.sh`: Phase7を全モードで実行

### 既知の問題
1. **gemma3_12bのメモリ不足**: 12BモデルはGPUメモリが不足するため、バッチサイズを1にしても実行できない。CPUオフロードや8bit量子化などの対応が必要。
2. **Phase4の異なるd_model間のアライメント**: 小型モデル（d_model=768）と中規模モデル（d_model=4096）間のアライメントは現在サポートされていない。Procrustes分析の実装を修正する必要がある。

## 高速化オプション
- `--batch-size`: バッチ処理のサイズ（デフォルト: 8-32）。メモリに応じて調整。
- `--device`: 計算デバイス（mps/cuda/cpu）。未指定時は自動選択。
- `--use-torch / --no-use-torch`: PCA/Procrustes計算でtorch（GPU/MPS加速）を使用するか（Phase 3/4）。
- `--n-jobs`: bootstrap並列計算のジョブ数（Phase 7）。-1で全CPU使用。

M4 Max等の高性能マシンで大幅な高速化が可能（従来比10-100倍以上）。CUDA環境では llama/gemma/qwen などの大モデルを実用的な時間で処理できる。

## フル実験目標（baseline, gpt2_small）
- Phase2: 全層(0-11) × 225サンプル/感情
- Phase3: PCA 8次元
- Phase4: 自己アライメント（必要に応じて他モデル比較）
- Phase5: 残差パッチング（層0/3/6/9/11、random=50以上）
- Phase6: ヘッドアブレーション & 全層スクリーニング
- Phase7.5: 統計集約（residual/random/head_patching/head_screening）

## Phase 7 出力ファイル（後続分析用）

Phase 7の統計解析結果は以下のパスに保存されます：

- **効果量**: `results/<profile>/statistics/effect_sizes.csv`
  - 各パッチング実験の効果量（Cohen's d）、p値、信頼区間、多重比較補正結果
- **検出力分析**: `results/<profile>/statistics/power_analysis.csv` と `power_analysis.json`
  - Post-hoc power、必要サンプル数の推定結果
- **k選択**: `results/<profile>/statistics/k_selection.csv`

**重要**: 後続分析でこれらのファイルを使用する際は、`--profile`で指定したプロファイル名（`baseline` または `baseline_smoke`）とパスを一致させること。

例:
```python
import pandas as pd
effect_df = pd.read_csv("results/baseline/statistics/effect_sizes.csv")
```

## 注意
- 旧v1スクリプト・レポートは削除済み。過去結果が必要な場合はGit履歴から取得。
- 実行時間は長くなる前提。計算資源に合わせて分割実行してもよいが、baselineは225件/感情を基本とする。
- **MPS環境**: Phase 3/4のSVDでCPUフォールバック警告が出る場合があるが、計算は継続して完了する。MPSが利用不可の場合はCPUに自動フォールバックする。
- **ランダム対照**: Phase5の `--random-control --num-random N` は効果量の分布確認に必須ではない。必要に応じて有効化し、Phase7を再実行する。

## テスト実行状況（2025-11-19更新）
- **データセット再生成**: baseline_smoke（20行）、baseline（900行）完了
- **スモールテスト**: baseline_smoke × gpt2_small で Phase2-7完了
- **中規模モデルスモーク**: baseline_smoke × llama3_8b で Phase2/3/5完了
- **網羅テスト**: baseline × gpt2_small と baseline × llama3_8b で Phase2-7完了
- **統計解析**: 両プロファイルでeffect_sizes.csvとpower_analysis.csv生成完了
- **詳細**: `docs/test_execution_summary.md` を参照

## 包括的多モデル実験の進捗（2025-01-XX更新）
- **詳細**: `docs/comprehensive_experiment_progress.md` を参照
- **Phase1-3**: 完了（5モデル）
- **Phase4**: スキップ（異なるd_model間のアライメント未サポート）
- **Phase5**: 実行中（gpt2_small完了、残り4モデル実行中）
- **Phase6-7**: 未実行（Phase5完了後）
