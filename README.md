# Emotion Circuits Project

軽量〜中規模LLMで感情回路を分析するための標準パイプラインです。旧版のスクリプト/結果は整理済みで、ここに記載の構成が正式な実行経路です。

## プロファイルと実験目標

### 研究ゴール
- 感情（gratitude/anger/apology/neutral）の内部表現を層・ヘッド・サブスペース単位で解析し、因果的介入（残差/ヘッドパッチング）で効果を測定する。
- 感情方向の共通構造を複数モデルで比較し、統計的検定とランダム対照で再現性を保証する。
- baselineプロファイルでは **1感情225件前後** のJSONLデータを想定。配線確認用のbaseline_smokeは各感情3–5件。

### プロファイル一覧
| Profile | 感情ごとの目標サンプル数 | 入力データ | 用途 |
| --- | --- | --- | --- |
| `baseline` | 225前後 | `data/emotion_dataset.jsonl` | 正式な再実験ルート。Phase2〜7をフルで走らせる |
| `baseline_smoke` | 3–5 | `data/emotion_dataset_smoke.jsonl` | ハードウェア/配線確認用の最小セット |

データは `python -m src.data.build_dataset --profile <profile> --input <path>` で標準フォーマット化し、`src/config/project_profiles.py` が各Phase CLIsへパス等を提供する。

## コア構成（現行）
- プロファイル/データ: `src/config/project_profiles.py`, `src/data/build_dataset.py`
- モデル/活性API: `src/models/model_registry.py`, `src/models/activation_api.py`
- 評価: `src/analysis/evaluation.py`（公開HFモデル: sentiment/politeness/goemotions）
- CLIs（Phase2〜6, 7.5統計）:
  - `src/analysis/run_phase2_activations.py`
  - `src/analysis/run_phase3_vectors.py`
  - `src/analysis/run_phase4_alignment.py`
  - `src/analysis/run_phase5_residual_patching.py`
  - `src/analysis/run_phase6_head_patching.py`
  - `src/analysis/run_phase6_head_screening.py`
- Phase7統計: `src/analysis/run_phase7_statistics.py`

## フェーズ別CLI概要

| Phase | スクリプト | 目的 | 主要依存関係 | 主な出力 |
| --- | --- | --- | --- | --- |
| Dataset build | `src/data/build_dataset.py` | CSV/JSONLをプロファイル別データセットへ変換 | `data/<profile>.jsonl`, `src/config/project_profiles.py` | `data/emotion_dataset*.jsonl` |
| Phase2 | `run_phase2_activations.py` | resid_pre/resid_postを抽出 | dataset build結果, `src/models/activation_api.py` | `results/<profile>/activations/<model>.pkl` |
| Phase3 | `run_phase3_vectors.py` | 感情差分ベクトル/PCAサブスペース | Phase2出力, `src/analysis/evaluation.py` | `results/<profile>/emotion_vectors/*.pkl`, `emotion_subspaces/*.pkl` |
| Phase4 | `run_phase4_alignment.py` | モデル間サブスペースアライメント | Phase3出力 | `results/<profile>/alignment/*.pkl` |
| Phase5 | `run_phase5_residual_patching.py` | Multi-token残差パッチング + 任意のランダム対照 | Phase3出力, 評価器 | `results/<profile>/patching/residual/*.pkl`, `patching_random/*.pkl` |
| Phase6a | `run_phase6_head_screening.py` | 全ヘッド重要度スクリーニング | dataset build結果, 評価器 | `results/<profile>/screening/head_scores_<model>.json` |
| Phase6b | `run_phase6_head_patching.py` | ヘッド単位パッチング/アブレーション | Phase6aでの候補（任意）, 評価器 | `results/<profile>/patching/head_patching/<model>_head_ablation.pkl` |
| Phase7 | `run_phase7_statistics.py` | 統計集約（effect/power/k） | Phase5/6出力, 任意でPhase4 | `results/<profile>/statistics/*.csv` |

## フェーズ時間計測とレポート
- **時間計測の自動記録**: `build_dataset` と Phase2〜7の各CLIは終了時に `results/<profile>/timing/phase_timings.jsonl` へ実行メタデータを追記する。各行には `phase`, `model`, `device`, `samples`, `elapsed_seconds`, `metadata`（層/ヘッド/サンプリング設定など）と実行コマンドが保存され、モデル規模ごとの実行時間を後から集計できる。
- **記録の使い方**: JSONLなので `pandas.read_json(..., lines=True)` で読み出し、`phase` や `model` で groupby すると各フェーズの時間分布を即座に取得可能。CUDA移行後の性能比較はこのログを参照する。
- **レポート生成CLI**: `python -m src.reporting.generate_phase_report --phase phase5 --profile baseline --model gpt2_small --output docs/report/phase5_baseline_gpt2_small.md` のように呼び出すと、該当テンプレート (`docs/report_template/phase5_residual_patching_template.md`) をコピーし、直近のタイミングエントリを末尾の「自動生成サマリ」に埋め込む。`--timing-run-id` を指定すれば特定ランの情報でレポートを再生成可能。
- **LLMレビュー向け**: サマリブロックには実行ID・所要時間・サンプル数・主要ハイパラが列挙されるため、査読者やLLMが参照する際にフェーズ間の比較が容易。テンプレート本文はこれまで通り手動で詳細を記述できる。

## ドキュメントマップと依存関係

| ドキュメント | 役割 | 依存元 | 下流/参照先 |
| --- | --- | --- | --- |
| `docs/proposal/masterplan.md` | 感情回路研究のビジョン・フェーズ仕様のソースオブトゥルース | プロジェクト仮説、`docs/proposal/emotion_circuits_project.md` | 他ドキュメントの全体方針、レビュー時の根拠 |
| `docs/implementation_plan.md` | 現行パイプラインの実装計画とPhase別要件 | masterplan、`src/config/project_profiles.py`, README | `docs/dev/PIPELINE_NOTES.md`, 各Phase実装/QA |
| `docs/dev/PIPELINE_NOTES.md` | 実行ログ/ランブック。フル実験やスモークの詳細なステップ | implementation_plan, `src/analysis/run_phase*` | `docs/report/*`, `results/<profile>/...` の更新チェック |
| `docs/dev/SETUP.md`, `docs/dev/setup_ubuntu_gpu.md` | MPS/CUDA環境構築ガイド | `pyproject.toml`, READMEの依存一覧 | 開発環境構築、Phase2以降の実行 |
| `docs/dev/IMPLEMENTATION_STATUS.md`, `NEXT_STEPS.md` | 現状の進捗とTODO | masterplan, PIPELINE_NOTES | スプリント計画、PR説明 |
| `docs/report/*.md` | フェーズ別レポートテンプレ/実績 | `results/<profile>/...`, PIPELINE_NOTES | レビュー資料、チーム報告 |
| `docs/report_template/` | Phase別サマリ雛形 | READMEのPhase説明 | 新規レポート作成 |
| `docs/archive/*` | 旧版（v1）レポート。歴史参照のみ | 過去の実験結果 | 参照専用。現行更新には使用しない |

旧ドキュメントの仕様を再販する場合は、上表の「依存元」を確認し、README→masterplan→implementation_planの順で整合性を取ること。

## データセット
- baseline: `data/emotion_dataset.jsonl`（4感情×225件を目標）。ユーザーが用意したCSV/JSONLを `build_dataset.py` で変換。
- baseline_smoke: `data/emotion_dataset_smoke.jsonl`（各感情3–5件の配線確認用）。

## 典型的な実行例（baseline、高速バッチ処理版）

```bash
# Phase2: 活性抽出（バッチ処理対応）
python -m src.analysis.run_phase2_activations --profile baseline --model gpt2_small --layers 0 1 2 3 4 5 6 7 8 9 10 11 --device mps --max-samples-per-emotion 225 --batch-size 32

# Phase3: 感情ベクトル/サブスペース（torch/sklearn切り替え可能）
python -m src.analysis.run_phase3_vectors --profile baseline --model gpt2_small --n-components 8 --use-torch --device mps

# Phase4: 自己アライメント（torch/numpy切り替え可能）
python -m src.analysis.run_phase4_alignment --profile baseline --model-a gpt2_small --model-b gpt2_small --k-max 8 --use-torch --device mps

# Phase5: 残差パッチング（バッチ処理対応）
# 小型（HookedTransformer）
python -m src.analysis.run_phase5_residual_patching --profile baseline --model gpt2_small --layers 0 1 2 3 4 5 6 7 8 9 10 11 --patch-window 3 --sequence-length 30 --alpha 1.0 --max-samples-per-emotion 8 --device mps --batch-size 16
# 大型（LargeHFModel, 例: llama3_8b） ※ランダム対照は任意（標準はオフ） / CUDA推奨
python -m src.analysis.run_phase5_residual_patching --profile baseline --model llama3_8b --layers 0 3 6 9 11 --patch-window 3 --sequence-length 30 --alpha 1.0 --max-samples-per-emotion 50 --device cuda --batch-size 8

# Phase6: ヘッドパッチング（バッチ処理対応）
# 小型（HookedTransformer）
python -m src.analysis.run_phase6_head_patching --profile baseline --model gpt2_small --heads 0:0-11 3:0-11 6:0-11 9:0-11 11:0-11 --max-samples 8 --sequence-length 30 --device mps --batch-size 8
# 大型（LargeHFModel, CUDA推奨）
python -m src.analysis.run_phase6_head_patching --profile baseline --model llama3_8b --heads 0:0-11 3:0-11 6:0-11 9:0-11 11:0-11 --max-samples 50 --sequence-length 30 --device cuda --batch-size 8

# Phase6: ヘッドスクリーニング（バッチ処理対応）
# 小型（HookedTransformer）
python -m src.analysis.run_phase6_head_screening --profile baseline --model gpt2_small --layers 0 1 2 3 4 5 6 7 8 9 10 11 --max-samples 8 --sequence-length 30 --device mps --batch-size 8
# 大型（LargeHFModel, CUDA推奨）
python -m src.analysis.run_phase6_head_screening --profile baseline --model llama3_8b --layers 0 3 6 9 11 --max-samples 50 --sequence-length 30 --device cuda --batch-size 8

# Phase7: 統計（並列処理対応）
python -m src.analysis.run_phase7_statistics --profile baseline --mode all --n-jobs 4
```

## Phase 7 出力ファイル（後続分析用）

Phase 7の統計解析結果は以下のパスに保存されます：

- **効果量**: `results/<profile>/statistics/effect_sizes.csv`
  - 各パッチング実験の効果量（Cohen's d）、p値、信頼区間、多重比較補正結果
  - カラム: `profile`, `phase`, `model_name`, `metric_name`, `layer`, `head`, `cohens_d`, `p_value`, `p_value_bonferroni`, `p_value_bh_fdr`, など
- **検出力分析**: `results/<profile>/statistics/power_analysis.csv`
  - Post-hoc power、必要サンプル数の推定結果
- **検出力サマリー**: `results/<profile>/statistics/power_analysis.json`
  - 目標効果量ごとの必要サンプル数の要約

**注意**: 後続分析でこれらのファイルを使用する際は、`--profile`で指定したプロファイル名（`baseline` または `baseline_smoke`）とパスを一致させること。

例:
```python
import pandas as pd
# baselineプロファイルの効果量を読み込む
effect_df = pd.read_csv("results/baseline/statistics/effect_sizes.csv")
```

## 高速化オプション

各Phase CLIで利用可能なオプション:

- **--batch-size**: バッチ処理のサイズ（デフォルト: 8-32）。メモリに応じて調整。
- **--device**: 計算デバイス（mps/cuda/cpu）。未指定時は自動選択。
- **--use-torch / --no-use-torch**: PCA/Procrustes計算でtorch（GPU/MPS加速）を使用するか（Phase 3/4）。
- **--n-jobs**: bootstrap並列計算のジョブ数（Phase 7）。-1で全CPU使用。
- **--random-control --num-random N**: Phase5のランダム対照（オプション）。標準フローではオフ。
- **CUDA推奨**: llama3 等の大モデルは CUDA 環境での実行を推奨（MPSは大幅に遅い）。

これらのオプションにより、M4 Max等の高性能マシンで大幅な高速化が可能です（従来比10-100倍以上）。

## MPS環境での注意事項

- **Phase 4**: `torch.linalg.pinv`内部でSVDがCPUにフォールバックする警告が表示される場合がありますが、処理は正常に継続します。MPS上で計算が完了するため、性能影響は限定的です。
- **Phase 3**: PCA計算のSVDでも同様の警告が表示される場合がありますが、処理は正常に完了します。

## 開発メモ
- 追加の運用手順や実行ログは `docs/dev/PIPELINE_NOTES.md` を参照してください。
- 本リポジトリには旧版成果物は含まれていません。過去の結果が必要な場合はGit履歴から取得してください。
- `docs/archive/*` 配下は歴史的な参照のみ。現行システムの更新は README → masterplan → implementation_plan → pipeline_notes の順で整合を取ること。
