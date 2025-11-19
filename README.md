# Emotion Circuits Project

軽量〜中規模LLMで感情回路を分析するための標準パイプラインです。旧版のスクリプト/結果は整理済みで、ここに記載の構成が正式な実行経路です。

## 目標
- 感情（gratitude/anger/apology/neutral）の内部表現を層・ヘッド・サブスペース単位で解析し、因果的介入（残差/ヘッドパッチング）で効果を測定する。
- baselineプロファイルでは **1感情225件前後** のJSONLデータを想定。配線確認用のbaseline_smokeは各感情3–5件。

## コア構成 (現行)
- プロファイル/データ: `src/config/project_profiles.py`, `src/data/build_dataset.py`
- モデル/活性API: `src/models/model_registry.py`, `src/models/activation_api.py`
- 評価: `src/analysis/evaluation.py`（公開HFモデル: sentiment/politeness/goemotions）
- CLIs (Phase2–6, 7.5統計):  
  - `src/analysis/run_phase2_activations.py`  
  - `src/analysis/run_phase3_vectors.py`  
  - `src/analysis/run_phase4_alignment.py`  
  - `src/analysis/run_phase5_residual_patching.py`  
  - `src/analysis/run_phase6_head_patching.py`  
  - `src/analysis/run_phase6_head_screening.py`  
- `src/analysis/run_phase7_statistics.py`
- ドキュメント: `docs/dev/PIPELINE_NOTES.md`, `docs/implementation_plan.md`, `masterplan.md`

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
# 大型（LargeHFModel, 例: llama3_8b） ※ランダム対照は任意（標準はオフ）
python -m src.analysis.run_phase5_residual_patching --profile baseline --model llama3_8b --layers 0 3 6 9 11 --patch-window 3 --sequence-length 30 --alpha 1.0 --max-samples-per-emotion 50 --device mps --batch-size 4

# Phase6: ヘッドパッチング（バッチ処理対応）
# 小型（HookedTransformer）
python -m src.analysis.run_phase6_head_patching --profile baseline --model gpt2_small --heads 0:0-11 3:0-11 6:0-11 9:0-11 11:0-11 --max-samples 8 --sequence-length 30 --device mps --batch-size 8
# 大型（LargeHFModel）
python -m src.analysis.run_phase6_head_patching --profile baseline --model llama3_8b --heads 0:0-11 3:0-11 6:0-11 9:0-11 11:0-11 --max-samples 50 --sequence-length 30 --device mps --batch-size 4

# Phase6: ヘッドスクリーニング（バッチ処理対応）
# 小型（HookedTransformer）
python -m src.analysis.run_phase6_head_screening --profile baseline --model gpt2_small --layers 0 1 2 3 4 5 6 7 8 9 10 11 --max-samples 8 --sequence-length 30 --device mps --batch-size 8
# 大型（LargeHFModel）
python -m src.analysis.run_phase6_head_screening --profile baseline --model llama3_8b --layers 0 3 6 9 11 --max-samples 50 --sequence-length 30 --device mps --batch-size 4

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

これらのオプションにより、M4 Max等の高性能マシンで大幅な高速化が可能です（従来比10-100倍以上）。

## MPS環境での注意事項

- **Phase 4**: `torch.linalg.pinv`内部でSVDがCPUにフォールバックする警告が表示される場合がありますが、処理は正常に継続します。MPS上で計算が完了するため、性能影響は限定的です。
- **Phase 3**: PCA計算のSVDでも同様の警告が表示される場合がありますが、処理は正常に完了します。

## 開発メモ
- 追加の運用手順や実行ログは `docs/dev/PIPELINE_NOTES.md` を参照してください。
- 本リポジトリには旧版成果物は含まれていません。過去の結果が必要な場合はGit履歴から取得してください。
