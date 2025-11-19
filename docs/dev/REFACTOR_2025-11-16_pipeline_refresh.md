# リファクタログ (2025-11-16, パイプライン整理)

- baselineプロファイルを「1感情225件前後」を前提とする構成に整理（旧版の70件は検出力不足のため廃止）。baseline_smokeは各感情3–5件の配線確認用。
- 旧バージョンのコード/結果/レポートを原則削除し、現在の標準パイプライン構成に統一。
- 現行パイプラインの構成要素:
  - プロファイル/データ: `src/config/project_profiles.py`, `src/data/build_dataset.py`
  - モデル/活性API: `src/models/model_registry.py`, `src/models/activation_api.py`
  - 評価: `src/analysis/evaluation.py`（公開HF: sentiment/politeness/goemotions）
  - CLIs: `run_phase2_activations.py`, `run_phase3_vectors.py`, `run_phase4_alignment.py`, `run_phase5_residual_patching.py`, `run_phase6_head_patching.py`, `run_phase6_head_screening.py`, `run_phase7_statistics.py`
- 過去の成果物が必要な場合はGit履歴から取得すること。

## サニティチェック結果 (2025-01-XX, GPT-2 small baseline pipeline定義)

### baselineプロファイルの定義確認
- ✅ `src/config/project_profiles.py`: baselineプロファイルは「1感情225件前後を目標」と明記されている（57-59行目）
- ✅ `src/data/build_dataset.py`: コメントに「baseline: 感情4種 × 各225サンプル前後（合計約900行）を想定」と記載（13行目）

### モデルレジストリ確認
- ✅ `src/models/model_registry.py`: `gpt2_small`が登録されている（41-48行目、hf_id="gpt2", n_layers_hint=12）

### 評価モデル確認
- ✅ `src/analysis/evaluation.py`: 公開HFモデルIDが正しく設定されている
  - sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest` (25行目)
  - politeness: `NOVA-vision-language/polite_bert` (26行目)
  - goemotions: `SamLowe/roberta-base-go_emotions` (27行目)

### "v2/refined"用語の残存確認
- ✅ コードベース全体で検索した結果、変数名として使用されているもののみ（`v2 = vec2[layer_idx]`など）。パイプライン関連の"v2"/"refined"用語は見つからなかった。

### コードの不整合確認
- ✅ `src/analysis/run_phase5_residual_patching.py`: 93行目で`torch.cuda.is_available()`を使用していたが、torchのインポートが不足していた。修正済み（`import torch`を追加）。

## GPT-2 small baseline pipeline実行状況 (2025-01-XX)

### 実行完了
- ✅ Phase 2: activations（全層0-11、225サンプル/感情） - 完了
  - 結果ファイル: `results/baseline/activations/gpt2_small.pkl`
  - サンプル数確認: 各感情225サンプル（合計900サンプル）
- ✅ Phase 3: emotion vectors + subspaces（n_components=8） - 完了
  - 結果ファイル: `results/baseline/emotion_vectors/gpt2_small_vectors_token_based.pkl`, `results/baseline/emotion_subspaces/gpt2_small_subspaces.pkl`
- ✅ Phase 4: alignment（self-alignment, k-max=8） - 完了
  - 結果ファイル: `results/baseline/alignment/gpt2_small_vs_gpt2_small_token_based_full.pkl`
- ✅ Phase 6-a: head screening（全層0-11、225サンプル） - 完了
  - 結果ファイル: `results/baseline/screening/head_scores_gpt2_small.json`
  - 12層分のヘッドスコアデータあり
- ✅ Phase 6-b: head patching（全層全ヘッド、225サンプル） - 完了
  - 結果ファイル: `results/baseline/patching/head_patching/gpt2_small_head_ablation.pkl`
  - 144ヘッド（12層 × 12ヘッド）分のデータあり

### 実行中/未完了
- ⏳ Phase 5: residual patching（全層0-11、225サンプル、random=50） - KeyError修正後に再実行中
  - 問題: `metric_map`のKeyErrorを修正（`patched_texts.keys()`を使用するように変更）
  - 結果ファイル: 未確認（`results/baseline/patching/residual/` に存在しない）
  - ランダム対照結果: 未確認（`results/baseline/patching_random/` に存在しない）
- ⏳ Phase 7: 統計解析 - Phase 5完了後に実行予定
  - 結果ファイル: 未存在（`results/baseline/statistics/` に存在しない）

### レポート生成状況
- ✅ Phase 2レポート: `docs/report/phase2_activations_gpt2_baseline.md` - 生成完了
- ✅ Phase 3レポート: `docs/report/phase3_vectors_gpt2_baseline.md` - 生成完了
- ✅ Phase 4レポート: `docs/report/phase4_alignment_gpt2_baseline.md` - 生成完了
- ⏳ Phase 5レポート: Phase 5完了後に生成予定
- ⏳ Phase 6-aレポート: Phase 6-a完了後に生成予定
- ⏳ Phase 6-bレポート: Phase 6-b完了後に生成予定
- ⏳ Phase 7レポート: Phase 5-6完了後に生成予定

### 実施した修正・改善
- ✅ 各Phaseスクリプトに所要時間ログを追加（Phase 2-7）
- ✅ tqdmを削除してキリの良い単位（10件/50件ごと）での進捗表示に変更
- ✅ Phase 5のKeyError修正: `metric_map`作成時に`patched_texts.keys()`を使用するように変更
- ✅ ドキュメントを更新してMPSをデフォルトに設定（`PIPELINE_NOTES.md`, `README.md`）
- ✅ データセットを225サンプル/感情に拡張（合計900サンプル）

### 注意事項
- Phase 5はKeyError修正後に再実行中。完了を待ってからPhase 7（統計解析）を実行する必要がある
- Phase 6-a/bは完了しているが、レポートは未生成
- デバイス: MPS（Metal）をデフォルトとして使用（M4 Mac環境）
