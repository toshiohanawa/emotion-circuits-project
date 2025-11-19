# 実装計画（現行パイプライン）

本計画は現在の標準パイプラインを前提とする。baselineは4感情×225件前後のJSONLデータを目標とする。

## プロファイルとデータ
- `baseline`: 感情4種×225件を想定。`data/emotion_dataset.jsonl` を `build_dataset.py` で変換。
- `baseline_smoke`: 各感情3–5件の配線確認用。

## モデルと評価
- モデル: `src/models/model_registry.py`（小型: gpt2_small, pythia-160m, gpt-neo-125m）
- 活性API: `src/models/activation_api.py`
- 評価: `src/analysis/evaluation.py`（公開HFモデル: sentiment/politeness/goemotions）

## フェーズ別CLI（標準ルート）
- Phase2: `python -m src.analysis.run_phase2_activations --profile <profile> --model <model> --layers ... --max-samples-per-emotion ... --device <device> --batch-size <size>`
- Phase3: `python -m src.analysis.run_phase3_vectors --profile <profile> --model <model> --n-components 8 --use-torch --device <device>`
- Phase4: `python -m src.analysis.run_phase4_alignment --profile <profile> --model-a <model> --model-b <model> --use-torch --device <device>`
- Phase5: `python -m src.analysis.run_phase5_residual_patching --profile <profile> --model <model> --layers ... --patch-window ... --sequence-length ... --alpha ... --random-control --num-random ... --device <device> --batch-size <size>`
- Phase6 (head ablation): `python -m src.analysis.run_phase6_head_patching --profile <profile> --model <model> --heads ... --device <device> --batch-size <size>`
- Phase6 (head screening): `python -m src.analysis.run_phase6_head_screening --profile <profile> --model <model> --layers ... --device <device> --batch-size <size>`
- Phase7 (統計): `python -m src.analysis.run_phase7_statistics --profile <profile> --mode effect --n-jobs <jobs>`

## 高速化オプション
各Phase CLIで利用可能なオプション:
- `--batch-size`: バッチ処理のサイズ（デフォルト: 8-32）。メモリに応じて調整。
- `--device`: 計算デバイス（mps/cuda/cpu）。未指定時は自動選択。
- `--use-torch / --no-use-torch`: PCA/Procrustes計算でtorch（GPU/MPS加速）を使用するか（Phase 3/4）。
- `--n-jobs`: bootstrap並列計算のジョブ数（Phase 7）。-1で全CPU使用。

M4 Max等の高性能マシンで大幅な高速化が可能（従来比10-100倍以上）。

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

**重要**: 後続分析でこれらのファイルを使用する際は、`--profile`で指定したプロファイル名（`baseline` または `baseline_smoke`）とパスを一致させること。

例:
```python
import pandas as pd
effect_df = pd.read_csv("results/baseline/statistics/effect_sizes.csv")
```

## 注意
- 旧v1スクリプト・レポートは削除済み。過去結果が必要な場合はGit履歴から取得。
- 実行時間は長くなる前提。計算資源に合わせて分割実行してもよいが、baselineは225件/感情を基本とする。
- **MPS環境**: Phase 4で`torch.linalg.pinv`内部のSVDがCPUにフォールバックする警告が表示される場合がありますが、処理は正常に継続します。*** End Patch
