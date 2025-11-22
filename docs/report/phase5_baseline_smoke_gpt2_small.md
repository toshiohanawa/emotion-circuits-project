# Phase 5 — 残差パッチング（multi-token＋ランダム対照）

## 🎯 目的
- 感情ベクトル/サブスペース方向を残差に注入し、生成文の sentiment / politeness / GoEmotions がどの程度変化するかを測定する。
- ランダム方向との比較で因果効果が特異的かどうかを検証する。

## 📦 生成物
- `results/<profile>/patching/residual/<model>_residual_sweep.pkl`
- `results/<profile>/patching_random/<model>_random_sweep.pkl`（--random-control 実行時）
- 観測メトリクス: sentiment / politeness / goemotions など（`effect_sizes.csv` の入力）
- 本レポート（例: `docs/phase5_residual_patching_report.md`）

## 🚀 実行コマンド例
```bash
python -m src.analysis.run_phase5_residual_patching \
  --profile baseline \
  --model gpt2_small \
  --layers 0 3 6 9 11 \
  --patch-window 3 \
  --sequence-length 30 \
  --alpha 0.8 \
  --max-samples-per-emotion 225 \
  --device mps \
  --batch-size 16 \
  --random-control \
  --num-random 50
```
※ 配線確認なら baseline_smoke と少数サンプルで実行。
※ `--device mps` でApple Silicon加速、`--batch-size` でメモリに応じた並列処理が可能。

## 📄 レポート項目
1. 実行設定
   - プロファイル / モデル / 層 / patch_window / sequence_length / alpha / random本数 / サンプル数
2. メトリクス変化（baseline vs patched vs random）
   - sentiment / politeness / goemotions の平均変化と標準誤差
   - レイヤー別・感情別のグラフ
3. テキスト例
   - baseline vs patched 出力の抜粋（崩れやスタイル変化の例示）
4. 考察
   - 効果が大きい層や感情方向
   - ランダム対照との差分（特異性）
5. 次のアクション
   - alpha スイープ、ウィンドウ変更、他モデル適用など

---
## 自動生成サマリ（LLMレビュー用）
- 実行ID: `a2fc6ddb-9e70-4b54-813e-96787ead59df`
- プロファイル: `baseline_smoke`
- フェーズ: `phase5`
- モデル: `gpt2_small`
- デバイス: `cuda`
- サンプル数: 3
- 計測時間: 4.64 秒 (0.08 分)
- 実行日時: 2025-11-19T08:50:53.495755+00:00
- タイミングログ: `results/baseline_smoke/timing/phase_timings.jsonl`
- 実行コマンド: `python --profile baseline_smoke --model gpt2_small --layers 0 6 --patch-window 3 --sequence-length 20 --alpha 0.8 --max-samples-per-emotion 3 --device cuda --batch-size 4`
- メタデータ:
  - alpha: 0.8
  - batch_size: 4
  - layers: [0, 6]
  - num_random: 0
  - patch_window: 3
  - random_control: false
  - random_result_path: null
  - result_path: "results/baseline_smoke/patching/residual/gpt2_small_residual_sweep.pkl"
  - sequence_length: 20
