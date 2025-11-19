# Phase 7 — 統計集約（run_phase7_statistics）

## 🎯 目的
- Phase5/6（残差パッチング・ヘッドパッチング/スクリーニング）の結果を統合し、効果量・p値・信頼区間を算出する。
- 検出力分析やk選択（アライメント結果があれば）を整理し、必要サンプル数の目安を示す。

## 📦 生成物
- `results/<profile>/statistics/effect_sizes.csv`
- `results/<profile>/statistics/power_analysis.csv` / `power_analysis.json`（--mode power 実行時）
- `results/<profile>/statistics/k_selection.csv`（アライメントk-sweepがある場合）
- 本レポート（例: `docs/phase7_statistics_report.md`）

## 🚀 実行コマンド例
```bash
python -m src.analysis.run_phase7_statistics \
  --profile baseline \
  --mode all \
  --phase-filter residual,head,random,head_screening \
  --n-bootstrap 500 \
  --effect-targets 0.2 0.5 \
  --power-target 0.85 \
  --seed 42 \
  --n-jobs 4
```
※ baseline_smoke では小さな n で配線確認し、本番は baseline で 225件/感情と十分な random 本数を前提。
※ `--n-jobs` で bootstrap並列計算を制御（-1で全CPU使用）。高速化に有効。

## 📄 レポート項目
1. 実行設定
   - プロファイル / 対象フェーズ（residual/head/random/head_screening） / ブートストラップ回数 / α
2. 効果量・有意性
   - metric×層/ヘッド別の mean_diff, Cohen’s d, 95% CI, p値, 補正後有意性
3. 検出力分析
   - 観測効果量の分布、post-hoc power、目標効果量ごとの必要サンプル数
4. k選択（該当時）
   - k別 overlap の集計と最適kの示唆
5. 考察
   - 効果が顕著な層/ヘッド、ランダムとの差分、今後増やすべきサンプル数
