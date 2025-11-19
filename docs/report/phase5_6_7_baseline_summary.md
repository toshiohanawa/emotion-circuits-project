# Phase 5–7 実行サマリー（baseline, 225 samples/感情）

## 概要
- 対象モデル: gpt2_small, pythia-160m, gpt-neo-125m
- サンプル数: 225/感情（3感情）
- バッチ設定: 生成32 / 評価32（MPS, 128GB RAM）
- ランダム対照: 10本
- 実行モード: Phase 5 → Phase 6 → Phase 7

## 実行時間（実測）
- Phase 5 残差パッチング: 約8時間  
  - gpt2_small (ランダムなし): 約6分  
  - gpt2_small (ランダム10本): 約2時間39分  
  - pythia-160m: 約2時間40分  
  - gpt-neo-125m: 約2時間34分  
- Phase 6 Headスクリーニング+パッチング: 約1時間10分  
  - スクリーニング: 約23分/モデル（3モデル計）  
  - パッチング: 約30秒/モデル（3モデル計）  
- Phase 7 統計解析: 約1時間7分（中間2回＋全体2回）  
- 総計: 約10時間17分（推定12–15時間に対し短縮）

## 生成ファイル（主要）
- Phase 5 残差パッチング:  
  - `results/baseline/patching/residual/gpt2_small_residual_sweep.pkl`  
  - `results/baseline/patching/residual/pythia-160m_residual_sweep.pkl`  
  - `results/baseline/patching/residual/gpt-neo-125m_residual_sweep.pkl`  
- Phase 6 Headスクリーニング:  
  - `results/baseline/screening/head_scores_gpt2_small.json`  
  - `results/baseline/screening/head_scores_pythia-160m.json`  
  - `results/baseline/screening/head_scores_gpt-neo-125m.json`  
- Phase 6 Headパッチング:  
  - `results/baseline/patching/head_patching/gpt2_small_head_ablation.pkl`  
  - `results/baseline/patching/head_patching/pythia-160m_head_ablation.pkl`  
  - `results/baseline/patching/head_patching/gpt-neo-125m_head_ablation.pkl`  
- Phase 7 統計:  
  - `results/baseline/statistics/effect_sizes.csv`  
  - `results/baseline/statistics/power_analysis.csv`  
  - `results/baseline/statistics/power_analysis.json`  
- ランダム対照: 現状10本（`patching_random`未生成）

## 主要メモ/所感
- 推定12–15時間に対し実測約10時間で完了。バッチ32がMPS環境で有効に利いた。  
- ランダム対照は10本のみ実施。追加で20/50本に拡張する場合はPhase5→Phase7を再計算。  
- スクリーニングJSONが ~118MB/モデルと大きいため、長期保存時は圧縮・アーカイブ推奨。  
- Phase4はpinv(SVD)でCPU警告が出るが処理は継続、総時間への影響は小。  
- バッチ32で問題なし。より大きなバッチはMPSではスループット頭打ちのことが多い。  

## 次の一手（必要なら）
1. ランダム対照を20/50本に拡張し、Phase5→Phase7を再実行。  
2. よく使うメトリクスの可視化（Layer×Emotionヒートマップ、Head上位ランキングなど）を追加。  
3. スクリーン結果の圧縮・バックアップでストレージ最適化。  

