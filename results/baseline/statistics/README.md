# Phase 7.5 統計的厳密性分析結果の説明

このディレクトリには、Phase 7.5（統計的厳密性の強化）で生成された統計分析結果が含まれています。
各CSVファイルの構造と意味を以下に説明します。

## ファイル一覧

1. **effect_sizes.csv** - 効果量分析の結果（525行）
2. **power_analysis.csv** - 検出力分析の結果（525行）
3. **power_analysis.json** - 検出力分析のメタデータとサンプルサイズ要件
4. **k_selection.csv** - k選択の統計的検証結果（12行）
5. **k_sweep_raw.csv** - kスイープの生データ（60行）

---

## 1. effect_sizes.csv

### 目的
残差パッチング実験における効果量（effect size）と統計的有意性を評価した結果です。
各実験条件（モデル、感情、層、α値）ごとに、パッチング前後のメトリクス変化を統計的に検証しています。

### 列の説明

| 列名 | 型 | 説明 |
|------|-----|------|
| `profile` | string | データセットプロファイル（`baseline` または `extended`） |
| `phase` | string | 実験フェーズ（`residual` = 残差パッチング、`head` = ヘッドパッチング） |
| `model_name` | string | 使用したモデル名（例: `gpt2`, `EleutherAI-pythia-160m`） |
| `emotion` | string | 感情カテゴリ（`gratitude`, `anger`, `apology`） |
| `layer` | int | パッチングを適用した層番号（3, 5, 7, 9, 11） |
| `head` | int/empty | ヘッド番号（残差パッチングの場合は空） |
| `head_spec` | string/empty | ヘッド指定（例: `"3:5"` = Layer 3 Head 5、残差パッチングの場合は空） |
| `alpha` | float | パッチング強度（-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0） |
| `comparison` | string | 比較タイプ（`patched_vs_neutral` = パッチング後 vs 中立ベースライン） |
| `metric_name` | string | 評価メトリクス名 |
| | | - `sentiment`: センチメントスコア（0-1、高いほどポジティブ） |
| | | - `politeness`: 礼儀正しさスコア（0-1、高いほど丁寧） |
| | | - `emotion_keywords.anger`: 怒りキーワード出現数 |
| | | - `emotion_keywords.apology`: 謝罪キーワード出現数 |
| | | - `emotion_keywords.gratitude`: 感謝キーワード出現数 |
| `paired` | bool | 対応のある検定かどうか（`True` = 同じプロンプトでパッチング前後を比較） |
| `n_baseline` | int | ベースライン条件のサンプル数（通常70） |
| `n_patched` | int | パッチング条件のサンプル数（通常70） |
| `n_effective` | int | 有効サンプル数（対応のある検定では通常70） |
| `mean_baseline` | float | ベースライン条件の平均値 |
| `mean_patched` | float | パッチング条件の平均値 |
| `std_baseline` | float | ベースライン条件の標準偏差 |
| `std_patched` | float | パッチング条件の標準偏差 |
| `mean_diff` | float | 平均値の差分（`mean_patched - mean_baseline`） |
| `cohen_d` | float/empty | Cohen's d（効果量の標準化指標） |
| | | - 空欄: 標準偏差が0で計算不可 |
| | | - 0.2未満: 小さい効果 |
| | | - 0.2-0.5: 中程度の効果 |
| | | - 0.5-0.8: 大きい効果 |
| | | - 0.8以上: 非常に大きい効果 |
| `ci_lower` | float | 平均値差分の95%信頼区間（下限） |
| `ci_upper` | float | 平均値差分の95%信頼区間（上限） |
| `effect_size_ci_lower` | float | Cohen's dの95%信頼区間（下限） |
| `effect_size_ci_upper` | float | Cohen's dの95%信頼区間（上限） |
| `t_statistic` | float/empty | t検定統計量 |
| `p_value` | float/empty | 両側t検定のp値 |
| `p_value_bonferroni` | float/empty | Bonferroni補正後のp値 |
| `p_value_fdr` | float/empty | Benjamini-Hochberg FDR補正後のp値 |
| `significant_bonferroni` | bool | Bonferroni補正後も有意か（α=0.05） |
| `significant_fdr` | bool | FDR補正後も有意か（α=0.05） |

### データの読み方

- **効果量の解釈**: `cohen_d`が0.2以上で実用的に意味のある効果とみなされます
- **統計的有意性**: `p_value_fdr < 0.05` または `significant_fdr == True` で、FDR補正後も有意
- **信頼区間**: `ci_lower`と`ci_upper`が0を含まない場合、統計的に有意な差がある可能性が高い

### 例
```
baseline,residual,gpt2,gratitude,9,,,1.0,patched_vs_neutral,sentiment,True,70,70,70,0.0043,0.0121,0.0204,0.0234,0.0078,0.35,0.0021,0.0135,0.15,0.55,2.45,0.017,0.85,0.023,False,True
```
この行は：
- GPT-2モデル、gratitude感情、Layer 9、α=1.0での残差パッチング
- sentimentメトリクスが0.0043 → 0.0121に増加（平均差分+0.0078）
- Cohen's d = 0.35（中程度の効果）
- FDR補正後も有意（`significant_fdr = True`）

---

## 2. power_analysis.csv

### 目的
各実験条件における検出力（statistical power）を事後的に評価した結果です。
現在のサンプルサイズで、検出した効果量をどれだけ確実に検出できるかを示します。

### 列の説明

`effect_sizes.csv`の全列に加えて、以下の列が追加されます：

| 列名 | 型 | 説明 |
|------|-----|------|
| `power` | float/empty | 事後検出力（0-1、高いほど検出力が高い） |
| | | - 0.8以上: 十分な検出力 |
| | | - 0.5-0.8: 中程度の検出力 |
| | | - 0.5未満: 検出力不足 |

### データの読み方

- **検出力の解釈**: `power`が0.8以上で、効果が存在する場合に80%以上の確率で検出可能
- **検出力不足の場合**: `power < 0.8`なら、より大きなサンプルサイズが必要な可能性がある

---

## 3. power_analysis.json

### 目的
特定の効果量ターゲット（d=0.2, 0.5）と検出力目標（0.85）に対して、必要なサンプルサイズを計算した結果です。

### 構造

```json
{
  "profile": "baseline",
  "alpha": 0.05,
  "target_power": 0.85,
  "requirements": [
    {
      "profile": "baseline",
      "phase": "residual",
      "model_name": "gpt2",
      "metric_name": "sentiment",
      "paired": true,
      "target_effect_size": 0.2,
      "target_power": 0.85,
      "alpha": 0.05,
      "required_n_effective": 225.0
    },
    ...
  ],
  "post_hoc_power_csv": "results/baseline/statistics/power_analysis.csv"
}
```

### フィールドの説明

- `alpha`: 有意水準（通常0.05）
- `target_power`: 目標検出力（0.85 = 85%）
- `requirements`: 各メトリクス×効果量ターゲットごとの必要サンプルサイズ
  - `target_effect_size`: 検出したい効果量（Cohen's d）
  - `required_n_effective`: 必要な有効サンプル数

### データの読み方

- **d=0.2（小さい効果）を検出するには**: 通常225サンプル以上が必要
- **d=0.5（中程度の効果）を検出するには**: 通常36サンプル以上が必要
- 現在のサンプルサイズ（70）では、中程度以上の効果は検出可能だが、小さい効果の検出には不足の可能性がある

---

## 4. k_selection.csv

### 目的
サブスペース次元k（2, 5, 10, 20）ごとのモデル間overlapを統計的に比較し、最適なkを選択するための結果です。

### 列の説明

| 列名 | 型 | 説明 |
|------|-----|------|
| `profile` | string | データセットプロファイル |
| `model_a` | string | 第1モデル名（例: `gpt2`） |
| `model_b` | string | 第2モデル名（例: `EleutherAI-pythia-160m`） |
| `emotion` | string | 感情カテゴリ（`gratitude`, `anger`, `apology`） |
| `k` | int | サブスペース次元（2, 5, 10, 20） |
| `n_layers` | int | 対象層数（通常5層: 3, 5, 7, 9, 11） |
| `mean_overlap` | float | 平均overlap（cos²、0-1、高いほど類似） |
| `std_overlap` | float | overlapの標準偏差 |
| `ci_lower` | float | 95%信頼区間（下限） |
| `ci_upper` | float | 95%信頼区間（上限） |

### データの読み方

- **k=2でのoverlapが最も高い**: 低次元に感情のコア因子が存在する可能性
- **信頼区間**: `ci_lower`と`ci_upper`の範囲が狭いほど、結果が安定している

### 例
```
baseline,gpt2,EleutherAI-pythia-160m,gratitude,2,5,0.0027,0.0012,0.0019,0.0037
```
この行は：
- GPT-2とPythia-160Mのgratitudeサブスペース比較
- k=2での平均overlap = 0.0027
- 95%信頼区間: [0.0019, 0.0037]

---

## 5. k_sweep_raw.csv

### 目的
各層×感情×kの組み合わせごとの詳細なoverlapデータです。
`k_selection.csv`の集約前の生データです。

### 列の説明

| 列名 | 型 | 説明 |
|------|-----|------|
| `profile` | string | データセットプロファイル |
| `model_a` | string | 第1モデル名 |
| `model_b` | string | 第2モデル名 |
| `layer` | int | 層番号（3, 5, 7, 9, 11） |
| `emotion` | string | 感情カテゴリ |
| `k` | int | サブスペース次元（2, 5, 10, 20） |
| `overlap` | float | その層×感情×kでのoverlap（cos²） |
| `mean_principal_angle` | float | 主角度の平均（ラジアン、0-π/2、小さいほど類似） |

### データの読み方

- **層ごとの違い**: 深い層（9, 11）でoverlapが高い場合、高層で感情表現がより共通している
- **kごとの違い**: k=2でoverlapが高い場合、低次元にコア因子が存在

---

## 統計的検定の解釈ガイド

### p値の解釈
- `p_value < 0.05`: 統計的に有意（偶然では5%未満の確率でしか起きない）
- `p_value_bonferroni < 0.05`: Bonferroni補正後も有意（多重比較を考慮）
- `p_value_fdr < 0.05`: FDR補正後も有意（偽発見率を制御）

### 効果量の解釈（Cohen's d）
- `|d| < 0.2`: 実用的に無視できる効果
- `0.2 ≤ |d| < 0.5`: 小さい効果
- `0.5 ≤ |d| < 0.8`: 中程度の効果
- `|d| ≥ 0.8`: 大きい効果

### 検出力の解釈
- `power ≥ 0.8`: 十分な検出力（効果が存在する場合、80%以上の確率で検出可能）
- `0.5 ≤ power < 0.8`: 中程度の検出力
- `power < 0.5`: 検出力不足（効果があっても検出できない可能性が高い）

---

## 使用例

### Pythonでの読み込み例

```python
import pandas as pd

# 効果量分析結果の読み込み
effect_df = pd.read_csv('results/baseline/statistics/effect_sizes.csv')

# FDR補正後も有意な結果を抽出
significant = effect_df[effect_df['significant_fdr'] == True]

# 中程度以上の効果量を持つ結果を抽出
medium_effect = effect_df[
    (effect_df['cohen_d'].abs() >= 0.5) & 
    (effect_df['cohen_d'].abs() < 0.8)
]

# 検出力分析結果の読み込み
power_df = pd.read_csv('results/baseline/statistics/power_analysis.csv')

# 検出力不足の結果を抽出
low_power = power_df[
    (power_df['power'].notna()) & 
    (power_df['power'] < 0.8)
]

# k選択結果の読み込み
k_df = pd.read_csv('results/baseline/statistics/k_selection.csv')

# k=2でのoverlapが最も高い感情を抽出
k2_overlap = k_df[k_df['k'] == 2].sort_values('mean_overlap', ascending=False)
```

---

## 注意事項

1. **空欄の扱い**: 標準偏差が0の場合、Cohen's dやt検定は計算できず空欄になります
2. **多重比較**: 複数のメトリクス・条件を同時に検定しているため、`p_value_bonferroni`や`p_value_fdr`を使用してください
3. **検出力**: 現在のサンプルサイズ（70）では、小さい効果（d<0.2）の検出には不足の可能性があります
4. **再現性**: `--seed`オプションを使用した場合、ブートストラップ結果は完全に再現可能です

---

## 関連ファイル

- 生成スクリプト: `src/analysis/run_statistics.py`
- データローダー: `src/analysis/statistics/data_loading.py`
- 効果量計算: `src/analysis/statistics/effect_sizes.py`
- 検出力分析: `src/analysis/statistics/power_analysis.py`
- k選択: `src/analysis/statistics/k_selection.py`
```

このドキュメントを`results/baseline/statistics/README.md`として保存してください。生成AIが各CSVファイルの構造と意味を理解できるようになります。
