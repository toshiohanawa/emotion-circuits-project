# Phase 3.5: Integrated Token-Based & Subspace Analysis Report

## Execution Date
2024年12月19日

## Overview
Phase 3.5では、baselineとextendedの両方のデータセットに対して、token-based感情ベクトル抽出とサブスペース解析を行いました。文末ベースのベクトルと比較し、より適切な感情表現の抽出方法を検証しました。

## Implementation

### CLI Usage
The existing CLIs support dataset-aware workflows:
- `src/analysis/emotion_vectors_token_based.py`: `--activations_dir`, `--output`
- `src/analysis/emotion_subspace.py`: `--activations_dir`, `--output`, `--n-components`
- `src/analysis/cross_model_subspace.py`: `--subspaces_dir`, `--models`, `--output_table`

### Execution

#### Baseline Dataset
```bash
# Token-based vectors
python -m src.analysis.emotion_vectors_token_based --activations_dir results/baseline/activations/gpt2 --output results/baseline/emotion_vectors/gpt2_vectors_token_based.pkl
# ... (repeated for all models)

# Subspace analysis
python -m src.analysis.emotion_subspace --activations_dir results/baseline/activations/gpt2 --output results/baseline/emotion_subspaces/gpt2_subspaces.pkl --n-components 10
# ... (repeated for all models)

# Cross-model subspace overlap
python -m src.analysis.cross_model_subspace --subspaces_dir results/baseline/emotion_subspaces --models gpt2 EleutherAI-pythia-160m EleutherAI-gpt-neo-125M --output_table results/baseline/cross_model_subspace_overlap.csv
```

#### Extended Dataset
```bash
# Same commands but with extended paths
python -m src.analysis.emotion_vectors_token_based --activations_dir results/extended/activations/gpt2 --output results/extended/emotion_vectors/gpt2_vectors_token_based.pkl
# ... (repeated for all models and subspace analysis)
```

## Results

### Token-Based Emotion Vectors

#### Baseline Dataset

**GPT-2**:
- gratitude vs anger: 0.1480 (much lower than sentence-end: 0.4817)
- gratitude vs apology: 0.3592 (lower than sentence-end: 0.6856)
- anger vs apology: 0.7031 (similar to sentence-end: 0.7473)

**Pythia-160M**:
- gratitude vs anger: -0.7573 (negative! Much lower than sentence-end: 0.9753)
- gratitude vs apology: -0.7259 (negative! Much lower than sentence-end: 0.9832)
- anger vs apology: 0.9667 (similar to sentence-end: 0.9884)
- **Key Finding**: Token-based method reveals that gratitude and anger/apology are actually opposite directions in Pythia-160M

**GPT-Neo-125M**:
- gratitude vs anger: 0.0847 (much lower than sentence-end: 0.5914)
- gratitude vs apology: 0.2769 (much lower than sentence-end: 0.7441)
- anger vs apology: 0.7656 (similar to sentence-end: 0.8097)

#### Extended Dataset

**GPT-2**:
- gratitude vs anger: 0.1383
- gratitude vs apology: 0.3166
- anger vs apology: 0.7344

**Pythia-160M**:
- gratitude vs anger: -0.7750
- gratitude vs apology: -0.7364
- anger vs apology: 0.9502

**GPT-Neo-125M**:
- gratitude vs anger: 0.0772
- gratitude vs apology: 0.1909
- anger vs apology: 0.7935

**Key Finding**: Token-based vectors show more distinct emotion representations compared to sentence-end vectors, especially for Pythia-160M where gratitude is opposite to anger/apology.

### Subspace Analysis

#### Baseline Dataset - Intra-Model Overlaps

**GPT-2**:
- gratitude vs anger: 0.7262
- gratitude vs apology: 0.7810
- anger vs apology: 0.7941
- **Explained Variance** (top 5 components): ~0.14-0.15

**Pythia-160M**:
- gratitude vs anger: 0.8071
- gratitude vs apology: 0.8245
- anger vs apology: 0.8376
- **Explained Variance**: ~0.199 (very high, suggesting concentrated representations)

**GPT-Neo-125M**:
- gratitude vs anger: 0.8082
- gratitude vs apology: 0.8424
- anger vs apology: 0.8565
- **Explained Variance**: ~0.15-0.16

#### Extended Dataset - Intra-Model Overlaps

**GPT-2**:
- gratitude vs anger: 0.7579
- gratitude vs apology: 0.8090
- anger vs apology: 0.8175
- **Explained Variance**: ~0.13-0.14

**Pythia-160M**:
- gratitude vs anger: 0.8343
- gratitude vs apology: 0.8578
- anger vs apology: 0.8563
- **Explained Variance**: ~0.199

**GPT-Neo-125M**:
- gratitude vs anger: 0.8384
- gratitude vs apology: 0.8651
- anger vs apology: 0.8694
- **Explained Variance**: ~0.14-0.16

### Cross-Model Subspace Overlaps

#### Baseline Dataset
| Emotion | Model Pair | Avg Overlap | Std | Min | Max |
|---------|-----------|-------------|-----|-----|-----|
| gratitude | gpt2 ↔ pythia-160m | 0.147195 | 0.006737 | 0.136349 | 0.157806 |
| gratitude | gpt2 ↔ gpt-neo-125M | 0.136662 | 0.006273 | 0.128631 | 0.148510 |
| gratitude | pythia-160m ↔ gpt-neo-125M | 0.138091 | 0.009677 | 0.120038 | 0.155427 |
| anger | gpt2 ↔ pythia-160m | 0.146956 | 0.013512 | 0.130123 | 0.181759 |
| anger | gpt2 ↔ gpt-neo-125M | 0.147716 | 0.007739 | 0.134670 | 0.157940 |
| anger | pythia-160m ↔ gpt-neo-125M | 0.141138 | 0.015671 | 0.108540 | 0.161293 |
| apology | gpt2 ↔ pythia-160m | 0.154639 | 0.008417 | 0.135703 | 0.164998 |
| apology | gpt2 ↔ gpt-neo-125M | 0.150310 | 0.011938 | 0.130789 | 0.167635 |
| apology | pythia-160m ↔ gpt-neo-125M | 0.139979 | 0.006808 | 0.127756 | 0.154706 |

**Key Finding**: Cross-model subspace overlaps are **0.13-0.15**, which is **higher than random baseline (0.0-0.1)**, suggesting shared subspace structure across models despite different coordinate systems.

#### Extended Dataset
| Emotion | Model Pair | Avg Overlap | Std | Min | Max |
|---------|-----------|-------------|-----|-----|-----|
| gratitude | gpt2 ↔ pythia-160m | 0.145165 | 0.006527 | 0.132314 | 0.156542 |
| gratitude | gpt2 ↔ gpt-neo-125M | 0.139282 | 0.008552 | 0.130629 | 0.163578 |
| gratitude | pythia-160m ↔ gpt-neo-125M | 0.138087 | 0.009935 | 0.118889 | 0.150878 |
| anger | gpt2 ↔ pythia-160m | 0.147159 | 0.014612 | 0.120716 | 0.180033 |
| anger | gpt2 ↔ gpt-neo-125M | 0.144281 | 0.009513 | 0.118229 | 0.154779 |
| anger | pythia-160m ↔ gpt-neo-125M | 0.139547 | 0.015343 | 0.109579 | 0.162867 |
| apology | gpt2 ↔ pythia-160m | 0.151474 | 0.008757 | 0.135242 | 0.168743 |
| apology | gpt2 ↔ gpt-neo-125M | 0.149206 | 0.008118 | 0.134705 | 0.165975 |
| apology | pythia-160m ↔ gpt-neo-125M | 0.142753 | 0.006028 | 0.129199 | 0.150097 |

**Key Finding**: Extended dataset shows similar cross-model subspace overlaps (0.13-0.15), confirming the robustness of the finding that models share subspace structure.

## MLflow Logging

All metrics were logged to MLflow:

### Parameters
- `phase`: phase3.5
- `task`: token_based_subspace_analysis
- `models`: ['gpt2', 'EleutherAI-pythia-160m', 'EleutherAI-gpt-neo-125M']
- `n_components`: 10

### Metrics
- Token-based intra-model similarities for baseline and extended
- Subspace intra-model overlaps for baseline and extended
- Cross-model subspace overlaps for baseline and extended

## Key Observations

1. **Token-Based Vectors Reveal More Distinct Representations**: Token-based method shows much lower (or negative) similarities compared to sentence-end method, especially for Pythia-160M where gratitude is opposite to anger/apology.

2. **Cross-Model Subspace Overlaps Are Above Random**: Overlaps of 0.13-0.15 (vs random 0.0-0.1) suggest shared subspace structure across models, supporting the hypothesis that "coordinate systems differ but underlying structure is similar."

3. **Extended Dataset Confirms Findings**: Extended dataset shows similar patterns, indicating robustness of the subspace overlap findings.

4. **Pythia-160M High Explained Variance**: Pythia-160M shows very high explained variance (~0.199), suggesting more concentrated emotion representations.

## Next Steps

Phase 3.5 is complete. Token-based vectors and subspace analyses are ready for:
- Phase 4: Activation patching experiments (using token-based vectors)
- Phase 6: Subspace alignment experiments

## Conclusion

Phase 3.5 successfully extracted token-based emotion vectors and performed subspace analysis on both baseline and extended datasets. The key finding is that **cross-model subspace overlaps (0.13-0.15) are above random baseline**, suggesting shared subspace structure across models despite different coordinate systems. This finding is consistent across both baseline and extended datasets, indicating robustness.

