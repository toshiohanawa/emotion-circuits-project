# Phase 3: Integrated Emotion Vectors & Visualization Report

## Execution Date
2024年12月19日

## Overview
Phase 3では、baselineとextendedの両方のデータセットから感情方向ベクトルを抽出し、可視化とモデル間比較を行いました。既存のCLIスクリプトが`--activations_dir`と`--output`パラメータをサポートしているため、両方のデータセットを同じワークフローで処理できました。

## Implementation

### CLI Usage
The existing CLIs support dataset-aware workflows:
- `src/analysis/emotion_vectors.py`: `--activations_dir`, `--output`
- `src/visualization/emotion_plots.py`: `--vectors_file`, `--output_dir`
- `src/analysis/cross_model_analysis.py`: `--vectors_dir`, `--models`, `--output_dir`, `--output_table`

### Execution

#### Baseline Dataset
```bash
# Extract emotion vectors
python -m src.analysis.emotion_vectors --activations_dir results/baseline/activations/gpt2 --output results/baseline/emotion_vectors/gpt2_vectors.pkl
# ... (repeated for all models)

# Create visualizations
python -m src.visualization.emotion_plots --vectors_file results/baseline/emotion_vectors/gpt2_vectors.pkl --output_dir results/baseline/plots/gpt2
# ... (repeated for all models)

# Cross-model analysis
python -m src.analysis.cross_model_analysis \
  --vectors_dir results/baseline/emotion_vectors \
  --models gpt2 EleutherAI-pythia-160m EleutherAI-gpt-neo-125M \
  --output_dir results/baseline/plots/cross_model \
  --output_table results/baseline/cross_model_similarity.csv
```

#### Extended Dataset
```bash
# Same commands but with extended paths
python -m src.analysis.emotion_vectors --activations_dir results/extended/activations/gpt2 --output results/extended/emotion_vectors/gpt2_vectors.pkl
# ... (repeated for all models and visualization/analysis)
```

## Results

### Baseline Dataset - Emotion Vectors

#### GPT-2
- **Vector Shapes**: (12 layers, 768 dimensions) for each emotion
- **Intra-Model Similarities**:
  - gratitude vs anger: 0.4817
  - gratitude vs apology: 0.6856
  - anger vs apology: 0.7473

#### Pythia-160M
- **Vector Shapes**: (12 layers, 768 dimensions) for each emotion
- **Intra-Model Similarities**:
  - gratitude vs anger: 0.9753
  - gratitude vs apology: 0.9832
  - anger vs apology: 0.9884
- **Note**: Very high similarities suggest less distinct emotion representations

#### GPT-Neo-125M
- **Vector Shapes**: (12 layers, 768 dimensions) for each emotion
- **Intra-Model Similarities**:
  - gratitude vs anger: 0.5914
  - gratitude vs apology: 0.7441
  - anger vs apology: 0.8097

### Extended Dataset - Emotion Vectors

#### GPT-2
- **Intra-Model Similarities**:
  - gratitude vs anger: 0.4936
  - gratitude vs apology: 0.6900
  - anger vs apology: 0.7604

#### Pythia-160M
- **Intra-Model Similarities**:
  - gratitude vs anger: 0.9614
  - gratitude vs apology: 0.9637
  - anger vs apology: 0.9714
- **Note**: Still very high similarities, but slightly lower than baseline

#### GPT-Neo-125M
- **Intra-Model Similarities**:
  - gratitude vs anger: 0.6030
  - gratitude vs apology: 0.7151
  - anger vs apology: 0.8155

### Cross-Model Similarities

#### Baseline Dataset
| Emotion | Model Pair | Avg Similarity | Std | Min | Max |
|---------|-----------|----------------|-----|-----|-----|
| gratitude | gpt2 ↔ pythia-160m | 0.000146 | 0.018545 | -0.038390 | 0.022893 |
| gratitude | gpt2 ↔ gpt-neo-125M | -0.001770 | 0.026768 | -0.033639 | 0.051921 |
| gratitude | pythia-160m ↔ gpt-neo-125M | -0.005124 | 0.024486 | -0.050876 | 0.028768 |
| anger | gpt2 ↔ pythia-160m | 0.005486 | 0.021417 | -0.032401 | 0.041708 |
| anger | gpt2 ↔ gpt-neo-125M | -0.010037 | 0.020262 | -0.044485 | 0.021601 |
| anger | pythia-160m ↔ gpt-neo-125M | -0.002478 | 0.021921 | -0.048309 | 0.027654 |
| apology | gpt2 ↔ pythia-160m | 0.003770 | 0.016896 | -0.031789 | 0.027198 |
| apology | gpt2 ↔ gpt-neo-125M | -0.001210 | 0.020651 | -0.040054 | 0.031777 |
| apology | pythia-160m ↔ gpt-neo-125M | -0.009837 | 0.023524 | -0.044504 | 0.042619 |

**Key Finding**: Cross-model similarities are near zero (range: -0.01 to 0.01), indicating that emotion vectors are model-specific and not directly comparable across models.

#### Extended Dataset
| Emotion | Model Pair | Avg Similarity | Std | Min | Max |
|---------|-----------|----------------|-----|-----|-----|
| gratitude | gpt2 ↔ pythia-160m | -0.000277 | 0.017152 | -0.035811 | 0.026582 |
| gratitude | gpt2 ↔ gpt-neo-125M | 0.004921 | 0.025244 | -0.024131 | 0.058774 |
| gratitude | pythia-160m ↔ gpt-neo-125M | 0.008009 | 0.030368 | -0.050566 | 0.039777 |
| anger | gpt2 ↔ pythia-160m | 0.002744 | 0.017423 | -0.023418 | 0.030511 |
| anger | gpt2 ↔ gpt-neo-125M | 0.003337 | 0.016517 | -0.036945 | 0.026966 |
| anger | pythia-160m ↔ gpt-neo-125M | 0.009069 | 0.027305 | -0.047246 | 0.042097 |
| apology | gpt2 ↔ pythia-160m | 0.000754 | 0.013524 | -0.026363 | 0.023306 |
| apology | gpt2 ↔ gpt-neo-125M | 0.004378 | 0.023278 | -0.035457 | 0.033673 |
| apology | pythia-160m ↔ gpt-neo-125M | -0.000618 | 0.026744 | -0.044265 | 0.059038 |

**Key Finding**: Extended dataset shows similar near-zero cross-model similarities, confirming the model-specific nature of emotion representations.

## Visualizations Generated

### Baseline Dataset
- **Per-Model Plots** (3 models × 3 plots = 9 files):
  - Layer norms (L2) plots
  - Emotion distance plots
  - Similarity heatmaps
- **Cross-Model Plots** (3 emotions × 1 plot = 3 files):
  - Cross-model similarity visualizations

### Extended Dataset
- **Per-Model Plots** (3 models × 3 plots = 9 files):
  - Layer norms (L2) plots
  - Emotion distance plots
  - Similarity heatmaps
- **Cross-Model Plots** (3 emotions × 1 plot = 3 files):
  - Cross-model similarity visualizations

## MLflow Logging

All metrics were logged to MLflow:

### Parameters
- `phase`: phase3
- `task`: emotion_vectors_extraction
- `models`: ['gpt2', 'EleutherAI-pythia-160m', 'EleutherAI-gpt-neo-125M']

### Metrics
- Per-layer L2 norms for each model×emotion×dataset combination
- Intra-model emotion similarities (gratitude vs anger, gratitude vs apology, anger vs apology)
- Cross-model similarities for all emotion×model pair combinations

## Key Observations

1. **Model-Specific Representations**: Cross-model similarities are near zero, confirming that emotion vectors are model-specific
2. **Pythia-160M High Similarities**: Pythia-160M shows very high intra-model similarities (0.96-0.99), suggesting less distinct emotion representations
3. **Extended Dataset Consistency**: Extended dataset shows similar patterns to baseline, indicating robustness of findings
4. **Unified Workflow Success**: Same CLIs successfully processed both datasets without modification

## Next Steps

Phase 3 is complete. Emotion vectors are ready for:
- Phase 3.5: Token-based vectors and subspace analysis
- Phase 4: Activation patching experiments
- All subsequent phases requiring emotion vectors

## Conclusion

Phase 3 successfully extracted emotion vectors from both baseline and extended datasets across all 3 models. The unified CLI approach worked seamlessly, and the results confirm the model-specific nature of emotion representations. Cross-model similarities remain near zero, suggesting that direct comparison requires alignment techniques (Phase 6).

