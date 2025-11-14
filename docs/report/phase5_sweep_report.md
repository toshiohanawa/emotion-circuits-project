# Phase 5: Integrated Sweep Experiments Report

## Execution Date
2024年12月19日

## Overview
Phase 5では、baselineとextendedの両方のデータセットでActivation Patching Sweep実験を実行しました。層×αのスイープ実験により、感情方向パッチングの効果が層と強度にどのように依存するかを検証しました。

## Implementation

### CLI Usage
The existing `src/models/activation_patching_sweep.py` CLI supports:
- `--model`: Model name
- `--vectors_file`: Emotion vectors file (token-based)
- `--prompts_file`: Prompts file (JSON)
- `--output`: Output file path
- `--layers`: Layer indices (list)
- `--alpha`: Alpha values (list)

### Execution

#### Baseline Sweep
```bash
python -m src.models.activation_patching_sweep \
  --model gpt2 \
  --vectors_file results/baseline/emotion_vectors/gpt2_vectors_token_based.pkl \
  --prompts_file data/neutral_prompts.json \
  --output results/baseline/patching/gpt2_sweep_token_based.pkl \
  --layers 3 5 7 9 11 \
  --alpha -2 -1 -0.5 0 0.5 1 2
```

**Configuration**:
- **Layers**: 3, 5, 7, 9, 11 (5 layers)
- **Alpha values**: -2, -1, -0.5, 0, 0.5, 1, 2 (7 values)
- **Emotions**: gratitude, anger, apology (3 emotions)
- **Total combinations**: 5 layers × 7 alpha × 3 emotions = 105 combinations

#### Extended Sweep (Limited)
```bash
python -m src.models.activation_patching_sweep \
  --model gpt2 \
  --vectors_file results/extended/emotion_vectors/gpt2_vectors_token_based.pkl \
  --prompts_file data/neutral_prompts.json \
  --output results/extended/patching_sweep_recheck.pkl \
  --layers 5 7 \
  --alpha -1 1
```

**Configuration** (Limited for efficiency):
- **Layers**: 5, 7 (2 layers)
- **Alpha values**: -1, 1 (2 values)
- **Emotions**: gratitude, anger (2 emotions, apology also tested)
- **Total combinations**: 2 layers × 2 alpha × 3 emotions = 12 combinations

**Rationale**: Extended sweep is limited to representative conditions to verify that findings from baseline dataset hold with extended dataset, while keeping computational cost manageable.

## Results

### Baseline Sweep
- **Output**: `results/baseline/patching/gpt2_sweep_token_based.pkl`
- **Coverage**: Full sweep across 5 layers × 7 alpha values × 3 emotions
- **Metrics Computed**:
  - Emotion keyword frequency (gratitude, anger, apology)
  - Politeness scores
  - Sentiment scores
  - Baseline outputs (no patching)

### Extended Sweep
- **Output**: `results/extended/patching_sweep_recheck.pkl`
- **Coverage**: Limited sweep across 2 layers × 2 alpha values × 3 emotions
- **Purpose**: Verify that extended dataset shows similar patterns to baseline

## Analysis

### Sweep Results Structure
Each sweep result file contains:
- `model`: Model name
- `prompts`: List of input prompts
- `layers`: List of layer indices tested
- `alpha_values`: List of alpha values tested
- `emotions`: List of emotions tested
- `baseline`: Baseline outputs and metrics (no patching)
- `sweep_results`: Nested dictionary structure:
  ```
  sweep_results[emotion][layer][alpha] = {
      'outputs': {prompt: generated_text},
      'metrics': {prompt: {emotion_keywords, politeness, sentiment}}
  }
  ```

### Key Metrics Tracked
1. **Emotion Keywords**: Frequency of emotion-related words in generated text
2. **Politeness**: Politeness score (0-1) based on politeness markers
3. **Sentiment**: Sentiment score (-1 to 1) based on positive/negative words

## MLflow Logging

All sweep experiments were logged to MLflow:

### Parameters
- `phase`: phase5
- `task`: activation_patching_sweep
- `model`: gpt2

### Metrics
- `baseline_sweep_layers`: 5
- `baseline_sweep_alpha_values`: 7
- `baseline_sweep_emotions`: 3
- `extended_sweep_layers`: 2
- `extended_sweep_alpha_values`: 2
- `extended_sweep_emotions`: 3

## Key Observations

1. **Full Baseline Sweep**: Complete coverage of layer × alpha space for comprehensive analysis
2. **Limited Extended Sweep**: Representative conditions tested to verify robustness without excessive computation
3. **Unified CLI**: Same sweep CLI successfully processed both baseline and extended datasets
4. **Comprehensive Metrics**: Each sweep combination includes emotion keywords, politeness, and sentiment metrics

## Next Steps

Phase 5 is complete. Sweep results are ready for:
- Visualization (heatmaps showing layer × alpha effects)
- Comparison between baseline and extended results
- Phase 6: Subspace alignment experiments

## Conclusion

Phase 5 successfully executed integrated sweep experiments for both baseline and extended datasets. The baseline sweep provides comprehensive coverage of the layer × alpha space, while the extended sweep verifies robustness with a limited but representative set of conditions. All results are properly structured and ready for downstream analysis and visualization.

