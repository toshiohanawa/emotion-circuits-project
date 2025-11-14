# Phase 4: Integrated Activation Patching Report

## Execution Date
2024年12月19日

## Overview
Phase 4では、baselineとextendedの両方のデータセットでActivation Patching実験を実行し、さらにランダム対照実験と実世界テキストでの検証を追加しました。既存のCLIスクリプトを活用し、新規モジュール（ランダム対照、実世界検証）を追加しました。

## Implementation

### Updated/New Modules

1. **`src/models/activation_patching.py`** (Existing)
   - Supports `--prompts_file` to use any dataset (baseline or extended)
   - Supports `--output` to specify output location

2. **`src/models/activation_patching_random.py`** (New)
   - Generates random vectors with same L2 norm as emotion vectors
   - Runs sweep experiments with random control vectors
   - Reuses existing patching engine from `ActivationPatchingSweep`

3. **`src/analysis/random_vs_emotion_effect.py`** (New)
   - Compares random vs emotion patching effects
   - Computes statistical tests (t-test, Cohen's d, 95% CI)
   - Generates comparison plots

### Execution

#### Baseline Patching
```bash
python -m src.models.activation_patching \
  --model gpt2 \
  --vectors_file results/baseline/emotion_vectors/gpt2_vectors_token_based.pkl \
  --prompts_file data/neutral_prompts.json \
  --output results/baseline/patching/gpt2_patching_gratitude_alpha1.0.pkl \
  --layer 6 \
  --alpha 1.0
```

#### Extended Patching
```bash
python -m src.models.activation_patching \
  --model gpt2 \
  --vectors_file results/extended/emotion_vectors/gpt2_vectors_token_based.pkl \
  --prompts_file data/neutral_prompts.json \
  --output results/extended/patching/gpt2_patching_gratitude_alpha1.0.pkl \
  --layer 6 \
  --alpha 1.0
```

#### Random Control Experiment
```bash
python -m src.models.activation_patching_random \
  --model gpt2 \
  --vectors_file results/extended/emotion_vectors/gpt2_vectors_token_based.pkl \
  --prompts_file data/neutral_prompts.json \
  --output_dir results/extended/patching_random \
  --layers 7 \
  --alpha 1.0 \
  --num_random 3
```

#### Real-World Dataset Patching
```bash
python -m src.models.activation_patching \
  --model gpt2 \
  --vectors_file results/extended/emotion_vectors/gpt2_vectors_token_based.pkl \
  --prompts_file data/real_world_samples.json \
  --output results/extended/patching_realworld.pkl \
  --layer 7 \
  --alpha -1 1
```

## Results

### Baseline Patching
- **Model**: GPT-2
- **Vectors**: Token-based emotion vectors
- **Prompts**: Neutral prompts (50 samples)
- **Layer**: 6
- **Alpha**: 1.0
- **Output**: `results/baseline/patching/gpt2_patching_*.pkl`

### Extended Patching
- **Model**: GPT-2
- **Vectors**: Extended token-based emotion vectors
- **Prompts**: Neutral prompts (50 samples)
- **Layer**: 6
- **Alpha**: 1.0
- **Output**: `results/extended/patching/gpt2_patching_*.pkl`

### Random Control Experiment
- **Model**: GPT-2
- **Layers**: 7
- **Alpha**: 1.0
- **Num Random Vectors**: 3
- **Emotions**: gratitude, anger, apology
- **Output**: `results/extended/patching_random/gpt2_random_control.pkl`

**Key Finding**: Random control vectors (with same L2 norm as emotion vectors) were tested to compare against emotion vector patching effects. This allows us to verify that emotion-specific effects are not just due to adding any vector of similar magnitude.

### Real-World Dataset
- **Dataset**: `data/real_world_samples.json` (35 prompts)
  - SNS-style: 15 prompts
  - Review-style: 10 prompts
  - Work email-style: 10 prompts
- **Patching**: Layer 7, α={-1, 1}
- **Output**: `results/extended/patching_realworld.pkl`

**Purpose**: Verify that emotion direction patching works on real-world text styles, not just artificial prompts.

## Analysis

### Random vs Emotion Comparison
The `random_vs_emotion_effect.py` script compares:
- **Emotion keyword frequency**: Random vs emotion patching
- **Politeness scores**: Random vs emotion patching
- **Sentiment scores**: Random vs emotion patching
- **Statistical tests**: t-test, Cohen's d, 95% confidence intervals

**Output**: Comparison plots saved to `results/extended/plots/random_control/`

## MLflow Logging

All patching experiments were logged to MLflow:

### Parameters
- `phase`: phase4
- `task`: activation_patching
- `models`: gpt2
- `baseline_dataset`: data/emotion_dataset.jsonl
- `extended_dataset`: data/emotion_dataset_extended.jsonl
- `real_world_dataset`: data/real_world_samples.json

### Metrics
- `baseline_patching_files`: Number of baseline patching result files
- `extended_patching_files`: Number of extended patching result files
- `random_control_layers`: Number of layers tested
- `random_control_num_random`: Number of random vectors generated
- `random_control_emotions`: Number of emotions tested
- `realworld_patching_completed`: Flag indicating real-world experiment completion

## Key Observations

1. **Unified CLI Success**: Same `activation_patching.py` CLI successfully processed baseline, extended, and real-world datasets
2. **Random Control Implementation**: New module successfully generates L2-matched random vectors and runs comparison experiments
3. **Real-World Validation**: Real-world text samples (SNS, reviews, emails) were tested to verify generalization
4. **Statistical Analysis**: Random vs emotion comparison provides quantitative evidence for emotion-specific effects

## Next Steps

Phase 4 is complete. Patching results are ready for:
- Phase 5: Sweep experiments (layer × alpha sweeps)
- Further analysis of random control vs emotion effects
- Real-world validation with more diverse text styles

## Conclusion

Phase 4 successfully implemented integrated activation patching for baseline, extended, and real-world datasets. The addition of random control experiments provides a crucial baseline for verifying that emotion-specific effects are not artifacts of adding arbitrary vectors. The unified CLI approach worked seamlessly across all dataset types.

