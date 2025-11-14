# Phase 1: Integrated Dataset Construction Report

## Execution Date
2024年12月19日

## Overview
Phase 1では、baselineとextendedの両方のデータセットを統合的に構築しました。既存のCLIスクリプトを拡張し、両方のデータセットを同じワークフローで作成できるようにしました。

## Implementation

### Updated Scripts

1. **`src/data/create_individual_prompt_files.py`**
   - Extended to create all 4 emotion categories (gratitude, anger, apology, neutral)
   - Added `--extended` flag to create extended version files
   - Added CLI arguments for flexibility

2. **`src/data/build_dataset.py`** (New)
   - New CLI that takes multiple prompt JSON files and builds a JSONL dataset
   - Supports automatic emotion label detection from filenames
   - Calculates and reports dataset statistics

### Dataset Creation

#### Baseline Dataset
- **Source**: `data/{emotion}_prompts.json` (70 prompts per category)
- **Output**: `data/emotion_dataset.jsonl`
- **Total Samples**: 280 (70 per emotion category)
- **Text Length**: avg=27.7, min=6, max=50 characters
- **Word Count**: avg=4.8, min=1, max=9 words

#### Extended Dataset
- **Source**: `data/{emotion}_prompts_extended.json` (100 prompts per category)
- **Output**: `data/emotion_dataset_extended.jsonl`
- **Total Samples**: 400 (100 per emotion category)
- **Text Length**: avg=33.0, min=6, max=71 characters
- **Word Count**: Higher diversity in sentence length

### Execution Commands

```bash
# Create baseline prompt files
python -m src.data.create_individual_prompt_files --data_dir data

# Build baseline dataset
python -m src.data.build_dataset \
  --prompts data/gratitude_prompts.json data/anger_prompts.json \
            data/apology_prompts.json data/neutral_prompts.json \
  --output data/emotion_dataset.jsonl

# Create extended prompt files (with additional 30 prompts per category)
# (Created via Python script that extends EMOTION_PROMPTS)

# Build extended dataset
python -m src.data.build_dataset \
  --prompts data/gratitude_prompts_extended.json data/anger_prompts_extended.json \
            data/apology_prompts_extended.json data/neutral_prompts_extended.json \
  --output data/emotion_dataset_extended.jsonl

# Validate both datasets
python -m src.data.validate_dataset data/emotion_dataset.jsonl
python -m src.data.validate_dataset data/emotion_dataset_extended.jsonl
```

## Dataset Statistics

### Baseline Dataset
- **Total Samples**: 280
- **Emotion Distribution**:
  - anger: 70 samples (25.0%)
  - apology: 70 samples (25.0%)
  - gratitude: 70 samples (25.0%)
  - neutral: 70 samples (25.0%)
- **Language**: 100% English
- **Text Length**: Average 27.7 characters (range: 6-50)

### Extended Dataset
- **Total Samples**: 400
- **Emotion Distribution**:
  - anger: 100 samples (25.0%)
  - apology: 100 samples (25.0%)
  - gratitude: 100 samples (25.0%)
  - neutral: 100 samples (25.0%)
- **Language**: 100% English
- **Text Length**: Average 33.0 characters (range: 6-71)

## MLflow Logging

All dataset statistics were logged to MLflow in a single Phase 1 run:

### Parameters
- `phase`: phase1
- `task`: dataset_construction

### Metrics
- `total_samples_baseline`: 280
- `total_samples_extended`: 400
- `avg_length_baseline`: 27.7
- `avg_length_extended`: 33.0
- `min_length_baseline`: 6
- `min_length_extended`: 6
- `max_length_baseline`: 50
- `max_length_extended`: 71
- Per-emotion counts for both baseline and extended datasets

## Key Improvements

1. **Unified Workflow**: Both baseline and extended datasets are created using the same CLI tools
2. **Extended Prompts**: Added 30 more prompts per category to reach 100 prompts for extended dataset
3. **Flexible CLI**: `build_dataset.py` can handle any combination of prompt files
4. **Comprehensive Logging**: All statistics logged to MLflow for tracking

## Next Steps

Phase 1 is complete. Both baseline and extended datasets are ready for:
- Phase 2: Activation extraction (can use either dataset via `--dataset` parameter)
- All subsequent phases with dataset-aware CLIs

## Conclusion

Phase 1 successfully created both baseline (280 samples) and extended (400 samples) datasets using unified CLI tools. The extended dataset provides more diversity and robustness for downstream experiments while maintaining the same structure and quality standards.

