# Phase 2: Integrated Activation Extraction Report

## Execution Date
2024年12月19日

## Overview
Phase 2では、baselineとextendedの両方のデータセットから、3つのモデル（GPT-2, Pythia-160M, GPT-Neo-125M）の内部活性を統合的に抽出しました。既存のCLIスクリプト（`extract_activations.py`）が`--dataset`と`--output`パラメータをサポートしているため、両方のデータセットを同じワークフローで処理できました。

## Implementation

### CLI Usage
The existing `src/models/extract_activations.py` CLI already supports:
- `--model`: Model name (required)
- `--dataset`: Dataset path (default: `data/emotion_dataset.jsonl`)
- `--output`: Output directory (required)
- `--emotion`: Emotion label (optional, filters dataset)

### Execution

#### Baseline Dataset Activations
```bash
# GPT-2
python -m src.models.extract_activations --model gpt2 --dataset data/emotion_dataset.jsonl --output results/baseline/activations/gpt2 --emotion gratitude
python -m src.models.extract_activations --model gpt2 --dataset data/emotion_dataset.jsonl --output results/baseline/activations/gpt2 --emotion anger
python -m src.models.extract_activations --model gpt2 --dataset data/emotion_dataset.jsonl --output results/baseline/activations/gpt2 --emotion apology
python -m src.models.extract_activations --model gpt2 --dataset data/emotion_dataset.jsonl --output results/baseline/activations/gpt2 --emotion neutral

# Pythia-160M
python -m src.models.extract_activations --model EleutherAI/pythia-160m --dataset data/emotion_dataset.jsonl --output results/baseline/activations/EleutherAI-pythia-160m --emotion {emotion}

# GPT-Neo-125M
python -m src.models.extract_activations --model EleutherAI/gpt-neo-125M --dataset data/emotion_dataset.jsonl --output results/baseline/activations/EleutherAI-gpt-neo-125M --emotion {emotion}
```

#### Extended Dataset Activations
```bash
# Same commands but with extended dataset and output directory
python -m src.models.extract_activations --model gpt2 --dataset data/emotion_dataset_extended.jsonl --output results/extended/activations/gpt2 --emotion {emotion}
# ... (repeated for all models and emotions)
```

## Results

### Baseline Dataset
- **Models**: GPT-2, Pythia-160M, GPT-Neo-125M
- **Emotions**: gratitude, anger, apology, neutral (4 categories)
- **Total Files**: 12 files (3 models × 4 emotions)
- **Samples per File**: 70 samples
- **Total Samples Processed**: 840 samples (3 models × 4 emotions × 70 samples)
- **Output Location**: `results/baseline/activations/{model}/activations_{emotion}.pkl`

### Extended Dataset
- **Models**: GPT-2, Pythia-160M, GPT-Neo-125M
- **Emotions**: gratitude, anger, apology, neutral (4 categories)
- **Total Files**: 12 files (3 models × 4 emotions)
- **Samples per File**: 100 samples
- **Total Samples Processed**: 1,200 samples (3 models × 4 emotions × 100 samples)
- **Output Location**: `results/extended/activations/{model}/activations_{emotion}.pkl`

### Activation Data Structure
Each activation file contains:
- **Residual stream layers**: 12 layers (for all models)
- **MLP output layers**: 12 layers
- **Attention layers**: 0 (not saved by default)

## Processing Performance

- **Processing Speed**: ~60-63 samples/second (consistent across models)
- **Total Processing Time**: 
  - Baseline: ~14 seconds per model×emotion combination
  - Extended: ~20 seconds per model×emotion combination
- **Total Time**: ~2-3 minutes for all baseline extractions, ~3-4 minutes for all extended extractions

## MLflow Logging

All activation extraction metrics were logged to MLflow:

### Parameters
- `phase`: phase2
- `task`: activation_extraction
- `models`: ['gpt2', 'EleutherAI-pythia-160m', 'EleutherAI-gpt-neo-125M']
- `emotions`: ['gratitude', 'anger', 'apology', 'neutral']

### Metrics
- Individual file sizes (MB) for each model×emotion×dataset combination
- `baseline_total_files`: 12
- `baseline_total_size_mb`: Total size of all baseline activation files
- `extended_total_files`: 12
- `extended_total_size_mb`: Total size of all extended activation files

## Key Observations

1. **Unified Workflow**: The same CLI successfully processed both baseline and extended datasets without modification
2. **Consistent Performance**: Processing speed was consistent across models (~60 samples/second)
3. **Complete Coverage**: All 3 models × 4 emotions × 2 datasets = 24 activation files successfully created
4. **Proper Isolation**: Extended activations stored in `results/extended/` separate from baseline `results/baseline/`

## Next Steps

Phase 2 is complete. Activation files are ready for:
- Phase 3: Emotion vector extraction (sentence-end based)
- Phase 3.5: Token-based vectors and subspace analysis
- All subsequent phases that require activation data

## Conclusion

Phase 2 successfully extracted activations from both baseline (280 samples) and extended (400 samples) datasets across all 3 models. The unified CLI approach worked seamlessly, demonstrating that the existing codebase already supports dataset-aware workflows. All activation files are properly organized and ready for downstream analysis.
