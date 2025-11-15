"""
Lightweight consistency checks between paper claims and repo artifacts.
Current checks:
- Dataset sizes for baseline/extended JSONL
- Presence of key result files (emotion_vectors, subspace overlaps)
- CLI argument presence for major scripts
Exit non-zero if any check fails.
"""
import sys
import json
from pathlib import Path


def check_dataset_sizes():
    expected = {
        "data/emotion_dataset.jsonl": 280,
        "data/emotion_dataset_extended.jsonl": 400,
    }
    for rel_path, exp in expected.items():
        path = Path(rel_path)
        if not path.exists():
            print(f"Missing dataset: {path}")
            return False
        count = sum(1 for _ in path.open())
        if count != exp:
            print(f"Dataset size mismatch for {path}: got {count}, expected {exp}")
            return False
    return True


def check_required_files():
    required = [
        "results/baseline/cross_model_subspace_overlap.csv",
        "results/baseline/emotion_vectors/gpt2_vectors.pkl",
        "results/baseline/alignment/model_alignment_gpt2_pythia.pkl",
    ]
    ok = True
    for rel in required:
        if not Path(rel).exists():
            print(f"Missing required artifact: {rel}")
            ok = False
    return ok


def check_cli_args():
    # verify that key CLI flags mentioned in docs exist in scripts
    targets = {
        "src/models/activation_patching.py": ["--max-new-tokens", "--patch-window"],
        "src/models/activation_patching_sweep.py": ["--alpha"],
        "src/models/head_patching.py": ["--patch-mode"],
    }
    ok = True
    for rel, flags in targets.items():
        content = Path(rel).read_text()
        for flag in flags:
            if flag not in content:
                print(f"Flag {flag} not found in {rel}")
                ok = False
    return ok


def main():
    checks = [
        check_dataset_sizes,
        check_required_files,
        check_cli_args,
    ]
    success = True
    for fn in checks:
        if not fn():
            success = False
    if not success:
        sys.exit(1)
    print("Consistency checks passed.")


if __name__ == "__main__":
    main()
