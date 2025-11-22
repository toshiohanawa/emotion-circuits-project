"""
Utility script for verifying that refactors preserve numeric outputs.

It compares artifacts (CSV/JSON/JSONL/NPY/PKL) from two directories
and reports the largest differences. Typical usage:

    python scripts/verify_refactoring.py \
      --before /path/to/baseline/results \
      --after  /path/to/new/results \
      --profiles baseline baseline_smoke
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np
import pandas as pd

try:  # torch is optional at runtime
    import torch
except Exception:  # pragma: no cover - fallback when torch is unavailable
    torch = None


SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".pkl", ".pickle", ".npy", ".npz"}
DEFAULT_PROFILES: Sequence[str] = ("baseline", "baseline_smoke")
DEFAULT_ARTIFACTS: Sequence[str] = (
    "statistics/effect_sizes.csv",
    "statistics/power_analysis.csv",
    "statistics/power_analysis.json",
)


@dataclass(frozen=True)
class ComparisonConfig:
    atol: float = 1e-3
    rtol: float = 1e-4
    max_report: int = 20
    allow_missing: bool = False


def _numeric_pair(series_a: pd.Series, series_b: pd.Series) -> tuple[np.ndarray, np.ndarray] | None:
    """Attempt to coerce two Series into numeric arrays."""
    if pd.api.types.is_numeric_dtype(series_a) and pd.api.types.is_numeric_dtype(series_b):
        return series_a.to_numpy(dtype=float), series_b.to_numpy(dtype=float)
    coerced_a = pd.to_numeric(series_a, errors="coerce")
    coerced_b = pd.to_numeric(series_b, errors="coerce")
    valid = (~coerced_a.isna()) | (~coerced_b.isna())
    if valid.sum() == 0:
        return None
    coverage = valid.sum() / len(valid)
    if coverage < 0.8:
        return None
    return coerced_a.to_numpy(dtype=float), coerced_b.to_numpy(dtype=float)


def _format_row_diffs(indices: Iterable[int], lhs: Sequence[Any], rhs: Sequence[Any], limit: int = 3) -> str:
    rows = []
    for idx in list(indices)[:limit]:
        a_val = lhs[idx]
        b_val = rhs[idx]
        rows.append(f"row {idx}: {a_val!r} vs {b_val!r}")
    return "; ".join(rows)


def compare_csv(path_a: Path, path_b: Path, config: ComparisonConfig) -> List[str]:
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)
    issues: List[str] = []
    if df_a.shape != df_b.shape:
        issues.append(f"shape mismatch: {df_a.shape} vs {df_b.shape}")
    missing_cols = sorted(set(df_a.columns) ^ set(df_b.columns))
    if missing_cols:
        issues.append(f"column mismatch: {missing_cols}")
    shared_cols = [c for c in df_a.columns if c in df_b.columns]
    for col in shared_cols:
        series_a = df_a[col]
        series_b = df_b[col]
        numeric_pair = _numeric_pair(series_a, series_b)
        if numeric_pair:
            arr_a, arr_b = numeric_pair
            if arr_a.shape != arr_b.shape:
                issues.append(f"{col}: length mismatch {arr_a.shape} vs {arr_b.shape}")
                continue
            mask = ~np.isclose(
                arr_a,
                arr_b,
                atol=config.atol,
                rtol=config.rtol,
                equal_nan=True,
            )
            if mask.any():
                diffs = np.abs(arr_a - arr_b)
                max_diff = float(np.max(diffs[mask]))
                idxs = np.where(mask)[0]
                sample = _format_row_diffs(idxs, series_a.tolist(), series_b.tolist())
                issues.append(f"{col}: {mask.sum()} rows differ (max Δ={max_diff:.3g}; {sample})")
        else:
            lhs = series_a.fillna("").astype(str).tolist()
            rhs = series_b.fillna("").astype(str).tolist()
            mismatch = [i for i, (a, b) in enumerate(zip(lhs, rhs)) if a != b]
            if mismatch:
                sample = _format_row_diffs(mismatch, lhs, rhs)
                issues.append(f"{col}: {len(mismatch)} rows differ ({sample})")
        if len(issues) >= config.max_report:
            break
    return issues


def _normalize_leaf(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.generic):
        return value.item()
    return value


def compare_structures(a: Any, b: Any, path: str, config: ComparisonConfig, issues: List[str]) -> None:
    if len(issues) >= config.max_report:
        return
    a = _normalize_leaf(a)
    b = _normalize_leaf(b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        arr_a = np.asarray(a)
        arr_b = np.asarray(b)
        if arr_a.shape != arr_b.shape:
            issues.append(f"{path}: shape mismatch {arr_a.shape} vs {arr_b.shape}")
            return
        if arr_a.dtype == object or arr_b.dtype == object:
            compare_structures(arr_a.tolist(), arr_b.tolist(), path, config, issues)
            return
        if not np.allclose(arr_a, arr_b, atol=config.atol, rtol=config.rtol, equal_nan=True):
            diff = float(np.max(np.abs(arr_a - arr_b)))
            issues.append(f"{path}: max Δ={diff:.3g}")
        return

    from collections.abc import Mapping, Sequence
    import numbers

    if isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
        if math.isnan(a) and math.isnan(b):
            return
        if not math.isclose(
            float(a),
            float(b),
            rel_tol=config.rtol,
            abs_tol=config.atol,
        ):
            diff = abs(float(a) - float(b))
            issues.append(f"{path}: {a} vs {b} (Δ={diff:.3g})")
        return

    if isinstance(a, str) and isinstance(b, str):
        if a != b:
            issues.append(f"{path}: '{a}' != '{b}'")
        return

    if isinstance(a, bool) and isinstance(b, bool):
        if a != b:
            issues.append(f"{path}: {a} != {b}")
        return

    if a is None or b is None:
        if a != b:
            issues.append(f"{path}: {a} != {b}")
        return

    if isinstance(a, Mapping) and isinstance(b, Mapping):
        keys = set(a.keys()) | set(b.keys())
        for key in sorted(keys):
            if key not in a or key not in b:
                issues.append(f"{path}[{key!r}]: missing key")
                if len(issues) >= config.max_report:
                    return
                continue
            compare_structures(a[key], b[key], f"{path}[{key!r}]", config, issues)
            if len(issues) >= config.max_report:
                return
        return

    if isinstance(a, Sequence) and not isinstance(a, (str, bytes)):
        if isinstance(b, Sequence) and not isinstance(b, (str, bytes)):
            if len(a) != len(b):
                issues.append(f"{path}: length mismatch {len(a)} vs {len(b)}")
                return
            for idx, (aval, bval) in enumerate(zip(a, b)):
                compare_structures(aval, bval, f"{path}[{idx}]", config, issues)
                if len(issues) >= config.max_report:
                    return
            return

    if type(a) != type(b):
        issues.append(f"{path}: type mismatch {type(a)} vs {type(b)}")
    elif a != b:
        issues.append(f"{path}: {a!r} != {b!r}")


def load_structured_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as fh:
            return pickle.load(fh)
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix == ".npz":
        npz = np.load(path, allow_pickle=True)
        return {key: npz[key] for key in npz.files}
    raise ValueError(f"Unsupported structured file: {path}")


def collect_files(
    before_root: Path,
    manual_paths: Sequence[Path],
    profiles: Sequence[str],
    include_defaults: bool,
    compare_all: bool,
) -> List[Path]:
    files: set[Path] = set()
    if include_defaults:
        for profile in profiles:
            for rel in DEFAULT_ARTIFACTS:
                files.add(Path("results") / profile / rel)
    for rel in manual_paths:
        files.add(rel)
    if compare_all:
        for candidate in before_root.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
                files.add(candidate.relative_to(before_root))
    return sorted(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare before/after artifacts with numeric tolerances.")
    parser.add_argument("--before", type=Path, required=True, help="Baseline directory (old results).")
    parser.add_argument("--after", type=Path, required=True, help="New directory to verify.")
    parser.add_argument(
        "--profiles",
        type=str,
        nargs="*",
        default=None,
        help=f"Profiles to include when building default artifact paths (default: {', '.join(DEFAULT_PROFILES)})",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="*",
        default=None,
        help="Additional relative file paths to compare.",
    )
    parser.add_argument(
        "--skip-defaults",
        action="store_true",
        help="Do not automatically include standard statistics artifacts.",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help=f"Compare every file with suffix in {sorted(SUPPORTED_SUFFIXES)} under --before.",
    )
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for numeric differences.")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for numeric differences.")
    parser.add_argument("--max-report", type=int, default=20, help="Maximum number of issues to report per file.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip files that are missing from either directory instead of failing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    before_root = args.before.resolve()
    after_root = args.after.resolve()
    profiles = tuple(args.profiles or DEFAULT_PROFILES)
    manual_paths = [Path(p) for p in (args.files or [])]
    for rel in manual_paths:
        if rel.is_absolute():
            raise ValueError(f"--files must be relative paths: {rel}")

    compare_paths = collect_files(
        before_root=before_root,
        manual_paths=manual_paths,
        profiles=profiles,
        include_defaults=not args.skip_defaults,
        compare_all=args.compare_all,
    )
    if not compare_paths:
        raise ValueError("No files to compare. Pass --files or enable defaults/all comparison.")

    config = ComparisonConfig(
        atol=args.atol,
        rtol=args.rtol,
        max_report=args.max_report,
        allow_missing=args.allow_missing,
    )

    total = 0
    failures = 0
    for rel_path in compare_paths:
        before_file = before_root / rel_path
        after_file = after_root / rel_path
        if not before_file.exists() or not after_file.exists():
            msg = f"[SKIP] {rel_path} (missing from {'before' if not before_file.exists() else 'after'})"
            if args.allow_missing:
                print(msg)
                continue
            raise FileNotFoundError(msg)

        suffix = before_file.suffix.lower()
        if suffix == ".csv":
            problems = compare_csv(before_file, after_file, config)
        else:
            data_before = load_structured_file(before_file)
            data_after = load_structured_file(after_file)
            problems: List[str] = []
            compare_structures(data_before, data_after, path=str(rel_path), config=config, issues=problems)
        total += 1
        if problems:
            failures += 1
            print(f"[FAIL] {rel_path}")
            for issue in problems[: config.max_report]:
                print(f"    - {issue}")
        else:
            print(f"[OK]   {rel_path}")

    print(f"Compared {total} files: {total - failures} OK, {failures} failed.")
    if failures > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
