"""Phase 8 大規模モデル（llama3/gemma3/qwen3 等）のアライメント結果サマリ."""
from __future__ import annotations

import argparse
import datetime
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config.project_profiles import list_profiles
from src.utils.project_context import ProjectContext, profile_help_text
from src.models.phase8_large.registry import MODEL_REGISTRY, get_spec


def load_alignment(alignment_path: Path) -> Dict:
    with alignment_path.open("rb") as f:
        data = pickle.load(f)
    return data


def _rows_from_alignment(overlaps: Dict) -> List[Dict]:
    rows: List[Dict] = []
    for layer_entry in overlaps.get("per_layer", []):
        layer = layer_entry.get("layer")
        emotions: Dict = layer_entry.get("emotions", {})
        for emotion, vals in emotions.items():
            rows.append(
                {
                    "layer": layer,
                    "emotion": emotion,
                    "overlap_before": float(vals.get("overlap_before", float("nan"))),
                    "overlap_after": float(vals.get("overlap_after", float("nan"))),
                }
            )
    return rows


def build_summaries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grp = df.groupby("layer", as_index=False).agg(
        mean_overlap_before=("overlap_before", "mean"),
        mean_overlap_after=("overlap_after", "mean"),
    )
    grp["delta"] = grp["mean_overlap_after"] - grp["mean_overlap_before"]
    return grp


def print_summary(source: str, target: str, profile: str, k: Optional[int], df_layer: pd.DataFrame) -> None:
    k_str = str(k) if k is not None else "N/A"
    print("\n" + "=" * 80)
    print(f"Phase 8: {source} 対 {target} サブスペースアライメント解析")
    print(f"プロファイル: {profile}")
    print(f"PCA次元数 (k): {k_str}")
    print("=" * 80)
    if df_layer.empty:
        print("データがありません。")
        return
    print("\n層ごとの平均 overlap:")
    print(f"{'層':<6} {'before (平均)':<18} {'after (平均)':<18} {'差分 (Δ)':<15}")
    print("-" * 60)
    for _, row in df_layer.iterrows():
        layer = int(row["layer"])
        mean_before = row["mean_overlap_before"]
        mean_after = row["mean_overlap_after"]
        delta = row["delta"]
        print(f"{layer:<6} {mean_before:<18.6f} {mean_after:<18.6f} {delta:<15.6f}")
    print()


def write_report(
    output_path: Path,
    profile: str,
    source_model: str,
    target_model: str,
    target_pretty: str,
    k: Optional[int],
    df_layer: pd.DataFrame,
    layers_used: Optional[List[int]] = None,
    n_components: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    k_str = str(k) if k is not None else (str(n_components) if n_components is not None else "N/A")
    lines: List[str] = []
    lines.append(f"# Phase 8: 中規模モデル（{target_pretty}）の感情サブスペース整合性解析")
    lines.append(f"_Last updated: {today}_")
    lines.append("")
    lines.append("## 🎯 目的")
    lines.append("")
    lines.append(f"{source_model} と {target_pretty} の感情サブスペース整合性を解析し、線形写像前後の overlap を比較。")
    lines.append("")
    lines.append("## 🚀 実験設定")
    lines.append(f"- source: {source_model}")
    lines.append(f"- target: {target_pretty}")
    lines.append(f"- プロファイル: {profile}")
    if layers_used:
        lines.append(f"- 対象層: {layers_used}")
    if n_components or k:
        lines.append(f"- PCA 次元 (k): {k_str}")
    if max_samples:
        lines.append(f"- 感情ごとの最大サンプル: {max_samples}")
    lines.append("- 手法: token-based 感情ベクトル → 多サンプルPCA → neutral から線形写像学習 → before/after overlap")
    lines.append("")
    lines.append("## 📊 結果概要（層平均）")
    if df_layer.empty:
        lines.append("データがありません。")
    else:
        lines.append("| Layer | Mean overlap (before) | Mean overlap (after) | Δ (after - before) |")
        lines.append("|-------|-----------------------|----------------------|--------------------|")
        for _, row in df_layer.iterrows():
            lines.append(
                f"| {int(row['layer'])} | {row['mean_overlap_before']:.6f} | "
                f"{row['mean_overlap_after']:.6f} | {row['delta']:.6f} |"
            )
    lines.append("")
    lines.append("## 💡 考察（簡潔に追記してください）")
    lines.append("- どの層で改善が大きいか / 小さいか。")
    lines.append("- 小型モデルの Phase 6 結果と類似傾向か、強弱はどうか。")
    lines.append("- 本設定はベースラインであり、更なるサンプル数/k増加で改善余地あり。")
    lines.append("")
    lines.append("## 🔭 今後のステップ")
    lines.append("- サンプル数や PCA 次元を増やした再実験。")
    lines.append("- 他モデル（Gemma3/Qwen3）との比較・横展開。")
    lines.append("- Phase 5/7.5 の統計パイプラインに大規模モデルを統合。")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdownレポートを出力しました: {output_path}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 8 大規模モデル アライメントサマリ (CSV/Markdown)")
    parser.add_argument("--profile", type=str, default="baseline", choices=list_profiles(), help=f"プロファイル ({profile_help_text()})")
    parser.add_argument("--large-model", type=str, choices=list(MODEL_REGISTRY.keys()), default="llama3_8b")
    parser.add_argument("--alignment-path", type=str, default=None, help="アライメントpickleパス（未指定なら自動推定）")
    parser.add_argument("--output-csv", type=str, default=None, help="層平均サマリCSV（未指定なら自動推定）")
    parser.add_argument("--write-report", type=str, default=None, help="Markdownレポートを書き出すパス（任意）")
    args = parser.parse_args(argv)

    spec = get_spec(args.large_model)
    context = ProjectContext(profile_name=args.profile)

    alignment_path = Path(args.alignment_path) if args.alignment_path else context.results_dir() / "alignment" / f"gpt2_vs_{spec.name}_token_based_full.pkl"
    output_csv = Path(args.output_csv) if args.output_csv else context.results_dir() / "statistics" / f"phase8_{spec.name}_alignment_summary.csv"

    data = load_alignment(alignment_path)
    overlaps = data.get("overlaps", {})
    source_model = data.get("source_model", "gpt2")
    target_model = data.get("target_model", spec.name)
    k = data.get("k")

    rows = _rows_from_alignment(overlaps)
    df = pd.DataFrame(rows)
    layer_summary = build_summaries(df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    layer_summary.to_csv(output_csv, index=False)
    print(f"層平均サマリを保存しました: {output_csv}")

    print_summary(source_model, target_model, args.profile, k, layer_summary)

    if args.write_report:
        write_report(
            output_path=Path(args.write_report),
            profile=args.profile,
            source_model=source_model,
            target_model=target_model,
            target_pretty=spec.pretty_name or spec.name,
            k=k,
            df_layer=layer_summary,
            layers_used=[int(x) for x in layer_summary["layer"].tolist()] if not layer_summary.empty else None,
            n_components=k,
        )


if __name__ == "__main__":
    main()
