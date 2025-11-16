"""Llama3 8B Phase 8 結果のサマリー生成スクリプト（CSV + 文章出力 + 任意でMarkdown）.

このスクリプトは、Phase 8で生成されたアライメントpickleファイルを読み込み、
層ごとの平均overlap（before/after）を計算してCSVに保存し、コンソールにサマリーを表示します。
オプションでMarkdownレポートも生成できます。

注意:
- 古いpickleファイルにはk（PCA次元数）が含まれていない場合があります。
  その場合は「N/A」と表示されますが、エラーにはなりません。
- このスクリプトはモジュールとしてインポート可能で、mainロジックは
  if __name__ == "__main__": の下にあります。
"""
from __future__ import annotations

import argparse
import datetime
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.utils.project_context import profile_help_text
from src.config.project_profiles import list_profiles


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
    """コンソールにサマリーを表示（日本語ラベル付き）"""
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
        layer = int(row['layer'])
        mean_before = row['mean_overlap_before']
        mean_after = row['mean_overlap_after']
        delta = row['delta']
        print(f"{layer:<6} {mean_before:<18.6f} {mean_after:<18.6f} {delta:<15.6f}")
    print()


def write_report(
    output_path: Path,
    profile: str,
    source_model: str,
    target_model: str,
    k: Optional[int],
    df_layer: pd.DataFrame,
    layers_used: Optional[List[int]] = None,
    n_components: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> None:
    """Markdownレポートを生成（既存のphaseレポートのスタイルに合わせる）"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    k_str = str(k) if k is not None else (str(n_components) if n_components is not None else "N/A")
    
    # モデル名を読みやすい形式に変換
    model_display = {
        "gpt2": "GPT-2 small (124M)",
        "llama3_8b": "Llama3 8B (Meta-Llama-3.1-8B)",
    }
    source_display = model_display.get(source_model, source_model)
    target_display = model_display.get(target_model, target_model)
    
    lines: List[str] = []
    lines.append("# Phase 8: 中規模モデル（Llama3 8B）の感情サブスペース整合性解析")
    lines.append(f"_Last updated: {today}_")
    lines.append("")
    lines.append("## 🎯 目的")
    lines.append("")
    lines.append("中規模モデル（Llama3 8B）とGPT-2 small間の感情サブスペースの整合性を解析し、")
    lines.append("線形写像によるアライメント前後のoverlapを比較しました。")
    lines.append("")
    lines.append("## 📦 生成物")
    lines.append("")
    lines.append(f"- `{output_path.name}` - 本レポート")
    lines.append("")
    lines.append("## 🚀 実行コマンド")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m src.analysis.summarize_phase8_llama3 \\")
    lines.append(f"  --profile {profile} \\")
    lines.append(f"  --alignment-path results/{profile}/alignment/gpt2_vs_llama3_8b_token_based_full.pkl \\")
    lines.append(f"  --output-csv results/{profile}/statistics/phase8_llama3_alignment_summary.csv \\")
    lines.append(f"  --write-report {output_path}")
    lines.append("```")
    lines.append("")
    lines.append("## 📄 レポート項目")
    lines.append("")
    lines.append("### 1. 実験設定")
    lines.append("")
    lines.append(f"- **モデル**: {source_display} (source), {target_display} (target)")
    lines.append(f"- **プロファイル**: {profile}")
    lines.append("- **データ**: 感情プロンプト × 4感情（gratitude / anger / apology / neutral）")
    lines.append(f"- **手法概要**: token-based emotion vector, 多サンプルPCA (k={k_str}), neutralサブスペースからの線形写像, GPT-2 vs Llama3 のsubspace overlap (before/after)")
    if layers_used:
        layers_str = f"{min(layers_used)}–{max(layers_used)}" if len(layers_used) > 1 else str(layers_used[0])
        lines.append(f"- **対象層**: {layers_str} ({len(layers_used)}層)")
    if max_samples:
        lines.append(f"- **感情ごとの最大サンプル数**: {max_samples}")
    lines.append("")
    lines.append("### 2. ハイパーパラメータ")
    lines.append("")
    lines.append(f"- **PCA次元数 (k)**: {k_str}")
    if layers_used:
        lines.append(f"- **対象層**: {layers_used}")
    if max_samples:
        lines.append(f"- **max-samples-per-emotion**: {max_samples}")
    else:
        lines.append("- **max-samples-per-emotion**: N/A（メタデータに記録されていません）")
    lines.append("")
    lines.append("### 3. 結果概要")
    lines.append("")
    if df_layer.empty:
        lines.append("データがありません。")
    else:
        # Markdown表
        lines.append("| Layer | Mean overlap (before) | Mean overlap (after) | Δ (after - before) |")
        lines.append("|-------|-----------------------|----------------------|--------------------|")
        for _, row in df_layer.iterrows():
            lines.append(
                f"| {int(row['layer'])} | {row['mean_overlap_before']:.6f} | "
                f"{row['mean_overlap_after']:.6f} | {row['delta']:.6f} |"
            )
    lines.append("")
    lines.append("### 4. 簡単な考察")
    lines.append("")
    if not df_layer.empty:
        # データから自動的に傾向を抽出
        max_delta_layer = int(df_layer.loc[df_layer['delta'].idxmax(), 'layer'])
        max_delta_value = df_layer['delta'].max()
        min_delta_layer = int(df_layer.loc[df_layer['delta'].idxmin(), 'layer'])
        min_delta_value = df_layer['delta'].min()
        avg_before = df_layer['mean_overlap_before'].mean()
        avg_after = df_layer['mean_overlap_after'].mean()
        
        lines.append(f"- **改善が最も大きい層**: Layer {max_delta_layer} (Δ = {max_delta_value:.6f})")
        lines.append(f"- **改善が最も小さい層**: Layer {min_delta_layer} (Δ = {min_delta_value:.6f})")
        lines.append(f"- **平均overlap (before)**: {avg_before:.6f}")
        lines.append(f"- **平均overlap (after)**: {avg_after:.6f}")
        lines.append(f"- **平均改善幅**: {avg_after - avg_before:.6f}")
        lines.append("")
        lines.append("線形写像によるアライメント後、すべての層でoverlapが改善しているかどうかを確認してください。")
        lines.append("中間層で特に改善が大きい場合、感情表現の抽象化が進んでいる可能性があります。")
    else:
        lines.append("- データが不足しているため、詳細な考察はできません。")
    lines.append("")
    lines.append("小型モデル（Phase 6）との定性的な比較:")
    lines.append("- Phase 6ではGPT-2とPythia-160M間の比較を行いましたが、本Phase 8ではより大きなモデル（Llama3 8B）との比較を行っています。")
    lines.append("- モデルサイズが大きくなることで、感情サブスペースの表現がどのように変化するかを観察できます。")
    lines.append("")
    lines.append("### 5. 今後のステップ")
    lines.append("")
    lines.append("- **Gemma2 / Qwen への展開**: 他の大規模モデルでも同様の解析を実施し、モデル間の一般性を検証")
    lines.append("- **サンプル数やkの増加**: より多くのサンプルやPCA次元数で再実験し、結果の頑健性を確認")
    lines.append("- **Phase 5 / 7.5 の統計パイプラインへの組み込み**: 効果量や検出力分析を実施し、統計的有意性を検証")
    lines.append("- **複数モデル間の比較**: GPT-2、Pythia-160M、Llama3 8Bの3モデル間での比較解析")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdownレポートを出力しました: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Llama3 8B Phase 8 サマリ生成 (CSV/Markdown)")
    parser.add_argument(
        "--profile",
        type=str,
        default="baseline",
        choices=list_profiles(),
        help=f"Dataset profile (例: baseline). {profile_help_text()}",
    )
    parser.add_argument(
        "--alignment-path",
        type=str,
        default="results/baseline/alignment/gpt2_vs_llama3_8b_token_based_full.pkl",
        help="アライメント結果のpickleパス",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/baseline/statistics/phase8_llama3_alignment_summary.csv",
        help="層平均サマリを書き出す CSV パス",
    )
    parser.add_argument(
        "--write-report",
        type=str,
        default=None,
        help="Markdown レポートを書き出すパス（指定時のみ生成）",
    )
    args = parser.parse_args()

    alignment_path = Path(args.alignment_path)
    data = load_alignment(alignment_path)
    overlaps = data.get("overlaps", {})
    source_model = data.get("source_model", "gpt2")
    target_model = data.get("target_model", "llama3_8b")
    k = data.get("k")

    rows = _rows_from_alignment(overlaps)
    df = pd.DataFrame(rows)
    layer_summary = build_summaries(df)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    layer_summary.to_csv(output_csv, index=False)
    print(f"層平均サマリを保存しました: {output_csv}")

    # コンソール出力
    print_summary(source_model, target_model, args.profile, k, layer_summary)

    # レポート出力（任意）
    if args.write_report:
        # 層リストをソートして取得
        layers_used = sorted(layer_summary["layer"].tolist()) if not layer_summary.empty else None
        write_report(
            output_path=Path(args.write_report),
            profile=args.profile,
            source_model=source_model,
            target_model=target_model,
            k=k,
            df_layer=layer_summary,
            layers_used=layers_used,
            n_components=k,
        )


if __name__ == "__main__":
    main()
