"""
標準パイプラインにおける Phase 6（Head Screening）のスクリプト。

masterplan.md の Phase 6「Head スクリーニング & パッチング」の一部として、
全 head を順次アブレート（ゼロ化）して指標変化を計測し、
どの head が感情情報に敏感かをスコアリングする。

baseline_smoke（少数サンプル）と baseline（1感情225件前後を想定）に対応し、
統計ローダ互換のJSONを出力する。
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformer_lens import HookedTransformer

from src.analysis.evaluation import TextEvaluator
from src.analysis.statistics.data_loading import _flatten_metric_dict  # 再利用
from src.config.project_profiles import list_profiles
from src.data.dataset_loader import load_dataset_for_profile
from src.models.model_registry import get_model_spec
from src.utils.project_context import ProjectContext, profile_help_text
from src.models.phase8_large.head_patcher import LargeHeadAblator


def _parse_layers(layers: Sequence[int] | None, n_layers_hint: int | None) -> List[int]:
    if layers:
        return list(layers)
    if n_layers_hint is None:
        raise ValueError("層数が不明です。--layers を指定してください。")
    return list(range(n_layers_hint))


def _select_prompts(rows: List[Dict[str, str]], limit: int | None) -> List[str]:
    prompts: List[str] = []
    for row in rows:
        if row.get("emotion") != "neutral":
            continue
        if limit is not None and len(prompts) >= limit:
            break
        prompts.append(row["text"])
    return prompts


def _scores_to_dict(scores: Dict[str, np.ndarray], index: int) -> Dict[str, float]:
    """
    TextEvaluator.evaluate_batch()の返り値を、_flatten_metric_dictと同じ形式に変換。

    Args:
        scores: TextEvaluator.evaluate_batch()の返り値（Dict[str, np.ndarray]）
        index: サンプルインデックス

    Returns:
        ネストしたメトリクス辞書（Phase 7統計処理と互換）
    """
    result: Dict[str, Any] = {}
    for key, values in scores.items():
        # key format: "sentiment/POSITIVE" -> result["sentiment"]["POSITIVE"]
        parts = key.split("/", 1)
        if len(parts) == 2:
            metric_name, label = parts
            if metric_name not in result:
                result[metric_name] = {}
            result[metric_name][label] = float(values[index])
        else:
            result[key] = float(values[index])
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Head Screening（小型モデル用）")
    parser.add_argument("--profile", type=str, choices=list_profiles(), default="baseline", help=profile_help_text())
    parser.add_argument("--model", type=str, required=True, help="モデル名（小型）")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="スクリーニングする層（未指定は全層ヒント）")
    parser.add_argument("--max-samples", type=int, default=8, help="neutralプロンプトの最大件数")
    parser.add_argument("--sequence-length", type=int, default=20, help="生成トークン数")
    parser.add_argument("--device", type=str, default=None, help="cuda/mps/cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="生成・評価のバッチサイズ（デフォルト: 8）")
    args = parser.parse_args()

    ctx = ProjectContext(args.profile)
    spec = get_model_spec(args.model)
    layers = _parse_layers(args.layers, spec.n_layers_hint)
    rows = load_dataset_for_profile(args.profile)
    prompts = _select_prompts(rows, args.max_samples)
    if not prompts:
        raise ValueError("neutralプロンプトが見つかりません。")

    from src.utils.device import get_default_device_str

    device = args.device or get_default_device_str()
    
    start_time = time.time()
    print(f"[Phase 6] 評価器を初期化中...")
    text_evaluator = TextEvaluator(device=device)
    elapsed = time.time() - start_time
    print(f"[Phase 6] 評価器初期化完了: {elapsed:.2f}秒")

    # モデル/生成関数を分岐
    if spec.is_large:
        print(f"[Phase 6] 大モデル（HF）を読み込み中...")
        head_runner = LargeHeadAblator(spec, device=device)
        elapsed = time.time() - start_time
        print(f"[Phase 6] モデル読み込み完了: {elapsed:.2f}秒 (n_heads: {head_runner.n_heads})")
        
        def generate_batch(prompts_batch: Sequence[str], ablate: Tuple[int, int] | None, batch_size: int = 8) -> List[str]:
            heads = [ablate] if ablate is not None else None
            return head_runner.generate(list(prompts_batch), ablate_heads=heads, max_new_tokens=args.sequence_length, batch_size=batch_size)
        
        n_heads_total = head_runner.n_heads
    else:
        print(f"[Phase 6] モデルを読み込み中...")
        model = HookedTransformer.from_pretrained(spec.hf_id, device=device)
        model.cfg.use_attn_result = True
        model.eval()
        elapsed = time.time() - start_time
        print(f"[Phase 6] モデル読み込み完了: {elapsed:.2f}秒")
        
        gen_cfg = {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": None,
            "stop_at_eos": True,
            "return_type": "tokens",
        }
        
        def generate_batch(prompts_batch: Sequence[str], ablate: Tuple[int, int] | None, batch_size: int = 8) -> List[str]:
            all_texts = []
            handles = []
            if ablate is not None:
                target_layer, target_head = ablate

                def zero_head(activation, hook):
                    act = activation.clone()
                    # activation shape: [batch, seq, n_heads, d_head]
                    if target_head < act.shape[2]:
                        act[:, :, target_head, :] = 0
                    return act

                hook_name = f"blocks.{target_layer}.attn.hook_result"
                handles.append(model.add_hook(hook_name, zero_head))

            try:
                # バッチごとに処理
                for i in range(0, len(prompts_batch), batch_size):
                    batch_prompts = prompts_batch[i:i + batch_size]
                    tokens_list = [model.to_tokens(p) for p in batch_prompts]
                    prompt_lens = [t.shape[1] for t in tokens_list]
                    
                    # 最大長に合わせてパディング
                    max_len = max(prompt_lens)
                    padded_tokens = []
                    for t in tokens_list:
                        pad_len = max_len - t.shape[1]
                        if pad_len > 0:
                            pad_token = model.tokenizer.eos_token_id
                            pad = torch.full((1, pad_len), pad_token, dtype=t.dtype, device=t.device)
                            t = torch.cat([t, pad], dim=1)
                        padded_tokens.append(t)
                    
                    batch_tokens = torch.cat(padded_tokens, dim=0)
                    
                    with torch.no_grad():
                        generated = model.generate(batch_tokens, max_new_tokens=args.sequence_length, **gen_cfg)
                    
                    # 各サンプルをデコード
                    for j, prompt_len in enumerate(prompt_lens):
                        new_tokens = generated[j, prompt_len:]
                        text = model.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
                        all_texts.append((batch_prompts[j] + " " + text).strip())
            finally:
                for h in handles:
                    if hasattr(h, "remove"):
                        h.remove()
            return all_texts
        
        n_heads_total = model.cfg.n_heads

    # ベースライン生成を一度だけ（バッチ処理）
    start_time = time.time()
    print(f"[Phase 6] ベースライン生成を開始... (サンプル数: {len(prompts)})")
    baseline_texts = generate_batch(prompts, ablate=None, batch_size=args.batch_size)
    baseline_scores = text_evaluator.evaluate_batch(baseline_texts, batch_size=args.batch_size)
    baseline_flat = [_scores_to_dict(baseline_scores, i) for i in range(len(baseline_texts))]
    elapsed = time.time() - start_time
    print(f"[Phase 6] ベースライン生成完了: {elapsed:.2f}秒 ({elapsed/60:.2f}分)")

    total_heads = len(layers) * n_heads_total
    processed = 0
    
    phase6_start = time.time()
    print(f"[Phase 6] Head Screening を開始... (層数: {len(layers)}, ヘッド数/層: {n_heads_total}, 総ヘッド数: {total_heads}, サンプル数: {len(prompts)})")
    
    head_scores: List[Dict] = []
    head_samples: List[Dict] = []
    for layer_idx in layers:
        n_heads = n_heads_total
        for head_idx in range(n_heads):
            head_start = time.time()
            patched_texts = generate_batch(prompts, ablate=(layer_idx, head_idx), batch_size=args.batch_size)
            patched_scores = text_evaluator.evaluate_batch(patched_texts, batch_size=args.batch_size)
            patched_flat = [_scores_to_dict(patched_scores, i) for i in range(len(patched_texts))]

            # ネスト辞書をフラット化（例: {"sentiment": {"POSITIVE": 0.8}} -> {"sentiment.POSITIVE": 0.8}）
            def _flatten_nested(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
                result = {}
                for k, v in d.items():
                    key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        result.update(_flatten_nested(v, key))
                    else:
                        result[key] = float(v)
                return result

            baseline_flat_metrics = [_flatten_nested(m) for m in baseline_flat]
            patched_flat_metrics = [_flatten_nested(m) for m in patched_flat]

            metric_names = set()
            for mf in baseline_flat_metrics + patched_flat_metrics:
                metric_names.update(mf.keys())
            for metric in sorted(metric_names):
                deltas: List[float] = []
                for base_m, patch_m in zip(baseline_flat_metrics, patched_flat_metrics):
                    if metric in base_m and metric in patch_m:
                        deltas.append(patch_m[metric] - base_m[metric])
                if not deltas:
                    continue
                head_scores.append(
                    {
                        "layer": layer_idx,
                        "head": head_idx,
                        "metric": metric,
                        "delta_mean": float(sum(deltas) / len(deltas)) if deltas else 0.0,
                        "delta_std": float((sum((x - (sum(deltas) / len(deltas))) ** 2 for x in deltas) / len(deltas)) ** 0.5) if len(deltas) > 1 else 0.0,
                        "n_samples": len(deltas),
                    }
                )
            head_samples.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "samples": [
                        {"prompt": p, "baseline": b, "patched": q}
                        for p, b, q in zip(prompts, baseline_flat, patched_flat)
                    ],
                }
            )
            processed += 1
            head_elapsed = time.time() - head_start
            total_elapsed = time.time() - phase6_start
            if processed % 10 == 0 or processed == total_heads:
                print(f"  [Phase 6] Layer {layer_idx} Head {head_idx} 完了: {head_elapsed:.1f}秒 (進捗: {processed}/{total_heads}, 累計: {total_elapsed/60:.1f}分)")
    
    phase6_elapsed = time.time() - phase6_start
    print(f"[Phase 6] Head Screening 完了: {phase6_elapsed:.2f}秒 ({phase6_elapsed/60:.2f}分)")

    out_dir = ctx.results_dir() / "screening"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"head_scores_{spec.name}.json"
    payload = {
        "profile": args.profile,
        "model": spec.name,
        "layers": layers,
        "prompts": prompts,
        "sequence_length": args.sequence_length,
        "head_scores": head_scores,
        "head_samples": head_samples,
    }
    
    start_time = time.time()
    print(f"[Phase 6] 結果を保存中...")
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elapsed = time.time() - start_time
    print(f"[Phase 6] 保存完了: {elapsed:.2f}秒")
    print(f"✓ Head Screening 結果を保存: {out_path}")


if __name__ == "__main__":
    main()
