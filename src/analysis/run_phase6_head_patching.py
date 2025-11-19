"""
標準パイプラインにおける Phase 6（Head Ablation）のスクリプト。

masterplan.md の Phase 6「Head スクリーニング & パッチング」の一部として、
指定された head をゼロ化（ablation）し、生成テキストの指標変化を計測する。

baseline_smoke（少数サンプル）と baseline（1感情225件前後を想定）に対応し、
統計ローダ互換の形式で保存する。
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformer_lens import HookedTransformer

from src.analysis.evaluation import TextEvaluator
from src.analysis.statistics.data_loading import _flatten_metric_dict
from src.config.project_profiles import list_profiles
from src.data.dataset_loader import load_dataset_for_profile
from src.models.model_registry import get_model_spec
from src.utils.device import get_default_device_str
from src.utils.project_context import ProjectContext, profile_help_text
from src.models.phase8_large.head_patcher import LargeHeadAblator


def _expand_head_range(range_str: str, n_heads: int) -> List[int]:
    if range_str in {"*", "all"}:
        return list(range(n_heads))
    if "-" in range_str:
        start, end = range_str.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(range_str)]


def _parse_heads(head_specs: List[str], n_heads: int) -> List[Tuple[int, int]]:
    heads: List[Tuple[int, int]] = []
    for spec in head_specs:
        if ":" not in spec:
            continue
        l, h = spec.split(":")
        try:
            layer_idx = int(l)
        except ValueError:
            continue
        try:
            for head_idx in _expand_head_range(h, n_heads):
                heads.append((layer_idx, head_idx))
        except ValueError:
            continue
    return heads


def _select_neutral_prompts(rows, limit: int | None) -> List[str]:
    prompts: List[str] = []
    for row in rows:
        if row.get("emotion") != "neutral":
            continue
        if limit is not None and len(prompts) >= limit:
            break
        prompts.append(row["text"])
    return prompts


def _scores_to_dict(scores: Dict[str, Any], index: int) -> Dict[str, Any]:
    """TextEvaluator.evaluate_batch()の返り値をネスト辞書に変換（Phase 7統計処理と互換）。"""
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
    parser = argparse.ArgumentParser(description="Phase 6: Head Ablation（小型モデル、統計互換フォーマット）")
    parser.add_argument("--profile", type=str, choices=list_profiles(), default="baseline", help=profile_help_text())
    parser.add_argument("--model", type=str, required=True, help="モデル名（小型のみ）")
    parser.add_argument("--heads", type=str, nargs="+", required=True, help="パッチ対象head (例: 0:0 1:3)")
    parser.add_argument("--max-samples", type=int, default=8, help="neutralプロンプトの最大件数")
    parser.add_argument("--sequence-length", type=int, default=20, help="生成トークン数")
    parser.add_argument("--device", type=str, default=None, help="cuda/mps/cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="生成のバッチサイズ（デフォルト: 8）")
    args = parser.parse_args()

    ctx = ProjectContext(args.profile)
    spec = get_model_spec(args.model)

    rows = load_dataset_for_profile(args.profile)
    prompts = _select_neutral_prompts(rows, args.max_samples)
    if not prompts:
        raise ValueError("neutralプロンプトが見つかりません。")

    device = args.device or get_default_device_str()
    
    # モデルロード分岐
    if spec.is_large:
        start_time = time.time()
        print(f"[Phase 6] 大モデル（HF）を読み込み中...")
        head_runner = LargeHeadAblator(spec, device=device)
        heads = _parse_heads(args.heads, head_runner.n_heads)
        if not heads:
            raise ValueError("head指定が不正です。例: --heads 0:0 1:3 または 0:0-11")
        elapsed = time.time() - start_time
        print(f"[Phase 6] モデル読み込み完了: {elapsed:.2f}秒 (対象ヘッド数: {len(heads)})")

        def _generate_batch(prompts_batch: List[str], hook: bool = False, batch_size: int = 8) -> List[str]:
            ablate = heads if hook else None
            return head_runner.generate(prompts_batch, ablate_heads=ablate, max_new_tokens=args.sequence_length, batch_size=batch_size)

        n_heads = head_runner.n_heads
    else:
        start_time = time.time()
        print(f"[Phase 6] モデルを読み込み中...")
        model = HookedTransformer.from_pretrained(spec.hf_id, device=device)
        n_heads = model.cfg.n_heads
        heads = _parse_heads(args.heads, n_heads)
        if not heads:
            raise ValueError("head指定が不正です。例: --heads 0:0 1:3 または 0:0-11")
        model.cfg.use_attn_result = True
        model.eval()
        elapsed = time.time() - start_time
        print(f"[Phase 6] モデル読み込み完了: {elapsed:.2f}秒 (対象ヘッド数: {len(heads)})")
        
        gen_cfg = {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": None,
            "stop_at_eos": True,
            "return_type": "tokens",
        }

        def _generate_batch(prompts_batch: List[str], hook: bool = False, batch_size: int = 8) -> List[str]:
            """バッチで生成（hook対応）"""
            all_texts = []
            handles = []
            
            if hook:
                def zero_head(activation, hook):
                    act = activation.clone()
                    for layer_idx, head_idx in heads:
                        if hook.name != f"blocks.{layer_idx}.attn.hook_result":
                            continue
                        # activation shape: [batch, seq, n_heads, d_head]
                        if head_idx < act.shape[2]:
                            act[:, :, head_idx, :] = 0
                    return act
                for layer_idx, _ in heads:
                    hname = f"blocks.{layer_idx}.attn.hook_result"
                    handles.append(model.add_hook(hname, zero_head))
            
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

    # 評価器を初期化
    start_time = time.time()
    print(f"[Phase 6] 評価器を初期化中...")
    text_evaluator = TextEvaluator(device=device)
    elapsed = time.time() - start_time
    print(f"[Phase 6] 評価器初期化完了: {elapsed:.2f}秒")

    start_time = time.time()
    print(f"[Phase 6] ベースライン生成を開始... (サンプル数: {len(prompts)})")
    baseline_texts = _generate_batch(prompts, hook=False, batch_size=args.batch_size)
    elapsed = time.time() - start_time
    print(f"[Phase 6] ベースライン生成完了: {elapsed:.2f}秒 ({elapsed/60:.2f}分)")

    # ベースライン評価
    start_time = time.time()
    print(f"[Phase 6] ベースライン評価を実行中...")
    baseline_scores = text_evaluator.evaluate_batch(baseline_texts, batch_size=args.batch_size)
    baseline_metrics = [_scores_to_dict(baseline_scores, i) for i in range(len(baseline_texts))]
    elapsed = time.time() - start_time
    print(f"[Phase 6] ベースライン評価完了: {elapsed:.2f}秒 ({elapsed/60:.2f}分)")

    start_time = time.time()
    print(f"[Phase 6] Head Ablation を開始... (ヘッド数: {len(heads)}, サンプル数: {len(prompts)})")
    patched_texts = _generate_batch(prompts, hook=True, batch_size=args.batch_size)
    elapsed = time.time() - start_time
    print(f"[Phase 6] Head Ablation 完了: {elapsed:.2f}秒 ({elapsed/60:.2f}分)")

    # パッチ後の評価
    start_time = time.time()
    print(f"[Phase 6] パッチ後評価を実行中...")
    patched_scores = text_evaluator.evaluate_batch(patched_texts, batch_size=args.batch_size)
    patched_metrics = [_scores_to_dict(patched_scores, i) for i in range(len(patched_texts))]
    elapsed = time.time() - start_time
    print(f"[Phase 6] パッチ後評価完了: {elapsed:.2f}秒 ({elapsed/60:.2f}分)")

    payload = {
        "model": spec.name,
        "baseline_texts": baseline_texts,
        "patched_texts": patched_texts,
        "baseline_metrics": baseline_metrics,
        "patched_metrics": patched_metrics,
        "heads": heads,
        "profile": args.profile,
        "prompts": prompts,
    }

    out_dir = ctx.results_dir() / "patching" / "head_patching"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.name}_head_ablation.pkl"
    
    start_time = time.time()
    print(f"[Phase 6] 結果を保存中...")
    with out_path.open("wb") as f:
        pickle.dump(payload, f)
    elapsed = time.time() - start_time
    print(f"[Phase 6] 保存完了: {elapsed:.2f}秒")
    print(f"✓ Head Ablation 結果を保存: {out_path}")


if __name__ == "__main__":
    main()
