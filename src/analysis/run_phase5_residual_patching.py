"""
標準パイプラインにおける Phase 5（multi-token 残差パッチング）のスクリプト。
baseline_smoke（少数サンプル）と baseline（1感情225件前後を想定）に対し、残差パッチングとランダム対照を実行し、
共通評価器/統計ローダ互換の出力を生成する。
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, List, Sequence
import sys

import numpy as np
import torch

from src.analysis.evaluation import TextEvaluator
from src.config.project_profiles import list_profiles
from src.data.dataset_loader import load_dataset_for_profile
from src.models.activation_patching import ActivationPatcher
from src.models.activation_patching_large import LargeActivationPatcher
from src.models.model_registry import get_model_spec
from src.utils.device import get_default_device_str
from src.utils.project_context import ProjectContext, profile_help_text
from src.utils.timing import record_phase_timing


def _load_emotion_vectors(path: Path) -> tuple[Dict[str, np.ndarray], Dict]:
    with path.open("rb") as f:
        data = pickle.load(f)
    return data["emotion_vectors"], data.get("metadata", {})


def _alpha_schedule(alpha: float, steps: int) -> List[float]:
    return [float(alpha) for _ in range(steps)]


def _select_neutral_prompts(rows: List[Dict[str, str]], max_samples: int | None) -> List[str]:
    prompts: List[str] = []
    for row in rows:
        if row.get("emotion") != "neutral":
            continue
        if max_samples is not None and len(prompts) >= max_samples:
            break
        prompts.append(row["text"])
    return prompts


def _ensure_layer_vectors(vectors: np.ndarray, layer_indices: Sequence[int]) -> Dict[int, np.ndarray]:
    return {layer: vectors[idx] for idx, layer in enumerate(layer_indices)}


def main():
    parser = argparse.ArgumentParser(description="Phase5: multi-token残差パッチング（小型モデル向け）")
    parser.add_argument("--profile", type=str, choices=list_profiles(), default="baseline", help=profile_help_text())
    parser.add_argument("--model", type=str, required=True, help="モデル名（model_registryに登録された名前）")
    parser.add_argument("--layers", type=int, nargs="+", required=True, help="パッチ適用層のリスト")
    parser.add_argument("--patch-window", type=int, default=3, help="パッチを適用するトークン幅")
    parser.add_argument("--sequence-length", type=int, default=30, help="生成する最大トークン数")
    parser.add_argument("--alpha", type=float, default=1.0, help="スカラーα（全ステップで固定）")
    parser.add_argument("--max-samples-per-emotion", type=int, default=8, help="neutralプロンプトの最大使用数")
    parser.add_argument("--device", type=str, default=None, help="cuda/mps/cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="生成・評価のバッチサイズ（デフォルト: 8）")
    parser.add_argument("--random-control", action="store_true", help="ランダム方向の対照実験も同時に保存する")
    parser.add_argument("--num-random", type=int, default=3, help="ランダム方向の本数（--random-control時）")
    args = parser.parse_args()

    ctx = ProjectContext(args.profile)
    spec = get_model_spec(args.model)

    dataset_rows = load_dataset_for_profile(args.profile)
    neutral_prompts = _select_neutral_prompts(dataset_rows, args.max_samples_per_emotion)
    if not neutral_prompts:
        raise ValueError("neutral プロンプトが見つかりません。")

    vec_path = ctx.results_dir() / "emotion_vectors" / f"{spec.name}_vectors_token_based.pkl"
    if not vec_path.exists():
        raise FileNotFoundError(f"感情ベクトルが見つかりません: {vec_path}")
    emotion_vectors, meta = _load_emotion_vectors(vec_path)
    layer_indices: List[int] = list(meta.get("layer_indices", []))
    if not layer_indices:
        raise ValueError("感情ベクトルに layer_indices メタデータがありません。")
    vector_map: Dict[str, Dict[int, np.ndarray]] = {}
    for emo, arr in emotion_vectors.items():
        vector_map[emo] = _ensure_layer_vectors(arr, layer_indices)

    device = args.device or get_default_device_str()
    if spec.is_large:
        patcher = LargeActivationPatcher(spec, device=device)
    else:
        patcher = ActivationPatcher(spec.hf_id, device=device)
    alpha_sched = _alpha_schedule(args.alpha, args.sequence_length)
    phase_started = time.perf_counter()

    # 評価器を初期化（スクリプト全体で1回だけ）
    start_time = time.time()
    print(f"[Phase 5] 評価器を初期化中...")
    text_evaluator = TextEvaluator(device=device)
    elapsed = time.time() - start_time
    print(f"[Phase 5] 評価器初期化完了: {elapsed:.2f}秒")

    # ベースライン生成（バッチ処理）
    start_time = time.time()
    print(f"[Phase 5] ベースライン生成を開始... (サンプル数: {len(neutral_prompts)})")
    baseline_texts_list = patcher._generate_text_batch(neutral_prompts, max_new_tokens=args.sequence_length, batch_size=args.batch_size)
    baseline_texts = {prompt: text for prompt, text in zip(neutral_prompts, baseline_texts_list)}
    elapsed = time.time() - start_time
    print(f"[Phase 5] ベースライン生成完了: {elapsed:.2f}秒 ({elapsed/60:.2f}分)")

    # ベースライン評価（バッチ処理）
    start_time = time.time()
    print(f"[Phase 5] ベースライン評価を実行中...")
    baseline_scores = text_evaluator.evaluate_batch(baseline_texts_list, batch_size=args.batch_size)
    # 後方互換性のため、Dict[str, Dict[str, float]]形式に変換
    baseline_metrics: Dict[str, Dict[str, float]] = {}
    for i, prompt in enumerate(neutral_prompts):
        metrics: Dict[str, float] = {}
        for key, values in baseline_scores.items():
            # key format: "sentiment/POSITIVE" -> metrics["sentiment"]["POSITIVE"]
            parts = key.split("/", 1)
            if len(parts) == 2:
                metric_name, label = parts
                if metric_name not in metrics:
                    metrics[metric_name] = {}
                metrics[metric_name][label] = float(values[i])
        baseline_metrics[prompt] = metrics
    elapsed = time.time() - start_time
    print(f"[Phase 5] ベースライン評価完了: {elapsed:.2f}秒 ({elapsed/60:.2f}分)")

    sweep_results: Dict[str, Dict[int, Dict[float, Dict]]] = {}
    emotion_entries: Dict[str, Dict[int, Dict[float, List[Dict]]]] = {}
    rng = np.random.default_rng(seed=0)

    # 総処理数を計算
    total_emotions = len([e for e in vector_map.keys() if e != "neutral"])
    total_layers = len(args.layers)
    total_combinations = total_emotions * total_layers
    processed = 0
    
    phase5_start = time.time()
    print(f"[Phase 5] 残差パッチングを開始... (感情数: {total_emotions}, 層数: {total_layers}, サンプル数: {len(neutral_prompts)})")
    
    # 進捗表示をバッファリング（I/O改善）
    progress_buffer: List[str] = []
    PROGRESS_INTERVAL = 5  # 5件ごとにまとめて出力
    
    for emotion, layer_vecs in vector_map.items():
        if emotion == "neutral":
            continue
        sweep_results[emotion] = {}
        emotion_entries[emotion] = {}
        for layer_idx in args.layers:
            if layer_idx not in layer_vecs:
                continue
            vec = layer_vecs[layer_idx]
            sweep_results[emotion][layer_idx] = {}
            emotion_entries[emotion][layer_idx] = {}

            layer_start = time.time()
            # バッチでパッチ生成
            patched_texts_list = patcher.generate_with_patching_batch(
                prompts=neutral_prompts,
                emotion_vector=vec,
                layer_idx=layer_idx,
                alpha=args.alpha,
                max_new_tokens=args.sequence_length,
                patch_window=args.patch_window,
                patch_positions=None,
                alpha_schedule=alpha_sched,
                alpha_decay_rate=None,
                patch_new_tokens_only=True,
                batch_size=args.batch_size,
            )
            patched_texts = {prompt: text for prompt, text in zip(neutral_prompts, patched_texts_list)}

            # バッチで評価
            patched_scores = text_evaluator.evaluate_batch(patched_texts_list, batch_size=args.batch_size)
            # 後方互換性のため、Dict[str, Dict[str, float]]形式に変換
            metric_map: Dict[str, Dict[str, float]] = {}
            for i, prompt in enumerate(neutral_prompts):
                metrics: Dict[str, float] = {}
                for key, values in patched_scores.items():
                    parts = key.split("/", 1)
                    if len(parts) == 2:
                        metric_name, label = parts
                        if metric_name not in metrics:
                            metrics[metric_name] = {}
                        metrics[metric_name][label] = float(values[i])
                metric_map[prompt] = metrics
            sweep_results[emotion][layer_idx][args.alpha] = {
                "texts": patched_texts,
                "metrics": metric_map,
            }

            # emotion_entriesはランダム対照で使う形に合わせて保持
            emotion_entries[emotion][layer_idx][args.alpha] = [
                {"prompt": p, "text": patched_texts[p], "metrics": metric_map[p]} for p in patched_texts.keys()
            ]
            
            processed += 1
            layer_elapsed = time.time() - layer_start
            total_elapsed = time.time() - phase5_start
            # 進捗情報をバッファに追加（即座に出力しない）
            progress_buffer.append(f"  [Phase 5] {emotion} Layer {layer_idx} 完了: {layer_elapsed:.1f}秒 (進捗: {processed}/{total_combinations}, 累計: {total_elapsed/60:.1f}分)")
            
            # 一定間隔でまとめて出力（I/O改善）
            if len(progress_buffer) >= PROGRESS_INTERVAL or processed == total_combinations:
                print("\n".join(progress_buffer))
                progress_buffer.clear()
    
    phase5_elapsed = time.time() - phase5_start
    print(f"[Phase 5] 残差パッチング完了: {phase5_elapsed:.2f}秒 ({phase5_elapsed/60:.2f}分)")

    out_dir = ctx.results_dir() / "patching" / "residual"
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_payload = {
        "profile": args.profile,
        "model": spec.name,
        "prompts": neutral_prompts,
        "baseline": {"texts": baseline_texts, "metrics": baseline_metrics},
        "sweep_results": sweep_results,
        "alpha_values": [args.alpha],
        "patch_window": args.patch_window,
        "sequence_length": args.sequence_length,
    }
    sweep_path = out_dir / f"{spec.name}_residual_sweep.pkl"
    
    start_time = time.time()
    print(f"[Phase 5] 残差パッチング結果を保存中...")
    with sweep_path.open("wb") as f:
        pickle.dump(sweep_payload, f)
    elapsed = time.time() - start_time
    print(f"[Phase 5] 保存完了: {elapsed:.2f}秒")
    print(f"✓ 残差パッチング結果を保存: {sweep_path}")

    random_path_str = None
    if args.random_control:
        random_start = time.time()
        total_random = total_emotions * total_layers * args.num_random
        print(f"[Phase 5] ランダム対照実験を開始... (総組み合わせ: {total_random})")
        random_results: Dict[str, Dict[int, Dict[int, Dict[float, List[Dict]]]]] = {}
        random_processed = 0
        for emotion, layer_dict in vector_map.items():
            if emotion == "neutral":
                continue
            random_results[emotion] = {}
            for layer_idx in args.layers:
                if layer_idx not in layer_dict:
                    continue
                emo_vec = layer_dict[layer_idx]
                emo_norm = np.linalg.norm(emo_vec) + 1e-8
                random_results[emotion][layer_idx] = {}
                for r_idx in range(args.num_random):
                    rand_vec = rng.standard_normal(emo_vec.shape).astype(np.float32)
                    rand_vec *= emo_norm / (np.linalg.norm(rand_vec) + 1e-8)
                    random_results[emotion][layer_idx][r_idx] = {}

                    # バッチでパッチ生成
                    patched_texts_list = patcher.generate_with_patching_batch(
                        prompts=neutral_prompts,
                        emotion_vector=rand_vec,
                        layer_idx=layer_idx,
                        alpha=args.alpha,
                        max_new_tokens=args.sequence_length,
                        patch_window=args.patch_window,
                        patch_positions=None,
                        alpha_schedule=alpha_sched,
                        alpha_decay_rate=None,
                        patch_new_tokens_only=True,
                        batch_size=args.batch_size,
                    )
                    patched_texts = {prompt: text for prompt, text in zip(neutral_prompts, patched_texts_list)}
                    
                    # バッチで評価
                    patched_scores = text_evaluator.evaluate_batch(patched_texts_list, batch_size=args.batch_size)
                    # 後方互換性のため、Dict形式に変換
                    entries = []
                    for i, prompt in enumerate(neutral_prompts):
                        metrics: Dict[str, float] = {}
                        for key, values in patched_scores.items():
                            parts = key.split("/", 1)
                            if len(parts) == 2:
                                metric_name, label = parts
                                if metric_name not in metrics:
                                    metrics[metric_name] = {}
                                metrics[metric_name][label] = float(values[i])
                        entries.append({"prompt": prompt, "text": patched_texts[prompt], "metrics": metrics})
                    random_results[emotion][layer_idx][r_idx][args.alpha] = entries
                    random_processed += 1
                    # 進捗表示をバッファリング（I/O改善）
                    if random_processed % 10 == 0 or random_processed == total_random:
                        elapsed = time.time() - random_start
                        print(f"  [Phase 5] ランダム対照進捗: {random_processed}/{total_random} ({elapsed/60:.1f}分)")

        random_elapsed = time.time() - random_start
        print(f"[Phase 5] ランダム対照実験完了: {random_elapsed:.2f}秒 ({random_elapsed/60:.2f}分)")

        rand_dir = ctx.results_dir() / "patching_random"
        rand_dir.mkdir(parents=True, exist_ok=True)
        rand_payload = {
            "profile": args.profile,
            "model": spec.name,
            "alpha_values": [args.alpha],
            "random_results": random_results,
            "emotion_results": emotion_entries,
            "num_random": args.num_random,
        }
        rand_path = rand_dir / f"{spec.name}_random_sweep.pkl"
        
        start_time = time.time()
        print(f"[Phase 5] ランダム対照結果を保存中...")
        with rand_path.open("wb") as f:
            pickle.dump(rand_payload, f)
        elapsed = time.time() - start_time
        print(f"[Phase 5] 保存完了: {elapsed:.2f}秒")
        print(f"✓ ランダム対照を保存: {rand_path}")
        random_path_str = str(rand_path)

    record_phase_timing(
        context=ctx,
        phase="phase5",
        started_at=phase_started,
        model=spec.name,
        device=device,
        samples=len(neutral_prompts),
        metadata={
            "layers": args.layers,
            "patch_window": args.patch_window,
            "sequence_length": args.sequence_length,
            "alpha": args.alpha,
            "batch_size": args.batch_size,
            "random_control": args.random_control,
            "num_random": args.num_random if args.random_control else 0,
            "result_path": str(sweep_path),
            "random_result_path": random_path_str,
        },
        cli_args=sys.argv[1:],
    )

if __name__ == "__main__":
    main()
