"""Phase 8 pipeline for large models (Llama3/Gemma/Qwen).

主な用途:
- Llama3 8B などの大規模モデルで、感情トークンベースのベクトル → 多サンプルPCAサブスペース → GPT-2 とのサブスペースアライメント(before/after)を実行。
- 出力は `results/<profile>/emotion_vectors/`, `emotion_subspaces/`, `alignment/` に保存。

実行例（CPU）:
    # CPU example
    python -m src.analysis.run_phase8_pipeline \\
      --profile baseline \\
      --large-model llama3_8b \\
      --layers 0 1 2 3 4 5 6 7 8 9 10 11 \\
      --n-components 8 \\
      --max-samples-per-emotion 32 \\
      --device cpu

Apple Silicon (MPS が有効な場合):
    # Apple Silicon example (M2/M4 etc., if torch.backends.mps.is_available())
    python -m src.analysis.run_phase8_pipeline \\
      --profile baseline \\
      --large-model llama3_8b \\
      --layers 0 1 2 3 4 5 6 7 8 9 10 11 \\
      --n-components 8 \\
      --max-samples-per-emotion 32 \\
      --device mps

注意:
- CUDAは必須ではありません。CPUまたはApple SiliconのMPSでも実行可能です。
- 重い実行の場合は、--layers で層数を制限するか、--max-samples-per-emotion でサンプル数を制限することを推奨します。
- CUDA環境であれば --device cuda も指定できます。
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA

from src.config.project_profiles import EMOTION_LABELS, list_profiles
from src.utils.project_context import ProjectContext, profile_help_text
from src.models.phase8_large.registry import MODEL_REGISTRY, LargeModelName
from src.models.phase8_large.hf_wrapper import LargeHFModel, load_large_model
from src.analysis.emotion_vectors_token_based import EMOTION_TOKENS
from src.analysis.subspace_utils import compute_subspace_overlap


def _load_prompts(context: ProjectContext, max_samples: int | None) -> Dict[str, List[str]]:
    prompts: Dict[str, List[str]] = {}
    for emotion in EMOTION_LABELS:
        path = context.prompt_file(emotion)
        with open(path, "r", encoding="utf-8") as f:
            import json

            data = json.load(f)
            plist = data.get("prompts", [])
            if max_samples:
                plist = plist[:max_samples]
            prompts[emotion] = plist
    return prompts


def _find_emotion_positions(token_strings: List[str], emotion_label: str) -> List[int]:
    """Reuse the token-based emotion heuristic."""
    if emotion_label == "neutral":
        return [len(token_strings) - 1]
    tokens = EMOTION_TOKENS.get(emotion_label, set())
    positions: List[int] = []
    for idx, tok in enumerate(token_strings):
        normalized = tok.lower().strip(".,!?:;\"' ")
        if normalized in tokens:
            positions.append(idx)
    if not positions:
        positions = [len(token_strings) - 1]
    return positions


def _token_strings(tokenizer, ids: np.ndarray, pad_token_id: int) -> List[str]:
    # Trim padding to avoid counting pad positions
    if pad_token_id in ids:
        last_non_pad = int(np.max(np.nonzero(ids != pad_token_id)))
        trimmed = ids[: last_non_pad + 1]
    else:
        trimmed = ids
    return tokenizer.convert_ids_to_tokens(trimmed.tolist())


def compute_emotion_vectors_large(
    model: LargeHFModel,
    prompts: Dict[str, List[str]],
    layers: Sequence[int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[int, np.ndarray]], Dict[int, np.ndarray]]:
    """
    Compute token-based emotion vectors and per-sample差分を返す。

    Returns:
        mean_vectors: emotion -> [len(layers), d_model]
        per_sample_diffs: emotion -> {layer: [n_samples, d_model]}
        neutral_vectors: {layer: [n_samples, d_model]}  # neutral token activations
    """
    tokenizer = model.tokenizer
    pad_id = tokenizer.pad_token_id
    mean_vectors: Dict[str, np.ndarray] = {}
    per_sample_diffs: Dict[str, Dict[int, np.ndarray]] = {}
    neutral_vectors: Dict[int, np.ndarray] = {}

    # 先にneutralの活性を取得（全てのemotionで共通利用）
    neutral_prompts = prompts["neutral"]
    neu_batch = model.get_resid_activations(neutral_prompts, layers=layers, hook_pos="resid_post")
    for layer in layers:
        neu_samples: List[np.ndarray] = []
        for i, prompt in enumerate(neutral_prompts):
            tokens = _token_strings(tokenizer, neu_batch.token_ids[i].numpy(), pad_id)
            pos = _find_emotion_positions(tokens, "neutral")
            acts = neu_batch.layer_activations[layer][i]
            neu_samples.append(np.mean(acts[pos, :].numpy(), axis=0))
        neutral_vectors[layer] = np.stack(neu_samples, axis=0)

    for emotion in ["gratitude", "anger", "apology"]:
        emo_prompts = prompts[emotion]
        pair_count = min(len(emo_prompts), len(neutral_prompts))
        emo_prompts = emo_prompts[:pair_count]
        neu_prompts_cut = neutral_prompts[:pair_count]

        emo_batch = model.get_resid_activations(emo_prompts, layers=layers, hook_pos="resid_post")
        neu_batch_cut = model.get_resid_activations(neu_prompts_cut, layers=layers, hook_pos="resid_post")

        layer_vecs: List[np.ndarray] = []
        per_sample_diffs[emotion] = {}

        for layer in layers:
            diff_list: List[np.ndarray] = []
            for i in range(pair_count):
                tokens_emo = _token_strings(tokenizer, emo_batch.token_ids[i].numpy(), pad_id)
                tokens_neu = _token_strings(tokenizer, neu_batch_cut.token_ids[i].numpy(), pad_id)
                pos_emo = _find_emotion_positions(tokens_emo, emotion)
                pos_neu = _find_emotion_positions(tokens_neu, "neutral")
                acts_emo = emo_batch.layer_activations[layer][i]
                acts_neu = neu_batch_cut.layer_activations[layer][i]
                emo_vec = np.mean(acts_emo[pos_emo, :].numpy(), axis=0)
                neu_vec = np.mean(acts_neu[pos_neu, :].numpy(), axis=0)
                diff = emo_vec - neu_vec
                diff_list.append(diff)
            diff_arr = np.stack(diff_list, axis=0)  # [n_samples, d_model]
            per_sample_diffs[emotion][layer] = diff_arr
            layer_vecs.append(np.mean(diff_arr, axis=0))
        mean_vectors[emotion] = np.stack(layer_vecs, axis=0)  # [len(layers), d_model]

    return mean_vectors, per_sample_diffs, neutral_vectors


def save_emotion_vectors(
    context: ProjectContext,
    model_name: str,
    vectors: Dict[str, np.ndarray],
    layers: Sequence[int],
    prompts_per_emotion: int,
    per_sample_diffs: Dict[str, Dict[int, np.ndarray]] | None = None,
) -> Path:
    out_dir = context.results_dir() / "emotion_vectors"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name}_vectors_token_based.pkl"
    metadata = {
        "model_name": model_name,
        "n_layers": len(layers),
        "layers": list(layers),
        "d_model": vectors["gratitude"].shape[-1],
        "n_samples": prompts_per_emotion,
        "position_strategy": "first_token_match",
        "save_residual_stream": True,
    }
    distances = {}
    emotions = list(vectors.keys())
    for i, e1 in enumerate(emotions):
        for e2 in emotions[i + 1 :]:
            v1 = vectors[e1]
            v2 = vectors[e2]
            dots = np.sum(v1 * v2, axis=-1)
            norms = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
            sims = np.where(norms > 0, dots / norms, 0.0)
            distances[f"{e1}_vs_{e2}"] = sims
    payload = {
        "emotion_vectors": vectors,
        "emotion_distances": distances,
        "metadata": metadata,
        "position_strategy": "first",
        "use_mlp": False,
    }
    if per_sample_diffs:
        payload["per_sample_diffs"] = per_sample_diffs
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def compute_subspaces(
    per_sample_diffs: Dict[str, Dict[int, np.ndarray]],
    layers: Sequence[int],
    n_components: int = 10,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    真の多サンプルPCAでサブスペースを計算。
    Returns dict with keys subspaces/emotion -> [len(layers), n_components, d_model]
    """
    emotions = list(per_sample_diffs.keys())
    # 推定されるd_modelは最初の要素から取得
    first_emotion = emotions[0]
    first_layer = layers[0]
    d_model = per_sample_diffs[first_emotion][first_layer].shape[-1]
    subspaces: Dict[str, np.ndarray] = {}
    explained: Dict[str, np.ndarray] = {}
    for emotion in emotions:
        layer_bases: List[np.ndarray] = []
        layer_exps: List[np.ndarray] = []
        for layer in layers:
            X = per_sample_diffs[emotion][layer]  # [n_samples, d_model]
            X = X - np.mean(X, axis=0, keepdims=True)
            k = min(n_components, X.shape[0], X.shape[1])
            pca = PCA(n_components=k)
            pca.fit(X)
            basis = np.zeros((n_components, d_model))
            basis[:k, :] = pca.components_
            exp_ratio = np.zeros(n_components)
            exp_ratio[:k] = pca.explained_variance_ratio_
            layer_bases.append(basis)
            layer_exps.append(exp_ratio)
        subspaces[emotion] = np.stack(layer_bases, axis=0)
        explained[emotion] = np.stack(layer_exps, axis=0)
    return {"subspaces": subspaces, "explained_variances": explained, "d_model": d_model}


def save_subspaces(
    context: ProjectContext,
    model_name: str,
    subspace_dict: Dict[str, Dict[str, np.ndarray]],
    layers: Sequence[int],
) -> Path:
    out_dir = context.results_dir() / "emotion_subspaces"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name}_subspaces.pkl"
    payload = {
        "subspaces": subspace_dict["subspaces"],
        "explained_variances": subspace_dict["explained_variances"],
        "metadata": {
            "model_name": model_name,
            "n_components": next(iter(subspace_dict["subspaces"].values())).shape[1],
            "layers": list(layers),
        },
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def load_reference_subspaces(context: ProjectContext, model_name: str = "gpt2") -> Tuple[Dict[str, np.ndarray], List[int]]:
    path = context.results_dir() / "emotion_subspaces" / f"{model_name}_subspaces.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    layers = data.get("metadata", {}).get("layers")
    if layers is None:
        # Assume contiguous layers from 0
        n_layers = next(iter(data["subspaces"].values())).shape[0]
        layers = list(range(n_layers))
    return data["subspaces"], layers


def compute_alignment_overlaps(
    ref_subspaces: Dict[str, np.ndarray],
    ref_layers: List[int],
    target_subspaces: Dict[str, np.ndarray],
    target_layers: List[int],
    ref_neutral: Dict[int, np.ndarray],
    target_neutral: Dict[int, np.ndarray],
) -> Dict:
    """
    ref = gpt2 (d_ref), target = llama (d_tgt, larger).
    Overlap before: truncate target basis to d_ref.
    Overlap after: map target basis via Procrustes learned on neutral subspaces.
    """
    d_ref = next(iter(ref_subspaces.values())).shape[-1]
    overlaps: Dict = {"layers": []}
    common_layers = sorted(set(ref_layers) & set(target_layers))
    ref_layer_to_idx = {l: i for i, l in enumerate(ref_layers)}
    tgt_layer_to_idx = {l: i for i, l in enumerate(target_layers)}
    for layer in common_layers:
        layer_entry = {"layer": layer, "emotions": {}}
        ref_idx = ref_layer_to_idx[layer]
        tgt_idx = tgt_layer_to_idx[layer]
        # 学習用: neutral サンプル
        tgt_neu = target_neutral[layer]  # [n_samples, d_tgt]
        ref_neu = ref_neutral[layer]  # [n_samples, d_ref]
        m = min(tgt_neu.shape[0], ref_neu.shape[0])
        tgt_neu = tgt_neu[:m].astype(np.float64)
        ref_neu = ref_neu[:m].astype(np.float64)
        # 中心化
        tgt_neu_c = tgt_neu - np.mean(tgt_neu, axis=0, keepdims=True)
        ref_neu_c = ref_neu - np.mean(ref_neu, axis=0, keepdims=True)
        # 最小二乗で線形マップ W (d_tgt -> d_ref) を学習: W = pinv(X) @ Y
        W = np.linalg.pinv(tgt_neu_c) @ ref_neu_c  # (d_tgt, d_ref)

        for emotion in target_subspaces.keys():
            tgt_basis = target_subspaces[emotion][tgt_idx]  # (k, d_tgt)
            ref_basis = ref_subspaces.get(emotion)
            if ref_basis is None:
                continue
            ref_basis_layer = ref_basis[ref_idx]  # (k, d_ref)
            # before: 単純に次元をtruncate
            tgt_trunc = tgt_basis[:, :d_ref]
            overlap_before = compute_subspace_overlap(ref_basis_layer, tgt_trunc, method="cos_squared")
            # after: 学習した線形マップでref空間に射影
            mapped = tgt_basis @ W  # (k, d_ref)
            overlap_after = compute_subspace_overlap(ref_basis_layer, mapped, method="cos_squared")
            layer_entry["emotions"][emotion] = {
                "overlap_before": float(overlap_before),
                "overlap_after": float(overlap_after),
            }
        overlaps.setdefault("per_layer", []).append(layer_entry)
    return overlaps


def save_alignment(
    context: ProjectContext,
    source_model: str,
    target_model: str,
    overlaps: Dict,
    suffix: str = "full",
    k: int | None = None,
) -> Path:
    out_dir = context.results_dir() / "alignment"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{source_model}_vs_{target_model}_token_based_{suffix}.pkl"
    with open(path, "wb") as f:
        payload = {"overlaps": overlaps, "source_model": source_model, "target_model": target_model}
        if k is not None:
            payload["k"] = k
        pickle.dump(payload, f)
    return path


def run_phase8_phases(
    model_name: LargeModelName,
    profile: str,
    device: str,
    layers: Sequence[int],
    max_samples: int | None,
    n_components: int = 10,
) -> None:
    context = ProjectContext(profile_name=profile)
    spec = MODEL_REGISTRY[model_name]
    if spec.family != "llama3":
        raise NotImplementedError(f"{model_name} is stubbed; implement loading before use.")
    model = load_large_model(spec, device=device)
    prompts = _load_prompts(context, max_samples)
    vectors, per_sample_diffs, neutral_vectors = compute_emotion_vectors_large(model, prompts, layers)
    vec_path = save_emotion_vectors(context, model_name, vectors, layers, len(prompts["gratitude"]), per_sample_diffs)
    print(f"Saved emotion vectors to {vec_path}")

    subspaces = compute_subspaces(per_sample_diffs, layers, n_components=n_components)
    sub_path = save_subspaces(context, model_name, subspaces, layers)
    print(f"Saved subspaces to {sub_path}")

    ref_sub, ref_layers = load_reference_subspaces(context, "gpt2")
    overlaps_simple = compute_alignment_overlaps(
        ref_subspaces=ref_sub,
        ref_layers=ref_layers,
        target_subspaces=subspaces["subspaces"],
        target_layers=list(layers),
        ref_neutral=_load_neutral_vectors_gpt2(context, ref_layers, n_components),
        target_neutral=_prepare_neutral_for_layers(neutral_vectors, layers, n_components),
    )
    align_path = save_alignment(context, "gpt2", model_name, overlaps_simple, suffix="full", k=n_components)
    print(f"Saved alignment overlaps to {align_path}")


def _prepare_neutral_for_layers(neutral_vectors: Dict[int, np.ndarray], layers: Sequence[int], n_components: int) -> Dict[int, np.ndarray]:
    """Ensure neutral vectors exist for required layers (identity if missing)."""
    prepared: Dict[int, np.ndarray] = {}
    for layer in layers:
        if layer in neutral_vectors:
            prepared[layer] = neutral_vectors[layer]
    return prepared


def _load_neutral_vectors_gpt2(context: ProjectContext, layers: List[int], n_components: int) -> Dict[int, np.ndarray]:
    """
    Load neutral activations for gpt2 and compute PCA-basis-aligned vectors.
    We return raw neutral vectors (not PCA) because Procrustes expects samples.
    """
    path = context.results_dir() / "activations" / "gpt2" / "activations_neutral.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Neutral activations not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    residuals = data["residual_stream"]  # list over layers, each list of samples [seq, d_model]
    gpt2_layers = len(residuals)
    vectors: Dict[int, np.ndarray] = {}
    for layer in layers:
        if layer >= gpt2_layers:
            continue
        samples = []
        for sample in residuals[layer]:
            samples.append(sample[-1])  # 文末トークン
        vectors[layer] = np.stack(samples, axis=0)
    return vectors


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 8 pipeline for large models")
    parser.add_argument("--profile", type=str, choices=list_profiles(), default="baseline", help=f"Dataset profile ({profile_help_text()})")
    parser.add_argument("--large-model", type=str, choices=list(MODEL_REGISTRY.keys()), default="llama3_8b")
    parser.add_argument("--phases", type=str, nargs="+", default=["3", "3.5", "6"], help="Phases to run (currently non-branching).")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 3, 6, 12, 24, 31], help="Layers to sample.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples-per-emotion", type=int, default=None)
    parser.add_argument("--n-components", type=int, default=10, help="PCA components for subspaces.")
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    run_phase8_phases(
        model_name=args.large_model,  # type: ignore[arg-type]
        profile=args.profile,
        device=args.device,
        layers=args.layers,
        max_samples=args.max_samples_per_emotion,
        n_components=args.n_components,
    )


if __name__ == "__main__":
    main()
