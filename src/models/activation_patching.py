"""
Activation Patching実装
中立文に対する推論時にresidual streamを改変して、感情方向の操作が出力に与える因果的影響を検証
"""
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from transformer_lens import HookedTransformer
from tqdm import tqdm


class ActivationPatcher:
    """Activation Patchingを実行するクラス"""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        初期化
        
        Args:
            model_name: モデル名
            device: 使用するデバイス
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model: {model_name}")
        self.model = HookedTransformer.from_pretrained(model_name, device=self.device)
        self.model.eval()
        self.generation_config = {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": None,
            "stop_at_eos": True,
            "return_type": "tokens",
        }
        print(f"✓ Model loaded on {self.device}")
    
    def load_emotion_vectors(self, vectors_file: Path) -> Dict[str, np.ndarray]:
        """
        感情方向ベクトルを読み込む
        
        Args:
            vectors_file: ベクトルファイルのパス
            
        Returns:
            感情ラベルをキーとする方向ベクトルの辞書
        """
        with open(vectors_file, 'rb') as f:
            data = pickle.load(f)
        
        return data['emotion_vectors']
    
    def _decode_new_tokens(self, tokens: torch.Tensor, prompt_length: int) -> str:
        """Decode only the newly generated continuations (excluding the prompt)."""
        new_tokens = tokens[:, prompt_length:]
        if new_tokens.shape[1] == 0:
            return ""
        token_ids = new_tokens[0].tolist()
        decoded = self.model.tokenizer.decode(token_ids, skip_special_tokens=True)
        return decoded.strip()

    def _build_alpha_schedule(
        self,
        base_alpha: float,
        max_steps: int,
        alpha_schedule: Optional[Sequence[float]] = None,
        decay_rate: Optional[float] = None,
    ) -> List[float]:
        """Return alpha per generation step."""
        schedule: List[float] = []
        for step in range(max_steps):
            if alpha_schedule:
                if step < len(alpha_schedule):
                    schedule.append(float(alpha_schedule[step]))
                else:
                    schedule.append(float(alpha_schedule[-1]))
            elif decay_rate is not None:
                schedule.append(float(base_alpha * (decay_rate ** step)))
            else:
                schedule.append(float(base_alpha))
        return schedule

    def _resolve_patch_positions(
        self,
        seq_len: int,
        prompt_len: int,
        patch_window: Optional[int],
        patch_positions: Optional[Sequence[int]],
        patch_new_tokens_only: bool,
    ) -> List[int]:
        """Return absolute token indices to patch for the current forward pass."""
        positions: List[int] = []

        if patch_positions:
            for pos in patch_positions:
                idx = pos if pos >= 0 else seq_len + pos
                if patch_new_tokens_only and idx < prompt_len:
                    continue
                if 0 <= idx < seq_len:
                    positions.append(idx)
            if positions:
                return sorted(set(positions))

        window = patch_window or 1
        start = max(seq_len - window, 0)
        positions = list(range(start, seq_len))
        if patch_new_tokens_only:
            positions = [idx for idx in positions if idx >= prompt_len]
        if not positions:
            positions = [seq_len - 1]
        return positions

    def _generate_tokens_with_patch(
        self,
        prompt: str,
        layer_idx: int,
        emotion_vector: np.ndarray,
        alpha_schedule: List[float],
        patch_window: Optional[int],
        patch_positions: Optional[Sequence[int]],
        patch_new_tokens_only: bool,
        max_new_tokens: int,
    ) -> Tuple[torch.Tensor, int]:
        """Generate tokens while applying the residual patch at every step."""
        tokens = self.model.to_tokens(prompt)
        prompt_len = tokens.shape[1]
        patch_vector = torch.tensor(emotion_vector, device=self.device, dtype=self.model.cfg.dtype)

        def patch_hook(activation, hook):
            if hook.name != f"blocks.{layer_idx}.hook_resid_pre":
                return activation
            if activation.shape[1] == 0:
                return activation
            seq_len = activation.shape[1]
            generated_len = max(seq_len - prompt_len, 0)
            step = min(generated_len, max_new_tokens - 1)
            alpha_value = alpha_schedule[step] if alpha_schedule else 0.0
            if alpha_value == 0.0:
                return activation
            indices = self._resolve_patch_positions(
                seq_len,
                prompt_len,
                patch_window,
                patch_positions,
                patch_new_tokens_only,
            )
            if not indices:
                return activation
            activation = activation.clone()
            for idx in indices:
                if 0 <= idx < activation.shape[1]:
                    activation[0, idx, :] += alpha_value * patch_vector.to(activation.dtype)
            return activation

        hook_name = f"blocks.{layer_idx}.hook_resid_pre"
        handle = self.model.add_hook(hook_name, patch_hook)
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    tokens,
                    max_new_tokens=max_new_tokens,
                    **self.generation_config,
                )
        finally:
            if hook_name in self.model.hook_dict:
                hook_point = self.model.hook_dict[hook_name]
                hook_point.fwd_hooks = []
        return generated, prompt_len

    def _generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 20,
    ) -> str:
        """Deterministic generation without patching."""
        tokens = self.model.to_tokens(prompt)
        with torch.no_grad():
            generated = self.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                **self.generation_config,
            )
        continuation = self._decode_new_tokens(generated, tokens.shape[1])
        return (prompt + " " + continuation).strip()
    
    def generate_with_patching(
        self,
        prompt: str,
        emotion_vector: np.ndarray,
        layer_idx: int,
        alpha: float = 1.0,
        max_new_tokens: int = 20,
        patch_window: Optional[int] = None,
        patch_positions: Optional[Sequence[int]] = None,
        alpha_schedule: Optional[Sequence[float]] = None,
        alpha_decay_rate: Optional[float] = None,
        patch_new_tokens_only: bool = False,
    ) -> str:
        """
        パッチを適用してテキストを生成（multi-token対応版）
        
        Args:
            prompt: 入力プロンプト
            emotion_vector: 感情方向ベクトル [d_model]（特定の層のベクトル）
            layer_idx: パッチを適用する層
            alpha: パッチの強度
            max_new_tokens: 生成する最大トークン数
            patch_window: 末尾から何トークン分をパッチするか（デフォルト: 1）
            patch_positions: 明示的にパッチする位置（負数は末尾から）
            alpha_schedule: 各生成ステップのα（リスト指定）
            alpha_decay_rate: αを逓減させる場合のレート
            patch_new_tokens_only: Trueの場合、生成済み（prompt）トークンには適用しない
        """
        schedule = self._build_alpha_schedule(
            base_alpha=alpha,
            max_steps=max_new_tokens,
            alpha_schedule=alpha_schedule,
            decay_rate=alpha_decay_rate,
        )
        generated, prompt_len = self._generate_tokens_with_patch(
            prompt=prompt,
            layer_idx=layer_idx,
            emotion_vector=emotion_vector,
            alpha_schedule=schedule,
            patch_window=patch_window,
            patch_positions=patch_positions,
            patch_new_tokens_only=patch_new_tokens_only,
            max_new_tokens=max_new_tokens,
        )
        continuation = self._decode_new_tokens(generated, prompt_len)
        return (prompt + " " + continuation).strip()
    
    def evaluate_patching_effect(
        self,
        prompts: List[str],
        emotion_vectors: Dict[str, np.ndarray],
        layer_idx: int = 6,
        alpha_values: List[float] = [0.0, 0.5, 1.0, 1.5, -0.5, -1.0],
        max_new_tokens: int = 20,
        patch_window: Optional[int] = None,
        patch_positions: Optional[Sequence[int]] = None,
        alpha_schedule: Optional[Sequence[float]] = None,
        alpha_decay_rate: Optional[float] = None,
        patch_new_tokens_only: bool = False,
    ) -> Dict:
        """
        パッチングの効果を評価
        
        Args:
            prompts: 入力プロンプトのリスト
            emotion_vectors: 感情ラベルをキーとする方向ベクトル [n_layers, d_model]
            layer_idx: パッチを適用する層
            alpha_values: パッチの強度のリスト
            
        Returns:
            評価結果の辞書
        """
        results = {
            'prompts': prompts,
            'baseline': {},
            'patched': {}
        }
        
        # Baseline（パッチなし）の生成
        print("Generating baseline outputs...")
        for prompt in tqdm(prompts, desc="Baseline"):
            results['baseline'][prompt] = self._generate_text(
                prompt,
                max_new_tokens=max_new_tokens,
            )
        
        # 各感情方向でパッチング
        for emotion_label, emotion_vec in emotion_vectors.items():
            layer_vector = emotion_vec[layer_idx]  # [d_model]
            
            results['patched'][emotion_label] = {}
            
            for alpha in alpha_values:
                print(f"\nPatching {emotion_label} with alpha={alpha}...")
                alpha_results = {}
                
                for prompt in tqdm(prompts, desc=f"{emotion_label} (α={alpha})"):
                    try:
                        generated = self.generate_with_patching(
                            prompt,
                            layer_vector,
                            layer_idx,
                            alpha,
                            max_new_tokens=max_new_tokens,
                            patch_window=patch_window,
                            patch_positions=patch_positions,
                            alpha_schedule=alpha_schedule,
                            alpha_decay_rate=alpha_decay_rate,
                            patch_new_tokens_only=patch_new_tokens_only,
                        )
                        alpha_results[prompt] = generated
                    except Exception as e:
                        print(f"Error with prompt '{prompt}': {e}")
                        alpha_results[prompt] = f"ERROR: {str(e)}"
                
                results['patched'][emotion_label][alpha] = alpha_results
        
        return results


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Activation patching experiment")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--vectors_file", type=str, required=True, help="Emotion vectors file")
    parser.add_argument("--prompts_file", type=str, default=None, help="Prompts file (JSON)")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--layer", type=int, default=6, help="Layer index for patching")
    parser.add_argument("--alpha", type=float, nargs='+', default=[0.0, 0.5, 1.0, 1.5, -0.5, -1.0], help="Alpha values")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--patch-window", type=int, default=None, help="Patch the last N positions each step")
    parser.add_argument("--patch-positions", type=int, nargs='+', default=None, help="Explicit token positions to patch (supports negative indices)")
    parser.add_argument("--patch-new-only", action="store_true", help="Patch only newly generated tokens")
    parser.add_argument("--alpha-schedule", type=float, nargs='+', default=None, help="Alpha schedule per generation step")
    parser.add_argument("--alpha-decay-rate", type=float, default=None, help="Decay rate for alpha per generation step (e.g., 0.9)")
    
    args = parser.parse_args()
    
    # パッチャーを作成
    patcher = ActivationPatcher(args.model)
    
    # 感情ベクトルを読み込み
    emotion_vectors = patcher.load_emotion_vectors(Path(args.vectors_file))
    print(f"Loaded emotion vectors: {list(emotion_vectors.keys())}")
    
    # プロンプトを読み込みまたはデフォルトを使用
    if args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            prompts_data = json.load(f)
            prompts = prompts_data.get('prompts', [])
    else:
        # デフォルトのプロンプト
        prompts = [
            "The weather today is",
            "I need to check",
            "Can you help me with",
            "What is the best way to",
            "I would like to know about"
        ]
    
    print(f"Using {len(prompts)} prompts")
    
    # パッチング実験を実行
    results = patcher.evaluate_patching_effect(
        prompts,
        emotion_vectors,
        layer_idx=args.layer,
        alpha_values=args.alpha,
        max_new_tokens=args.max_new_tokens,
        patch_window=args.patch_window,
        patch_positions=args.patch_positions,
        alpha_schedule=args.alpha_schedule,
        alpha_decay_rate=args.alpha_decay_rate,
        patch_new_tokens_only=args.patch_new_only,
    )
    
    # 結果を保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to: {output_path}")
    
    # 結果を表示
    print("\n" + "=" * 80)
    print("Patching Results Summary:")
    print("=" * 80)
    
    for prompt in prompts[:3]:  # 最初の3つを表示
        print(f"\nPrompt: {prompt}")
        print(f"Baseline: {results['baseline'][prompt][:100]}...")
        
        for emotion_label in emotion_vectors.keys():
            if emotion_label in results['patched']:
                print(f"\n{emotion_label.upper()} (α=1.0):")
                if 1.0 in results['patched'][emotion_label]:
                    patched_text = results['patched'][emotion_label][1.0][prompt]
                    print(f"  {patched_text[:100]}...")


if __name__ == "__main__":
    main()
