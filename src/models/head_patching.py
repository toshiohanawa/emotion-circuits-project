"""
Head patching実験
感謝文のあるhead出力を、中立文に「移植」するSwap Patchingのhead版
"""
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence
from transformer_lens import HookedTransformer
from tqdm import tqdm

from src.analysis.sentiment_eval import SentimentEvaluator


class HeadPatcher:
    """Head patchingを実行するクラス"""
    
    def __init__(self, model_name: str, device: Optional[str] = None, use_attn_result: bool = True):
        """
        初期化
        
        Args:
            model_name: モデル名（例: "gpt2"）
            device: 使用するデバイス
            use_attn_result: head出力を直接操作するための設定（Trueの場合、attn.hook_resultを有効化）
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_attn_result = use_attn_result
        
        print(f"Loading model: {model_name}")
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=self.device,
        )
        # Set use_attn_result after loading (not a from_pretrained parameter)
        self.model.cfg.use_attn_result = use_attn_result
        self.model.eval()
        self.generation_config = {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": None,
            "stop_at_eos": True,
            "return_type": "tokens",
        }
        
        print(f"✓ Model loaded on {self.device}")
        
        # Sentiment評価器を初期化
        self.sentiment_evaluator = SentimentEvaluator(model_name, device=device)
    
    def parse_head_spec(self, head_spec: str) -> List[Tuple[int, int]]:
        """
        Head指定文字列をパース
        
        Args:
            head_spec: "layer:head"のカンマ区切り（例: "3:5,7:2"）
            
        Returns:
            [(layer_idx, head_idx), ...] のリスト
        """
        heads = []
        for spec in head_spec.split(','):
            spec = spec.strip()
            if ':' in spec:
                layer_str, head_str = spec.split(':')
                try:
                    layer_idx = int(layer_str.strip())
                    head_idx = int(head_str.strip())
                    heads.append((layer_idx, head_idx))
                except ValueError:
                    print(f"Warning: Invalid head spec '{spec}', skipping...")
        return heads
    
    def capture_emotion_head_outputs(
        self,
        emotion_prompts: List[str],
        head_specs: List[Tuple[int, int]],
        position: int = -1
    ) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:
        """
        感情プロンプトでhead出力/パターン/Vをキャプチャし、全プロンプト平均を返す
        """
        captured_outputs = {(layer, head): {"pattern": [], "v": [], "head_output": []} for layer, head in head_specs}
        
        cache = {}
        
        def capture_pattern_hook(activation, hook):
            layer_idx = int(hook.name.split('.')[1])
            cache.setdefault(layer_idx, {})['pattern'] = activation.detach().clone()
            return activation
        
        def capture_v_hook(activation, hook):
            layer_idx = int(hook.name.split('.')[1])
            cache.setdefault(layer_idx, {})['v'] = activation.detach().clone()
            return activation
        
        hook_handles = []
        for layer_idx, head_idx in head_specs:
            pattern_hook_name = f"blocks.{layer_idx}.attn.hook_pattern"
            v_hook_name = f"blocks.{layer_idx}.attn.hook_v"
            handle1 = self.model.add_hook(pattern_hook_name, capture_pattern_hook)
            handle2 = self.model.add_hook(v_hook_name, capture_v_hook)
            hook_handles.append((pattern_hook_name, handle1))
            hook_handles.append((v_hook_name, handle2))
        
        try:
            with torch.no_grad():
                for prompt in tqdm(emotion_prompts, desc="Capturing emotion head outputs"):
                    tokens = self.model.to_tokens(prompt)
                    cache.clear()
                    _ = self.model(tokens)
                    
                    for layer_idx, head_idx in head_specs:
                        c = cache.get(layer_idx, {})
                        if 'pattern' not in c or 'v' not in c:
                            continue
                        pattern = c['pattern']  # [batch, head, pos, pos]
                        v = c['v']  # [batch, pos, head, d_head]
                        v_reshaped = v.permute(0, 2, 1, 3)  # [batch, head, pos, d_head]
                        head_output = torch.einsum('bhqp,bhpd->bhqd', pattern, v_reshaped)  # [batch, head, pos, d_head]
                        
                        pos = -1 if position >= head_output.shape[2] else position
                        head_output_value = head_output[0, head_idx, pos, :].clone()
                        
                        captured_outputs[(layer_idx, head_idx)]["pattern"].append(pattern[0, head_idx].clone())
                        captured_outputs[(layer_idx, head_idx)]["v"].append(v[0, :, head_idx, :].clone())
                        captured_outputs[(layer_idx, head_idx)]["head_output"].append(head_output_value)
        finally:
            for hook_name, handle in hook_handles:
                if hook_name in self.model.hook_dict:
                    hook_point = self.model.hook_dict[hook_name]
                    hook_point.fwd_hooks = []
        
        # 平均を取る
        averaged = {}
        for key, data in captured_outputs.items():
            averaged[key] = {}
            for k2, lst in data.items():
                if lst:
                    averaged[key][k2] = torch.mean(torch.stack(lst), dim=0)
        return averaged
    
    def _generate_tokens(
        self,
        tokens: torch.Tensor,
        hook_fns: List[Tuple[str, callable]],
        max_new_tokens: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        handles = []
        for hook_name, hook_fn in hook_fns:
            handle = self.model.add_hook(hook_name, hook_fn)
            handles.append((hook_name, handle))
        try:
            # Apply overrides if provided
            gen_cfg = self.generation_config.copy()
            if temperature is not None:
                gen_cfg["temperature"] = float(temperature)
            if top_p is not None:
                gen_cfg["top_p"] = float(top_p)
            
            with torch.no_grad():
                generated = self.model.generate(
                    tokens,
                    max_new_tokens=max_new_tokens,
                    **gen_cfg,
                )
        finally:
            for hook_name, handle in handles:
                if hook_name in self.model.hook_dict:
                    hook_point = self.model.hook_dict[hook_name]
                    hook_point.fwd_hooks = []
        return generated

    def generate_with_patching(
        self,
        neutral_prompt: str,
        emotion_head_outputs: Dict[Tuple[int, int], Dict[str, torch.Tensor]],
        max_new_tokens: int = 30,
        position: int = -1,
        patch_mode: str = "v_only",
        qk_overrides: Optional[Dict[Tuple[int, int], Dict[str, torch.Tensor]]] = None,
        ov_overrides: Optional[Dict[Tuple[int, int], torch.Tensor]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Head patchingを適用して生成
        
        Args:
            neutral_prompt: 中立プロンプト
            emotion_head_outputs: 感情プロンプトから取得したhead出力
            max_new_tokens: 生成する最大トークン数
            position: パッチを適用する位置
            patch_mode: 'v_only', 'pattern_v', 'result'
            qk_overrides: {(layer, head): {"q": tensor, "k": tensor}}
            ov_overrides: {(layer, head): tensor}  # attn.hook_resultに直接適用
            temperature: サンプリング温度（Noneの場合はデフォルト値を使用）
            top_p: nucleus samplingのパラメータ（Noneの場合はデフォルト値を使用）
        """
        if patch_mode not in {"v_only", "pattern_v", "result"}:
            raise ValueError(f"Unsupported patch_mode '{patch_mode}'. Choose from ['v_only','pattern_v','result'].")

        tokens = self.model.to_tokens(neutral_prompt)
        target_pos = position

        def patch_q_hook(activation, hook):
            if not qk_overrides:
                return activation
            layer_idx = int(hook.name.split('.')[1])
            updated = activation
            patched = False
            for (patch_layer, patch_head), patch_dict in qk_overrides.items():
                if patch_layer != layer_idx or "q" not in patch_dict:
                    continue
                if not patched:
                    updated = activation.clone()
                    patched = True
                q_value = patch_dict["q"].to(updated.device)
                if q_value.shape == updated[:, :, patch_head, :].shape:
                    updated[0, :, patch_head, :] = q_value
                elif q_value.shape == updated.shape:
                    updated.copy_(q_value)
            return updated

        def patch_k_hook(activation, hook):
            if not qk_overrides:
                return activation
            layer_idx = int(hook.name.split('.')[1])
            updated = activation
            patched = False
            for (patch_layer, patch_head), patch_dict in qk_overrides.items():
                if patch_layer != layer_idx or "k" not in patch_dict:
                    continue
                if not patched:
                    updated = activation.clone()
                    patched = True
                k_value = patch_dict["k"].to(updated.device)
                if k_value.shape == updated[:, :, patch_head, :].shape:
                    updated[0, :, patch_head, :] = k_value
                elif k_value.shape == updated.shape:
                    updated.copy_(k_value)
            return updated

        def patch_v_hook(activation, hook):
            layer_idx = int(hook.name.split('.')[1])
            for (patch_layer, patch_head), patch_dict in emotion_head_outputs.items():
                if patch_layer == layer_idx and 'v' in patch_dict:
                    activation = activation.clone()
                    v_value = patch_dict['v'].to(activation.device)
                    pos = target_pos
                    if pos == -1 or pos >= activation.shape[1]:
                        pos = activation.shape[1] - 1
                    activation[0, pos, patch_head, :] = v_value
            return activation

        def patch_pattern_hook(activation, hook):
            if patch_mode != "pattern_v":
                return activation
            layer_idx = int(hook.name.split('.')[1])
            for (patch_layer, patch_head), patch_dict in emotion_head_outputs.items():
                if patch_layer == layer_idx and 'pattern' in patch_dict:
                    activation = activation.clone()
                    pat_value = patch_dict['pattern'].to(activation.device)
                    if pat_value.shape == activation[0, patch_head].shape:
                        activation[0, patch_head, :, :] = pat_value
            return activation

        def patch_ov_hook(activation, hook):
            if not ov_overrides:
                return activation
            layer_idx = int(hook.name.split('.')[1])
            updated = activation
            patched = False
            for (patch_layer, patch_head), ov_value in ov_overrides.items():
                if patch_layer != layer_idx:
                    continue
                if not patched:
                    updated = activation.clone()
                    patched = True
                ov_tensor = ov_value.to(updated.device)
                if updated.ndim == 4 and updated.shape[1] == self.model.cfg.n_heads:
                    pos = target_pos
                    if pos == -1 or pos >= updated.shape[2]:
                        pos = updated.shape[2] - 1
                    if ov_tensor.ndim == 1:
                        updated[0, patch_head, pos, :] = ov_tensor
                    elif ov_tensor.ndim == 2 and ov_tensor.shape[0] == updated.shape[2]:
                        updated[0, patch_head, :ov_tensor.shape[0], :] = ov_tensor
                elif updated.ndim == 4 and updated.shape[2] == self.model.cfg.n_heads:
                    pos = target_pos
                    if pos == -1 or pos >= updated.shape[1]:
                        pos = updated.shape[1] - 1
                    if ov_tensor.ndim == 1:
                        updated[0, pos, patch_head, :] = ov_tensor
                    elif ov_tensor.ndim == 2 and ov_tensor.shape[0] == updated.shape[1]:
                        updated[0, :, patch_head, :] = ov_tensor
            return updated

        def patch_result_hook(activation, hook):
            if patch_mode != "result":
                return activation
            layer_idx = int(hook.name.split('.')[1])
            for (patch_layer, patch_head), patch_dict in emotion_head_outputs.items():
                if patch_layer == layer_idx and 'head_output' in patch_dict:
                    patched = patch_dict['head_output'].to(activation.device)
                    act = activation.clone()
                    # Support shapes [batch, head, pos, d_head] or [batch, pos, head, d_head]
                    if act.ndim == 4 and act.shape[1] == self.model.cfg.n_heads:
                        pos = target_pos
                        if pos == -1 or pos >= act.shape[2]:
                            pos = act.shape[2] - 1
                        act[0, patch_head, pos, :] = patched
                    elif act.ndim == 4 and act.shape[2] == self.model.cfg.n_heads:
                        pos = target_pos
                        if pos == -1 or pos >= act.shape[1]:
                            pos = act.shape[1] - 1
                        act[0, pos, patch_head, :] = patched
                    return act
            return activation

        hook_fns: List[Tuple[str, callable]] = []
        target_layers = {layer_idx for layer_idx, _ in emotion_head_outputs.keys()}
        target_layers.update({layer_idx for layer_idx, _ in (qk_overrides or {}).keys()})
        target_layers.update({layer_idx for layer_idx, _ in (ov_overrides or {}).items()})

        for layer_idx in sorted(target_layers):
            hook_fns.append((f"blocks.{layer_idx}.attn.hook_v", patch_v_hook))
            if qk_overrides:
                hook_fns.append((f"blocks.{layer_idx}.attn.hook_q", patch_q_hook))
                hook_fns.append((f"blocks.{layer_idx}.attn.hook_k", patch_k_hook))
            if patch_mode == "pattern_v":
                hook_fns.append((f"blocks.{layer_idx}.attn.hook_pattern", patch_pattern_hook))
            if patch_mode == "result" and self.use_attn_result:
                hook_fns.append((f"blocks.{layer_idx}.attn.hook_result", patch_result_hook))
            if ov_overrides and self.use_attn_result:
                hook_fns.append((f"blocks.{layer_idx}.attn.hook_result", patch_ov_hook))

        generated_tokens = self._generate_tokens(
            tokens,
            hook_fns=hook_fns,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return ' '.join(self.model.to_str_tokens(generated_tokens[0]))
    
    def evaluate_texts(
        self,
        texts: List[str]
    ) -> Dict:
        """
        テキストを評価（感情キーワード頻度とsentimentスコア）
        
        Args:
            texts: テキストのリスト
            
        Returns:
            評価結果の辞書
        """
        all_keyword_counts = {
            'gratitude': [],
            'anger': [],
            'apology': []
        }
        all_sentiment_scores = []
        
        for text in texts:
            # 感情キーワード頻度
            keyword_counts = self.sentiment_evaluator.count_emotion_keywords(text)
            for emotion in ['gratitude', 'anger', 'apology']:
                all_keyword_counts[emotion].append(keyword_counts[emotion])
            
            # Sentimentスコア
            sentiment = self.sentiment_evaluator.calculate_sentiment_score(text)
            if sentiment:
                all_sentiment_scores.append(sentiment.get('POSITIVE', 0.0))
        
        metrics = {
            'keyword_counts': {
                emotion: {
                    'mean': float(np.mean(counts)),
                    'std': float(np.std(counts)),
                    'total': int(np.sum(counts))
                }
                for emotion, counts in all_keyword_counts.items()
            },
            'sentiment': {
                'mean': float(np.mean(all_sentiment_scores)) if all_sentiment_scores else 0.0,
                'std': float(np.std(all_sentiment_scores)) if all_sentiment_scores else 0.0
            }
        }
        
        return metrics
    
    def run_patching_experiment(
        self,
        neutral_prompts: List[str],
        emotion_prompts: List[str],
        head_specs: List[Tuple[int, int]],
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_p: float = 0.9,
        position: int = -1,
        patch_mode: str = "v_only",
    ) -> Dict:
        """
        Patching実験を実行
        
        Args:
            neutral_prompts: 中立プロンプトのリスト
            emotion_prompts: 感情プロンプトのリスト
            head_specs: パッチするheadのリスト
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度
            top_p: nucleus samplingのパラメータ
            position: パッチを適用する位置
            patch_mode: 'v_only'等のパッチ方法を記録
            
        Returns:
            実験結果の辞書
        """
        # 感情プロンプトからhead出力をキャプチャ（全プロンプト平均）
        print("Capturing emotion head outputs (averaging over all emotion prompts)...")
        emotion_head_outputs = self.capture_emotion_head_outputs(
            emotion_prompts,
            head_specs,
            position=position
        )
        
        baseline_texts = []
        patched_texts = []
        
        print(f"Running patching experiment with {len(head_specs)} heads...")
        print(f"Patched heads: {head_specs}")
        
        for neutral_prompt in tqdm(neutral_prompts, desc="Generating texts"):
            # ベースライン生成
            baseline_text = self.sentiment_evaluator.generate_long_text(
                neutral_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            baseline_texts.append(baseline_text)
            
            # Patching生成
            patched_text = self.generate_with_patching(
                neutral_prompt,
                emotion_head_outputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                position=position,
                patch_mode=patch_mode,
            )
            patched_texts.append(patched_text)
        
        # 評価
        baseline_metrics = self.evaluate_texts(baseline_texts)
        patched_metrics = self.evaluate_texts(patched_texts)
        
        results = {
            "model": self.model_name,
            "heads": head_specs,
            "patch_mode": patch_mode,
            "neutral_prompts": neutral_prompts,
            "emotion_prompts": emotion_prompts,
            "baseline_texts": baseline_texts,
            "patched_texts": patched_texts,
            "baseline_metrics": baseline_metrics,
            "patched_metrics": patched_metrics
        }
        
        return results


def main():
    """メイン関数"""
    import argparse
    from src.config.project_profiles import list_profiles
    from src.utils.project_context import ProjectContext, profile_help_text
    
    parser = argparse.ArgumentParser(description="Head patching experiment")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--profile", type=str, choices=list_profiles(), default="baseline",
                        help=f"Dataset profile ({profile_help_text()})")
    parser.add_argument("--head-spec", type=str, required=True, help="Head specification (e.g., '3:5,7:2')")
    parser.add_argument("--neutral-prompts", type=str, default=None, help="Neutral prompts file (JSON, overrides profile default)")
    parser.add_argument("--emotion-prompts", type=str, default=None, help="Emotion prompts file (JSON, overrides profile default)")
    parser.add_argument("--emotion", type=str, default="gratitude", choices=["gratitude", "anger", "apology"],
                        help="Emotion label (used when --emotion-prompts is not specified)")
    parser.add_argument("--output", type=str, default=None, help="Output file path (overrides profile default)")
    parser.add_argument("--max-tokens", type=int, default=30, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--position", type=int, default=-1, help="Position to patch (-1 for last)")
    parser.add_argument("--patch-mode", type=str, choices=["v_only", "pattern_v"], default="v_only",
                        help="Patching strategy (v_only or pattern_v)")
    
    args = parser.parse_args()
    
    # ProjectContextを使用してパスを解決
    context = ProjectContext(profile_name=args.profile)
    results_dir = context.results_dir()
    
    # Head指定をパース
    patcher = HeadPatcher(args.model, device=args.device)
    head_specs = patcher.parse_head_spec(args.head_spec)
    
    if not head_specs:
        raise ValueError(f"No valid heads found in spec: {args.head_spec}")
    
    # プロンプトファイルを解決
    if args.neutral_prompts:
        neutral_file = Path(args.neutral_prompts)
    else:
        neutral_file = context.prompt_file("neutral")
    
    if args.emotion_prompts:
        emotion_file = Path(args.emotion_prompts)
    else:
        emotion_file = context.prompt_file(args.emotion)
    
    if not neutral_file.exists():
        raise FileNotFoundError(f"Neutral prompts file not found: {neutral_file}")
    if not emotion_file.exists():
        raise FileNotFoundError(f"Emotion prompts file not found: {emotion_file}")
    
    with open(neutral_file, 'r') as f:
        neutral_data = json.load(f)
        neutral_prompts = neutral_data.get('prompts', [])
    
    with open(emotion_file, 'r') as f:
        emotion_data = json.load(f)
        emotion_prompts = emotion_data.get('prompts', [])
    
    print(f"Using {len(neutral_prompts)} neutral prompts")
    print(f"Using {len(emotion_prompts)} emotion prompts")
    
    # Patching実験を実行
    results = patcher.run_patching_experiment(
        neutral_prompts,
        emotion_prompts,
        head_specs,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        position=args.position,
        patch_mode=args.patch_mode,
    )
    
    # 出力パスを解決
    if args.output:
        output_path = Path(args.output)
    else:
        # デフォルト: results/{profile}/patching/head_patching/{model}_{emotion}_{head_spec}.pkl
        head_spec_str = args.head_spec.replace(',', '_').replace(':', '')
        output_path = results_dir / "patching" / "head_patching" / f"{args.model.replace('/', '_')}_{args.emotion}_{head_spec_str}.pkl"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")
    
    # サマリーを表示
    print("\nBaseline metrics:")
    print(f"  Sentiment mean: {results['baseline_metrics']['sentiment']['mean']:.4f}")
    print(f"  Gratitude keywords: {results['baseline_metrics']['keyword_counts']['gratitude']['total']}")
    print(f"  Anger keywords: {results['baseline_metrics']['keyword_counts']['anger']['total']}")
    print(f"  Apology keywords: {results['baseline_metrics']['keyword_counts']['apology']['total']}")
    
    print("\nPatched metrics:")
    print(f"  Sentiment mean: {results['patched_metrics']['sentiment']['mean']:.4f}")
    print(f"  Gratitude keywords: {results['patched_metrics']['keyword_counts']['gratitude']['total']}")
    print(f"  Anger keywords: {results['patched_metrics']['keyword_counts']['anger']['total']}")
    print(f"  Apology keywords: {results['patched_metrics']['keyword_counts']['apology']['total']}")
    
    print("\nDifference:")
    sentiment_diff = results['patched_metrics']['sentiment']['mean'] - results['baseline_metrics']['sentiment']['mean']
    print(f"  Sentiment change: {sentiment_diff:+.4f}")


if __name__ == "__main__":
    main()
