"""
Swap Activation Patching実装
中立プロンプトAのresidualを感謝プロンプトBのresidualに置換して生成
"""
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformer_lens import HookedTransformer
from tqdm import tqdm


class SwapActivationPatcher:
    """Residual swap patchingを実行するクラス"""
    
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
        
        print(f"✓ Model loaded on {self.device}")
    
    def swap_and_generate(
        self,
        neutral_prompt: str,
        emotion_prompt: str,
        layer_idx: int,
        position: int = -1,
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> Tuple[str, str]:
        """
        中立プロンプトのresidualを感情プロンプトのresidualに置換して生成
        
        Args:
            neutral_prompt: 中立プロンプトA
            emotion_prompt: 感情プロンプトB（例：感謝文）
            layer_idx: パッチを適用する層
            position: パッチを適用する位置（-1は最後）
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度
            top_p: nucleus samplingのパラメータ
            
        Returns:
            (baseline生成テキスト, swap後の生成テキスト)
        """
        neutral_tokens = self.model.to_tokens(neutral_prompt)
        emotion_tokens = self.model.to_tokens(emotion_prompt)
        
        neutral_length = neutral_tokens.shape[1]
        emotion_length = emotion_tokens.shape[1]
        
        # 感情プロンプトのresidualを取得
        emotion_residual = None
        
        def capture_emotion_residual(activation, hook):
            """感情プロンプトのresidualをキャプチャ"""
            nonlocal emotion_residual
            if hook.name == f"blocks.{layer_idx}.hook_resid_pre":
                if position == -1:
                    emotion_residual = activation[0, -1, :].clone()
                else:
                    emotion_residual = activation[0, position, :].clone()
            return activation
        
        # 感情プロンプトでforwardしてresidualをキャプチャ
        hook_name = f"blocks.{layer_idx}.hook_resid_pre"
        hook_handle = self.model.add_hook(hook_name, capture_emotion_residual)
        
        try:
            with torch.no_grad():
                _ = self.model(emotion_tokens)
        finally:
            # Hookを削除
            if hook_name in self.model.hook_dict:
                hook_point = self.model.hook_dict[hook_name]
                hook_point.fwd_hooks = []
        
        if emotion_residual is None:
            raise ValueError("Failed to capture emotion residual")
        
        # Baseline: 中立プロンプトをそのまま生成
        baseline_tokens = neutral_tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.model(baseline_tokens)
                next_token_logits = logits[0, -1, :] / temperature
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                baseline_tokens = torch.cat([baseline_tokens, next_token.unsqueeze(0)], dim=1)
        
        baseline_text_tokens = self.model.to_str_tokens(baseline_tokens[0])
        baseline_generated = ' '.join(baseline_text_tokens[neutral_length:])
        
        # Swap: 中立プロンプトのresidualを感情プロンプトのresidualに置換して生成
        swap_tokens = neutral_tokens.clone()
        
        def swap_residual(activation, hook):
            """Residualをswapするhook"""
            if hook.name == f"blocks.{layer_idx}.hook_resid_pre":
                activation = activation.clone()
                if position == -1:
                    activation[0, -1, :] = emotion_residual.clone()
                else:
                    activation[0, position, :] = emotion_residual.clone()
            return activation
        
        hook_handle = self.model.add_hook(hook_name, swap_residual)
        
        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    logits = self.model(swap_tokens)
                    next_token_logits = logits[0, -1, :] / temperature
                    
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    swap_tokens = torch.cat([swap_tokens, next_token.unsqueeze(0)], dim=1)
        finally:
            # Hookを削除
            if hook_name in self.model.hook_dict:
                hook_point = self.model.hook_dict[hook_name]
                hook_point.fwd_hooks = []
        
        swap_text_tokens = self.model.to_str_tokens(swap_tokens[0])
        swap_generated = ' '.join(swap_text_tokens[neutral_length:])
        
        return baseline_generated, swap_generated
    
    def batch_swap_and_generate(
        self,
        neutral_prompts: List[str],
        emotion_prompts: List[str],
        layer_idx: int,
        position: int = -1,
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> List[Dict]:
        """
        複数のプロンプトペアに対してバッチでswap patchingを実行
        
        Args:
            neutral_prompts: 中立プロンプトのリスト
            emotion_prompts: 感情プロンプトのリスト
            layer_idx: パッチを適用する層
            position: パッチを適用する位置
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度
            top_p: nucleus samplingのパラメータ
            
        Returns:
            結果のリスト
        """
        if len(neutral_prompts) != len(emotion_prompts):
            raise ValueError("neutral_prompts and emotion_prompts must have the same length")
        
        results = []
        
        for neutral_prompt, emotion_prompt in tqdm(
            zip(neutral_prompts, emotion_prompts),
            desc="Swap patching",
            total=len(neutral_prompts)
        ):
            try:
                baseline_text, swap_text = self.swap_and_generate(
                    neutral_prompt,
                    emotion_prompt,
                    layer_idx,
                    position,
                    max_new_tokens,
                    temperature,
                    top_p
                )
                
                results.append({
                    'neutral_prompt': neutral_prompt,
                    'emotion_prompt': emotion_prompt,
                    'baseline_generated': baseline_text,
                    'swap_generated': swap_text
                })
            except Exception as e:
                print(f"Error with prompts: {e}")
                results.append({
                    'neutral_prompt': neutral_prompt,
                    'emotion_prompt': emotion_prompt,
                    'error': str(e)
                })
        
        return results


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Swap activation patching")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--neutral_prompts_file", type=str, required=True, help="Neutral prompts file (JSON)")
    parser.add_argument("--emotion_prompts_file", type=str, required=True, help="Emotion prompts file (JSON)")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--layer", type=int, default=7, help="Layer index for patching")
    parser.add_argument("--position", type=int, default=-1, help="Position to patch (-1 for last)")
    parser.add_argument("--max_tokens", type=int, default=30, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    
    args = parser.parse_args()
    
    # パッチャーを作成
    patcher = SwapActivationPatcher(args.model)
    
    # プロンプトを読み込み
    with open(args.neutral_prompts_file, 'r') as f:
        neutral_data = json.load(f)
        neutral_prompts = neutral_data.get('prompts', [])
    
    with open(args.emotion_prompts_file, 'r') as f:
        emotion_data = json.load(f)
        emotion_prompts = emotion_data.get('prompts', [])
    
    if len(neutral_prompts) != len(emotion_prompts):
        print(f"Warning: Different number of prompts ({len(neutral_prompts)} vs {len(emotion_prompts)})")
        min_len = min(len(neutral_prompts), len(emotion_prompts))
        neutral_prompts = neutral_prompts[:min_len]
        emotion_prompts = emotion_prompts[:min_len]
    
    print(f"Using {len(neutral_prompts)} prompt pairs")
    
    # バッチ生成を実行
    results = patcher.batch_swap_and_generate(
        neutral_prompts,
        emotion_prompts,
        args.layer,
        args.position,
        args.max_tokens,
        args.temperature,
        args.top_p
    )
    
    # 結果を保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump({
            'model': args.model,
            'layer': args.layer,
            'position': args.position,
            'max_new_tokens': args.max_tokens,
            'results': results
        }, f)
    
    print(f"\nResults saved to: {output_path}")
    
    # サマリーを表示
    print("\n" + "=" * 80)
    print("Swap Patching Summary")
    print("=" * 80)
    print(f"Layer: {args.layer}")
    print(f"Position: {args.position}")
    print(f"Processed {len([r for r in results if 'baseline_generated' in r])} prompt pairs")
    
    # サンプル出力を表示
    print("\nSample outputs:")
    for i, result in enumerate(results[:3]):
        if 'baseline_generated' in result:
            print(f"\n{i+1}. Neutral prompt: {result['neutral_prompt']}")
            print(f"   Emotion prompt: {result['emotion_prompt']}")
            print(f"   Baseline: {result['baseline_generated'][:100]}...")
            print(f"   Swap: {result['swap_generated'][:100]}...")


if __name__ == "__main__":
    main()

