"""
Iterative Activation Patching実装
各生成ステップでhookを発火し、特定層のresidualにα * directionを加算
"""
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformer_lens import HookedTransformer
from tqdm import tqdm


class IterativeActivationPatcher:
    """各生成ステップでpatchingを適用するクラス"""
    
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
    
    def load_emotion_vectors(self, vectors_file: Path) -> Dict[str, np.ndarray]:
        """
        感情方向ベクトルを読み込む
        
        Args:
            vectors_file: ベクトルファイルのパス
            
        Returns:
            感情ラベルをキーとする方向ベクトルの辞書 [n_layers, d_model]
        """
        with open(vectors_file, 'rb') as f:
            data = pickle.load(f)
        
        return data['emotion_vectors']
    
    def generate_with_iterative_patching(
        self,
        prompt: str,
        emotion_vector: np.ndarray,
        layer_idx: int,
        alpha: float = 1.0,
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_p: float = 0.9,
        log_activations: bool = False
    ) -> Tuple[str, Optional[Dict]]:
        """
        各生成ステップでpatchingを適用してテキストを生成
        
        Args:
            prompt: 入力プロンプト
            emotion_vector: 感情方向ベクトル [d_model]（特定の層のベクトル）
            layer_idx: パッチを適用する層
            alpha: パッチの強度
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度
            top_p: nucleus samplingのパラメータ
            log_activations: 各ステップの内部値をログに記録するか
            
        Returns:
            (生成されたテキスト, ログデータ)
        """
        tokens = self.model.to_tokens(prompt)
        original_length = tokens.shape[1]
        
        generated_tokens = tokens.clone()
        activation_logs = [] if log_activations else None
        
        # Hookを定義
        def patch_hook(activation, hook):
            """Residual streamをパッチするhook"""
            if hook.name == f"blocks.{layer_idx}.hook_resid_pre":
                activation = activation.clone()
                # 最後の位置にパッチを適用
                activation[0, -1, :] += alpha * torch.tensor(emotion_vector, device=self.device, dtype=activation.dtype)
                
                # ログを記録
                if log_activations:
                    activation_logs.append({
                        'step': len(generated_tokens[0]) - original_length,
                        'activation_norm': torch.norm(activation[0, -1, :]).item(),
                        'patch_norm': torch.norm(alpha * torch.tensor(emotion_vector, device=self.device)).item()
                    })
            
            return activation
        
        # Hookを登録
        hook_name = f"blocks.{layer_idx}.hook_resid_pre"
        hook_handle = self.model.add_hook(hook_name, patch_hook)
        
        try:
            with torch.no_grad():
                for step in range(max_new_tokens):
                    # 現在のトークン列で推論
                    logits = self.model(generated_tokens)
                    
                    # 最後のトークンのlogitsを取得
                    next_token_logits = logits[0, -1, :] / temperature
                    
                    # Top-p (nucleus) sampling
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # 累積確率がtop_pを超える最初のインデックス
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # サンプリング
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # 生成されたトークンを追加
                    generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)
        finally:
            # Hookを削除
            if hook_name in self.model.hook_dict:
                hook_point = self.model.hook_dict[hook_name]
                hook_point.fwd_hooks = []
        
        # トークンをテキストに変換
        generated_text_tokens = self.model.to_str_tokens(generated_tokens[0])
        full_text = ' '.join(generated_text_tokens)
        
        # 生成部分のみを抽出
        generated_part = ' '.join(generated_text_tokens[original_length:])
        
        log_data = {
            'activation_logs': activation_logs,
            'full_text': full_text,
            'generated_part': generated_part,
            'num_tokens_generated': len(generated_text_tokens) - original_length
        } if log_activations else None
        
        return generated_part, log_data
    
    def batch_generate_with_patching(
        self,
        prompts: List[str],
        emotion_vector: np.ndarray,
        layer_idx: int,
        alpha: float = 1.0,
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_p: float = 0.9,
        log_activations: bool = False
    ) -> List[Dict]:
        """
        複数のプロンプトに対してバッチでiterative patchingを実行
        
        Args:
            prompts: プロンプトのリスト
            emotion_vector: 感情方向ベクトル [d_model]
            layer_idx: パッチを適用する層
            alpha: パッチの強度
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度
            top_p: nucleus samplingのパラメータ
            log_activations: 各ステップの内部値をログに記録するか
            
        Returns:
            結果のリスト
        """
        results = []
        
        for prompt in tqdm(prompts, desc="Iterative patching"):
            try:
                generated_text, log_data = self.generate_with_iterative_patching(
                    prompt,
                    emotion_vector,
                    layer_idx,
                    alpha,
                    max_new_tokens,
                    temperature,
                    top_p,
                    log_activations
                )
                
                results.append({
                    'prompt': prompt,
                    'generated_text': generated_text,
                    'log_data': log_data
                })
            except Exception as e:
                print(f"Error with prompt '{prompt}': {e}")
                results.append({
                    'prompt': prompt,
                    'error': str(e)
                })
        
        return results


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Iterative activation patching")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--vectors_file", type=str, required=True, help="Emotion vectors file")
    parser.add_argument("--prompts_file", type=str, required=True, help="Prompts file (JSON)")
    parser.add_argument("--emotion", type=str, required=True, choices=['gratitude', 'anger', 'apology'], help="Emotion label")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--layer", type=int, default=7, help="Layer index for patching")
    parser.add_argument("--alpha", type=float, default=1.0, help="Patching strength")
    parser.add_argument("--max_tokens", type=int, default=30, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--log_activations", action="store_true", help="Log activation values")
    
    args = parser.parse_args()
    
    # パッチャーを作成
    patcher = IterativeActivationPatcher(args.model)
    
    # 感情ベクトルを読み込み
    emotion_vectors = patcher.load_emotion_vectors(Path(args.vectors_file))
    
    if args.emotion not in emotion_vectors:
        raise ValueError(f"Emotion '{args.emotion}' not found in vectors file")
    
    emotion_vec = emotion_vectors[args.emotion]
    layer_vector = emotion_vec[args.layer]  # [d_model]
    
    print(f"Using {args.emotion} vector from layer {args.layer}")
    
    # プロンプトを読み込み
    with open(args.prompts_file, 'r') as f:
        prompts_data = json.load(f)
        prompts = prompts_data.get('prompts', [])
    
    print(f"Using {len(prompts)} prompts")
    
    # バッチ生成を実行
    results = patcher.batch_generate_with_patching(
        prompts,
        layer_vector,
        args.layer,
        args.alpha,
        args.max_tokens,
        args.temperature,
        args.top_p,
        args.log_activations
    )
    
    # 結果を保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump({
            'model': args.model,
            'emotion': args.emotion,
            'layer': args.layer,
            'alpha': args.alpha,
            'max_new_tokens': args.max_tokens,
            'results': results
        }, f)
    
    print(f"\nResults saved to: {output_path}")
    
    # サマリーを表示
    print("\n" + "=" * 80)
    print("Iterative Patching Summary")
    print("=" * 80)
    print(f"Emotion: {args.emotion}")
    print(f"Layer: {args.layer}")
    print(f"Alpha: {args.alpha}")
    print(f"Generated {len([r for r in results if 'generated_text' in r])} texts")
    
    # サンプル出力を表示
    print("\nSample outputs:")
    for i, result in enumerate(results[:3]):
        if 'generated_text' in result:
            print(f"\n{i+1}. Prompt: {result['prompt']}")
            print(f"   Generated: {result['generated_text'][:100]}...")


if __name__ == "__main__":
    main()

