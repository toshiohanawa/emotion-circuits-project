"""
Activation Patching実装
中立文に対する推論時にresidual streamを改変して、感情方向の操作が出力に与える因果的影響を検証
"""
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
    
    def patch_residual_stream(
        self,
        tokens: torch.Tensor,
        emotion_vector: np.ndarray,
        layer_idx: int,
        alpha: float = 1.0,
        position: int = -1
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Residual streamをパッチして推論を実行
        
        Args:
            tokens: 入力トークン [batch, pos]
            emotion_vector: 感情方向ベクトル [n_layers, d_model] または [d_model]
            layer_idx: パッチを適用する層インデックス
            alpha: パッチの強度
            position: パッチを適用する位置（-1は最後）
            
        Returns:
            (logits, generated_tokens)
        """
        patched_activations = {}
        
        def patch_hook(activation, hook):
            """Residual streamをパッチするhook"""
            if hook.name == f"blocks.{layer_idx}.hook_resid_pre":
                # パッチを適用
                activation = activation.clone()
                if position == -1:
                    # 最後の位置にパッチ
                    activation[0, -1, :] += alpha * torch.tensor(emotion_vector, device=self.device, dtype=activation.dtype)
                else:
                    activation[0, position, :] += alpha * torch.tensor(emotion_vector, device=self.device, dtype=activation.dtype)
            return activation
        
        # Hookを登録
        hook_handle = self.model.add_hook(f"blocks.{layer_idx}.hook_resid_pre", patch_hook)
        
        # 推論実行
        with torch.no_grad():
            logits = self.model(tokens)
        
        # Hookを削除（hook_dictから削除）
        hook_name = f"blocks.{layer_idx}.hook_resid_pre"
        if hook_name in self.model.hook_dict:
            hook_point = self.model.hook_dict[hook_name]
            hook_point.fwd_hooks = []
        
        # 生成されたトークンを取得
        generated_tokens = self.model.to_str_tokens(logits.argmax(dim=-1)[0])
        
        return logits, generated_tokens
    
    def generate_with_patching(
        self,
        prompt: str,
        emotion_vector: np.ndarray,
        layer_idx: int,
        alpha: float = 1.0,
        max_new_tokens: int = 10
    ) -> str:
        """
        パッチを適用してテキストを生成（簡易版：1トークンのみ）
        
        Args:
            prompt: 入力プロンプト
            emotion_vector: 感情方向ベクトル [d_model]（特定の層のベクトル）
            layer_idx: パッチを適用する層
            alpha: パッチの強度
            max_new_tokens: 生成する最大トークン数（簡易版では使用しない）
            
        Returns:
            生成されたテキスト（元のプロンプト + 次の数トークン）
        """
        tokens = self.model.to_tokens(prompt)
        
        # パッチを適用して推論
        logits, generated_tokens = self.patch_residual_stream(
            tokens,
            emotion_vector,
            layer_idx,
            alpha,
            position=-1
        )
        
        # 次のトークンを予測
        next_token_logits = logits[0, -1, :]
        top_k = 5
        top_k_tokens = torch.topk(next_token_logits, top_k)
        
        # トップ5のトークンを取得
        top_tokens = []
        for idx in top_k_tokens.indices:
            token_str = self.model.to_string(idx.item())
            top_tokens.append(token_str)
        
        # 元のプロンプト + 次のトークン候補を返す
        generated_text = prompt + " [" + ", ".join(top_tokens[:3]) + "]"
        
        return generated_text
    
    def evaluate_patching_effect(
        self,
        prompts: List[str],
        emotion_vectors: Dict[str, np.ndarray],
        layer_idx: int = 6,
        alpha_values: List[float] = [0.0, 0.5, 1.0, 1.5, -0.5, -1.0]
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
            tokens = self.model.to_tokens(prompt)
            with torch.no_grad():
                logits = self.model(tokens)
            generated = self.model.to_str_tokens(logits.argmax(dim=-1)[0])
            results['baseline'][prompt] = ' '.join(generated)
        
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
                            max_new_tokens=20
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
        alpha_values=args.alpha
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

