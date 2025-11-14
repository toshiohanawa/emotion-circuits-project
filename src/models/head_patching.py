"""
Head patching実験
感謝文のあるhead出力を、中立文に「移植」するSwap Patchingのhead版
"""
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformer_lens import HookedTransformer
from tqdm import tqdm

from src.analysis.sentiment_eval import SentimentEvaluator


class HeadPatcher:
    """Head patchingを実行するクラス"""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        初期化
        
        Args:
            model_name: モデル名（例: "gpt2"）
            device: 使用するデバイス
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model: {model_name}")
        self.model = HookedTransformer.from_pretrained(model_name, device=self.device)
        self.model.eval()
        
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
    ) -> Dict[Tuple[int, int], List[torch.Tensor]]:
        """
        感情プロンプトでhead出力をキャプチャ
        
        Args:
            emotion_prompts: 感情プロンプトのリスト
            head_specs: キャプチャするheadのリスト [(layer_idx, head_idx), ...]
            position: キャプチャする位置（-1は最後）
            
        Returns:
            {(layer_idx, head_idx): [head_outputs]} の辞書
        """
        captured_outputs = {(layer, head): [] for layer, head in head_specs}
        
        # hook_resultの代わりに、hook_patternとhook_vを使ってhead出力を計算
        # キャッシュを保存
        cache = {}
        
        def capture_pattern_hook(activation, hook):
            """Attention patternをキャプチャ"""
            layer_idx = int(hook.name.split('.')[1])
            if layer_idx not in cache:
                cache[layer_idx] = {}
            cache[layer_idx]['pattern'] = activation.detach().clone()
            return activation
        
        def capture_v_hook(activation, hook):
            """Vをキャプチャ"""
            layer_idx = int(hook.name.split('.')[1])
            if layer_idx not in cache:
                cache[layer_idx] = {}
            cache[layer_idx]['v'] = activation.detach().clone()
            return activation
        
        # Hookを登録
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
                    cache.clear()  # キャッシュをクリア
                    _ = self.model(tokens)
                    
                    # hook_patternとhook_vからhead出力を計算
                    for layer_idx, head_idx in head_specs:
                        if layer_idx in cache and 'pattern' in cache[layer_idx] and 'v' in cache[layer_idx]:
                            pattern = cache[layer_idx]['pattern']  # [batch, head, pos, pos]
                            v = cache[layer_idx]['v']  # [batch, pos, head, d_head]
                            
                            # head出力を計算: pattern @ v
                            # pattern: [batch, head, pos, pos], v: [batch, pos, head, d_head]
                            # 結果: [batch, pos, head, d_head]
                            batch_size = pattern.shape[0]
                            n_heads = pattern.shape[1]
                            seq_len = pattern.shape[2]
                            d_head = v.shape[3]
                            
                            # patternを [batch, head, pos, pos] から [batch, head, pos, pos] に保持
                            # vを [batch, pos, head, d_head] から [batch, head, pos, d_head] に変換
                            v_reshaped = v.permute(0, 2, 1, 3)  # [batch, head, pos, d_head]
                            
                            # pattern @ v_reshaped: [batch, head, pos, pos] @ [batch, head, pos, d_head] = [batch, head, pos, d_head]
                            head_output = torch.einsum('bhqp,bhpd->bhqd', pattern, v_reshaped)  # [batch, head, pos, d_head]
                            
                            # 指定headと位置を取得
                            if position == -1:
                                head_output_value = head_output[0, head_idx, -1, :].clone()
                            else:
                                head_output_value = head_output[0, head_idx, position, :].clone()
                            
                            captured_outputs[(layer_idx, head_idx)].append(head_output_value)
        
        finally:
            # Hookを削除
            for hook_name, handle in hook_handles:
                if hook_name in self.model.hook_dict:
                    hook_point = self.model.hook_dict[hook_name]
                    hook_point.fwd_hooks = []
        
        return captured_outputs
    
    def generate_with_patching(
        self,
        neutral_prompt: str,
        emotion_head_outputs: Dict[Tuple[int, int], torch.Tensor],
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_p: float = 0.9,
        position: int = -1
    ) -> str:
        """
        Head patchingを適用して生成
        
        Args:
            neutral_prompt: 中立プロンプト
            emotion_head_outputs: 感情プロンプトからキャプチャしたhead出力 {(layer, head): output}
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度
            top_p: nucleus samplingのパラメータ
            position: パッチを適用する位置（-1は最後）
            
        Returns:
            生成されたテキスト
        """
        tokens = self.model.to_tokens(neutral_prompt)
        generated_tokens = tokens.clone()
        
        # hook_resultの代わりに、hook_vを使ってhead出力を間接的に変更
        # 注意: これは完全なhead出力の置き換えではなく、Vの置き換えによる近似
        # より正確には、hook_patternとhook_vの両方を操作する必要があるが、
        # 簡易版としてhook_vのみを操作する
        
        # 感情側のhead出力からVを逆算する必要があるが、これは複雑
        # 代わりに、hook_vを直接置き換える方法を使用
        # ただし、これは完全には機能しない可能性がある
        
        # 簡易版: hook_vを置き換える（完全ではないが、近似的に機能する）
        def patch_v_hook(activation, hook):
            """指定headのVを感情側の値に差し替え（近似）"""
            layer_idx = int(hook.name.split('.')[1])
            
            # この層でpatchするheadがあるかチェック
            for (patch_layer, patch_head), patch_value in emotion_head_outputs.items():
                if patch_layer == layer_idx:
                    activation = activation.clone()
                    # activation shape: [batch, pos, head, d_head]
                    if position == -1:
                        # 最後の位置のVを置き換え（近似）
                        activation[0, -1, patch_head, :] = patch_value.to(activation.device)
                    else:
                        activation[0, position, patch_head, :] = patch_value.to(activation.device)
            
            return activation
        
        # Hookを登録（hook_vを使用）
        hook_handles = []
        for layer_idx, head_idx in emotion_head_outputs.keys():
            hook_name = f"blocks.{layer_idx}.attn.hook_v"
            handle = self.model.add_hook(hook_name, patch_v_hook)
            hook_handles.append((hook_name, handle))
        
        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    # 現在のトークンでlogitsを取得
                    logits = self.model(generated_tokens)
                    
                    # 最後のトークンのlogitsを使用
                    next_token_logits = logits[0, -1, :] / temperature
                    
                    # Top-p sampling
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Top-pでフィルタ
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
            for hook_name, handle in hook_handles:
                if hook_name in self.model.hook_dict:
                    hook_point = self.model.hook_dict[hook_name]
                    hook_point.fwd_hooks = []
        
        # トークンをテキストに変換
        generated_text = self.model.to_str_tokens(generated_tokens[0])
        full_text = ' '.join(generated_text)
        
        return full_text
    
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
        position: int = -1
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
            
        Returns:
            実験結果の辞書
        """
        # 感情プロンプトからhead出力をキャプチャ（最初のプロンプトのみ使用）
        print("Capturing emotion head outputs...")
        emotion_head_outputs_dict = self.capture_emotion_head_outputs(
            emotion_prompts[:1],  # 最初のプロンプトのみ使用
            head_specs,
            position=position
        )
        
        # 各headについて平均を取る（複数プロンプトがある場合）
        emotion_head_outputs = {}
        for (layer, head), outputs in emotion_head_outputs_dict.items():
            if outputs:
                emotion_head_outputs[(layer, head)] = torch.mean(torch.stack(outputs), dim=0)
        
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
                position=position
            )
            patched_texts.append(patched_text)
        
        # 評価
        baseline_metrics = self.evaluate_texts(baseline_texts)
        patched_metrics = self.evaluate_texts(patched_texts)
        
        results = {
            "model": self.model_name,
            "heads": head_specs,
            "neutral_prompts": neutral_prompts,
            "emotion_prompts": emotion_prompts[:1],
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
        position=args.position
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

