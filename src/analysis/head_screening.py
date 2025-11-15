"""
感情語トークンに強く反応するheadをスクリーニング
各層・各headが感情語トークンに対してどれくらい特異的に反応しているかを定量化
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformer_lens import HookedTransformer
from tqdm import tqdm

from src.analysis.emotion_vectors_token_based import EMOTION_TOKENS
from src.config.project_profiles import list_profiles
from src.utils.project_context import ProjectContext, profile_help_text


class HeadScreener:
    """Headの感情反応度をスクリーニングするクラス"""
    
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
        
        self.n_layers = self.model.cfg.n_layers
        self.n_heads = self.model.cfg.n_heads
        
        print(f"✓ Model loaded on {self.device}")
        print(f"  - Layers: {self.n_layers}")
        print(f"  - Heads per layer: {self.n_heads}")
    
    def find_emotion_token_positions(
        self,
        token_strings: List[str],
        emotion_label: str
    ) -> List[int]:
        """
        感情語トークンの位置を特定
        
        Args:
            token_strings: トークン文字列のリスト
            emotion_label: 感情ラベル
            
        Returns:
            感情語トークンの位置のリスト
        """
        if emotion_label == "neutral":
            return [len(token_strings) - 1]
        
        if emotion_label not in EMOTION_TOKENS:
            return [len(token_strings) - 1]
        
        emotion_tokens = EMOTION_TOKENS[emotion_label]
        positions = []
        
        for idx, token_str in enumerate(token_strings):
            normalized = token_str.lower().strip('.,!?;:')
            if normalized in emotion_tokens:
                positions.append(idx)
        
        if not positions:
            positions = [len(token_strings) - 1]
        
        return positions
    
    def extract_head_activations(
        self,
        prompts: List[str],
        emotion_label: str
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        各層・各headのattentionパターンとMLP出力を抽出
        
        Args:
            prompts: プロンプトのリスト
            emotion_label: 感情ラベル
            
        Returns:
            {layer_idx: {head_idx: attention_patterns}} の辞書
        """
        # 各層・各headのattentionパターンを保存
        attention_patterns = {layer: {head: [] for head in range(self.n_heads)} 
                              for layer in range(self.n_layers)}
        mlp_outputs = {layer: [] for layer in range(self.n_layers)}
        
        # Hook関数を定義
        def save_attention_pattern(activation, hook):
            """Attentionパターンを保存"""
            layer_idx = int(hook.name.split('.')[1])
            # activation shape: [batch, head, query_pos, key_pos]
            batch_size = activation.shape[0]
            
            # 各バッチ（通常は1）ごとに処理
            for batch_idx in range(batch_size):
                for head_idx in range(self.n_heads):
                    # 各headのattentionパターンを保存
                    attn_pattern = activation[batch_idx, head_idx, :, :].detach().cpu().numpy()
                    attention_patterns[layer_idx][head_idx].append(attn_pattern)
            return activation
        
        def save_mlp_output(activation, hook):
            """MLP出力を保存"""
            layer_idx = int(hook.name.split('.')[1])
            # activation shape: [batch, seq, d_model]
            mlp_output = activation.detach().cpu().numpy()
            mlp_outputs[layer_idx].append(mlp_output)
            return activation
        
        # Hookを登録
        hook_names_attn = [f"blocks.{i}.attn.hook_pattern" for i in range(self.n_layers)]
        hook_names_mlp = [f"blocks.{i}.hook_mlp_out" for i in range(self.n_layers)]
        
        for hook_name in hook_names_attn:
            self.model.add_hook(hook_name, save_attention_pattern)
        for hook_name in hook_names_mlp:
            self.model.add_hook(hook_name, save_mlp_output)
        
        # 各プロンプトで推論
        token_positions_per_prompt = []
        
        with torch.no_grad():
            for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"Processing {emotion_label} prompts")):
                tokens = self.model.to_tokens(prompt)
                token_strings = self.model.to_str_tokens(tokens[0])
                
                # 感情語トークン位置を特定
                emotion_positions = self.find_emotion_token_positions(token_strings, emotion_label)
                token_positions_per_prompt.append(emotion_positions)
                
                # Forward（この時点でhookが呼ばれ、attentionパターンが保存される）
                _ = self.model(tokens)
        
        # Hookを削除
        for hook_name in hook_names_attn:
            if hook_name in self.model.hook_dict:
                hook_point = self.model.hook_dict[hook_name]
                hook_point.fwd_hooks = []
        for hook_name in hook_names_mlp:
            if hook_name in self.model.hook_dict:
                hook_point = self.model.hook_dict[hook_name]
                hook_point.fwd_hooks = []
        
        return attention_patterns, mlp_outputs, token_positions_per_prompt
    
    def compute_head_scores(
        self,
        emotion_attention: Dict[int, Dict[int, List[np.ndarray]]],
        neutral_attention: Dict[int, Dict[int, List[np.ndarray]]],
        emotion_positions: List[List[int]],
        neutral_positions: List[List[int]]
    ) -> List[Dict]:
        """
        Headごとの感情反応度スコアを計算
        
        Args:
            emotion_attention: 感情プロンプトのattentionパターン
            neutral_attention: 中立プロンプトのattentionパターン
            emotion_positions: 各プロンプトの感情語トークン位置
            neutral_positions: 各プロンプトの位置（中立の場合は最後）
            
        Returns:
            スコアのリスト
        """
        scores = []
        
        # プロンプト数を確認（emotion_positionsの長さを使用）
        n_emotion_prompts = len(emotion_positions)
        n_neutral_prompts = len(neutral_positions)
        
        for layer_idx in range(self.n_layers):
            for head_idx in range(self.n_heads):
                # 感情プロンプトでのattentionを集計
                emotion_attn_values = []
                emotion_attn_list = emotion_attention[layer_idx][head_idx]
                
                # プロンプト数とattentionパターン数の整合性を確認
                for prompt_idx in range(min(n_emotion_prompts, len(emotion_attn_list))):
                    if prompt_idx >= len(emotion_positions):
                        continue
                    attn_patterns = emotion_attn_list[prompt_idx]
                    positions = emotion_positions[prompt_idx]
                    valid_query_positions = [pos for pos in positions if pos < attn_patterns.shape[0]]
                    if not valid_query_positions:
                        valid_query_positions = [attn_patterns.shape[0] - 1]
                    valid_key_positions = [pos for pos in positions if pos < attn_patterns.shape[1]]
                    if not valid_key_positions:
                        valid_key_positions = [attn_patterns.shape[1] - 1]
                    submatrix = attn_patterns[np.ix_(valid_query_positions, valid_key_positions)]
                    attn_value = float(np.mean(submatrix))
                    emotion_attn_values.append(attn_value)
                
                # 中立プロンプトでのattentionを集計
                neutral_attn_values = []
                neutral_attn_list = neutral_attention[layer_idx][head_idx]
                
                # プロンプト数とattentionパターン数の整合性を確認
                for prompt_idx in range(min(n_neutral_prompts, len(neutral_attn_list))):
                    if prompt_idx >= len(neutral_positions):
                        continue
                    attn_patterns = neutral_attn_list[prompt_idx]
                    positions = neutral_positions[prompt_idx]
                    valid_query_positions = [pos for pos in positions if pos < attn_patterns.shape[0]]
                    if not valid_query_positions:
                        valid_query_positions = [attn_patterns.shape[0] - 1]
                    valid_key_positions = [pos for pos in positions if pos < attn_patterns.shape[1]]
                    if not valid_key_positions:
                        valid_key_positions = [attn_patterns.shape[1] - 1]
                    submatrix = attn_patterns[np.ix_(valid_query_positions, valid_key_positions)]
                    neutral_attn_values.append(float(np.mean(submatrix)))
                
                # 差分を計算
                if emotion_attn_values and neutral_attn_values:
                    delta_attn = np.mean(emotion_attn_values) - np.mean(neutral_attn_values)
                else:
                    delta_attn = 0.0
                
                scores.append({
                    "layer": layer_idx,
                    "head": head_idx,
                    "delta_attn": float(delta_attn),
                    "emotion_mean_attn": float(np.mean(emotion_attn_values)) if emotion_attn_values else 0.0,
                    "neutral_mean_attn": float(np.mean(neutral_attn_values)) if neutral_attn_values else 0.0,
                    "samples_emotion": len(emotion_attn_values),
                    "samples_neutral": len(neutral_attn_values),
                })
        
        return scores
    
    def screen_heads(
        self,
        emotion_prompts: Dict[str, List[str]]
    ) -> Dict:
        """
        Headスクリーニングを実行
        
        Args:
            emotion_prompts: {emotion_label: [prompts]} の辞書
            
        Returns:
            スクリーニング結果
        """
        results = {
            "model": self.model_name,
            "layers": self.n_layers,
            "heads_per_layer": self.n_heads,
            "scores": []
        }
        
        # 各感情カテゴリのattentionパターンを抽出
        emotion_activations = {}
        emotion_positions_dict = {}
        
        for emotion_label, prompts in emotion_prompts.items():
            print(f"\nExtracting activations for {emotion_label}...")
            attn_patterns, mlp_outputs, positions = self.extract_head_activations(prompts, emotion_label)
            emotion_activations[emotion_label] = attn_patterns
            emotion_positions_dict[emotion_label] = positions
        
        # 各感情について、中立との差分を計算
        if "neutral" in emotion_prompts:
            neutral_attention = emotion_activations["neutral"]
            neutral_positions = emotion_positions_dict["neutral"]
            
            for emotion_label in ["gratitude", "anger", "apology"]:
                if emotion_label in emotion_prompts:
                    print(f"\nComputing scores for {emotion_label}...")
                    emotion_attention = emotion_activations[emotion_label]
                    emotion_positions = emotion_positions_dict[emotion_label]
                    
                    scores = self.compute_head_scores(
                        emotion_attention,
                        neutral_attention,
                        emotion_positions,
                        neutral_positions
                    )
                    
                    # 感情ラベルを追加
                    for score in scores:
                        score["emotion"] = emotion_label
                        results["scores"].append(score)
        
        return results


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Head screening for emotion tokens")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--profile", type=str, choices=list_profiles(), default="baseline",
                        help=f"Dataset profile used to auto-resolve prompts ({profile_help_text()})")
    parser.add_argument("--gratitude-prompts", type=str, default=None, help="Optional override for gratitude prompts file")
    parser.add_argument("--anger-prompts", type=str, default=None, help="Optional override for anger prompts file")
    parser.add_argument("--apology-prompts", type=str, default=None, help="Optional override for apology prompts file")
    parser.add_argument("--neutral-prompts", type=str, default=None, help="Optional override for neutral prompts file")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory for profile resolution")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    
    args = parser.parse_args()
    
    context = ProjectContext(args.profile, data_dir=Path(args.data_dir))
    
    # プロンプトを読み込み
    emotion_prompts = {}
    
    for emotion_label, specified_path in [
        ("gratitude", args.gratitude_prompts),
        ("anger", args.anger_prompts),
        ("apology", args.apology_prompts),
        ("neutral", args.neutral_prompts)
    ]:
        if specified_path:
            prompt_file = Path(specified_path)
        else:
            prompt_file = context.prompt_file(emotion_label)
        if prompt_file and prompt_file.exists():
            with open(prompt_file, 'r') as f:
                data = json.load(f)
                prompts = data.get('prompts', [])
                emotion_prompts[emotion_label] = prompts
                print(f"Loaded {len(prompts)} {emotion_label} prompts from {prompt_file}")
        else:
            print(f"Warning: No prompt file found for {emotion_label}")
    
    if not emotion_prompts:
        raise ValueError("No prompts loaded. Please check file paths.")
    
    # Headスクリーニングを実行
    screener = HeadScreener(args.model, device=args.device)
    results = screener.screen_heads(emotion_prompts)
    
    # 結果を保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")
    print(f"Total scores: {len(results['scores'])}")
    
    # 上位headを表示
    sorted_scores = sorted(results['scores'], key=lambda x: abs(x['delta_attn']), reverse=True)
    print("\nTop 10 heads by |delta_attn|:")
    for i, score in enumerate(sorted_scores[:10]):
        print(f"  {i+1}. Layer {score['layer']}, Head {score['head']}, Emotion: {score['emotion']}, "
              f"Δattn: {score['delta_attn']:.6f}")


if __name__ == "__main__":
    main()
