"""
汎用的な内部活性抽出スクリプト
TransformerLensを使用して、各層のresidual stream、MLP出力、attentionをhookで取得
"""
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from transformer_lens import HookedTransformer
from tqdm import tqdm


class ActivationExtractor:
    """モデルの内部活性を抽出するクラス"""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        初期化
        
        Args:
            model_name: モデル名（例: "gpt2", "EleutherAI/pythia-160m"）
            device: 使用するデバイス（Noneの場合は自動選択）
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model: {model_name}")
        self.model = HookedTransformer.from_pretrained(model_name, device=self.device)
        self.model.eval()
        
        print(f"✓ Model loaded on {self.device}")
        print(f"  - Layers: {self.model.cfg.n_layers}")
        print(f"  - Hidden size: {self.model.cfg.d_model}")
        print(f"  - Vocab size: {self.model.cfg.d_vocab}")
    
    def extract_activations(
        self,
        texts: List[str],
        save_residual_stream: bool = True,
        save_mlp_output: bool = True,
        save_attention: bool = False,
    ) -> Dict:
        """
        テキストのリストから内部活性を抽出
        
        Args:
            texts: 入力テキストのリスト
            save_residual_stream: residual streamを保存するか
            save_mlp_output: MLP出力を保存するか
            save_attention: attentionを保存するか（メモリ使用量が多い）
            
        Returns:
            抽出された活性の辞書
        """
        activations = {
            'residual_stream': [],
            'mlp_output': [],
            'attention': [],
            'tokens': [],
            'token_strings': [],
        }
        
        # 各テキストごとの活性を保存するリスト
        text_activations = []
        
        def save_residual_stream_hook(activation, hook):
            """Residual streamを保存"""
            layer_idx = int(hook.name.split('.')[1])
            # 現在のテキストインデックスを取得（グローバル変数から）
            text_idx = len(text_activations) - 1
            if text_idx >= 0:
                text_activations[text_idx]['residual_stream'][layer_idx] = activation.detach().cpu()
        
        def save_mlp_output_hook(activation, hook):
            """MLP出力を保存"""
            layer_idx = int(hook.name.split('.')[1])
            text_idx = len(text_activations) - 1
            if text_idx >= 0:
                text_activations[text_idx]['mlp_output'][layer_idx] = activation.detach().cpu()
        
        def save_attention_hook(activation, hook):
            """Attention重みを保存"""
            layer_idx = int(hook.name.split('.')[1])
            text_idx = len(text_activations) - 1
            if text_idx >= 0:
                # Attention shape: [batch, head, pos, pos]
                text_activations[text_idx]['attention'][layer_idx] = activation.detach().cpu()
        
        # Hook名のリストを保存（後で削除するため）
        hook_names = []
        
        # Hookを登録
        if save_residual_stream:
            for layer_idx in range(self.model.cfg.n_layers):
                hook_name = f"blocks.{layer_idx}.hook_resid_pre"
                self.model.add_hook(hook_name, save_residual_stream_hook)
                hook_names.append(hook_name)
        
        if save_mlp_output:
            for layer_idx in range(self.model.cfg.n_layers):
                hook_name = f"blocks.{layer_idx}.hook_mlp_out"
                self.model.add_hook(hook_name, save_mlp_output_hook)
                hook_names.append(hook_name)
        
        if save_attention:
            for layer_idx in range(self.model.cfg.n_layers):
                hook_name = f"blocks.{layer_idx}.attn.hook_pattern"
                self.model.add_hook(hook_name, save_attention_hook)
                hook_names.append(hook_name)
        
        # 各テキストを処理
        with torch.no_grad():
            for text in tqdm(texts, desc="Extracting activations"):
                # このテキスト用の活性辞書を初期化
                text_act = {
                    'residual_stream': {},
                    'mlp_output': {},
                    'attention': {},
                }
                text_activations.append(text_act)
                
                # トークナイズ
                tokens = self.model.to_tokens(text)
                token_strings = self.model.to_str_tokens(text)
                
                # 推論実行（hookが活性を保存）
                _ = self.model(tokens)
                
                # トークン情報を保存
                activations['tokens'].append(tokens.cpu())
                activations['token_strings'].append(token_strings)
        
        # Hookを削除（hook_dictから削除）
        for hook_name in hook_names:
            if hook_name in self.model.hook_dict:
                hook_point = self.model.hook_dict[hook_name]
                hook_point.fwd_hooks = []
        
        # 活性を整理（層ごとにまとめる）
        if save_residual_stream:
            for layer_idx in range(self.model.cfg.n_layers):
                layer_acts = []
                for text_act in text_activations:
                    if layer_idx in text_act['residual_stream']:
                        # [batch, pos, d_model] -> [pos, d_model] (batch=1なので)
                        layer_acts.append(text_act['residual_stream'][layer_idx][0].numpy())
                activations['residual_stream'].append(layer_acts)
        
        if save_mlp_output:
            for layer_idx in range(self.model.cfg.n_layers):
                layer_acts = []
                for text_act in text_activations:
                    if layer_idx in text_act['mlp_output']:
                        layer_acts.append(text_act['mlp_output'][layer_idx][0].numpy())
                activations['mlp_output'].append(layer_acts)
        
        if save_attention:
            for layer_idx in range(self.model.cfg.n_layers):
                layer_acts = []
                for text_act in text_activations:
                    if layer_idx in text_act['attention']:
                        layer_acts.append(text_act['attention'][layer_idx][0].numpy())
                activations['attention'].append(layer_acts)
        
        return activations
    
    def process_dataset(
        self,
        dataset_path: Path,
        output_dir: Path,
        emotion_label: Optional[str] = None,
        save_residual_stream: bool = True,
        save_mlp_output: bool = True,
        save_attention: bool = False,
    ) -> None:
        """
        データセット全体を処理して活性を抽出・保存
        
        Args:
            dataset_path: JSONL形式のデータセットファイルのパス
            output_dir: 出力ディレクトリ
            emotion_label: 感情ラベル（Noneの場合は全データを処理）
            save_residual_stream: residual streamを保存するか
            save_mlp_output: MLP出力を保存するか
            save_attention: attentionを保存するか
        """
        # データセットを読み込み
        texts = []
        labels = []
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if emotion_label is None or entry['emotion'] == emotion_label:
                        texts.append(entry['text'])
                        labels.append(entry['emotion'])
        
        print(f"\nProcessing {len(texts)} texts (emotion: {emotion_label or 'all'})")
        
        # 活性を抽出
        activations = self.extract_activations(
            texts,
            save_residual_stream=save_residual_stream,
            save_mlp_output=save_mlp_output,
            save_attention=save_attention,
        )
        
        # メタデータを追加
        metadata = {
            'model_name': self.model_name,
            'n_layers': self.model.cfg.n_layers,
            'd_model': self.model.cfg.d_model,
            'd_vocab': self.model.cfg.d_vocab,
            'n_samples': len(texts),
            'emotion_label': emotion_label,
            'save_residual_stream': save_residual_stream,
            'save_mlp_output': save_mlp_output,
            'save_attention': save_attention,
        }
        
        activations['metadata'] = metadata
        activations['labels'] = labels
        
        # 出力ディレクトリを作成
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存
        output_file = output_dir / f"activations_{emotion_label or 'all'}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(activations, f)
        
        print(f"✓ Activations saved to: {output_file}")
        print(f"  - Residual stream layers: {len(activations.get('residual_stream', []))}")
        print(f"  - MLP output layers: {len(activations.get('mlp_output', []))}")
        print(f"  - Attention layers: {len(activations.get('attention', []))}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract model activations")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, default="data/emotion_dataset.jsonl", help="Dataset path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--emotion", type=str, default=None, help="Emotion label (optional)")
    parser.add_argument("--no-residual", action="store_true", help="Don't save residual stream")
    parser.add_argument("--no-mlp", action="store_true", help="Don't save MLP output")
    parser.add_argument("--attention", action="store_true", help="Save attention weights")
    
    args = parser.parse_args()
    
    # Extractorを作成
    extractor = ActivationExtractor(args.model)
    
    # データセットを処理
    extractor.process_dataset(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output),
        emotion_label=args.emotion,
        save_residual_stream=not args.no_residual,
        save_mlp_output=not args.no_mlp,
        save_attention=args.attention,
    )


if __name__ == "__main__":
    main()

