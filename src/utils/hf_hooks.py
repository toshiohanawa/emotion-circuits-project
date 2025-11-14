"""
HuggingFaceモデル用フック
GPT-2 medium/large、Pythia-410M、Llama-3などからresidual streamやattention出力を取得
"""
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_hf_causal_lm(model_name: str, device: Optional[str] = None) -> Tuple:
    """
    HuggingFaceのCausalLMモデルとトークナイザーをロード
    
    Args:
        model_name: モデル名（例: "gpt2-medium", "EleutherAI/pythia-410m-deduped"）
        device: 使用するデバイス（Noneの場合は自動選択）
        
    Returns:
        (model, tokenizer) のタプル
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Pad tokenが設定されていない場合、eos_tokenを使用
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device if device == "cuda" else None
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    
    return model, tokenizer


@contextmanager
def capture_residuals(model, layers: List[int]):
    """
    Residual streamをキャプチャするcontext manager
    
    Args:
        model: HuggingFaceモデル
        layers: キャプチャする層のインデックスリスト
        
    Yields:
        cache辞書: {"resid": {layer_idx: tensor}, ...}
    """
    cache = {"resid": {}}
    hooks = []
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # GPT-2系の場合、outputはtupleで、最初の要素がhidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Residual streamを保存
            cache["resid"][layer_idx] = hidden_states.detach().cpu()
            return output
        return hook_fn
    
    # Hookを登録
    for layer_idx in layers:
        # GPT-2系の場合、transformer.h.{layer_idx} が層
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layer_module = model.transformer.h[layer_idx]
        elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
            # Pythia系の場合
            layer_module = model.gpt_neox.layers[layer_idx]
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Llama系の場合
            layer_module = model.model.layers[layer_idx]
        else:
            print(f"Warning: Could not find layer {layer_idx}, skipping...")
            continue
        
        # 層の出力にhookを登録
        handle = layer_module.register_forward_hook(make_hook(layer_idx))
        hooks.append(handle)
    
    try:
        yield cache
    finally:
        # Hookを削除
        for handle in hooks:
            handle.remove()


@contextmanager
def capture_attention(model, layers: List[int]):
    """
    Attention weightsをキャプチャするcontext manager
    
    Args:
        model: HuggingFaceモデル
        layers: キャプチャする層のインデックスリスト
        
    Yields:
        cache辞書: {"attn": {layer_idx: tensor}, ...}
    """
    cache = {"attn": {}}
    hooks = []
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # Attention weightsを取得
            # GPT-2系の場合、attn_weightsはattention層の属性に保存されることが多い
            if hasattr(module, 'attn') and hasattr(module.attn, 'attn_weights'):
                attn_weights = module.attn.attn_weights
                cache["attn"][layer_idx] = attn_weights.detach().cpu()
            return output
        return hook_fn
    
    # Hookを登録
    for layer_idx in layers:
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layer_module = model.transformer.h[layer_idx]
        elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
            layer_module = model.gpt_neox.layers[layer_idx]
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layer_module = model.model.layers[layer_idx]
        else:
            print(f"Warning: Could not find layer {layer_idx}, skipping...")
            continue
        
        # Attention層にhookを登録
        if hasattr(layer_module, 'attn'):
            handle = layer_module.attn.register_forward_hook(make_hook(layer_idx))
            hooks.append(handle)
    
    try:
        yield cache
    finally:
        # Hookを削除
        for handle in hooks:
            handle.remove()


@contextmanager
def capture_mlp_output(model, layers: List[int]):
    """
    MLP出力をキャプチャするcontext manager
    
    Args:
        model: HuggingFaceモデル
        layers: キャプチャする層のインデックスリスト
        
    Yields:
        cache辞書: {"mlp": {layer_idx: tensor}, ...}
    """
    cache = {"mlp": {}}
    hooks = []
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # MLP出力を取得
            # GPT-2系の場合、mlpの出力は層の出力の一部
            if hasattr(module, 'mlp'):
                # MLPモジュールの出力を取得
                mlp_output = module.mlp(input[0] if isinstance(input, tuple) else input)
                cache["mlp"][layer_idx] = mlp_output.detach().cpu()
            return output
        return hook_fn
    
    # Hookを登録
    for layer_idx in layers:
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layer_module = model.transformer.h[layer_idx]
        elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
            layer_module = model.gpt_neox.layers[layer_idx]
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layer_module = model.model.layers[layer_idx]
        else:
            print(f"Warning: Could not find layer {layer_idx}, skipping...")
            continue
        
        handle = layer_module.register_forward_hook(make_hook(layer_idx))
        hooks.append(handle)
    
    try:
        yield cache
    finally:
        # Hookを削除
        for handle in hooks:
            handle.remove()


def get_model_config(model_name: str) -> Dict:
    """
    モデルの設定情報を取得
    
    Args:
        model_name: モデル名
        
    Returns:
        設定情報の辞書
    """
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(model_name)
    
    info = {
        "model_name": model_name,
        "vocab_size": config.vocab_size if hasattr(config, 'vocab_size') else None,
        "hidden_size": config.hidden_size if hasattr(config, 'hidden_size') else None,
        "num_layers": config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else None,
        "num_attention_heads": config.num_attention_heads if hasattr(config, 'num_attention_heads') else None,
    }
    
    return info


if __name__ == "__main__":
    # テスト用
    import argparse
    
    parser = argparse.ArgumentParser(description="Test HF hooks")
    parser.add_argument("--model", type=str, default="gpt2-medium", help="Model name")
    parser.add_argument("--device", type=str, default=None, help="Device")
    
    args = parser.parse_args()
    
    model, tokenizer = load_hf_causal_lm(args.model, device=args.device)
    
    # テスト用の入力
    text = "The weather today is"
    inputs = tokenizer(text, return_tensors="pt").to(args.device or "cpu")
    
    # Residual streamをキャプチャ
    with capture_residuals(model, layers=[3, 5, 7]) as cache:
        with torch.no_grad():
            outputs = model(**inputs)
    
    print("\nCaptured residual streams:")
    for layer_idx, resid in cache["resid"].items():
        print(f"  Layer {layer_idx}: shape {resid.shape}")
    
    config = get_model_config(args.model)
    print("\nModel config:")
    for key, value in config.items():
        print(f"  {key}: {value}")

