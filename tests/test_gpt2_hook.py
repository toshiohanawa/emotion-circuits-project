"""
GPT-2 smallモデルでTransformerLensのhook動作を確認するテスト
"""
import torch
from transformer_lens import HookedTransformer


def test_gpt2_model_loading():
    """GPT-2 smallモデルが正常に読み込めることを確認"""
    model = HookedTransformer.from_pretrained("gpt2")
    assert model is not None
    assert model.cfg.model_name == "gpt2"
    print(f"✓ Model loaded: {model.cfg.model_name}")
    print(f"✓ Number of layers: {model.cfg.n_layers}")
    print(f"✓ Hidden size: {model.cfg.d_model}")


def test_gpt2_inference():
    """基本的な推論が動作することを確認"""
    model = HookedTransformer.from_pretrained("gpt2")
    text = "Hello, how are you?"
    
    tokens = model.to_tokens(text)
    logits = model(tokens)
    
    assert logits is not None
    assert logits.shape[0] == 1  # batch size
    assert logits.shape[1] == tokens.shape[1]  # sequence length
    assert logits.shape[2] == model.cfg.d_vocab  # vocab size
    print(f"✓ Inference successful: input shape {tokens.shape}, output shape {logits.shape}")


def test_gpt2_hook_residual_stream():
    """Residual streamのhookが正常に動作することを確認"""
    model = HookedTransformer.from_pretrained("gpt2")
    text = "Thank you very much."
    
    tokens = model.to_tokens(text)
    activations = {}
    
    def save_activation(activation, hook):
        activations[hook.name] = activation.detach()
    
    # 各層のresidual streamをhook
    hook_names = []
    for layer_idx in range(model.cfg.n_layers):
        hook_name = f"blocks.{layer_idx}.hook_resid_pre"
        hook_names.append(hook_name)
        model.add_hook(hook_name, save_activation)
    
    # 推論実行
    _ = model(tokens)
    
    # hookが正常に動作したか確認
    assert len(activations) == len(hook_names)
    for hook_name in hook_names:
        assert hook_name in activations
        activation = activations[hook_name]
        assert activation.shape[0] == 1  # batch size
        assert activation.shape[1] == tokens.shape[1]  # sequence length
        assert activation.shape[2] == model.cfg.d_model  # hidden size
    
    print(f"✓ Hooked {len(activations)} residual stream activations")
    print(f"✓ Activation shape: {activations[hook_names[0]].shape}")


def test_gpt2_hook_mlp_output():
    """MLP出力のhookが正常に動作することを確認"""
    model = HookedTransformer.from_pretrained("gpt2")
    text = "I apologize for the mistake."
    
    tokens = model.to_tokens(text)
    activations = {}
    
    def save_activation(activation, hook):
        activations[hook.name] = activation.detach()
    
    # 最初の層のMLP出力をhook
    hook_name = "blocks.0.hook_mlp_out"
    model.add_hook(hook_name, save_activation)
    
    # 推論実行
    _ = model(tokens)
    
    # hookが正常に動作したか確認
    assert hook_name in activations
    activation = activations[hook_name]
    assert activation.shape[0] == 1  # batch size
    assert activation.shape[1] == tokens.shape[1]  # sequence length
    assert activation.shape[2] == model.cfg.d_model  # hidden size
    
    print(f"✓ Hooked MLP output: {hook_name}")
    print(f"✓ Activation shape: {activation.shape}")


if __name__ == "__main__":
    print("Running GPT-2 hook tests...")
    test_gpt2_model_loading()
    test_gpt2_inference()
    test_gpt2_hook_residual_stream()
    test_gpt2_hook_mlp_output()
    print("\n✓ All tests passed!")

