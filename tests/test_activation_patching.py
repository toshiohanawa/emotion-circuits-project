"""
Tests for the upgraded ActivationPatcher (multi-token support and helpers).
"""
import numpy as np

from src.models.activation_patching import ActivationPatcher


def test_alpha_schedule_with_decay():
    """Verify that alpha decay schedules are generated correctly."""
    patcher = ActivationPatcher.__new__(ActivationPatcher)
    schedule = ActivationPatcher._build_alpha_schedule(
        patcher,
        base_alpha=1.0,
        max_steps=3,
        alpha_schedule=None,
        decay_rate=0.5,
    )
    assert schedule == [1.0, 0.5, 0.25]


def test_resolve_patch_positions_windowed():
    """Windowed patching should target the suffix positions."""
    patcher = ActivationPatcher.__new__(ActivationPatcher)
    positions = ActivationPatcher._resolve_patch_positions(
        patcher,
        seq_len=6,
        prompt_len=4,
        patch_window=2,
        patch_positions=None,
        patch_new_tokens_only=False,
    )
    assert positions == [4, 5]


def test_generate_with_patching_multi_token():
    """Ensure the new generation path returns multi-token continuations."""
    patcher = ActivationPatcher("gpt2")
    random_vector = np.random.randn(patcher.model.cfg.d_model)
    prompt = "Hello, thank you for"
    generated = patcher.generate_with_patching(
        prompt,
        random_vector,
        layer_idx=0,
        alpha=0.1,
        max_new_tokens=2,
    )
    assert isinstance(generated, str)
    assert len(generated.split()) >= len(prompt.split())
