"""
内部活性抽出スクリプトの単体テスト
"""
import json
import tempfile
from pathlib import Path
import torch
from src.models.extract_activations import ActivationExtractor


def test_activation_extractor_initialization():
    """ActivationExtractorの初期化テスト"""
    extractor = ActivationExtractor("gpt2")
    assert extractor.model is not None
    assert extractor.model_name == "gpt2"
    print("✓ ActivationExtractor initialized successfully")


def test_extract_activations_single_text():
    """単一テキストの活性抽出テスト"""
    extractor = ActivationExtractor("gpt2")
    texts = ["Thank you very much."]
    
    activations = extractor.extract_activations(
        texts,
        save_residual_stream=True,
        save_mlp_output=True,
        save_attention=False,
    )
    
    # 基本的な構造を確認
    assert 'residual_stream' in activations
    assert 'mlp_output' in activations
    assert 'tokens' in activations
    assert 'token_strings' in activations
    
    # Residual streamの確認
    assert len(activations['residual_stream']) == extractor.model.cfg.n_layers
    # 各層はテキストごとの活性のリスト
    assert len(activations['residual_stream'][0]) == len(texts)
    assert activations['residual_stream'][0][0].ndim == 2  # [pos, d_model]
    
    # MLP出力の確認
    assert len(activations['mlp_output']) == extractor.model.cfg.n_layers
    assert len(activations['mlp_output'][0]) == len(texts)
    assert activations['mlp_output'][0][0].ndim == 2  # [pos, d_model]
    
    print(f"✓ Extracted activations for single text")
    print(f"  - Residual stream layers: {len(activations['residual_stream'])}")
    print(f"  - MLP output layers: {len(activations['mlp_output'])}")


def test_extract_activations_multiple_texts():
    """複数テキストの活性抽出テスト"""
    extractor = ActivationExtractor("gpt2")
    texts = [
        "Thank you very much.",
        "I'm sorry for the mistake.",
        "I'm frustrated with this.",
    ]
    
    activations = extractor.extract_activations(
        texts,
        save_residual_stream=True,
        save_mlp_output=False,
        save_attention=False,
    )
    
    # テキスト数が正しいことを確認
    assert len(activations['residual_stream'][0]) == len(texts)
    assert len(activations['tokens']) == len(texts)
    assert len(activations['token_strings']) == len(texts)
    
    print(f"✓ Extracted activations for {len(texts)} texts")
    print(f"  - Number of texts: {len(activations['residual_stream'][0])}")


def test_process_dataset():
    """データセット処理のテスト"""
    # 一時的なデータセットファイルを作成
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset.jsonl"
        output_dir = Path(tmpdir) / "output"
        
        # テストデータセットを作成
        test_data = [
            {"text": "Thank you very much.", "emotion": "gratitude", "lang": "en"},
            {"text": "I'm sorry.", "emotion": "apology", "lang": "en"},
        ]
        
        with open(dataset_path, 'w', encoding='utf-8') as f:
            for entry in test_data:
                f.write(json.dumps(entry) + '\n')
        
        # Extractorを作成
        extractor = ActivationExtractor("gpt2")
        
        # データセットを処理（gratitudeのみ）
        extractor.process_dataset(
            dataset_path=dataset_path,
            output_dir=output_dir,
            emotion_label="gratitude",
            save_residual_stream=True,
            save_mlp_output=True,
            save_attention=False,
        )
        
        # 出力ファイルが作成されたことを確認
        output_file = output_dir / "activations_gratitude.pkl"
        assert output_file.exists()
        
        print(f"✓ Processed dataset and saved to {output_file}")


if __name__ == "__main__":
    print("Running activation extraction tests...")
    print()
    
    test_activation_extractor_initialization()
    print()
    
    test_extract_activations_single_text()
    print()
    
    test_extract_activations_multiple_texts()
    print()
    
    test_process_dataset()
    print()
    
    print("✓ All tests passed!")

