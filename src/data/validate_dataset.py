"""
データセットの検証スクリプト
統計情報を出力し、カテゴリごとの分布を確認
"""
import json
from pathlib import Path
from collections import Counter
from typing import Dict, List


def validate_dataset(dataset_path: Path) -> Dict:
    """
    データセットを検証して統計情報を返す
    
    Args:
        dataset_path: データセットファイルのパス
        
    Returns:
        統計情報の辞書
    """
    dataset = []
    
    # JSONLファイルを読み込み
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    
    # 統計情報を計算
    total_samples = len(dataset)
    emotion_counts = Counter(entry['emotion'] for entry in dataset)
    lang_counts = Counter(entry['lang'] for entry in dataset)
    
    # テキスト長の統計
    text_lengths = [len(entry['text']) for entry in dataset]
    avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    min_length = min(text_lengths) if text_lengths else 0
    max_length = max(text_lengths) if text_lengths else 0
    
    # 単語数の統計
    word_counts = [len(entry['text'].split()) for entry in dataset]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    min_words = min(word_counts) if word_counts else 0
    max_words = max(word_counts) if word_counts else 0
    
    stats = {
        'total_samples': total_samples,
        'emotion_distribution': dict(emotion_counts),
        'language_distribution': dict(lang_counts),
        'text_length': {
            'average': avg_length,
            'min': min_length,
            'max': max_length
        },
        'word_count': {
            'average': avg_words,
            'min': min_words,
            'max': max_words
        }
    }
    
    return stats, dataset


def print_statistics(stats: Dict, dataset: List[Dict]) -> None:
    """
    統計情報を整形して出力
    
    Args:
        stats: 統計情報の辞書
        dataset: データセットのリスト
    """
    print("=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    print(f"\nTotal samples: {stats['total_samples']}")
    
    print("\nEmotion distribution:")
    for emotion, count in sorted(stats['emotion_distribution'].items()):
        percentage = (count / stats['total_samples']) * 100
        print(f"  - {emotion:12s}: {count:3d} samples ({percentage:5.1f}%)")
    
    print("\nLanguage distribution:")
    for lang, count in sorted(stats['language_distribution'].items()):
        percentage = (count / stats['total_samples']) * 100
        print(f"  - {lang}: {count} samples ({percentage:.1f}%)")
    
    print("\nText length statistics:")
    print(f"  - Average: {stats['text_length']['average']:.1f} characters")
    print(f"  - Min: {stats['text_length']['min']} characters")
    print(f"  - Max: {stats['text_length']['max']} characters")
    
    print("\nWord count statistics:")
    print(f"  - Average: {stats['word_count']['average']:.1f} words")
    print(f"  - Min: {stats['word_count']['min']} words")
    print(f"  - Max: {stats['word_count']['max']} words")
    
    # 各感情カテゴリのサンプル例を表示
    print("\n" + "=" * 60)
    print("Sample entries by emotion:")
    print("=" * 60)
    
    for emotion in sorted(stats['emotion_distribution'].keys()):
        samples = [entry for entry in dataset if entry['emotion'] == emotion]
        print(f"\n{emotion.upper()}:")
        for i, entry in enumerate(samples[:3], 1):  # 最初の3つを表示
            print(f"  {i}. {entry['text']}")
        if len(samples) > 3:
            print(f"  ... and {len(samples) - 3} more")


def validate_data_quality(dataset: List[Dict]) -> bool:
    """
    データの品質を検証
    
    Args:
        dataset: データセットのリスト
        
    Returns:
        検証が成功したかどうか
    """
    errors = []
    
    # 必須フィールドの確認
    required_fields = ['text', 'emotion', 'lang']
    for i, entry in enumerate(dataset):
        for field in required_fields:
            if field not in entry:
                errors.append(f"Entry {i}: Missing field '{field}'")
    
    # 感情カテゴリの確認
    valid_emotions = {'gratitude', 'anger', 'apology', 'neutral'}
    for i, entry in enumerate(dataset):
        if entry.get('emotion') not in valid_emotions:
            errors.append(f"Entry {i}: Invalid emotion '{entry.get('emotion')}'")
    
    # 言語の確認
    for i, entry in enumerate(dataset):
        if entry.get('lang') != 'en':
            errors.append(f"Entry {i}: Invalid language '{entry.get('lang')}' (expected 'en')")
    
    # テキストが空でないことを確認
    for i, entry in enumerate(dataset):
        if not entry.get('text', '').strip():
            errors.append(f"Entry {i}: Empty text field")
    
    if errors:
        print("\n" + "=" * 60)
        print("Data Quality Issues Found:")
        print("=" * 60)
        for error in errors:
            print(f"  ✗ {error}")
        return False
    else:
        print("\n" + "=" * 60)
        print("✓ Data quality validation passed!")
        print("=" * 60)
        return True


if __name__ == "__main__":
    dataset_path = Path("data/emotion_dataset.jsonl")
    
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        exit(1)
    
    # データセットを検証
    stats, dataset = validate_dataset(dataset_path)
    
    # 統計情報を出力
    print_statistics(stats, dataset)
    
    # データ品質を検証
    is_valid = validate_data_quality(dataset)
    
    if not is_valid:
        exit(1)
    
    print("\n✓ Dataset validation completed successfully!")

