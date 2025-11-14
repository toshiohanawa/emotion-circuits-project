"""
プロンプトJSONファイルからJSONLデータセットを構築。
プロファイル（baseline/extendedなど）を指定すると、一貫したファイルセットを自動解決する。
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

from src.config.project_profiles import list_profiles
from src.utils.project_context import ProjectContext, profile_help_text


def find_prompt_file(data_dir: Path, emotion: str, prefer_extended: bool = False) -> Optional[Path]:
    """
    プロンプトファイルを検索（_extendedを優先）
    
    Args:
        data_dir: データディレクトリのパス
        emotion: 感情ラベル（例: "anger", "gratitude"）
        prefer_extended: Trueの場合、_extendedファイルを優先
    
    Returns:
        見つかったファイルのパス、見つからない場合はNone
    """
    if prefer_extended:
        # まず_extendedファイルを探す
        extended_file = data_dir / f"{emotion}_prompts_extended.json"
        if extended_file.exists():
            return extended_file
        # なければ通常版を探す
        regular_file = data_dir / f"{emotion}_prompts.json"
        if regular_file.exists():
            return regular_file
    else:
        # 通常版を優先
        regular_file = data_dir / f"{emotion}_prompts.json"
        if regular_file.exists():
            return regular_file
        extended_file = data_dir / f"{emotion}_prompts_extended.json"
        if extended_file.exists():
            return extended_file
    
    return None


def build_dataset_from_prompts(
    prompt_files: List[Path] = None,
    output_path: Path = None,
    data_dir: Path = None,
    emotions: List[str] = None,
    emotion_mapping: Dict[str, str] = None,
    prefer_extended: bool = False
) -> Dict:
    """
    プロンプトJSONファイルからJSONLデータセットを構築
    
    Args:
        prompt_files: プロンプトJSONファイルのパスリスト（指定された場合）
        output_path: 出力JSONLファイルのパス
        data_dir: データディレクトリ（prompt_filesがNoneの場合に使用）
        emotions: 感情ラベルのリスト（prompt_filesがNoneの場合に使用）
        emotion_mapping: ファイル名から感情ラベルへのマッピング（Noneの場合は自動推測）
        prefer_extended: Trueの場合、_extendedファイルを優先的に使用
    
    Returns:
        データセット統計情報の辞書
    """
    dataset = []
    emotion_counts = {}
    
    # prompt_filesが指定されていない場合、data_dirとemotionsから自動検索
    if prompt_files is None:
        if data_dir is None:
            data_dir = Path("data")
        if emotions is None:
            emotions = ["gratitude", "anger", "apology", "neutral"]
        
        prompt_files = []
        for emotion in emotions:
            found_file = find_prompt_file(data_dir, emotion, prefer_extended)
            if found_file:
                prompt_files.append(found_file)
                print(f"Found prompt file for {emotion}: {found_file.name}")
            else:
                print(f"Warning: No prompt file found for {emotion}")
    
    # 感情マッピングが指定されていない場合、ファイル名から推測
    if emotion_mapping is None:
        emotion_mapping = {}
        for prompt_file in prompt_files:
            # ファイル名から感情ラベルを抽出（例: gratitude_prompts.json -> gratitude）
            stem = prompt_file.stem.replace("_prompts", "").replace("_extended", "")
            emotion_mapping[str(prompt_file)] = stem
    
    for prompt_file in prompt_files:
        if not prompt_file.exists():
            print(f"Warning: {prompt_file} not found, skipping...")
            continue
        
        # プロンプトファイルを読み込み
        with open(prompt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            prompts = data.get('prompts', [])
        
        # 感情ラベルを取得
        emotion = emotion_mapping.get(str(prompt_file), prompt_file.stem.replace("_prompts", "").replace("_extended", ""))
        
        # データセットエントリを作成
        for text in prompts:
            entry = {
                "text": text,
                "emotion": emotion,
                "lang": "en"
            }
            dataset.append(entry)
        
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + len(prompts)
    
    # JSONL形式で保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # 統計情報を計算
    text_lengths = [len(entry['text']) for entry in dataset]
    stats = {
        'total_samples': len(dataset),
        'emotion_counts': emotion_counts,
        'text_length': {
            'avg': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            'min': min(text_lengths) if text_lengths else 0,
            'max': max(text_lengths) if text_lengths else 0
        }
    }
    
    print(f"✓ Dataset created: {output_path}")
    print(f"✓ Total samples: {stats['total_samples']}")
    for emotion, count in sorted(stats['emotion_counts'].items()):
        print(f"  - {emotion}: {count} samples")
    print(f"✓ Text length: avg={stats['text_length']['avg']:.1f}, min={stats['text_length']['min']}, max={stats['text_length']['max']}")
    
    return stats


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Build JSONL dataset from prompt JSON files. "
                    "Use --profile for deterministic prompt selection."
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=list_profiles(),
        default=None,
        help=f"Dataset profile name ({profile_help_text()}).",
    )
    parser.add_argument("--prompts", type=str, nargs='+', default=None, 
                       help="Prompt JSON file paths (optional, if not specified, auto-finds from --data_dir)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file path (defaults to profile dataset location if --profile is supplied)",
    )
    parser.add_argument("--data_dir", type=str, default="data", 
                       help="Data directory (used when --prompts is not specified)")
    parser.add_argument("--emotions", type=str, nargs='+', default=None,
                       help="Emotion labels to include (default: gratitude, anger, apology, neutral)")
    parser.add_argument("--emotion_mapping", type=str, nargs='+', default=None, 
                       help="Emotion mapping (format: file:emotion, e.g., gratitude_prompts.json:gratitude)")
    parser.add_argument(
        "--prefer_extended",
        action="store_true",
        default=False,
        help="Prefer *_extended files when auto-detecting prompts (deprecated; use --profile).",
    )
    parser.add_argument(
        "--no_prefer_extended",
        action="store_false",
        dest="prefer_extended",
        help="Disable preference for *_extended files (default).",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    context = ProjectContext(args.profile, data_dir=data_dir) if args.profile else None
    
    output_path = Path(args.output) if args.output else None
    if context and output_path is None:
        output_path = context.dataset_path()
    
    if output_path is None:
        raise ValueError("Output path must be provided via --output or inferred from --profile.")
    
    # プロンプトファイルのパスを変換
    prompt_files = None
    if args.prompts:
        prompt_files = [Path(p) for p in args.prompts]
    elif context:
        prompt_files = list(context.prompt_files(args.emotions).values())
    
    # 感情マッピングを解析
    emotion_mapping = None
    if args.emotion_mapping:
        emotion_mapping = {}
        for mapping in args.emotion_mapping:
            if ':' in mapping:
                file_path, emotion = mapping.split(':', 1)
                emotion_mapping[file_path] = emotion
    
    # データセットを構築
    stats = build_dataset_from_prompts(
        prompt_files=prompt_files,
        output_path=output_path,
        data_dir=data_dir,
        emotions=args.emotions,
        emotion_mapping=emotion_mapping,
        prefer_extended=args.prefer_extended
    )
    
    print("\n✓ Dataset build completed!")
    return stats


if __name__ == "__main__":
    main()
