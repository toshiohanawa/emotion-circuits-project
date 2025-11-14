"""
個別の感情プロンプトJSONファイルを作成
全感情カテゴリ（gratitude, anger, apology, neutral）のファイルを作成
拡張版（extended）も作成可能
"""
import json
import argparse
from pathlib import Path
from src.data.create_emotion_dataset import EMOTION_PROMPTS


def create_individual_prompt_files(
    data_dir: Path = Path("data"),
    emotions: list = None,
    extended: bool = False
) -> None:
    """
    各感情カテゴリごとの個別JSONファイルを作成
    
    Args:
        data_dir: データディレクトリのパス
        emotions: 作成する感情カテゴリのリスト（Noneの場合は全カテゴリ）
        extended: Trueの場合、拡張版ファイルを作成
    """
    data_dir.mkdir(exist_ok=True)
    
    if emotions is None:
        emotions = list(EMOTION_PROMPTS.keys())
    
    for emotion in emotions:
        if emotion in EMOTION_PROMPTS:
            prompts = EMOTION_PROMPTS[emotion]
            suffix = "_extended" if extended else ""
            output_file = data_dir / f"{emotion}_prompts{suffix}.json"
            
            data = {
                "prompts": prompts
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"✓ Created: {output_file} ({len(prompts)} prompts)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create individual emotion prompt JSON files")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--emotions", type=str, nargs='+', default=None, help="Emotion categories to create")
    parser.add_argument("--extended", action="store_true", help="Create extended version files")
    
    args = parser.parse_args()
    
    create_individual_prompt_files(
        data_dir=Path(args.data_dir),
        emotions=args.emotions,
        extended=args.extended
    )
    print("\n✓ Individual prompt files creation completed!")

