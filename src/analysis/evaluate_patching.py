"""
Activation Patchingの結果を評価するスクリプト
感謝語の出現頻度、文末の丁寧さ、全体トーンを評価
"""
import pickle
import re
from pathlib import Path
from typing import Dict, List
from collections import Counter


# 感謝語のキーワード
GRATITUDE_KEYWORDS = [
    "thank", "thanks", "grateful", "appreciate", "appreciation",
    "gratitude", "thankful"
]

# 謝罪語のキーワード
APOLOGY_KEYWORDS = [
    "sorry", "apologize", "apology", "regret", "apologetic"
]

# 怒りのキーワード（評価用、安全性に注意）
ANGER_KEYWORDS = [
    "angry", "frustrated", "terrible", "annoyed", "upset", "mad"
]


def count_keywords(text: str, keywords: List[str]) -> int:
    """
    テキスト内のキーワード出現数をカウント
    
    Args:
        text: テキスト
        keywords: キーワードのリスト
        
    Returns:
        出現数
    """
    text_lower = text.lower()
    count = 0
    for keyword in keywords:
        count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
    return count


def evaluate_politeness(text: str) -> float:
    """
    文末の丁寧さを評価（簡易版）
    
    Args:
        text: テキスト
        
    Returns:
        丁寧さスコア（0-1）
    """
    # 丁寧な表現のパターン
    polite_patterns = [
        r'please\b',
        r'would you\b',
        r'could you\b',
        r'\.$',  # 文末のピリオド
        r'\?$',  # 疑問文
    ]
    
    score = 0.0
    for pattern in polite_patterns:
        if re.search(pattern, text.lower()):
            score += 0.2
    
    return min(score, 1.0)


def evaluate_patching_results(results_file: Path) -> Dict:
    """
    Patching結果を評価
    
    Args:
        results_file: 結果ファイルのパス
        
    Returns:
        評価結果の辞書
    """
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    evaluation = {
        'baseline': {},
        'patched': {}
    }
    
    prompts = results['prompts']
    
    # Baselineの評価
    for prompt in prompts:
        baseline_text = results['baseline'][prompt]
        evaluation['baseline'][prompt] = {
            'gratitude_keywords': count_keywords(baseline_text, GRATITUDE_KEYWORDS),
            'apology_keywords': count_keywords(baseline_text, APOLOGY_KEYWORDS),
            'anger_keywords': count_keywords(baseline_text, ANGER_KEYWORDS),
            'politeness_score': evaluate_politeness(baseline_text)
        }
    
    # Patching結果の評価
    for emotion_label in results['patched'].keys():
        evaluation['patched'][emotion_label] = {}
        
        for alpha, alpha_results in results['patched'][emotion_label].items():
            evaluation['patched'][emotion_label][alpha] = {}
            
            for prompt in prompts:
                patched_text = alpha_results[prompt]
                
                evaluation['patched'][emotion_label][alpha][prompt] = {
                    'gratitude_keywords': count_keywords(patched_text, GRATITUDE_KEYWORDS),
                    'apology_keywords': count_keywords(patched_text, APOLOGY_KEYWORDS),
                    'anger_keywords': count_keywords(patched_text, ANGER_KEYWORDS),
                    'politeness_score': evaluate_politeness(patched_text),
                    'text': patched_text[:200]  # 最初の200文字
                }
    
    return evaluation


def print_evaluation_summary(evaluation: Dict):
    """
    評価結果のサマリーを表示
    
    Args:
        evaluation: 評価結果の辞書
    """
    print("=" * 80)
    print("Activation Patching Evaluation Summary")
    print("=" * 80)
    
    prompts = list(evaluation['baseline'].keys())
    
    for emotion_label in ['gratitude', 'anger', 'apology']:
        if emotion_label not in evaluation['patched']:
            continue
        
        print(f"\n{emotion_label.upper()} Direction Patching:")
        print("-" * 80)
        
        # Alpha値ごとの平均変化を計算
        for alpha in [0.5, 1.0, -0.5, -1.0]:
            if alpha not in evaluation['patched'][emotion_label]:
                continue
            
            avg_gratitude_change = 0.0
            avg_apology_change = 0.0
            avg_anger_change = 0.0
            avg_politeness_change = 0.0
            n_prompts = 0
            
            for prompt in prompts:
                baseline = evaluation['baseline'][prompt]
                patched = evaluation['patched'][emotion_label][alpha][prompt]
                
                avg_gratitude_change += patched['gratitude_keywords'] - baseline['gratitude_keywords']
                avg_apology_change += patched['apology_keywords'] - baseline['apology_keywords']
                avg_anger_change += patched['anger_keywords'] - baseline['anger_keywords']
                avg_politeness_change += patched['politeness_score'] - baseline['politeness_score']
                n_prompts += 1
            
            if n_prompts > 0:
                avg_gratitude_change /= n_prompts
                avg_apology_change /= n_prompts
                avg_anger_change /= n_prompts
                avg_politeness_change /= n_prompts
                
                print(f"\n  Alpha = {alpha:+.1f}:")
                print(f"    Gratitude keywords change: {avg_gratitude_change:+.2f}")
                print(f"    Apology keywords change: {avg_apology_change:+.2f}")
                print(f"    Anger keywords change: {avg_anger_change:+.2f}")
                print(f"    Politeness score change: {avg_politeness_change:+.3f}")
        
        # サンプル出力を表示
        print(f"\n  Sample outputs (α=1.0):")
        for prompt in prompts[:2]:
            patched_text = evaluation['patched'][emotion_label][1.0][prompt]['text']
            print(f"    Prompt: {prompt}")
            print(f"    Output: {patched_text}...")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate activation patching results")
    parser.add_argument("--results_file", type=str, required=True, help="Patching results file")
    parser.add_argument("--output", type=str, default=None, help="Output evaluation file")
    
    args = parser.parse_args()
    
    # 評価を実行
    evaluation = evaluate_patching_results(Path(args.results_file))
    
    # サマリーを表示
    print_evaluation_summary(evaluation)
    
    # 保存
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(evaluation, f)
        
        print(f"\nEvaluation saved to: {output_path}")


if __name__ == "__main__":
    main()

