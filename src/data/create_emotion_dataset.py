"""
感情プロンプトデータセットを作成するスクリプト
英語のみで、Gratitude、Anger、Apology、Neutralの4カテゴリを収集
"""
import json
from pathlib import Path
from typing import List, Dict


# 感情カテゴリごとのプロンプト例
EMOTION_PROMPTS = {
    "gratitude": [
        # 丁寧な表現
        "Thank you very much for your help.",
        "I really appreciate your assistance.",
        "I'm so grateful for your support.",
        "Thank you for taking the time to help me.",
        "I can't thank you enough for what you've done.",
        "I'm deeply grateful for your kindness.",
        "Thank you sincerely for your help.",
        "I appreciate your efforts very much.",
        "Thank you for being so understanding.",
        "I'm truly thankful for your generosity.",
        
        # 砕けた表現
        "Thanks a lot!",
        "Thanks so much!",
        "I really appreciate it!",
        "You're the best!",
        "Thanks for everything!",
        "Much appreciated!",
        "Thanks a million!",
        "You're awesome!",
        "Thanks for being there!",
        "I owe you one!",
        
        # 直接的表現
        "Thank you.",
        "Thanks.",
        "I appreciate it.",
        "Thank you for that.",
        "I'm grateful.",
        "Thanks for your help.",
        "Thank you for your time.",
        "I appreciate your help.",
        "Thank you for everything.",
        "Thanks for the support.",
        
        # 暗示的表現
        "I couldn't have done it without you.",
        "You made a huge difference.",
        "This means a lot to me.",
        "You've been incredibly helpful.",
        "I'm so glad you were here.",
        "You've been wonderful.",
        "This wouldn't have been possible without you.",
        "You've been a great help.",
        "I'm lucky to have your support.",
        "You've made this so much easier.",
        
        # より多様な表現
        "Thank you for your patience.",
        "I'm grateful for your understanding.",
        "Thank you for your consideration.",
        "I appreciate your thoughtfulness.",
        "Thank you for going above and beyond.",
        "I'm thankful for your guidance.",
        "Thank you for your expertise.",
        "I appreciate your professionalism.",
        "Thank you for your dedication.",
        "I'm grateful for your commitment.",
        
        "Thank you for listening.",
        "I appreciate your feedback.",
        "Thank you for your input.",
        "I'm thankful for your advice.",
        "Thank you for your perspective.",
        "I appreciate your insight.",
        "Thank you for sharing.",
        "I'm grateful for your contribution.",
        "Thank you for your participation.",
        "I appreciate your cooperation.",
        
        "Thank you for your patience with me.",
        "I'm grateful for your flexibility.",
        "Thank you for accommodating me.",
        "I appreciate your understanding.",
        "Thank you for being patient.",
        "I'm thankful for your tolerance.",
        "Thank you for your kindness.",
        "I appreciate your compassion.",
        "Thank you for your empathy.",
        "I'm grateful for your support.",
    ],
    
    "anger": [
        # 丁寧な表現
        "I'm quite frustrated with this situation.",
        "I'm very disappointed with this outcome.",
        "This is extremely frustrating.",
        "I'm not satisfied with this at all.",
        "I find this completely unacceptable.",
        "I'm deeply concerned about this.",
        "This is very problematic.",
        "I'm quite upset about this.",
        "I'm very displeased with this.",
        "This is highly unsatisfactory.",
        
        # 砕けた表現
        "This is so annoying!",
        "I'm really frustrated!",
        "This is ridiculous!",
        "I'm so mad right now!",
        "This is infuriating!",
        "I can't stand this!",
        "This is so frustrating!",
        "I'm really upset!",
        "This is terrible!",
        "I'm so angry!",
        
        # 直接的表現
        "I'm angry.",
        "I'm frustrated.",
        "I'm upset.",
        "I'm annoyed.",
        "I'm disappointed.",
        "This is unacceptable.",
        "I'm not happy about this.",
        "This is wrong.",
        "I'm dissatisfied.",
        "This is frustrating.",
        
        # 暗示的表現
        "This doesn't work for me.",
        "I'm not okay with this.",
        "This is a problem.",
        "I have concerns about this.",
        "This needs to be addressed.",
        "I'm not comfortable with this.",
        "This is concerning.",
        "I have issues with this.",
        "This is problematic.",
        "I'm not pleased with this.",
        
        # より多様な表現
        "I'm frustrated with the delay.",
        "I'm upset about the mistake.",
        "I'm angry about the error.",
        "I'm disappointed with the service.",
        "I'm annoyed by the inconvenience.",
        "I'm frustrated with the lack of communication.",
        "I'm upset about the poor quality.",
        "I'm angry about the unfair treatment.",
        "I'm disappointed with the results.",
        "I'm annoyed by the constant problems.",
        
        "This is unacceptable behavior.",
        "I'm frustrated with the process.",
        "I'm upset about the situation.",
        "I'm angry about the decision.",
        "I'm disappointed with the response.",
        "I'm annoyed by the attitude.",
        "I'm frustrated with the system.",
        "I'm upset about the policy.",
        "I'm angry about the change.",
        "I'm disappointed with the outcome.",
        
        "I'm frustrated that this keeps happening.",
        "I'm upset that nothing is being done.",
        "I'm angry that this was allowed.",
        "I'm disappointed that this wasn't prevented.",
        "I'm annoyed that this continues.",
        "I'm frustrated with the lack of action.",
        "I'm upset about the negligence.",
        "I'm angry about the incompetence.",
        "I'm disappointed with the management.",
        "I'm annoyed by the bureaucracy.",
    ],
    
    "apology": [
        # 丁寧な表現
        "I sincerely apologize for the inconvenience.",
        "I deeply regret the mistake I made.",
        "I'm truly sorry for what happened.",
        "I apologize for any trouble I may have caused.",
        "I'm very sorry for the error.",
        "I sincerely apologize for my actions.",
        "I deeply apologize for the misunderstanding.",
        "I'm truly sorry for the confusion.",
        "I apologize for any harm I may have caused.",
        "I'm very sorry for the delay.",
        
        # 砕けた表現
        "Sorry about that!",
        "My bad!",
        "I'm really sorry!",
        "Sorry for the trouble!",
        "I apologize!",
        "Sorry!",
        "My mistake!",
        "I'm sorry!",
        "Sorry about the mix-up!",
        "I'm really sorry about that!",
        
        # 直接的表現
        "I apologize.",
        "I'm sorry.",
        "I'm sorry for that.",
        "I apologize for the mistake.",
        "I'm sorry about the error.",
        "I apologize for the confusion.",
        "I'm sorry for the inconvenience.",
        "I apologize for the delay.",
        "I'm sorry for the trouble.",
        "I apologize for my mistake.",
        
        # 暗示的表現
        "I take full responsibility for this.",
        "I acknowledge my error.",
        "I understand I made a mistake.",
        "I recognize that I was wrong.",
        "I accept responsibility for this.",
        "I know I should have done better.",
        "I understand the impact of my actions.",
        "I realize I made an error.",
        "I acknowledge my fault.",
        "I take responsibility for the mistake.",
        
        # より多様な表現
        "I apologize for the confusion I caused.",
        "I'm sorry for the misunderstanding.",
        "I apologize for not being clear.",
        "I'm sorry for the miscommunication.",
        "I apologize for the oversight.",
        "I'm sorry for missing that.",
        "I apologize for the error in judgment.",
        "I'm sorry for the poor decision.",
        "I apologize for the lack of attention.",
        "I'm sorry for not being more careful.",
        
        "I apologize for any offense I may have caused.",
        "I'm sorry if I hurt your feelings.",
        "I apologize for my insensitive comment.",
        "I'm sorry for being inconsiderate.",
        "I apologize for my thoughtless behavior.",
        "I'm sorry for not thinking things through.",
        "I apologize for my carelessness.",
        "I'm sorry for not being more mindful.",
        "I apologize for my lack of consideration.",
        "I'm sorry for not being more aware.",
        
        "I apologize for the inconvenience this has caused.",
        "I'm sorry for disrupting your day.",
        "I apologize for taking up your time.",
        "I'm sorry for the trouble this has caused.",
        "I apologize for any frustration I may have caused.",
        "I'm sorry for the stress this may have caused.",
        "I apologize for the problems this has created.",
        "I'm sorry for any difficulties I may have caused.",
        "I apologize for the negative impact.",
        "I'm sorry for any harm I may have done.",
    ],
    
    "neutral": [
        # 情報提供・質問
        "What is the weather like today?",
        "Can you tell me the time?",
        "How does this work?",
        "What is the capital of France?",
        "Can you explain this concept?",
        "What are the main features?",
        "How do I use this?",
        "What is the purpose of this?",
        "Can you provide more information?",
        "What does this mean?",
        
        # 日常的な会話
        "Hello, how are you?",
        "Good morning.",
        "Nice to meet you.",
        "How was your day?",
        "What did you do today?",
        "I see.",
        "That makes sense.",
        "I understand.",
        "That's interesting.",
        "I agree.",
        
        # 事実の記述
        "The meeting is scheduled for 3 PM.",
        "The document is ready.",
        "The report has been completed.",
        "The data shows an increase.",
        "The results indicate success.",
        "The system is operational.",
        "The process is complete.",
        "The task is finished.",
        "The project is on schedule.",
        "The deadline is approaching.",
        
        # より多様な表現
        "I need to check the schedule.",
        "Let me review the information.",
        "I'll look into that.",
        "I'll get back to you.",
        "I'll consider your suggestion.",
        "I'll think about it.",
        "I'll let you know.",
        "I'll keep that in mind.",
        "I'll take note of that.",
        "I'll remember that.",
        
        "The data is available.",
        "The information is correct.",
        "The details are clear.",
        "The explanation is helpful.",
        "The instructions are straightforward.",
        "The process is simple.",
        "The method is effective.",
        "The approach is reasonable.",
        "The solution is practical.",
        "The plan is feasible.",
        
        "I'm working on the project.",
        "I'm reviewing the materials.",
        "I'm analyzing the data.",
        "I'm preparing the report.",
        "I'm organizing the files.",
        "I'm checking the details.",
        "I'm verifying the information.",
        "I'm updating the records.",
        "I'm processing the request.",
        "I'm handling the matter.",
        
        "The weather is nice today.",
        "The temperature is moderate.",
        "The sky is clear.",
        "The day is pleasant.",
        "The conditions are favorable.",
        "The situation is stable.",
        "The environment is calm.",
        "The atmosphere is peaceful.",
        "The setting is comfortable.",
        "The circumstances are normal.",
    ]
}


def create_dataset(output_path: Path, min_samples_per_category: int = 50) -> None:
    """
    感情データセットを作成してJSONL形式で保存
    
    Args:
        output_path: 出力ファイルのパス
        min_samples_per_category: カテゴリごとの最小サンプル数
    """
    dataset = []
    
    for emotion, prompts in EMOTION_PROMPTS.items():
        # 各カテゴリから指定数のサンプルを選択
        selected_prompts = prompts[:min(len(prompts), min_samples_per_category)]
        
        for text in selected_prompts:
            entry = {
                "text": text,
                "emotion": emotion,
                "lang": "en"
            }
            dataset.append(entry)
    
    # JSONL形式で保存
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✓ Dataset created: {output_path}")
    print(f"✓ Total samples: {len(dataset)}")
    for emotion in EMOTION_PROMPTS.keys():
        count = sum(1 for entry in dataset if entry['emotion'] == emotion)
        print(f"  - {emotion}: {count} samples")


if __name__ == "__main__":
    # データディレクトリを作成
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # データセットを作成
    output_file = data_dir / "emotion_dataset.jsonl"
    create_dataset(output_file, min_samples_per_category=50)
    
    print("\n✓ Dataset creation completed!")

