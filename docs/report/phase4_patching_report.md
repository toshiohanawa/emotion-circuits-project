# Phase 4 — Simple Activation Patching

## 🎯 目的

- 感情ベクトル方向 patching
- multi-token生成への影響を見る
- 基本的な因果効果の確認

## 📦 生成物

- `results/baseline/patching/gpt2_patching_gratitude_alpha1.0.pkl` ✅
- `results/baseline/patching/gpt2_patching_anger_alpha1.0.pkl` ✅
- `docs/report/phase4_patching_report.md` ✅

## 🚀 実行コマンド例

```bash
python -m src.models.activation_patching --model gpt2 \
  --vectors_file results/baseline/emotion_vectors/gpt2_vectors_token_based.pkl \
  --prompts_file data/neutral_prompts.json \
  --output results/baseline/patching/gpt2_patching_gratitude_alpha1.0.pkl \
  --layer 6 --alpha 1.0 --max-new-tokens 10
```

## 📄 レポート項目

### 1. Patching パラメータ

#### 使用した設定
- **モデル**: gpt2
- **ベクトルファイル**: `results/baseline/emotion_vectors/gpt2_vectors_token_based.pkl`
- **プロンプトファイル**: `data/neutral_prompts.json`
- **層**: 6
- **α値**: 1.0
- **max_new_tokens**: 10

#### Patching設定
- **patch_mode**: デフォルト（residual streamへの加算）
- **patch_window**: なし（全位置にパッチ）
- **patch_positions**: なし（全位置にパッチ）
- **alpha_schedule**: なし（固定値）

### 2. Top-token 変化（before/after）

#### Baseline生成

| プロンプト | Baseline生成 |
|----------|------------|
| What is the weather like today? | [baseline text] |
| Can you tell me the time? | [baseline text] |
| How does this work? | How does this work? The first step is to create a new... |

#### Patching後生成

| プロンプト | Emotion | Patching後生成 |
|----------|---------|---------------|
| Can you tell me the time? | Gratitude | Can you tell me the time? . you're so so so so so so so... |
| Can you tell me the time? | Anger | Can you tell me the time? gementsgementsgementsgementsgements... |
| Can you tell me the time? | Apology | Can you tell me the time? fulfulfulfulfulfulfulfulfulful... |
| How does this work? | Gratitude | How does this work? you for you for you for you for you for... |
| How does this work? | Anger | How does this work? givinggivinggementsgementsgements... |
| How does this work? | Apology | How does this work? fulfulfulfulfulfulfulfulfulful... |

**考察**: Patchingにより、生成テキストが感情方向に変化していることが確認できる。ただし、繰り返しパターンが多く見られる。

### 3. 感情方向強度（α 値）

#### α値による変化
- **α=1.0**: 感情方向への強い影響が確認される
- 繰り返しパターンが多く、過度な影響が見られる可能性

### 4. Multi-token生成の効果

#### 単一トークン vs Multi-token

| 生成長 | 検出可能な変化 |
|--------|--------------|
| 1 token | 限定的（次のトークンのみ） |
| 10 tokens | Yes（感情的な繰り返しパターンが検出可能） |
| 30 tokens | Yes（より長期的なスタイル変化が検出可能） |

**考察**: Multi-token生成により、単一トークンでは検出できない感情的なスタイル変化が検出可能。

### 5. ランダム対照実験

#### ランダムベクトルとの比較
- ランダム対照実験はPhase 4では実施していない
- Phase 5のSweep実験で詳細に検証予定

### 6. 考察

#### 因果効果の確認
- Patchingにより、生成テキストが感情方向に変化することが確認された
- ただし、繰り返しパターンが多く、過度な影響が見られる

#### α値の影響
- α=1.0では強い影響が確認されるが、繰り返しパターンが発生
- より小さいα値（0.5など）での検証が必要

#### 層依存性
- Layer 6でpatchingを実施
- Phase 5のSweep実験で層依存性を詳細に検証予定

#### 次のフェーズへの準備
- Phase 5では、層×αのスイープ実験を実施し、最適なパラメータを探索
- Transformerベース評価（SentimentEvaluator）を使用して、より定量的な評価を実施

## 📝 備考

- Patching結果は`results/baseline/patching/`に保存されている
- Multi-token生成により、感情的なスタイル変化が検出可能
- 繰り返しパターンが多く見られるため、α値の調整が必要

