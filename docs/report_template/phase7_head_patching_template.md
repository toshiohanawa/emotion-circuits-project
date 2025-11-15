# Phase 7 — Head Patching

## 🎯 目的

- head-level の causal patching
- Δsentiment, ΔGoEmotions
- head importance への因果的根拠

## 📦 生成物

- `head_patching.pkl`
- `docs/phase7_report.md`

## 🚀 実行コマンド例

```bash
python -m src.models.head_patching \
  --model gpt2 \
  --head-spec "3:5,7:2" \
  --neutral-prompts data/neutral_prompts.json \
  --emotion-prompts data/gratitude_prompts.json \
  --output results/baseline/patching/head_patching_gpt2_gratitude.pkl \
  --max-tokens 30 \
  --temperature 0.8 \
  --top-p 0.9 \
  --patch-mode result
```

## 📄 レポート項目

### 1. Head Patching設定

#### 実験パラメータ
- モデル: [gpt2 / pythia-160m / gpt-neo-125M]
- 対象Heads: [層:Headのリスト]
- Patch mode: [v_only / pattern_v / result]
- use_attn_result: [True/False]
- 中立プロンプト数: [数]
- 感情プロンプト数: [数]
- max_new_tokens: [数]
- temperature: [値]
- top_p: [値]

### 2. Baseline vs Patched比較

#### メトリクス変化（Δ）

| メトリクス | Baseline | Patched | Δ | 統計的有意性 |
|-----------|---------|---------|---|------------|
| Sentiment (POSITIVE) | [値] | [値] | [値] | [p値] |
| Sentiment (NEGATIVE) | [値] | [値] | [値] | [p値] |
| Sentiment (NEUTRAL) | [値] | [値] | [値] | [p値] |
| Politeness | [値] | [値] | [値] | [p値] |
| Gratitude keywords | [値] | [値] | [値] | [p値] |
| Anger keywords | [値] | [値] | [値] | [p値] |
| Apology keywords | [値] | [値] | [値] | [p値] |

#### GoEmotions変化

| Emotion | Baseline | Patched | Δ |
|---------|---------|---------|---|
| joy | [値] | [値] | [値] |
| anger | [値] | [値] | [値] |
| sadness | [値] | [値] | [値] |
| ... | ... | ... | ... |

### 3. Patch Mode比較

#### v_only vs pattern_v vs result

| Patch Mode | Sentiment Δ | Politeness Δ | 効果の大きさ |
|-----------|------------|-------------|------------|
| v_only | [値] | [値] | [評価] |
| pattern_v | [値] | [値] | [評価] |
| result | [値] | [値] | [評価] |

### 4. Head別の効果

#### 個別Headの効果

| 層:Head | Sentiment Δ | Politeness Δ | 総合評価 |
|---------|------------|-------------|---------|
| 0:0 | [値] | [値] | [評価] |
| 3:5 | [値] | [値] | [評価] |
| 7:2 | [値] | [値] | [評価] |

#### 複数Head同時Patching

| Head組み合わせ | Sentiment Δ | 相乗効果 |
|---------------|------------|---------|
| [0:0] | [値] | - |
| [3:5] | [値] | - |
| [0:0, 3:5] | [値] | [Yes/No] |

### 5. 生成テキストの変化

#### サンプル例

##### Baseline
```
[生成テキスト例1]
[生成テキスト例2]
```

##### Patched (Gratitude)
```
[生成テキスト例1]
[生成テキスト例2]
```

### 6. Multi-token生成の重要性

#### 単一トークン vs Multi-token

| 生成長 | 検出可能な変化 | 理由 |
|--------|--------------|------|
| 1 token | [Yes/No] | [理由] |
| 30 tokens | [Yes/No] | [理由] |

### 7. 考察

#### Head重要性の因果的根拠
- [Head Patchingによる因果的根拠の確認]

#### Patch Modeの選択
- [どのPatch Modeが最も効果的か]

#### Head組み合わせ効果
- [複数Headの相乗効果の有無]

#### 次のフェーズへの準備
- [Phase 8でのOV/QK回路解析への示唆]

## 📝 備考

[その他の注意事項やメモ]

