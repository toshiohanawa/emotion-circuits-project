# **Title**

**Mechanistic Analysis of Emotion Representations in Small Language Models:

A Corrected and Reproducible Study from Activations to Subspaces and Head Circuits**

---

# **Abstract**

本研究は、GPT-2・Pythia-160M・GPT-Neo-125M の3つの小規模言語モデルを対象に、

感情（gratitude / anger / apology）表現の内部構造を分析したものである。

当初の分析では、activation patching や head-level 介入が文章スタイルに大きな変化をもたらすと報告していたが、Codex を使ったコードベース検証の結果、多くの記述が実装と整合していないことが判明した。本稿 Rev.2 では、**コードおよび実験結果に基づき、事実に整合した形で研究内容を再構成**する。

再検証の結果：

1. **データセット、活性抽出、emotion vectors、subspace overlaps、neutral-space alignment の結果は正確であり、論文記述と整合している。**
2. 一方で、**activation patching（Phase 4）、layer–alpha sweep（Phase 5）、random control、real-world patching、head-level patching（Phase 7）に関する因果的解釈は誤り・過大評価であった。**
3. 実装上、**パッチングは「next-token logits 1ステップ」のみであり、文体変化を評価できる構造ではない**ことが明らかになった。

本稿 Rev.2 では、上記誤りを訂正し、再現可能性と実装の実態に基づき、研究の本質的な貢献点を明確化した。

---

# **1. Introduction**

言語モデル内部の感情表現を理解することは、機構的解釈（mechanistic interpretability）の文脈で重要なテーマである。本研究は、小規模モデルに対し、以下の観点から体系的に解析を行った：

* データセット作成（baseline / extended）
* 活性抽出（per-layer activations）
* 感情方向ベクトル推定（sentence-end / token-based）
* サブスペース解析とモデル間アライメント
* パッチング実験による因果介入
* Attention head の寄与分析

本 Rev.2 では、Codex による厳密なコード照合を経て、事実に基づいた研究内容を再提示する。

---

# **2. Dataset Construction（Phase 1）**

本研究では 4 種類の感情カテゴリ（gratitude, anger, apology, neutral）について：

* baseline：70 prompts × 4 = **280 samples**
* extended：100 prompts × 4 = **400 samples**

を構築した。

英語のみ、長さ分布もコード上の統計と完全に一致する。

**← この部分は元論文と一致しており、正確である。**

---

# **3. Activation Extraction（Phase 2）**

各モデル × 各感情 グループごとに 1 ファイルが生成されるため：

* 3 models × 4 emotions =  **12 activation files** （baseline）
* 同じく 12（extended）

速度ログは記録されておらず、

**以前の「840 files / 60 samples/sec」は誤りである。**

---

# **4. Emotion Vector Estimation（Phase 3, 3.5）**

## **4.1 Sentence-End Based Vectors**

実装：最終 residual を平均化

再現結果：

* GPT-2：0.48 / 0.68 / 0.75
* Pythia-160M：0.96〜0.99（非常に高い相関）

**← コードおよびアーティファクトと完全一致**

## **4.2 Token-Based Vectors**

実装：emotion token の変化量ベース

Pythia-160M では gratitude が anger/apology と逆向きになる

（負の cosine）

**← こちらもアーティファクトと一致**

---

# **5. Subspace Analysis（Phase 3.5, 6）**

## **5.1 Raw Subspace Overlap**

結果：すべて **0.13–0.15**

→ model-specific coordinate system により、直接比較は不可。

## **5.2 Neutral-Space Alignment**

neutral の residual を使って線形マッピングを学習

結果：cos² overlap **0.001 → 0.96–0.99**

**← 実装と完全一致。主要な正しい発見。**

## **5.3 Procrustes Alignment**

改善はごく小さく、~0.01 程度

**← 一致**

---

# **6. Activation Patching（Phase 4）［訂正有り］**

## **6.1 実装仕様（重要な制約）**

Codex により確認された重要事実：

* **生成は「1トークンのみ」**
* トップ3候補の logits を返すだけ
* 文体（style）や長文 sentiment を計算する仕組みはない
* evaluate_patching_effect は **文字列ログのみ**

つまり……

### ❌ 論文 Rev.1 の主張（誤り）

* α>0 で丁寧になる
* α<0 で怒りが増える
* スタイルが変化する
* real-world prompt でも効果が確認された

### ✔ 論文 Rev.2 の正しい主張

Activation patching（Phase 4）は

**「next-token の確率分布がどう変わるかを観察する実験」**

であり、

**文体変化（politeness, positivity）は評価できない。**

---

# **7. Layer–Alpha Sweep（Phase 5）［訂正有り］**

実装は：

* deterministic generation（do_sample=False）
* keyword-based heuristic sentiment/politeness
* 全フレームで効果量は **ほぼゼロ**

つまり……

### ❌ 「Layer 6–7 が最大効果」は誤り

### ✔ 正しくは

「本実装では明確な層依存性は観察されず、

効果量もごく小さい（ほぼ0）。」

---

# **8. Random Control（Phase 4.5）［訂正有り］**

* random_vs_emotion_effect.py は存在するが
* 結果ログ（CSV, PNG）はコミットされていない
* 差分比較のデータが不足

結論：

### ❌ 「random vector は効果を持たない」は未証明

### ✔ 正しくは

「比較結果はコミットされておらず、検証は未完了」

---

# **9. Real-World Patching（Phase 4）［訂正有り］**

* 1-step logits 出力のみ
* sentiment/politeness を測る仕組みは存在しない

### ❌ 「実世界でもスタイル変化が見られた」は誤り

### ✔ 「実世界 prompt に対し next-token 変化のみ観察」と訂正

---

# **10. Head-Level Analysis（Phase 7）［訂正有り］**

## **10.1 Head Screening**

* 実際には Layer 3〜11 が大きく反応
* Layer 0 dominance という主張は誤り

## **10.2 Head Ablation**

* sentiment：−0.0349

  **← 一致（正しい）**

## **10.3 Head Patching**

* sentiment gain：+0.0149（小さい）
* +0.1102 は誤り
* しかも 測定は next-token ベース

### 正しい記述：

「Head patching による sentiment 変化は小さく、

感情生成の因果回路を同定するにはさらなる改善が必要。」

---

# **11. Discussion（修正版）**

本研究の正しく実装に基づく主要結論：

1. **Emotion vectors（sentence-end / token-based）は正確に再現されており、

   モデル内部の感情方向の違いは確かに存在する。**
2. **Subspace alignment による “モデル間の共有構造” は明確で、

   小規模モデルでも emotion manifold が安定していることが分かった。**
3. 一方で、**activation patching・layer sweep・head patching などの 因果介入実験は、現在の実装では「next-token logits の変化」を観察するのみで、文体変化を評価するには不十分である。**
4. よって、**感情回路の因果的解明は未完であり、

   multi-token generation・高度な sentiment scorer・head-level patching 改善などを要する。**

---

# **12. Conclusion（修正版）**

本研究は、感情表現のベクトル構造とサブスペース構造に関しては説得力のある実証を提供した。一方、因果介入によるスタイル制御や attention head レベルの寄与については、実装の制約が大きく、結論を出す段階には至っていない。

今後は、より強力なパッチング実装と長文生成を伴う検証が必要である。

---
