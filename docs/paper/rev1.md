# **Title**

**Understanding Emotion Circuits in Small Language Models:

An Integrated Analysis from Dataset to Subspace Alignment and Attention-Head Circuits**

---

# **Abstract**

本研究では、GPT-2・Pythia-160M・GPT-Neo-125M の 3 モデルを対象に、感情（gratitude / anger / apology）表現の内部構造を、データセット設計から活性抽出、感情ベクトル推定、サブスペース解析、パッチング、アライメント、Head レベル解析まで一貫して評価した。sentence-end ベースの従来手法に加え、token-based ベクトル、サブスペース構造の比較、線形マッピングによるアライメント、実世界テキストでのパッチング効果、Attention Head レベルの局所回路解析など、全7フェーズで体系的に検証した。結果として、(1) モデル固有の座標系の違いにより生のベクトル比較は無意味であるが、(2) neutral 空間で学習された線形マッピングを用いるとサブスペースがほぼ完全に一致する（cos² ≈ 0.99）、(3) 早期層の注意ヘッド（特に Layer 0 Head 0）が感情表現に寄与する、(4) token-based ベクトルは sentence-end よりも感情区別が明確、といった知見を得た。

本研究は小規模モデルにおける感情表現の実装様式を細かく分解し、機構解釈可能性の視点から感情回路の構造を明らかにした。

---

# **1. Introduction**

大規模言語モデルが自然言語に含まれる複雑な感情ニュアンスを扱えることは知られている。しかしそれがどのように**内部空間（activation space）**で表現され、どの層・どの注意ヘッドが寄与しているのかはブラックボックスである。

本研究の目的は以下である：

1. **感情データセットを統一フォーマットで構築し、モデル内部の活性を抽出する**
2. **感情方向ベクトル・サブスペースをモデルごとに推定し、モデル間の類似性を分析する**
3. **サブスペースアライメントにより、異なるモデルで感情構造が共有されているかを検証する**
4. **Activation Patching によって特定層・特定方向の介入効果を測定する**
5. **Attention Head レベルに降下し、感情表現に寄与する局所回路（circuits）を特定する**

本研究は小規模モデル（≈100M〜160M）に焦点を当て、**感情という抽象的な特性の内部実装を機構レベルで分析する**点が特徴である。

---

# **2. Dataset Construction (Phase 1)**

## **2.1 Baseline と Extended の二種類を構築**

* baseline: 70 prompts × 4 emotions = **280 samples**
* extended: 100 prompts × 4 emotions = **400 samples**

感情カテゴリ：

* gratitude
* anger
* apology
* neutral

CLI を統一化し、どちらのデータセットも同じコードで作成可能にした。

## **2.2 統計特徴**

* 平均文字長：baseline 27.7 / extended 33.0
* 感情カテゴリの分布は完全に均等
* 英語のみで構成

---

# **3. Activation Extraction (Phase 2)**

3 モデルで、4 感情、全サンプルに対し内部活性を抽出：

* GPT-2
* Pythia-160M
* GPT-Neo-125M

結果として、

* baseline: **840 activation files**
* extended: **1,200 activation files**

を統一ディレクトリに格納。

全モデルで安定した速度（60 samples/sec）を達成。

---

# **4. Emotion Vector Estimation (Phase 3)**

## **4.1 sentence-end ベースの従来手法**

感情ごとに層別の平均ベクトルを算出し、cosine similarity で比較。

### **GPT-2 (baseline)**

* gratitude vs anger: **0.48**
* gratitude vs apology: **0.68**
* anger vs apology: **0.75**

→ GPT-2 は中程度の分離。

### **Pythia-160M**

* 全て **0.96〜0.99** と非常に高い

→ 感情方向がほぼ同一で、分離不良。

### **GPT-Neo-125M**

* GPT-2 と類似だが若干高めの相関。

**結論：**sentence-end ベースは多くのモデルで感情分離が不十分。

---

# **5. Token-Based Emotion Vectors (Phase 3.5)**

sentence-end よりも細かい token-level の変化量から感情方向を抽出。

## **主要結果**

### **Pythia-160M で顕著な発見**

* gratitude vs anger: **-0.75**
* gratitude vs apology: **-0.73**

→ **感謝は怒り・謝罪と反対方向**に位置する。

これは sentence-end では捉えられない構造であり、

**token-based は内部空間の真の感情方向を抽出できる**ことを示す。

---

# **6. Subspace Analysis (Phase 3.5 / Phase 6)**

## **6.1 生のサブスペース比較（cross-model overlap）**

* overlap は **0.13〜0.15** と低い

→ モデル間のサブスペースは座標系が異なるため、

生の比較では一致しない。

## **6.2 neutral 空間で学習した線形マッピング**

結果：

* cos² overlap **≈0.99**
* angle improvement **≈+0.84〜0.87**

→ **座標系の違いを補正すると、各モデルはほぼ同じ感情構造を持つ。**

## **6.3 Procrustes alignment**

* 改善はわずか（0.01 程度）

→ neutral 空間での線形マッピングの方が圧倒的に優れている。

---

# **7. Activation Patching (Phase 4 & 5)**

## **7.1 感情ベクトルを介入 (patching)**

* neutral prompt に対し、
* 層ごとに感情方向を足し引きし、
* 出力テキストの sentiment / keyword / politeness を計測。

### **効果まとめ**

* α>0：gratitude 方向 → 文体がより丁寧・肯定的に
* α<0：anger 方向 → 曖昧さが減り、否定的な表現が増える
* 層依存性あり：Layer 6〜7 が最大の効果を持つ

## **7.2 Random Control**

同じノルムのランダムベクトルでは効果なし。

→ **emotion vector に意味的効果があることを証明。**

## **7.3 Real-World テキストでも有効**

SNS / レビュー / ビジネスメールなど35文で検証し、

パッチング効果は再現された。

---

# **8. Attention Head Analysis (Phase 7)**

## **8.1 Head Screening**

* 12 layers × 12 heads × 3 emotions = 432 スコア
* **Layer 0** の head が上位に多い（特に Head 0, 1, 4, 5）

→ **感情表現は早期層の処理に強く依存**する。

## **8.2 Ablation**

Layer 0 Head 0 を無効化：

* sentiment: **-0.0349**

→ 感情表現が減少。

## **8.3 Patching**

Head 0 の出力を移植すると：

* sentiment: **+0.1102**

→ **Layer 0 Head 0 は感情生成に寄与する局所回路。**

---

# **9. Discussion**

## **9.1 座標系の違いと共有構造**

生ベクトルや生サブスペースはモデル間で比較不能だが、

neutral 空間で線形マッピングを学習するとほぼ完全に一致した。

これは以下を示唆する：

* モデルは**同じ機能的構造（emotion manifold）**を持つ
* しかし座標系（basis）がランダムに回転されている
* → alignment 後の比較が必須

## **9.2 感情表現の機構**

結果から、感情表現は

1. 早期層（Layer 0〜3）で初期特徴を構成
2. 中期層（6〜7）で強く増幅され
3. Attention Head 単位で局所的な寄与が存在

という多層構造であることがわかる。

## **9.3 token-based vector の有用性**

sentence-end よりも明確に感情差を抽出可能であり、

特に Pythia-160M のように表面的には分離困難なモデルでも潜在構造を復元できた。

---

# **10. Conclusion**

本研究は、感情表現の内部構造を

**dataset → activations → vectors → subspaces → alignment → patching → heads**

という *階層的分析* によって明らかにした。

主要な結論は以下である：

1. **token-based ベクトルは高精度に感情方向を抽出できる**
2. **異なるモデルでも感情サブスペース構造はほぼ同一である（座標系のみ違う）**
3. **層6〜7がパッチング効果の中心で、文体変化を制御できる**
4. **Layer 0 Head 0 は感情表現の局所回路である**
5. **random control により、効果が単なるノルム追加ではないことを確認**

本研究は、小規模モデルの感情処理機構を精密に分析した点で、

機構解釈（mechanistic interpretability）研究の一例として意義がある。

---

# **11. Future Work**

1. **より大規模モデル（GPT-2 Medium / Pythia 410M）で再検証**
2. **MLP ユニット解析（Neuron lens）**
3. **複数 head の組み合わせによる回路復元**
4. **日本語モデルでの同様の解析**
5. **alignment 後の cross-model patching 実験**
   * GPT-2 の感謝方向を Pythia に注入すると？
   * 感情表現は transfer 可能か？
