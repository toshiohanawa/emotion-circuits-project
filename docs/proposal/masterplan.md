# Emotion Circuits Masterplan（刷新版パイプライン設計）  
更新日: 2025-11-16

この文書は、小型〜中規模 Transformer 言語モデルにおける「感情回路」の構造と因果効果を明らかにするための、**刷新版パイプライン**の設計書である。

過去のパイロット実験（少数サンプル・single-token patching・部分的な実装）は、本ドキュメントでは歴史的経緯として扱い、今後の実装とリランは **本書に記載されたフェーズ構成と仕様** に従う。

---

## 1. ビジョンとゴール

### 1.1 ビジョン

- 機械（LLM）は、**感情に相当する情報を内部でどのように表現し、流し、出力へ反映しているのか？**
- 異なるアーキテクチャ・パラメータ規模のモデル同士で、**「座標系は違うが本質的には同じ」感情サブスペースや回路構造** は存在するのか？
- それを人間が理解できる粒度まで分解し、**因果的に操作可能な単位（方向・サブスペース・ヘッド・小さな回路）** として捉えたい。

### 1.2 本研究のゴール

1. **単一モデル内の感情処理フロー（write-in → propagation → read-out → 出力）** を、活性・サブスペース・ヘッド単位で記述する。
2. **小型〜中規模モデル間で共通する感情サブスペース構造** を発見し、線形アライメントで「座標系の違い」を取り除く。
3. 感情方向・サブスペースを **残差ベクトル patching / head patching** により因果的に操作し、「小さいが一貫した効果」を統計的に示す。
4. 将来的には、複数 head と MLP を含む **小さな感情回路モジュール** を特定する。

---

## 2. 研究のコアな問い

- **Q1: Intra-model emotion flow**  
  単一モデルの内部では、「感謝」「怒り」「謝罪」といった感情はどのような方向・サブスペースとして書き込まれ、どの層・head を通って出力に現れるのか？

- **Q2: Cross-model shared structure**  
  GPT-2 small / Pythia-160M / GPT-Neo-125M などの小型モデルと、Llama3 8B / Gemma3 12B / Qwen3 8B などの中規模モデルの間で、**共通する感情サブスペース構造** は存在するのか？ それは線形写像でどの程度アライメントできるのか？

- **Q3: Causal manipulability**  
  発見された感情方向・サブスペース・head・小さな回路は、**残差ベクトル patching / head patching / ablation** によって因果的に操れるのか？ その効果量はどの程度か？  

---

## 3. 技術フレームワーク

### 3.1 感情処理の概念モデル（4ステップ）

1. **書き込み（write-in）**  
   - 入力トークン（感情を示すフレーズやコンテキスト）から、残差ストリームに感情情報が書き込まれる。

2. **伝搬（propagation）**  
   - self-attention と MLP を通じて、感情情報が層をまたいで伝搬・変形される。

3. **読み出し（read-out）**  
   - 特定の head / MLP / 最終層などが、感情情報を読んで次トークン分布やスタイルに影響を与える。

4. **出力影響（behavioral effect）**  
   - 生成されたテキストの sentiment / politeness / GoEmotions などに、感情回路の影響が現れる。

### 3.2 技術的 3 本柱（T1–T3）

- **T1: Multi-token Activation Patching**  
  - 単一トークンではなく、**生成中の複数トークン window に対して残差ベクトルを注入**し、スタイルや評価指標の変化を見る。

- **T2: Transformer-based Evaluation**  
  - 外部の Transformer ベースの評価器（sentiment, politeness, multi-label emotion）を用いて、  
    感情 patching の効果を一貫した尺度で測定する。

- **T3: Random Baseline & Statistical Testing**  
  - ランダムベクトルによる対照実験、Cohen’s d、t-test、ブートストラップ、  
    Bonferroni / BH-FDR、多重比較補正、power analysis により、  
    「再現性のある小さなシグナル」であることを統計的に確認する。

---

## 4. フェーズ別ロードマップ（刷新版）

フェーズは **「何をするか（分析ステップ）」** で区切り、  
小型〜中規模モデルは各フェーズ内の **対象モデル集合** として扱う。

### Phase 0: インフラ・環境整備

**目的**: すべてのフェーズで共通に使う実行基盤を整える。

- Python 環境・依存関係（Poetry / venv など）
- Hugging Face 認証（Llama3 / Gemma3 / Qwen 系モデル用）
- GPU / MPS / CPU の切り替え（Apple Silicon を含む）
- MLflow・ログ・結果ディレクトリ構造

**対象モデル**: 全モデル（小型・中規模）

**アウトプット**:
- 安定して動作する実行環境
- 共通の `ProjectContext` / registry / モデルラッパ

---

### Phase 1: データセット & 評価設計（全モデル共通の土台）

**目的**: すべてのモデルで共通利用する「感情プロンプト」と「評価指標」を確定する。

- 感情カテゴリ:
  - gratitude / anger / apology / neutral（必要に応じて追加）
- サンプル数:
  - 目標: **約 200〜250 サンプル / 感情**
  - 同一プロンプト集合を全モデルで使い回せる形式にする（CSV 等）
- 評価指標:
  - sentiment（ポジ・ネガ中立）
  - politeness（丁寧さ）
  - GoEmotions 等の多ラベル感情分類  \
    → **Phase 5, 6, 7 でもこの評価器セットを一貫して用いる**

**対象モデル**: 全モデル（小型〜中規模）

**アウトプット**:
- `data/` 以下の標準化されたデータファイル
- 評価器モジュール（T2）のセットアップ

---

### Phase 2: 活性抽出（全モデルの内部表現を揃える）

**目的**: 共通データセット上で、各モデルの hidden states / residual を取得する。

- 小型モデル:
  - GPT-2 small, Pythia-160M, GPT-Neo-125M
- 中規模モデル:
  - Llama3 8B, Gemma3 12B, Qwen3 8B（など、利用可能なもの）

**やること**:
- token-based 位置決定（感情トリガートークンなど）
- 各モデルごとに
  - resid_pre / resid_post など、必要なテンソルを抽出
  - `results/<profile>/activations/<model>.pkl` 等に保存

**アウトプット**:
- モデル × 層 × トークン単位の活性テンソル（共通フォーマット）

---

### Phase 3: 感情ベクトル & サブスペース構築（モデル内構造）

**目的**: 各モデル・各層で、感情方向と感情サブスペースを定義する。

**やること**:
- emotion_vec(layer, emotion) =  \
  mean(resid_emotion) − mean(resid_neutral)
- 差分ベクトル集合に対して PCA を適用し、  \
  **k 次元サブスペース** を構築（例: k=8〜10）
- モデル × 層 × 感情ごとの
  - 感情ベクトル
  - サブスペース（主成分ベクトル＋固有値）

**対象モデル**: 全モデル（小型〜中規模）

**アウトプット**:
- `results/<profile>/emotion_vectors/<model>_vectors_token_based.pkl`
- `results/<profile>/emotion_subspaces/<model>_subspaces.pkl`

---

### Phase 4: モデル間アライメント（サブスペースレベル）

**目的**: 各モデル間で感情サブスペースの共通性を測り、  \
「座標系の違い」を線形写像で取り除けるかを検証する。

**やること**:
- neutral サンプルを用いて、モデル A→B の線形写像（Procrustes / 最小二乗）を学習
- 「before/after」でサブスペース overlap（cos²）を層ごと・k ごとに算出
  - 例: GPT-2 ↔ Pythia / GPT-Neo / Llama3 / Gemma3 / Qwen3
- k=1..k_max の overlap カーブを計算し、  \
  「どの k で最も共通構造が強いか」を調べる

**対象モデル**:
- ペアリング: GPT-2 基準で他モデルと比較（＋必要に応じて中規模同士）

**アウトプット**:
- `results/<profile>/alignment/<modelA>_vs_<modelB>_token_based_full.pkl`
- k ごとの overlap 結果（Phase 7 で統計処理）

---

### Phase 5: 残差ベクトルの causal patching（multi-token 統一）

**目的**: 感情ベクトル / 感情サブスペースが、実際に出力のスタイル・評価指標に影響を与えるかを、**multi-token patching** で検証する。

**やること**:
- 共通仕様（例）:
  - 生成長: 30 トークン
  - patch window: 3〜5 トークン
  - αスケジュール: 固定 or exponential decay（プロジェクト内で1つに決める）
- 各モデルについて:
  - neutral 文に対して emotion_vec / サブスペース方向を残差に注入
  - Phase 1 で決めた評価器（T2）で、sentiment / politeness / GoEmotions の変化を測定
- ランダムベクトルによる対照実験（T3）もここで実行

**対象モデル**:
- まず小型モデル（GPT-2 等）で確立 → 中規模モデルへ展開

**アウトプット**:
- patching 実験の per-sample メトリクス（stats の入力）
- `results/<profile>/patching/residual/<model>_...pkl` など

---

### Phase 6: Head スクリーニング & パッチング（回路の入り口）

**目的**: どの head / 層が感情情報に敏感か、  \
head パッチング／アブレーションにより因果効果を測る。

**やること**:
- head ごとの attention / value 書き込みを、感情 vs neutral で比較
- 上位 head をスコアリング（「感情ヘッド候補」を抽出）
- 上位 head について:
  - head patching（中立→感情／感情→中立）
  - head ablation（出力ゼロ化 等）
- 評価器（T2）とランダム対照（T3）を用いて因果効果を測定

**対象モデル**:
- 計算コストの観点から、主に小型モデル（GPT-2 系）で重点的に実行  \
  （必要に応じて中規模モデルの一部層でも試験）

**アウトプット**:
- head スコアリング結果
- head patching / ablation メトリクス

---

### Phase 7: 統計的厳密性（効果量・有意性・k 選択）

**目的**: Phase 5–6 の patching 結果と Phase 4 のサブスペースアライメントについて、  \
**統計的に厳密な評価** を行う。

**やること**:
- Effect size & significance:
  - Cohen’s d
  - paired / unpaired t-test
  - p 値
  - Bonferroni / BH-FDR 多重比較補正
- Power analysis:
  - 現状サンプル数における post-hoc power
  - 目標効果量（例 d=0.2, 0.3, 0.5）に対する必要サンプル数
- k-selection:
  - k=1..k_max の overlap をブートストラップ
  - k=2（など）の最適性を統計的に検証

**対象モデル**:
- 基本は全モデルの結果を含めるが、必要に応じて小型モデルにフォーカス

**アウトプット**:
- `results/<profile>/statistics/effect_sizes.csv`
- `results/<profile>/statistics/power_analysis.csv / .json`
- `results/<profile>/statistics/k_selection.csv`
- レポート用の図表・要約

---

### 将来の Phase 8: 回路レベル解剖（OV/QK＋multi-head 回路）

**目的**: 単一 head ではなく、複数 head＋MLP を含む **小さな感情回路モジュール** として構造を描き出す。

**やること（構想段階）**:
- 重要 head の OV/QK 行列を解析し、「どのトークンから何を読み取り、どのトークンに何を書いているか」を特定
- 「感情を拾う head」「感情を集約する head」「感情を出力に反映する head」など、役割分担を推定
- 複数 head を同時に patch / ablate し、回路単位の因果効果を測定

---

## 5. 論文パッケージ構想（刷新版）

- **Paper 1**:  \
  「小型〜中規模モデルに共通する感情サブスペース構造と線形アライメント」
  - 対応フェーズ: Phase 1〜4 + 7（統計）
  - 主題: サブスペース構造／座標系の違い／中規模モデルへの汎化

- **Paper 2（将来）**:  \
  「感情方向・サブスペース・head patching による因果的スタイル制御」
  - 対応フェーズ: Phase 5〜7
  - 主題: 小さいが一貫した効果／統計的厳密性／モデル間比較

- **Paper 3（将来）**:  \
  「OV/QK と multi-head 回路による感情回路の構造的解剖」
  - 対応フェーズ: Phase 8 以降
  - 主題: 小さな回路モジュールとしての感情処理

---
