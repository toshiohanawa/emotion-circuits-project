# 感情回路探索プロジェクト実装計画（完全版）

企画書に基づき、フェーズ0からフェーズ8まで段階的に実装します。3つの軽量モデル（GPT-2 small、Pythia-160M、GPT-Neo-125M）を使用します。最初のパイロットでは英語データのみで実施します。

**更新履歴**: 
- フェーズ3の結果（モデル間類似度がほぼ0）を受けて、フェーズ3.5とフェーズ4ライトを追加
- 本質的な研究クエスチョン「異なるLLMの中に、座標系は違うが本質的には同じ感情表現の部分空間は存在するのか？そしてそれを因果的に操作できるのか？」に基づき、フェーズ5-8を追加
- フェーズ5完了: 層×αのスイープ実験と比較実験を実装完了
- フェーズ6の評価系強化完了: 長文生成・sentiment評価・iterative/swap patchingを実装完了
- Phase 6-8の詳細実装計画を追加: 12モジュールの実装計画と優先順位を明確化
- **Phase 0-6統合実行完了（2024年12月）**: Baseline + Extendedデータセットでの完全再実行、MLflow統合、ランダム対照実験、実世界検証を追加

## プロジェクトの本質的な問い

このプロジェクトは、表向きは「感謝・怒り・謝罪の感情回路探索」ですが、本質は以下の問いです：

**「異なるLLMの中に、"座標系は違うが本質的には同じ" 感情表現の部分空間は存在するのか？そしてそれを因果的に操作できるのか？」**

## フェーズ0: 環境構築（1-2日）✅ 完了

### ディレクトリ構造の作成
- `data/` - データセット保存用
- `notebooks/` - Jupyterノートブック
- `src/` - ソースコード
  - `src/data/` - データ処理モジュール
  - `src/models/` - モデル関連モジュール
  - `src/analysis/` - 分析モジュール
  - `src/visualization/` - 可視化モジュール
- `docs/` - ドキュメント（既存）
- `results/` - 実験結果保存用

### Python環境構築
- `uv`を使用して仮想環境を作成
- Jupyterカーネルの設定
- 依存関係の定義（`pyproject.toml`）

## フェーズ1: 感情プロンプトデータ作成（2-3日）✅ 完了

### データセット作成
- 感情カテゴリごとに50-200文を収集（英語のみ）
  - Gratitude（感謝）: Baseline 70文、Extended 100文
  - Anger（怒り）: Baseline 70文、Extended 100文
  - Apology（謝罪）: Baseline 70文、Extended 100文
  - Neutral（中立）: Baseline 70文、Extended 100文

### 実装済みモジュール
- `src/data/create_individual_prompt_files.py`: 個別プロンプトJSONファイル作成 ✅
- `src/data/build_dataset.py`: プロンプトJSONからJSONLデータセット構築 ✅
- `src/data/validate_dataset.py`: データセット検証 ✅

### MLflow統合
- データセット統計（総サンプル数、感情別カウント、テキスト長統計）をMLflowに記録 ✅

## フェーズ2: 内部活性の抽出（1-2週間）✅ 完了

### 対象モデル（3モデル）
1. **GPT-2 small (124M)** - 古典的構造の基準モデルとして必須
2. **Pythia-160M** - GPT-Neo系アーキテクチャ、訓練データが透明
3. **GPT-Neo-125M** - GPT-Neo系（TinyLlamaの代替）

### 内部活性抽出スクリプト
- `src/models/extract_activations.py`を作成 ✅
- 3モデル全てで全感情カテゴリの活性抽出完了 ✅

## フェーズ3: 感情方向ベクトルの抽出・可視化（1-2週間）✅ 完了

### 感情方向ベクトル抽出
- `src/analysis/emotion_vectors.py`を作成 ✅
- 感情方向ベクトルの計算（文末位置を使用）：
  ```
  emotion_vec[layer] = mean(resid_emotion[layer]) - mean(resid_neutral[layer])
  ```

### 層ごとの強度分析
- L1 norm、L2 normの計算 ✅
- cos-simによる安定性評価 ✅
- 層ごとの強度を可視化（line plot） ✅

### モデル内での感情距離分析
- 感謝 vs 怒り、感謝 vs 謝罪、怒り vs 謝罪のcos-sim計算 ✅
- 可視化 ✅

### モデル間での感情方向類似度
- 3モデル間の感情方向ベクトルのcos-sim計算 ✅
- **重要な発見**: モデル間での類似度がほぼ0（-0.02から0.01の範囲）

### 可視化モジュール
- `src/visualization/emotion_plots.py`を作成 ✅

## フェーズ3.5: 感情方向の再検証（1週間）✅ 完了

### 目的
フェーズ3の結果（モデル間類似度がほぼ0）を受けて、より適切な方法で感情方向を検証し、「モデルをまたぐ感情構造」の示唆があるかを判断する。

### ステップ3.5-1: 感情語トークンでベクトルを取り直す ✅ 完了
- `src/analysis/emotion_vectors_token_based.py`を作成 ✅
- **重要な発見**: Pythia-160Mの「全部0.99」現象が大幅に解消

### ステップ3.5-2: 感情サブスペースでの比較 ✅ 完了
- `src/analysis/emotion_subspace.py`を作成 ✅
- **重要な発見**: モデル間のサブスペースoverlapが0.13-0.15（ランダムより高い）

## フェーズ4ライト: 各モデル内での因果性確認（1週間）✅ 完了

### 目的
モデル間の比較でモヤモヤしているときは、まず「各モデル内で、この感情方向が本当に効いているか」を確かめる。

### Activation Patching実装（簡易版）✅ 完了
- `src/models/activation_patching.py`を作成 ✅
- GPT-2とGPT-Neo-125Mで実施 ✅
- Baseline + Extendedデータセットで実行済み ✅

### ランダム対照実験 ✅ 追加実装
- `src/models/activation_patching_random.py`: ランダム対照ベクトルでのPatching実験 ✅
- `src/analysis/random_vs_emotion_effect.py`: ランダム vs 感情ベクトルの効果比較 ✅
- 同じL2ノルムを持つランダムベクトルを生成し、感情ベクトルとの効果を統計的に比較
- Extendedデータセットで実行済み ✅

### 実世界検証 ✅ 追加実装
- `data/real_world_samples.json`: 実世界テキストサンプル（SNS、レビュー、メール）35文 ✅
- 実世界テキストでのPatching実験を実行 ✅

## フェーズ5: 因果性をちゃんと殴る（Activation Patching完全版）（2-3週間）✅ 完了

### 目的
「この方向 / サブスペースは本当に感情を担っている」と言い切るための実験をやりきる。

### MLflow統合 ✅
- 全Sweep実験のメトリクス（層数、alpha値、感情数）をMLflowに記録 ✅
- Baseline + Extendedデータセットの結果を統合的に追跡 ✅

### ステップ5-1: 層×αのスイープ実験 ✅ 完了

**実装内容**:
- `src/models/activation_patching_sweep.py`を作成 ✅
- 中立プロンプトを30-50文に増やす
- 感謝・怒り・謝罪のtoken-based emotion_vecを使用
- 次のグリッドでpatchingを実行：
  - **層**: L = {3, 5, 7, 9, 11}
  - **スケールα**: {-2, -1, -0.5, 0, 0.5, 1, 2}
- 各設定で生成テキストを保存し、以下の指標でスコアリング：
  - 感謝語・怒り語・謝罪語の頻度
  - 文のポジ/ネガ（sentiment classifier）
  - "丁寧さ" proxy（please, kindly, thank you, sorryなどの出現）

**期待される成果物**:
- 層×αのヒートマップ（論文の1枚図レベルの可視化）
- 「どの層 + どのαが、どの感情トーンに一番効くか」の明確な結果

### ステップ5-2: 文末 vs 感情語ベクトルの「因果力」比較 ✅ 完了

**実装内容**:
- `src/analysis/compare_patching_methods.py`を作成 ✅
- `src/visualization/patching_heatmaps.py`を作成 ✅
- 同じpatching実験を以下で比較：
  - 文末ベースのemotion_vec
  - 感情語トークンベースのemotion_vec
- 全く同じ条件で比較 ✅

**期待される結果**:
- 感情語トークンベースの方が、少ない層・小さいαでトーンが変わる
- 文末ベースはノイズが多く、効果が弱い／不安定
- → 「感情は局所トークンに宿る」という主張を因果実験付きで言える

## フェーズ6: 構造の正体を暴く（サブスペース & アライメント）（2-3週間）✅ 完了

### 目的
「overlap 0.13-0.15 > ランダム」という結果の先に、「何が同じで何が違うのか」を一段深く掘る。

### ステップ6-0: 評価系の強化（実施済み）✅ 完了

**目的**: フェーズ6の核心実験を実行するための評価系の基盤構築

**実装内容**:
- `src/analysis/sentiment_eval.py`を作成 ✅
  - 30トークン程度の長文生成サポート
  - 感謝語・怒り語・謝罪語の頻度計算
  - HuggingFaceのsentimentモデル統合（distilbert-base-uncased-finetuned-sst-2-english）
  - JSON/Pickle形式で結果保存
- `src/visualization/sentiment_plots.py`を作成 ✅
  - sentimentスコアの可視化
  - 感情キーワード頻度の可視化
- `src/models/activation_patching_iterative.py`を作成 ✅
  - 各生成ステップでhookを発火
  - 特定層のresidualにα * directionを加算
  - 10〜50トークン生成に対応
- `src/models/activation_patching_swap.py`を作成 ✅
  - 中立プロンプトAと感謝プロンプトBをforward
  - Aの層L, posPのresidualを感謝文Bのresidualに置換
  - その後Aを継続生成

**結果**: 評価系の基盤が整い、より包括的な評価が可能になった。詳細は`docs/phase1_expansion_report.md`を参照。

### ステップ6-0.5: サブスペース解析ユーティリティ（共通基盤）✅ 優先実装

**目的**: フェーズ6の各実験で共通して使用するPCA・主角度・overlap計算・アライメント機能を共通化

**実装内容**:
- **新規ファイル**: `src/analysis/subspace_utils.py`（最優先で実装）
  - **PCAサブスペース計算**: 活性データからPCAでサブスペースを抽出
  - **主角度（principal angles）計算**: 2つのサブスペース間の主角度を計算
  - **Overlap計算（cos^2）**: サブスペース間のoverlap scoreを計算
  - **Procrustesアライメント**: 直交行列での回転アライメント
  - **CCAアライメント**: Canonical Correlation Analysis
  - **層ごとのresidual読み込み補助関数**: 活性データから層ごとのresidualを読み込むユーティリティ

**使用されるモジュール**:
- `src/analysis/model_alignment.py`が呼び出す
- `src/analysis/subspace_k_sweep.py`が呼び出す
- `src/analysis/subspace_alignment.py`が呼び出す
- `src/analysis/principal_angles.py`が呼び出す（または統合）

**実装方針**:
- NumPyベースで実装（sklearn依存を最小化）
- 各関数は独立して使用可能
- 層ごとの処理を効率化するためのバッチ処理関数も提供

### ステップ6-1: Neutralで線形写像を学習し、Emotionで検証する ✅ 完了

**狙い**: 「中立文の表現空間でモデル同士を線形に合わせると、感情サブスペースはどれくらい揃うのか？」

**実装内容**:
- **実装済み**: `src/analysis/model_alignment.py` ✅
  - GPT-2とPythiaで、中立プロンプトのresidualを大量に取得
  - 層ごとに線形写像Wを学習: W = argmin ||WX - Y||（最小二乗）
  - 学習したWを使って感情サブスペースを写像し、overlapを計算
  - 「写像前のoverlap（~0.001）」と「写像後のoverlap（~0.99）」を比較
  - **重要な発見**: 線形写像によりoverlapが0.001から0.99に大幅改善
  - Baseline + Extendedデータセットで実行済み ✅

**解釈**:
- overlapが大きく増える → 実はほぼ同じ構造で、座標変換だけ違う
- あまり増えない → 感情の埋め方が本質的に違う

**必要な補助モジュール**:
- `src/analysis/subspace_utils.py`: 共通ユーティリティ（PCA、主角度、overlap計算）
- `src/visualization/alignment_plots.py`: アライメント結果の可視化

### ステップ6-2: サブスペース次元を振ってみる（k = 2, 5, 10, 20）✅ 完了

**実装内容**:
- **実装済み**: `src/analysis/subspace_k_sweep.py` ✅
  - 各感情ごとにPCAの次元数kを変えてoverlapを測定
  - k = {2, 5, 10, 20}で実験
  - 各kについて感情サブスペースをPCA(k)で抽出し、モデル間overlapを計算
  - 結果を`results/baseline/alignment/k_sweep.json`に保存
  - kを増やしたときのoverlapの伸び方をプロット
  - 可視化: `src/visualization/layer_subspace_plots.py`を使用 ✅
  - Baseline + Extendedデータセットで実行済み ✅

**見たいもの**:
- 小さいk（2-3）で既にoverlapが高い → 「コアとなるshared factorが小さい次元で存在」
- kを増やさないとoverlapが出ない → 「広いが薄い、にじんだ共通性」

**補助分析**:
- `src/analysis/subspace_clusterability.py`: サブスペースのクラスタリング可能性を分析

## フェーズ7: 局所回路への降下（Head / Unitレベル）（2-3週間）

### 目的
「ベクトル・サブスペース」レベルから、「どの頭（head）がやっているの？」に降りる。

### 共通ルール
- **既存構造を尊重**: 既存のディレクトリ構造・モジュール構成は変更しない
- **インポートパスの統一**: `from src.〜`形式に統一（相対インポートは使わない）
- **CLIスクリプト化**: すべての新規モジュールは`argparse`でCLI実行可能にする
- **保存先の一貫性**: 解析結果は`results/{profile}/`以下、データは`.pkl`/`.json`、図は`.png`
- **GPU/メモリ配慮**: `torch.no_grad()`使用、hook使用後は`handle.remove()`で解除

### ステップ7-1: 感情語トークンに強く反応するheadをスクリーニング（未実施）

**実装ファイル**: `src/analysis/head_screening.py`（新規）

**目的**: 各層・各headが感情語トークンに対してどれくらい特異的に反応しているかを定量化

**入力**:
- モデル名（例: `gpt2`）
- プロンプトJSON:
  - `data/gratitude_prompts.json`
  - `data/anger_prompts.json`
  - `data/apology_prompts.json`
  - `data/neutral_prompts.json`

**CLI仕様**:
```bash
python -m src.analysis.head_screening \
  --model gpt2 \
  --device cuda \
  --output results/baseline/alignment/head_scores_gpt2.json
```

**実装要件**:
1. TransformerLensでGPT-2をロード
2. 各感情カテゴリごとにプロンプトを読み込み
3. Forward時に以下をhookで回収：
   - `blocks.{layer}.attn.hook_pattern`（[batch, head, query_pos, key_pos]）
   - `blocks.{layer}.hook_mlp_out`（MLP出力）
4. 感情語トークン位置を特定（既存のtoken-basedベクトル抽出ロジックを流用）
5. 各headについて：
   - 感情カテゴリごとの平均attention（例：感謝 vs 中立）
   - `score = mean(Δattn over emotion tokens)` を計算
   - 必要ならMLP出力のノルム差も計算
6. 結果をJSON保存：
   ```json
   {
     "model": "gpt2",
     "layers": 12,
     "heads_per_layer": 12,
     "scores": [
       {
         "layer": 3,
         "head": 5,
         "emotion": "gratitude",
         "delta_attn": 0.123,
         "delta_mlp_norm": 0.045
       }
     ]
   }
   ```

**保存先**: `results/baseline/alignment/head_scores_{model}.json`

### ステップ7-2: Head ablation実験（未実施）

**実装ファイル**: `src/models/head_ablation.py`（新規）

**目的**: 「怪しいhead」をゼロアウトして生成させ、感情トーンやsentimentがどう変わるかを見る

**CLI仕様**:
```bash
python -m src.models.head_ablation \
  --model gpt2 \
  --device cuda \
  --head-spec "3:5,7:2" \
  --prompts-file data/gratitude_prompts.json \
  --output results/baseline/patching/head_ablation_gpt2_gratitude.pkl
```

**実装要件**:
1. GPT-2をTransformerLensでロード
2. `--head-spec`をパース: `"layer:head"`のカンマ区切り → `[(layer_idx, head_idx), ...]`
3. Generateまたはトークン逐次生成をhook付きで実行
4. Hook対象: `blocks.{layer}.attn.hook_result`
5. Ablation方法: 対象headのテンソルを0にする（他headはそのまま）
6. ベースラインとの比較：
   - ベースライン生成（no ablation）
   - Ablation生成（対象headゼロ）
   - それぞれに対して以下を計算：
     - 感情キーワード頻度（既存のキーワードリストを再利用）
     - Sentimentスコア（distilbert-base-uncased-finetuned-sst-2-englishを再利用）
7. 結果保存（pkl形式）:
   ```python
   {
     "model": "gpt2",
     "heads": [(3,5), (7,2)],
     "prompts": [...],
     "baseline_texts": [...],
     "ablation_texts": [...],
     "baseline_metrics": {...},
     "ablation_metrics": {...}
   }
   ```

**保存先**: `results/baseline/patching/head_ablation/`

### ステップ7-3: Head patching実験（未実施）

**実装ファイル**: `src/models/head_patching.py`（新規）

**目的**: 感謝文のあるhead出力を、中立文に「移植」するSwap Patchingのhead版

**CLI仕様**:
```bash
python -m src.models.head_patching \
  --model gpt2 \
  --device cuda \
  --head-spec "3:5,7:2" \
  --neutral-prompts data/neutral_prompts.json \
  --emotion-prompts data/gratitude_prompts.json \
  --output results/baseline/patching/head_patching_gpt2_gratitude.pkl
```

**実装要件**:
1. 2段階処理：
   - ステップ1：感謝プロンプトでforwardし、指定headの出力を保存
   - ステップ2：中立プロンプトでforward中に、該当層・位置のhead出力を「感謝側」の値に差し替え
2. Hook対象: `blocks.{layer}.attn.hook_result`
3. 位置の扱い：シンプルに「最後のトークン位置」または感情語トークン位置を対象
4. 評価：
   - 生成テキスト（patchなし / patchあり）を比較
   - 感情キーワード頻度
   - Sentimentスコア
5. 結果形式：`head_ablation.py`と同様のstructでpklに保存

**保存先**: `results/baseline/patching/head_patching/`

### ステップ7-4: Head解析結果の可視化（未実施）

**実装ファイル**: `src/visualization/head_plots.py`（新規）

**目的**: head_screening・head_ablation・head_patchingの結果を可視化

**CLI仕様**:
```bash
python -m src.visualization.head_plots \
  --head-scores results/baseline/alignment/head_scores_gpt2.json \
  --ablation-file results/baseline/patching/head_ablation_gpt2_gratitude.pkl \
  --patching-file results/baseline/patching/head_patching_gpt2_gratitude.pkl \
  --output-dir results/baseline/plots/heads
```

**生成する図**:
1. Head反応度heatmap: x=head index, y=layer index, 色=Δactivation（emotion vs neutral）
2. Ablationによるsentiment変化: 棒グラフ or 箱ひげ図（baseline vs ablationのsentiment mean）
3. Patchingによる感情キーワード増減: patch前後でのキーワード出現数を比較する棒グラフ

**保存先**: `results/baseline/plots/heads/`

## フェーズ8: モデルサイズと普遍性（1-2週間）

### 目的
小型モデルでやりきった後、より大きなモデルでの検証。モデルサイズと感情サブスペースの普遍性の関係を探る。

### 共通ルール
- **既存構造を尊重**: 既存のディレクトリ構造・モジュール構成は変更しない
- **インポートパスの統一**: `from src.〜`形式に統一
- **CLIスクリプト化**: すべての新規モジュールは`argparse`でCLI実行可能にする
- **保存先の一貫性**: 解析結果は`results/{profile}/`以下に保存
- **GPU/メモリ配慮**: `torch.no_grad()`使用、hook使用後は適切に解除

### ステップ8-1: HuggingFaceモデル用フック（未実施）

**実装ファイル**: `src/utils/hf_hooks.py`（新規）

**目的**: HuggingFaceモデル（GPT-2 medium/large、Pythia-410M、Llama-3など）からresidual streamやattention出力を簡便に取得するための共通ヘルパー

**API例**:
```python
from src.utils.hf_hooks import load_hf_causal_lm, capture_residuals

model, tokenizer = load_hf_causal_lm("gpt2-medium", device="cuda")

with capture_residuals(model, layers=[3,5,7]) as cache:
    outputs = model(**inputs)

resid_layer5 = cache["resid", 5]  # [batch, seq, d_model]
```

**実装要件**:
1. `load_hf_causal_lm(model_name: str, device: str) -> (model, tokenizer)`
2. `capture_residuals(model, layers: List[int])` → contextmanagerでhook登録 & 解除
3. 必要ならattention、MLP出力も同様にサポート

**対応モデル**:
- `gpt2`, `gpt2-medium`, `gpt2-large`
- `EleutherAI/pythia-410m-deduped`
- `meta-llama/Meta-Llama-3-8B`（オプション）

### ステップ8-2: 既存解析のマルチモデル対応拡張（未実施）

**対象モジュール**:
- `src/analysis/emotion_subspace.py`
- `src/analysis/subspace_k_sweep.py`
- `src/analysis/model_alignment.py`
- `src/analysis/subspace_alignment.py`

**拡張方針**:
1. すべてのスクリプトに`--model-a`, `--model-b`などの引数を追加し、GPT-2 small以外にも指定可能にする
2. モデルロード部分を関数に切り出し、`gpt2`, `gpt2-medium`, `gpt2-large`, `EleutherAI/pythia-410m-deduped`などを切り替え可能にする
3. 既存のGPT-2 small用のロジックは維持しつつ、HuggingFaceモデルの場合は`hf_hooks.py`を使ってresidualを取得

**対象モデル**:
- GPT-2 small (124M) ✅ 既に実施済み
- GPT-2 medium (355M)
- GPT-2 large (774M)
- Pythia-410M
- Llama-3 8B（オプション）

**実行例**:
```bash
python -m src.analysis.subspace_k_sweep \
  --model-a gpt2 \
  --model-b gpt2-medium \
  --layers 3 5 7 9 11 \
  --k-values 2 5 10 20 \
  --output results/baseline/alignment/k_sweep_gpt2_gpt2medium.json
```

### ステップ8-3: モデルサイズとサブスペース共通性の解析（未実施）

**実装パターン**:
すべて以下の流れを再利用：
1. 各モデルで感謝/怒り/謝罪サブスペース（PCA, k=10）を計算
2. モデルペアごとに：
   - Principal angles / overlapを計算
   - Procrustesアライメントを適用
   - Neutral空間での線形写像（Phase 6-1の手法）も適用
   - k-sweepでk={2,5,10,20}の変化も見る

**想定するCLI実行例**:
```bash
# gpt2 vs gpt2-medium
python -m src.analysis.model_alignment \
  --model-a gpt2 \
  --model-b gpt2-medium \
  --layers 3 5 7 9 11 \
  --output results/baseline/alignment/model_alignment_gpt2_gpt2medium.pkl

python -m src.analysis.subspace_k_sweep \
  --model-a gpt2 \
  --model-b gpt2-medium \
  --layers 3 5 7 9 11 \
  --k-values 2 5 10 20 \
  --output results/baseline/alignment/k_sweep_gpt2_gpt2medium.json
```

**研究クエスチョン**:
- 「モデルサイズが大きくなると、感情サブスペースの共通性はどう変化するか？」
- 「線形写像によるアライメント効果はモデルサイズに依存するか？」
- 「低次元でのコア因子（k=2）はモデルサイズに依存するか？」

## フェーズ9: 結果整理と評価（1週間）

### 結果の統合
- 各フェーズの結果をまとめる
- チェック項目の確認：
  - 感情方向の安定性
  - モデル間の類似度（サブスペースレベル）
  - Activation patchingの効果
  - 線形写像によるアライメント効果
  - Headレベルの回路特定
  - 小型モデルでの分析可能性
  - コードの拡張性

### レポート作成
- 実験結果のまとめ
- 可視化結果の整理
- 「観察 → 因果 → 構造 → アライメント」までの一貫した物語の構築
- 今後の拡張案の検討

## 研究の物語性

このプロジェクトは以下の流れで「一つの物語」として閉じます：

1. **観察**: フェーズ3で「モデル間類似度がほぼ0」を発見
2. **再検証**: フェーズ3.5で「感情語トークン」「サブスペース」で改善を発見
3. **因果**: フェーズ5で「この方向が本当に効いている」を証明
4. **構造**: フェーズ6で「座標系は違うが部分空間は似ている」を数式レベルで証明
5. **局所回路**: フェーズ7で「どのheadが担っているか」を特定

「小さい世界を最後まで理解する」ことに集中し、GPT-2 / Pythia / Neoという"箱庭"の中で、一貫した物語を構築します。

---

## Phase 6-8 実装モジュール一覧

### 第一優先（Phase 6の核心実験）

0. **`src/analysis/subspace_utils.py`** ⭐ 最優先（共通基盤）
   - PCAサブスペース計算
   - 主角度（principal angles）計算
   - Overlap計算（cos^2）
   - Procrustesアライメント
   - CCAアライメント
   - 層ごとのresidual読み込み補助関数
   - 他の全モジュールがこのユーティリティを使用

1. **`src/analysis/model_alignment.py`**
   - Neutral residualを用いてGPT-2 → Pythiaの線形写像Wを学習
   - 感情サブスペースをWで変換しoverlapを計算
   - `subspace_utils.py`を使用してoverlap計算
   - 保存先: `results/{profile}/linear_maps/layer_{L}.pt`

2. **`src/analysis/subspace_k_sweep.py`**
   - k={2,5,10,20}に対するサブスペースoverlapを計算
   - `subspace_utils.py`を使用してPCAとoverlap計算
   - 保存先: `results/baseline/alignment/k_sweep.json`

3. **`src/analysis/principal_angles.py`**（オプション）
   - `subspace_utils.py`の関数をラップする形で実装
   - または`subspace_utils.py`に統合

4. **`src/analysis/subspace_alignment.py`** ✅ 実装済み
   - PCA後の10次元サブスペースに対し
   - Procrustesアライメントを適用
   - before/afterのoverlapを比較
   - Baseline + Extendedデータセットで実行済み ✅

### 第二優先（Phase 7のhead-level解析）

5. **`src/analysis/head_screening.py`**
   - 各headのΔactivationによる感情反応度をランキング
   - 保存先: `results/baseline/alignment/head_scores.json`

6. **`src/models/head_ablation.py`**
   - 指定headのzero-outによるcausal ablation
   - 保存先: `results/baseline/patching/head_ablation/`

7. **`src/models/head_patching.py`**
   - 感情文のhead出力を中立文へswapするpatching
   - 保存先: `results/baseline/patching/head_patching/`

### 第三優先（補助モジュール・可視化）

8. **`src/analysis/subspace_clusterability.py`**
   - サブスペースのクラスタリング可能性を分析

9. **`src/visualization/alignment_plots.py`** ✅ 実装済み
   - アライメント結果の可視化
   - Baseline + Extendedデータセットで実行済み ✅

10. **`src/visualization/layer_subspace_plots.py`** ✅ 実装済み
    - 層ごとのサブスペース可視化
    - k-sweep結果の可視化
    - Baseline + Extendedデータセットで実行済み ✅

11. **`src/visualization/head_plots.py`**
    - Head-level解析結果の可視化

12. **`src/utils/hf_hooks.py`**
    - Llama, Mistral対応のHuggingFace hooks
    - Residual stream取得のテンプレート

## ディレクトリ構造（Phase 6-8で追加）

```
emotion-circuits/
├── results/
│   └── {profile}/
│       ├── linear_maps/          ← 新規（W写像）
│       │   └── layer_{L}.pt
│       ├── alignment/            ← 新規
│       │   ├── k_sweep.json
│       │   └── head_scores.json
│       └── plots/
│           ├── alignment/        ← 新規
│           ├── layers/           ← 新規
│           └── heads/            ← 新規
├── src/
│   ├── models/
│   │   ├── head_patching.py  ← 新規
│   │   └── head_ablation.py  ← 新規
│   ├── analysis/
│   │   ├── subspace_utils.py           ← 新規（最優先、共通基盤）
│   │   ├── principal_angles.py          ← 新規（オプション、subspace_utils.pyに統合可）
│   │   ├── subspace_alignment.py       ← 新規
│   │   ├── subspace_clusterability.py  ← 新規
│   │   ├── model_alignment.py          ← 新規
│   │   ├── subspace_k_sweep.py         ← 新規
│   │   └── head_screening.py           ← 新規
│   ├── visualization/
│   │   ├── alignment_plots.py          ← 新規
│   │   ├── layer_subspace_plots.py     ← 新規
│   │   └── head_plots.py               ← 新規
│   └── utils/
│       └── hf_hooks.py                  ← 新規（拡張）
```

## 実装ルール（Phase 6-8共通）

1. 既存ファイルを破壊せず新規ファイルを追加
2. インポートは`from src.〜`の形で統一
3. 生成物はすべて`results/{profile}/`以下に保存
4. パラメータ（層、α、モデル名、k値）は引数で変更可能に
5. Notebook前提ではなくPython CLIで動作
6. 大規模モデルでは`torch.no_grad()`を必ず使用
7. GPUメモリ対策としてHook解除を実装
8. すべてのモジュールはCLI実行可能にする

## 統合実行の成果（2024年12月）

### 完了した統合実行
- **Phase 0-6の完全再実行**: Baseline + Extendedデータセットでの統合実行完了 ✅
- **MLflow統合**: 全フェーズで実験追跡対応 ✅
- **新規モジュール追加**:
  - `src/data/build_dataset.py`: 統合データセット構築 ✅
  - `src/models/activation_patching_random.py`: ランダム対照実験 ✅
  - `src/analysis/random_vs_emotion_effect.py`: ランダム vs 感情効果比較 ✅
- **実世界検証**: 実世界テキストサンプルでの検証追加 ✅

### データセット
- **Baseline**: 280サンプル（70文 × 4感情）
- **Extended**: 400サンプル（100文 × 4感情）
- **実世界**: 35サンプル（SNS、レビュー、メール）

### 主要な発見（統合実行版）
1. **Token-basedベクトルの有効性**: 感情語トークンベースのベクトルがより明確な感情表現を抽出
2. **サブスペースレベルでの共通性**: モデル間overlap 0.13-0.15（ランダムより高い）
3. **線形写像による大幅改善**: Neutral空間での線形写像によりoverlapが0.001から0.99に改善
4. **Extendedデータセットでの検証**: 拡張データセットでも同様のパターンが確認され、発見の頑健性が示された

### レポート
- `docs/report/phase0_setup_report.md` - Phase 0: 環境セットアップ ✅
- `docs/report/phase1_data_report.md` - Phase 1: 統合データセット構築 ✅
- `docs/report/phase2_activations_report.md` - Phase 2: 統合活性抽出 ✅
- `docs/report/phase3_vectors_report.md` - Phase 3: 統合感情ベクトル ✅
- `docs/report/phase3.5_subspace_report.md` - Phase 3.5: 統合サブスペース解析 ✅
- `docs/report/phase4_patching_report.md` - Phase 4: 統合Patching + ランダム対照 + 実世界 ✅
- `docs/report/phase5_sweep_report.md` - Phase 5: 統合Sweep実験 ✅
- `docs/report/phase6_alignment_report.md` - Phase 6: 統合サブスペースアライメント ✅
