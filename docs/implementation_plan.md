# 感情回路探索プロジェクト実装計画（完全版）

企画書に基づき、フェーズ0からフェーズ8まで段階的に実装します。3つの軽量モデル（GPT-2 small、Pythia-160M、GPT-Neo-125M）を使用します。最初のパイロットでは英語データのみで実施します。

**更新履歴**: 
- フェーズ3の結果（モデル間類似度がほぼ0）を受けて、フェーズ3.5とフェーズ4ライトを追加
- 本質的な研究クエスチョン「異なるLLMの中に、座標系は違うが本質的には同じ感情表現の部分空間は存在するのか？そしてそれを因果的に操作できるのか？」に基づき、フェーズ5-8を追加
- フェーズ5完了: 層×αのスイープ実験と比較実験を実装完了
- フェーズ6の評価系強化完了: 長文生成・sentiment評価・iterative/swap patchingを実装完了
- Phase 6-8の詳細実装計画を追加: 12モジュールの実装計画と優先順位を明確化
- **Phase 0-6統合実行完了（2024年12月）**: Baseline + Extendedデータセットでの完全再実行、MLflow統合、ランダム対照実験、実世界検証を追加
- **Phase 7完了（2024年11月）**: Head screening、ablation、patching実験を実装・実行完了 ✅
- **インフラ改善（2024年11月）**: MLflow自動フォールバック、モデルディレクトリスラッグ統一化、ログ記録改善、Activation Patching Sweepのメトリクス集計強化 ✅
- **Issue 1-5実装完了（2024年11月）**: Multi-token Activation Patching、Transformerベース評価、ランダム対照強化、実世界評価、Head Patching強化を実装完了 ✅
- **Issue 6-7, 9-10実装完了（2024年11月）**: ニューロン解析、クロスモデルパッチング、MLflow拡張、CI/CDパイプラインを実装完了 ✅
- **Epic Issue: OV/QK Circuit Analysis完了（2024年11月）**: TransformerLens新バックエンド対応、OV/QK回路解析パイプライン、回路可視化・サマリー生成を実装完了 ✅
- **Phase 7.5完了（2025年11月）**: 統計的厳密性の強化（効果量、検出力分析、k選択検証）を実装完了 ✅
- **Phase 8完了（2025年11月）**: 中規模モデル（Llama3 8B / Gemma3 12B / Qwen3 8B）へのスケーリング検証を実装完了 ✅

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
- `scripts/phase2_extract_all_activations.py`: 全モデル×全感情のスイープ実行 ✅

### MLflow統合とログ記録 ✅
- **モデルディレクトリスラッグの統一化**: `src/utils/model_utils.py`で一貫した命名規則を適用 ✅
- **実行ログの記録**: 各実行のstdout/stderrをMLflowアーティファクトとして記録 ✅
- **エラー追跡**: 失敗した実行のログを後で確認可能 ✅

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

### Activation Patching実装（Multi-token対応版）✅ 完了・強化
- `src/models/activation_patching.py`を全面刷新 ✅
  - Multi-token生成に対応（`HookedTransformer.generate`を使用）
  - αスケジュール/減衰を実装（`_build_alpha_schedule`）
  - ウィンドウ/位置指定に対応（`_resolve_patch_positions`）
  - 生成中の継続パッチを実装（`_generate_tokens_with_patch`）
  - CLIオプション追加（`--max-new-tokens`, `--patch-window`, `--patch-positions`, `--patch-new-only`, `--alpha-schedule`, `--alpha-decay-rate`）
- GPT-2とGPT-Neo-125Mで実施 ✅
- Baseline + Extendedデータセットで実行済み ✅

### ランダム対照実験 ✅ 追加実装・強化完了
- `src/models/activation_patching_random.py`: ランダム対照ベクトルでのPatching実験 ✅
  - デフォルトのランダムベクトル数を100に設定 ✅
  - MLflowログオプション追加 ✅
  - Transformerベース評価（`SentimentEvaluator`）を使用 ✅
- `src/analysis/random_vs_emotion_effect.py`: ランダム vs 感情ベクトルの効果比較 ✅
  - 同じL2ノルムを持つランダムベクトルを生成し、感情ベクトルとの効果を統計的に比較
  - Cohen's d計算 ✅
  - パーミュテーション検定 ✅
  - ブートストラップ信頼区間 ✅
  - CSV出力（`--output_csv`, `--distributions_csv`） ✅
  - ヒートマップ/バイオリンプロット生成 ✅
- Extendedデータセットで実行済み ✅

### 実世界検証 ✅ 追加実装・強化完了
- `data/real_world_samples.json`: 実世界テキストサンプル（SNS、レビュー、メール）35文 ✅
- `src/analysis/real_world_patching.py`: multi-token patchingとTransformer指標を用いた実世界プロンプト評価 ✅
  - ベースライン＋各emotion/αでのテキストとメトリクスを保存
  - `ActivationPatcher`と`SentimentEvaluator`を使用
- `src/visualization/real_world_plots.py`: 実世界Patching結果の可視化 ✅
  - 前後比較のバイオリン/バープロット生成
  - ペアt検定（mean_diff/t/p）を出力
  - フラットCSV出力（`real_world_metrics.csv`）
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
- **実際のテキスト生成**: `HookedTransformer.generate`を使用したdeterministic生成 ✅
- **Multi-token生成**: 全生成ステップでhookを効かせる継続パッチ ✅
- **Transformerベース評価**: `SentimentEvaluator`によるRoBERTa系スコアを使用 ✅
  - CardiffNLP sentiment (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
  - Stanford Politeness (`michellejieli/Stanford_politeness_roberta`)
  - GoEmotions (`bhadresh-savani/roberta-base-go-emotions`)
- ヒューリスティック指標を廃止し、Transformerベース評価に完全移行 ✅

**成果物の構造**:
- `baseline_summary`: ベースラインの集約メトリクス（ネスト辞書） ✅
- `aggregated_metrics`: 絶対値メトリクス（感情別、層別、α別、ネスト辞書） ✅
- `delta_metrics`: ベースラインからの差分メトリクス（感情別、層別、α別、ネスト辞書） ✅
- `sweep_results`: 詳細な結果（各プロンプト×層×αの生成テキストとメトリクス） ✅
- MLflowにflattenしてログ化（ネスト辞書対応） ✅

**期待される成果物**:
- 層×αのヒートマップ（論文の1枚図レベルの可視化、`patching_heatmaps.py`で生成） ✅
- バイオリンプロット（分布の比較） ✅
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

### ステップ7-1: 感情語トークンに強く反応するheadをスクリーニング ✅ 完了

**実装ファイル**: `src/analysis/head_screening.py` ✅ 実装済み

**目的**: 各層・各headが感情語トークンに対してどれくらい特異的に反応しているかを定量化

**実装状況**:
- ✅ 実装完了・修正完了・動作確認済み
- ✅ プロファイル対応済み（`--profile`オプション）
- ✅ GPT-2で実行完了（`results/baseline/alignment/head_scores_gpt2.json`）
- ✅ 新しい統計情報を取得（感情語トークンへのattention、サンプル数） ✅

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
  --profile baseline \
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
   - 感情語トークンへのattentionを測定（行全体の平均ではなく、感情語トークン位置のみ） ✅
   - 感情カテゴリごとの平均attention（例：感謝 vs 中立）
   - `score = mean(Δattn over emotion tokens)` を計算
   - サンプル数を記録（`samples_emotion`, `samples_neutral`） ✅
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
         "emotion_mean_attn": 0.037713,
         "neutral_mean_attn": 0.013689,
         "samples_emotion": 70,
         "samples_neutral": 70
       }
     ]
   }
   ```

**保存先**: `results/baseline/alignment/head_scores_{model}.json`

### ステップ7-2: Head ablation実験 ✅ 実装済み（テスト待ち）

**実装ファイル**: `src/models/head_ablation.py` ✅ 実装済み

**目的**: 「怪しいhead」をゼロアウトして生成させ、感情トーンやsentimentがどう変わるかを見る

**実装状況**:
- ✅ コード実装完了
- ✅ プロファイル対応済み（`--profile`オプション）
- ✅ GPT-2で実行完了（`results/baseline/patching/head_ablation/gpt2_gratitude_00.pkl`）

**CLI仕様**:
```bash
python -m src.models.head_ablation \
  --model gpt2 \
  --profile baseline \
  --head-spec "0:0" \
  --emotion gratitude \
  --max-tokens 15
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

### ステップ7-3: Head patching実験 ✅ 実装済み（テスト待ち）

**実装ファイル**: `src/models/head_patching.py` ✅ 実装済み

**目的**: 感謝文のあるhead出力を、中立文に「移植」するSwap Patchingのhead版

**実装状況**:
- ✅ コード実装完了
- ✅ プロファイル対応済み（`--profile`オプション）
- ✅ GPT-2で実行完了（`results/baseline/patching/head_patching/gpt2_gratitude_00.pkl`）
- ✅ 感情プロンプト全体から`pattern`/`V`/`head_output`を平均して使用 ✅
- ✅ `patch_mode`に`v_only`/`pattern_v`/`result`を追加 ✅
- ✅ `use_attn_result=True`起動で`hook_result`による直接パッチ対応 ✅
- ✅ `generate` APIでマルチトークン生成に対応 ✅
- ✅ 複数head同時パッチに対応 ✅

**CLI仕様**:
```bash
python -m src.models.head_patching \
  --model gpt2 \
  --profile baseline \
  --head-spec "0:0,3:5" \
  --emotion gratitude \
  --max-tokens 30 \
  --patch-mode result \
  --use-attn-result
```

**実装要件**:
1. 2段階処理：
   - ステップ1：感謝プロンプト全体でforwardし、指定headの出力を平均して保存（`hook_pattern`と`hook_v`から計算、または`hook_result`を使用）
   - ステップ2：中立プロンプトで`generate` API実行中に、該当層・位置のhead出力を「感謝側」の値に差し替え
2. Hook対象: `patch_mode`に応じて切り替え ✅
   - `v_only`: `hook_v`を使用
   - `pattern_v`: `hook_pattern`と`hook_v`を使用
   - `result`: `hook_result`を使用（`use_attn_result=True`が必要）
3. 位置の扱い：シンプルに「最後のトークン位置」または感情語トークン位置を対象
4. 評価：
   - 生成テキスト（patchなし / patchあり）を比較
   - TransformerベースのSentiment/Politeness/Emotion評価（`SentimentEvaluator`）
5. 結果形式：`head_ablation.py`と同様のstructでpklに保存
6. **メタデータ**: `patch_mode`フィールドで使用したpatching手法を記録 ✅

**保存先**: `results/baseline/patching/head_patching/`

### ステップ7-4: Head解析結果の可視化 ✅ 実装済み（テスト待ち）

**実装ファイル**: `src/visualization/head_plots.py` ✅ 実装済み

**目的**: head_screening・head_ablation・head_patchingの結果を可視化

**実装状況**:
- ✅ コード実装完了
- ✅ プロファイル対応済み（`--profile`オプション）
- ✅ GPT-2で実行完了（`results/baseline/plots/heads/`に4つの図を生成）

**CLI仕様**:
```bash
python -m src.visualization.head_plots \
  --profile baseline \
  --head-scores results/baseline/alignment/head_scores_gpt2.json \
  --ablation-file results/baseline/patching/head_ablation/gpt2_gratitude_00.pkl \
  --patching-file results/baseline/patching/head_patching/gpt2_gratitude_00.pkl \
  --output-dir results/baseline/plots/heads \
  --top-n 20
```

**生成する図**:
1. Head反応度heatmap: x=head index, y=layer index, 色=Δactivation（emotion vs neutral）
2. Ablationによるsentiment変化: 棒グラフ or 箱ひげ図（baseline vs ablationのsentiment mean）
3. Patchingによる感情キーワード増減: patch前後でのキーワード出現数を比較する棒グラフ

**保存先**: `results/baseline/plots/heads/`

## フェーズ8: モデルサイズと普遍性（1-2週間）✅ 完了

### 目的
小型モデルでやりきった後、より大きなモデルでの検証。モデルサイズと感情サブスペースの普遍性の関係を探る。

### 実装済みモジュール

**Phase 8 専用モジュール**:
- `src/models/phase8_large/registry.py`: 中規模モデル（Llama3 8B、Gemma3 12B、Qwen3 8B）の定義 ✅
- `src/analysis/run_phase8_pipeline.py`: Phase 8 自動化パイプライン ✅
- `src/analysis/summarize_phase8_large.py`: Phase 8 結果のサマリー生成 ✅

### ステップ8-1: 中規模モデルのレジストリ ✅ 完了

**実装ファイル**: `src/models/phase8_large/registry.py`

**実装内容**:
```python
MODEL_REGISTRY = {
    "llama3_8b": LargeModelSpec(
        name="llama3_8b",
        family="llama3",
        hf_model_name="meta-llama/Meta-Llama-3.1-8B",
        n_layers_hint=32,
        d_model_hint=4096
    ),
    "gemma3_12b": LargeModelSpec(
        name="gemma3_12b",
        family="gemma3",
        hf_model_name="google/gemma-3-12b-it",
        n_layers_hint=48,
        d_model_hint=3072
    ),
    "qwen3_8b": LargeModelSpec(
        name="qwen3_8b",
        family="qwen3",
        hf_model_name="Qwen/Qwen3-8B-Base",
        n_layers_hint=36
    ),
}
```

### ステップ8-2: サブスペースアライメントパイプライン ✅ 完了

**実装ファイル**: `src/analysis/run_phase8_pipeline.py`

**実装内容**:
- GPT-2の感情サブスペースを中規模モデルへ線形写像でアライメント
- token-based感情ベクトル → 多サンプルPCA → neutral から線形写像学習 → before/after overlap計算
- HuggingFace Transformersの`device_map="auto"`による自動デバイス分割対応
- ゲートモデルへのアクセス（HuggingFaceトークン必要）

**CLI仕様**:
```bash
python3 -m src.analysis.run_phase8_pipeline \
  --target llama3_8b \
  --profile baseline \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 \
  --k 8 \
  --n-samples 50 \
  --device auto
```

### ステップ8-3: 結果と知見 ✅ 完了

**実験結果**:

| モデル | アライメント成功度 | 最大改善幅（Δoverlap） | 特徴 |
|--------|-------------------|------------------------|------|
| **Llama3 8B** | 優秀 | 0.713（Layer 11） | 全層で大幅改善、上位層ほど効果大 |
| **Qwen3 8B** | 中程度 | 0.105（Layer 4） | 中間層で安定した改善 |
| **Gemma3 12B** | 低調 | <0.001 | 数値的不安定性あり |

**主要な発見**:
1. **アーキテクチャ依存性が決定的**: 同じ「感情」概念でも、モデルファミリーによって内部表現の構造が大きく異なる
2. **線形写像の限界**: Gemma3の結果は、単純な線形変換では捉えられない非線形構造の存在を示唆
3. **スケーリングの非自明性**: モデルサイズが大きくなっても、小型モデルの知見が直接適用できるとは限らない

**保存先**:
- `results/{profile}/phase8/`: Phase 8実験結果
- `docs/report/phase8_llama3_scaling_report.md`: Llama3スケーリングレポート
- `docs/report/phase8_comparative_scaling_report.md`: 比較スケーリングレポート

### 共通ルール
- **既存構造を尊重**: 既存のディレクトリ構造・モジュール構成は変更しない
- **インポートパスの統一**: `from src.〜`形式に統一
- **CLIスクリプト化**: すべての新規モジュールは`argparse`でCLI実行可能にする
- **保存先の一貫性**: 解析結果は`results/{profile}/`以下に保存
- **GPU/メモリ配慮**: `torch.no_grad()`使用、hook使用後は適切に解除
- **大規模モデル対応**: `device_map="auto"`による自動デバイス分割対応

### 研究クエスチョンへの回答
- **「モデルサイズが大きくなると、感情サブスペースの共通性はどう変化するか？」**
  → アーキテクチャに依存。Llama3では高い共通性、Gemma3では低い共通性
- **「線形写像によるアライメント効果はモデルサイズに依存するか？」**
  → モデルサイズよりもアーキテクチャ特性に依存
- **「低次元でのコア因子はモデルサイズに依存するか？」**
  → k=2が最適である傾向は小型モデルと共通（Phase 7.5で統計的に検証済み）

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
- **Phase 0-8の完全再実行**: Baselineデータセットでの統合実行完了 ✅
- **MLflow統合**: 全フェーズで実験追跡対応 ✅
- **MLflow改善**: 自動フォールバック（リモートサーバーがダウン時はローカルファイルストアを使用） ✅
- **Phase 7.5追加**: 統計的厳密性の強化（効果量、検出力分析、k選択検証） ✅
- **Phase 8追加**: 中規模モデル（Llama3 8B / Gemma3 12B / Qwen3 8B）へのスケーリング検証 ✅
- **新規モジュール追加**:
  - `src/data/build_dataset.py`: 統合データセット構築 ✅
  - `src/models/activation_patching_random.py`: ランダム対照実験 ✅
  - `src/analysis/random_vs_emotion_effect.py`: ランダム vs 感情効果比較（統計検定付き） ✅
  - `src/utils/model_utils.py`: モデルディレクトリスラッグの統一化 ✅
  - `src/models/head_ablation.py`: Head ablation実験 ✅
  - `src/models/head_patching.py`: Head patching実験（multi-token、複数patch_mode対応） ✅
  - `src/analysis/head_screening.py`: Headスクリーニング ✅
  - `src/visualization/head_plots.py`: Head解析結果可視化 ✅
  - `src/analysis/real_world_patching.py`: 実世界プロンプト評価 ✅
  - `src/visualization/real_world_plots.py`: 実世界Patching結果可視化 ✅
  - `src/models/neuron_ablation.py`: ニューロン単位アブレーション実験 ✅
  - `src/analysis/neuron_saliency.py`: ニューロンサリエンシー解析 ✅
  - `src/analysis/cross_model_patching.py`: クロスモデルパッチング（アライメント写像使用） ✅
  - `src/analysis/circuit_ov_qk.py`: OV/QK回路解析ユーティリティ（TransformerLens新バックエンド対応） ✅
  - `src/analysis/circuit_experiments.py`: OV/QK回路実験パイプライン（OV ablation、QK routing patching、統合実験） ✅
  - `src/analysis/circuit_report.py`: OV/QK回路解析結果の可視化とサマリー生成 ✅
  - `scripts/consistency_check.py`: コード-論文整合性チェック ✅
  - `.github/workflows/code_paper_consistency.yml`: GitHub Actions CI/CD ✅
  - `src/analysis/statistics/data_loading.py`: Phase 7.5 統計データローダー ✅
  - `src/analysis/statistics/effect_sizes.py`: 効果量・補正パイプライン ✅
  - `src/analysis/statistics/power_analysis.py`: 検出力分析 ✅
  - `src/analysis/statistics/k_selection.py`: k選択統計検証 ✅
  - `src/analysis/run_statistics.py`: Phase 7.5 統合CLI ✅
  - `src/models/phase8_large/registry.py`: Phase 8 中規模モデルレジストリ ✅
  - `src/analysis/run_phase8_pipeline.py`: Phase 8 自動化パイプライン ✅
  - `src/analysis/summarize_phase8_large.py`: Phase 8 結果サマリー生成 ✅
- **モジュール強化**:
  - `src/models/activation_patching.py`: Multi-token生成、αスケジュール、ウィンドウ/位置指定対応 ✅
  - `src/models/activation_patching_sweep.py`: Transformerベース評価、ネストメトリクス対応 ✅
  - `src/analysis/sentiment_eval.py`: CardiffNLP sentiment、Stanford Politeness、GoEmotions統合 ✅
  - `src/visualization/patching_heatmaps.py`: ネストメトリクス対応、ヒートマップ/バイオリン生成 ✅
  - `src/utils/mlflow_utils.py`: ネストメトリクスログ（`log_nested_metrics`）、タグ設定（`set_run_tags`）追加 ✅
  - `src/models/head_patching.py`: `generate_with_patching`に`temperature`/`top_p`パラメータ追加 ✅
- **実世界検証**: 実世界テキストサンプルでの検証追加・強化 ✅
- **ログ記録改善**: Phase 2で各実行のstdout/stderrをMLflowアーティファクトとして記録 ✅

### データセット
- **Baseline**: 280サンプル（70文 × 4感情）
- **Extended**: 400サンプル（100文 × 4感情）
- **実世界**: 35サンプル（SNS、レビュー、メール）

### 主要な発見（統合実行版）
1. **Token-basedベクトルの有効性**: 感情語トークンベースのベクトルがより明確な感情表現を抽出
2. **サブスペースレベルでの共通性**: モデル間overlap 0.13-0.15（ランダムより高い）
3. **線形写像による大幅改善**: Neutral空間での線形写像によりoverlapが0.001から0.99に改善
4. **Extendedデータセットでの検証**: 拡張データセットでも同様のパターンが確認され、発見の頑健性が示された
5. **Layer 1の重要性**: Head screeningにより、Layer 1 Head 10がgratitude感情に最も強く反応（Δattn: 0.340434） ✅
6. **Head patchingの効果**: Layer 0 Head 0のpatchingでsentimentが増加（+0.0149） ✅
7. **Multi-token生成の重要性**: 単一トークン生成では検出できないスタイル変化や感情パターンがmulti-token生成で検出可能 ✅
8. **Transformerベース評価の有効性**: ヒューリスティック指標では検出できなかった効果がTransformerベース評価で検出可能 ✅
9. **クロスモデルパッチングの可能性**: アライメント写像により、異なるモデル間で感情ベクトルを転送してパッチング可能 ✅
10. **ニューロンレベル解析の重要性**: HeadだけでなくMLPニューロンも感情回路に重要な役割を果たす可能性 ✅
11. **統計的厳密性の確認**: 効果量は小さい（d < 0.2）が一貫しており、k=2が最適サブスペース次元として統計的に検証済み ✅
12. **中規模モデルへのスケーリング**: アーキテクチャ依存性が決定的であり、Llama3は高いアライメント、Gemma3は低いアライメントを示す ✅

### レポート
- `docs/report/phase0_setup_report.md` - Phase 0: 環境セットアップ ✅
- `docs/report/phase1_data_report.md` - Phase 1: 統合データセット構築 ✅
- `docs/report/phase2_activations_report.md` - Phase 2: 統合活性抽出 ✅
- `docs/report/phase3_vectors_report.md` - Phase 3: 統合感情ベクトル ✅
- `docs/report/phase3.5_subspace_report.md` - Phase 3.5: 統合サブスペース解析 ✅
- `docs/report/phase4_patching_report.md` - Phase 4: 統合Patching + ランダム対照 + 実世界 ✅
- `docs/report/phase5_sweep_report.md` - Phase 5: 統合Sweep実験 ✅
- `docs/report/phase6_alignment_report.md` - Phase 6: 統合サブスペースアライメント ✅
- `docs/report/phase7_head_analysis_report.md` - Phase 7: Head/Unitレベル解析 ✅
- `docs/report/phase7.5_statistics_report.md` - Phase 7.5: 統計的厳密性検証 ✅
- `docs/report/phase8_llama3_scaling_report.md` - Phase 8: Llama3スケーリング解析 ✅
- `docs/report/phase8_comparative_scaling_report.md` - Phase 8: 比較スケーリング解析 ✅
