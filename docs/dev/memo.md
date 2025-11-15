# プロジェクト続行メモ

## 現在の状況（2024年11月14日時点）

### 実装完了状況

- ✅ **Phase 0-7**: 完全に実装完了
  - Phase 0: 環境構築
  - Phase 1: 感情プロンプトデータ作成
  - Phase 2: 内部活性の抽出
  - Phase 3: 感情方向ベクトルの抽出・可視化
  - Phase 3.5: 感情語トークンベースの再検証
  - Phase 4 Light: 簡易Activation Patching
  - Phase 5: 層×αスイープ実験と因果力比較
  - Phase 6: サブスペース構造とアライメント（線形写像学習、k-sweep、Procrustesアライメント）
  - Phase 7: Head/Unitレベル解析（head_screening.py, head_ablation.py, head_patching.py, head_plots.py）

- 🔄 **Phase 8**: 部分的に実装済み
  - ✅ `src/utils/hf_hooks.py`: HuggingFaceモデル用フック（実装済み）
  - ❌ 既存モジュールのマルチモデル対応拡張（未実施）
  - ❌ モデルサイズとサブスペース共通性の解析実験（未実施）

### 実装済みモジュール数

- **Pythonモジュール**: 31ファイル
- **ドキュメント**: 8レポート
- **データファイル**: 2プロンプトファイル（neutral, gratitude）

## 次にすべきこと（優先順位順）

### 1. Phase 7の実験実行（最優先）

Phase 7のモジュールは実装済みですが、まだ実験を実行していません。まずはGPT-2 smallで実験を実行して動作確認を行います。

#### 1-1. Headスクリーニング実験

```bash
# GPT-2 smallでheadスクリーニングを実行
python -m src.analysis.head_screening \
  --model gpt2 \
  --device cuda \
  --output results/baseline/alignment/head_scores_gpt2.json
```

**必要なデータ**:
- `data/gratitude_prompts.json` ✅ 存在
- `data/anger_prompts.json` ❌ 作成が必要
- `data/apology_prompts.json` ❌ 作成が必要
- `data/neutral_prompts.json` ✅ 存在

**次のステップ**: 不足しているプロンプトデータを作成するか、既存のデータで実行可能な範囲で実験を開始

#### 1-2. Head ablation実験

```bash
# Headスクリーニング結果から「怪しいhead」を特定後、ablation実験を実行
python -m src.models.head_ablation \
  --model gpt2 \
  --device cuda \
  --head-spec "3:5,7:2" \
  --prompts-file data/gratitude_prompts.json \
  --output results/baseline/patching/head_ablation_gpt2_gratitude.pkl
```

#### 1-3. Head patching実験

```bash
python -m src.models.head_patching \
  --model gpt2 \
  --device cuda \
  --head-spec "3:5,7:2" \
  --neutral-prompts data/neutral_prompts.json \
  --emotion-prompts data/gratitude_prompts.json \
  --output results/baseline/patching/head_patching_gpt2_gratitude.pkl
```

#### 1-4. Head解析結果の可視化

```bash
python -m src.visualization.head_plots \
  --head-scores results/baseline/alignment/head_scores_gpt2.json \
  --ablation-file results/baseline/patching/head_ablation_gpt2_gratitude.pkl \
  --patching-file results/baseline/patching/head_patching_gpt2_gratitude.pkl \
  --output-dir results/baseline/plots/heads
```

### 2. 不足しているデータファイルの作成

#### 2-1. 感情プロンプトデータの作成

```bash
# 怒りプロンプトと謝罪プロンプトを作成
python -m src.data.create_emotion_dataset \
  --output data/emotion_dataset.jsonl \
  --min-samples-per-category 50
```

または、手動でJSONファイルを作成：
- `data/anger_prompts.json`
- `data/apology_prompts.json`

### 3. Phase 8の既存モジュールのマルチモデル対応拡張

Phase 8の実験を実行するため、以下のモジュールをマルチモデル対応に拡張する必要があります。

#### 3-1. 対象モジュール

- `src/analysis/emotion_subspace.py`
- `src/analysis/subspace_k_sweep.py`
- `src/analysis/model_alignment.py`
- `src/analysis/subspace_alignment.py`

#### 3-2. 拡張内容

1. **引数の追加**: `--model-a`, `--model-b`などの引数を追加
2. **モデルロード関数の統一**: TransformerLensモデルとHuggingFaceモデルを切り替え可能に
3. **Residual取得の統一**: `hf_hooks.py`を使用してHuggingFaceモデルからresidualを取得

#### 3-3. 実装例

```python
# モデルロード関数の例
def load_model(model_name: str, device: str):
    """モデルをロード（TransformerLensまたはHuggingFace）"""
    if model_name in ["gpt2", "EleutherAI/pythia-160m", "EleutherAI/gpt-neo-125M"]:
        # TransformerLensでロード
        model = HookedTransformer.from_pretrained(model_name, device=device)
        return model, None  # tokenizerは不要
    else:
        # HuggingFaceでロード
        from src.utils.hf_hooks import load_hf_causal_lm
        model, tokenizer = load_hf_causal_lm(model_name, device=device)
        return model, tokenizer
```

### 4. Phase 8の実験実行

#### 4-1. 対象モデルペア

- GPT-2 small (124M) vs GPT-2 medium (355M)
- GPT-2 small (124M) vs GPT-2 large (774M)
- GPT-2 small (124M) vs Pythia-410M

#### 4-2. 実行する実験

```bash
# k-sweep実験（GPT-2 small vs GPT-2 medium）
python -m src.analysis.subspace_k_sweep \
  --model-a gpt2 \
  --model-b gpt2-medium \
  --layers 3 5 7 9 11 \
  --k-values 2 5 10 20 \
  --output results/baseline/alignment/k_sweep_gpt2_gpt2medium.json

# 線形写像アライメント（GPT-2 small vs GPT-2 medium）
python -m src.analysis.model_alignment \
  --model-a gpt2 \
  --model-b gpt2-medium \
  --neutral_prompts_file data/neutral_prompts.json \
  --model1_activations_dir results/baseline/activations/gpt2 \
  --model2_activations_dir results/baseline/activations/gpt2-medium \
  --output results/baseline/alignment/model_alignment_gpt2_gpt2medium.pkl \
  --n-components 10 \
  --layers 3 5 7 9 11
```

**注意**: より大きなモデル（GPT-2 medium/large）の活性データを先に抽出する必要があります。

### 5. 結果の統合とレポート作成

#### 5-1. Phase 7レポートの作成

- Headスクリーニング結果の分析
- Ablation/patching実験の定量結果
- 可視化結果の解釈

#### 5-2. Phase 8レポートの作成

- モデルサイズとサブスペース共通性の関係
- 線形写像アライメント効果のモデルサイズ依存性
- k-sweep結果のモデルサイズ依存性

#### 5-3. 最終レポートの更新

- `docs/final_report.md`を更新
- 全フェーズの結果を統合
- 研究クエスチョンへの回答をまとめる

## 実装上の注意点

### Phase 7の実装に関する注意

1. **Head ablation/patchingのhook実装**: TransformerLensの`hook_result`のshapeに注意
   - 実際のshapeは`[batch, pos, head, d_head]`または`[batch, head, pos, d_head]`の可能性
   - モデルによって異なるため、実際の実行時に確認が必要

2. **感情語トークン位置の特定**: `head_screening.py`では既存の`emotion_vectors_token_based.py`のロジックを流用

3. **評価指標**: 既存の`sentiment_eval.py`の関数を再利用

### Phase 8の実装に関する注意

1. **モデルロードの統一**: TransformerLensとHuggingFaceの両方に対応
2. **Residual取得の統一**: `hf_hooks.py`の`capture_residuals`を使用
3. **既存コードの互換性**: GPT-2 small用の既存ロジックは維持

## 推奨される実行順序

### 短期（1-2週間）

1. **不足データの作成**: `data/anger_prompts.json`, `data/apology_prompts.json`
2. **Phase 7の実験実行**: GPT-2 smallでheadスクリーニング、ablation、patching
3. **Phase 7レポート作成**: 結果の分析と可視化

### 中期（2-3週間）

4. **Phase 8のモジュール拡張**: 既存モジュールのマルチモデル対応
5. **より大きなモデルの活性抽出**: GPT-2 medium/large, Pythia-410M
6. **Phase 8の実験実行**: モデルサイズとサブスペース共通性の解析

### 長期（1-2週間）

7. **Phase 8レポート作成**: モデルサイズ依存性の分析
8. **最終レポート更新**: 全フェーズの統合
9. **コードの整理とドキュメント化**: GitHub公開準備

## 技術的な課題と解決策

### 課題1: Head ablation/patchingのhook実装

**問題**: TransformerLensの`hook_result`のshapeがモデルによって異なる可能性

**解決策**: 
- 実際の実行時にshapeを確認
- 条件分岐で両方のパターンに対応
- エラーメッセージを詳細にしてデバッグしやすくする

### 課題2: より大きなモデルのメモリ使用量

**問題**: GPT-2 medium/largeはGPUメモリを多く消費

**解決策**:
- バッチサイズを小さくする
- CPUモードで実行（遅いが可能）
- モデルの一部のみをロード（可能な場合）

### 課題3: HuggingFaceモデルのhook実装

**問題**: モデルアーキテクチャによってhookの登録方法が異なる

**解決策**:
- `hf_hooks.py`で主要なアーキテクチャ（GPT-2, Pythia, Llama）に対応
- エラーハンドリングを追加
- モデル構造を自動検出する機能を追加

## チェックリスト

### Phase 7の実験実行

- [ ] `data/anger_prompts.json`を作成
- [ ] `data/apology_prompts.json`を作成
- [ ] Headスクリーニング実験を実行（GPT-2 small）
- [ ] Head ablation実験を実行（GPT-2 small）
- [ ] Head patching実験を実行（GPT-2 small）
- [ ] Head解析結果を可視化
- [ ] Phase 7レポートを作成

### Phase 8の実装と実験

- [ ] `emotion_subspace.py`をマルチモデル対応に拡張
- [ ] `subspace_k_sweep.py`をマルチモデル対応に拡張
- [ ] `model_alignment.py`をマルチモデル対応に拡張
- [ ] `subspace_alignment.py`をマルチモデル対応に拡張
- [ ] GPT-2 mediumの活性データを抽出
- [ ] GPT-2 largeの活性データを抽出（オプション）
- [ ] Pythia-410Mの活性データを抽出（オプション）
- [ ] Phase 8の実験を実行（GPT-2 small vs GPT-2 medium）
- [ ] Phase 8レポートを作成

### 最終整理

- [ ] 全フェーズのレポートを統合
- [ ] `docs/final_report.md`を更新
- [ ] コードの整理とリファクタリング
- [ ] READMEの最終確認
- [ ] GitHub公開準備

## 参考資料

- **実装計画**: `docs/implementation_plan.md`
- **Phase 6レポート**: `docs/phase6_expansion_report.md`
- **Phase 5レポート**: `docs/phase5_report.md`
- **Phase 3.5/4 Lightレポート**: `docs/phase3.5_and_4light_report.md`

## メモ

- Phase 7の実装は完了しているが、実験実行前に不足データの作成が必要
- Phase 8の`hf_hooks.py`は実装済みだが、既存モジュールとの統合が必要
- より大きなモデルでの実験は、GPUメモリと時間を考慮して計画する必要がある
- すべての実験結果を統合して、研究クエスチョンへの回答をまとめることが最終目標
