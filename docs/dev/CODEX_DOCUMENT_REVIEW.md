# Codex Instruction Document レビュー結果

## 実態との不一致点

### 1. Phase定義の不一致

#### Phase 3.5 → Phase 4
- **ドキュメント**: Phase 3.5「サブスペース解析 & アライメント」
- **実態**: `run_phase4_alignment.py` として Phase 4 に実装
- **修正**: masterplan.md に合わせて Phase 4 として統一

#### Phase 7.5 → Phase 7
- **ドキュメント**: Phase 7.5「統計解析」
- **実態**: `run_phase7_statistics.py` として Phase 7 に実装
- **修正**: masterplan.md に合わせて Phase 7 として統一

#### Phase 7（Head-level Patching）
- **ドキュメント**: Phase 7「Head-level Patching（pattern_v / v_only）」
- **実態**: 未実装（Phase 6 で ablation のみ実装済み）
- **修正**: Phase 7 は統計解析として記載し、Head-level Patching は未実装として明記

---

### 2. ファイルパスの不一致

#### 存在しないファイルへの参照

| ドキュメント記載 | 実態 |
|---|---|
| `src/analysis/run_statistics.py` | `src/analysis/run_phase7_statistics.py` |
| `src/analysis/activations.py` | `src/analysis/run_phase2_activations.py` |
| `src/analysis/emotion_vectors.py` | `src/analysis/run_phase3_vectors.py` |
| `src/analysis/emotion_vectors_token_based.py` | `src/analysis/run_phase3_vectors.py`（統合） |
| `src/analysis/emotion_subspace.py` | `src/analysis/run_phase3_vectors.py`（統合） |
| `src/analysis/cross_model_subspace.py` | 存在しない |
| `src/analysis/model_alignment.py` | 存在しない |
| `src/analysis/subspace_alignment.py` | `src/analysis/run_phase4_alignment.py` |
| `src/analysis/head_screening.py` | `src/analysis/run_phase6_head_screening.py` |
| `src/models/head_ablation.py` | 存在しない（Phase 6 スクリプトに統合） |
| `src/models/head_patching.py` | 存在しない（未実装） |
| `src/models/activation_patching_sweep.py` | 存在しない（`activation_patching.py` に統合） |
| `src/analysis/run_phase8_pipeline.py` | 存在しない（未実装） |
| `src/analysis/summarize_phase8_large.py` | 存在しない（未実装） |

#### 正しいファイル構造

**Phase 2**: `src/analysis/run_phase2_activations.py`
**Phase 3**: `src/analysis/run_phase3_vectors.py`（感情ベクトルとサブスペースを統合）
**Phase 4**: `src/analysis/run_phase4_alignment.py`
**Phase 5**: `src/analysis/run_phase5_residual_patching.py`
**Phase 6**: 
- `src/analysis/run_phase6_head_screening.py`（全headスクリーニング）
- `src/analysis/run_phase6_head_patching.py`（head ablation）
**Phase 7**: `src/analysis/run_phase7_statistics.py`
**Phase 8**: 未実装（`src/models/phase8_large/` にモジュールのみ存在）

---

### 3. 実装済み最適化の未反映

#### ✅ 実装済み（ドキュメントに未記載）

1. **デバイスユーティリティ**
   - ✅ `src/utils/device.py` に `get_default_device()`, `get_default_device_str()`, `move_to_device()` を実装済み

2. **TextEvaluator**
   - ✅ `src/analysis/evaluation.py` に `TextEvaluator` クラスを実装済み
   - ✅ `evaluate_batch()` メソッドでバッチ評価を実装済み

3. **Phase 2 のバッチ化**
   - ✅ `activation_api.py` の `_capture_small()` でバッチ処理を実装済み
   - ✅ `--batch-size` CLI引数を追加済み（デフォルト: 16）

4. **Phase 3 のtorch化**
   - ✅ `run_phase3_vectors.py` で `_pca_torch()` を実装済み（GPU/MPS対応）

5. **Phase 4 のtorch化**
   - ✅ `run_phase4_alignment.py` で `_procrustes_torch()`, `_subspace_overlap_torch()` を実装済み
   - ✅ `--use-torch` / `--no-use-torch` CLI引数を追加済み

6. **Phase 5 のバッチ化**
   - ✅ `activation_patching.py` に `generate_with_patching_batch()` を実装済み
   - ✅ `TextEvaluator.evaluate_batch()` を使用
   - ✅ `--batch-size` CLI引数を追加済み（デフォルト: 8）

7. **Phase 6 のバッチ化**
   - ✅ `run_phase6_head_patching.py` と `run_phase6_head_screening.py` でバッチ処理を実装済み
   - ✅ `--batch-size` CLI引数を追加済み（デフォルト: 8）

8. **Phase 7 の並列化**
   - ✅ `effect_sizes.py` で bootstrap 計算を並列化済み（joblib使用）
   - ✅ `k_selection.py` でも並列化済み
   - ✅ `--n-jobs` CLI引数を追加済み（デフォルト: 1）

---

### 4. Phase定義の修正が必要な箇所

#### Phase 3.5 → Phase 4
```markdown
# 修正前
- **Phase 3.5**: サブスペース解析 & アライメント（PCA, Procrustes, 線形写像）
- **Phase 4**: Simple Activation Patching（単層・単α）

# 修正後
- **Phase 3**: 感情ベクトル構築（sentence-end, token-based）+ サブスペース構築（PCA）
- **Phase 4**: モデル間アライメント（Procrustes, 線形写像）
```

#### Phase 7.5 → Phase 7
```markdown
# 修正前
- **Phase 7**: Head-level Patching（pattern_v / v_only）
- **Phase 7.5**: 統計解析（effect size, power, k-selection）

# 修正後
- **Phase 7**: 統計的厳密性（effect size, power, k-selection）
- **Phase 7.5**: （未実装）Head-level Patching（pattern_v / v_only）
```

---

### 5. ファイルパス参照の修正が必要な箇所

#### セクション 1（リポジトリ概要）
```markdown
# 修正前
- `src/analysis/run_statistics.py` : Phase 7.5 統計パイプライン

# 修正後
- `src/analysis/run_phase7_statistics.py` : Phase 7 統計パイプライン
- `src/analysis/statistics/` : Phase 7 統計計算モジュール群
```

#### セクション 4.2（Phase 2）
```markdown
# 修正前
対象と思われるコード例:
src/analysis/activations.py, src/analysis/run_phase2_activations.py のようなスクリプト群。

# 修正後
対象:
src/analysis/run_phase2_activations.py
```

#### セクション 4.3（Phase 3）
```markdown
# 修正前
対象:
src/analysis/emotion_vectors.py, src/analysis/emotion_vectors_token_based.py 等。

# 修正後
対象:
src/analysis/run_phase3_vectors.py（感情ベクトルとサブスペース構築を統合）
```

#### セクション 4.4（Phase 3.5 → Phase 4）
```markdown
# 修正前
対象:
src/analysis/emotion_subspace.py, src/analysis/cross_model_subspace.py,
src/analysis/model_alignment.py, src/analysis/subspace_alignment.py 等。

# 修正後
対象:
src/analysis/run_phase4_alignment.py
```

#### セクション 4.6（Phase 5）
```markdown
# 修正前
対象:
src/analysis/run_phase5_residual_patching.py
src/models/activation_patching_sweep.py
src/analysis/evaluation.py など。

# 修正後
対象:
src/analysis/run_phase5_residual_patching.py
src/models/activation_patching.py（sweep機能を含む）
src/analysis/evaluation.py
```

#### セクション 4.7（Phase 6）
```markdown
# 修正前
対象:
src/analysis/head_screening.py
src/models/head_ablation.py
src/visualization/head_plots.py 等。

# 修正後
対象:
src/analysis/run_phase6_head_screening.py（全headスクリーニング）
src/analysis/run_phase6_head_patching.py（head ablation）
src/visualization/head_plots.py 等。
```

#### セクション 4.8（Phase 7）
```markdown
# 修正前
対象:
src/models/head_patching.py など。

主な処理:
特定の head (e.g. Layer 1 Head 10) の pattern_v / v_only で causal patching。

# 修正後
対象:
未実装（将来の拡張として検討）

注意:
- Phase 6 で head ablation は実装済み
- pattern_v / v_only での head-level patching は未実装
```

#### セクション 4.9（Phase 7.5 → Phase 7）
```markdown
# 修正前
対象:
src/analysis/run_statistics.py など。

# 修正後
対象:
src/analysis/run_phase7_statistics.py
src/analysis/statistics/effect_sizes.py（bootstrap並列化済み）
src/analysis/statistics/power_analysis.py
src/analysis/statistics/k_selection.py（bootstrap並列化済み）
```

#### セクション 4.10（Phase 8）
```markdown
# 修正前
対象:
src/analysis/run_phase8_pipeline.py
src/analysis/summarize_phase8_large.py
および各モデル用の alignment ロジック。

# 修正後
対象:
未実装（CLIスクリプトは存在しない）

既存モジュール:
- src/models/phase8_large/registry.py（モデル定義）
- src/models/phase8_large/hf_wrapper.py（LargeHFModelラッパー）

注意:
- Phase 4 のアライメント計算ロジックを再利用可能
- Phase 4 のtorchベース化が完了しているため、Phase 8 実装時の一貫性が確保されている
```

---

### 6. 実装済み最適化の追記が必要な箇所

#### セクション 3.1（デバイスユーティリティ）
```markdown
# 追加
実装状況:
- ✅ `src/utils/device.py` に実装済み
- ✅ `get_default_device()`: torch.device を返す
- ✅ `get_default_device_str()`: 文字列を返す
- ✅ `move_to_device()`: 再帰的にデバイス移動
```

#### セクション 3.2（評価器ユーティリティ）
```markdown
# 追加
実装状況:
- ✅ `src/analysis/evaluation.py` に `TextEvaluator` クラスを実装済み
- ✅ `evaluate_batch()` メソッドでバッチ評価を実装済み
- ✅ Phase 5, 6 で使用中
```

#### セクション 4.2（Phase 2）
```markdown
# 追加
実装状況:
- ✅ `activation_api.py` の `_capture_small()` でバッチ処理を実装済み
- ✅ `--batch-size` CLI引数を追加済み（デフォルト: 16）
- ✅ モデルロードは1回のみ
```

#### セクション 4.3（Phase 3）
```markdown
# 追加
実装状況:
- ✅ `_pca_torch()` を実装済み（GPU/MPS対応）
- ✅ `--use-torch` CLI引数を追加済み（デフォルト: True）
```

#### セクション 4.4（Phase 4）
```markdown
# 追加
実装状況:
- ✅ `_procrustes_torch()`, `_subspace_overlap_torch()` を実装済み
- ✅ `--use-torch` / `--no-use-torch` CLI引数を追加済み
- ✅ `--device` CLI引数でデバイス指定可能
```

#### セクション 4.6（Phase 5）
```markdown
# 追加
実装状況:
- ✅ `generate_with_patching_batch()` を実装済み
- ✅ `TextEvaluator.evaluate_batch()` を使用
- ✅ `--batch-size` CLI引数を追加済み（デフォルト: 8）
- ✅ 評価器の初期化はループ外で1回のみ
```

#### セクション 4.7（Phase 6）
```markdown
# 追加
実装状況:
- ✅ バッチ処理を実装済み
- ✅ `--batch-size` CLI引数を追加済み（デフォルト: 8）
- ✅ `TextEvaluator.evaluate_batch()` を使用
```

#### セクション 4.9（Phase 7）
```markdown
# 追加
実装状況:
- ✅ bootstrap 計算を並列化済み（joblib使用）
- ✅ `--n-jobs` CLI引数を追加済み（デフォルト: 1）
- ✅ `effect_sizes.py` と `k_selection.py` の両方で並列化済み
```

---

### 7. 実装ロードマップの更新

#### セクション 7（実装ロードマップ）
```markdown
# 修正前
共通ユーティリティ整備
- get_default_device() などのデバイスユーティリティ
- TextEvaluator のようなバッチ対応評価器

Phase 5 の完全バッチ化
- 生成のバッチ化
- 評価のバッチ化
- モデル・評価器の単一ロード

Phase 4 / 7（patching 系）のバッチ化
- 共通 patching ロジックで再利用

Phase 2（activations）のバッチ化
- hooks の最適化

Phase 6（head screening/ablation）の行列化

Phase 7.5（statistics）のベクトル化・並列化

Phase 8（large model alignment）のバッチ化・安定化

# 修正後
✅ 共通ユーティリティ整備（完了）
- ✅ get_default_device() などのデバイスユーティリティ
- ✅ TextEvaluator のようなバッチ対応評価器

✅ Phase 5 の完全バッチ化（完了）
- ✅ 生成のバッチ化
- ✅ 評価のバッチ化
- ✅ モデル・評価器の単一ロード
- ✅ バッチサイズの設定可能化

✅ Phase 4 のtorchベース化（完了）
- ✅ Procrustes計算のtorch化
- ✅ サブスペースoverlap計算のtorch化

✅ Phase 2（activations）のバッチ化（完了）
- ✅ hooks の最適化
- ✅ バッチサイズの設定可能化

✅ Phase 6（head screening/ablation）のバッチ化（完了）
- ✅ バッチサイズの設定可能化

✅ Phase 7（statistics）の並列化（完了）
- ✅ bootstrap計算の並列化（joblib）

🔄 Phase 8（large model alignment）のパイプライン実装（未実装）
- Phase 4 のtorchベース化が完了しているため、実装時の一貫性が確保されている

📝 Phase 7.5（Head-level Patching）の実装（未実装）
- pattern_v / v_only での head-level patching
- Phase 6 の拡張として実装可能
```

---

## 推奨される修正

1. **Phase定義の統一**: masterplan.md に合わせて Phase 3.5 → Phase 4、Phase 7.5 → Phase 7 に修正
2. **ファイルパスの更新**: 存在しないファイルへの参照を削除し、正しいファイルパスに更新
3. **実装状況の追記**: 実装済みの最適化を各セクションに追記
4. **未実装機能の明記**: Phase 7.5（Head-level Patching）と Phase 8 パイプラインが未実装であることを明記

