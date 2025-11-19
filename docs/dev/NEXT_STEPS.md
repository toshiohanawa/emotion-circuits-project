# 次のステップ（残課題の優先順位）

## 完了した作業

✅ Phaseマッピングの統一
- Phase 6, 7 のコメントとdocstringを masterplan.md に基づいて統一完了
- すべてのスクリプトが正しいPhase名で統一された

---

## 残課題の優先順位

### 🔴 高優先度

#### 1. Phase 4 の torch ベース化（GPU/MPS加速） ✅ **完了**
**対象**: `src/analysis/run_phase4_alignment.py`

**実装内容**:
- ✅ `_procrustes_torch()` を実装（`torch.linalg.lstsq` 使用、MPS時は `torch.linalg.pinv` でGPU/MPSを維持）
- ✅ `_subspace_overlap_torch()` を実装（`torch.linalg.qr` 使用）
- ✅ `_orthonormalize_torch()` を実装
- ✅ `get_default_device()` を使用してデバイス管理を統一
- ✅ `--device` と `--use-torch` / `--no-use-torch` CLI引数を追加
- ✅ 後方互換性のため numpy ベースの実装も保持

**期待される効果**:
- GPU/MPS 環境で 2-3倍の高速化
- 大規模データセットでの計算時間短縮
- Phase 8 実装時の一貫性向上

**使用方法**:
```bash
# torch ベース（デフォルト、GPU/MPS加速）
python -m src.analysis.run_phase4_alignment --model-a gpt2 --model-b llama3_8b

# 明示的にデバイス指定
python -m src.analysis.run_phase4_alignment --model-a gpt2 --model-b llama3_8b --device mps

# numpy ベース（後方互換性）
python -m src.analysis.run_phase4_alignment --model-a gpt2 --model-b llama3_8b --no-use-torch
```

---

#### 2. Phase 7 統計解析の並列化（bootstrap計算） ✅ **完了**
**対象**: `src/analysis/statistics/effect_sizes.py`, `k_selection.py`

**実装内容**:
- ✅ `_bootstrap_ci()` と `_bootstrap_unpaired()` を並列化可能な形にリファクタリング
- ✅ `_bootstrap_sample_mean()`, `_bootstrap_sample_effect_size()`, `_bootstrap_unpaired_sample()` を追加
- ✅ `joblib.Parallel` を使用して並列化（joblib がない場合は逐次処理にフォールバック）
- ✅ `--n-jobs` CLI引数を追加（デフォルト: 1、環境依存を考慮）
- ✅ `summarize_k_selection()` の bootstrap 計算も並列化
- ✅ `EffectComputationConfig` に `n_jobs` パラメータを追加
- ✅ `pyproject.toml` に `joblib>=1.3.0` を追加

**期待される効果**:
- 4-8コア環境で 3-5倍の高速化
- 大規模データセットでの統計計算時間短縮
- `n_bootstrap=2000` のデフォルトで、並列化により大幅な時間短縮

**使用方法**:
```bash
# 逐次処理（デフォルト）
python -m src.analysis.run_phase7_statistics --profile baseline --mode all

# 並列処理（4コア使用）
python -m src.analysis.run_phase7_statistics --profile baseline --mode all --n-jobs 4

# 全CPU使用
python -m src.analysis.run_phase7_statistics --profile baseline --mode all --n-jobs -1
```

---

### 🟡 中優先度

#### 3. バッチサイズの設定可能化 ✅ **完了**
**対象**: 全Phase（特に Phase 2, 5, 6）

**実装内容**:
- ✅ `run_phase2_activations.py` に `--batch-size` 引数を追加（デフォルト: 16）
- ✅ `run_phase5_residual_patching.py` に `--batch-size` 引数を追加（デフォルト: 8）
- ✅ `run_phase6_head_patching.py` に `--batch-size` 引数を追加（デフォルト: 8）
- ✅ `run_phase6_head_screening.py` に `--batch-size` 引数を追加（デフォルト: 8）
- ✅ `activation_patching.py` の `_generate_text_batch()` に `batch_size` 引数を追加
- ✅ `activation_api.py` の `get_activations()` に `batch_size` 引数を追加
- ✅ すべての `evaluate_batch()` 呼び出しに `batch_size` 引数を追加

**影響範囲**:
- ユーザーが環境に応じて最適なバッチサイズを設定可能
- 大規模モデルでのメモリ管理が容易に
- メモリ制約に応じた調整が可能

**使用方法**:
```bash
# Phase 2（活性抽出）
python -m src.analysis.run_phase2_activations --model gpt2 --layers 0 6 --batch-size 32

# Phase 5（残差パッチング）
python -m src.analysis.run_phase5_residual_patching --model gpt2 --layers 0 6 --batch-size 16
  # ランダム対照はオプション（--random-control --num-random N）。標準ではオフ。

# Phase 6（Head Patching）小型（HookedTransformer）
python -m src.analysis.run_phase6_head_patching --model gpt2 --heads 0:0 --batch-size 4

# Phase 6（Head Patching）大モデル（LargeHFModel, 例: llama3_8b）
python -m src.analysis.run_phase6_head_patching --model llama3_8b --heads 0:0-11 3:0-11 --batch-size 4 --max-samples 50 --sequence-length 30 --device mps

# Phase 6（Head Screening）小型
python -m src.analysis.run_phase6_head_screening --model gpt2 --layers 0 1 --batch-size 4

# Phase 6（Head Screening）大モデル
python -m src.analysis.run_phase6_head_screening --model llama3_8b --layers 0 3 6 9 11 --batch-size 4 --max-samples 50 --sequence-length 30 --device mps
```

---

#### 4. Phase 8 パイプラインの実装
**対象**: 新規スクリプト `src/analysis/run_phase8_pipeline.py`

**現状**:
- `src/models/phase8_large/` にモジュール（registry, hf_wrapper）は存在
- CLIスクリプトが未実装
- `docs/process_flow_diagram.html` で言及されているが実体なし

**実装内容**:
- Phase 3/4 相当の処理を大規模モデル向けに実装
- `LargeHFModel` を使用した活性抽出（バッチ処理）
- Phase 4 のアライメント計算ロジックを再利用
- 結果を `results/<profile>/phase8/` に保存

**前提条件**:
- Phase 4 の torch ベース化が完了していること（一貫性のため）

---

### 🟢 低優先度

#### 5. pattern_v/v_only の実装（Phase 6 拡張）
**対象**: `src/analysis/run_phase6_head_patching.py` または新規スクリプト

**現状**:
- Codex Instruction Document で言及されているが未実装
- 現在は ablation（ゼロ化）のみ

**実装内容**:
- pattern_v モード: attention pattern を感情プロンプトから取得して注入
- v_only モード: value ベクトルのみを注入
- Phase 5 のバッチ生成ロジックを再利用

**優先度の理由**:
- 現状の ablation でも因果効果は測定可能
- 実装コストが高い（hook の複雑な操作が必要）

---

#### 6. パフォーマンスログの追加
**対象**: 全Phaseスクリプト

**内容**:
- 各処理ステップの実行時間を詳細に記録
- MLflow への自動記録（オプション）
- ボトルネックの可視化

---

#### 7. 数値検証スクリプト
**対象**: 新規スクリプト `scripts/verify_refactoring.py`

**内容**:
- Before/After の出力比較
- 許容範囲内の差分確認（1e-3〜1e-2）
- スモークテスト用の簡易スクリプト

---

## 推奨される実装順序

1. **Phase 4 の torch ベース化** ← 最も影響が大きく、実装が比較的簡単
2. **Phase 7 の並列化** ← 統計計算の高速化が重要
3. **バッチサイズの設定可能化** ← ユーザビリティ向上
4. **Phase 8 パイプライン実装** ← Phase 4 の torch 化が前提

---

## 注意事項

- **後方互換性**: すべての変更で既存のCLI引数と出力フォーマットを維持
- **テスト**: 各最適化後にスモークテストを実行して数値の整合性を確認
- **ドキュメント**: 新しい機能（--batch-size, --n-jobs など）の使用例を README に追加
