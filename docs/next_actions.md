# 次のアクション（研究結果改善のための継続作業）

更新日: 2025-11-19

## 現在の状況

✅ **全テスト完了**: データセット再生成、Phase2-7実行、統計解析、包括的テストが全て正常に完了
✅ **検証基盤整備**: `scripts/verify_refactoring.py`が利用可能
✅ **タイミングログ**: 全Phaseで記録済み
✅ **統計結果**: 両プロファイルで生成完了

## 推奨される次のステップ

### 1. コード変更時の検証フロー確立

**目的**: リファクタリングや機能追加時に数値整合性を自動検証

**使用方法**:
```bash
# リファクタリング前の結果をベースラインとして保存
cp -r results results_baseline

# コード変更後、検証スクリプトを実行
python scripts/verify_refactoring.py \
  --before results_baseline \
  --after results \
  --profiles baseline baseline_smoke \
  --atol 1e-3 --rtol 1e-4
```

**期待される効果**:
- リファクタリング後の数値整合性を即座に確認
- 回帰テストの自動化
- レビュー時間の削減

### 2. Phase5/6のI/O改善（優先度: 中）

**現状**:
- Phase5: 18箇所のprint/logging文
- Phase6 (screening): 14箇所のprint/logging文
- Phase6 (patching): 17箇所のprint/logging文

**改善内容**:
1. **ループ内のprint/loggingをループ外に集約**
   - 進捗表示をバッファリングしてまとめて出力
   - エラーログを集約して最後に表示

2. **高頻度I/Oのバッファリング**
   - 評価結果の書き込みをバッチ化
   - 中間ファイルの書き込み頻度を削減

3. **ログの構造化**
   - タイミング情報を構造化ログとして出力
   - デバッグモードと通常モードの切り替え

**実装順序**:
1. Phase5（最重フェーズ、I/Oオーバーヘッドが大きい）
2. Phase6 head_screening
3. Phase6 head_patching

**期待される効果**:
- I/Oオーバーヘッドの削減（10-20%の高速化見込み）
- ログの可読性向上
- デバッグの容易化

### 3. 統計解析の拡張（優先度: 高）

**Phase7の追加モード実行**:
```bash
# powerモード: 検出力分析
python3 -m src.analysis.run_phase7_statistics \
  --profile baseline \
  --mode power \
  --phase-filter residual,head

# kモード: k選択分析
python3 -m src.analysis.run_phase7_statistics \
  --profile baseline \
  --mode k \
  --phase-filter residual,head
```

**期待される効果**:
- より詳細な統計的検証
- サンプルサイズの最適化
- 研究結果の信頼性向上

### 4. 追加実験（優先度: 中）

**ランダム対照実験**:
```bash
python3 -m src.analysis.run_phase5_residual_patching \
  --profile baseline \
  --model gpt2_small \
  --layers 0 3 6 9 11 \
  --random-control \
  --num-random 50 \
  --device cuda
```

**期待される効果**:
- 因果効果の特異性検証
- より厳密な統計的検証

### 5. 可視化とレポート生成（優先度: 中）

**各Phaseの詳細レポート生成**:
```bash
# Phase2-7のレポートを生成
for phase in phase2 phase3 phase4 phase5 phase6_head_screening phase6_head_patching phase7_statistics; do
  python3 -m src.reporting.generate_phase_report \
    --phase $phase \
    --profile baseline \
    --model gpt2_small \
    --output docs/report/${phase}_baseline_gpt2_small.md
done
```

**期待される効果**:
- 研究結果の可視化
- 再現性の向上
- 論文執筆の準備

## 実装時の注意事項

### 検証フローの確立
- **コード変更前**: `verify_refactoring.py`でベースライン結果を保存
- **コード変更後**: 必ず`verify_refactoring.py`を実行して数値整合性を確認
- **CI/CD**: 可能であれば自動化を検討

### I/O改善の実装
- **後方互換性**: 既存のCLI引数と出力フォーマットを維持
- **段階的実装**: 一度に全てを変更せず、Phaseごとに実装・検証
- **パフォーマンス測定**: 改善前後の実行時間を記録

### 統計解析の拡張
- **計算リソース**: power/kモードは計算時間が長くなる可能性がある
- **並列処理**: `--n-jobs`オプションを活用して高速化

## 完了基準

### Phase5/6のI/O改善
- [ ] Phase5のループ内print/loggingを集約
- [ ] Phase6 (screening)のループ内print/loggingを集約
- [ ] Phase6 (patching)のループ内print/loggingを集約
- [ ] 実行時間の改善を確認（10%以上の高速化）
- [ ] `verify_refactoring.py`で数値整合性を確認

### 統計解析の拡張
- [ ] Phase7 powerモードの実行
- [ ] Phase7 kモードの実行
- [ ] 結果の検証と解釈

### 追加実験
- [ ] ランダム対照実験の実行
- [ ] 結果の統計的検証

## 参考資料

- `docs/test_execution_summary.md`: テスト実行の詳細サマリー
- `docs/dev/IMPLEMENTATION_STATUS.md`: 実装状況の詳細
- `scripts/verify_refactoring.py`: 数値検証スクリプトの使用方法

