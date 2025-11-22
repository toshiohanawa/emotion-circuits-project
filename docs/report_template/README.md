# レポートテンプレート

各フェーズの実験レポートを作成するためのテンプレート集です。

## 📁 テンプレート一覧（現行パイプライン）

- `phase0_setup_template.md` - Phase 0: 環境構築
- `phase1_dataset_template.md` - Phase 1: データセット構築
- `phase2_activations_template.md` - Phase 2: 活性抽出（run_phase2_activations）
- `phase3_vectors_template.md` - Phase 3: 感情ベクトル・サブスペース構築（run_phase3_vectors）
- `phase4_alignment_template.md` - Phase 4: サブスペースアライメント（run_phase4_alignment）
- `phase5_residual_patching_template.md` - Phase 5: 残差パッチング＋ランダム対照（run_phase5_residual_patching）
- `phase6_head_template.md` - Phase 6: ヘッドパッチング/スクリーニング（run_phase6_head_patching, run_phase6_head_screening）
- `phase7_statistics_template.md` - Phase 7: 統計集約（run_phase7_statistics）

## 🚀 使用方法

1. 該当するフェーズのテンプレートをコピー
2. `docs/report/`ディレクトリに配置（例: `docs/report/phase0_setup_report.md`）
3. テンプレート内の`[値]`や`[説明]`を実際の結果で置き換え
4. 必要に応じてセクションを追加・削除

### ⏱ 自動サマリ付きレポート生成
タイミングログ（`results/<profile>/timing/phase_timings.jsonl`）が生成済みなら、以下のCLIでテンプレート＋実行サマリ付きのレポートを自動生成できる。

```bash
python -m src.reporting.generate_phase_report \
  --phase phase3 \
  --profile baseline_smoke \
  --model gpt2_small \
  --output docs/report/phase3_baseline_smoke_gpt2_small.md
```

`--timing-run-id` を指定すると特定ランの記録を埋め込める。テンプレート本文は手動で加筆し、末尾の「自動生成サマリ」でLLM査読者に向けた実験メタデータを共有できる。

## 📝 テンプレートの構造

各テンプレートは以下の構造を持ちます：

- **🎯 目的**: フェーズの目的
- **📦 生成物**: このフェーズで生成されるファイル
- **🚀 実行コマンド例**: 実際の実行コマンド
- **📄 レポート項目**: 記録すべき項目のリスト
- **📝 備考**: その他の注意事項

## 💡 テンプレートのカスタマイズ

テンプレートは必要に応じてカスタマイズしてください：

- プロジェクト固有の項目を追加
- 不要なセクションを削除
- 表の形式を変更
- 図表の参照を追加

## 📚 関連ドキュメント

- `docs/implementation_plan.md` - 実装計画の詳細
- `docs/report/` - 実際のレポート例
