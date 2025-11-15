# Phase 0 — Setup & Environment

## 🎯 目的

- プロジェクト全体の環境構築
- TransformerLens（新バックエンド）を利用可能にする
- MLflow tracking / GPU / Python モジュールの初期設定

## 📦 生成物

- `docs/phase0_report.md`
- ログ: `results/setup/environment_info.json`
- 動作確認: "Hello Attention Hooks"

## 🚀 実行コマンド例

```bash
python -m scripts.check_environment
python -m scripts.test_hooks --model gpt2
```

## 📄 レポート項目

### 1. インストールバージョン一覧

#### Python
- バージョン: [X.X.X]
- 仮想環境: [.venv / conda / その他]

#### Torch
- バージョン: [X.X.X]
- CUDA対応: [Yes/No]
- CUDAバージョン: [X.X] (該当する場合)

#### TransformerLens
- バージョン: [X.X.X]
- 新バックエンド対応: [Yes/No]

#### MLflow
- バージョン: [X.X.X]
- Tracking URI: [http://localhost:5001 / file://...]

#### その他
- [依存パッケージ名]: [バージョン]

### 2. 環境の正常性チェック

#### CUDA / MPS 状態
- CUDA利用可能: [Yes/No]
- MPS利用可能: [Yes/No] (macOS)
- 使用デバイス: [cuda / cpu / mps]

#### dtype = float32 / bfloat16 の確認
- デフォルトdtype: [float32 / bfloat16]
- 動作確認: [Pass/Fail]

#### Hook が正しく動作するか
- Hook登録: [Pass/Fail]
- Hook実行: [Pass/Fail]
- use_attn_result=True動作確認: [Pass/Fail]

### 3. セットアップ時の注意点

#### use_attn_result=True の設定方法
- 設定方法: [model.cfg.use_attn_result = True を設定]
- 確認方法: [model.cfg.use_attn_result を確認]

#### 注意モジュールへの hook 設置成功の確認
- hook_pattern: [Pass/Fail]
- hook_q: [Pass/Fail]
- hook_k: [Pass/Fail]
- hook_v: [Pass/Fail]
- hook_result: [Pass/Fail] (use_attn_result=True時)

### 4. トラブルシューティング

#### 発生した問題
- [問題の説明]

#### 解決方法
- [解決手順]

### 5. 次のフェーズへの準備

- [ ] データセット準備完了
- [ ] モデルロード確認完了
- [ ] Hook動作確認完了
- [ ] MLflow接続確認完了

## 📝 備考

[その他の注意事項やメモ]

