# LaTeX論文コンパイルガイド

このディレクトリには、ACL形式のLaTeX論文が含まれています。

## ファイル

- `Emotion Circuits in Small LLMs_v1_20251115.tex` - メインのLaTeXファイル
- `compile.sh` - コンパイルスクリプト

## コンパイル方法

### 方法1: ローカルでコンパイル（推奨）

1. **BasicTeXをインストール**:
   ```bash
   brew install --cask basictex
   ```

2. **ターミナルを再起動**するか、以下を実行:
   ```bash
   eval "$(/usr/libexec/path_helper)"
   ```

3. **ACLスタイルファイルをダウンロード**（自動）:
   ```bash
   curl -L -o acl.sty https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl.sty
   ```

4. **コンパイルスクリプトを実行**:
   ```bash
   ./compile.sh
   ```

   または手動で:
   ```bash
   pdflatex "Emotion Circuits in Small LLMs_v1_20251115.tex"
   pdflatex "Emotion Circuits in Small LLMs_v1_20251115.tex"
   ```

### 方法2: オンラインサービスを使用

以下のオンラインLaTeXエディタを使用できます:

1. **Overleaf** (https://www.overleaf.com/)
   - アカウント作成（無料）
   - 新規プロジェクトを作成
   - `.tex`ファイルをアップロード
   - ACLテンプレートを使用する場合は、ACLスタイルファイルも必要

2. **LaTeX Base** (https://latexbase.com/)
   - ブラウザで直接コンパイル可能

## ACLスタイルファイル

ACLスタイルファイル（`acl.sty`）は以下のリポジトリから取得できます:
- https://github.com/acl-org/acl-style-files

`compile.sh`スクリプトは自動的にダウンロードを試みます。

## トラブルシューティング

### pdflatexが見つからない

```bash
# パスを確認
which pdflatex

# BasicTeXがインストールされている場合、パスを更新
eval "$(/usr/libexec/path_helper)"
```

### ACLスタイルファイルが見つからない

手動でダウンロード:
```bash
curl -L -o acl.sty https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl.sty
```

### コンパイルエラー

- LaTeXログファイル（`.log`）を確認
- 必要なパッケージがインストールされているか確認
- ACLスタイルファイルが正しいバージョンか確認

## プレビュー

PDFが生成されたら、以下のコマンドで開けます:

```bash
open "Emotion Circuits in Small LLMs_v1_20251115.pdf"
```

