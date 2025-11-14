#!/bin/bash
# LaTeXコンパイルスクリプト

set -e

TEX_FILE="Emotion Circuits in Small LLMs_v1_20251115.tex"
PDF_FILE="Emotion Circuits in Small LLMs_v1_20251115.pdf"

echo "=== LaTeXコンパイルスクリプト ==="
echo ""

# ACLスタイルファイルのダウンロード（必要な場合）
if [ ! -f "acl.sty" ]; then
    echo "ACLスタイルファイルをダウンロード中..."
    curl -L -o acl.sty https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl.sty || {
        echo "警告: ACLスタイルファイルのダウンロードに失敗しました"
        echo "手動でダウンロードしてください: https://github.com/acl-org/acl-style-files"
    }
fi

# pdflatexの確認
if ! command -v pdflatex &> /dev/null; then
    echo "エラー: pdflatexが見つかりません"
    echo ""
    echo "インストール方法:"
    echo "  1. Homebrewを使用: brew install --cask basictex"
    echo "  2. インストール後、ターミナルを再起動するか、以下を実行:"
    echo "     eval \"\$(/usr/libexec/path_helper)\""
    echo ""
    echo "または、オンラインサービスを使用:"
    echo "  - Overleaf: https://www.overleaf.com/"
    echo "  - LaTeX Base: https://latexbase.com/"
    exit 1
fi

echo "pdflatexが見つかりました: $(which pdflatex)"
echo ""

# コンパイル（複数回実行して相互参照を解決）
echo "1回目のコンパイル..."
pdflatex -interaction=nonstopmode "$TEX_FILE" || {
    echo "コンパイルエラーが発生しました"
    exit 1
}

echo ""
echo "2回目のコンパイル（相互参照を解決）..."
pdflatex -interaction=nonstopmode "$TEX_FILE" || {
    echo "コンパイルエラーが発生しました"
    exit 1
}

# 補助ファイルをクリーンアップ（オプション）
# rm -f *.aux *.log *.out

if [ -f "$PDF_FILE" ]; then
    echo ""
    echo "✓ コンパイル成功: $PDF_FILE"
    echo ""
    echo "PDFを開きますか？ (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        open "$PDF_FILE"
    fi
else
    echo "エラー: PDFファイルが生成されませんでした"
    exit 1
fi
