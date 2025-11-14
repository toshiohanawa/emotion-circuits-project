#!/bin/bash
# LaTeXパッケージインストールスクリプト（詳細版）

echo "=== LaTeXパッケージインストール ==="
echo ""

eval "$(/usr/libexec/path_helper)"

echo "1. tlmgrのバージョン確認..."
tlmgr --version

echo ""
echo "2. リポジトリを更新中..."
sudo tlmgr update --self

echo ""
echo "3. 必要なパッケージをインストール中..."
sudo tlmgr install multirow subcaption enumitem

echo ""
echo "4. インストール確認..."
tlmgr list --only-installed | grep -E "(multirow|subcaption|enumitem)"

echo ""
echo "5. ファイルの場所確認..."
kpsewhich multirow.sty
kpsewhich subcaption.sty
kpsewhich enumitem.sty

echo ""
echo "完了！"
