#!/bin/bash
# LaTeXパッケージインストールスクリプト

echo "必要なLaTeXパッケージをインストールします..."
echo "管理者パスワードの入力が必要です。"
echo ""

eval "$(/usr/libexec/path_helper)"

# 必要なパッケージをインストール
sudo tlmgr install multirow subcaption enumitem

echo ""
echo "インストール完了！"
