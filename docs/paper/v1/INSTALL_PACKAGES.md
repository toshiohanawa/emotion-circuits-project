# LaTeXパッケージのインストール

BasicTeXには基本的なパッケージしか含まれていないため、論文で使用している追加パッケージをインストールする必要があります。

## 必要なパッケージ

- `multirow` - 表の複数行結合
- `subcaption` - サブキャプション
- `enumitem` - リストのカスタマイズ

## インストール方法

以下のコマンドを実行してください（管理者パスワードが必要です）:

```bash
eval "$(/usr/libexec/path_helper)"
sudo tlmgr install multirow subcaption enumitem
```

または、インストールスクリプトを使用:

```bash
./install_packages.sh
```

## インストール後の確認

パッケージが正しくインストールされたか確認:

```bash
tlmgr list --only-installed | grep -E "(multirow|subcaption|enumitem)"
```

## トラブルシューティング

### tlmgrが見つからない

パスを更新:
```bash
eval "$(/usr/libexec/path_helper)"
```

### パッケージが見つからない

リポジトリを更新:
```bash
sudo tlmgr update --self
sudo tlmgr update --all
```

