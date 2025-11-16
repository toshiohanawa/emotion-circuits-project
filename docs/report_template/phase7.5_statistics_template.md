Phase 7.5 — 統計的検証
🎯 目的

残差パッチングおよび Head パッチング実験の効果量（mean difference, Cohen’s d）、信頼区間、p 値をまとめ、結果の統計的有意性を確認する。

検出力 (power) 分析により、観測された効果量に対して十分なサンプルサイズを推定し、今後のデータ収集の目安とする。

感情サブスペースの次元数 k がモデル間の重なり (overlap) に与える影響を比較し、低次元コア因子の存在を検証する。

📦 生成物

effect_sizes.csv – 残差 / head / random patching に対する効果量と p 値を含む表。

power_analysis.csv, power_analysis.json – 効果量に基づく検出力分析結果と、効果量ごとの必要サンプルサイズ。

k_selection.csv, k_sweep_raw.csv – k 次元サブスペースの overlap 値とそのブートストラップ統計。

docs/phase7_5_report.md – 本テンプレートを用いたレポート。

🚀 実行コマンド例
python3 -m src.analysis.run_statistics \
  --profile baseline \
  --mode all \
  --phase-filter residual,head,random \
  --n-bootstrap 500 \
  --effect-targets 0.2 0.5 \
  --power-target 0.85 \
  --seed 42


上記は baseline プロファイルに対して残差・head・ランダム patching 全ての統計解析を実行する例である。 --effect-targets と --power-target は検出力分析の目標値を指定する。

📄 レポート項目
1. 統計解析設定

モデル: [gpt2 / pythia‑160m / gpt‑neo‑125M]

プロファイル: [baseline / extended]

対象フェーズ: [residual / head / random] – 複数指定可

ブートストラップ回数: [数] – 信頼区間推定に使用

使用メトリクス: [Sentiment, Politeness, Emotion Keywords など]

有意水準 (α): [値] – p 値の判定基準

2. 効果量・有意性の結果
メトリクス別効果量一覧
メトリクス	レイヤー/ヘッド	mean_diff	Cohen’s d	95% CI	p 値	有意性 (FDR)
Sentiment (POSITIVE)	[layer/head]	[値]	[値]	[下–上]	[値]	[Yes/No]
Politeness	[layer/head]	[値]	[値]	[下–上]	[値]	[Yes/No]
Gratitude keywords	[layer/head]	[値]	[値]	[下–上]	[値]	[Yes/No]
Anger keywords	[layer/head]	[値]	[値]	[下–上]	[値]	[Yes/No]
Apology keywords	[layer/head]	[値]	[値]	[下–上]	[値]	[Yes/No]
…	…	…	…	…	…	…

※ mean_diff は baseline と patched の差 (patched − baseline)、Cohen’s d は効果の大きさを示す標準化指標。Bonferroni/FDR 補正後の有意性も併記する。

3. 検出力分析

観測された効果量 (Cohen’s d): [範囲] — 効果の大小を概観する。

事後的検出力 (post‑hoc power): [値]（サンプル数 n = [値], 有意水準 α=[値]） – 現状のデータで効果が検出可能かどうか。

効果量 0.2 を検出する必要サンプル数: [値] – small effect を検出するのに必要な n。

効果量 0.5 を検出する必要サンプル数: [値] – medium effect を検出するのに必要な n。

コメント: [検出力が十分かどうか、サンプルサイズの不足が原因かなど]

4. k 値選択の検証

感情サブスペースを主成分解析 (PCA) により k 次元に圧縮したとき、異なるモデル間の overlap 値がどのように変化するかを示す。k に対する平均 overlap と信頼区間を比較し、最も情報を保持する次元数を考察する。

k	平均 overlap	95% CI	コメント
1	[値]	[下–上]	[記述]
2	[値]	[下–上]	[2 次元が最も高い場合の言及]
3	[値]	[下–上]	[記述]
4	[値]	[下–上]	[記述]
5	[値]	[下–上]	[記述]
5. 考察
効果量の解釈

観測された効果量がほとんど small である場合、残差パッチングおよび head パッチングによる感情変化はモデル内部で極めて僅かであることを示す。

有意な効果が検出された場合は、それがどの感情・層・head で起きたかを指摘し、その可能な回路的解釈を述べる。

検出力の課題

サンプル数が少ないとき small effect の検出力が低くなることに注意し、必要な追加データ量を検討する。

将来の拡張 (Phase 8 以降) では、extended プロファイルやより大きなモデルで再検証を行うことが望ましい。

k 値選択の示唆

k=2 において overlap が最大になる場合、感情表現のコア因子が 2 次元に凝縮されていることを示唆する。

k が増えるにつれて overlap が減少する場合は、余分な次元がノイズを含んでいる可能性がある。

次のフェーズへの提案

Phase 8 では中規模モデルへの拡張を行い、データ量を増やして効果量と検出力を向上させる。

統計的検証を継続し、head パッチングの因果効果やサブスペースの一般性を再評価する。

📝 備考

本テンプレートは、Phase 7.5 の統計解析結果をわかりやすく整理するためのガイドラインである。必要に応じて項目を追加・削除し、実験内容に合わせて調整してほしい。

報告書では、各テーブル・グラフに対して簡潔な本文による解説を添え、主要な発見や解釈を強調すること。