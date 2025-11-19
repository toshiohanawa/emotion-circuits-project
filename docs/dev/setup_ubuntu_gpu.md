# Ubuntu GPU Machine Setup Guide for LLM Research

**Target hardware (実績構成):**

- Ubuntu 22.04 LTS
- NVIDIA RTX 3090
- CUDA Toolkit 12.1
- PyTorch 2.5.1 (CUDA 12.1)
- vLLM（FlashInfer 無効版）
  - VLLM_ATTENTION_BACKEND=FLASHINFER_OFF

本ドキュメントは GPU マシンのセットアップを **再現性高く構築できるように整理した公式手順**です。
LLM 研究（Pythia, GPT-2, GPT-Neo, Llama 系）を行うために必要な環境構築情報を網羅します。

---

# 0. BIOS 設定（必須）

## ✔ Secure Boot を無効化

**NVIDIA ドライバが読み込まれなくなるため、必ず OFF。**
ASUS 例：
Advanced → Boot → Secure Boot → Secure Boot Mode: Custom
Secure Boot → Disabled

yaml
コードをコピーする

## ✔ OS Type: UEFI Mode

---

# 1. Ubuntu インストール後の初期セットアップ

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget python3-pip
2. NVIDIA Driver（apt 経由でインストール）
CUDA .run 版に含まれるドライバと競合するため、
ドライバは apt でインストールするのが最も安全。

bash
コードをコピーする
sudo ubuntu-drivers autoinstall
sudo reboot
確認：

bash
コードをコピーする
nvidia-smi
3. CUDA Toolkit 12.1（任意）
vLLM や PyTorch は 独自の CUDA Runtime を同梱しているため、
研究用には Toolkit は必須ではないが、コンパイル用途で便利。

公式 .run を使う場合は ドライバのチェックを必ず外す。

例：

bash
コードをコピーする
sudo sh cuda_12.1.1_530.30.02_linux.run
確認：

bash
コードをコピーする
nvcc --version
4. CUDA 環境変数の設定
~/.bashrc に追記：

bash
コードをコピーする
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
反映：

bash
コードをコピーする
source ~/.bashrc
5. PyTorch（CUDA 12.1 対応版）
PyTorch 2.5.1 + cu121 が研究用として最も安定。

bash
コードをコピーする
pip install "torch==2.5.1+cu121" \
            "torchvision==0.20.1+cu121" \
            "torchaudio==2.5.1" \
            --index-url https://download.pytorch.org/whl/cu121
確認：

python
コードをコピーする
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
6. vLLM セットアップ（FlashInfer 無効版）
FlashInfer は PyTorch 2.6+ 必須のため、この環境ではオフにする。

bash
コードをコピーする
pip install vllm
echo 'export VLLM_ATTENTION_BACKEND=FLASHINFER_OFF' >> ~/.bashrc
source ~/.bashrc
起動テスト：

python
コードをコピーする
from vllm import LLM
llm = LLM("meta-llama/Meta-Llama-3-8B-Instruct")
print("vLLM OK")
7. Hugging Face Login
bash
コードをコピーする
pip install huggingface_hub
python3 -m huggingface_hub login
ブラウザで発行した Token を貼り付ける。

8. 日本語入力（Fcitx5 + Mozc）
bash
コードをコピーする
sudo apt install -y fcitx5 fcitx5-mozc fcitx5-configtool
im-config -n fcitx5
再起動後：

scss
コードをコピーする
設定 → 地域と言語 → 入力ソース → Japanese(Mozc)
IME 切替は以下がおすすめ：

Ctrl + Space

Super + Space

9. 必要に応じて Node.js / npm（OpenAI SDK）
Node.js:

bash
コードをコピーする
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
OpenAI SDK:

bash
コードをコピーする
npm install -g openai
10. GPU 研究のための CUDA・PyTorch・vLLM 注意点
10.1 GPU とモデル規模
Llama 8B → bf16 で約 12GB VRAM

RTX 3090（24GB）は 8B~14B モデルの単カード推論に最適

10.2 CUDA バージョン互換（最重要）
PyTorch 2.5.1 → CUDA Runtime 12.1

vLLM → CUDA12 系が安定

FlashInfer → PyTorch 2.6 以上必須（今回は OFF）

10.3 メモリ節約のポイント
vLLM の量子化 (quantization="awq" / "fp8" / "int8")

kv_cache_dtype（fp8 / half）

rope_scaling の互換性を確認

11. この環境で可能な研究
TransformerLens / Neuron Viewer / MI 研究

GPT-2・Pythia・Llama ファミリの再現実験

vLLM による高速バッチ推論

感情方向ベクトルの抽出・可視化

あなたの「Emotion Circuits Project」全フェーズ

12. トラブルシューティング
❌ NVIDIA-SMI がエラー
Secure Boot → Disabled を確認。

❌ vLLM 起動時に FlashInfer エラー
環境変数で無効化：

bash
コードをコピーする
export VLLM_ATTENTION_BACKEND=FLASHINFER_OFF
❌ CUDA_HOME が無い
nvcc --version の結果に合わせて設定：

bash
コードをコピーする
export CUDA_HOME=/usr/local/cuda-12.1
13. バージョン固定（再現性のため）
このセットアップで安定動作が確認されたバージョン：

Component	Version
Ubuntu	22.04 LTS
NVIDIA Driver	535.xx（autoinstall）
CUDA Toolkit	12.1
PyTorch	2.5.1 + cu121
vLLM	0.4.2
HuggingFace Hub	0.36.x
Python	3.10
GPU	RTX 3090

14. 付録：環境のバックアップ
Python パッケージ一覧：

bash
コードをコピーする
pip freeze > requirements.txt
.bashrc バックアップ：

bash
コードをコピーする
cp ~/.bashrc docs/bashrc.backup
```
