"""Llama3 8B 専用ラッパ（汎用 summarize_phase8_large を呼び出すだけ）。"""
from __future__ import annotations

import argparse

from src.analysis.summarize_phase8_large import main as generic_main


def main() -> None:
    # llama3_8b をデフォルトで指定しつつ、その他の引数はそのまま渡す
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--large-model", default="llama3_8b")
    args, remaining = parser.parse_known_args()
    # 先頭に large-model を補う形で generic_main を呼ぶ
    argv = ["--large-model", args.large_model] + remaining
    generic_main(argv)


if __name__ == "__main__":
    main()
