"""
デバイス管理ユーティリティ。

MPS > CUDA > CPU の優先順位でデバイスを自動選択し、
テンソルやデータ構造を再帰的にデバイスに移動する機能を提供する。
"""
from __future__ import annotations

from typing import Any, Dict, List, Union

import torch


def get_default_device() -> torch.device:
    """
    利用可能な最適なデバイスを返す。
    
    優先順位: MPS > CUDA > CPU
    
    Returns:
        利用可能な最適なデバイス
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_default_device_str() -> str:
    """
    利用可能な最適なデバイスを文字列で返す。
    
    優先順位: MPS > CUDA > CPU
    
    Returns:
        デバイス名（"mps", "cuda", "cpu"）
    """
    device = get_default_device()
    return str(device)


def move_to_device(data: Any, device: Union[str, torch.device]) -> Any:
    """
    データ構造を再帰的に指定デバイスに移動する。
    
    dict, list, tuple, torch.Tensor を再帰的に処理し、
    すべてのテンソルを指定デバイスに移動する。
    
    Args:
        data: 移動するデータ（dict, list, tuple, torch.Tensor など）
        device: 移動先デバイス（文字列または torch.device）
    
    Returns:
        デバイスに移動されたデータ
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        moved = [move_to_device(item, device) for item in data]
        return type(data)(moved)  # list または tuple を保持
    else:
        # その他の型（int, float, str など）はそのまま返す
        return data

