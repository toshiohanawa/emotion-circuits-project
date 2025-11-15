"""
Utilities for working with model identifiers.

Many experiment scripts need to derive directory-friendly names from the
Hugging Face identifiers (e.g., ``EleutherAI/pythia-160m``).  Historically the
codebase produced a mix of ``pythia-160m`` and ``EleutherAI-pythia-160m`` paths,
which made it hard to tell experiments apart.  This module centralises the
logic so every caller can generate consistent, “sanitised” names.
"""
from __future__ import annotations

from pathlib import Path


def sanitize_model_name(model_name: str) -> str:
    """
    Convert a Hugging Face model identifier into a filesystem-friendly slug.

    Examples
    --------
    >>> sanitize_model_name("EleutherAI/pythia-160m")
    'EleutherAI-pythia-160m'
    >>> sanitize_model_name("meta-llama/Llama-2-7b-hf")
    'meta-llama-Llama-2-7b-hf'
    """
    safe = model_name.replace("/", "-").replace(":", "-")
    safe = safe.replace(" ", "_")
    # Windows filesystems dislike trailing dots; strip them defensively.
    return safe.rstrip(".")


def ensure_model_subdir(root: Path, model_name: str) -> Path:
    """
    Create (if necessary) and return the experiment subdirectory for a model.

    The helper exists so scripts do not have to repeat ``sanitize_model_name``
    and ``mkdir`` boilerplate.
    """
    safe_name = sanitize_model_name(model_name)
    path = root / safe_name
    path.mkdir(parents=True, exist_ok=True)
    return path
