"""Config loading and output-path resolution.

Config-driven per CLAUDE.md: no hyperparameters live in code. Every run reads a
single YAML file. Output paths in the YAML are relative and get resolved against
`output_dir` so the same config works locally and on Kaggle (/kaggle/working).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping.")
    return cfg


def output_path(cfg: dict[str, Any], relative: str) -> Path:
    """Resolve a config-relative output path against cfg['output_dir'].

    Absolute paths are returned unchanged so a config can override individual
    locations if needed.
    """
    rel = Path(relative)
    if rel.is_absolute():
        return rel
    base = Path(cfg.get("output_dir", "."))
    return base / rel


def ensure_parent(path: str | os.PathLike) -> Path:
    """Create the parent directory of `path` if missing; return the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
