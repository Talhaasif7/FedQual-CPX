"""
YAML-based configuration system for FedQual-CPX experiments.

Loads experiment configs from YAML files, supports nested access,
and records the config alongside results for reproducibility.

Usage:
    from src.utils.config import load_config
    cfg = load_config("configs/cifar10.yaml")
    print(cfg.dataset.name)         # "CIFAR10"
    print(cfg.fl.num_clients)       # 100
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


class Config:
    """Dot-accessible, nested configuration container.

    Supports dict-style and attribute-style access.
    Immutable after creation unless explicitly modified.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        if data is None:
            data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        """Convert back to a plain dictionary (recursive)."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def load_config(path: str | Path) -> Config:
    """Load a YAML config file and return a Config object.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Config object with dot-accessible attributes.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    return Config(data)


def save_config(cfg: Config, path: str | Path) -> None:
    """Save a Config object to a YAML file.

    Args:
        cfg: Config object to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False, sort_keys=False)


def get_reproducibility_info() -> dict[str, str]:
    """Collect system info for the reproducibility tuple.

    Returns:
        Dictionary with git commit, Python/PyTorch versions, hardware, timestamp.
    """
    info: dict[str, str] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda or "N/A"
        info["gpu_name"] = torch.cuda.get_device_name(0)

    # Try to get git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["git_commit"] = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["git_commit"] = "unknown"

    return info
