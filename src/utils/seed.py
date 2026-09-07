"""
Deterministic seeding utility for reproducible experiments.

Usage:
    from src.utils.seed import set_seed
    set_seed(42)

Guarantees:
    same seed → same partition → same training → reproducible run
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility.

    Sets seeds for: Python random, NumPy, PyTorch CPU, PyTorch CUDA.
    Also configures PyTorch for deterministic behavior.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic operations (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed for reproducible hashing
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Return the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
