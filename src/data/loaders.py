"""
Dataset loaders for FedQual-CPX.

Loads processed datasets from data/processed/<dataset>/ and provides
PyTorch-compatible datasets for federated training.

Currently supports:
    - CIFAR-10 (from processed .pt files)
    - FEMNIST and Shakespeare loaders will be added incrementally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset loaded from processed .pt tensors.

    Loads the pre-saved train.pt / test.pt files created by download_data.py.
    Images are normalized to [0, 1] float32 and transposed to CHW format.

    Args:
        processed_dir: Path to data/processed/cifar10/.
        train: If True, load train split; otherwise test split.
        transform: Optional callable transform applied to each image tensor.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        train: bool = True,
        transform: Any | None = None,
    ) -> None:
        processed_dir = Path(processed_dir)
        filename = "train.pt" if train else "test.pt"
        filepath = processed_dir / filename

        project_root = Path(__file__).resolve().parent.parent.parent
        alt_filepath = project_root / processed_dir / filename

        if not filepath.exists():
            if alt_filepath.exists():
                filepath = alt_filepath
            else:
                # Attempt auto-download and processing
                try:
                    from scripts.download_data import download_cifar10
                    print(f"[Dataset] Processed dataset not found at {filepath}. Auto-downloading...")
                    download_cifar10()
                    if alt_filepath.exists():
                        filepath = alt_filepath
                    elif (processed_dir / filename).exists():
                        filepath = processed_dir / filename
                except Exception as e:
                    print(f"[Dataset] Auto-download attempt failed: {e}")

        if not filepath.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {filepath}\n"
                f"Run: python scripts/download_data.py --dataset cifar10 --force"
            )

        data = torch.load(filepath, weights_only=True)
        # Images: uint8 [N, 32, 32, 3] → float32 [N, 3, 32, 32]
        self.images = data["images"].float().div(255.0).permute(0, 3, 1, 2)
        self.labels = data["labels"]
        self.transform = transform
        self.num_classes = 10

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class FederatedClientDataset(Dataset):
    """A dataset representing a single federated client's local data.

    Wraps a subset of the global dataset using index arrays.
    Supports train/val split for local validation.

    Args:
        base_dataset: The full global dataset.
        indices: Array of indices belonging to this client.
        val_fraction: Fraction of local data to hold out for validation.
        split: Which split to use — 'train' or 'val'.
        seed: Random seed for train/val split.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        indices: np.ndarray,
        val_fraction: float = 0.1,
        split: str = "train",
        seed: int = 42,
    ) -> None:
        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(indices)

        n_val = max(1, int(len(shuffled) * val_fraction))

        if split == "val":
            self.indices = shuffled[:n_val]
        else:
            self.indices = shuffled[n_val:]

        self.base_dataset = base_dataset
        self.split = split

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        global_idx = self.indices[idx]
        return self.base_dataset[global_idx]


def get_client_dataloader(
    client_dataset: FederatedClientDataset,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for a single federated client.

    Args:
        client_dataset: The client's local dataset.
        batch_size: Batch size for local training.
        shuffle: Whether to shuffle data each epoch.

    Returns:
        PyTorch DataLoader.
    """
    return DataLoader(
        client_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,  # Windows compatibility
        pin_memory=torch.cuda.is_available(),
    )
