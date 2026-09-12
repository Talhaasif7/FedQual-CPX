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


class FEMNISTDataset(Dataset):
    """EMNIST-ByClass (62-class) character dataset.

    Images: normalized to [0, 1] float32, shape [N, 1, 28, 28].
    Labels: [N] long integers, 62 classes (10 digits + 26 upper + 26 lower).
    Partitioned across federated clients via Dirichlet non-IID distribution.
    """

    def __init__(
        self,
        processed_dir: str | Path = "data/processed/femnist",
        train: bool = True,
        transform: Any | None = None,
        auto_download: bool = True,
    ) -> None:
        processed_dir = Path(processed_dir)
        filename = "train.pt" if train else "test.pt"
        filepath = processed_dir / filename

        project_root = Path(__file__).resolve().parent.parent.parent
        alt_filepath = project_root / processed_dir / filename

        if not filepath.exists() and alt_filepath.exists():
            filepath = alt_filepath

        if not filepath.exists():
            processed_dir.mkdir(parents=True, exist_ok=True)
            if auto_download:
                try:
                    from torchvision.datasets import EMNIST
                    raw_dir = project_root / "data" / "raw" / "emnist"
                    emnist = EMNIST(root=str(raw_dir), split="byclass", train=train, download=True)
                    images = emnist.data.unsqueeze(1).float().div(255.0)  # [N, 1, 28, 28]
                    labels = emnist.targets.long()
                    # Deterministically shuffle before truncating to preserve class balance across 62 classes
                    perm = torch.randperm(len(labels), generator=torch.Generator().manual_seed(42 if train else 142))
                    n_samples = min(len(labels), 50000 if train else 10000)
                    selected_idx = perm[:n_samples]
                    images = images[selected_idx]
                    labels = labels[selected_idx]
                    torch.save({"images": images, "labels": labels}, filepath)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to download and process EMNIST-ByClass dataset: {e}\n"
                        f"Please run 'python scripts/download_data.py --dataset femnist' with network access."
                    ) from e

            if not filepath.exists():
                raise FileNotFoundError(
                    f"Processed EMNIST-ByClass dataset not found at {filepath}.\n"
                    f"Run: python scripts/download_data.py --dataset femnist"
                )

        data = torch.load(filepath, weights_only=True)
        img = data["images"]
        if img.dim() == 3:
            img = img.unsqueeze(1)
        if img.dtype == torch.uint8:
            img = img.float().div(255.0)
        self.images = img
        self.labels = data["labels"].long()
        self.transform = transform
        self.num_classes = 62

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class ShakespeareDataset(Dataset):
    """Shakespeare next-character prediction dataset.

    Tokenized sequences of length seq_len (default 80), predicting next character (vocab_size 90).
    """

    def __init__(
        self,
        processed_dir: str | Path = "data/processed/shakespeare",
        train: bool = True,
        seq_len: int = 80,
        vocab_size: int = 90,
    ) -> None:
        processed_dir = Path(processed_dir)
        filename = "train.pt" if train else "test.pt"
        filepath = processed_dir / filename

        project_root = Path(__file__).resolve().parent.parent.parent
        alt_filepath = project_root / processed_dir / filename

        if not filepath.exists() and alt_filepath.exists():
            filepath = alt_filepath

        if not filepath.exists():
            raise FileNotFoundError(
                f"Processed Shakespeare dataset not found at {filepath}.\n"
                f"Synthetic uniform tokenization has been permanently disabled per data integrity standards.\n"
                f"Please provide authentic text sequences or evaluate on CIFAR-10 / EMNIST-ByClass."
            )

        data = torch.load(filepath, weights_only=True)
        self.sequences = data["sequences"].long()
        self.labels = data["labels"].long()
        self.num_classes = vocab_size

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


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
