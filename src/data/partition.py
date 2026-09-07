"""
Federated data partitioning for FedQual-CPX.

Implements Dirichlet non-IID partitioning for CIFAR-10 and provides
utilities to inspect and validate partition quality.

Per Section 26 of the plan:
    alpha ∈ {0.1, 0.5, 1.0}
    0.1 = highly heterogeneous
    0.5 = medium
    1.0 = relatively milder

Usage:
    from src.data.partition import DirichletPartitioner
    partitioner = DirichletPartitioner(num_clients=100, alpha=0.5, seed=42)
    client_indices = partitioner.partition(labels)
    stats = partitioner.get_statistics(labels, client_indices)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class DirichletPartitioner:
    """Partition a dataset across clients using Dirichlet distribution.

    For each class, the Dirichlet distribution determines what fraction
    of that class's samples goes to each client. This creates natural
    heterogeneity: small alpha → highly non-IID, large alpha → near-IID.

    Args:
        num_clients: Number of federated clients (N).
        alpha: Dirichlet concentration parameter.
        seed: Random seed for reproducibility.
        min_samples_per_client: Minimum samples each client must have.
    """

    def __init__(
        self,
        num_clients: int = 100,
        alpha: float = 0.5,
        seed: int = 42,
        min_samples_per_client: int = 10,
    ) -> None:
        self.num_clients = num_clients
        self.alpha = alpha
        self.seed = seed
        self.min_samples_per_client = min_samples_per_client

    def partition(self, labels: np.ndarray) -> dict[int, np.ndarray]:
        """Partition dataset indices across clients using Dirichlet allocation.

        Args:
            labels: Array of integer labels for the full dataset.

        Returns:
            Dictionary mapping client_id → array of global dataset indices.
        """
        rng = np.random.RandomState(self.seed)
        num_classes = len(np.unique(labels))
        num_samples = len(labels)

        # Group indices by class
        class_indices: dict[int, np.ndarray] = {}
        for c in range(num_classes):
            class_indices[c] = np.where(labels == c)[0]

        # Initialize client assignments
        client_indices: dict[int, list[int]] = {
            i: [] for i in range(self.num_clients)
        }

        # For each class, draw Dirichlet proportions and allocate
        for c in range(num_classes):
            indices_c = class_indices[c]
            rng.shuffle(indices_c)

            # Draw proportions from Dirichlet
            proportions = rng.dirichlet(
                np.full(self.num_clients, self.alpha)
            )

            # Convert proportions to actual counts
            # Use cumulative sum to split indices
            proportions = proportions / proportions.sum()
            counts = (proportions * len(indices_c)).astype(int)

            # Fix rounding: distribute remainder to clients with most allocation
            remainder = len(indices_c) - counts.sum()
            if remainder > 0:
                top_clients = np.argsort(-proportions)[:remainder]
                counts[top_clients] += 1

            # Assign indices to clients
            start = 0
            for client_id in range(self.num_clients):
                end = start + counts[client_id]
                client_indices[client_id].extend(indices_c[start:end].tolist())
                start = end

        # Convert to numpy arrays
        result: dict[int, np.ndarray] = {}
        for client_id in range(self.num_clients):
            arr = np.array(client_indices[client_id], dtype=np.int64)
            rng.shuffle(arr)  # shuffle within client
            result[client_id] = arr

        # Validate minimum samples
        empty_clients = [
            cid for cid, idx in result.items()
            if len(idx) < self.min_samples_per_client
        ]
        if empty_clients:
            print(
                f"  Warning: {len(empty_clients)} clients have fewer than "
                f"{self.min_samples_per_client} samples. "
                f"Consider increasing alpha or decreasing num_clients."
            )

        return result

    def get_statistics(
        self,
        labels: np.ndarray,
        client_indices: dict[int, np.ndarray],
    ) -> dict[str, Any]:
        """Compute partition statistics for validation and logging.

        Args:
            labels: Full dataset labels.
            client_indices: Output from partition().

        Returns:
            Dictionary with partition statistics.
        """
        num_classes = len(np.unique(labels))
        samples_per_client = [len(idx) for idx in client_indices.values()]

        # Per-client class distributions
        class_counts_per_client = []
        classes_per_client = []

        for client_id in range(self.num_clients):
            idx = client_indices[client_id]
            if len(idx) == 0:
                class_counts_per_client.append([0] * num_classes)
                classes_per_client.append(0)
                continue

            client_labels = labels[idx]
            counts = np.bincount(client_labels, minlength=num_classes)
            class_counts_per_client.append(counts.tolist())
            classes_per_client.append(int(np.sum(counts > 0)))

        stats = {
            "num_clients": self.num_clients,
            "alpha": self.alpha,
            "seed": self.seed,
            "total_samples": int(len(labels)),
            "samples_per_client": {
                "min": int(min(samples_per_client)),
                "max": int(max(samples_per_client)),
                "mean": round(float(np.mean(samples_per_client)), 2),
                "std": round(float(np.std(samples_per_client)), 2),
                "median": int(np.median(samples_per_client)),
            },
            "classes_per_client": {
                "min": int(min(classes_per_client)),
                "max": int(max(classes_per_client)),
                "mean": round(float(np.mean(classes_per_client)), 2),
            },
            "empty_clients": int(sum(1 for s in samples_per_client if s == 0)),
        }

        return stats

    def save_partition(
        self,
        client_indices: dict[int, np.ndarray],
        labels: np.ndarray,
        output_dir: str | Path,
    ) -> None:
        """Save partition to disk for reproducibility.

        Saves:
            - partition.npz: client_id → index arrays
            - partition_manifest.json: partition metadata and statistics

        Args:
            client_indices: Output from partition().
            labels: Full dataset labels.
            output_dir: Directory to save partition files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save index arrays
        np.savez(
            output_dir / "partition.npz",
            **{str(k): v for k, v in client_indices.items()},
        )

        # Save manifest with statistics
        stats = self.get_statistics(labels, client_indices)
        manifest = {
            "partition_method": "dirichlet",
            "alpha": self.alpha,
            "num_clients": self.num_clients,
            "seed": self.seed,
            "min_samples_per_client": self.min_samples_per_client,
            "statistics": stats,
        }

        with open(output_dir / "partition_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"  Partition saved to: {output_dir}")
        print(f"  Samples/client: min={stats['samples_per_client']['min']}, "
              f"max={stats['samples_per_client']['max']}, "
              f"mean={stats['samples_per_client']['mean']}")
        print(f"  Classes/client: min={stats['classes_per_client']['min']}, "
              f"max={stats['classes_per_client']['max']}, "
              f"mean={stats['classes_per_client']['mean']}")

    @staticmethod
    def load_partition(partition_dir: str | Path) -> dict[int, np.ndarray]:
        """Load a previously saved partition.

        Args:
            partition_dir: Directory containing partition.npz.

        Returns:
            Dictionary mapping client_id → array of global dataset indices.
        """
        partition_dir = Path(partition_dir)
        data = np.load(partition_dir / "partition.npz")
        return {int(k): data[k] for k in data.files}
