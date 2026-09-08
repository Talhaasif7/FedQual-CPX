"""
FedQual-CPX Dataset Acquisition Script
=======================================

Downloads, preprocesses, and stores all datasets required for the
FedQual-CPX federated learning experiments.

Datasets:
    - CIFAR-10:      via torchvision (auto-download)
    - FEMNIST:       via flwr-datasets / Hugging Face (flwrlabs/femnist)
    - Shakespeare:   via Hugging Face datasets (flwrlabs/shakespeare)

Usage:
    python scripts/download_data.py --dataset cifar10
    python scripts/download_data.py --dataset femnist
    python scripts/download_data.py --dataset shakespeare
    python scripts/download_data.py --all

All data is stored under data/raw/<dataset>/ and data/processed/<dataset>/.
Experiments load only from local project directories — no live downloads
during training.

This script is idempotent: re-running does not corrupt or duplicate data.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root (one level up from scripts/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def _ensure_dirs(*dirs: Path) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def _write_manifest(path: Path, manifest: dict) -> None:
    """Write a JSON manifest atomically."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Manifest written: {path}")


def _manifest_exists(dataset_name: str) -> bool:
    """Check whether a processed manifest and data files already exist."""
    folder = DATA_PROCESSED / dataset_name
    return (
        (folder / "manifest.json").exists()
        and (folder / "train.pt").exists()
        and (folder / "test.pt").exists()
    )


# ===================================================================
# CIFAR-10
# ===================================================================
def download_cifar10(force: bool = False) -> None:
    """Download CIFAR-10 via torchvision and save processed tensors."""

    dataset_name = "cifar10"
    raw_dir = DATA_RAW / dataset_name
    proc_dir = DATA_PROCESSED / dataset_name

    if _manifest_exists(dataset_name) and not force:
        print(f"[cifar10] Already processed. Skipping. (use --force to re-download)")
        return

    _ensure_dirs(raw_dir, proc_dir)

    # -- lazy imports so the script can report helpful errors ----------------
    try:
        import torch
        from torchvision.datasets import CIFAR10
    except ImportError:
        print("[cifar10] ERROR: torch / torchvision not installed.")
        print("  Run: pip install torch torchvision")
        sys.exit(1)

    print("[cifar10] Downloading via torchvision ...")
    train_ds = CIFAR10(root=str(raw_dir), train=True, download=True)
    test_ds = CIFAR10(root=str(raw_dir), train=False, download=True)

    # Save as tensors for fast loading later
    import numpy as np

    train_images = torch.tensor(np.array(train_ds.data), dtype=torch.uint8)
    train_labels = torch.tensor(train_ds.targets, dtype=torch.long)
    test_images = torch.tensor(np.array(test_ds.data), dtype=torch.uint8)
    test_labels = torch.tensor(test_ds.targets, dtype=torch.long)

    torch.save({"images": train_images, "labels": train_labels},
               proc_dir / "train.pt")
    torch.save({"images": test_images, "labels": test_labels},
               proc_dir / "test.pt")

    # Class distribution
    classes = train_ds.classes
    train_dist = {c: int((train_labels == i).sum()) for i, c in enumerate(classes)}
    test_dist = {c: int((test_labels == i).sum()) for i, c in enumerate(classes)}

    manifest = {
        "dataset": "CIFAR-10",
        "source": "torchvision.datasets.CIFAR10",
        "source_url": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "download_date": datetime.now(timezone.utc).isoformat(),
        "num_classes": len(classes),
        "classes": classes,
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "image_shape": list(train_images.shape[1:]),
        "train_class_distribution": train_dist,
        "test_class_distribution": test_dist,
        "partition_method": "synthetic_dirichlet",
        "notes": "Requires Dirichlet partitioning into federated clients at experiment time."
    }
    _write_manifest(proc_dir / "manifest.json", manifest)

    print(f"[cifar10] Done. {len(train_ds)} train / {len(test_ds)} test samples.")
    print(f"  Classes: {classes}")
    print(f"  Saved to: {proc_dir}")


# ===================================================================
# FEMNIST
# ===================================================================
def download_femnist(force: bool = False) -> None:
    """Download FEMNIST via flwr-datasets, preserving writer_id → client mapping."""

    dataset_name = "femnist"
    raw_dir = DATA_RAW / dataset_name
    proc_dir = DATA_PROCESSED / dataset_name

    if _manifest_exists(dataset_name) and not force:
        print(f"[femnist] Already processed. Skipping. (use --force to re-download)")
        return

    _ensure_dirs(raw_dir, proc_dir)

    try:
        import torch
        import numpy as np
        from flwr_datasets import FederatedDataset
        from flwr_datasets.partitioner import NaturalIdPartitioner
    except ImportError:
        print("[femnist] ERROR: required packages not installed.")
        print("  Run: pip install torch flwr-datasets")
        sys.exit(1)

    print("[femnist] Downloading via flwr-datasets (flwrlabs/femnist) ...")
    print("  This may take several minutes on first run.")

    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={
            "train": NaturalIdPartitioner(partition_by="writer_id")
        },
    )

    num_partitions = fds.partitioners["train"].num_partitions
    print(f"  Found {num_partitions} natural clients (writers).")

    # Collect per-client statistics
    client_stats = []
    all_labels = []

    for pid in range(num_partitions):
        partition = fds.load_partition(partition_id=pid)
        n_samples = len(partition)

        # Extract labels for distribution
        labels = []
        for sample in partition:
            if "label" in sample:
                labels.append(sample["label"])
            elif "character" in sample:
                labels.append(sample["character"])

        client_stats.append({
            "partition_id": pid,
            "num_samples": n_samples,
        })
        all_labels.extend(labels)

        if pid % 500 == 0 and pid > 0:
            print(f"    Processed {pid}/{num_partitions} partitions ...")

    # Save raw partitioner info
    total_samples = sum(c["num_samples"] for c in client_stats)
    samples_per_client = [c["num_samples"] for c in client_stats]

    manifest = {
        "dataset": "FEMNIST",
        "source": "flwrlabs/femnist (Hugging Face via flwr-datasets)",
        "source_url": "https://huggingface.co/datasets/flwrlabs/femnist",
        "original_source": "https://github.com/TalwalkarLab/leaf",
        "download_date": datetime.now(timezone.utc).isoformat(),
        "num_clients": num_partitions,
        "total_train_samples": total_samples,
        "num_classes": 62,
        "class_description": "10 digits + 26 uppercase + 26 lowercase letters",
        "partition_method": "natural_writer_id",
        "client_identity": "writer_id",
        "samples_per_client_min": min(samples_per_client),
        "samples_per_client_max": max(samples_per_client),
        "samples_per_client_mean": round(sum(samples_per_client) / len(samples_per_client), 2),
        "notes": "Natural federated partition by writer. Do NOT re-shuffle client structure."
    }
    _write_manifest(proc_dir / "manifest.json", manifest)

    # Save client stats for quick reference
    with open(proc_dir / "client_stats.json", "w", encoding="utf-8") as f:
        json.dump(client_stats, f, indent=2)

    print(f"[femnist] Done. {num_partitions} clients, {total_samples} total samples.")
    print(f"  Samples/client: min={min(samples_per_client)}, "
          f"max={max(samples_per_client)}, "
          f"mean={manifest['samples_per_client_mean']}")
    print(f"  Saved to: {proc_dir}")


# ===================================================================
# Shakespeare
# ===================================================================
def download_shakespeare(force: bool = False) -> None:
    """Download Shakespeare via HuggingFace datasets, preserving character/play client mapping."""

    dataset_name = "shakespeare"
    raw_dir = DATA_RAW / dataset_name
    proc_dir = DATA_PROCESSED / dataset_name

    if _manifest_exists(dataset_name) and not force:
        print(f"[shakespeare] Already processed. Skipping. (use --force to re-download)")
        return

    _ensure_dirs(raw_dir, proc_dir)

    try:
        import torch
        from datasets import load_dataset
    except ImportError:
        print("[shakespeare] ERROR: required packages not installed.")
        print("  Run: pip install torch datasets")
        sys.exit(1)

    print("[shakespeare] Downloading via HuggingFace (flwrlabs/shakespeare) ...")

    ds = load_dataset("flwrlabs/shakespeare")

    # Inspect available splits
    print(f"  Available splits: {list(ds.keys())}")

    total_samples = 0
    split_info = {}

    for split_name, split_data in ds.items():
        n = len(split_data)
        total_samples += n
        split_info[split_name] = n
        print(f"  Split '{split_name}': {n} samples")

        # Save each split as JSON for portability
        split_data.to_json(str(proc_dir / f"{split_name}.json"))

    # Try to identify client/user column
    sample = ds[list(ds.keys())[0]][0]
    columns = list(sample.keys())
    print(f"  Columns: {columns}")

    manifest = {
        "dataset": "Shakespeare",
        "source": "flwrlabs/shakespeare (Hugging Face)",
        "source_url": "https://huggingface.co/datasets/flwrlabs/shakespeare",
        "original_source": "https://github.com/TalwalkarLab/leaf",
        "download_date": datetime.now(timezone.utc).isoformat(),
        "total_samples": total_samples,
        "splits": split_info,
        "columns": columns,
        "task": "next_character_prediction",
        "partition_method": "natural_character_play",
        "client_identity": "character/play combination",
        "notes": "Natural federated partition by character/play user. "
                 "Do NOT re-shuffle client structure."
    }
    _write_manifest(proc_dir / "manifest.json", manifest)

    print(f"[shakespeare] Done. {total_samples} total samples across {len(split_info)} splits.")
    print(f"  Saved to: {proc_dir}")


# ===================================================================
# CLI
# ===================================================================
DATASET_HANDLERS = {
    "cifar10": download_cifar10,
    "femnist": download_femnist,
    "shakespeare": download_shakespeare,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FedQual-CPX Dataset Acquisition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_data.py --dataset cifar10
    python scripts/download_data.py --dataset femnist
    python scripts/download_data.py --dataset shakespeare
    python scripts/download_data.py --all
    python scripts/download_data.py --all --force
        """,
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_HANDLERS.keys()),
        help="Which dataset to download.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if already processed.",
    )

    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.print_help()
        sys.exit(1)

    print("=" * 60)
    print("FedQual-CPX Dataset Acquisition")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data dir: {DATA_RAW}")
    print(f"Processed dir: {DATA_PROCESSED}")
    print("=" * 60)

    if args.all:
        for name, handler in DATASET_HANDLERS.items():
            print(f"\n{'─' * 40}")
            handler(force=args.force)
    else:
        print()
        DATASET_HANDLERS[args.dataset](force=args.force)

    print(f"\n{'=' * 60}")
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
