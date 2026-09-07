"""
FedQual-CPX Smoke Test (Section 53)

Quick validation that the entire pipeline works end-to-end:
    Dataset → Partition → Clients → FedAvg → Evaluate → Log

Uses minimal settings for fast execution:
    N=20 clients, K=5 per round, T=10 rounds, 1 seed

Usage:
    python experiments/smoke_test.py
    python experiments/smoke_test.py --config configs/cifar10.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, Config
from src.fl.simulator import FederatedSimulator


def get_smoke_config(base_config_path: str | None = None) -> Config:
    """Create a minimal smoke-test configuration.

    Overrides the base config with small values for fast execution.
    """
    if base_config_path:
        cfg = load_config(base_config_path)
        data = cfg.to_dict()
    else:
        data = {
            "dataset": {
                "name": "CIFAR10",
                "processed_dir": "data/processed/cifar10",
                "num_classes": 10,
                "image_shape": [32, 32, 3],
            },
            "model": {
                "name": "SmallCNN",
                "num_classes": 10,
            },
            "training": {
                "local_epochs": 1,
                "batch_size": 32,
                "optimizer": "sgd",
                "learning_rate": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
            },
            "normalization": {
                "method": "robust_mad",
                "z_max": 3.0,
                "epsilon": 1e-8,
            },
            "output": {
                "results_dir": "results/raw",
                "save_checkpoints": False,
            },
        }

    # Override for smoke test (small and fast)
    data["partition"] = {
        "method": "dirichlet",
        "alpha": 0.5,
        "num_clients": 20,
        "seed": 42,
        "min_samples_per_client": 10,
        "val_fraction": 0.1,
    }
    data["fl"] = {
        "num_rounds": 10,
        "num_clients": 20,
        "clients_per_round": 5,
        "aggregation": "fedavg",
    }
    data["selection"] = {"method": "random"}
    data["evaluation"] = {"test_every": 1, "save_checkpoint_every": 100}
    data["seed"] = 42

    return Config(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="FedQual-CPX Smoke Test")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Base config to override (optional).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  FedQual-CPX SMOKE TEST")
    print("  N=20, K=5, T=10, seed=42")
    print("=" * 60)

    cfg = get_smoke_config(args.config)
    sim = FederatedSimulator(cfg, experiment_id="smoke_test")
    sim.setup()
    summary = sim.run()

    # Validate basic expectations
    acc = summary["best_accuracy"]
    print(f"\n--- Smoke Test Validation ---")
    print(f"Best accuracy: {acc:.4f}")

    if acc > 0.10:
        print("[PASS] Model is learning (accuracy > random chance)")
    else:
        print("[WARNING] Accuracy is at random chance level")

    if summary["participation"]["coverage"] > 0.5:
        print("[PASS] Client coverage is reasonable")
    else:
        print("[WARNING] Low client coverage")

    print("\n[PASS] Smoke test complete -- pipeline works end-to-end")


if __name__ == "__main__":
    main()
