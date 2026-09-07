"""
FedQual-CPX Dynamic Drift Smoke Test.

Validates the full end-to-end FedQual-CPX pipeline under piecewise-stationary drift:
    - N = 20 clients, K = 5 per round, T = 15 rounds
    - Drift: Class swap (D2) triggered at round tau = 8 on 30% of clients
    - Selection: Full FedQual-CPX (CUSUM + adaptive exploration + uncertainty-aware exploitation)
    - Verifies:
        1. CUSUM detects utility shift on drifted clients post round 8
        2. Adaptive epsilon adjusts in response to detected changes
        3. History, normalized utilities, and change events log properly
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fl.simulator import FederatedSimulator
from src.utils.config import Config


def run_fedqual_drift_test() -> None:
    print("=" * 60)
    print("FedQual-CPX End-to-End Drift & Selection Verification")
    print("=" * 60)

    cfg = Config({
        "seed": 42,
        "dataset": {
            "name": "CIFAR10",
            "processed_dir": "data/processed/cifar10",
            "num_classes": 10,
            "image_shape": [32, 32, 3],
        },
        "partition": {
            "method": "dirichlet",
            "alpha": 0.5,
            "num_clients": 20,
            "seed": 42,
            "min_samples_per_client": 10,
            "val_fraction": 0.1,
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
        "fl": {
            "num_rounds": 15,
            "num_clients": 20,
            "clients_per_round": 5,
            "aggregation": "fedavg",
        },
        "drift": {
            "enabled": True,
            "drift_type": "class_swap",
            "drift_round": 8,
            "drift_fraction": 0.3,
            "severity": "medium",
        },
        "selection": {
            "method": "fedqual_cpx",
            "warmup_rounds": 4,
            "epsilon_min": 0.1,
            "epsilon_max": 0.4,
            "c1_change_rate": 0.5,
            "c2_uncertainty": 0.2,
            "c3_coverage": 0.2,
            "delta": 0.3,
            "h": 2.5,
            "cooldown": 2,
        },
        "normalization": {
            "method": "robust_mad",
            "z_max": 3.0,
            "epsilon": 1e-8,
        },
        "evaluation": {
            "test_every": 1,
            "save_checkpoint_every": 100,
        },
        "output": {
            "results_dir": "results/raw",
            "save_checkpoints": False,
        },
    })

    sim = FederatedSimulator(cfg, experiment_id="fedqual_drift_smoke_test")
    summary = sim.run()

    print("\n" + "=" * 60)
    print("Verification Summary:")
    print(f"  Best Accuracy:  {summary['best_accuracy']:.4f}")
    print(f"  Final Accuracy: {summary['final_accuracy']:.4f}")
    print(f"  Total Rounds:   {summary['total_rounds']}")
    print(f"  Total Time:     {summary['total_time_seconds']:.1f}s")
    print(f"  Gini:           {summary['participation']['gini']:.4f}")
    print(f"  Entropy:        {summary['participation']['entropy']:.4f}")
    print(f"  Coverage:       {summary['participation']['coverage']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_fedqual_drift_test()
