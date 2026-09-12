"""
Phase 15: Participation Crossover Sweep (K/N Sweep).

Investigates the participation ratio threshold rho = K / N where change-aware
client selection transitions from lagging behind random sampling to outperforming it.

Evaluates K in {5, 10, 25, 50, 100} at N = 100:
    - K/N = 0.05 (K=5): Severe partial observability, delay ~100 rounds
    - K/N = 0.10 (K=10): Standard FL setting, delay ~51 rounds
    - K/N = 0.25 (K=25): Moderate observability, delay ~20 rounds
    - K/N = 0.50 (K=50): High observability, delay ~10 rounds
    - K/N = 1.00 (K=100): Full observability, delay ~5 rounds

Outputs:
    - Raw results in results/raw/kn_sweep/
    - Consolidated tables in results/tables/kn_sweep_summary.csv and .json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fl.simulator import FederatedSimulator
from src.utils.config import Config


def build_kn_config(
    k_val: int,
    method_key: str,
    num_rounds: int = 100,
    seed: int = 42,
) -> Config:
    """Construct configuration for a specific K/N condition."""
    selection_cfg: dict[str, Any] = {}
    norm_cfg: dict[str, Any] = {}

    if method_key == "random":
        selection_cfg = {"method": "random"}
        norm_cfg = {"method": "none"}
    elif method_key == "fedqual_cpx":
        selection_cfg = {
            "method": "fedqual_cpx",
            "detector_name": "cusum",
            "warmup_rounds": max(5, int(50 // max(1, (100 // k_val)))),
            "epsilon_min": 0.05,
            "epsilon_max": 0.30,
            "delta": 0.3,
            "h": 2.5,
            "cooldown": 2,
            "use_uncertainty": True,
            "use_adaptive_epsilon": True,
            "use_change_bonus": True,
        }
        norm_cfg = {"method": "robust_mad", "z_max": 3.0, "epsilon": 1e-8}

    return Config({
        "seed": seed,
        "dataset": {
            "name": "CIFAR10",
            "processed_dir": "data/processed/cifar10",
            "num_classes": 10,
            "image_shape": [32, 32, 3],
        },
        "partition": {
            "method": "dirichlet",
            "alpha": 0.5,
            "num_clients": 100,
            "seed": seed,
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
            "num_rounds": num_rounds,
            "num_clients": 100,
            "clients_per_round": k_val,
            "aggregation": "fedavg",
        },
        "drift": {
            "enabled": True,
            "drift_type": "class_swap",
            "drift_round": num_rounds // 2,
            "drift_end_round": (num_rounds // 2) + 20,
            "severity": "medium",
            "drift_fraction": 0.3,
        },
        "selection": selection_cfg,
        "normalization": norm_cfg,
    })


def run_kn_sweep(
    k_list: list[int] | None = None,
    methods: list[str] | None = None,
    num_rounds: int = 100,
    seed: int = 42,
    output_dir: str = "results/tables",
) -> list[dict[str, Any]]:
    """Execute the participation crossover sweep."""
    if k_list is None:
        k_list = [5, 10, 25, 50]
    if methods is None:
        methods = ["random", "fedqual_cpx"]

    records = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FedQual-CPX Phase 15: Participation Crossover (K/N) Sweep")
    print(f"K values: {k_list} | N: 100 | Rounds: {num_rounds} | Seed: {seed}")
    print("=" * 80)

    for k_val in k_list:
        ratio = k_val / 100.0
        for method_key in methods:
            print(f"\n---> Running K={k_val} (rho={ratio:.2f}), Method={method_key}")
            cfg = build_kn_config(k_val, method_key, num_rounds=num_rounds, seed=seed)

            exp_id = f"kn_k{k_val}_{method_key}_seed{seed}"
            sim = FederatedSimulator(cfg, experiment_id=exp_id)
            results = sim.run()

            best_acc = float(results.get("best_accuracy", 0.0))
            final_acc = float(results.get("final_accuracy", 0.0))
            part = results.get("participation", {})
            gini = float(part.get("gini", 0.0))
            coverage = float(part.get("coverage", 0.0))

            rec = {
                "k": k_val,
                "ratio": ratio,
                "method": method_key,
                "best_accuracy": round(best_acc, 4),
                "final_accuracy": round(final_acc, 4),
                "gini": round(gini, 4),
                "coverage": round(coverage, 4),
            }
            records.append(rec)
            print(f"Recorded: Best Acc={best_acc:.2%}, Final Acc={final_acc:.2%}, Gini={gini:.4f}, Cov={coverage:.1%}")

    # Save summary tables
    csv_path = Path(output_dir) / "kn_sweep_summary.csv"
    json_path = Path(output_dir) / "kn_sweep_summary.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "ratio", "method", "best_accuracy", "final_accuracy", "gini", "coverage"])
        writer.writeheader()
        writer.writerows(records)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nSaved K/N sweep summary to {csv_path} and {json_path}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run participation crossover sweep.")
    parser.add_argument("--quick", action="store_true", help="Quick mode (15 rounds, subset of K)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.quick:
        run_kn_sweep(k_list=[10, 25], num_rounds=15, seed=args.seed)
    else:
        run_kn_sweep(k_list=[5, 10, 25, 50], num_rounds=100, seed=args.seed)
