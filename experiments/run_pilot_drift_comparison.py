"""
FedQual-CPX Pilot Comparison Experiment under Piecewise-Stationary Drift.

Compares 4 core strategies under identical non-IID partitioning and drift:
    1. B0: Random Selection (FedAvg)
    2. B2: Utility Greedy
    3. B4: Fixed Exploration (epsilon = 0.1)
    4. B8: FedQual-CPX (proposed sequential CUSUM + adaptive exploration)

Protocol:
    - Dataset: CIFAR-10, Dirichlet alpha = 0.5
    - Clients: N = 30, K = 5 per round, T = 30 rounds
    - Piecewise-Stationary Drift: Class swap (D2) at tau = 15 on 30% of clients
    - Output: Metrics saved to results/raw/<exp_id>/, summary table in results/tables/

Answers:
    RQ1: Does sequential change detection identify utility shifts?
    RQ2: Does change-point-aware selection improve accuracy post-drift?
    RQ3: Does FedQual-CPX recover faster from client distribution changes?
    RQ4: Does adaptive exploration outperform fixed exploration and greedy selection?
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fl.simulator import FederatedSimulator
from src.utils.config import Config


def get_base_experiment_config(
    method: str,
    exp_id: str,
    num_clients: int = 30,
    clients_per_round: int = 5,
    num_rounds: int = 30,
    drift_round: int = 15,
    seed: int = 42,
) -> Config:
    """Construct configuration for an experimental condition."""

    selection_dict: dict[str, Any] = {"method": method}
    if method == "fedqual_cpx":
        selection_dict.update({
            "warmup_rounds": 5,
            "epsilon_min": 0.08,
            "epsilon_max": 0.35,
            "c1_change_rate": 0.5,
            "c2_uncertainty": 0.2,
            "c3_coverage": 0.2,
            "delta": 0.3,
            "h": 2.5,
            "cooldown": 2,
        })
    elif method == "fixed_exploration":
        selection_dict.update({
            "epsilon": 0.15,
            "window_size": 5,
        })
    elif method == "sliding_window":
        selection_dict.update({
            "window_size": 5,
        })

    cfg = Config({
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
            "num_clients": num_clients,
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
            "num_clients": num_clients,
            "clients_per_round": clients_per_round,
            "aggregation": "fedavg",
        },
        "drift": {
            "enabled": True,
            "drift_type": "class_swap",
            "drift_round": drift_round,
            "drift_fraction": 0.3,
            "severity": "medium",
        },
        "selection": selection_dict,
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
    return cfg


def run_pilot_suite(
    methods: tuple[str, ...] = ("random", "utility_greedy", "fixed_exploration", "fedqual_cpx"),
    num_rounds: int = 25,
    num_clients: int = 25,
    clients_per_round: int = 5,
    drift_round: int = 12,
    seed: int = 42,
) -> None:
    print("=" * 70)
    print("FedQual-CPX Pilot Comparison Suite")
    print(f"Methods: {methods}")
    print(f"N={num_clients}, K={clients_per_round}, T={num_rounds}, tau={drift_round}, Seed={seed}")
    print("=" * 70)

    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    suite_results = []

    for method in methods:
        exp_id = f"pilot_{method}_seed{seed}"
        print(f"\n>>> Running Condition: {method.upper()} ({exp_id}) <<<")

        cfg = get_base_experiment_config(
            method=method,
            exp_id=exp_id,
            num_clients=num_clients,
            clients_per_round=clients_per_round,
            num_rounds=num_rounds,
            drift_round=drift_round,
            seed=seed,
        )

        sim = FederatedSimulator(cfg, experiment_id=exp_id)
        summary = sim.run()

        row = {
            "method": method,
            "best_accuracy": summary["best_accuracy"],
            "final_accuracy": summary["final_accuracy"],
            "total_time_seconds": summary["total_time_seconds"],
            "gini": summary["participation"]["gini"],
            "entropy": summary["participation"]["entropy"],
            "coverage": summary["participation"]["coverage"],
        }
        suite_results.append(row)

    # Save summary table
    summary_path = tables_dir / "pilot_comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(suite_results, f, indent=2)

    csv_path = tables_dir / "pilot_comparison_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(suite_results[0].keys()))
        writer.writeheader()
        writer.writerows(suite_results)

    print("\n" + "=" * 70)
    print("Pilot Suite Complete! Summary:")
    print(f"{'Method':<20} | {'Best Acc':<10} | {'Final Acc':<10} | {'Gini':<8} | {'Coverage':<8}")
    print("-" * 70)
    for r in suite_results:
        print(f"{r['method']:<20} | {r['best_accuracy']:<10.4f} | {r['final_accuracy']:<10.4f} | {r['gini']:<8.4f} | {r['coverage']:<8.4f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--clients", type=int, default=25)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--drift-round", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_pilot_suite(
        num_rounds=args.rounds,
        num_clients=args.clients,
        clients_per_round=args.k,
        drift_round=args.drift_round,
        seed=args.seed,
    )
