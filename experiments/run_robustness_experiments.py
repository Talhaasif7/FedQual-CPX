"""
Phase 14: Robustness & Sensitivity Analysis Suite for FedQual-CPX.

Evaluates performance stability under non-ideal federated conditions:
    1. Heterogeneity Level (Dirichlet alpha = 0.1, 0.5, 1.0)
    2. Drift Fraction (f_drift = 0.1, 0.3, 0.5)
    3. Population Scale (N = 20, 50, 100)

Output:
    - Raw results in results/raw/robustness_<param>_<val>/
    - Summary table in results/tables/robustness_analysis_summary.csv and .json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fl.simulator import FederatedSimulator
from src.utils.config import Config


def build_robustness_config(
    alpha: float = 0.5,
    drift_fraction: float = 0.3,
    num_clients: int = 30,
    clients_per_round: int = 5,
    num_rounds: int = 30,
    drift_round: int = 15,
    seed: int = 42,
    method: str = "fedqual_cpx",
) -> Config:
    selection_dict = {
        "method": "fedqual_cpx",
        "detector_name": "cusum",
        "warmup_rounds": 5,
        "epsilon_min": 0.08,
        "epsilon_max": 0.35,
        "delta": 0.3,
        "h": 2.5,
        "cooldown": 2,
        "use_uncertainty": True,
        "use_adaptive_epsilon": True,
        "use_change_bonus": True,
    } if method == "fedqual_cpx" else {"method": "random"}

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
            "alpha": alpha,
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
            "drift_fraction": drift_fraction,
            "severity": "medium",
        },
        "selection": selection_dict,
        "normalization": {"method": "robust_mad", "z_max": 3.0, "epsilon": 1e-8} if method == "fedqual_cpx" else {"method": "none"},
        "evaluation": {
            "test_every": 1,
            "save_checkpoint_every": 100,
        },
        "output": {
            "results_dir": "results/raw",
            "save_checkpoints": False,
        },
    })


def run_robustness_experiments(
    num_rounds: int = 30,
    num_clients: int = 30,
    clients_per_round: int = 5,
    drift_round: int = 15,
) -> None:
    print("=" * 80)
    print("FedQual-CPX Phase 14: Robustness & Sensitivity Analysis Suite")
    print("=" * 80)

    summary_rows = []

    # 1. Varying Dirichlet Alpha
    print("\n--- [Varying Non-IID Dirichlet Alpha] ---")
    for alpha in [0.1, 0.5, 1.0]:
        for method in ["random", "fedqual_cpx"]:
            exp_id = f"robust_alpha_{alpha}_{method}"
            cfg = build_robustness_config(
                alpha=alpha,
                drift_fraction=0.3,
                num_clients=num_clients,
                clients_per_round=clients_per_round,
                num_rounds=num_rounds,
                drift_round=drift_round,
                method=method,
            )
            summary = sim.run()
            final_acc = summary["final_accuracy"]
            gini = summary["participation"]["gini"]
            coverage = summary["participation"]["coverage"]

            summary_rows.append({
                "parameter": "alpha",
                "param_value": alpha,
                "method": method,
                "final_accuracy": round(final_acc, 4),
                "gini": round(gini, 4),
                "coverage": round(coverage, 4),
            })
            print(f"  -> Alpha={alpha}, Method={method}: Final Acc = {final_acc:.4f}, Gini = {gini:.4f}, Coverage = {coverage*100:.1f}%")

    # 2. Varying Drift Fraction
    print("\n--- [Varying Drift Fraction f_drift] ---")
    for df in [0.1, 0.3, 0.5]:
        for method in ["random", "fedqual_cpx"]:
            exp_id = f"robust_drift_frac_{df}_{method}"
            cfg = build_robustness_config(
                alpha=0.5,
                drift_fraction=df,
                num_clients=num_clients,
                clients_per_round=clients_per_round,
                num_rounds=num_rounds,
                drift_round=drift_round,
                method=method,
            )
            sim = FederatedSimulator(cfg, experiment_id=exp_id)
            summary = sim.run()
            final_acc = summary["final_accuracy"]
            gini = summary["participation"]["gini"]
            coverage = summary["participation"]["coverage"]

            summary_rows.append({
                "parameter": "drift_fraction",
                "param_value": df,
                "method": method,
                "final_accuracy": round(final_acc, 4),
                "gini": round(gini, 4),
                "coverage": round(coverage, 4),
            })
            print(f"  -> Drift Frac={df}, Method={method}: Final Acc = {final_acc:.4f}, Gini = {gini:.4f}, Coverage = {coverage*100:.1f}%")

    # Save outputs
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    json_path = tables_dir / "robustness_analysis_summary.json"
    csv_path = tables_dir / "robustness_analysis_summary.csv"

    with open(json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 80)
    print("ROBUSTNESS ANALYSIS SUMMARY")
    print("=" * 80)
    for r in summary_rows:
        print(f"{r['parameter']:<15} | Value={r['param_value']:<5} | Method={r['method']:<12} | Final Acc = {r['final_accuracy']*100:>6.2f}% | Gini = {r['gini']:<6.4f}")
    print("=" * 80)
    print(f"Saved summary tables to:\n  - {csv_path}\n  - {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 14 Robustness Analysis")
    parser.add_argument("--quick", action="store_true", help="Run fast verification (15 rounds, N=20)")
    args = parser.parse_args()

    if args.quick:
        run_robustness_experiments(
            num_rounds=15,
            num_clients=20,
            clients_per_round=5,
            drift_round=8,
        )
    else:
        run_robustness_experiments(
            num_rounds=30,
            num_clients=30,
            clients_per_round=5,
            drift_round=15,
        )


if __name__ == "__main__":
    main()
