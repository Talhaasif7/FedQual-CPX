"""
Sanity Reference Experiment for CIFAR-10 (Reviewer #7).

Evaluates standard FedAvg on near-IID data (Dirichlet alpha = 100.0) without concept drift:
- N = 100 total clients, K = 10 selected per round (rho = 0.10)
- T = 100 rounds
- Near-IID distribution (alpha = 100.0)
- No drift injected (drift.enabled = False)
- 3 random seeds: [42, 43, 44]
- Evaluated every 5 rounds

Outputs:
- results/raw/sanity_iid_seed<S>/
- results/tables/sanity_reference_summary.csv and .json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.fl.simulator import FederatedSimulator
from src.utils.config import Config
from src.evaluation.statistical_tests import compute_bootstrap_ci


def build_sanity_config(seed: int, num_rounds: int = 100, test_every: int = 5) -> Config:
    """Build configuration for near-IID, drift-free FedAvg on CIFAR-10."""
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
            "alpha": 100.0,
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
            "clients_per_round": 10,
            "aggregation": "fedavg",
        },
        "drift": {
            "enabled": False,
        },
        "evaluation": {
            "test_every": test_every,
        },
        "selection": {
            "method": "random",
        },
        "normalization": {
            "method": "none",
        },
    })


def run_sanity_experiments(
    seeds: list[int] | None = None,
    num_rounds: int = 100,
    test_every: int = 5,
    output_dir: str = "results/tables",
) -> list[dict[str, Any]]:
    if seeds is None:
        seeds = [42, 43, 44]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FedQual-CPX: CIFAR-10 Sanity Reference Experiment (Near-IID, No Drift)")
    print(f"alpha: 100.0 | Drift: None | Rounds: {num_rounds} | K=10, N=100 | Seeds: {seeds}")
    print("=" * 80)

    final_accs = []
    best_accs = []
    ginis = []
    coverages = []
    seed_details = []

    for seed in seeds:
        exp_id = f"sanity_iid_seed{seed}"
        exp_dir = Path("results/raw") / exp_id
        sum_file = exp_dir / "summary.json"

        if sum_file.exists():
            print(f"\n[Seed {seed}] Found cached results at {sum_file}, skipping execution.")
            with open(sum_file, "r", encoding="utf-8") as f:
                summary = json.load(f)
            final_acc = float(summary["final_accuracy"])
            best_acc = float(summary["best_accuracy"])
            part = summary.get("participation", {})
            gini = float(part.get("gini", 0.0))
            cov = float(part.get("coverage", 0.0))
        else:
            print(f"\n[Seed {seed}] Starting simulation: {exp_id}")
            cfg = build_sanity_config(seed, num_rounds=num_rounds, test_every=test_every)
            sim = FederatedSimulator(cfg, experiment_id=exp_id)
            summary = sim.run()

            final_acc = float(summary.get("final_accuracy", 0.0))
            best_acc = float(summary.get("best_accuracy", 0.0))
            part = summary.get("participation", {})
            gini = float(part.get("gini", 0.0))
            cov = float(part.get("coverage", 0.0))

        final_accs.append(final_acc * 100.0)
        best_accs.append(best_acc * 100.0)
        ginis.append(gini)
        coverages.append(cov * 100.0 if cov <= 1.0 else cov)

        seed_details.append({
            "seed": seed,
            "final_accuracy": round(final_acc * 100.0, 2),
            "best_accuracy": round(best_acc * 100.0, 2),
            "gini": round(gini, 4),
            "coverage": round(cov * 100.0 if cov <= 1.0 else cov, 1),
        })

        print(f"  --> Seed {seed}: Final Acc = {final_acc:.2%}, Best Acc = {best_acc:.2%}, Gini = {gini:.4f}, Cov = {cov:.1%}")

    f_mean, f_ci_l, f_ci_h = compute_bootstrap_ci(final_accs)
    b_mean, b_ci_l, b_ci_h = compute_bootstrap_ci(best_accs)
    g_mean = float(np.mean(ginis))
    c_mean = float(np.mean(coverages))

    summary_result = {
        "condition": "CIFAR-10 Near-IID (alpha=100.0, No Drift)",
        "method": "Random / FedAvg",
        "num_rounds": num_rounds,
        "num_seeds": len(seeds),
        "final_acc_mean": round(f_mean, 2),
        "final_acc_ci_95": f"[{f_ci_l:.2f}, {f_ci_h:.2f}]",
        "best_acc_mean": round(b_mean, 2),
        "best_acc_ci_95": f"[{b_ci_l:.2f}, {b_ci_h:.2f}]",
        "gini_mean": round(g_mean, 4),
        "coverage_mean": round(c_mean, 1),
        "seed_details": seed_details,
    }

    print("\n" + "=" * 80)
    print(f"Sanity Reference Results: Final Acc = {f_mean:.2f}% [{f_ci_l:.2f}, {f_ci_h:.2f}] | Best Acc = {b_mean:.2f}% [{b_ci_l:.2f}, {b_ci_h:.2f}]")
    print("=" * 80)

    # Save to JSON and CSV
    json_path = output_path / "sanity_reference_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_result, f, indent=2)

    csv_path = output_path / "sanity_reference_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition", "method", "num_rounds", "num_seeds",
                "final_acc_mean", "final_acc_ci_95", "best_acc_mean", "best_acc_ci_95",
                "gini_mean", "coverage_mean"
            ]
        )
        writer.writeheader()
        row = {k: v for k, v in summary_result.items() if k != "seed_details"}
        writer.writerow(row)

    print(f"Saved sanity reference results to {json_path} and {csv_path}")
    return [summary_result]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CIFAR-10 sanity reference.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds")
    parser.add_argument("--num-rounds", type=int, default=100, help="Number of FL rounds")
    parser.add_argument("--test-every", type=int, default=5, help="Test frequency")
    args = parser.parse_args()

    run_sanity_experiments(seeds=args.seeds, num_rounds=args.num_rounds, test_every=args.test_every)
