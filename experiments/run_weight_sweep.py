"""
Exploration Weight Sweep and Guaranteed Priority Experiment (Reviewer #6).

Investigates whether detector inertness is an artifact of explore weighting (w_c)
or an inescapable consequence of partial observability (rho = 0.10).

Configurations:
- change_explore_weight in {0.20, 0.50, 1.00, 2.00, 5.00}
- Guaranteed Priority: Reserves exploration slot for any change-flagged client regardless of score

Outputs:
- results/raw/weight_sweep_*/
- results/tables/weight_sweep_summary.json
- results/tables/weight_sweep_summary.csv
- results/figures/fig_weight_sweep.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.fl.simulator import FederatedSimulator
from src.utils.config import Config
from src.evaluation.statistical_tests import compute_bootstrap_ci


def build_weight_config(
    seed: int,
    change_weight: float = 0.20,
    guaranteed_priority: bool = False,
    num_rounds: int = 100,
    test_every: int = 5,
) -> Config:
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
            "clients_per_round": 10,
            "aggregation": "fedavg",
        },
        "drift": {
            "enabled": True,
            "drift_type": "class_swap",
            "drift_round": 50,
            "drift_fraction": 0.3,
            "severity": "medium",
        },
        "selection": {
            "method": "fedqual_cpx",
            "detector_name": "cusum",
            "change_explore_weight": change_weight,
            "guaranteed_change_priority": guaranteed_priority,
        },
        "normalization": {
            "method": "robust_mad",
        },
        "evaluation": {
            "test_every": test_every,
        },
    })


def compute_recovery_accuracy(global_metrics_csv: Path, drift_round: int = 50, window: int = 10) -> float:
    """Calculate mean test accuracy in rounds [drift_round, drift_round + window]."""
    if not global_metrics_csv.exists():
        return 0.0
    accs = []
    with open(global_metrics_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = int(row["round"])
            if drift_round <= r <= drift_round + window and row.get("test_accuracy"):
                accs.append(float(row["test_accuracy"]))
    return float(np.mean(accs)) * 100.0 if accs else 0.0


def run_weight_sweep(
    weights: list[float] | None = None,
    include_guaranteed: bool = True,
    seeds: list[int] | None = None,
    num_rounds: int = 100,
    test_every: int = 5,
    output_dir: str = "results/tables",
) -> list[dict[str, Any]]:
    if weights is None:
        weights = [0.20, 0.50, 1.00, 2.00, 5.00]
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    conditions: list[tuple[str, float, bool]] = []
    for w in weights:
        conditions.append((f"w_change={w:.2f}", w, False))
    if include_guaranteed:
        conditions.append(("Guaranteed Priority", 1.00, True))

    print("=" * 80)
    print("FedQual-CPX: Exploration Weight Sweep (Reviewer #6)")
    print(f"Conditions: {[c[0] for c in conditions]}")
    print(f"Seeds: {seeds} | Rounds: {num_rounds} | K=10, N=100")
    print("=" * 80)

    summary_rows = []

    for cond_name, w_val, guar in conditions:
        print(f"\n>>> Running Condition: {cond_name} <<<")
        final_accs = []
        recovery_accs = []
        ginis = []
        coverages = []

        for seed in seeds:
            tag = "guaranteed" if guar else f"w{w_val:.2f}"
            exp_id = f"weight_sweep_{tag}_seed{seed}"
            exp_dir = Path("results/raw") / exp_id
            sum_file = exp_dir / "summary.json"
            metrics_file = exp_dir / "global_metrics.csv"

            if sum_file.exists():
                print(f"  [Seed {seed}] Found cached results at {sum_file}, skipping.")
                with open(sum_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                final_acc = float(summary.get("final_accuracy", 0.0)) * 100.0
                rec_acc = compute_recovery_accuracy(metrics_file)
                part = summary.get("participation", {})
                gini = float(part.get("gini", 0.0))
                cov = float(part.get("coverage", 0.0))
            else:
                print(f"  [Seed {seed}] Running simulation {exp_id} ...")
                cfg = build_weight_config(
                    seed=seed,
                    change_weight=w_val,
                    guaranteed_priority=guar,
                    num_rounds=num_rounds,
                    test_every=test_every,
                )
                sim = FederatedSimulator(cfg, experiment_id=exp_id)
                summary = sim.run()
                final_acc = float(summary.get("final_accuracy", 0.0)) * 100.0
                rec_acc = compute_recovery_accuracy(metrics_file)
                part = summary.get("participation", {})
                gini = float(part.get("gini", 0.0))
                cov = float(part.get("coverage", 0.0))

            final_accs.append(final_acc)
            recovery_accs.append(rec_acc)
            ginis.append(gini)
            coverages.append(cov * 100.0 if cov <= 1.0 else cov)
            print(f"    Seed {seed}: Final = {final_acc:.2f}%, Recovery = {rec_acc:.2f}%, Gini = {gini:.4f}")

        f_mean, f_l, f_h = compute_bootstrap_ci(final_accs)
        r_mean, r_l, r_h = compute_bootstrap_ci(recovery_accs)
        g_mean = float(np.mean(ginis))
        c_mean = float(np.mean(coverages))

        row = {
            "condition": cond_name,
            "change_weight": w_val,
            "guaranteed_priority": guar,
            "num_seeds": len(seeds),
            "final_acc_mean": round(f_mean, 2),
            "final_acc_ci": f"[{f_l:.2f}, {f_h:.2f}]",
            "recovery_acc_mean": round(r_mean, 2),
            "recovery_acc_ci": f"[{r_l:.2f}, {r_h:.2f}]",
            "gini_mean": round(g_mean, 4),
            "coverage_mean": round(c_mean, 1),
        }
        summary_rows.append(row)
        print(f"  => {cond_name}: Final = {f_mean:.2f}% [{f_l:.2f}, {f_h:.2f}], Recovery = {r_mean:.2f}% [{r_l:.2f}, {r_h:.2f}]")

    # Save outputs
    json_file = output_path / "weight_sweep_summary.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    csv_file = output_path / "weight_sweep_summary.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved weight sweep results to {json_file} and {csv_file}")
    return summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run exploration weight sweep.")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.20, 0.50, 1.00, 2.00, 5.00])
    parser.add_argument("--no-guaranteed", action="store_true", help="Skip guaranteed priority condition")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--test-every", type=int, default=5)
    args = parser.parse_args()

    run_weight_sweep(
        weights=args.weights,
        include_guaranteed=not args.no_guaranteed,
        seeds=args.seeds,
        num_rounds=args.num_rounds,
        test_every=args.test_every,
    )
