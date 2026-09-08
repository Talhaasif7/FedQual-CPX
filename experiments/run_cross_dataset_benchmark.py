"""
Cross-Dataset Validation Suite for FedQual-CPX on LEAF Benchmarks (Phase 17).

Evaluates FedQual-CPX and key baselines on:
    1. FEMNIST (62 classes, FEMNISTCNN, 28x28 grayscale)
    2. Shakespeare (90 character vocabulary, ShakespeareLSTM)

Usage:
    python experiments/run_cross_dataset_benchmark.py --dataset femnist --quick
    python experiments/run_cross_dataset_benchmark.py --dataset femnist
    python experiments/run_cross_dataset_benchmark.py --dataset shakespeare --quick
    python experiments/run_cross_dataset_benchmark.py --dataset shakespeare
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.fl.simulator import FederatedSimulator
from src.utils.config import Config
from src.evaluation.statistical_tests import compute_bootstrap_ci


CROSS_DATASET_METHODS: dict[str, dict[str, Any]] = {
    "random": {
        "description": "Random / FedAvg (B0)",
        "selection": {"method": "random"},
        "normalization": {"method": "none"},
    },
    "utility_greedy": {
        "description": "Utility Greedy (B2)",
        "selection": {"method": "utility_greedy"},
        "normalization": {"method": "none"},
    },
    "fixed_exploration": {
        "description": "Fixed Exploration (B4, eps=0.15)",
        "selection": {"method": "fixed_exploration", "epsilon": 0.15, "window_size": 10},
        "normalization": {"method": "none"},
    },
    "fedqual_cpx": {
        "description": "FedQual-CPX (B8, Proposed)",
        "selection": {
            "method": "fedqual_cpx",
            "detector_name": "cusum",
            "warmup_rounds": 10,
            "epsilon_min": 0.05,
            "epsilon_max": 0.30,
            "delta": 0.30,
            "h": 2.5,
            "cooldown": 2,
            "use_uncertainty": True,
            "use_adaptive_epsilon": True,
            "use_change_bonus": True,
        },
        "normalization": {
            "method": "robust_mad",
            "z_max": 3.0,
            "epsilon": 1e-8,
        },
    },
}


def build_cross_dataset_config(
    dataset: str,
    method_key: str,
    seed: int,
    num_clients: int = 100,
    clients_per_round: int = 10,
    num_rounds: int = 100,
    drift_round: int = 50,
) -> Config:
    info = CROSS_DATASET_METHODS[method_key]
    dataset = dataset.lower()

    if dataset == "femnist":
        ds_cfg = {
            "name": "FEMNIST",
            "processed_dir": "data/processed/femnist",
            "num_classes": 62,
            "image_shape": [28, 28, 1],
        }
        model_cfg = {
            "name": "FEMNISTCNN",
            "num_classes": 62,
        }
        train_cfg = {
            "local_epochs": 1,
            "batch_size": 32,
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0001,
        }
    elif dataset == "shakespeare":
        ds_cfg = {
            "name": "Shakespeare",
            "processed_dir": "data/processed/shakespeare",
            "sequence_length": 80,
            "vocab_size": 90,
            "num_classes": 90,
        }
        model_cfg = {
            "name": "ShakespeareLSTM",
            "num_classes": 90,
        }
        train_cfg = {
            "local_epochs": 1,
            "batch_size": 32,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return Config({
        "seed": seed,
        "dataset": ds_cfg,
        "partition": {
            "method": "dirichlet",
            "alpha": 0.5,
            "num_clients": num_clients,
            "seed": seed,
            "min_samples_per_client": 10,
            "val_fraction": 0.1,
        },
        "model": model_cfg,
        "training": train_cfg,
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
        "selection": info["selection"],
        "normalization": info["normalization"],
        "evaluation": {
            "test_every": 1,
            "save_checkpoint_every": 100,
        },
        "output": {
            "results_dir": "results/raw",
            "save_checkpoints": False,
        },
    })


def run_cross_dataset_benchmark(
    dataset: str = "femnist",
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    methods: Sequence[str] = ("random", "utility_greedy", "fixed_exploration", "fedqual_cpx"),
    num_rounds: int = 100,
    num_clients: int = 100,
    clients_per_round: int = 10,
    drift_round: int = 50,
) -> None:
    print("=" * 80)
    print(f"FedQual-CPX Phase 17: Cross-Dataset Benchmark ({dataset.upper()})")
    print(f"Seeds: {list(seeds)} | Clients N={num_clients}, K={clients_per_round}, T={num_rounds}, tau={drift_round}")
    print("=" * 80)

    results_by_method: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}

    for method in methods:
        print(f"\n[RUNNING METHOD] {CROSS_DATASET_METHODS[method]['description']}")
        for seed in seeds:
            exp_id = f"cross_{dataset}_{method}_seed{seed}"
            exp_dir = Path("results/raw") / exp_id
            eval_history: list[dict[str, Any]] = []

            if (exp_dir / "summary.json").exists() and (exp_dir / "global_metrics.csv").exists():
                print(f"  -> Found cached results for {exp_id}, loading...")
                with open(exp_dir / "summary.json", "r", encoding="utf-8") as f:
                    summary = json.load(f)
                with open(exp_dir / "global_metrics.csv", "r", encoding="utf-8") as f:
                    eval_history = list(csv.DictReader(f))
            else:
                cfg = build_cross_dataset_config(
                    dataset=dataset,
                    method_key=method,
                    seed=seed,
                    num_clients=num_clients,
                    clients_per_round=clients_per_round,
                    num_rounds=num_rounds,
                    drift_round=drift_round,
                )
                sim = FederatedSimulator(cfg, experiment_id=exp_id)
                summary = sim.run()

                if sim.logger and getattr(sim.logger, "round_history", None):
                    eval_history = sim.logger.round_history
                elif sim.logger and getattr(sim.logger, "global_rows", None):
                    eval_history = sim.logger.global_rows
                elif (sim.output_dir / "global_metrics.csv").exists():
                    with open(sim.output_dir / "global_metrics.csv", "r", encoding="utf-8") as f:
                        eval_history = list(csv.DictReader(f))

            final_acc = summary["final_accuracy"]
            best_acc = summary["best_accuracy"]

            post_drift_accs = [
                float(e["test_accuracy"])
                for e in eval_history
                if e.get("test_accuracy") != "" and drift_round <= int(e["round"]) <= drift_round + 10
            ]
            recovery_acc = float(sum(post_drift_accs) / len(post_drift_accs)) if post_drift_accs else final_acc

            fairness = summary["participation"]
            gini = fairness.get("gini", 0.0)
            entropy = fairness.get("entropy", 0.0)
            coverage = fairness.get("coverage", 0.0)

            run_result = {
                "seed": seed,
                "final_accuracy": final_acc,
                "best_accuracy": best_acc,
                "recovery_accuracy": recovery_acc,
                "gini": gini,
                "entropy": entropy,
                "coverage": coverage,
            }
            results_by_method[method].append(run_result)
            cov_pct = coverage * 100 if coverage <= 1.0 else coverage
            print(f"  -> Seed {seed}: Final Acc = {final_acc:.4f}, Recovery Acc = {recovery_acc:.4f}, Gini = {gini:.4f}, Coverage = {cov_pct:.1f}%")

    # Aggregate summaries across seeds
    summary_rows = []
    for method in methods:
        runs = results_by_method[method]
        final_accs = [r["final_accuracy"] * 100 for r in runs]
        rec_accs = [r["recovery_accuracy"] * 100 for r in runs]
        ginis = [r["gini"] for r in runs]
        covs = [r["coverage"] * 100 if r["coverage"] <= 1.0 else r["coverage"] for r in runs]

        final_mean, final_ci_l, final_ci_h = compute_bootstrap_ci(final_accs)
        rec_mean, rec_ci_l, rec_ci_h = compute_bootstrap_ci(rec_accs)

        summary_rows.append({
            "dataset": dataset,
            "method": CROSS_DATASET_METHODS[method]["description"],
            "method_key": method,
            "final_acc_mean": round(final_mean, 2),
            "final_acc_ci_95": f"[{final_ci_l:.2f}, {final_ci_h:.2f}]",
            "recovery_acc_mean": round(rec_mean, 2),
            "recovery_acc_ci_95": f"[{rec_ci_l:.2f}, {rec_ci_h:.2f}]",
            "gini_mean": round(float(np.mean(ginis)), 4),
            "coverage_mean": round(float(np.mean(covs)), 1),
            "seeds": [r["seed"] for r in runs],
        })

    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    json_path = tables_dir / f"cross_dataset_summary_{dataset}.json"
    csv_path = tables_dir / f"cross_dataset_summary_{dataset}.csv"

    with open(json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 80)
    print(f"CROSS-DATASET BENCHMARK SUMMARY ({dataset.upper()})")
    print("=" * 80)
    print(f"{'Method':<25} | {'Final Acc (95% CI)':<22} | {'Recovery Acc (95% CI)':<22} | {'Gini':<8} | {'Coverage':<8}")
    print("-" * 90)
    for r in summary_rows:
        print(
            f"{r['method']:<25} | {r['final_acc_mean']:>5.2f}% {r['final_acc_ci_95']:<15} | "
            f"{r['recovery_acc_mean']:>5.2f}% {r['recovery_acc_ci_95']:<15} | {r['gini_mean']:<8.4f} | {r['coverage_mean']:<7.1f}%"
        )
    print("=" * 80)
    print(f"Saved summary tables to:\n  - {csv_path}\n  - {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-Dataset Validation Suite")
    parser.add_argument("--dataset", type=str, default="femnist", choices=["femnist", "shakespeare"])
    parser.add_argument("--quick", action="store_true", help="Run fast verification (2 seeds, 10 rounds, N=20)")
    args = parser.parse_args()

    if args.quick:
        run_cross_dataset_benchmark(
            dataset=args.dataset,
            seeds=(42, 43),
            methods=("random", "utility_greedy", "fedqual_cpx"),
            num_rounds=10,
            num_clients=20,
            clients_per_round=5,
            drift_round=5,
        )
    else:
        run_cross_dataset_benchmark(
            dataset=args.dataset,
            seeds=(42, 43, 44, 45, 46),
            methods=("random", "utility_greedy", "fixed_exploration", "fedqual_cpx"),
            num_rounds=100,
            num_clients=100,
            clients_per_round=10,
            drift_round=50,
        )


if __name__ == "__main__":
    main()
