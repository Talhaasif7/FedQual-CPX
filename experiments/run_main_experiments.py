"""
Phase 12: Main Scale Multi-Seed Experiment Suite for FedQual-CPX.

Executes full-scale federated benchmarks across multiple random seeds and dynamic drift conditions:
    - N = 100 total clients, K = 10 selected per round
    - T = 100 rounds
    - Multi-seed validation: 5 random seeds (42, 43, 44, 45, 46)
    - Baseline algorithms:
        1. Random / FedAvg (B0)
        2. Utility Greedy (B2)
        3. Sliding Window (B3, W=10)
        4. Fixed Exploration (B4, eps=0.15)
        5. Page-Hinckley Adaptive (B6, FLEX competitor)
        6. FedQual-CPX (B8, Proposed CUSUM + MAD Norm + Adaptive Exploration)
    - Dynamic Drift Types:
        1. Abrupt Label Swap (class swap at tau=50 on 30% clients)
        2. Feature Shift (Gaussian noise at tau=50 on 30% clients)
        3. Gradual Drift (linear label transition tau in [30, 70])

Output:
    - Raw results in results/raw/<exp_id>/
    - Consolidated tables in results/tables/main_experiments_summary.csv and .json
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
from src.evaluation.statistical_tests import compare_paired_methods, compute_bootstrap_ci


MAIN_METHOD_CONFIGS: dict[str, dict[str, Any]] = {
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
    "sliding_window": {
        "description": "Sliding Window (B3, W=10)",
        "selection": {"method": "sliding_window", "window_size": 10},
        "normalization": {"method": "none"},
    },
    "fixed_exploration": {
        "description": "Fixed Exploration (B4, eps=0.15)",
        "selection": {"method": "fixed_exploration", "epsilon": 0.15, "window_size": 10},
        "normalization": {"method": "none"},
    },
    "page_hinckley_adaptive": {
        "description": "Page-Hinckley Adaptive (B6, FLEX)",
        "selection": {
            "method": "page_hinckley_adaptive",
            "detector_name": "page_hinckley",
            "warmup_rounds": 10,
            "epsilon_min": 0.05,
            "epsilon_max": 0.30,
            "delta": 0.3,
            "h": 2.5,
            "cooldown": 2,
            "use_uncertainty": True,
            "use_adaptive_epsilon": True,
            "use_change_bonus": True,
        },
        "normalization": {"method": "robust_mad", "z_max": 3.0, "epsilon": 1e-8},
    },
    "fedqual_cpx": {
        "description": "FedQual-CPX (B8 - Proposed CUSUM)",
        "selection": {
            "method": "fedqual_cpx",
            "detector_name": "cusum",
            "warmup_rounds": 10,
            "epsilon_min": 0.05,
            "epsilon_max": 0.30,
            "delta": 0.3,
            "h": 2.5,
            "cooldown": 2,
            "use_uncertainty": True,
            "use_adaptive_epsilon": True,
            "use_change_bonus": True,
        },
        "normalization": {"method": "robust_mad", "z_max": 3.0, "epsilon": 1e-8},
    },
}


def build_main_config(
    method_key: str,
    seed: int,
    drift_type: str = "class_swap",
    num_clients: int = 100,
    clients_per_round: int = 10,
    num_rounds: int = 100,
    drift_round: int = 50,
) -> Config:
    """Build standardized Config object for main scale runs."""
    info = MAIN_METHOD_CONFIGS[method_key]
    is_gradual = drift_type in ("gradual", "gradual_drift")
    actual_drift_round = int(round(num_rounds * 0.3)) if is_gradual else drift_round
    actual_drift_end = int(round(num_rounds * 0.7)) if is_gradual else (drift_round + 20)

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
            "drift_type": drift_type,
            "drift_round": actual_drift_round,
            "drift_end_round": actual_drift_end,
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


def run_main_experiments(
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    methods: Sequence[str] = (
        "random",
        "utility_greedy",
        "sliding_window",
        "fixed_exploration",
        "page_hinckley_adaptive",
        "fedqual_cpx",
    ),
    drift_type: str = "class_swap",
    num_rounds: int = 100,
    num_clients: int = 100,
    clients_per_round: int = 10,
    drift_round: int = 50,
) -> None:
    is_gradual = drift_type in ("gradual", "gradual_drift")
    actual_drift_round = int(round(num_rounds * 0.3)) if is_gradual else drift_round
    actual_drift_end = int(round(num_rounds * 0.7)) if is_gradual else (drift_round + 20)

    print("=" * 80)
    print(f"FedQual-CPX Phase 12: Main Scale Multi-Seed Experiment Suite")
    print(f"Drift Type: {drift_type} | Seeds: {list(seeds)}")
    if is_gradual:
        print(f"Clients N={num_clients}, Selection K={clients_per_round}, Rounds T={num_rounds}, tau=[{actual_drift_round}, {actual_drift_end}]")
    else:
        print(f"Clients N={num_clients}, Selection K={clients_per_round}, Rounds T={num_rounds}, tau={actual_drift_round}")
    print("=" * 80)

    results_by_method: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}

    for method in methods:
        print(f"\n[RUNNING METHOD] {MAIN_METHOD_CONFIGS[method]['description']}")
        for seed in seeds:
            exp_id = f"main_{drift_type}_{method}_seed{seed}"
            exp_dir = Path("results/raw") / exp_id
            eval_history: list[dict[str, Any]] = []

            if (exp_dir / "summary.json").exists() and (exp_dir / "global_metrics.csv").exists():
                print(f"  -> Found cached results for {exp_id}, loading...")
                with open(exp_dir / "summary.json", "r", encoding="utf-8") as f:
                    summary = json.load(f)
                with open(exp_dir / "global_metrics.csv", "r", encoding="utf-8") as f:
                    eval_history = list(csv.DictReader(f))
            else:
                cfg = build_main_config(
                    method_key=method,
                    seed=seed,
                    drift_type=drift_type,
                    num_clients=num_clients,
                    clients_per_round=clients_per_round,
                    num_rounds=num_rounds,
                    drift_round=actual_drift_round,
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

            # Recovery accuracy (post-drift window)
            eval_recovery_start = 70 if is_gradual else actual_drift_round
            post_drift_accs = [
                float(e["test_accuracy"])
                for e in eval_history
                if e.get("test_accuracy") != "" and eval_recovery_start <= int(e["round"]) <= eval_recovery_start + 10
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
            print(
                f"  -> Seed {seed}: Final Acc = {final_acc:.4f}, Recovery Acc = {recovery_acc:.4f}, "
                f"Gini = {gini:.4f}, Coverage = {coverage*100:.1f}%"
            )

    # Summarize results
    summary_rows = []
    for method in methods:
        runs = results_by_method[method]
        final_accs = [r["final_accuracy"] for r in runs]
        recovery_accs = [r["recovery_accuracy"] for r in runs]
        ginis = [r["gini"] for r in runs]
        coverages = [r["coverage"] for r in runs]

        final_mean, final_ci_l, final_ci_h = compute_bootstrap_ci(final_accs)
        rec_mean, rec_ci_l, rec_ci_h = compute_bootstrap_ci(recovery_accs)
        gini_mean, gini_ci_l, gini_ci_h = compute_bootstrap_ci(ginis)
        cov_mean, cov_ci_l, cov_ci_h = compute_bootstrap_ci(coverages)

        summary_rows.append({
            "method": method,
            "description": MAIN_METHOD_CONFIGS[method]["description"],
            "drift_type": drift_type,
            "num_seeds": len(seeds),
            "final_acc_mean": round(final_mean * 100, 2),
            "final_acc_ci_95": f"[{final_ci_l*100:.2f}, {final_ci_h*100:.2f}]",
            "recovery_acc_mean": round(rec_mean * 100, 2),
            "recovery_acc_ci_95": f"[{rec_ci_l*100:.2f}, {rec_ci_h*100:.2f}]",
            "gini_mean": round(gini_mean, 4),
            "gini_ci_95": f"[{gini_ci_l:.4f}, {gini_ci_h:.4f}]",
            "coverage_mean": round(cov_mean * 100, 1),
            "coverage_ci_95": f"[{cov_ci_l*100:.1f}, {cov_ci_h*100:.1f}]",
        })

    # Save outputs
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    json_path = tables_dir / f"main_experiments_summary_{drift_type}.json"
    csv_path = tables_dir / f"main_experiments_summary_{drift_type}.csv"

    with open(json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 80)
    print(f"MAIN EXPERIMENTAL SUMMARY ({drift_type})")
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
    parser = argparse.ArgumentParser(description="Run Phase 12 Main Scale Experiments")
    parser.add_argument("--quick", action="store_true", help="Run fast verification (2 seeds, 20 rounds, N=20)")
    parser.add_argument("--drift-type", type=str, default="class_swap", choices=["class_swap", "feature_shift", "gradual_drift"])
    args = parser.parse_args()

    if args.quick:
        run_main_experiments(
            seeds=(42, 43),
            methods=("random", "utility_greedy", "fixed_exploration", "fedqual_cpx"),
            drift_type=args.drift_type,
            num_rounds=20,
            num_clients=20,
            clients_per_round=5,
            drift_round=10,
        )
    else:
        run_main_experiments(
            seeds=(42, 43, 44, 45, 46),
            methods=(
                "random",
                "utility_greedy",
                "sliding_window",
                "fixed_exploration",
                "page_hinckley_adaptive",
                "fedqual_cpx",
            ),
            drift_type=args.drift_type,
            num_rounds=100,
            num_clients=100,
            clients_per_round=10,
            drift_round=50,
        )


if __name__ == "__main__":
    main()
