"""
Section 36 & 37: Multi-Seed Statistical Validation Suite for FedQual-CPX.

Executes 5 independent random seeds across the core client selection baselines:
    - B0: Random / FedAvg
    - B2: Utility Greedy
    - B4: Fixed Exploration (eps=0.15)
    - B8: FedQual-CPX (Proposed)

Computes:
    - Mean ± Standard Deviation for Final Test Accuracy, Gini, Coverage
    - Non-parametric 95% Bootstrap Confidence Intervals (Section 37)
    - Paired Wilcoxon Signed-Rank Tests against FedQual-CPX (Section 36)
    - Paired Student's t-Tests and Cohen's d effect sizes (Section 36)

Saves consolidated tables:
    - results/tables/multi_seed_summary.json & .csv
    - results/tables/statistical_significance_tests.json & .csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fl.simulator import FederatedSimulator
from src.utils.config import Config
from src.evaluation.statistical_tests import compare_paired_methods, compute_bootstrap_ci


METHOD_CONFIGS = {
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
        "selection": {"method": "fixed_exploration", "epsilon": 0.15, "window_size": 5},
        "normalization": {"method": "none"},
    },
    "fedqual_cpx": {
        "description": "FedQual-CPX (B8 - Proposed)",
        "selection": {
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
        },
        "normalization": {"method": "robust_mad", "z_max": 3.0, "epsilon": 1e-8},
    },
}


def build_experiment_config(
    method_key: str,
    seed: int,
    num_clients: int = 20,
    clients_per_round: int = 5,
    num_rounds: int = 15,
    drift_round: int = 8,
) -> Config:
    info = METHOD_CONFIGS[method_key]
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


def run_multi_seed_suite(
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    methods: Sequence[str] = ("random", "utility_greedy", "fixed_exploration", "fedqual_cpx"),
    num_rounds: int = 15,
    num_clients: int = 20,
    clients_per_round: int = 5,
    drift_round: int = 8,
) -> None:
    print("=" * 70)
    print("FedQual-CPX Multi-Seed Statistical Validation Suite (Sections 36 & 37)")
    print(f"Seeds: {list(seeds)}")
    print(f"Methods: {list(methods)}")
    print(f"N={num_clients}, K={clients_per_round}, T={num_rounds}, tau={drift_round}")
    print("=" * 70)

    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary: method -> list of run summaries across seeds
    runs_by_method: dict[str, list[dict]] = {m: [] for m in methods}

    for seed in seeds:
        print(f"\n--- SEED {seed} ---")
        for m in methods:
            exp_id = f"multiseed_{m}_seed{seed}"

            # Check if this run exists in pilot or multiseed directory
            alt_exp_id = f"pilot_{m}_seed{seed}"
            summary_path = Path("results/raw") / exp_id / "summary.json"
            alt_path = Path("results/raw") / alt_exp_id / "summary.json"

            if summary_path.exists():
                print(f"  [CACHED] {exp_id} -> loading {summary_path}")
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            elif alt_path.exists():
                print(f"  [CACHED] {exp_id} -> found existing pilot run {alt_path}")
                with open(alt_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            else:
                print(f"  [RUNNING] {exp_id} ...")
                cfg = build_experiment_config(
                    method_key=m,
                    seed=seed,
                    num_clients=num_clients,
                    clients_per_round=clients_per_round,
                    num_rounds=num_rounds,
                    drift_round=drift_round,
                )
                sim = FederatedSimulator(cfg, experiment_id=exp_id)
                summary = sim.run()

            runs_by_method[m].append({
                "seed": seed,
                "best_accuracy": summary["best_accuracy"],
                "final_accuracy": summary["final_accuracy"],
                "gini": summary["participation"]["gini"],
                "coverage": summary["participation"]["coverage"],
                "total_time_seconds": summary["total_time_seconds"],
            })

    # Compile Method Aggregates (Mean ± Std, 95% Bootstrap CIs)
    summary_table = []
    for m in methods:
        finals = [r["final_accuracy"] * 100 for r in runs_by_method[m]]
        ginis = [r["gini"] for r in runs_by_method[m]]
        covs = [r["coverage"] * 100 for r in runs_by_method[m]]

        mean_f, low_f, high_f = compute_bootstrap_ci(finals)
        mean_g, low_g, high_g = compute_bootstrap_ci(ginis)
        mean_c, low_c, high_c = compute_bootstrap_ci(covs)

        summary_table.append({
            "method": m,
            "description": METHOD_CONFIGS[m]["description"],
            "num_seeds": len(seeds),
            "final_acc_mean": round(mean_f, 2),
            "final_acc_ci_95": f"[{low_f:.2f}, {high_f:.2f}]",
            "gini_mean": round(mean_g, 4),
            "gini_ci_95": f"[{low_g:.4f}, {high_g:.4f}]",
            "coverage_mean": round(mean_c, 1),
            "coverage_ci_95": f"[{low_c:.1f}, {high_c:.1f}]",
        })

    # Save summary tables
    json_path = tables_dir / "multi_seed_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_table, f, indent=2)

    csv_path = tables_dir / "multi_seed_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_table[0].keys()))
        writer.writeheader()
        writer.writerows(summary_table)

    # Perform Paired Hypothesis Tests against FedQual-CPX (Section 36)
    fedqual_finals = [r["final_accuracy"] for r in runs_by_method["fedqual_cpx"]]
    test_results = []
    for m in methods:
        if m == "fedqual_cpx":
            continue
        base_finals = [r["final_accuracy"] for r in runs_by_method[m]]
        res = compare_paired_methods(
            fedqual_finals,
            base_finals,
            method_a_name="FedQual-CPX",
            method_b_name=METHOD_CONFIGS[m]["description"],
        )
        test_results.append(res)

    tests_json_path = tables_dir / "statistical_significance_tests.json"
    with open(tests_json_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    tests_csv_path = tables_dir / "statistical_significance_tests.csv"
    with open(tests_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(test_results[0].keys()))
        writer.writeheader()
        writer.writerows(test_results)

    print("\n" + "=" * 70)
    print("Multi-Seed Aggregate Results Table:")
    print(f"{'Method':<20} | {'Final Acc (Mean [95% CI])':<26} | {'Gini (Mean [95% CI])':<24} | {'Coverage (%)':<12}")
    print("-" * 70)
    for r in summary_table:
        acc_str = f"{r['final_acc_mean']:.2f}% {r['final_acc_ci_95']}"
        gini_str = f"{r['gini_mean']:.4f} {r['gini_ci_95']}"
        cov_str = f"{r['coverage_mean']:.1f}%"
        print(f"{r['method']:<20} | {acc_str:<26} | {gini_str:<24} | {cov_str:<12}")

    print("\n" + "=" * 70)
    print("Paired Statistical Hypothesis Tests (Section 36):")
    for t in test_results:
        sig_str = "YES (p < 0.05)" if t["statistically_significant_05"] else "NO (p >= 0.05)"
        print(f"FedQual-CPX vs. {t['method_b']}:")
        print(f"  Mean Difference: {t['mean_difference']:+.4f} | Cohen's d: {t['cohens_d']:.3f}")
        print(f"  Wilcoxon Signed-Rank stat: {t['wilcoxon_stat']:.3f} (p = {t['wilcoxon_p_value']:.4f}) | Significant: {sig_str}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--methods", nargs="+", default=["random", "utility_greedy", "fixed_exploration", "fedqual_cpx"])
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--drift-round", type=int, default=8)
    args = parser.parse_args()

    run_multi_seed_suite(
        seeds=args.seeds,
        methods=args.methods,
        num_rounds=args.rounds,
        num_clients=args.clients,
        clients_per_round=args.k,
        drift_round=args.drift_round,
    )
