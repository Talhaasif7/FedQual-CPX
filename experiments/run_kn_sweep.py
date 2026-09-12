"""
Phase 15: Participation Crossover Sweep (K/N Sweep).

Investigates the participation ratio threshold rho = K / N where change-aware
client selection transitions from lagging behind random sampling to outperforming it.

Evaluates K in {5, 10, 25, 50} at N = 100 with T = 100 communication rounds,
drift at tau = 50, across multiple random seeds with 95% bootstrap confidence intervals.

Outputs:
    - Raw results in results/raw/kn_k<K>_<method>_seed<S>/
    - Consolidated tables in results/tables/kn_sweep_summary.csv and .json
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


def build_kn_config(
    k_val: int,
    method_key: str,
    num_rounds: int = 100,
    seed: int = 42,
    test_every: int = 5,
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
            "change_explore_weight": 0.20,
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
        "evaluation": {
            "test_every": test_every,
        },
        "selection": selection_cfg,
        "normalization": norm_cfg,
    })


def run_kn_sweep(
    k_list: list[int] | None = None,
    methods: list[str] | None = None,
    seeds: list[int] | None = None,
    num_rounds: int = 100,
    test_every: int = 5,
    output_dir: str = "results/tables",
) -> list[dict[str, Any]]:
    """Execute the multi-seed participation crossover sweep."""
    if k_list is None:
        k_list = [5, 10, 25, 50]
    if methods is None:
        methods = ["random", "fedqual_cpx"]
    if seeds is None:
        seeds = [42, 43, 44]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FedQual-CPX Phase 15: Multi-Seed Participation Crossover (K/N) Sweep")
    print(f"K values: {k_list} | N: 100 | Rounds: {num_rounds} | Seeds: {seeds}")
    print("=" * 80)

    summary_rows = []

    for k_val in k_list:
        ratio = k_val / 100.0
        for method_key in methods:
            print(f"\n============================================================")
            print(f"Sweep Condition: K={k_val} (rho={ratio:.2f}), Method={method_key}")
            print(f"============================================================")

            final_accs = []
            rec_accs = []
            ginis = []
            coverages = []
            for seed in seeds:
                exp_id = f"kn_k{k_val}_{method_key}_seed{seed}"
                sum_file = Path("results/raw") / exp_id / "summary.json"
                exp_dir = Path("results/raw") / exp_id

                if not sum_file.exists() and k_val == 10:
                    alt_file = Path("results/raw") / f"main_class_swap_{method_key}_seed{seed}" / "summary.json"
                    if alt_file.exists():
                        sum_file = alt_file
                        exp_dir = Path("results/raw") / f"main_class_swap_{method_key}_seed{seed}"

                # Check if already completed
                if sum_file.exists():
                    print(f"Found existing results for {sum_file}, loading cached summary.")
                    with open(sum_file, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    final_acc = summary["final_accuracy"]
                    best_acc = summary["best_accuracy"]
                    fairness = summary["participation"]
                    gini = fairness.get("gini", 0.0)
                    cov = fairness.get("coverage", 0.0)

                    # Try to extract recovery acc from global_metrics.csv
                    gm_file = exp_dir / "global_metrics.csv"
                    if gm_file.exists():
                        with open(gm_file, "r", encoding="utf-8") as gf:
                            rows = list(csv.DictReader(gf))
                        drift_r = num_rounds // 2
                        post = [float(r["test_accuracy"]) for r in rows if r.get("test_accuracy") != "" and int(r["round"]) >= drift_r]
                        recovery_acc = float(np.mean(post)) if post else final_acc
                    else:
                        recovery_acc = final_acc
                else:
                    cfg = build_kn_config(k_val, method_key, num_rounds=num_rounds, seed=seed, test_every=test_every)
                    sim = FederatedSimulator(cfg, experiment_id=exp_id)
                    summary = sim.run()

                    final_acc = float(summary.get("final_accuracy", 0.0))
                    best_acc = float(summary.get("best_accuracy", 0.0))
                    part = summary.get("participation", {})
                    gini = float(part.get("gini", 0.0))
                    cov = float(part.get("coverage", 0.0))

                    gm_file = sim.output_dir / "global_metrics.csv"
                    if gm_file.exists():
                        with open(gm_file, "r", encoding="utf-8") as gf:
                            rows = list(csv.DictReader(gf))
                        drift_r = num_rounds // 2
                        post = [float(r["test_accuracy"]) for r in rows if r.get("test_accuracy") != "" and int(r["round"]) >= drift_r]
                        recovery_acc = float(np.mean(post)) if post else final_acc
                    else:
                        recovery_acc = final_acc

                final_accs.append(final_acc * 100.0)
                rec_accs.append(recovery_acc * 100.0)
                ginis.append(gini)
                coverages.append(cov * 100.0 if cov <= 1.0 else cov)

                print(f"  Seed {seed}: Final Acc={final_acc:.2%}, Rec Acc={recovery_acc:.2%}, Gini={gini:.4f}, Cov={cov:.1%}")

            # Compute bootstrap CIs
            f_mean, f_ci_l, f_ci_h = compute_bootstrap_ci(final_accs)
            r_mean, r_ci_l, r_ci_h = compute_bootstrap_ci(rec_accs)
            g_mean = float(np.mean(ginis))
            c_mean = float(np.mean(coverages))

            summary_rows.append({
                "k": k_val,
                "ratio": ratio,
                "method": method_key,
                "final_acc_mean": round(f_mean, 2),
                "final_acc_ci_95": f"[{f_ci_l:.2f}, {f_ci_h:.2f}]",
                "recovery_acc_mean": round(r_mean, 2),
                "recovery_acc_ci_95": f"[{r_ci_l:.2f}, {r_ci_h:.2f}]",
                "gini_mean": round(g_mean, 4),
                "coverage_mean": round(c_mean, 1),
                "num_seeds": len(seeds),
            })

            print(f"==> K={k_val}, Method={method_key}: Final Acc={f_mean:.2f}% [{f_ci_l:.2f}, {f_ci_h:.2f}], Recovery={r_mean:.2f}% [{r_ci_l:.2f}, {r_ci_h:.2f}], Gini={g_mean:.4f}, Cov={c_mean:.1f}%")

    # Save to CSV and JSON
    csv_path = output_path / "kn_sweep_summary.csv"
    json_path = output_path / "kn_sweep_summary.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "k", "ratio", "method", "final_acc_mean", "final_acc_ci_95",
            "recovery_acc_mean", "recovery_acc_ci_95", "gini_mean", "coverage_mean", "num_seeds"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nSuccessfully wrote K/N sweep summary to {csv_path} and {json_path}")
    return summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run participation crossover sweep.")
    parser.add_argument("--k-list", type=int, nargs="+", default=[5, 10, 25, 50], help="List of K values")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds")
    parser.add_argument("--num-rounds", type=int, default=100, help="Number of FL communication rounds")
    parser.add_argument("--test-every", type=int, default=5, help="Test evaluation frequency")
    args = parser.parse_args()

    run_kn_sweep(
        k_list=args.k_list,
        seeds=args.seeds,
        num_rounds=args.num_rounds,
        test_every=args.test_every,
    )
