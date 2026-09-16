"""
Phase 17: FL Normalization Head-to-Head Comparison Benchmark.

Compares three normalization strategies inside actual federated training under concept drift:
    1. cross_sectional: Robust MAD across contemporaneous batch (FedQual-CPX default)
    2. temporal: Per-client running historical baseline (preserves common-mode shifts)
    3. raw: Pass-through unnormalized loss differences (method='none')

Configuration:
    - CIFAR-10, Dirichlet alpha=0.5, N=100, K=10 (rho=0.10)
    - T=100 rounds, Class Swap concept drift at tau=50
    - Seeds: [42, 43, 44, 45, 46]

Outputs:
    - Raw results: results/raw/fl_norm_<norm_method>_seed<S>/
    - Consolidated summary: results/tables/fl_normalization_comparison.csv and .json
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


NORM_CONDITIONS = {
    "cross_sectional": {
        "description": "Cross-Sectional Robust MAD (Contemporaneous Batch)",
        "normalization": {"method": "robust_mad", "z_max": 3.0, "epsilon": 1e-8},
    },
    "temporal": {
        "description": "Per-Client Temporal Baseline (Historical)",
        "normalization": {"method": "temporal", "z_max": 3.0, "epsilon": 1e-8},
    },
    "raw": {
        "description": "Raw Utility Differences (No Normalization)",
        "normalization": {"method": "none"},
    },
}


def build_norm_config(
    norm_key: str,
    seed: int,
    num_rounds: int = 100,
    drift_round: int = 50,
) -> Config:
    info = NORM_CONDITIONS[norm_key]
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
        "drift": {
            "enabled": True,
            "drift_type": "class_swap",
            "drift_round": drift_round,
            "drift_end_round": drift_round + 20,
            "severity": "medium",
            "drift_fraction": 0.30,
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
            "evaluation_frequency": 1,
            "save_checkpoint_every": 100,
        },
        "normalization": info["normalization"],
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
            "change_explore_weight": 0.20,
            "guaranteed_change_priority": False,
        },
        "output": {
            "results_dir": "results/raw",
            "save_checkpoints": False,
        },
    })


def run_fl_normalization_comparison(
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    num_rounds: int = 100,
    drift_round: int = 50,
) -> list[dict[str, Any]]:
    # Safety Guard
    if num_rounds < 50 or drift_round >= num_rounds:
        raise ValueError(
            f"Safety Guard: Refusing to run FL normalization comparison with num_rounds={num_rounds} (< 50) "
            f"or drift_round={drift_round} (>= {num_rounds})."
        )

    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FedQual-CPX Phase 17: FL Normalization Head-to-Head Comparison")
    print(f"Conditions: {list(NORM_CONDITIONS.keys())} | Seeds: {list(seeds)} | Rounds: {num_rounds}")
    print("=" * 80)

    summary_rows = []

    for norm_key, info in NORM_CONDITIONS.items():
        print(f"\n[EVALUATING CONDITION] {info['description']}")
        final_accs = []
        rec_accs = []
        ginis = []
        coverages = []

        for seed in seeds:
            exp_id = f"fl_norm_{norm_key}_seed{seed}"
            exp_dir = Path("results/raw") / exp_id
            sum_file = exp_dir / "summary.json"

            # Check if cross_sectional can reuse existing baseline cache
            if norm_key == "cross_sectional" and not sum_file.exists():
                alt_dir = Path("results/raw") / f"main_class_swap_fedqual_cpx_seed{seed}"
                if (alt_dir / "summary.json").exists():
                    exp_dir = alt_dir
                    sum_file = alt_dir / "summary.json"

            if sum_file.exists():
                with open(sum_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                if summary.get("total_rounds", 0) >= 50:
                    print(f"  -> Found cached results for {sum_file}, loading...")
                    final_acc = float(summary["final_accuracy"])
                    best_acc = float(summary["best_accuracy"])
                    gini = float(summary["participation"].get("gini", 0.0))
                    cov = float(summary["participation"].get("coverage", 0.0))

                    gm_file = exp_dir / "global_metrics.csv"
                    if gm_file.exists():
                        with open(gm_file, "r", encoding="utf-8") as gf:
                            rows = list(csv.DictReader(gf))
                        post = [float(r["test_accuracy"]) for r in rows if r.get("test_accuracy") != "" and int(r["round"]) >= drift_round]
                        recovery_acc = float(np.mean(post)) if post else final_acc
                    else:
                        recovery_acc = final_acc
                else:
                    print(f"  -> Warning: Cached {sum_file} has only {summary.get('total_rounds', 0)} rounds (< 50). Rerunning fresh...")
                    summary = None
            else:
                summary = None

            if summary is None:
                cfg = build_norm_config(norm_key, seed, num_rounds=num_rounds, drift_round=drift_round)
                sim = FederatedSimulator(cfg, experiment_id=exp_id)
                summary = sim.run()

                final_acc = float(summary["final_accuracy"])
                best_acc = float(summary["best_accuracy"])
                gini = float(summary["participation"].get("gini", 0.0))
                cov = float(summary["participation"].get("coverage", 0.0))

                gm_file = sim.output_dir / "global_metrics.csv"
                if gm_file.exists():
                    with open(gm_file, "r", encoding="utf-8") as gf:
                        rows = list(csv.DictReader(gf))
                    post = [float(r["test_accuracy"]) for r in rows if r.get("test_accuracy") != "" and int(r["round"]) >= drift_round]
                    recovery_acc = float(np.mean(post)) if post else final_acc
                else:
                    recovery_acc = final_acc

            final_accs.append(final_acc * 100.0)
            rec_accs.append(recovery_acc * 100.0)
            ginis.append(gini)
            coverages.append(cov * 100.0 if cov <= 1.0 else cov)

            print(f"  Seed {seed}: Final Acc={final_acc:.2%}, Rec Acc={recovery_acc:.2%}, Gini={gini:.4f}, Cov={cov:.1%}")

        f_mean, f_ci_l, f_ci_h = compute_bootstrap_ci(final_accs)
        r_mean, r_ci_l, r_ci_h = compute_bootstrap_ci(rec_accs)
        g_mean = float(np.mean(ginis))
        c_mean = float(np.mean(coverages))

        summary_rows.append({
            "normalization_strategy": norm_key,
            "description": info["description"],
            "num_seeds": len(seeds),
            "final_acc_mean": round(f_mean, 2),
            "final_acc_ci_95": f"[{f_ci_l:.2f}, {f_ci_h:.2f}]",
            "recovery_acc_mean": round(r_mean, 2),
            "recovery_acc_ci_95": f"[{r_ci_l:.2f}, {r_ci_h:.2f}]",
            "gini_mean": round(g_mean, 4),
            "coverage_mean": round(c_mean, 1),
        })

    # Save to CSV and JSON
    csv_path = tables_dir / "fl_normalization_comparison.csv"
    json_path = tables_dir / "fl_normalization_comparison.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print("\n" + "=" * 80)
    print("FL NORMALIZATION HEAD-TO-HEAD COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Strategy':<18} | {'Final Acc (95% CI)':<22} | {'Recovery Acc (95% CI)':<22} | {'Gini':<8} | {'Coverage':<8}")
    print("-" * 90)
    for r in summary_rows:
        print(
            f"{r['normalization_strategy']:<18} | {r['final_acc_mean']:>5.2f}% {r['final_acc_ci_95']:<15} | "
            f"{r['recovery_acc_mean']:>5.2f}% {r['recovery_acc_ci_95']:<15} | {r['gini_mean']:<8.4f} | {r['coverage_mean']:<7.1f}%"
        )
    print("=" * 80)
    print(f"Saved summary tables to:\n  - {csv_path}\n  - {json_path}")
    return summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FL normalization head-to-head comparison.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="Random seeds")
    parser.add_argument("--num-rounds", type=int, default=100, help="Number of rounds (must be >= 50)")
    parser.add_argument("--drift-round", type=int, default=50, help="Round at which drift begins")
    args = parser.parse_args()

    run_fl_normalization_comparison(
        seeds=args.seeds,
        num_rounds=args.num_rounds,
        drift_round=args.drift_round,
    )
