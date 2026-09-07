"""
Section 31 & 32: Mandatory Ablation Suite for FedQual-CPX.

Isolates the individual contributions of:
    1. Sequential Change Detection (CUSUM vs Page-Hinckley vs None)
    2. Robust MAD Normalization (MAD scaling vs Raw utility)
    3. Adaptive Exploration (Dynamic epsilon_t vs Fixed epsilon)
    4. Epistemic Uncertainty Scoring (Uncertainty bonus vs None)

Ablation Matrix (Section 31):
    A0: Bare Greedy (No Detector, No Adapt, No Normalization, No Uncertainty)
    A1: Fixed Exploration (No Detector, Fixed eps=0.15, No Normalization, No Uncertainty)
    A2: No Detector (No Detector, Adaptive eps_t, Robust MAD, Uncertainty)
    A3: No Normalization (CUSUM, Adaptive eps_t, Raw Utility, Uncertainty)
    A4: Page-Hinckley Competitor (Page-Hinckley, Adaptive eps_t, Robust MAD, Uncertainty)
    A5: Full FedQual-CPX (CUSUM, Adaptive eps_t, Robust MAD, Uncertainty, Change Bonus)

Saves summary tables to results/tables/ablation_summary.json & .csv.
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


ABLATION_CONDITIONS = {
    "A0_bare_greedy": {
        "description": "Bare Greedy (Exploitation only, no normalization)",
        "selection": {"method": "utility_greedy"},
        "normalization": {"method": "none"},
    },
    "A1_fixed_explore": {
        "description": "Fixed Exploration (Static epsilon=0.15, windowed utility)",
        "selection": {"method": "fixed_exploration", "epsilon": 0.15, "window_size": 5},
        "normalization": {"method": "none"},
    },
    "A2_no_detector": {
        "description": "No Detector (Adaptive exploration + MAD normalization, but no detector)",
        "selection": {
            "method": "fedqual_cpx",
            "detector_name": "none",
            "warmup_rounds": 5,
            "epsilon_min": 0.08,
            "epsilon_max": 0.35,
            "use_uncertainty": True,
            "use_adaptive_epsilon": True,
            "use_change_bonus": False,
        },
        "normalization": {"method": "robust_mad", "z_max": 3.0, "epsilon": 1e-8},
    },
    "A3_no_normalization": {
        "description": "No Normalization (CUSUM + Adaptive exploration on raw utilities)",
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
        "normalization": {"method": "none"},
    },
    "A4_page_hinckley": {
        "description": "Page-Hinckley Detector (FLEX detector + adaptive exploration)",
        "selection": {
            "method": "fedqual_cpx",
            "detector_name": "page_hinckley",
            "warmup_rounds": 5,
            "epsilon_min": 0.08,
            "epsilon_max": 0.35,
            "delta": 0.15,
            "h": 6.0,
            "cooldown": 2,
            "use_uncertainty": True,
            "use_adaptive_epsilon": True,
            "use_change_bonus": True,
        },
        "normalization": {"method": "robust_mad", "z_max": 3.0, "epsilon": 1e-8},
    },
    "A5_full_fedqual_cpx": {
        "description": "Full FedQual-CPX (CUSUM + MAD Normalization + Adaptive exploration + Uncertainty + Bonus)",
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


def build_ablation_config(
    condition_id: str,
    num_clients: int = 20,
    clients_per_round: int = 5,
    num_rounds: int = 15,
    drift_round: int = 8,
    seed: int = 42,
) -> Config:
    info = ABLATION_CONDITIONS[condition_id]

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
    return cfg


def run_ablation_suite(
    conditions: Sequence[str] = tuple(ABLATION_CONDITIONS.keys()),
    num_rounds: int = 15,
    num_clients: int = 20,
    clients_per_round: int = 5,
    drift_round: int = 8,
    seed: int = 42,
) -> None:
    print("=" * 70)
    print("FedQual-CPX Mandatory Ablation Suite (Sections 31 & 32)")
    print(f"Conditions: {conditions}")
    print(f"N={num_clients}, K={clients_per_round}, T={num_rounds}, tau={drift_round}, Seed={seed}")
    print("=" * 70)

    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for cond_id in conditions:
        exp_id = f"ablation_{cond_id}_seed{seed}"
        print(f"\n>>> Running Ablation: {cond_id.upper()} ({exp_id}) <<<")
        desc = ABLATION_CONDITIONS[cond_id]["description"]
        print(f"    Description: {desc}")

        summary_path = Path("results/raw") / exp_id / "summary.json"
        if summary_path.exists():
            print(f"    [CACHED] Found existing completed run for {exp_id}, loading summary...")
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        else:
            cfg = build_ablation_config(
                condition_id=cond_id,
                num_clients=num_clients,
                clients_per_round=clients_per_round,
                num_rounds=num_rounds,
                drift_round=drift_round,
                seed=seed,
            )

            sim = FederatedSimulator(cfg, experiment_id=exp_id)
            summary = sim.run()

        row = {
            "variant": cond_id[:2],
            "condition": cond_id,
            "description": desc,
            "best_accuracy": summary["best_accuracy"],
            "final_accuracy": summary["final_accuracy"],
            "total_time_seconds": summary["total_time_seconds"],
            "gini": summary["participation"]["gini"],
            "entropy": summary["participation"]["entropy"],
            "coverage": summary["participation"]["coverage"],
        }
        summary_rows.append(row)

    # Save summary files
    json_path = tables_dir / "ablation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    csv_path = tables_dir / "ablation_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 70)
    print("Ablation Suite Completed! Summary Table (Section 31):")
    print(f"{'Variant':<8} | {'Best Acc':<10} | {'Final Acc':<10} | {'Gini':<8} | {'Coverage':<8} | {'Description'}")
    print("-" * 70)
    for r in summary_rows:
        print(f"{r['variant']:<8} | {r['best_accuracy']:<10.4f} | {r['final_accuracy']:<10.4f} | {r['gini']:<8.4f} | {r['coverage']:<8.4f} | {r['description']}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=list(ABLATION_CONDITIONS.keys()))
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--drift-round", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_ablation_suite(
        conditions=args.conditions,
        num_rounds=args.rounds,
        num_clients=args.clients,
        clients_per_round=args.k,
        drift_round=args.drift_round,
        seed=args.seed,
    )
