"""
Phase 13: Comprehensive Ablation Matrix Suite for FedQual-CPX.

Systematically isolates and evaluates each core building block:
    1. Detector Choice: CUSUM (Proposed) vs Page-Hinckley vs EWMA vs No Detector
    2. Normalization Scheme: Robust MAD (Proposed) vs Z-Score vs Min-Max vs None
    3. Adaptive Exploration Components: Full vs No Uncertainty vs No Change Bonus vs Fixed Epsilon

Outputs:
    - Results in results/raw/<ablation_id>/
    - Consolidated tables in results/tables/ablation_comprehensive_summary.csv and .json
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


ABLATION_CONDITIONS: dict[str, dict[str, Any]] = {
    # ── Group 1: Detector Choice ──
    "A1_cusum_full": {
        "group": "Detector",
        "description": "Full FedQual-CPX (CUSUM Detector + Robust MAD + Adaptive Exploration)",
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
    "A2_page_hinckley": {
        "group": "Detector",
        "description": "Page-Hinckley Detector (FLEX Detector + Robust MAD)",
        "selection": {
            "method": "page_hinckley_adaptive",
            "detector_name": "page_hinckley",
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
    "A3_ewma_detector": {
        "group": "Detector",
        "description": "EWMA Detector + Robust MAD",
        "selection": {
            "method": "fedqual_cpx",
            "detector_name": "ewma",
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
    "A4_no_detector": {
        "group": "Detector",
        "description": "No Detector (Adaptive Exploration on raw utility without change events)",
        "selection": {
            "method": "fedqual_cpx",
            "detector_name": "cusum",
            "warmup_rounds": 5,
            "epsilon_min": 0.08,
            "epsilon_max": 0.35,
            "use_uncertainty": True,
            "use_adaptive_epsilon": False,
            "use_change_bonus": False,
        },
        "normalization": {"method": "robust_mad", "z_max": 3.0, "epsilon": 1e-8},
    },

    # ── Group 2: Normalization Scheme ──
    "B1_robust_mad": {
        "group": "Normalization",
        "description": "Robust MAD Normalization (Proposed)",
        "selection": {
            "method": "fedqual_cpx",
            "detector_name": "cusum",
            "warmup_rounds": 5,
            "epsilon_min": 0.08,
            "epsilon_max": 0.35,
            "use_uncertainty": True,
            "use_adaptive_epsilon": True,
            "use_change_bonus": True,
        },
        "normalization": {"method": "robust_mad", "z_max": 3.0, "epsilon": 1e-8},
    },
    "B2_zscore_norm": {
        "group": "Normalization",
        "description": "Standard Z-Score Normalization",
        "selection": {
            "method": "fedqual_cpx",
            "detector_name": "cusum",
            "warmup_rounds": 5,
            "epsilon_min": 0.08,
            "epsilon_max": 0.35,
            "use_uncertainty": True,
            "use_adaptive_epsilon": True,
            "use_change_bonus": True,
        },
        "normalization": {"method": "zscore", "z_max": 3.0, "epsilon": 1e-8},
    },
    "B3_minmax_norm": {
        "group": "Normalization",
        "description": "Min-Max Normalization",
        "selection": {
            "method": "fedqual_cpx",
            "detector_name": "cusum",
            "warmup_rounds": 5,
            "epsilon_min": 0.08,
            "epsilon_max": 0.35,
            "use_uncertainty": True,
            "use_adaptive_epsilon": True,
            "use_change_bonus": True,
        },
        "normalization": {"method": "minmax", "epsilon": 1e-8},
    },
    "B4_no_norm": {
        "group": "Normalization",
        "description": "No Normalization (Raw Utilities)",
        "selection": {
            "method": "fedqual_cpx",
            "detector_name": "cusum",
            "warmup_rounds": 5,
            "epsilon_min": 0.08,
            "epsilon_max": 0.35,
            "use_uncertainty": True,
            "use_adaptive_epsilon": True,
            "use_change_bonus": True,
        },
        "normalization": {"method": "none"},
    },

    # ── Group 3: Adaptive Exploration Terms ──
    "C1_no_uncertainty": {
        "group": "Exploration",
        "description": "No Uncertainty Bonus (Uncertainty term disabled)",
        "selection": {
            "method": "fedqual_cpx",
            "detector_name": "cusum",
            "warmup_rounds": 5,
            "epsilon_min": 0.08,
            "epsilon_max": 0.35,
            "use_uncertainty": False,
            "use_adaptive_epsilon": True,
            "use_change_bonus": True,
        },
        "normalization": {"method": "robust_mad", "z_max": 3.0, "epsilon": 1e-8},
    },
    "C2_no_change_bonus": {
        "group": "Exploration",
        "description": "No Change Bonus (Change detection bonus disabled)",
        "selection": {
            "method": "fedqual_cpx",
            "detector_name": "cusum",
            "warmup_rounds": 5,
            "epsilon_min": 0.08,
            "epsilon_max": 0.35,
            "use_uncertainty": True,
            "use_adaptive_epsilon": True,
            "use_change_bonus": False,
        },
        "normalization": {"method": "robust_mad", "z_max": 3.0, "epsilon": 1e-8},
    },
}


def build_ablation_config(
    ablation_key: str,
    seed: int = 42,
    num_clients: int = 30,
    clients_per_round: int = 5,
    num_rounds: int = 30,
    drift_round: int = 15,
) -> Config:
    info = ABLATION_CONDITIONS[ablation_key]
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


def run_comprehensive_ablations(
    keys: Sequence[str] | None = None,
    seed: int = 42,
    num_rounds: int = 30,
    num_clients: int = 30,
    clients_per_round: int = 5,
    drift_round: int = 15,
) -> None:
    if keys is None:
        keys = list(ABLATION_CONDITIONS.keys())

    print("=" * 80)
    print("FedQual-CPX Phase 13: Comprehensive Ablation Matrix")
    print(f"Conditions: {len(keys)} | Seed: {seed} | N={num_clients}, K={clients_per_round}, T={num_rounds}")
    print("=" * 80)

    summary_rows = []
    for key in keys:
        info = ABLATION_CONDITIONS[key]
        print(f"\n[RUNNING ABLATION {key}] {info['description']}")
        cfg = build_ablation_config(
            ablation_key=key,
            seed=seed,
            num_clients=num_clients,
            clients_per_round=clients_per_round,
            num_rounds=num_rounds,
            drift_round=drift_round,
        )
        sim = FederatedSimulator(cfg, experiment_id=f"ablation_{key}_seed{seed}")
        summary = sim.run()

        final_acc = summary["final_accuracy"]
        best_acc = summary["best_accuracy"]

        fairness = summary["participation"]
        gini = fairness.get("gini", 0.0)
        entropy = fairness.get("entropy", 0.0)
        coverage = fairness.get("coverage", 0.0)

        summary_rows.append({
            "key": key,
            "group": info["group"],
            "description": info["description"],
            "best_accuracy": round(best_acc, 4),
            "final_accuracy": round(final_acc, 4),
            "gini": round(gini, 4),
            "entropy": round(entropy, 4),
            "coverage": round(coverage, 4),
        })

        print(
            f"  -> Best Acc = {best_acc:.4f} | Final Acc = {final_acc:.4f} | "
            f"Gini = {gini:.4f} | Coverage = {coverage*100:.1f}%"
        )

    # Save outputs
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    json_path = tables_dir / "ablation_comprehensive_summary.json"
    csv_path = tables_dir / "ablation_comprehensive_summary.csv"

    with open(json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 80)
    print("COMPREHENSIVE ABLATION SUMMARY")
    print("=" * 80)
    print(f"{'Key':<20} | {'Group':<12} | {'Final Acc':<10} | {'Gini':<8} | {'Coverage':<8}")
    print("-" * 65)
    for r in summary_rows:
        print(f"{r['key']:<20} | {r['group']:<12} | {r['final_accuracy']*100:>8.2f}% | {r['gini']:<8.4f} | {r['coverage']*100:<7.1f}%")
    print("=" * 80)
    print(f"Saved summary tables to:\n  - {csv_path}\n  - {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 13 Comprehensive Ablations")
    parser.add_argument("--quick", action="store_true", help="Run fast verification (20 rounds, N=20)")
    args = parser.parse_args()

    if args.quick:
        run_comprehensive_ablations(
            num_rounds=15,
            num_clients=20,
            clients_per_round=5,
            drift_round=8,
        )
    else:
        run_comprehensive_ablations(
            num_rounds=30,
            num_clients=30,
            clients_per_round=5,
            drift_round=15,
        )


if __name__ == "__main__":
    main()
