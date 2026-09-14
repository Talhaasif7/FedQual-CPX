"""
FedQual-CPX: High-Performance GPU Benchmark Suite for Google Colab / Kaggle.

Executes the high-impact experimental runs on GPU:
  1. Main 7-baseline benchmark (CIFAR-10 class swap, 100 rounds)
  2. 10-seed statistical evaluation suite (Wilcoxon & Cohen's d)
  3. Participation threshold sweep (rho in {0.05, 0.10, 0.25, 0.36, 0.50})
  4. Exploration weight sweep (w_c in {0.2, 0.5, 1.0, 2.0, 5.0} + guaranteed priority)

Usage:
  python experiments/run_colab_suite.py --all
  python experiments/run_colab_suite.py --task main
  python experiments/run_colab_suite.py --task threshold
  python experiments/run_colab_suite.py --task weights
  python experiments/run_colab_suite.py --task multiseed
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def check_gpu() -> None:
    print("=" * 70)
    print("Hardware & Runtime Environment Check")
    print("=" * 70)
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
        print(f"Available Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("[WARNING] CUDA is NOT available. Simulation will run on CPU.")
    print("=" * 70)


def run_command(cmd: list[str]) -> None:
    print(f"\n[EXEC] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with code {result.returncode}")
        sys.exit(result.returncode)


def prepare_data() -> None:
    print("\n[Step 1/2] Ensuring CIFAR-10 data is processed...")
    train_pt = Path("data/processed/cifar10/train.pt")
    if not train_pt.exists():
        run_command([sys.executable, "scripts/download_data.py", "--dataset", "cifar10"])
    else:
        print("  -> CIFAR-10 data already processed.")


def run_main_benchmark() -> None:
    print("\n" + "=" * 70)
    print("Running Main 7-Baseline Benchmark (CIFAR-10, 100 Rounds)")
    print("=" * 70)
    cmd = [
        sys.executable,
        "experiments/run_main_experiments.py",
        "--drift-type", "class_swap",
    ]
    run_command(cmd)


def run_multiseed_evaluation() -> None:
    print("\n" + "=" * 70)
    print("Running 10-Seed Statistical Validation Suite (Wilcoxon & Cohen's d)")
    print("=" * 70)
    seeds = [str(s) for s in range(42, 52)]
    cmd = [
        sys.executable,
        "experiments/run_multi_seed_evaluation.py",
        "--seeds", *seeds,
    ]
    run_command(cmd)


def run_threshold_sweep() -> None:
    print("\n" + "=" * 70)
    print("Running Participation Threshold Sweep (rho in {0.05, 0.10, 0.25, 0.36, 0.50})")
    print("=" * 70)
    cmd = [
        sys.executable,
        "experiments/run_kn_sweep.py",
        "--k-list", "5", "10", "25", "36", "50",
        "--seeds", "42", "43", "44",
        "--num-rounds", "100",
        "--test-every", "5",
    ]
    run_command(cmd)


def run_weight_sweep() -> None:
    print("\n" + "=" * 70)
    print("Running Exploration Weight & Guaranteed Priority Sweep")
    print("=" * 70)
    cmd = [
        sys.executable,
        "experiments/run_weight_sweep.py",
        "--weights", "0.20", "0.50", "1.00", "2.00", "5.00",
        "--seeds", "42", "43", "44", "45", "46",
        "--num-rounds", "100",
    ]
    run_command(cmd)


def package_results() -> None:
    print("\n" + "=" * 70)
    print("Packaging Results for Download")
    print("=" * 70)
    zip_path = Path("fedqual_cpx_colab_results.zip")
    if zip_path.exists():
        zip_path.unlink()

    shutil.make_archive("fedqual_cpx_colab_results", "zip", "results")
    print(f"  -> Created archive: {zip_path.resolve()} ({zip_path.stat().st_size / (1024**2):.2f} MB)")
    print("  -> Download this file and extract its contents into your local 'results/' folder.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FedQual-CPX GPU Suite on Colab.")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "main", "threshold", "weights", "multiseed"],
        help="Which benchmark task to execute.",
    )
    args = parser.parse_args()

    check_gpu()
    prepare_data()

    if args.task in ("all", "main"):
        run_main_benchmark()
    if args.task in ("all", "multiseed"):
        run_multiseed_evaluation()
    if args.task in ("all", "threshold"):
        run_threshold_sweep()
    if args.task in ("all", "weights"):
        run_weight_sweep()

    package_results()
    print("\n[COMPLETE] All requested benchmark runs finished successfully!")


if __name__ == "__main__":
    main()
