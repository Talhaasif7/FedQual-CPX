"""
Phase 16: Publication Figure Generator for FedQual-CPX.

Generates 300-DPI high-quality publication vector/raster figures per Section 40:
    Figure 1: System pipeline & sequential CUSUM change detection concept
    Figure 2: Sequential detector delay vs shift magnitude Delta
    Figure 3: Global test accuracy learning curves under concept drift
    Figure 4: Post-drift recovery performance
    Figure 5: Client participation distribution / coverage comparison
    Figure 6: Fairness Gini coefficient comparison
    Figure 7: Dynamic exploration probability (epsilon_t) trajectory
    Figure 8: Component-wise ablation study breakdown

Output directory:
    results/figures/
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


# Publication styling configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

COLORS = {
    "random": "#7f7f7f",            # Gray
    "utility_greedy": "#d62728",    # Red
    "sliding_window": "#9467bd",    # Purple
    "fixed_exploration": "#ff7f0e",# Orange
    "page_hinckley_adaptive": "#2ca02c", # Green
    "fedqual_cpx": "#1f77b4",       # Deep Blue (Proposed)
    "cusum": "#1f77b4",
    "page_hinckley": "#2ca02c",
    "ewma": "#8c564b",
}

LABELS = {
    "random": "Random / FedAvg (B0)",
    "utility_greedy": "Utility Greedy (B2)",
    "sliding_window": "Sliding Window (B3)",
    "fixed_exploration": "Fixed Exploration (B4)",
    "page_hinckley_adaptive": "Page-Hinckley (FLEX)",
    "fedqual_cpx": "FedQual-CPX (B8 - Proposed)",
    "cusum": "CUSUM Detector",
    "page_hinckley": "Page-Hinckley Detector",
    "ewma": "EWMA Detector",
}


def generate_figure1_detector_concept(output_dir: Path) -> None:
    """Figure 1: Synthetic single-client utility trajectory + CUSUM detector signal."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    np.random.seed(42)
    rounds = np.arange(1, 51)
    
    # Utility trajectory with abrupt jump at round 25
    u_true = np.concatenate([np.random.normal(0.4, 0.05, 24), np.random.normal(0.8, 0.05, 26)])
    
    # CUSUM statistic S_t^+
    s_pos = np.zeros(50)
    mu_0 = 0.4
    delta = 0.3
    h = 2.5
    for t in range(1, 50):
        s_pos[t] = max(0.0, s_pos[t-1] + (u_true[t] - mu_0 - delta / 2))

    # Top plot: Client Utility
    ax1.plot(rounds, u_true, color="#1f77b4", label="Observed Utility $U_{i,t}$", alpha=0.8)
    ax1.axvline(x=25, color="#d62728", linestyle="--", label="True Concept Shift ($\\tau=25$)")
    ax1.axhline(y=0.4, color="gray", linestyle=":", label="Pre-drift Mean $\\mu_0$")
    ax1.set_ylabel("Normalized Utility")
    ax1.set_title("FedQual-CPX Change-Point Detection Concept")
    ax1.legend(loc="upper left")

    # Bottom plot: CUSUM Statistic
    ax2.plot(rounds, s_pos, color="#2ca02c", label="CUSUM Statistic $S_t^+$", linewidth=2.2)
    ax2.axhline(y=h, color="#d62728", linestyle="-.", label=f"Threshold $h={h}$")
    ax2.axvline(x=25, color="#d62728", linestyle="--")
    ax2.set_xlabel("Communication Round ($t$)")
    ax2.set_ylabel("CUSUM Accumulator")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    fig_path = output_dir / "fig1_cusum_concept.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Fig 1] Generated CUSUM concept figure: {fig_path}")


def generate_figure2_detector_delay(benchmark_json: Path, output_dir: Path) -> None:
    """Figure 2: Sequential change detection delay vs shift magnitude Delta."""
    if not benchmark_json.exists():
        fallback = Path("results/raw/synthetic_detector_benchmark/summary.json")
        if fallback.exists():
            benchmark_json = fallback
        else:
            print(f"[Warning] {benchmark_json} not found. Skipping Fig 2.")
            return

    with open(benchmark_json, "r") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    deltas = [0.25, 0.50, 1.00]
    detectors = ["cusum", "page_hinckley", "ewma"]
    x = np.arange(len(deltas))
    width = 0.25

    for idx, det in enumerate(detectors):
        means = []
        stds = []
        for d in deltas:
            match = next((r for r in data if r["detector"] == det and abs(r["delta"] - d) < 1e-3), None)
            if match:
                means.append(match["mean_delay"])
                stds.append(match["std_delay"])
            else:
                means.append(0.0)
                stds.append(0.0)

        offset = (idx - 1) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=LABELS.get(det, det),
            color=COLORS.get(det, "#333333"),
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Small Shift\n($\\Delta=0.25$)", f"Medium Shift\n($\\Delta=0.50$)", f"Large Shift\n($\\Delta=1.00$)"])
    ax.set_ylabel("Mean Detection Delay (Rounds)")
    ax.set_title("Sequential Detector Latency vs. Shift Magnitude (RQ1 & RQ5)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig_path = output_dir / "fig2_detector_delay_benchmark.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Fig 2] Generated detector delay benchmark figure: {fig_path}")


def generate_figure3_learning_curves(results_dir: Path, output_dir: Path) -> None:
    """Figure 3: Global Test Accuracy Learning Curves."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    is_main_scale = (results_dir / "main_class_swap_fedqual_cpx_seed42" / "global_metrics.csv").exists()

    if is_main_scale:
        conditions = [
            ("random", "main_class_swap_random_seed42"),
            ("utility_greedy", "main_class_swap_utility_greedy_seed42"),
            ("sliding_window", "main_class_swap_sliding_window_seed42"),
            ("fixed_exploration", "main_class_swap_fixed_exploration_seed42"),
            ("fedqual_cpx", "main_class_swap_fedqual_cpx_seed42"),
        ]
        drift_round = 50
        title_suffix = "100-Round Main Scale Multi-Seed Suite (N=100, K=10)"
    else:
        conditions = [
            ("random", "pilot_random_seed42"),
            ("utility_greedy", "pilot_utility_greedy_seed42"),
            ("fixed_exploration", "pilot_fixed_exploration_seed42"),
            ("fedqual_cpx", "pilot_fedqual_cpx_seed42"),
        ]
        drift_round = 15
        title_suffix = "Pilot Benchmark (N=20, K=5)"

    for key, folder in conditions:
        csv_path = results_dir / folder / "global_metrics.csv"
        if not csv_path.exists():
            continue

        rounds = []
        accs = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("test_accuracy"):
                    rounds.append(int(row["round"]))
                    accs.append(float(row["test_accuracy"]) * 100)

        marker = "o" if key == "fedqual_cpx" else ("s" if key == "fixed_exploration" else ("^" if key == "sliding_window" else "v"))
        markevery = 5 if len(rounds) > 40 else 2
        ax.plot(
            rounds,
            accs,
            label=LABELS.get(key, key),
            color=COLORS.get(key, "#333333"),
            marker=marker,
            markevery=markevery,
            linewidth=2.4 if key == "fedqual_cpx" else 1.8,
        )

    ax.axvline(x=drift_round, color="#d62728", linestyle="--", label=f"Drift Onset ($\\tau={drift_round}$)")
    ax.set_xlabel("Communication Round ($t$)")
    ax.set_ylabel("Global Test Accuracy (%)")
    ax.set_title(f"Global Model Convergence under Concept Drift\n{title_suffix}")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig_path = output_dir / "fig3_learning_curves.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Fig 3] Generated learning curves figure: {fig_path}")


def generate_figure4_fairness_gini(multi_seed_json: Path, output_dir: Path) -> None:
    """Figure 4: Client Participation Fairness (Gini Index) Comparison."""
    if not multi_seed_json.exists():
        fallback = Path("results/tables/main_experiments_summary_class_swap.json")
        if fallback.exists():
            multi_seed_json = fallback
        else:
            fallback2 = Path("results/tables/multi_seed_summary.json")
            if fallback2.exists():
                multi_seed_json = fallback2
            else:
                print(f"[Warning] Fairness summary json not found. Skipping Fig 4.")
                return

    with open(multi_seed_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    methods = [r["method"] for r in data]
    ginis = [r.get("gini_mean", r.get("gini", 0.0)) for r in data]
    labels = [r.get("description", LABELS.get(m, m)) for m, r in zip(methods, data)]
    colors = [COLORS.get(m, "#333333") for m in methods]

    bars = ax.bar(range(len(methods)), ginis, color=colors, edgecolor="black", width=0.55, alpha=0.85)

    for bar, val in zip(bars, ginis):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Participation Gini Index (Lower = Fairer)")
    ax.set_title("Client Selection Fairness Comparison (RQ8)")
    ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    fig_path = output_dir / "fig4_fairness_gini.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Fig 4] Generated Gini index fairness figure: {fig_path}")


def generate_figure5_ablation(ablation_json: Path, output_dir: Path) -> None:
    """Figure 5: Component-wise Ablation Study Comparison."""
    if not ablation_json.exists():
        fallback = Path("results/tables/ablation_comprehensive_summary.json")
        if fallback.exists():
            ablation_json = fallback
        else:
            fallback2 = Path("results/tables/ablation_summary.json")
            if fallback2.exists():
                ablation_json = fallback2
            else:
                print(f"[Warning] {ablation_json} not found. Skipping Fig 5.")
                return

    with open(ablation_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(8.5, 5))

    variants = [r.get("variant", r.get("key", "")) for r in data]
    accs = [float(r.get("final_accuracy", 0.0)) * 100 for r in data]
    labels = [r.get("description", v) for v, r in zip(variants, data)]

    bars = ax.barh(range(len(variants)), accs, color="#1f77b4", edgecolor="black", height=0.55, alpha=0.85)

    for bar, val in zip(bars, accs):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.2f}%",
            ha="left",
            va="center",
            fontsize=9.0,
            fontweight="bold",
        )

    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Final Test Accuracy (%)")
    ax.set_title("Component-Wise Ablation Breakdown (RQ4 & RQ6)")
    ax.set_xlim(25, max(accs) + 5 if accs else 40)
    ax.invert_yaxis()

    plt.tight_layout()
    fig_path = output_dir / "fig5_ablation_breakdown.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Fig 5] Generated ablation breakdown figure: {fig_path}")


def generate_figure6_multi_drift_comparison(tables_dir: Path, output_dir: Path) -> None:
    """Figure 6: Multi-Drift Modality Comparison (Abrupt vs. Feature Shift vs. Gradual Drift)."""
    drift_files = [
        ("Abrupt Swap", tables_dir / "main_experiments_summary_class_swap.json"),
        ("Feature Shift", tables_dir / "main_experiments_summary_feature_shift.json"),
        ("Gradual Drift", tables_dir / "main_experiments_summary_gradual_drift.json"),
    ]

    if not all(p.exists() for _, p in drift_files):
        print("[Warning] One or more multi-drift summary json files missing. Skipping Fig 6.")
        return

    methods = ["random", "utility_greedy", "fixed_exploration", "fedqual_cpx"]
    drift_labels = [label for label, _ in drift_files]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    x = np.arange(len(drift_labels))
    width = 0.18

    for idx, m in enumerate(methods):
        accs = []
        ginis = []
        for _, path in drift_files:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entry = next((r for r in data if r["method"] == m), None)
            if entry:
                accs.append(entry["final_acc_mean"])
                ginis.append(entry["gini_mean"])
            else:
                accs.append(0.0)
                ginis.append(0.0)

        offset = (idx - 1.5) * width
        ax1.bar(
            x + offset,
            accs,
            width,
            label=LABELS.get(m, m),
            color=COLORS.get(m, "#333333"),
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
        )
        ax2.bar(
            x + offset,
            ginis,
            width,
            label=LABELS.get(m, m),
            color=COLORS.get(m, "#333333"),
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(drift_labels)
    ax1.set_ylabel("Final Test Accuracy (%)")
    ax1.set_title("(a) Accuracy across Drift Regimes")
    ax1.legend(loc="lower right")
    ax1.set_ylim(25, 42)

    ax2.set_xticks(x)
    ax2.set_xticklabels(drift_labels)
    ax2.set_ylabel("Participation Gini Index (Lower = Fairer)")
    ax2.set_title("(b) Client Starvation & Inequality (Gini)")
    ax2.legend(loc="upper left")
    ax2.set_ylim(0.0, 1.05)

    plt.tight_layout()
    fig_path = output_dir / "fig6_multi_drift_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Fig 6] Generated multi-drift comparison figure: {fig_path}")


def generate_figure7_leaf_benchmarks(tables_dir: Path, output_dir: Path) -> None:
    """Figure 7: Cross-Dataset LEAF Benchmark (FEMNIST & Shakespeare)."""
    femnist_json = tables_dir / "cross_dataset_summary_femnist.json"
    shakespeare_json = tables_dir / "cross_dataset_summary_shakespeare.json"

    if not femnist_json.exists() or not shakespeare_json.exists():
        print("[Warning] FEMNIST or Shakespeare summary json missing. Skipping Fig 7.")
        return

    with open(femnist_json, "r", encoding="utf-8") as f:
        femnist_data = json.load(f)
    with open(shakespeare_json, "r", encoding="utf-8") as f:
        shakespeare_data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    methods = ["random", "utility_greedy", "fixed_exploration", "fedqual_cpx"]
    labels = [LABELS.get(m, m) for m in methods]
    colors = [COLORS.get(m, "#333333") for m in methods]

    # FEMNIST Accuracy
    femnist_accs = []
    for m in methods:
        entry = next((r for r in femnist_data if r.get("method_key") == m or r.get("method") == m), None)
        femnist_accs.append(entry["final_acc_mean"] if entry else 0.0)

    bars1 = ax1.bar(range(len(methods)), femnist_accs, color=colors, edgecolor="black", width=0.55, alpha=0.85)
    for bar, val in zip(bars1, femnist_accs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.3,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Final Test Accuracy (%)")
    ax1.set_title("(a) EMNIST-ByClass (62-Class Vision CNN)")
    ax1.set_ylim(65, 80)

    # Shakespeare Accuracy
    shakespeare_accs = []
    for m in methods:
        entry = next((r for r in shakespeare_data if r.get("method_key") == m or r.get("method") == m), None)
        shakespeare_accs.append(entry["final_acc_mean"] if entry else 0.0)

    bars2 = ax2.bar(range(len(methods)), shakespeare_accs, color=colors, edgecolor="black", width=0.55, alpha=0.85)
    for bar, val in zip(bars2, shakespeare_accs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Top-1 Character Accuracy (%)")
    ax2.set_title("(b) Shakespeare (Recurrent LSTM, Excluded)")
    ax2.set_ylim(0.5, 1.45)

    plt.tight_layout()
    fig_path = output_dir / "fig7_leaf_benchmarks.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Fig 7] Generated cross-dataset benchmarks figure: {fig_path}")


def generate_all_paper_figures() -> None:
    import shutil
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = Path("paper/figures")
    paper_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results/raw")
    tables_dir = Path("results/tables")

    print("=" * 80)
    print("Generating Publication-Ready Figures for FedQual-CPX (Phase 16)")
    print("=" * 80)

    generate_figure1_detector_concept(output_dir)
    generate_figure2_detector_delay(tables_dir / "synthetic_detector_benchmark.json", output_dir)
    generate_figure3_learning_curves(results_dir, output_dir)
    generate_figure4_fairness_gini(tables_dir / "main_experiments_summary_class_swap.json", output_dir)
    generate_figure5_ablation(tables_dir / "ablation_comprehensive_summary.json", output_dir)
    generate_figure6_multi_drift_comparison(tables_dir, output_dir)
    generate_figure7_leaf_benchmarks(tables_dir, output_dir)

    for png in output_dir.glob("*.png"):
        shutil.copy2(png, paper_dir / png.name)
    print(f"[Sync] Copied all publication figures to {paper_dir}/")

    print("=" * 80)
    print(f"All figures generated successfully in {output_dir}/ and {paper_dir}/")


if __name__ == "__main__":
    generate_all_paper_figures()
