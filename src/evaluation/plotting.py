"""
Publication figure generator for FedQual-CPX.

Implements standard scientific visualization per Section 40 of the implementation plan:
    Figure 1: Global Test Accuracy curves under concept drift (tau = 8)
    Figure 2: Sequential change detection delay vs shift magnitude Delta
    Figure 3: Client participation distribution and starvation comparison
    Figure 4: Participation Gini coefficient fairness comparison
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


# Academic styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

COLORS = {
    "random": "#7f7f7f",            # Gray
    "utility_greedy": "#d62728",    # Red
    "fixed_exploration": "#ff7f0e",# Orange
    "fedqual_cpx": "#1f77b4",       # Deep Blue (Ours)
    "cusum": "#1f77b4",
    "page_hinckley": "#2ca02c",     # Green
    "ewma": "#9467bd",              # Purple
}

LABELS = {
    "random": "Random / FedAvg (B0)",
    "utility_greedy": "Utility Greedy (B2)",
    "fixed_exploration": "Fixed Exploration (B4, eps=0.15)",
    "fedqual_cpx": "FedQual-CPX (B8 - Proposed)",
    "cusum": "CUSUM (FedQual-CPX)",
    "page_hinckley": "Page-Hinckley (FLEX)",
    "ewma": "EWMA Filter",
}


def plot_learning_curves(
    results_dir: Path, output_path: Path, drift_round: int = 8
) -> None:
    """Figure 1: Test accuracy curves across rounds under drift."""
    fig, ax = plt.subplots(figsize=(8, 5))

    conditions = [
        ("random", "pilot_random_seed42"),
        ("utility_greedy", "pilot_utility_greedy_seed42"),
        ("fixed_exploration", "pilot_fixed_exploration_seed42"),
        ("fedqual_cpx", "pilot_fedqual_cpx_seed42"),
    ]

    for key, folder in conditions:
        csv_path = results_dir / folder / "global_metrics.csv"
        if not csv_path.exists():
            continue

        rounds = []
        accuracies = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["test_accuracy"]:
                    rounds.append(int(row["round"]))
                    accuracies.append(float(row["test_accuracy"]) * 100)

        marker = "o" if key == "fedqual_cpx" else ("s" if key == "fixed_exploration" else "v")
        ax.plot(
            rounds,
            accuracies,
            label=LABELS.get(key, key),
            color=COLORS.get(key, "#333333"),
            marker=marker,
            markersize=5,
            linewidth=2.4 if key == "fedqual_cpx" else 1.8,
        )

    # Vertical line indicating concept drift onset
    ax.axvline(
        x=drift_round,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Concept Drift (tau = {drift_round})",
    )

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Test Accuracy (%)")
    ax.set_title("Test Accuracy Under Non-IID Dynamic Concept Drift (D2 Class Swap)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(1, 15)
    ax.set_ylim(8, 38)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Fig 1] Saved learning curves: {output_path}")


def plot_detector_benchmark(benchmark_summary: Path, output_path: Path) -> None:
    """Figure 2: Sequential change detection delay across shift magnitudes."""
    if not benchmark_summary.exists():
        return

    with open(benchmark_summary, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 4.8))

    deltas = [0.25, 0.50, 1.00]
    detectors = ["cusum", "page_hinckley", "ewma"]

    x = np.arange(len(deltas))
    width = 0.25

    for idx, det in enumerate(detectors):
        means = []
        stds = []
        for d in deltas:
            match = next((row for row in data if row["detector"] == det and abs(row["delta"] - d) < 1e-3), None)
            if match:
                means.append(match["mean_delay"])
                stds.append(match["std_delay"])
            else:
                means.append(0)
                stds.append(0)

        offset = (idx - 1) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=LABELS.get(det, det),
            color=COLORS.get(det, "#555"),
            alpha=0.88,
        )

    ax.set_xlabel("Utility Shift Magnitude (Delta)")
    ax.set_ylabel("Detection Delay (Rounds to Detect)")
    ax.set_title("Sequential Change Detection Delay (Lower is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Delta = {d:.2f}" for d in deltas])
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    ax.set_ylim(1, 300)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Fig 2] Saved detector benchmark: {output_path}")


def plot_participation_distribution(results_dir: Path, output_path: Path) -> None:
    """Figure 3: Client participation counts comparing Greedy starvation vs FedQual-CPX."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    configs = [
        ("utility_greedy", "pilot_utility_greedy_seed42", axes[0], "Utility Greedy (B2) - Starvation"),
        ("fedqual_cpx", "pilot_fedqual_cpx_seed42", axes[1], "FedQual-CPX (B8) - Equitable"),
    ]

    for key, folder, ax, title in configs:
        csv_path = results_dir / folder / "client_events.csv"
        if not csv_path.exists():
            continue

        counts: dict[int, int] = {i: 0 for i in range(20)}
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                c_id = int(row["client_id"])
                counts[c_id] = counts.get(c_id, 0) + 1

        clients = list(counts.keys())
        participations = [counts[c] for c in clients]

        bar_colors = [COLORS.get(key, "#1f77b4") if p > 0 else "#cccccc" for p in participations]
        ax.bar(clients, participations, color=bar_colors, edgecolor="#333333", linewidth=0.8)
        ax.set_xlabel("Client ID")
        ax.set_title(title)
        ax.set_xticks(range(0, 20, 2))
        ax.set_ylim(0, 16)

    axes[0].set_ylabel("Participation Count (out of 15 rounds)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Fig 3] Saved participation histogram: {output_path}")


def plot_fairness_gini(table_summary_path: Path, output_path: Path) -> None:
    """Figure 4: Participation Gini and Coverage bar comparison."""
    if not table_summary_path.exists():
        return

    with open(table_summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    methods = [row["method"] for row in data]
    ginis = [row["gini"] for row in data]
    coverages = [row["coverage"] * 100 for row in data]
    colors = [COLORS.get(m, "#555") for m in methods]
    labels = [m.replace("_", " ").title() for m in methods]

    # Gini plot (lower is better)
    bars1 = ax1.bar(labels, ginis, color=colors, alpha=0.85, edgecolor="#333")
    ax1.set_ylabel("Participation Gini Coefficient (Lower is More Fair)")
    ax1.set_title("Participation Inequality (Gini)")
    ax1.set_ylim(0, 0.9)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontsize=9)

    # Coverage plot (higher is better)
    bars2 = ax2.bar(labels, coverages, color=colors, alpha=0.85, edgecolor="#333")
    ax2.set_ylabel("Client Population Coverage (%)")
    ax2.set_title("Network Exploration Coverage (Higher is Better)")
    ax2.set_ylim(0, 115)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 2, f"{yval:.0f}%", ha="center", va="bottom", fontsize=9)

    for ax in (ax1, ax2):
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Fig 4] Saved fairness comparison: {output_path}")


def plot_ablation_comparison(ablation_summary_path: Path, output_path: Path) -> None:
    """Figure 5: Ablation study metrics comparing A2-A5."""
    if not ablation_summary_path.exists():
        return

    with open(ablation_summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))

    variants = [row["variant"] for row in data]
    labels = [f"{r['variant']}: {r['condition'].replace('ablation_', '').replace('_', ' ').title()}" for r in data]
    accuracies = [row["final_accuracy"] * 100 for row in data]
    ginis = [row["gini"] for row in data]

    variant_colors = ["#7f7f7f", "#d62728", "#2ca02c", "#1f77b4"]

    # Accuracy bar plot
    bars1 = ax1.bar(variants, accuracies, color=variant_colors, alpha=0.85, edgecolor="#333")
    ax1.set_ylabel("Final Test Accuracy (%)")
    ax1.set_title("Model Accuracy Across Ablation Variants")
    ax1.set_ylim(25, 36)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Gini coefficient plot
    bars2 = ax2.bar(variants, ginis, color=variant_colors, alpha=0.85, edgecolor="#333")
    ax2.set_ylabel("Participation Gini (Lower is More Fair)")
    ax2.set_title("Participation Inequality Across Variants")
    ax2.set_ylim(0.18, 0.32)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.005, f"{yval:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for ax in (ax1, ax2):
        ax.set_xlabel("Ablation Variant (Section 31)")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Fig 5] Saved ablation comparison: {output_path}")


def generate_all_figures() -> None:
    results_dir = Path("results/raw")
    figures_dir = Path("results/figures")
    tables_dir = Path("results/tables")

    plot_learning_curves(results_dir, figures_dir / "fig1_learning_curves.png")
    plot_detector_benchmark(
        results_dir / "synthetic_detector_benchmark" / "summary.json",
        figures_dir / "fig2_detector_delay_vs_delta.png",
    )
    plot_participation_distribution(results_dir, figures_dir / "fig3_client_participation_histogram.png")
    plot_fairness_gini(
        tables_dir / "pilot_comparison_summary.json",
        figures_dir / "fig4_fairness_gini_comparison.png",
    )
    plot_ablation_comparison(
        tables_dir / "ablation_summary.json",
        figures_dir / "fig5_ablation_comparison.png",
    )


if __name__ == "__main__":
    generate_all_figures()

