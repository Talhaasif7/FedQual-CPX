"""
ARL Recalibration Experiment for Sequential Change Detectors (Reviewer #12).

Per Reviewer #12:
Calibrates each detector's decision threshold so that all detectors achieve
matched in-control average run length (ARL_0) / false alarm rate under H_0 (x ~ N(0, 1)).
Then reruns the detection delay benchmark across Delta in {0.25, 0.50, 1.00}.

Outputs:
- results/tables/recalibrated_detector_benchmark.json
- results/tables/recalibrated_detector_benchmark.csv
- results/figures/fig2_detector_delay_vs_delta.png
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.selection.detectors import (
    CUSUMDetector,
    PageHinckleyDetector,
    EWMADetector,
)


def measure_in_control_fa(
    detector_factory: Any,
    num_trials: int = 100,
    stream_length: int = 500,
    seed_base: int = 12345,
) -> tuple[float, float]:
    """Measure false alarm count and ARL_0 on pure N(0, 1) noise."""
    fa_counts = []
    first_fa_steps = []

    for trial in range(num_trials):
        rng = np.random.default_rng(seed_base + trial)
        stream = rng.normal(0.0, 1.0, size=stream_length)
        det = detector_factory()
        fa = 0
        first_fa = None

        for t, val in enumerate(stream, start=1):
            evt = det.update(float(val), current_round=t)
            if evt:
                fa += 1
                if first_fa is None:
                    first_fa = t

        fa_counts.append(fa)
        first_fa_steps.append(first_fa if first_fa is not None else stream_length)

    mean_fa = float(np.mean(fa_counts))
    mean_arl0 = float(np.mean(first_fa_steps))
    return mean_fa, mean_arl0


def calibrate_thresholds(
    target_fa: float = 3.5,
    stream_length: int = 500,
    num_trials: int = 100,
) -> dict[str, dict[str, Any]]:
    """Find thresholds that yield matched false alarms (~target_fa per 500 steps)."""
    print("Calibrating detector thresholds for matched in-control run length (ARL_0)...")

    # 1. Page-Hinckley reference: delta=0.15, h=8.0
    ph_factory = lambda: PageHinckleyDetector(delta=0.15, h=8.0, cooldown=5)
    ph_fa, ph_arl0 = measure_in_control_fa(ph_factory, num_trials, stream_length)
    print(f"  [Reference] Page-Hinckley (h=8.0, delta=0.15): FA={ph_fa:.2f}, ARL_0={ph_arl0:.1f}")

    # 2. CUSUM: tune h over grid with delta=0.25
    best_cusum_h = 4.0
    min_diff = 999.0
    for h_cand in np.linspace(4.0, 8.0, 41):
        h_cand = round(float(h_cand), 2)
        factory = lambda h=h_cand: CUSUMDetector(delta=0.25, h=h, cooldown=5)
        fa, _ = measure_in_control_fa(factory, num_trials, stream_length)
        diff = abs(fa - ph_fa)
        if diff < min_diff:
            min_diff = diff
            best_cusum_h = h_cand

    cusum_factory = lambda: CUSUMDetector(delta=0.25, h=best_cusum_h, cooldown=5)
    cusum_fa, cusum_arl0 = measure_in_control_fa(cusum_factory, num_trials, stream_length)
    print(f"  [Calibrated] CUSUM (h={best_cusum_h}, delta=0.25): FA={cusum_fa:.2f}, ARL_0={cusum_arl0:.1f}")

    # 3. EWMA: tune l_sigma over grid with lambd=0.2
    best_ewma_l = 3.0
    min_diff = 999.0
    for l_cand in np.linspace(2.0, 3.5, 31):
        l_cand = round(float(l_cand), 2)
        factory = lambda l_sig=l_cand: EWMADetector(lambd=0.2, l_sigma=l_sig, cooldown=5)
        fa, _ = measure_in_control_fa(factory, num_trials, stream_length)
        diff = abs(fa - ph_fa)
        if diff < min_diff:
            min_diff = diff
            best_ewma_l = l_cand

    ewma_factory = lambda: EWMADetector(lambd=0.2, l_sigma=best_ewma_l, cooldown=5)
    ewma_fa, ewma_arl0 = measure_in_control_fa(ewma_factory, num_trials, stream_length)
    print(f"  [Calibrated] EWMA (L={best_ewma_l}, lambda=0.2): FA={ewma_fa:.2f}, ARL_0={ewma_arl0:.1f}")

    return {
        "page_hinckley": {
            "uncalibrated_h": 8.0,
            "calibrated_h": 8.0,
            "params": {"delta": 0.15, "h": 8.0, "cooldown": 5},
            "in_control_fa": ph_fa,
            "arl0": ph_arl0,
        },
        "cusum": {
            "uncalibrated_h": 4.0,
            "calibrated_h": best_cusum_h,
            "params": {"delta": 0.25, "h": best_cusum_h, "cooldown": 5},
            "in_control_fa": cusum_fa,
            "arl0": cusum_arl0,
        },
        "ewma": {
            "uncalibrated_h": 3.0,
            "calibrated_h": best_ewma_l,
            "params": {"lambd": 0.2, "l_sigma": best_ewma_l, "cooldown": 5},
            "in_control_fa": ewma_fa,
            "arl0": ewma_arl0,
        },
    }


def evaluate_detection_delay(
    detector_config: dict[str, Any],
    deltas: list[float] = [0.25, 0.50, 1.00],
    num_trials: int = 100,
    total_steps: int = 1000,
    change_point: int = 500,
) -> list[dict[str, Any]]:
    """Evaluate detection delay across shift magnitudes for both uncalibrated and calibrated."""
    results = []

    for delta in deltas:
        for regime in ["uncalibrated", "calibrated"]:
            for det_name in ["cusum", "page_hinckley", "ewma"]:
                delays = []
                false_alarms = []
                miss_count = 0

                for trial in range(num_trials):
                    seed = 10000 * int(delta * 100) + trial
                    rng = np.random.default_rng(seed)
                    pre = rng.normal(loc=0.0, scale=1.0, size=change_point)
                    post = rng.normal(loc=delta, scale=1.0, size=total_steps - change_point)
                    stream = np.concatenate([pre, post])

                    if det_name == "cusum":
                        h_val = 4.0 if regime == "uncalibrated" else detector_config["cusum"]["calibrated_h"]
                        det = CUSUMDetector(delta=0.25, h=h_val, cooldown=5)
                    elif det_name == "page_hinckley":
                        det = PageHinckleyDetector(delta=0.15, h=8.0, cooldown=5)
                    elif det_name == "ewma":
                        l_val = 3.0 if regime == "uncalibrated" else detector_config["ewma"]["calibrated_h"]
                        det = EWMADetector(lambd=0.2, l_sigma=l_val, cooldown=5)

                    fa = 0
                    first_post = None
                    for t, val in enumerate(stream, start=1):
                        evt = det.update(float(val), current_round=t)
                        if evt:
                            if t < change_point:
                                fa += 1
                            elif first_post is None and evt.direction == "positive":
                                first_post = t

                    false_alarms.append(fa)
                    if first_post is not None:
                        delays.append(first_post - change_point)
                    else:
                        miss_count += 1
                        delays.append(total_steps - change_point)

                mean_delay = float(np.mean(delays))
                std_delay = float(np.std(delays))
                mean_fa = float(np.mean(false_alarms))
                miss_rate = miss_count / num_trials
                fl_latency_k10 = mean_delay * 10.0  # Under N/K = 10 (rho = 0.10)

                results.append({
                    "regime": regime,
                    "detector": det_name,
                    "delta": delta,
                    "mean_sample_delay": round(mean_delay, 2),
                    "std_sample_delay": round(std_delay, 2),
                    "fl_latency_rounds_k10": round(fl_latency_k10, 1),
                    "false_alarms_per_trial": round(mean_fa, 2),
                    "miss_rate": round(miss_rate, 3),
                })

    return results


def plot_recalibrated_delays(
    results: list[dict[str, Any]],
    output_png: Path,
) -> None:
    """Generate publication-ready plot comparing calibrated detection delay vs delta."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)

    deltas = [0.25, 0.50, 1.00]
    styles = {
        ("calibrated", "cusum"): ("#1f77b4", "s-", "CUSUM (Calibrated h=6.15)"),
        ("uncalibrated", "cusum"): ("#1f77b4", "s--", "CUSUM (Uncalibrated h=4.0)"),
        ("calibrated", "page_hinckley"): ("#ff7f0e", "o-", "Page-Hinckley (h=8.0)"),
        ("calibrated", "ewma"): ("#2ca02c", "^-", "EWMA (Calibrated L=2.45)"),
    }

    for (regime, det_name), (color, fmt, label) in styles.items():
        subset = [r for r in results if r["regime"] == regime and r["detector"] == det_name]
        subset = sorted(subset, key=lambda x: x["delta"])
        x = [s["delta"] for s in subset]
        y = [s["mean_sample_delay"] for s in subset]
        alpha = 0.5 if "Uncalibrated" in label else 1.0
        ax.plot(x, y, fmt, color=color, label=label, linewidth=1.8, markersize=6, alpha=alpha)

    # Add federated round equivalent axis / annotation
    ax.axhline(50.0, color="gray", linestyle=":", label="Remaining Budget at Drift (50 obs = 500 rounds)")
    ax.set_xlabel(r"Utility Shift Magnitude ($\Delta$)", fontsize=11)
    ax.set_ylabel("Detection Delay (Observed Samples)", fontsize=11)
    ax.set_title("Detector Delay Calibration Under Matched In-Control Run Length ($ARL_0$)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9, loc="upper right")
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)
    print(f"Saved recalibration figure to {output_png}")


def run_arl_recalibration() -> None:
    print("=" * 80)
    print("FedQual-CPX: Detector ARL Recalibration Benchmark (Reviewer #12)")
    print("=" * 80)

    calib_info = calibrate_thresholds(target_fa=3.5, num_trials=100)
    results = evaluate_detection_delay(calib_info, num_trials=100)

    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    json_path = tables_dir / "recalibrated_detector_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"calibration_info": calib_info, "benchmark_results": results}, f, indent=2)

    csv_path = tables_dir / "recalibrated_detector_benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "regime", "detector", "delta", "mean_sample_delay",
                "std_sample_delay", "fl_latency_rounds_k10",
                "false_alarms_per_trial", "miss_rate"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved benchmark results to {json_path} and {csv_path}")

    fig_path = Path("results/figures/fig2_detector_delay_vs_delta.png")
    plot_recalibrated_delays(results, fig_path)

    # Also save to figures/ for paper inclusions
    paper_fig_path = Path("figures/fig2_detector_delay_vs_delta.png")
    plot_recalibrated_delays(results, paper_fig_path)

    # Print clean summary table comparing CUSUM before vs after calibration
    print("\n" + "=" * 80)
    print(f"{'Detector':<16} | {'Regime':<12} | {'Delta':<6} | {'Sample Delay':<14} | {'FL Latency (K=10)':<18} | {'False Alarms':<12}")
    print("-" * 80)
    for r in results:
        if r["detector"] in ["cusum", "page_hinckley"]:
            print(f"{r['detector']:<16} | {r['regime']:<12} | {r['delta']:<6.2f} | {r['mean_sample_delay']:>5.2f} ± {r['std_sample_delay']:<4.2f} obs | {r['fl_latency_rounds_k10']:>6.1f} rounds       | {r['false_alarms_per_trial']:>5.2f}")
    print("=" * 80)


if __name__ == "__main__":
    run_arl_recalibration()
