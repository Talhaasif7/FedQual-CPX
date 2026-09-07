"""
Section 59: First Experiment — Synthetic Detector Benchmark.

Per Section 59 of the plan:
    Generate:
        x_t ~ Normal(0, 1), t < 500
        x_t ~ Normal(Delta, 1), t >= 500

    Compare:
        - CUSUM
        - Page-Hinckley
        - EWMA

    For shift magnitudes:
        Delta in {0.25, 0.50, 1.00}

    Measure:
        - Detection delay (rounds from tau=500 to first post-drift detection)
        - False alarm count (detections before tau=500)
        - Miss rate (fraction of trials with no detection by t=1000)

Saves summary table and raw trial logs to results/raw/synthetic_detector_benchmark/.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.selection.detectors import (
    CUSUMDetector,
    PageHinckleyDetector,
    EWMADetector,
)


def run_single_trial(
    detector_name: str,
    delta: float,
    seed: int,
    total_steps: int = 1000,
    change_point: int = 500,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    # Generate synthetic stream
    pre = rng.normal(loc=0.0, scale=1.0, size=change_point)
    post = rng.normal(loc=delta, scale=1.0, size=total_steps - change_point)
    stream = np.concatenate([pre, post])

    # Instantiate detector
    if detector_name == "cusum":
        # Tuned allowance for given delta (allowance around delta/2 or 0.25)
        detector = CUSUMDetector(delta=0.25, h=4.0, cooldown=5)
    elif detector_name == "page_hinckley":
        detector = PageHinckleyDetector(delta=0.15, h=8.0, cooldown=5)
    elif detector_name == "ewma":
        detector = EWMADetector(lambd=0.2, l_sigma=3.0, cooldown=5)
    else:
        raise ValueError(f"Unknown detector: {detector_name}")

    false_alarms = 0
    first_post_detection: int | None = None
    all_events = []

    for t_step, val in enumerate(stream, start=1):
        evt = detector.update(float(val), current_round=t_step)
        if evt:
            all_events.append(evt)
            if t_step < change_point:
                false_alarms += 1
            elif first_post_detection is None and evt.direction == "positive":
                first_post_detection = t_step

    detected = first_post_detection is not None
    delay = (first_post_detection - change_point) if detected else (total_steps - change_point)

    return {
        "detector": detector_name,
        "delta": delta,
        "seed": seed,
        "false_alarms": false_alarms,
        "detected": detected,
        "delay": delay,
        "first_detection_round": first_post_detection,
    }


def run_benchmark(
    num_trials: int = 50,
    deltas: tuple[float, ...] = (0.25, 0.50, 1.00),
    detectors: tuple[str, ...] = ("cusum", "page_hinckley", "ewma"),
) -> dict[str, Any]:
    print("=" * 70)
    print("FedQual-CPX -- Section 59: Synthetic Detector Benchmark")
    print(f"Trials per condition: {num_trials} | Shifts Delta: {deltas}")
    print("=" * 70)

    results_dir = Path("results/raw/synthetic_detector_benchmark")
    results_dir.mkdir(parents=True, exist_ok=True)

    trial_rows = []
    summary_data = []

    for delta in deltas:
        print(f"\n--- Evaluating Shift Magnitude Delta = {delta:.2f} ---")
        for det_name in detectors:
            delays = []
            false_alarm_counts = []
            miss_count = 0

            for trial_idx in range(num_trials):
                seed = 10000 * int(delta * 100) + trial_idx
                trial_res = run_single_trial(det_name, delta, seed=seed)
                trial_rows.append(trial_res)

                false_alarm_counts.append(trial_res["false_alarms"])
                if trial_res["detected"]:
                    delays.append(trial_res["delay"])
                else:
                    miss_count += 1

            miss_rate = miss_count / num_trials
            mean_delay = float(np.mean(delays)) if delays else float("nan")
            std_delay = float(np.std(delays)) if delays else float("nan")
            mean_fa = float(np.mean(false_alarm_counts))

            summary_row = {
                "detector": det_name,
                "delta": delta,
                "num_trials": num_trials,
                "mean_delay": round(mean_delay, 2),
                "std_delay": round(std_delay, 2),
                "false_alarms_per_trial": round(mean_fa, 3),
                "miss_rate": round(miss_rate, 3),
            }
            summary_data.append(summary_row)

            print(
                f"[{det_name:<14}] Delay: {mean_delay:6.2f} ± {std_delay:5.2f} rounds | "
                f"False Alarms/trial: {mean_fa:.3f} | Miss Rate: {miss_rate * 100:4.1f}%"
            )

    # Save summary JSON
    summary_file = results_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save trials CSV
    csv_file = results_dir / "trials.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["detector", "delta", "seed", "false_alarms", "detected", "delay", "first_detection_round"]
        )
        writer.writeheader()
        writer.writerows(trial_rows)

    print("\n" + "=" * 70)
    print(f"Benchmark completed successfully! Artifacts saved to:")
    print(f"  - {summary_file}")
    print(f"  - {csv_file}")
    print("=" * 70)

    return {"summary": summary_data}


if __name__ == "__main__":
    run_benchmark(num_trials=50)
