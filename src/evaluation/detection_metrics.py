"""
Change-Point Detection Quality Metrics for FedQual-CPX.

Per Section 35 & RQ1:
    - Detection Delay (mean and std across detected changes)
    - False Alarm Rate (detections occurring before or outside true drift windows)
    - Missed Detection Rate (ground-truth changes with no detection within window)
    - Precision, Recall, and F1 score of change-point events
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


def compute_detection_performance(
    detected_events: list[dict],
    ground_truth_changes: list[dict],
    max_delay_window: int = 50,
    total_rounds: int = 100,
) -> dict[str, float]:
    """Compute detection delay, false alarms, miss rate, precision, recall, and F1.

    Args:
        detected_events: List of dicts with keys {"client_id": int, "round": int, "direction": str}.
        ground_truth_changes: List of dicts with keys {"client_id": int, "round": int, "direction": str}.
        max_delay_window: Maximum allowable rounds after true drift to match a detection.
        total_rounds: Total training rounds in the experiment.

    Returns:
        Dictionary of detection evaluation metrics.
    """
    if not ground_truth_changes:
        false_alarms = len(detected_events)
        far = false_alarms / max(1, total_rounds)
        return {
            "num_true_changes": 0,
            "num_detections": len(detected_events),
            "false_alarms": false_alarms,
            "false_alarm_rate_per_round": round(far, 4),
            "mean_detection_delay": float("nan"),
            "std_detection_delay": float("nan"),
            "miss_rate": 0.0,
            "precision": 0.0 if detected_events else 1.0,
            "recall": 1.0,
            "f1": 0.0,
        }

    matched_gt: set[int] = set()
    delays: list[int] = []
    true_positives = 0
    false_positives = 0

    # Sort events by round
    sorted_events = sorted(detected_events, key=lambda e: e.get("round", 0))

    for evt in sorted_events:
        c_id = evt.get("client_id")
        r_det = evt.get("round", 0)

        # Look for a matching ground-truth change for this client
        matched = False
        for gt_idx, gt in enumerate(ground_truth_changes):
            if gt_idx in matched_gt:
                continue
            if gt.get("client_id") == c_id:
                gt_round = gt.get("round", 0)
                if gt_round <= r_det <= gt_round + max_delay_window:
                    matched_gt.add(gt_idx)
                    delays.append(r_det - gt_round)
                    true_positives += 1
                    matched = True
                    break

        if not matched:
            false_positives += 1

    num_gt = len(ground_truth_changes)
    missed = num_gt - len(matched_gt)
    miss_rate = missed / num_gt if num_gt > 0 else 0.0

    precision = true_positives / max(1, (true_positives + false_positives))
    recall = true_positives / max(1, num_gt)
    f1 = (2.0 * precision * recall) / max(1e-8, (precision + recall))

    return {
        "num_true_changes": num_gt,
        "num_detections": len(detected_events),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "missed_changes": missed,
        "miss_rate": round(miss_rate, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_delay": round(float(np.mean(delays)), 2) if delays else float("nan"),
        "std_delay": round(float(np.std(delays)), 2) if delays else float("nan"),
        "median_delay": round(float(np.median(delays)), 2) if delays else float("nan"),
    }
