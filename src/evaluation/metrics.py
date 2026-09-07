"""
Learning Curve and Recovery Metrics for FedQual-CPX.

Per Section 35 & RQ2, RQ3:
    - Area Under the Learning Curve (AUC)
    - Convergence Rounds to Target Accuracy
    - Post-Drift Recovery Rounds (Rounds to return to pre-drift baseline accuracy)
    - Post-Drift Drop Magnitude (accuracy drop from pre-drift to post-drift nadir)
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


def compute_auc(accuracies: Sequence[float]) -> float:
    """Compute normalized Area Under the Learning Curve (trapezoidal rule).

    Normalized by total rounds, giving an average accuracy score in [0.0, 1.0].
    """
    arr = np.asarray(accuracies, dtype=np.float64)
    if arr.size <= 1:
        return float(arr[0]) if arr.size == 1 else 0.0
    # Portable trapezoidal rule: sum((y_i + y_{i+1}) / 2)
    val = np.sum((arr[:-1] + arr[1:]) / 2.0)
    return float(val / (arr.size - 1))


def compute_rounds_to_target(accuracies: Sequence[float], target_acc: float) -> int | None:
    """Return the first round (1-indexed) where accuracy >= target_acc, or None if never reached."""
    for idx, acc in enumerate(accuracies):
        if acc >= target_acc:
            return idx + 1
    return None


def compute_post_drift_recovery(
    accuracies: Sequence[float],
    drift_round: int,
    recovery_tolerance: float = 0.02,
) -> dict[str, float | int | None]:
    """Compute post-drift performance degradation and recovery metrics.

    Args:
        accuracies: Array of test accuracies across rounds 1..T.
        drift_round: The round tau at which concept drift occurred (1-indexed).
        recovery_tolerance: How close to pre-drift accuracy is considered 'recovered' (e.g. within 2%).

    Returns:
        Dict with pre_drift_acc, post_drift_nadir, drop_magnitude, recovery_round, recovery_time.
    """
    arr = np.asarray(accuracies, dtype=np.float64)
    tau_idx = drift_round - 1

    if tau_idx <= 0 or tau_idx >= len(arr):
        return {
            "pre_drift_accuracy": float("nan"),
            "post_drift_nadir": float("nan"),
            "drop_magnitude": float("nan"),
            "recovery_round": None,
            "recovery_delay": None,
        }

    # Pre-drift benchmark: mean accuracy over 3 rounds immediately preceding drift
    pre_window = arr[max(0, tau_idx - 3):tau_idx]
    pre_drift_acc = float(np.mean(pre_window)) if pre_window.size > 0 else float(arr[tau_idx - 1])

    # Post-drift trajectory
    post_traj = arr[tau_idx:]
    nadir = float(np.min(post_traj))
    drop = float(max(0.0, pre_drift_acc - nadir))

    target_recovery_acc = pre_drift_acc - recovery_tolerance
    recovery_round = None
    recovery_delay = None

    for offset, acc in enumerate(post_traj):
        if acc >= target_recovery_acc:
            recovery_round = drift_round + offset
            recovery_delay = offset
            break

    return {
        "pre_drift_accuracy": round(pre_drift_acc, 4),
        "post_drift_nadir": round(nadir, 4),
        "drop_magnitude": round(drop, 4),
        "recovery_round": recovery_round,
        "recovery_delay": recovery_delay,
    }
