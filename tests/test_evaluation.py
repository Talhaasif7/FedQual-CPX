"""
Unit tests for FedQual-CPX evaluation modules:
    - fairness.py
    - detection_metrics.py
    - metrics.py
    - statistical_tests.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.evaluation.fairness import (
    compute_gini_coefficient,
    compute_participation_entropy,
    compute_coverage,
    compute_all_fairness_metrics,
)
from src.evaluation.detection_metrics import compute_detection_performance
from src.evaluation.metrics import (
    compute_auc,
    compute_rounds_to_target,
    compute_post_drift_recovery,
)
from src.evaluation.statistical_tests import (
    compute_bootstrap_ci,
    compute_cohens_d,
    compare_paired_methods,
)


def test_fairness_metrics():
    # Perfectly equal participation
    counts_equal = [5, 5, 5, 5, 5]
    gini = compute_gini_coefficient(counts_equal)
    assert abs(gini) < 1e-4, f"Expected 0 Gini for equal counts, got {gini}"
    assert compute_coverage(counts_equal) == 1.0

    # Highly unequal participation (starvation)
    counts_unequal = [20, 0, 0, 0]
    gini_high = compute_gini_coefficient(counts_unequal)
    assert gini_high > 0.7, f"Expected high Gini for starved clients, got {gini_high}"
    assert compute_coverage(counts_unequal) == 0.25

    summary = compute_all_fairness_metrics([3, 4, 3, 5, 0])
    assert "gini" in summary and "entropy" in summary and "coverage" in summary
    print("[PASS] Fairness and participation diversity metrics verified.")


def test_detection_metrics():
    gt = [
        {"client_id": 1, "round": 10, "direction": "positive"},
        {"client_id": 5, "round": 20, "direction": "negative"},
    ]
    detected = [
        {"client_id": 1, "round": 13, "direction": "positive"},  # delay = 3
        {"client_id": 5, "round": 25, "direction": "negative"},  # delay = 5
        {"client_id": 8, "round": 50, "direction": "positive"},  # false alarm
    ]

    res = compute_detection_performance(detected, gt, max_delay_window=10, total_rounds=100)
    assert res["true_positives"] == 2
    assert res["false_positives"] == 1
    assert res["missed_changes"] == 0
    assert res["miss_rate"] == 0.0
    assert abs(res["mean_delay"] - 4.0) < 1e-4
    assert res["precision"] == round(2 / 3, 4)
    assert res["recall"] == 1.0
    print("[PASS] Sequential change detection metrics verified.")


def test_recovery_and_learning_metrics():
    accuracies = [0.10, 0.20, 0.30, 0.35, 0.25, 0.26, 0.34, 0.36]
    # Drift at round 5
    rec = compute_post_drift_recovery(accuracies, drift_round=5, recovery_tolerance=0.02)
    assert rec["recovery_round"] is not None
    assert rec["drop_magnitude"] > 0.0

    auc = compute_auc(accuracies)
    assert 0.0 < auc < 1.0
    r_target = compute_rounds_to_target(accuracies, target_acc=0.30)
    assert r_target == 3
    print("[PASS] Recovery trajectory and learning curve metrics verified.")


def test_statistical_tests():
    rng = np.random.default_rng(42)
    # Method A consistently outperforms Method B by 0.05
    b = rng.normal(0.60, 0.02, size=10)
    a = b + 0.05 + rng.normal(0.0, 0.005, size=10)

    mean, low, high = compute_bootstrap_ci(a)
    assert low <= mean <= high

    d = compute_cohens_d(a, b)
    assert d > 1.0, f"Expected large effect size, got {d}"

    res = compare_paired_methods(a, b, "FedQual-CPX", "Baseline")
    assert res["statistically_significant_05"] is True
    assert res["wilcoxon_p_value"] < 0.05
    print("[PASS] Statistical hypothesis testing and bootstrap CIs verified.")


if __name__ == "__main__":
    test_fairness_metrics()
    test_detection_metrics()
    test_recovery_and_learning_metrics()
    test_statistical_tests()
    print("\nAll evaluation unit tests passed!")
