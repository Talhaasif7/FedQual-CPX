"""
Fairness and Client Participation Diversity Metrics for FedQual-CPX.

Per Section 35 & RQ8:
    - Participation Gini Coefficient
    - Participation Shannon Entropy
    - Client Selection Coverage
    - Starvation Rate (fraction of clients never selected)
    - Minimum / Maximum Selection Counts
"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np


def compute_gini_coefficient(counts: Sequence[int | float] | np.ndarray) -> float:
    """Compute the Gini inequality coefficient for client participation.

    Gini = (sum_i sum_j |x_i - x_j|) / (2 * N * sum_i x_i)
    Values range from 0.0 (perfect equality) to 1.0 (maximal inequality/starvation).
    """
    arr = np.asarray(counts, dtype=np.float64)
    if arr.size == 0 or np.sum(arr) == 0:
        return 0.0

    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * sorted_arr)) / (n * np.sum(sorted_arr)) - (n + 1.0) / n)


def compute_participation_entropy(counts: Sequence[int | float] | np.ndarray) -> float:
    """Compute Shannon entropy of the client participation distribution.

    H = - sum_i p_i * log2(p_i)
    Higher entropy indicates more uniform, diverse client selection.
    """
    arr = np.asarray(counts, dtype=np.float64)
    total = np.sum(arr)
    if total == 0:
        return 0.0

    probs = arr / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def compute_coverage(counts: Sequence[int | float] | np.ndarray) -> float:
    """Fraction of unique clients selected at least once."""
    arr = np.asarray(counts)
    if arr.size == 0:
        return 0.0
    return float(np.sum(arr > 0) / arr.size)


def compute_starvation_rate(counts: Sequence[int | float] | np.ndarray) -> float:
    """Fraction of clients that have NEVER been selected (starvation)."""
    return 1.0 - compute_coverage(counts)


def compute_all_fairness_metrics(counts: Sequence[int | float] | np.ndarray) -> dict[str, float]:
    """Compute the full suite of client fairness and diversity metrics."""
    arr = np.asarray(counts)
    return {
        "gini": round(compute_gini_coefficient(arr), 4),
        "entropy": round(compute_participation_entropy(arr), 4),
        "coverage": round(compute_coverage(arr), 4),
        "starvation_rate": round(compute_starvation_rate(arr), 4),
        "min_participations": int(np.min(arr)) if arr.size > 0 else 0,
        "max_participations": int(np.max(arr)) if arr.size > 0 else 0,
        "mean_participations": round(float(np.mean(arr)), 2) if arr.size > 0 else 0.0,
    }
