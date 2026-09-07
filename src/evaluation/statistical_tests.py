"""
Statistical Hypothesis Testing and Confidence Intervals for FedQual-CPX.

Per Section 36 & 37:
    - Wilcoxon Signed-Rank Test (non-parametric paired comparison)
    - Paired Student's t-Test
    - Cohen's d Effect Size
    - Non-parametric Bootstrap 95% Confidence Intervals
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from scipy import stats


def compute_bootstrap_ci(
    data: Sequence[float],
    confidence_level: float = 0.95,
    num_bootstraps: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute empirical mean and non-parametric bootstrap confidence interval.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        v = float(arr[0])
        return v, v, v

    rng = np.random.default_rng(seed)
    boot_means = np.zeros(num_bootstraps)
    n = len(arr)

    for i in range(num_bootstraps):
        resample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(resample)

    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))
    mean_val = float(np.mean(arr))

    return round(mean_val, 4), round(ci_lower, 4), round(ci_upper, 4)


def compute_cohens_d(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Cohen's d effect size between two paired or independent samples.

    d = (mean_x - mean_y) / pooled_std
    """
    arr_x = np.asarray(x, dtype=np.float64)
    arr_y = np.asarray(y, dtype=np.float64)

    nx, ny = len(arr_x), len(arr_y)
    if nx < 2 or ny < 2:
        return float("nan")

    vx, vy = np.var(arr_x, ddof=1), np.var(arr_y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))

    if pooled_std == 0:
        return 0.0
    return float((np.mean(arr_x) - np.mean(arr_y)) / pooled_std)


def compare_paired_methods(
    method_a_scores: Sequence[float],
    method_b_scores: Sequence[float],
    method_a_name: str = "FedQual-CPX",
    method_b_name: str = "Baseline",
) -> dict[str, float | str]:
    """Perform rigorous paired statistical tests comparing two methods across identical seeds/partitions.

    Per Section 36: Wilcoxon signed-rank test, paired t-test, Cohen's d.
    """
    a = np.asarray(method_a_scores, dtype=np.float64)
    b = np.asarray(method_b_scores, dtype=np.float64)

    if len(a) != len(b):
        raise ValueError(f"Sample sizes must match for paired testing: len(a)={len(a)}, len(b)={len(b)}")

    n = len(a)
    diff = a - b
    mean_diff = float(np.mean(diff))

    # Wilcoxon signed-rank test
    if np.all(diff == 0):
        w_stat, w_p = 0.0, 1.0
    else:
        try:
            w_res = stats.wilcoxon(a, b, alternative="two-sided")
            w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_stat, w_p = float("nan"), float("nan")

    # Paired Student's t-test
    try:
        t_res = stats.ttest_rel(a, b)
        t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)
    except Exception:
        t_stat, t_p = float("nan"), float("nan")

    d = compute_cohens_d(a, b)

    mean_a, a_low, a_high = compute_bootstrap_ci(a)
    mean_b, b_low, b_high = compute_bootstrap_ci(b)

    return {
        "method_a": method_a_name,
        "method_b": method_b_name,
        "n_pairs": n,
        "mean_a": mean_a,
        "ci_95_a": f"[{a_low}, {a_high}]",
        "mean_b": mean_b,
        "ci_95_b": f"[{b_low}, {b_high}]",
        "mean_difference": round(mean_diff, 4),
        "cohens_d": round(d, 4),
        "wilcoxon_stat": round(w_stat, 4),
        "wilcoxon_p_value": round(w_p, 5),
        "ttest_stat": round(t_stat, 4),
        "ttest_p_value": round(t_p, 5),
        "statistically_significant_05": bool(w_p < 0.05 if not np.isnan(w_p) else False),
    }
