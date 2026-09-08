"""
Robust utility normalization for FedQual-CPX.

Per Section 10 of the implementation plan:
    Clients can have naturally higher loss scales because local distributions
    vary in difficulty. Therefore, utility signals must be normalized relative
    to comparable observations across clients or across time.

Baseline Robust MAD Formulation:
    median_t = median(observed utilities)
    MAD_t = median(|u_{i,t} - median_t|)
    z_{i,t} = (u_{i,t} - median_t) / (1.4826 * MAD_t + epsilon)
    z = clip(z, -z_max, z_max)
"""

from __future__ import annotations

import numpy as np


class RobustNormalizer:
    """Robust MAD (Median Absolute Deviation) Normalizer for client utilities.

    Args:
        z_max: Maximum absolute value for z-score clipping (default 3.0).
               Ablations test z_max in {2.0, 3.0, 5.0}.
        epsilon: Numerical stability constant (default 1e-8).
    """

    def __init__(self, z_max: float = 3.0, epsilon: float = 1e-8) -> None:
        self.z_max = float(z_max)
        self.epsilon = float(epsilon)

    def normalize_batch(self, utilities: list[float] | np.ndarray) -> np.ndarray:
        """Normalize a collection of contemporaneous utility observations.

        Args:
            utilities: List or array of raw utility values from round t.

        Returns:
            np.ndarray of clipped robust z-scores.
        """
        arr = np.asarray(utilities, dtype=np.float64)
        if len(arr) == 0:
            return np.array([], dtype=np.float64)

        if len(arr) == 1:
            # Single observation: neutral normalized score 0.0
            return np.array([0.0], dtype=np.float64)

        med = np.median(arr)
        abs_diff = np.abs(arr - med)
        mad = np.median(abs_diff)

        # Scale factor 1.4826 makes MAD an unbiased estimator of sigma for normal distributions
        scale = 1.4826 * mad + self.epsilon
        z = (arr - med) / scale
        return np.clip(z, -self.z_max, self.z_max)

    def normalize_single(
        self,
        value: float,
        reference_utilities: list[float] | np.ndarray,
    ) -> float:
        """Normalize a single utility value against a reference distribution.

        Args:
            value: Single raw utility value.
            reference_utilities: Baseline/historical distribution values.

        Returns:
            Normalized and clipped z-score.
        """
        arr = np.asarray(reference_utilities, dtype=np.float64)
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return 0.0

        med = np.median(valid)
        abs_diff = np.abs(valid - med)
        mad = np.median(abs_diff)

        scale = 1.4826 * mad + self.epsilon
        z = (value - med) / scale
        return float(np.clip(z, -self.z_max, self.z_max))


class ZScoreNormalizer:
    """Standard Mean/Std Z-Score Normalizer for ablation studies."""

    def __init__(self, z_max: float = 3.0, epsilon: float = 1e-8) -> None:
        self.z_max = float(z_max)
        self.epsilon = float(epsilon)

    def normalize_batch(self, utilities: list[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(utilities, dtype=np.float64)
        if len(arr) <= 1:
            return np.zeros_like(arr, dtype=np.float64)
        mean = np.mean(arr)
        std = np.std(arr) + self.epsilon
        z = (arr - mean) / std
        return np.clip(z, -self.z_max, self.z_max)

    def normalize_single(self, value: float, reference_utilities: list[float] | np.ndarray) -> float:
        arr = np.asarray(reference_utilities, dtype=np.float64)
        valid = arr[~np.isnan(arr)]
        if len(valid) <= 1:
            return 0.0
        mean = np.mean(valid)
        std = np.std(valid) + self.epsilon
        z = (value - mean) / std
        return float(np.clip(z, -self.z_max, self.z_max))


class MinMaxNormalizer:
    """Min-Max Normalizer [0, 1] for ablation studies."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = float(epsilon)

    def normalize_batch(self, utilities: list[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(utilities, dtype=np.float64)
        if len(arr) <= 1:
            return np.zeros_like(arr, dtype=np.float64)
        min_val = np.min(arr)
        max_val = np.max(arr)
        denom = (max_val - min_val) + self.epsilon
        return (arr - min_val) / denom

    def normalize_single(self, value: float, reference_utilities: list[float] | np.ndarray) -> float:
        arr = np.asarray(reference_utilities, dtype=np.float64)
        valid = arr[~np.isnan(arr)]
        if len(valid) <= 1:
            return 0.5
        min_val = np.min(valid)
        max_val = np.max(valid)
        denom = (max_val - min_val) + self.epsilon
        return float((value - min_val) / denom)


class IdentityNormalizer:
    """Pass-through normalizer (raw utility) for ablation studies (Section 31 A3)."""

    def normalize_batch(self, utilities: list[float] | np.ndarray) -> np.ndarray:
        return np.asarray(utilities, dtype=np.float64)

    def normalize_single(self, value: float, reference_utilities: list[float] | np.ndarray) -> float:
        return float(value)


def create_normalizer(
    method: str = "robust_mad",
    z_max: float = 3.0,
    epsilon: float = 1e-8,
) -> RobustNormalizer | ZScoreNormalizer | MinMaxNormalizer | IdentityNormalizer:
    """Factory function for normalizers."""
    method_lower = method.lower()
    if method_lower in ("none", "identity", "raw"):
        return IdentityNormalizer()
    elif method_lower in ("robust_mad", "mad"):
        return RobustNormalizer(z_max=z_max, epsilon=epsilon)
    elif method_lower in ("zscore", "z_score", "standard"):
        return ZScoreNormalizer(z_max=z_max, epsilon=epsilon)
    elif method_lower in ("minmax", "min_max"):
        return MinMaxNormalizer(epsilon=epsilon)
    else:
        raise ValueError(f"Unknown normalization method: '{method}'. Choose 'robust_mad', 'zscore', 'minmax', or 'none'.")
