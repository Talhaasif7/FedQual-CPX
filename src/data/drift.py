"""
Drift generation module for FedQual-CPX.

Implements controlled non-stationary data streams per Sections 21-24 of the plan:
    D1: Label-proportion shift
    D2: Client-specific class swap
    D3: Feature distribution shift (Gaussian noise, brightness/contrast)
    D4: Client quality shift (label noise, degraded samples)

Follows the piecewise-stationary drift protocol:
    - Phase A: Stationary (t < tau)
    - Phase B: Abrupt change at t = tau
    - Phase C: Post-change stationary (t >= tau)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DriftConfig:
    """Configuration for client drift injection."""
    enabled: bool = False
    drift_type: str = "none"            # 'none', 'class_swap', 'feature_noise', 'feature_shift', 'gradual', 'gradual_drift', 'label_noise'
    drift_round: int = 50              # Round tau when drift occurs (or tau_start for gradual)
    drift_end_round: int = 70          # Round tau_end for gradual drift
    drift_fraction: float = 0.3         # Fraction of clients that experience drift
    severity: str = "medium"            # 'small', 'medium', 'large'
    drift_clients: list[int] = field(default_factory=list)  # Explicit client IDs if set


class DriftedDataset(Dataset):
    """A wrapper dataset that applies dynamic drift transformations on the fly.

    Args:
        base_dataset: Underlying PyTorch dataset returning (x, y).
        transform_fn: Function (x, y) -> (x_drifted, y_drifted).
        is_active: Whether the drift transformation is currently active.
        drift_prob: Probability in [0.0, 1.0] of transforming a sample when active (for gradual drift).
        seed: Random seed for probabilistic sampling.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        transform_fn: Callable[[torch.Tensor, int], tuple[torch.Tensor, int]] | None = None,
        is_active: bool = False,
        drift_prob: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.base_dataset = base_dataset
        self.transform_fn = transform_fn
        self.is_active = is_active
        self.drift_prob = float(np.clip(drift_prob, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)

    def set_active(self, active: bool, drift_prob: float = 1.0) -> None:
        """Enable or disable drift transformation with optional probability."""
        self.is_active = active
        self.drift_prob = float(np.clip(drift_prob, 0.0, 1.0))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x, y = self.base_dataset[idx]
        if self.is_active and self.transform_fn is not None:
            if self.drift_prob >= 1.0 or self.rng.random() < self.drift_prob:
                return self.transform_fn(x, y)
        return x, y


class DriftManager:
    """Manages piecewise-stationary and gradual drift across federated clients.

    Tracks which clients are designated as drifting, their drift schedules,
    and constructs appropriate transformation functions for each drift type.
    """

    # Severity lookup tables
    FEATURE_NOISE_STD = {
        "small": 0.15,
        "medium": 0.35,
        "large": 0.70,
    }

    LABEL_NOISE_RATE = {
        "small": 0.20,
        "medium": 0.40,
        "large": 0.70,
    }

    def __init__(
        self,
        num_clients: int,
        config: DriftConfig | dict[str, Any] | None = None,
        num_classes: int = 10,
        seed: int = 42,
    ) -> None:
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if isinstance(config, dict):
            self.cfg = DriftConfig(**config)
        elif config is None:
            self.cfg = DriftConfig(enabled=False)
        else:
            self.cfg = config

        self.drift_round = self.cfg.drift_round
        self.drift_end_round = getattr(self.cfg, "drift_end_round", self.drift_round + 20)
        self.drift_type = self.cfg.drift_type
        self.enabled = self.cfg.enabled

        # Determine which clients drift
        if not self.enabled or self.drift_type == "none":
            self.drift_client_ids: set[int] = set()
        elif self.cfg.drift_clients:
            self.drift_client_ids = set(self.cfg.drift_clients)
        else:
            n_drift = max(1, int(round(num_clients * self.cfg.drift_fraction)))
            chosen = self.rng.choice(num_clients, size=n_drift, replace=False).tolist()
            self.drift_client_ids = set(chosen)

        # Build transform cache per client
        self._transforms: dict[int, Callable[[torch.Tensor, int], tuple[torch.Tensor, int]]] = {}
        self._init_transforms()

    def _init_transforms(self) -> None:
        """Pre-compute deterministic drift transforms for each drifted client."""
        if not self.enabled:
            return

        is_swap = self.drift_type in ("class_swap", "gradual", "gradual_drift")
        is_feature = self.drift_type in ("feature_noise", "feature_shift")
        is_label_noise = self.drift_type == "label_noise"

        for client_id in self.drift_client_ids:
            client_rng = np.random.default_rng(self.seed + 1000 + client_id)

            if is_swap:
                # D2: Swap two or more classes for this client
                if self.cfg.severity == "large":
                    # Circular shift of all classes: c -> (c + 1) % C
                    mapping = {c: (c + 1) % self.num_classes for c in range(self.num_classes)}
                else:
                    # Pairwise swap of two random distinct classes
                    pair = client_rng.choice(self.num_classes, size=2, replace=False)
                    c1, c2 = int(pair[0]), int(pair[1])
                    mapping = {c: c for c in range(self.num_classes)}
                    mapping[c1] = c2
                    mapping[c2] = c1

                def make_swap(m: dict[int, int]):
                    def _transform(x: torch.Tensor, y: int) -> tuple[torch.Tensor, int]:
                        return x, m.get(int(y), int(y))
                    return _transform

                self._transforms[client_id] = make_swap(mapping)

            elif is_feature:
                # D3: Gaussian feature noise / covariate shift
                noise_std = self.FEATURE_NOISE_STD.get(self.cfg.severity, 0.35)

                def make_feature_noise(std: float, c_rng: np.random.Generator):
                    def _transform(x: torch.Tensor, y: int) -> tuple[torch.Tensor, int]:
                        noise = torch.randn_like(x) * std
                        return torch.clamp(x + noise, 0.0, 1.0), y
                    return _transform

                self._transforms[client_id] = make_feature_noise(noise_std, client_rng)

            elif is_label_noise:
                # D4: Symmetric label noise
                noise_rate = self.LABEL_NOISE_RATE.get(self.cfg.severity, 0.40)
                n_classes = self.num_classes

                def make_label_noise(rate: float, c_rng: np.random.Generator):
                    def _transform(x: torch.Tensor, y: int) -> tuple[torch.Tensor, int]:
                        if c_rng.random() < rate:
                            new_y = int(c_rng.integers(0, n_classes))
                            return x, new_y
                        return x, y
                    return _transform

                self._transforms[client_id] = make_label_noise(noise_rate, client_rng)

    def is_client_drifted(self, client_id: int, current_round: int) -> bool:
        """Check if client i is experiencing drift at current_round."""
        if not self.enabled:
            return False
        return (client_id in self.drift_client_ids) and (current_round >= self.drift_round)

    def get_transform(
        self, client_id: int
    ) -> Callable[[torch.Tensor, int], tuple[torch.Tensor, int]] | None:
        """Get the drift transformation for a client, if defined."""
        return self._transforms.get(client_id, None)

    def wrap_client_dataset(
        self, client_id: int, dataset: Dataset, current_round: int = 0
    ) -> DriftedDataset:
        """Wrap a client's dataset in a DriftedDataset instance."""
        transform = self.get_transform(client_id)
        is_gradual = self.drift_type in ("gradual", "gradual_drift")
        active = self.is_client_drifted(client_id, current_round)
        drift_prob = 1.0
        if is_gradual:
            tau_start = self.drift_round
            tau_end = max(tau_start + 1, self.drift_end_round)
            drift_prob = float(np.clip((current_round - tau_start) / (tau_end - tau_start), 0.0, 1.0))

        return DriftedDataset(
            dataset,
            transform_fn=transform,
            is_active=active,
            drift_prob=drift_prob if active else 0.0,
            seed=self.seed + client_id,
        )

    def update_round(self, current_round: int, client_datasets: dict[int, DriftedDataset]) -> None:
        """Update active states and drift probabilities for all client datasets."""
        is_gradual = self.drift_type in ("gradual", "gradual_drift")
        tau_start = self.drift_round
        tau_end = max(tau_start + 1, self.drift_end_round)

        for client_id, ds in client_datasets.items():
            if client_id in self.drift_client_ids:
                if is_gradual:
                    if current_round < tau_start:
                        ds.set_active(False, drift_prob=0.0)
                    else:
                        prob = float(np.clip((current_round - tau_start) / (tau_end - tau_start), 0.0, 1.0))
                        ds.set_active(True, drift_prob=prob)
                else:
                    active = current_round >= self.drift_round
                    ds.set_active(active, drift_prob=1.0 if active else 0.0)
            else:
                ds.set_active(False, drift_prob=0.0)

    def get_drift_info(self) -> dict[str, Any]:
        """Summary of drift configuration for reproducibility and logging."""
        return {
            "enabled": self.enabled,
            "drift_type": self.drift_type,
            "drift_round": self.drift_round,
            "drift_end_round": self.drift_end_round,
            "severity": self.cfg.severity,
            "num_drift_clients": len(self.drift_client_ids),
            "drift_client_ids": sorted(list(self.drift_client_ids)),
        }
