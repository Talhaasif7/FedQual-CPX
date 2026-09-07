"""
Sequential change-point detection module for FedQual-CPX.

Implements sequential change-point detectors per Section 12 & 13 of the plan:
    1. CUSUM (Cumulative Sum Control Chart) — Section 12.1
    2. Page-Hinckley Detector — Section 12.2 (used by FLEX benchmark)
    3. EWMA (Exponentially Weighted Moving Average) — Section 12.3

All detectors operate on the normalized client utility stream and expose a
standardized interface returning change events and state tracking.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ChangeEvent:
    """Represents a detected change point in client utility."""
    client_id: int
    round: int
    direction: Literal["positive", "negative"]
    detector: str
    magnitude_proxy: float
    score: float
    delay: int | None = None  # Populated when ground truth change round is known


@dataclass
class ChangeState:
    """Current change tracking state for a client."""
    positive_score: float = 0.0
    negative_score: float = 0.0
    last_positive_change: int | None = None
    last_negative_change: int | None = None
    confidence: float = 0.0
    cooldown_remaining: int = 0


class BaseDetector(ABC):
    """Abstract base class for sequential change-point detectors."""

    def __init__(self, baseline_mean: float = 0.0, cooldown: int = 3) -> None:
        self.baseline_mean = baseline_mean
        self.cooldown = cooldown
        self.state = ChangeState()

    @abstractmethod
    def update(self, x: float, current_round: int, client_id: int = 0) -> ChangeEvent | None:
        """Process an observation x at current_round and return ChangeEvent if triggered."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal accumulator statistics."""
        pass

    def get_state(self) -> ChangeState:
        """Return the current change tracking state."""
        return self.state


class CUSUMDetector(BaseDetector):
    """Two-sided CUSUM detector for client utility change detection (Section 12.1).

    Maintains:
        S_pos(t) = max(0, S_pos(t-1) + x_t - mu - delta)
        S_neg(t) = max(0, S_neg(t-1) + mu - x_t - delta)

    Triggers when S_pos > H (positive) or S_neg > H (negative).

    Args:
        delta: Slack/allowance parameter (drift magnitude threshold, e.g. 0.5).
        h: Decision threshold (e.g. 3.0 to 5.0).
        baseline_mean: Expected mean under the in-control hypothesis (default 0.0).
        cooldown: Rounds to wait after detection before triggering again.
    """

    def __init__(
        self,
        delta: float = 0.5,
        h: float = 4.0,
        baseline_mean: float = 0.0,
        cooldown: int = 3,
    ) -> None:
        super().__init__(baseline_mean=baseline_mean, cooldown=cooldown)
        self.delta = float(delta)
        self.h = float(h)
        self.s_pos: float = 0.0
        self.s_neg: float = 0.0

    def update(self, x: float, current_round: int, client_id: int = 0) -> ChangeEvent | None:
        if self.state.cooldown_remaining > 0:
            self.state.cooldown_remaining -= 1

        # CUSUM accumulation
        self.s_pos = max(0.0, self.s_pos + x - self.baseline_mean - self.delta)
        self.s_neg = max(0.0, self.s_neg + self.baseline_mean - x - self.delta)

        self.state.positive_score = self.s_pos
        self.state.negative_score = self.s_neg

        event: ChangeEvent | None = None

        if self.state.cooldown_remaining == 0:
            if self.s_pos > self.h:
                event = ChangeEvent(
                    client_id=client_id,
                    round=current_round,
                    direction="positive",
                    detector="cusum",
                    magnitude_proxy=self.s_pos,
                    score=self.s_pos,
                )
                self.state.last_positive_change = current_round
                self.state.cooldown_remaining = self.cooldown
                self.s_pos = 0.0  # Reset positive accumulator

            elif self.s_neg > self.h:
                event = ChangeEvent(
                    client_id=client_id,
                    round=current_round,
                    direction="negative",
                    detector="cusum",
                    magnitude_proxy=self.s_neg,
                    score=self.s_neg,
                )
                self.state.last_negative_change = current_round
                self.state.cooldown_remaining = self.cooldown
                self.s_neg = 0.0  # Reset negative accumulator

        return event

    def reset(self) -> None:
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.state.positive_score = 0.0
        self.state.negative_score = 0.0


class PageHinckleyDetector(BaseDetector):
    """Two-sided Page-Hinckley test (Section 12.2).

    Mandatory comparison for FLEX (2026).
    Tracks cumulative deviation from running mean:
        U_pos(t) = sum(x_i - mu - delta)
        m(t) = min(U_pos(1..t))
        PH_pos = U_pos(t) - m(t) > H

        U_neg(t) = sum(x_i - mu + delta)
        M(t) = max(U_neg(1..t))
        PH_neg = M(t) - U_neg(t) > H

    Args:
        delta: Minimum allowable drift magnitude (default 0.2).
        h: Threshold parameter (default 10.0).
        alpha: Decay factor for running sum (default 0.99 for adaptive tracking).
        baseline_mean: In-control baseline mean.
        cooldown: Cooldown rounds.
    """

    def __init__(
        self,
        delta: float = 0.2,
        h: float = 10.0,
        alpha: float = 1.0,
        baseline_mean: float = 0.0,
        cooldown: int = 3,
    ) -> None:
        super().__init__(baseline_mean=baseline_mean, cooldown=cooldown)
        self.delta = float(delta)
        self.h = float(h)
        self.alpha = float(alpha)

        self.u_pos: float = 0.0
        self.min_u_pos: float = 0.0
        self.u_neg: float = 0.0
        self.max_u_neg: float = 0.0

    def update(self, x: float, current_round: int, client_id: int = 0) -> ChangeEvent | None:
        if self.state.cooldown_remaining > 0:
            self.state.cooldown_remaining -= 1

        # Update sums
        self.u_pos = self.alpha * self.u_pos + (x - self.baseline_mean - self.delta)
        self.min_u_pos = min(self.min_u_pos, self.u_pos)
        ph_pos = self.u_pos - self.min_u_pos

        self.u_neg = self.alpha * self.u_neg + (x - self.baseline_mean + self.delta)
        self.max_u_neg = max(self.max_u_neg, self.u_neg)
        ph_neg = self.max_u_neg - self.u_neg

        self.state.positive_score = ph_pos
        self.state.negative_score = ph_neg

        event: ChangeEvent | None = None

        if self.state.cooldown_remaining == 0:
            if ph_pos > self.h:
                event = ChangeEvent(
                    client_id=client_id,
                    round=current_round,
                    direction="positive",
                    detector="page_hinckley",
                    magnitude_proxy=ph_pos,
                    score=ph_pos,
                )
                self.state.last_positive_change = current_round
                self.state.cooldown_remaining = self.cooldown
                self.u_pos = 0.0
                self.min_u_pos = 0.0

            elif ph_neg > self.h:
                event = ChangeEvent(
                    client_id=client_id,
                    round=current_round,
                    direction="negative",
                    detector="page_hinckley",
                    magnitude_proxy=ph_neg,
                    score=ph_neg,
                )
                self.state.last_negative_change = current_round
                self.state.cooldown_remaining = self.cooldown
                self.u_neg = 0.0
                self.max_u_neg = 0.0

        return event

    def reset(self) -> None:
        self.u_pos = 0.0
        self.min_u_pos = 0.0
        self.u_neg = 0.0
        self.max_u_neg = 0.0
        self.state.positive_score = 0.0
        self.state.negative_score = 0.0


class EWMADetector(BaseDetector):
    """Exponentially Weighted Moving Average (EWMA) Control Chart (Section 12.3).

    z_t = lambda * x_t + (1 - lambda) * z_{t-1}
    sigma_z = sigma * sqrt(lambda / (2 - lambda) * (1 - (1 - lambda)^(2t)))
    Upper Limit = mu + L * sigma_z
    Lower Limit = mu - L * sigma_z

    Args:
        lambd: Smoothing constant (0 < lambda <= 1, default 0.2).
        l_sigma: Number of standard errors for limits (default 3.0).
        sigma: Standard deviation under in-control state (default 1.0).
        baseline_mean: In-control baseline mean (default 0.0).
        cooldown: Cooldown rounds.
    """

    def __init__(
        self,
        lambd: float = 0.2,
        l_sigma: float = 3.0,
        sigma: float = 1.0,
        baseline_mean: float = 0.0,
        cooldown: int = 3,
    ) -> None:
        super().__init__(baseline_mean=baseline_mean, cooldown=cooldown)
        self.lambd = float(lambd)
        self.l_sigma = float(l_sigma)
        self.sigma = float(sigma)
        self.z: float = baseline_mean
        self.t: int = 0

    def update(self, x: float, current_round: int, client_id: int = 0) -> ChangeEvent | None:
        if self.state.cooldown_remaining > 0:
            self.state.cooldown_remaining -= 1

        self.t += 1
        self.z = self.lambd * x + (1.0 - self.lambd) * self.z

        # Time-varying standard deviation of EWMA statistic
        factor = (self.lambd / (2.0 - self.lambd)) * (1.0 - (1.0 - self.lambd) ** (2 * self.t))
        sigma_z = self.sigma * math.sqrt(max(1e-9, factor))

        upper_limit = self.baseline_mean + self.l_sigma * sigma_z
        lower_limit = self.baseline_mean - self.l_sigma * sigma_z

        pos_dev = max(0.0, self.z - upper_limit)
        neg_dev = max(0.0, lower_limit - self.z)

        self.state.positive_score = pos_dev
        self.state.negative_score = neg_dev

        event: ChangeEvent | None = None

        if self.state.cooldown_remaining == 0:
            if self.z > upper_limit:
                event = ChangeEvent(
                    client_id=client_id,
                    round=current_round,
                    direction="positive",
                    detector="ewma",
                    magnitude_proxy=self.z - self.baseline_mean,
                    score=self.z,
                )
                self.state.last_positive_change = current_round
                self.state.cooldown_remaining = self.cooldown
                self.z = self.baseline_mean

            elif self.z < lower_limit:
                event = ChangeEvent(
                    client_id=client_id,
                    round=current_round,
                    direction="negative",
                    detector="ewma",
                    magnitude_proxy=self.baseline_mean - self.z,
                    score=self.z,
                )
                self.state.last_negative_change = current_round
                self.state.cooldown_remaining = self.cooldown
                self.z = self.baseline_mean

        return event

    def reset(self) -> None:
        self.z = self.baseline_mean
        self.t = 0
        self.state.positive_score = 0.0
        self.state.negative_score = 0.0


class DummyDetector(BaseDetector):
    """No-op detector for ablation studies (Sections 31 & 32)."""

    def __init__(self) -> None:
        super().__init__(baseline_mean=0.0, cooldown=0)

    def update(self, x: float, current_round: int, client_id: int = 0) -> ChangeEvent | None:
        return None

    def reset(self) -> None:
        pass


def create_detector(
    name: str,
    delta: float = 0.5,
    h: float = 4.0,
    cooldown: int = 3,
    **kwargs: Any,
) -> BaseDetector:
    """Factory function for change-point detectors."""
    name_lower = name.lower()
    if name_lower in ("none", "dummy", "no_detector"):
        return DummyDetector()
    elif name_lower == "cusum":
        return CUSUMDetector(delta=delta, h=h, cooldown=cooldown)
    elif name_lower in ("page_hinckley", "ph"):
        ph_delta_val = kwargs.get("ph_delta", delta)
        ph_h_val = kwargs.get("ph_h", h)
        return PageHinckleyDetector(
            delta=ph_delta_val,
            h=ph_h_val,
            cooldown=cooldown,
        )
    elif name_lower == "ewma":
        return EWMADetector(
            lambd=kwargs.get("ewma_lambda", 0.2),
            l_sigma=kwargs.get("ewma_l", 3.0),
            cooldown=cooldown,
        )
    else:
        raise ValueError(f"Unknown detector name: '{name}'. Choose from 'none', 'cusum', 'page_hinckley', 'ewma'.")
