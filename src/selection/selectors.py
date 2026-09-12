"""
Client selection algorithms for FedQual-CPX.

Implements the baseline suite (Section 30) and the proposed FedQual-CPX method (Section 14-18):
    B0: Random / FedAvg (uniform random)
    B2: Utility Greedy (latest observed utility)
    B3: Sliding-Window Utility (windowed average utility)
    B4: Fixed Exploration + Utility (fixed epsilon)
    B5: CUSUM-only (change detection without adaptive exploration)
    B6: Page-Hinckley + Adaptive Exploration (FLEX-matched competitor)
    B8: Full FedQual-CPX (change detection + adaptive exploration + uncertainty-aware exploitation)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from src.selection.detectors import (
    BaseDetector,
    CUSUMDetector,
    PageHinckleyDetector,
    EWMADetector,
    ChangeEvent,
    create_detector,
)


@dataclass
class SelectionResult:
    """Outcome of a round's client selection decision."""
    selected_client_ids: list[int]
    epsilon: float = 0.0
    change_events: list[ChangeEvent] = field(default_factory=list)
    client_scores: dict[int, dict[str, float]] = field(default_factory=dict)


class BaseSelector(ABC):
    """Abstract interface for all federated client selection policies."""

    @abstractmethod
    def select(
        self,
        round_num: int,
        num_clients: int,
        clients_per_round: int,
        client_history: dict[int, dict[str, Any]],
        rng: np.random.Generator,
    ) -> SelectionResult:
        """Select K clients for round round_num using only pre-round history."""
        pass


class RandomSelector(BaseSelector):
    """B0: Uniform random client selection."""

    def select(
        self,
        round_num: int,
        num_clients: int,
        clients_per_round: int,
        client_history: dict[int, dict[str, Any]],
        rng: np.random.Generator,
    ) -> SelectionResult:
        k = min(clients_per_round, num_clients)
        chosen = rng.choice(num_clients, size=k, replace=False).tolist()
        return SelectionResult(selected_client_ids=chosen, epsilon=1.0)


class UtilityGreedySelector(BaseSelector):
    """B2: Select clients with highest recent observed utility."""

    def select(
        self,
        round_num: int,
        num_clients: int,
        clients_per_round: int,
        client_history: dict[int, dict[str, Any]],
        rng: np.random.Generator,
    ) -> SelectionResult:
        scores = {}
        for c in range(num_clients):
            hist = client_history.get(c, {})
            # Look up last non-None utility
            utils = [u for u in hist.get("utility", []) if u is not None]
            scores[c] = utils[-1] if utils else 0.0

        # Sort clients descending by utility
        sorted_clients = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        chosen = sorted_clients[:clients_per_round]

        return SelectionResult(
            selected_client_ids=chosen,
            epsilon=0.0,
            client_scores={c: {"exploit_score": scores[c]} for c in chosen},
        )


class SlidingWindowSelector(BaseSelector):
    """B3: Select clients with highest moving-average utility over window W."""

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size

    def select(
        self,
        round_num: int,
        num_clients: int,
        clients_per_round: int,
        client_history: dict[int, dict[str, Any]],
        rng: np.random.Generator,
    ) -> SelectionResult:
        scores = {}
        for c in range(num_clients):
            hist = client_history.get(c, {})
            utils = [u for u in hist.get("utility", []) if u is not None]
            recent = utils[-self.window_size:] if utils else []
            scores[c] = float(np.mean(recent)) if recent else 0.0

        sorted_clients = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        chosen = sorted_clients[:clients_per_round]

        return SelectionResult(
            selected_client_ids=chosen,
            epsilon=0.0,
            client_scores={c: {"window_score": scores[c]} for c in chosen},
        )


class FixedExplorationSelector(BaseSelector):
    """B4: Epsilon-greedy utility selection with fixed epsilon."""

    def __init__(self, epsilon: float = 0.1, window_size: int = 5) -> None:
        self.epsilon = float(epsilon)
        self.window_size = window_size

    def select(
        self,
        round_num: int,
        num_clients: int,
        clients_per_round: int,
        client_history: dict[int, dict[str, Any]],
        rng: np.random.Generator,
    ) -> SelectionResult:
        k_exploit = int(math.floor((1.0 - self.epsilon) * clients_per_round))
        k_explore = clients_per_round - k_exploit

        # Exploitation scores (moving average utility)
        scores = {}
        for c in range(num_clients):
            hist = client_history.get(c, {})
            utils = [u for u in hist.get("utility", []) if u is not None]
            recent = utils[-self.window_size:] if utils else []
            scores[c] = float(np.mean(recent)) if recent else 0.0

        sorted_clients = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        exploit_chosen = sorted_clients[:k_exploit]

        remaining = [c for c in range(num_clients) if c not in exploit_chosen]
        k_explore = min(k_explore, len(remaining))
        explore_chosen = rng.choice(remaining, size=k_explore, replace=False).tolist() if k_explore > 0 else []

        selected = exploit_chosen + explore_chosen
        return SelectionResult(
            selected_client_ids=selected,
            epsilon=self.epsilon,
            client_scores={c: {"score": scores[c]} for c in selected},
        )


class FedQualCPXSelector(BaseSelector):
    """B8: Proposed FedQual-CPX Client Selection (Sections 14-18).

    Components:
        1. Sequential change-point detector per client (CUSUM by default, or Page-Hinckley/EWMA)
        2. Warm-up period (W rounds uniform random)
        3. Adaptive exploration probability epsilon_t based on global change rate, uncertainty, coverage
        4. Exploitation scoring with uncertainty bonus
        5. Exploration scoring with staleness, uncertainty, change suspicion, and fairness deficit
        6. Change bonus for detected positive/negative state shifts
    """

    def __init__(
        self,
        detector_name: str = "cusum",
        warmup_rounds: int = 5,
        epsilon_min: float = 0.05,
        epsilon_max: float = 0.35,
        c1_change_rate: float = 0.4,
        c2_uncertainty: float = 0.2,
        c3_coverage: float = 0.2,
        gamma_pos: float = 1.0,
        gamma_neg: float = 0.5,
        delta: float = 0.5,
        h: float = 4.0,
        cooldown: int = 3,
        use_uncertainty: bool = True,
        use_adaptive_epsilon: bool = True,
        use_change_bonus: bool = True,
        **kwargs: Any,
    ) -> None:
        self.detector_name = detector_name
        self.warmup_rounds = warmup_rounds
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.c1 = c1_change_rate
        self.c2 = c2_uncertainty
        self.c3 = c3_coverage
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

        self.delta = delta
        self.h = h
        self.cooldown = cooldown
        self.detector_kwargs = kwargs

        # Ablation study flags (Section 31)
        self.use_uncertainty = use_uncertainty
        self.use_adaptive_epsilon = use_adaptive_epsilon
        self.use_change_bonus = use_change_bonus

        # Per-client detector instances
        self.detectors: dict[int, BaseDetector] = {}
        self.recent_events: list[ChangeEvent] = []

    def _ensure_detectors(self, num_clients: int) -> None:
        """Initialize detector instances for any newly encountered clients."""
        for c in range(num_clients):
            if c not in self.detectors:
                self.detectors[c] = create_detector(
                    self.detector_name,
                    delta=self.delta,
                    h=self.h,
                    cooldown=self.cooldown,
                    **self.detector_kwargs,
                )

    def select(
        self,
        round_num: int,
        num_clients: int,
        clients_per_round: int,
        client_history: dict[int, dict[str, Any]],
        rng: np.random.Generator,
    ) -> SelectionResult:
        self._ensure_detectors(num_clients)

        # ── 1. Warm-up Phase (Section 19) ──
        if round_num <= self.warmup_rounds:
            k = min(clients_per_round, num_clients)
            chosen = rng.choice(num_clients, size=k, replace=False).tolist()
            return SelectionResult(
                selected_client_ids=chosen,
                epsilon=1.0,
                change_events=[],
            )

        # ── 2. Run detectors on newly arrived observations ──
        # Process the last observation for each client from previous rounds
        new_events: list[ChangeEvent] = []
        for c in range(num_clients):
            hist = client_history.get(c, {})
            norm_utils = hist.get("normalized_utility", [])
            part = hist.get("participated", [])

            # If client participated in the most recent round (len-1)
            if part and part[-1] and norm_utils and norm_utils[-1] is not None:
                val = float(norm_utils[-1])
                evt = self.detectors[c].update(val, current_round=round_num - 1, client_id=c)
                if evt:
                    new_events.append(evt)
                    self.recent_events.append(evt)

        # Keep only recent events (within last 10 rounds)
        self.recent_events = [e for e in self.recent_events if round_num - e.round <= 10]

        # ── 3. Compute Exploration Rate epsilon_t (Section 14 & Ablation A1) ──
        if self.use_adaptive_epsilon:
            changed_clients = {e.client_id for e in self.recent_events}
            global_change_rate = len(changed_clients) / max(1, num_clients)
            obs_counts = [hist.get("total_participations", 0) for hist in client_history.values()]
            mean_uncertainty = np.mean([1.0 / math.sqrt(n + 1) for n in obs_counts])
            coverage_deficit = 1.0 - (np.sum(np.array(obs_counts) > 0) / max(1, num_clients))

            epsilon_raw = (
                self.epsilon_min
                + self.c1 * global_change_rate
                + self.c2 * mean_uncertainty
                + self.c3 * coverage_deficit
            )
            epsilon_t = float(np.clip(epsilon_raw, self.epsilon_min, self.epsilon_max))
        else:
            changed_clients = {e.client_id for e in self.recent_events}
            epsilon_t = self.epsilon_min

        # ── 4. Compute Exploitation and Exploration Scores (Sections 15-17) ──
        exploit_scores: dict[int, float] = {}
        explore_scores: dict[int, float] = {}
        score_details: dict[int, dict[str, float]] = {}

        max_round = max(1, round_num)

        for c in range(num_clients):
            hist = client_history.get(c, {})
            n_obs = hist.get("total_participations", 0)
            last_seen = hist.get("last_seen", None)
            staleness = (round_num - last_seen) if last_seen is not None else round_num
            norm_staleness = min(1.0, staleness / max_round)

            uncertainty = (1.0 / math.sqrt(n_obs + 1)) if self.use_uncertainty else 0.0
            fairness_deficit = max(0.0, 1.0 - (n_obs / max(1, (round_num * clients_per_round / num_clients))))

            # Estimated utility from recent observations
            valid_utils = [u for u in hist.get("normalized_utility", []) if u is not None]
            est_utility = float(np.mean(valid_utils[-3:])) if valid_utils else 0.0

            # Change suspicion: is this client suspect of a state change?
            det_state = self.detectors[c].get_state()
            change_suspect = 1.0 if c in changed_clients else 0.0

            # Change bonus: reward positive change, re-check negative change cautiously
            if self.use_change_bonus:
                pos_conf = 1.0 if (det_state.last_positive_change and round_num - det_state.last_positive_change <= 5) else 0.0
                neg_conf = 1.0 if (det_state.last_negative_change and round_num - det_state.last_negative_change <= 5) else 0.0
                change_bonus = self.gamma_pos * pos_conf - self.gamma_neg * neg_conf
            else:
                change_bonus = 0.0

            # Section 16: ExploitScore = estimated_utility + uncertainty_bonus + change_bonus
            e_score = est_utility + (0.5 * uncertainty) + change_bonus
            exploit_scores[c] = e_score

            # Section 15: ExploreScore = a*staleness + b*uncertainty + c*change_suspect + d*fairness
            # When change bonus is active, boost change_weight to ensure flagged clients
            # actually receive exploration priority rather than being suppressed by unobserved clients.
            change_weight = 1.00 if self.use_change_bonus else 0.20
            exp_score = (
                0.35 * norm_staleness
                + (0.35 * uncertainty if self.use_uncertainty else 0.0)
                + change_weight * change_suspect
                + 0.10 * fairness_deficit
            )
            explore_scores[c] = exp_score

            score_details[c] = {
                "exploit_score": round(e_score, 4),
                "explore_score": round(exp_score, 4),
                "epsilon": round(epsilon_t, 4),
            }

        # ── 5. Selection Mechanism (Section 18) ──
        k_exploit = int(math.floor((1.0 - epsilon_t) * clients_per_round))
        k_exploit = max(1, min(clients_per_round - 1, k_exploit))
        k_explore = clients_per_round - k_exploit

        sorted_exploit = sorted(exploit_scores.keys(), key=lambda c: exploit_scores[c], reverse=True)
        exploit_chosen = sorted_exploit[:k_exploit]

        remaining = [c for c in range(num_clients) if c not in exploit_chosen]
        sorted_explore = sorted(remaining, key=lambda c: explore_scores[c], reverse=True)
        explore_chosen = sorted_explore[:k_explore]

        selected = exploit_chosen + explore_chosen

        return SelectionResult(
            selected_client_ids=selected,
            epsilon=epsilon_t,
            change_events=new_events,
            client_scores={c: score_details[c] for c in selected},
        )


def create_selector(name: str, **kwargs: Any) -> BaseSelector:
    """Factory function for client selection strategies."""
    name_lower = name.lower()
    if name_lower in ("random", "b0", "fedavg"):
        return RandomSelector()
    elif name_lower in ("greedy", "b2", "utility_greedy"):
        return UtilityGreedySelector()
    elif name_lower in ("sliding_window", "b3", "window"):
        return SlidingWindowSelector(window_size=kwargs.get("window_size", 5))
    elif name_lower in ("fixed_exploration", "b4", "epsilon_greedy"):
        return FixedExplorationSelector(
            epsilon=kwargs.get("epsilon", 0.1),
            window_size=kwargs.get("window_size", 5),
        )
    elif name_lower in ("page_hinckley_adaptive", "b6", "ph_adaptive"):
        det_name = kwargs.pop("detector_name", "page_hinckley")
        return FedQualCPXSelector(detector_name=det_name, **kwargs)
    elif name_lower in ("fedqual_cpx", "b8", "fedqual"):
        det_name = kwargs.pop("detector_name", "cusum")
        return FedQualCPXSelector(detector_name=det_name, **kwargs)
    else:
        raise ValueError(f"Unknown selector '{name}'. Supported: random, greedy, sliding_window, fixed_exploration, fedqual_cpx, page_hinckley_adaptive")
