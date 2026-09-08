"""
Federated learning server for FedQual-CPX.

Manages the global model, client selection, aggregation, and evaluation.
This is the orchestration layer that ties together the FL pipeline.

Per Section 7:
    Selection at round t must only use information that is legitimately
    available before round t begins.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.fl.client import ClientResult
from src.fl.normalization import RobustNormalizer
from src.selection.selectors import BaseSelector, RandomSelector, SelectionResult


class FederatedServer:
    """Central server for federated learning simulation.

    Manages:
        - Global model state
        - Client selection (pluggable strategy)
        - Model aggregation
        - Global evaluation
        - Per-client history tracking
        - Robust utility normalization

    Args:
        global_model: The initial global model.
        test_loader: DataLoader for global test evaluation.
        device: Torch device.
        normalizer: Robust utility normalizer.
        selector: Client selection algorithm (default RandomSelector).
    """

    def __init__(
        self,
        global_model: nn.Module,
        test_loader: DataLoader,
        device: torch.device | None = None,
        normalizer: RobustNormalizer | None = None,
        selector: BaseSelector | None = None,
    ) -> None:
        self.global_model = global_model
        self.test_loader = test_loader
        self.device = device or torch.device("cpu")
        self.global_model.to(self.device)
        self.normalizer = normalizer or RobustNormalizer(z_max=3.0)
        self.selector = selector or RandomSelector()

        # Per-client history (Section 9)
        self.client_history: dict[int, dict[str, Any]] = {}
        self.current_round = 0

    def initialize_client_history(self, num_clients: int) -> None:
        """Initialize tracking state for all clients.

        Per Section 9: maintain rounds, utility, participated,
        detector_state, last_seen for every client.
        """
        for i in range(num_clients):
            self.client_history[i] = {
                "rounds": [],
                "utility": [],
                "normalized_utility": [],
                "participated": [],
                "last_seen": None,
                "total_participations": 0,
            }

    def select_clients(
        self,
        round_num: int,
        num_clients: int,
        clients_per_round: int,
        rng: np.random.Generator | None = None,
    ) -> SelectionResult:
        """Select clients for round_num using the configured selector policy.

        Args:
            round_num: Current communication round.
            num_clients: Total client population size N.
            clients_per_round: Communication budget K.
            rng: NumPy random generator.

        Returns:
            SelectionResult with selected client IDs, epsilon, and score diagnostics.
        """
        if rng is None:
            rng = np.random.default_rng(42 + round_num)

        return self.selector.select(
            round_num=round_num,
            num_clients=num_clients,
            clients_per_round=clients_per_round,
            client_history=self.client_history,
            rng=rng,
        )

    def select_clients_random(
        self,
        num_clients: int,
        clients_per_round: int,
        available_clients: list[int] | None = None,
    ) -> list[int]:
        """Random client selection (Baseline B0).

        Args:
            num_clients: Total number of clients.
            clients_per_round: How many to select (K).
            available_clients: Optional subset of available client IDs.

        Returns:
            List of selected client IDs.
        """
        if available_clients is None:
            available_clients = list(range(num_clients))

        k = min(clients_per_round, len(available_clients))
        selected = np.random.choice(
            available_clients, size=k, replace=False
        ).tolist()
        return selected

    def update_client_history(
        self,
        round_num: int,
        client_results: Sequence[ClientResult],
        all_client_ids: list[int],
    ) -> None:
        """Update per-client history after a round.

        For selected clients: record utility and update last_seen.
        For unselected clients: record as missing (NOT zero).

        Per Section 9 & 10:
            not selected ≠ low utility
            Do NOT impute unobserved utility as zero.
            Robust MAD normalization applied across contemporaneous round utilities.
        """
        selected_ids = {r.client_id for r in client_results}

        # Normalize contemporaneous utilities (Section 10)
        if client_results:
            raw_utils = [r.utility_loss_gain for r in client_results]
            norm_utils = self.normalizer.normalize_batch(raw_utils)
            norm_map = {r.client_id: float(norm_utils[idx]) for idx, r in enumerate(client_results)}
        else:
            norm_map = {}

        for client_id in all_client_ids:
            history = self.client_history[client_id]
            history["rounds"].append(round_num)

            if client_id in selected_ids:
                # Find this client's result
                result = next(r for r in client_results if r.client_id == client_id)
                history["utility"].append(result.utility_loss_gain)
                history["normalized_utility"].append(norm_map[client_id])
                history["participated"].append(True)
                history["last_seen"] = round_num
                history["total_participations"] += 1
            else:
                # Unselected: utility is MISSING, not zero
                history["utility"].append(None)
                history["normalized_utility"].append(None)
                history["participated"].append(False)

    @torch.no_grad()
    def evaluate_global_model(self) -> dict[str, float]:
        """Evaluate the global model on the test set.

        Returns:
            Dictionary with test_accuracy and test_loss.
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in self.test_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            outputs = self.global_model(batch_x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if outputs.dim() == 3:
                outputs = outputs[:, -1, :]

            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * len(batch_y)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += len(batch_y)

        self.global_model.train()

        return {
            "test_accuracy": correct / max(total, 1),
            "test_loss": total_loss / max(total, 1),
        }

    def get_participation_stats(self) -> dict[str, float]:
        """Compute participation fairness statistics (Section 35).

        Returns:
            Dictionary with Gini coefficient, entropy, coverage, etc.
        """
        counts = [
            h["total_participations"]
            for h in self.client_history.values()
        ]

        if not counts or sum(counts) == 0:
            return {"gini": 0.0, "entropy": 0.0, "coverage": 0.0}

        counts = np.array(counts, dtype=np.float64)
        n = len(counts)

        # Gini coefficient
        sorted_counts = np.sort(counts)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_counts) / (n * np.sum(sorted_counts))) - (n + 1) / n
        gini = max(0.0, gini)

        # Participation entropy
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))

        # Coverage: fraction of clients selected at least once
        coverage = np.sum(counts > 0) / n

        return {
            "gini": round(float(gini), 4),
            "entropy": round(float(entropy), 4),
            "coverage": round(float(coverage), 4),
            "min_participations": int(counts.min()),
            "max_participations": int(counts.max()),
        }

    @staticmethod
    def get_server_resources() -> dict[str, float]:
        """Capture server resource usage."""
        process = psutil.Process()
        return {
            "server_memory_mb": round(process.memory_info().rss / (1024 * 1024), 2),
        }
