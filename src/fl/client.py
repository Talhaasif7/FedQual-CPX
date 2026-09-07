"""
Federated client implementation for FedQual-CPX.

Each client:
    1. Receives the global model
    2. Trains locally for E epochs
    3. Computes utility statistics
    4. Returns model update + metadata

Per Section 7 & 8 of the plan:
    - Utility is computed AFTER participation (never before selection)
    - Selection at round t uses only information available before round t
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class ClientResult:
    """Result returned by a client after local training.

    Contains the model update, utility signals, and metadata.
    The server uses this to aggregate and update client histories.
    """
    client_id: int
    model_state: dict[str, torch.Tensor]
    num_samples: int

    # Utility signals (Section 8)
    loss_before: float = 0.0        # loss on local data BEFORE training
    loss_after: float = 0.0         # loss on local data AFTER training
    utility_loss_gain: float = 0.0  # loss_before - loss_after (Utility A/B)

    # Optional validation utility
    val_loss_before: float = 0.0
    val_loss_after: float = 0.0
    val_accuracy: float = 0.0

    # Training metadata
    train_time_ms: float = 0.0


class FederatedClient:
    """A single federated learning client.

    Performs local training on its own data partition and computes
    utility statistics for the server's selection mechanism.

    Args:
        client_id: Unique client identifier.
        train_loader: DataLoader for local training data.
        val_loader: Optional DataLoader for local validation data.
        device: Torch device for training.
    """

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cpu")

    def train(
        self,
        global_model: nn.Module,
        local_epochs: int = 2,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        optimizer_name: str = "sgd",
    ) -> ClientResult:
        """Perform local training starting from the global model.

        Steps:
            1. Copy global model
            2. Evaluate loss BEFORE training (for utility computation)
            3. Train for E local epochs
            4. Evaluate loss AFTER training
            5. Compute utility = loss_before - loss_after
            6. Return updated model state + utility

        Args:
            global_model: The current global model.
            local_epochs: Number of local training epochs.
            lr: Learning rate.
            momentum: SGD momentum.
            weight_decay: L2 regularization.
            optimizer_name: 'sgd' or 'adam'.

        Returns:
            ClientResult with model state, utility, and metadata.
        """
        start_time = time.perf_counter()

        # Deep copy the global model for local training
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()

        # Create optimizer
        if optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                local_model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.SGD(
                local_model.parameters(), lr=lr,
                momentum=momentum, weight_decay=weight_decay
            )

        criterion = nn.CrossEntropyLoss()

        # ── Step 1: Evaluate BEFORE training ──
        loss_before = self._evaluate_loss(local_model, criterion)

        val_loss_before = 0.0
        if self.val_loader is not None:
            val_loss_before = self._evaluate_loss(
                local_model, criterion, loader=self.val_loader
            )

        # ── Step 2: Local training ──
        total_samples = 0
        for epoch in range(local_epochs):
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = local_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_samples += len(batch_y)

        # ── Step 3: Evaluate AFTER training ──
        loss_after = self._evaluate_loss(local_model, criterion)

        val_loss_after = 0.0
        val_accuracy = 0.0
        if self.val_loader is not None:
            val_loss_after = self._evaluate_loss(
                local_model, criterion, loader=self.val_loader
            )
            val_accuracy = self._evaluate_accuracy(
                local_model, loader=self.val_loader
            )

        # ── Step 4: Compute utility ──
        utility_loss_gain = loss_before - loss_after

        train_time = (time.perf_counter() - start_time) * 1000  # ms

        return ClientResult(
            client_id=self.client_id,
            model_state=local_model.state_dict(),
            num_samples=len(self.train_loader.dataset),
            loss_before=loss_before,
            loss_after=loss_after,
            utility_loss_gain=utility_loss_gain,
            val_loss_before=val_loss_before,
            val_loss_after=val_loss_after,
            val_accuracy=val_accuracy,
            train_time_ms=train_time,
        )

    @torch.no_grad()
    def _evaluate_loss(
        self,
        model: nn.Module,
        criterion: nn.Module,
        loader: DataLoader | None = None,
    ) -> float:
        """Compute average loss on a data loader."""
        model.eval()
        if loader is None:
            loader = self.train_loader

        total_loss = 0.0
        total_samples = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * len(batch_y)
            total_samples += len(batch_y)

        model.train()
        return total_loss / max(total_samples, 1)

    @torch.no_grad()
    def _evaluate_accuracy(
        self,
        model: nn.Module,
        loader: DataLoader | None = None,
    ) -> float:
        """Compute accuracy on a data loader."""
        model.eval()
        if loader is None:
            loader = self.train_loader

        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += len(batch_y)

        model.train()
        return correct / max(total, 1)
