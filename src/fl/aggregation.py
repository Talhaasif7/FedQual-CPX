"""
FedAvg aggregation for FedQual-CPX.

Implements sample-count weighted averaging as specified in Section 29.

    w_global = Σ (n_i / n_total) * w_i

where n_i is the number of training samples on client i.
"""

from __future__ import annotations

import copy
from typing import Sequence

import torch
import torch.nn as nn

from src.fl.client import ClientResult


def fedavg_aggregate(
    global_model: nn.Module,
    client_results: Sequence[ClientResult],
) -> nn.Module:
    """Aggregate client model updates using Federated Averaging.

    Uses sample-count weighted averaging. Each client's contribution is
    proportional to its local dataset size.

    Args:
        global_model: The current global model (used as template).
        client_results: List of ClientResult from selected clients.

    Returns:
        Updated global model with aggregated weights.
    """
    if not client_results:
        return global_model

    # Compute total samples across all participating clients
    total_samples = sum(r.num_samples for r in client_results)

    if total_samples == 0:
        return global_model

    # Initialize aggregated state dict
    aggregated_state = {}
    global_state = global_model.state_dict()

    for key in global_state:
        aggregated_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)

    # Weighted sum
    for result in client_results:
        weight = result.num_samples / total_samples
        for key in aggregated_state:
            aggregated_state[key] += weight * result.model_state[key].float()

    # Load aggregated weights into model
    updated_model = copy.deepcopy(global_model)
    updated_model.load_state_dict(aggregated_state)

    return updated_model
