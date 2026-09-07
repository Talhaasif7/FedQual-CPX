"""
Unit tests for client selectors (Section 30, 52).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.selection.selectors import (
    RandomSelector,
    UtilityGreedySelector,
    SlidingWindowSelector,
    FixedExplorationSelector,
    FedQualCPXSelector,
    create_selector,
)


def make_mock_history(num_clients: int, num_rounds: int = 5):
    """Generate mock history for tests."""
    history = {}
    for c in range(num_clients):
        history[c] = {
            "rounds": list(range(1, num_rounds + 1)),
            "utility": [float(c * 0.1) if r % 2 == 0 else None for r in range(1, num_rounds + 1)],
            "normalized_utility": [float(c * 0.2) if r % 2 == 0 else None for r in range(1, num_rounds + 1)],
            "participated": [r % 2 == 0 for r in range(1, num_rounds + 1)],
            "last_seen": num_rounds if (num_rounds % 2 == 0) else (num_rounds - 1),
            "total_participations": num_rounds // 2,
        }
    return history


def test_random_selector():
    rng = np.random.default_rng(42)
    sel = RandomSelector()
    res = sel.select(1, 20, 5, {}, rng)
    assert len(res.selected_client_ids) == 5
    assert len(set(res.selected_client_ids)) == 5
    assert all(0 <= c < 20 for c in res.selected_client_ids)


def test_utility_greedy_selector():
    rng = np.random.default_rng(42)
    history = make_mock_history(10)
    sel = UtilityGreedySelector()
    res = sel.select(6, 10, 3, history, rng)
    # Highest utility should be clients 9, 8, 7
    assert res.selected_client_ids == [9, 8, 7]


def test_fixed_exploration_selector():
    rng = np.random.default_rng(42)
    history = make_mock_history(10)
    sel = FixedExplorationSelector(epsilon=0.3)
    res = sel.select(6, 10, 4, history, rng)
    assert len(res.selected_client_ids) == 4
    assert len(set(res.selected_client_ids)) == 4


def test_fedqual_cpx_selector():
    rng = np.random.default_rng(42)
    sel = FedQualCPXSelector(warmup_rounds=3, epsilon_min=0.1, epsilon_max=0.4)

    # Warmup round
    res_warm = sel.select(2, 20, 5, {}, rng)
    assert len(res_warm.selected_client_ids) == 5
    assert res_warm.epsilon == 1.0

    # Post warmup
    history = make_mock_history(20, num_rounds=5)
    res_post = sel.select(6, 20, 5, history, rng)
    assert len(res_post.selected_client_ids) == 5
    assert len(set(res_post.selected_client_ids)) == 5
    assert 0.1 <= res_post.epsilon <= 0.4


if __name__ == "__main__":
    test_random_selector()
    test_utility_greedy_selector()
    test_fixed_exploration_selector()
    test_fedqual_cpx_selector()
    print("All selector unit tests passed!")
