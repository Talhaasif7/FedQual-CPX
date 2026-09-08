import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import torch
from torch.utils.data import TensorDataset

from src.data.drift import DriftConfig, DriftManager, DriftedDataset


def test_drift_inactive_before_tau():
    """Verify that drift is inactive when round < drift_round."""
    cfg = DriftConfig(
        enabled=True,
        drift_type="class_swap",
        drift_round=50,
        drift_clients=[0, 1],
    )
    dm = DriftManager(num_clients=10, config=cfg)

    # Client 0 should NOT be drifted at round 10
    assert not dm.is_client_drifted(0, current_round=10)
    # Client 0 SHOULD be drifted at round 50 and 60
    assert dm.is_client_drifted(0, current_round=50)
    assert dm.is_client_drifted(0, current_round=60)
    # Client 2 (not in drift list) should never be drifted
    assert not dm.is_client_drifted(2, current_round=60)


def test_drifted_dataset_wrapping():
    """Verify that dataset transforms are applied only when active."""
    x = torch.zeros(10, 3, 32, 32)
    y = torch.zeros(10, dtype=torch.long)
    base_ds = TensorDataset(x, y)

    cfg = DriftConfig(
        enabled=True,
        drift_type="class_swap",
        drift_round=10,
        drift_clients=[0],
        severity="small",
    )
    dm = DriftManager(num_clients=5, config=cfg)
    drifted_ds = dm.wrap_client_dataset(0, base_ds, current_round=5)

    # At round 5, drift is not active
    sample_x, sample_y = drifted_ds[0]
    assert sample_y == 0

    # Activate drift
    drifted_ds.set_active(True)
    sample_x, sample_y = drifted_ds[0]
    # Label should either swap or be preserved depending on the pair; if swapped, it changes
    # Now test feature noise:
    cfg_noise = DriftConfig(
        enabled=True,
        drift_type="feature_noise",
        drift_round=5,
        drift_clients=[1],
        severity="medium",
    )
    dm_noise = DriftManager(num_clients=5, config=cfg_noise)
    ds_noise = dm_noise.wrap_client_dataset(1, base_ds, current_round=1)
    # Inactive
    x_inact, _ = ds_noise[0]
    assert torch.all(x_inact == 0.0)

    # Active
    ds_noise.set_active(True)
    x_act, _ = ds_noise[0]
    assert not torch.all(x_act == 0.0)


def test_non_drift_control():
    """Verify that when drift is disabled, no clients drift."""
    cfg = DriftConfig(enabled=False)
    dm = DriftManager(num_clients=10, config=cfg)
    for c in range(10):
        assert not dm.is_client_drifted(c, current_round=100)
    assert dm.get_drift_info()["enabled"] is False


def test_feature_shift_alias():
    """Verify that feature_shift alias works equivalently to feature_noise and clamps to [0, 1]."""
    x = torch.zeros(10, 3, 32, 32)
    y = torch.zeros(10, dtype=torch.long)
    base_ds = TensorDataset(x, y)

    cfg = DriftConfig(
        enabled=True,
        drift_type="feature_shift",
        drift_round=10,
        drift_clients=[0],
        severity="medium",
    )
    dm = DriftManager(num_clients=5, config=cfg)
    ds = dm.wrap_client_dataset(0, base_ds, current_round=15)
    assert ds.is_active
    sample_x, sample_y = ds[0]
    assert not torch.all(sample_x == 0.0)
    assert torch.all(sample_x >= 0.0) and torch.all(sample_x <= 1.0)


def test_gradual_drift_probability_interpolation():
    """Verify that gradual drift linearly scales drift probability between tau_start and tau_end."""
    x = torch.zeros(100, 3, 32, 32)
    y = torch.zeros(100, dtype=torch.long)
    base_ds = TensorDataset(x, y)

    cfg = DriftConfig(
        enabled=True,
        drift_type="gradual_drift",
        drift_round=30,
        drift_end_round=70,
        drift_clients=[0],
        severity="medium",
    )
    dm = DriftManager(num_clients=5, config=cfg)
    ds = dm.wrap_client_dataset(0, base_ds, current_round=20)
    assert not ds.is_active
    assert ds.drift_prob == 0.0

    # At round 50 (midway between 30 and 70), prob should be 0.5
    client_dict = {0: ds}
    dm.update_round(50, client_dict)
    assert ds.is_active
    assert abs(ds.drift_prob - 0.5) < 1e-4

    # At round 70, prob should be 1.0
    dm.update_round(70, client_dict)
    assert ds.is_active
    assert abs(ds.drift_prob - 1.0) < 1e-4


if __name__ == "__main__":
    test_drift_inactive_before_tau()
    test_drifted_dataset_wrapping()
    test_non_drift_control()
    test_feature_shift_alias()
    test_gradual_drift_probability_interpolation()
    print("All drift unit tests passed!")
