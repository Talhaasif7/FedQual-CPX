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


if __name__ == "__main__":
    test_drift_inactive_before_tau()
    test_drifted_dataset_wrapping()
    test_non_drift_control()
    print("All drift unit tests passed!")
