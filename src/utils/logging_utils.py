"""
Experiment logging utilities for FedQual-CPX.

Provides structured CSV logging for per-round and per-client metrics,
matching the format specified in Section 41 of the implementation plan.

Usage:
    from src.utils.logging_utils import ExperimentLogger
    logger = ExperimentLogger(output_dir="results/raw/exp_001")
    logger.log_round(round_num=1, test_accuracy=0.45, ...)
    logger.log_client(round_num=1, client_id=3, utility=0.12, ...)
    logger.close()
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


# Column definitions per Section 41 of the plan
ROUND_COLUMNS = [
    "round",
    "test_accuracy",
    "test_loss",
    "num_selected",
    "total_clients",
    "epsilon",
    "change_events",
    "selection_time_ms",
    "server_memory_mb",
]

CLIENT_COLUMNS = [
    "round",
    "client_id",
    "selected",
    "utility",
    "normalized_utility",
    "detector",
    "cusum_pos",
    "cusum_neg",
    "change_detected",
    "change_direction",
    "last_seen",
    "age",
    "explore_score",
    "exploit_score",
    "final_score",
]


class ExperimentLogger:
    """Structured logger for federated learning experiments.

    Creates and manages CSV log files for round-level and client-level metrics.
    All files are created in the specified output directory.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Open CSV files
        self._round_file = open(
            self.output_dir / "global_metrics.csv", "w", newline="", encoding="utf-8"
        )
        self._round_writer = csv.DictWriter(
            self._round_file, fieldnames=ROUND_COLUMNS, extrasaction="ignore"
        )
        self._round_writer.writeheader()

        self._client_file = open(
            self.output_dir / "client_events.csv", "w", newline="", encoding="utf-8"
        )
        self._client_writer = csv.DictWriter(
            self._client_file, fieldnames=CLIENT_COLUMNS, extrasaction="ignore"
        )
        self._client_writer.writeheader()

        # In-memory history for quick access
        self.round_history: list[dict[str, Any]] = []

    def log_round(self, **kwargs: Any) -> None:
        """Log a single round's global metrics.

        Args:
            **kwargs: Metric key-value pairs matching ROUND_COLUMNS.
        """
        self._round_writer.writerow(kwargs)
        self._round_file.flush()
        self.round_history.append(kwargs)

    def log_client(self, **kwargs: Any) -> None:
        """Log a single client's per-round metrics.

        Args:
            **kwargs: Metric key-value pairs matching CLIENT_COLUMNS.
        """
        self._client_writer.writerow(kwargs)
        self._client_file.flush()

    def log_clients_batch(self, records: list[dict[str, Any]]) -> None:
        """Log multiple client records at once.

        Args:
            records: List of dicts, each matching CLIENT_COLUMNS.
        """
        for record in records:
            self._client_writer.writerow(record)
        self._client_file.flush()

    def save_summary(self, summary: dict[str, Any]) -> None:
        """Save an experiment summary JSON file.

        Args:
            summary: Dictionary with experiment summary information.
        """
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

    def close(self) -> None:
        """Close all open log files."""
        self._round_file.close()
        self._client_file.close()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
