"""
Federated learning simulation engine for FedQual-CPX.

Implements the complete FL round loop per Section 7:
    1. Server has global model
    2. Server chooses K clients
    3. Selected clients receive global model
    4. Each client trains locally
    5. Each client computes utility statistics
    6. Server receives updates + metadata
    7. Server aggregates updates
    8. Server evaluates global model
    9. Server updates history/detectors
    10. Server selects clients for next round

Usage:
    from src.fl.simulator import FederatedSimulator
    sim = FederatedSimulator(config)
    sim.run()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.drift import DriftConfig, DriftManager, DriftedDataset
from src.data.loaders import (
    CIFAR10Dataset,
    FEMNISTDataset,
    ShakespeareDataset,
    FederatedClientDataset,
    get_client_dataloader,
)
from src.data.partition import DirichletPartitioner
from src.fl.aggregation import fedavg_aggregate
from src.fl.client import FederatedClient
from src.fl.normalization import RobustNormalizer, create_normalizer
from src.fl.server import FederatedServer
from src.models.cnn import create_model
from src.selection.selectors import BaseSelector, SelectionResult, create_selector
from src.utils.config import Config, save_config, get_reproducibility_info
from src.utils.logging_utils import ExperimentLogger
from src.utils.seed import set_seed, get_device


class FederatedSimulator:
    """Complete federated learning simulation engine.

    Orchestrates the entire FL experiment: dataset loading, partitioning,
    client creation, round execution, evaluation, and logging.

    Args:
        cfg: Experiment configuration (loaded from YAML).
        experiment_id: Unique identifier for this experiment run.
    """

    def __init__(self, cfg: Config, experiment_id: str = "default") -> None:
        self.cfg = cfg
        self.experiment_id = experiment_id

        # Set seed for reproducibility
        self.seed = cfg.get("seed", 42)
        set_seed(self.seed)

        # Device
        self.device = get_device()
        print(f"Device: {self.device}")

        # Prepare output directory
        results_dir = cfg.get("output", Config({"results_dir": "results/raw"})).results_dir
        self.output_dir = Path(results_dir) / experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components (lazy — built in setup())
        self.rng = np.random.default_rng(self.seed)
        self.train_dataset: CIFAR10Dataset | None = None
        self.test_dataset: CIFAR10Dataset | None = None
        self.client_indices: dict[int, np.ndarray] = {}
        self.clients: dict[int, FederatedClient] = {}
        self.drift_manager: DriftManager | None = None
        self.drifted_train_datasets: dict[int, DriftedDataset] = {}
        self.server: FederatedServer | None = None
        self.logger: ExperimentLogger | None = None

    def setup(self) -> None:
        """Initialize all components: data, partitions, model, server, logger."""
        print("=" * 60)
        print(f"FedQual-CPX Experiment: {self.experiment_id}")
        print(f"Seed: {self.seed}")
        print("=" * 60)

        # ── 1. Load dataset ──
        print("\n[1/5] Loading dataset ...")
        dataset_cfg = self.cfg.dataset
        dname = dataset_cfg.name.lower()
        if dname == "femnist":
            self.train_dataset = FEMNISTDataset(
                processed_dir=dataset_cfg.processed_dir, train=True
            )
            self.test_dataset = FEMNISTDataset(
                processed_dir=dataset_cfg.processed_dir, train=False
            )
        elif dname == "shakespeare":
            self.train_dataset = ShakespeareDataset(
                processed_dir=dataset_cfg.processed_dir, train=True
            )
            self.test_dataset = ShakespeareDataset(
                processed_dir=dataset_cfg.processed_dir, train=False
            )
        else:
            self.train_dataset = CIFAR10Dataset(
                processed_dir=dataset_cfg.processed_dir, train=True
            )
            self.test_dataset = CIFAR10Dataset(
                processed_dir=dataset_cfg.processed_dir, train=False
            )
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Test:  {len(self.test_dataset)} samples")

        # ── 2. Partition data ──
        print("\n[2/5] Partitioning data ...")
        part_cfg = self.cfg.partition
        partitioner = DirichletPartitioner(
            num_clients=part_cfg.num_clients,
            alpha=part_cfg.alpha,
            seed=part_cfg.get("seed", self.seed),
            min_samples_per_client=part_cfg.get("min_samples_per_client", 10),
        )

        labels = self.train_dataset.labels.numpy()
        self.client_indices = partitioner.partition(labels)

        # Save partition
        partition_dir = Path("data/partitions") / dname / f"alpha{part_cfg.alpha}_seed{self.seed}"
        partitioner.save_partition(self.client_indices, labels, partition_dir)

        # ── 3. Create clients with drift wrapper ──
        print("\n[3/5] Creating federated clients ...")
        train_cfg = self.cfg.training
        val_fraction = part_cfg.get("val_fraction", 0.1)

        drift_cfg = self.cfg.get("drift", Config({"enabled": False}))
        drift_dict = drift_cfg.to_dict() if hasattr(drift_cfg, "to_dict") else (drift_cfg if isinstance(drift_cfg, dict) else {})
        self.drift_manager = DriftManager(
            num_clients=len(self.client_indices),
            config=drift_dict,
            num_classes=self.cfg.dataset.get("num_classes", 10),
            seed=self.seed,
        )
        print(f"  Drift status: {self.drift_manager.get_drift_info()}")

        for client_id, indices in self.client_indices.items():
            # Train split
            raw_train_ds = FederatedClientDataset(
                self.train_dataset, indices,
                val_fraction=val_fraction, split="train", seed=self.seed
            )
            # Wrap with dynamic drift transformation
            drifted_train_ds = self.drift_manager.wrap_client_dataset(
                client_id, raw_train_ds, current_round=0
            )
            self.drifted_train_datasets[client_id] = drifted_train_ds

            train_loader = get_client_dataloader(
                drifted_train_ds, batch_size=train_cfg.batch_size, shuffle=True
            )

            # Val split
            val_ds = FederatedClientDataset(
                self.train_dataset, indices,
                val_fraction=val_fraction, split="val", seed=self.seed
            )
            val_loader = get_client_dataloader(
                val_ds, batch_size=train_cfg.batch_size, shuffle=False
            )

            self.clients[client_id] = FederatedClient(
                client_id=client_id,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
            )

        print(f"  Created {len(self.clients)} clients")

        # ── 4. Create model, selector, and server ──
        print("\n[4/5] Initializing model and server ...")
        model_cfg = self.cfg.model
        global_model = create_model(
            name=model_cfg.name,
            num_classes=model_cfg.num_classes,
        )
        print(f"  Model: {model_cfg.name}")
        total_params = sum(p.numel() for p in global_model.parameters())
        print(f"  Parameters: {total_params:,}")

        # Test loader
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=train_cfg.batch_size * 4,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        norm_cfg = self.cfg.get("normalization", Config({"method": "robust_mad", "z_max": 3.0, "epsilon": 1e-8}))
        normalizer = create_normalizer(
            method=norm_cfg.get("method", "robust_mad"),
            z_max=norm_cfg.get("z_max", 3.0),
            epsilon=norm_cfg.get("epsilon", 1e-8),
        )

        sel_cfg = self.cfg.get("selection", Config({"method": "random"}))
        sel_dict = sel_cfg.to_dict() if hasattr(sel_cfg, "to_dict") else (sel_cfg if isinstance(sel_cfg, dict) else {})
        method_name = sel_dict.pop("method", "random")
        selector = create_selector(method_name, **sel_dict)
        print(f"  Selection policy: {method_name}")

        self.server = FederatedServer(
            global_model=global_model,
            test_loader=test_loader,
            device=self.device,
            normalizer=normalizer,
            selector=selector,
        )
        self.server.initialize_client_history(len(self.clients))

        # ── 5. Logger ──
        print("\n[5/5] Setting up logging ...")
        self.logger = ExperimentLogger(self.output_dir)

        # Save config for reproducibility
        save_config(self.cfg, self.output_dir / "config.yaml")

        print(f"  Output: {self.output_dir}")
        print("\nSetup complete.\n")

    def run(self) -> dict[str, Any]:
        """Execute the full federated learning experiment.

        Returns:
            Summary dictionary with final metrics.
        """
        if self.server is None:
            self.setup()

        fl_cfg = self.cfg.fl
        train_cfg = self.cfg.training
        num_rounds = fl_cfg.num_rounds
        clients_per_round = fl_cfg.clients_per_round
        all_client_ids = list(range(fl_cfg.num_clients))

        print(f"Starting training: {num_rounds} rounds, "
              f"K={clients_per_round} clients/round\n")

        best_accuracy = 0.0
        experiment_start = time.perf_counter()

        for round_num in range(1, num_rounds + 1):
            round_start = time.perf_counter()

            # ── Step 0: Update dynamic drift state ──
            if self.drift_manager is not None:
                self.drift_manager.update_round(round_num, self.drifted_train_datasets)

            # ── Step 1: Select clients using configured policy ──
            selection_start = time.perf_counter()
            sel_result = self.server.select_clients(
                round_num=round_num,
                num_clients=fl_cfg.num_clients,
                clients_per_round=clients_per_round,
                rng=self.rng,
            )
            selected = sel_result.selected_client_ids
            selection_time = (time.perf_counter() - selection_start) * 1000

            # ── Step 2: Local training ──
            client_results = []
            for client_id in selected:
                result = self.clients[client_id].train(
                    global_model=self.server.global_model,
                    local_epochs=train_cfg.local_epochs,
                    lr=train_cfg.learning_rate,
                    momentum=train_cfg.get("momentum", 0.9),
                    weight_decay=train_cfg.get("weight_decay", 0.0001),
                    optimizer_name=train_cfg.optimizer,
                )
                client_results.append(result)

            # ── Step 3: Aggregate ──
            self.server.global_model = fedavg_aggregate(
                self.server.global_model, client_results
            )

            # ── Step 4: Evaluate ──
            eval_cfg = self.cfg.get("evaluation", Config({"test_every": 1}))
            test_every = eval_cfg.get("test_every", 1)

            metrics = {}
            if round_num % test_every == 0 or round_num == num_rounds:
                metrics = self.server.evaluate_global_model()
                best_accuracy = max(best_accuracy, metrics["test_accuracy"])

            # ── Step 5: Update history ──
            self.server.update_client_history(
                round_num, client_results, all_client_ids
            )

            # ── Step 6: Log ──
            resources = self.server.get_server_resources()
            avg_utility = np.mean([r.utility_loss_gain for r in client_results])

            self.logger.log_round(
                round=round_num,
                test_accuracy=metrics.get("test_accuracy", ""),
                test_loss=metrics.get("test_loss", ""),
                num_selected=len(selected),
                total_clients=fl_cfg.num_clients,
                epsilon=round(sel_result.epsilon, 4),
                change_events=len(sel_result.change_events),
                selection_time_ms=round(selection_time, 2),
                server_memory_mb=resources["server_memory_mb"],
            )

            # Log per-client info for selected clients
            for result in client_results:
                norm_u = self.server.client_history[result.client_id]["normalized_utility"][-1]
                scores = sel_result.client_scores.get(result.client_id, {})

                client_evts = [e for e in sel_result.change_events if e.client_id == result.client_id]
                change_detected = len(client_evts) > 0
                change_direction = client_evts[0].direction if client_evts else ""
                detector_name = client_evts[0].detector if client_evts else "none"

                self.logger.log_client(
                    round=round_num,
                    client_id=result.client_id,
                    selected=True,
                    utility=round(result.utility_loss_gain, 6),
                    normalized_utility=round(norm_u, 4) if norm_u is not None else "",
                    detector=detector_name,
                    cusum_pos="",
                    cusum_neg="",
                    change_detected=change_detected,
                    change_direction=change_direction,
                    last_seen=round_num,
                    age=0,
                    explore_score=scores.get("explore_score", ""),
                    exploit_score=scores.get("exploit_score", ""),
                    final_score=scores.get("score", ""),
                )

            # ── Print progress ──
            round_time = (time.perf_counter() - round_start) * 1000
            if metrics:
                print(
                    f"Round {round_num:3d}/{num_rounds} | "
                    f"Acc: {metrics['test_accuracy']:.4f} | "
                    f"Loss: {metrics['test_loss']:.4f} | "
                    f"Utility: {avg_utility:.4f} | "
                    f"Time: {round_time:.0f}ms"
                )

        # ── Finalize ──
        total_time = (time.perf_counter() - experiment_start)
        participation = self.server.get_participation_stats()

        summary = {
            "experiment_id": self.experiment_id,
            "best_accuracy": round(best_accuracy, 4),
            "final_accuracy": round(metrics.get("test_accuracy", 0.0), 4),
            "total_rounds": num_rounds,
            "total_time_seconds": round(total_time, 2),
            "participation": participation,
            "reproducibility": get_reproducibility_info(),
            "config": self.cfg.to_dict(),
        }

        self.logger.save_summary(summary)
        self.logger.close()

        print(f"\n{'=' * 60}")
        print(f"Experiment complete: {self.experiment_id}")
        print(f"  Best accuracy:  {best_accuracy:.4f}")
        print(f"  Final accuracy: {metrics.get('test_accuracy', 0.0):.4f}")
        print(f"  Total time:     {total_time:.1f}s")
        print(f"  Participation Gini: {participation['gini']}")
        print(f"  Results saved to: {self.output_dir}")
        print(f"{'=' * 60}")

        return summary
