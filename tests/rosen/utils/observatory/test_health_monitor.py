from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from rosen.core.train import TrainerComponents, TrainerDataLoaders
from rosen.core.trainers import BaseTrainerWithLife
from rosen.models.base import BaseModel
from tests.support.model_fixtures import DummyCls
from rosen.utils.observatory import HealthMonitor, HealthMonitorConfig


class HealthToyModel(BaseModel):
    def __init__(self):
        super().__init__(DummyCls.MANAGER)
        self.tower = torch.nn.Sequential(
            torch.nn.Linear(1, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
        )

    def _feature(self, interaction):
        return interaction[DummyCls.USER_ID].float().view(-1, 1) / 10.0

    def calculate_loss(self, interaction):
        logits = self.tower(self._feature(interaction)).view(-1)
        labels = interaction[DummyCls.LABEL].float().view(-1)
        task = F.binary_cross_entropy_with_logits(logits, labels)
        if self.lsp.active:
            self.lsp.add("task", task)
        return task

    def predict(self, interaction):
        return torch.sigmoid(self.tower(self._feature(interaction)).view(-1))


class _DomainRoutedTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.domain_networks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU())
            for _ in range(3)
        ])

    def forward(self, x, domain_ids):
        outputs = torch.stack([network(x) for network in self.domain_networks], dim=1)
        rows = torch.arange(x.size(0), device=x.device)
        return outputs[rows, domain_ids.long()]


class DomainRoutedToyModel(BaseModel):
    def __init__(self):
        super().__init__(DummyCls.MANAGER)
        self.tower = _DomainRoutedTower()

    def calculate_loss(self, interaction):
        x = torch.stack([
            interaction[DummyCls.USER_ID].float(),
            interaction[DummyCls.ITEM_ID].float(),
        ], dim=-1)
        output = self.tower(x, interaction[DummyCls.DOMAIN])
        task = output.square().mean()
        if self.lsp.active:
            self.lsp.add("task", task)
        return task

    def predict(self, interaction):
        raise NotImplementedError


def _cfg():
    return SimpleNamespace(
        device="cpu",
        train_log_interval=100,
        max_epochs=1,
        ckpt_dir="",
        experiment_name="health",
        dataset_name="dummy",
    )


def test_default_snapshot_policy_starts_at_step_100():
    monitor = HealthMonitor()

    assert not monitor.should_capture(SimpleNamespace(global_step=0))
    assert not monitor.should_capture(SimpleNamespace(global_step=98))
    assert monitor.should_capture(SimpleNamespace(global_step=99))


def test_health_monitor_captures_aligned_module_and_point_loss_snapshots(tmp_path):
    jsonl = tmp_path / "health.jsonl"
    monitor = HealthMonitor(
        HealthMonitorConfig(
            snapshot_every=2,
            capture_first_step=False,
            output_jsonl=str(jsonl),
        )
    )
    monitor.observe_module("tower", prefix="tower", include_root=True)
    model = HealthToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = [DummyCls.make_interaction(batch_size=8) for _ in range(2)]
    trainer = BaseTrainerWithLife(
        model,
        optimizer,
        DummyCls.MANAGER,
        TrainerDataLoaders(train=batches, valid=[]),
        TrainerComponents(health_monitor=monitor),
        _cfg(),
    )

    trainer.train_epoch()

    assert len(monitor.store) == 1
    snapshot = monitor.store.latest()
    assert snapshot is not None
    assert snapshot.step == 2
    assert snapshot.interval_start == 2
    assert snapshot.interval_end == 2
    assert snapshot.signals["loss.task"].scope == "point"
    assert snapshot.signals["loss.task"].metrics["raw"] > 0.0
    assert snapshot.signals["loss.task"].metrics["weight"] == 1.0
    assert model.lsp.summary()["settlements"] == {}
    layer_signals = [
        signal for name, signal in snapshot.signals.items()
        if ".tower.layers." in name and signal.metrics.get("flat_dim", 0) > 1
    ]
    assert layer_signals
    assert all("effective_rank" in signal.metrics for signal in layer_signals)
    assert all(not isinstance(value, torch.Tensor) for signal in layer_signals for value in signal.metrics.values())
    payload = json.loads(jsonl.read_text(encoding="utf-8").strip())
    assert payload["snapshot_id"] == snapshot.snapshot_id

    csv_path = monitor.export_csv(tmp_path / "health.csv")
    assert csv_path.exists()
    assert "effective_rank" in csv_path.read_text(encoding="utf-8")


def test_explicit_probe_records_the_trained_eval_model_mode():
    monitor = HealthMonitor()
    monitor.observe_module("tower", prefix="tower", include_root=True)
    model = HealthToyModel().eval()
    batch = DummyCls.make_interaction(batch_size=8)
    ctx = SimpleNamespace(
        global_step=7,
        epoch=1,
        batch_idx=-1,
        stage="trained_audit",
        phase="default",
    )
    model.lsp.start_step(step=7, epoch=1, batch_idx=-1, stage="trained_audit")

    with torch.no_grad(), monitor.capture_scope(model, ctx, force=True), model.lsp.scope():
        loss = model.calculate_loss(batch)
        model.lsp.settle(loss)
    snapshot = monitor.record_probe(model, ctx)

    assert snapshot is not None
    assert snapshot.step == 7
    assert snapshot.stage == "trained_audit"
    assert snapshot.model_mode == "eval"
    assert "loss.task" in snapshot.signals
    assert any(".tower." in name for name in snapshot.signals)


def test_zero_energy_tensor_has_finite_zero_rank_metrics():
    monitor = HealthMonitor()

    metrics = monitor._tensor_metrics(torch.zeros(8, 4), ("basic", "rank"))

    for name in (
        "effective_rank",
        "rank_ratio",
        "participation_ratio",
        "stable_rank",
        "top1_energy_ratio",
    ):
        assert torch.isfinite(torch.tensor(metrics[name]))
    assert metrics["top1_energy_ratio"] == 0.0


def test_empty_domain_tensor_keeps_its_flat_dimension():
    monitor = HealthMonitor()

    metrics = monitor._tensor_metrics(torch.empty(0, 4, 8), ("basic", "rank"))

    assert metrics == {
        "sample_count": 0,
        "flat_dim": 32,
        "finite_ratio": 1.0,
        "vector_count": 0,
        "vector_dim": 8,
        "metric_vector_count": 0,
        "l2_norm_mean": 0.0,
        "l2_norm_std": 0.0,
        "l2_norm_min": 0.0,
        "l2_norm_max": 0.0,
    }


def test_l2_norm_metrics_use_the_last_axis_as_vector_dimension():
    monitor = HealthMonitor()
    tensor = torch.tensor([
        [[3.0, 4.0], [0.0, 0.0]],
        [[0.0, 5.0], [8.0, 15.0]],
    ])

    metrics = monitor._tensor_metrics(tensor, ("basic", "rank"))

    assert metrics["vector_count"] == 4
    assert metrics["l2_norm_mean"] == 6.75
    assert metrics["l2_norm_std"] == pytest.approx(6.259992, abs=1e-6)
    assert metrics["l2_norm_min"] == 0.0
    assert metrics["l2_norm_max"] == 17.0


def test_vector_collection_metrics_cover_scale_coordinates_pairs_and_population_rank():
    monitor = HealthMonitor()
    vectors = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        [[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]],
    ])

    metrics = monitor._tensor_metrics(vectors, ("basic", "vector_collection"))

    assert metrics["vector_count"] == 4
    assert metrics["vector_dim"] == 4
    assert metrics["l2_norm_mean"] == 1.0
    assert metrics["coordinate_participation_ratio"] == 0.25
    assert metrics["pairwise_sample_count"] == 6
    assert metrics["pairwise_abs_cosine_mean"] == pytest.approx(1.0 / 3.0)
    assert metrics["pairwise_abs_cosine_p95"] == 1.0
    assert metrics["population_effective_rank"] == pytest.approx(2.0)
    assert metrics["population_effective_rank_ratio"] == pytest.approx(2.0 / 3.0)


def test_vector_metrics_sample_on_vector_rows_before_cpu_summary():
    monitor = HealthMonitor(HealthMonitorConfig(max_metric_vectors=2))
    vectors = torch.tensor([
        [3.0, 4.0],
        [6.0, 8.0],
        [9.0, 12.0],
        [12.0, 16.0],
    ])

    metrics = monitor._tensor_metrics(vectors, ("basic", "vector_collection"))

    assert metrics["vector_count"] == 4
    assert metrics["metric_vector_count"] == 2
    assert metrics["l2_norm_mean"] == 12.5


def test_loss_signal_names_are_stable_when_extra_scopes_exist():
    signals = HealthMonitor._loss_signals({
        "settlements": {
            "main": {
                "total": 1.0,
                "terms": {
                    "task": {"raw": 1.0, "weight": 1.0, "weighted": 1.0},
                },
            },
            "generator": {
                "total": 2.0,
                "terms": {
                    "adversarial": {"raw": 2.0, "weight": 1.0, "weighted": 2.0},
                },
            },
        },
    })

    assert "loss.task" in signals
    assert "loss.main.task" not in signals
    assert "loss.generator.adversarial" in signals
    assert signals["loss.task"].metadata["settlement_scope"] == "main"
    assert signals["loss.generator.adversarial"].metadata["settlement_scope"] == "generator"


def test_domain_routed_probe_keeps_only_matching_samples_and_cleans_context():
    monitor = HealthMonitor(HealthMonitorConfig(snapshot_every=1))
    monitor.observe_module(
        "tower",
        prefix="domain_tower",
        include_root=False,
        domain_routed=True,
    )
    model = DomainRoutedToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    domains = torch.tensor([0, 0, 1, 2, 2, 2])
    batch = DummyCls.make_interaction(batch_size=6, domain=domains)
    trainer = BaseTrainerWithLife(
        model,
        optimizer,
        DummyCls.MANAGER,
        TrainerDataLoaders(train=[batch], valid=[]),
        TrainerComponents(health_monitor=monitor),
        _cfg(),
    )

    trainer.train_epoch()

    snapshot = monitor.store.latest()
    assert snapshot is not None
    expected_counts = {0: 2, 1: 1, 2: 3}
    routed = {
        name: signal
        for name, signal in snapshot.signals.items()
        if ".domain_tower.layers.domain_networks." in name
    }
    assert routed
    for signal in routed.values():
        domain = int(signal.coordinates["domain"])
        assert signal.metrics["sample_count"] == expected_counts[domain]
    assert monitor.probes._domain_context == {}
