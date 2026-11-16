from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rosen.core.train import TrainerComponents, TrainerDataLoaders
from rosen.core.trainers import BaseTrainer, BaseTrainerWithLife
from rosen.models.msr import build_model
from rosen.utils.observatory import HealthMonitor, HealthMonitorConfig
from tests.support.model_fixtures import DummyCls


def _cfg():
    return SimpleNamespace(
        device="cpu",
        train_log_interval=100,
        max_epochs=1,
        ckpt_dir="",
        experiment_name="sharedbottom_ports",
        dataset_name="dummy",
    )


def _sharedbottom():
    return DummyCls.build_model(
        hidden_dims=(16, 8),
        tower_hidden_dims=(4,),
        batch_norm=False,
    )


def test_sharedbottom_keeps_observation_port_empty_without_external_or_custom_observers():
    model = _sharedbottom()
    interaction = DummyCls.make_interaction(batch_size=6)

    assert hasattr(model, "obp")
    assert hasattr(model, "lsp")
    assert not hasattr(model, "losses")

    with model.obp.collecting(enabled=True):
        loss = model.calculate_loss(interaction)

    trace = model.obp.summary()
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert trace == {}
    # calculate_loss only contributes terms when a training transaction owns
    # an active LossPort scope; it never settles a step by itself.
    assert model.lsp.summary()["settlements"] == {}


def test_sharedbottom_default_train_step_settles_named_task_loss():
    model = _sharedbottom()
    trainer = BaseTrainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        DummyCls.MANAGER,
        TrainerDataLoaders(
            train=[DummyCls.make_interaction(batch_size=6)],
            valid=None,
        ),
        TrainerComponents(),
        _cfg(),
    )

    trainer.train_epoch()

    summary = model.lsp.summary()
    settlement = summary["settlements"]["main"]
    assert summary["step"] == 1
    assert summary["stage"] == "train"
    assert settlement["total"] == pytest.approx(
        settlement["terms"]["task"]["weighted"]
    )
    assert settlement["terms"]["task"]["weight"] == 1.0


def test_sharedbottom_health_snapshot_aligns_ports_and_domain_tower_layers():
    monitor = HealthMonitor(
        HealthMonitorConfig(snapshot_every=1, capture_first_step=False)
    )
    monitor.observe_module(
        "backbone",
        prefix="sharedbottom_backbone",
        include_root=True,
    )
    monitor.observe_module(
        "head",
        prefix="domain_tower",
        include_root=True,
        domain_routed=True,
    )
    model = _sharedbottom()
    domains = torch.tensor([0, 0, 1, 1, 2, 2])
    trainer = BaseTrainerWithLife(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        DummyCls.MANAGER,
        TrainerDataLoaders(
            train=[DummyCls.make_interaction(batch_size=6, domain=domains)],
            valid=None,
        ),
        TrainerComponents(health_monitor=monitor),
        _cfg(),
    )

    trainer.train_epoch()

    snapshot = monitor.store.latest()
    assert snapshot is not None
    assert snapshot.step == 1
    assert snapshot.interval_start == snapshot.interval_end == 1
    assert "loss.task" in snapshot.signals
    assert "SharedBottomModel.sharedbottom_backbone.output" in snapshot.signals
    assert "SharedBottomModel.domain_tower.output" in snapshot.signals

    tower_layers = {
        name: signal
        for name, signal in snapshot.signals.items()
        if ".domain_tower.layers.head.domain_networks." in name
    }
    assert tower_layers
    assert {int(signal.coordinates["domain"]) for signal in tower_layers.values()} == {0, 1, 2}
    assert all(signal.metrics["sample_count"] == 2 for signal in tower_layers.values())


def test_sharedbottom_codebook_reports_task_and_quantization_terms():
    model = build_model(
        DummyCls.MANAGER,
        DummyCls.NUM_DOMAINS,
        "sharedbottom_codebook",
        hidden_dims=(16, 8),
        tower_hidden_dims=(4,),
        batch_norm=False,
        codebook_kmeans_init=False,
    )
    interaction = DummyCls.make_interaction(batch_size=6)
    model.lsp.start_step(step=1, epoch=0, batch_idx=0, stage="train")

    with model.obp.collecting(enabled=True), model.lsp.scope():
        raw_loss = model.calculate_loss(interaction)
        settled = model.lsp.settle(raw_loss)

    assert settled is raw_loss
    terms = model.lsp.summary()["settlements"]["main"]["terms"]
    assert set(terms) == {"task", "vq.quantize"}
    assert terms["task"]["weight"] == 1.0
    assert terms["vq.quantize"]["weight"] == pytest.approx(model.quantize_weight)
    assert model.obp.summary() == {}
