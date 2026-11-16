from __future__ import annotations

from types import SimpleNamespace

import torch

from rosen.core.train import EMAConfig, CustomTrainStepProtocol, TrainerComponents, TrainerDataLoaders
from rosen.core.trainers import BaseTrainerWithLife, EMATrainer
from rosen.models.msr.m3oe import M3oEModel
from tests.support.model_fixtures import DummyCls
from rosen.utils.observatory import HealthMonitor, HealthMonitorConfig


def _cfg():
    return SimpleNamespace(
        device="cpu",
        train_log_interval=100,
        max_epochs=1,
        ckpt_dir="",
        experiment_name="m3oe_async",
        dataset_name="dummy",
    )


def _model():
    return M3oEModel(
        DummyCls.MANAGER,
        DummyCls.NUM_DOMAINS,
        star_dims=(16, 8),
        expert_dims=(8,),
        num_shared_experts=2,
        tower_hidden_dims=(4,),
        factor_mode="global",
        arch_lr=1e-3,
    )


def test_m3oe_uses_custom_step_and_epoch_hook_for_asynchronous_architecture_update():
    model = _model()
    arch_before = torch.cat([value.detach().reshape(-1) for value in model._arch_parameters()])
    monitor = HealthMonitor(HealthMonitorConfig(snapshot_every=1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = BaseTrainerWithLife(
        model,
        optimizer,
        DummyCls.MANAGER,
        TrainerDataLoaders(train=[DummyCls.make_interaction(batch_size=8)], valid=[]),
        TrainerComponents(health_monitor=monitor),
        _cfg(),
    )

    assert isinstance(model, CustomTrainStepProtocol)
    assert not hasattr(trainer, "lifecycle")
    trainer.train_epoch()

    arch_after = torch.cat([value.detach().reshape(-1) for value in model._arch_parameters()])
    assert not torch.allclose(arch_before, arch_after)
    cycle_snapshots = [
        snapshot for snapshot in monitor.store.list()
        if snapshot.stage == "epoch_end"
    ]
    assert len(cycle_snapshots) == 1
    cycle_snapshot = cycle_snapshots[0]
    assert cycle_snapshot.metadata["model_updated"] is True
    assert cycle_snapshot.metadata["updated_state_names"] == list(model._arch_state_names())
    architecture_loss = next(
        signal for name, signal in cycle_snapshot.signals.items()
        if name == "loss.epoch_end_arch_update.task"
    )
    assert architecture_loss.role == "loss"
    assert architecture_loss.metadata["settlement_scope"] == "epoch_end_arch_update"
    assert not any(".m3oe." in name for name in cycle_snapshot.signals)
    trainer_state = model.trainer_state_dict()
    assert set(trainer_state) == {"arch_optimizer"}
    assert trainer_state["arch_optimizer"] is not None


def test_m3oe_epoch_end_architecture_update_is_visible_to_ema_teacher():
    model = _model()
    initial = torch.cat([value.detach().reshape(-1) for value in model._arch_parameters()])
    arch_ids = {id(value) for value in model._arch_parameters()}
    main_name, main_parameter = next(
        (name, value)
        for name, value in model.named_parameters()
        if id(value) not in arch_ids
    )
    initial_main = main_parameter.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = EMATrainer(
        model,
        optimizer,
        DummyCls.MANAGER,
        TrainerDataLoaders(train=[DummyCls.make_interaction(batch_size=8)], valid=[]),
        TrainerComponents(),
        _cfg(),
        ema_config=EMAConfig(
            decays=(0.5,),
            eval_student=False,
            eval_ema=False,
        ),
    )

    trainer.train_epoch()

    final = torch.cat([value.detach().reshape(-1) for value in model._arch_parameters()])
    teacher = trainer.ema.teachers[0.5]
    teacher_arch = torch.cat([
        value.detach().reshape(-1)
        for value in teacher.backbone.arch_parameters()
        if value.requires_grad is False
    ])
    assert not torch.allclose(initial, final)
    assert torch.allclose(teacher_arch, initial * 0.5 + final * 0.5, atol=1e-6)
    final_main = dict(model.named_parameters())[main_name].detach()
    teacher_main = dict(teacher.named_parameters())[main_name].detach()
    assert torch.allclose(
        teacher_main,
        initial_main * 0.5 + final_main * 0.5,
        atol=1e-6,
    )
