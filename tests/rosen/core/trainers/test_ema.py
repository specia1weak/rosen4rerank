"""Tests for EMA training integration."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from rosen.core.train import EMAConfig, TrainerComponents, TrainerDataLoaders
from rosen.core.trainers import EMATrainer
from rosen.models.base import BaseModel
from tests.support.model_fixtures import DummyCls


class TinyBinaryModel(BaseModel):
    def __init__(self):
        super().__init__(DummyCls.MANAGER)
        self.linear = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.constant_(self.linear.weight, 1.0)

    def _feature(self, interaction):
        return interaction[DummyCls.USER_ID].float().view(-1, 1) / 10.0

    def calculate_loss(self, interaction):
        logits = self.linear(self._feature(interaction)).view(-1)
        labels = interaction[DummyCls.LABEL].float().view(-1)
        return F.binary_cross_entropy_with_logits(logits, labels)

    def predict(self, interaction):
        logits = self.linear(self._feature(interaction)).view(-1)
        return torch.sigmoid(logits)


def _cfg(tmp_path=None):
    return SimpleNamespace(
        device="cpu",
        train_log_interval=100,
        max_epochs=1,
        ckpt_dir=str(tmp_path) if tmp_path is not None else "",
        experiment_name="ema_test",
        dataset_name="dummy",
    )


def test_ema_trainer_updates_teacher_without_model_protocol():
    interaction = DummyCls.make_interaction(
        batch_size=4,
        **{
            DummyCls.USER_ID: torch.tensor([1, 2, 3, 4]),
            DummyCls.LABEL: torch.tensor([0.0, 1.0, 0.0, 1.0]),
        },
    )
    model = TinyBinaryModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    initial_weight = model.linear.weight.detach().clone()

    trainer = EMATrainer(
        model,
        optimizer,
        DummyCls.MANAGER,
        TrainerDataLoaders(train=[interaction], valid=[]),
        TrainerComponents(evaluator_manager=None),
        _cfg(),
        ema_config=EMAConfig(decays=(0.5,), eval_student=False),
    )
    teacher = trainer.ema.teachers[0.5]
    assert not any(param.requires_grad for param in teacher.parameters())

    trainer.train_epoch()

    student_weight = trainer.model.linear.weight.detach()
    teacher_weight = teacher.linear.weight.detach()
    expected_teacher_weight = initial_weight * 0.5 + student_weight * 0.5

    assert trainer.global_step == 1
    assert not torch.allclose(student_weight, initial_weight)
    assert torch.allclose(teacher_weight, expected_teacher_weight)


def test_ema_trainer_exposes_only_ema_config():
    parameters = inspect.signature(EMATrainer.__init__).parameters
    assert "ema_config" in parameters
    assert not {
        "ema_decays",
        "ema_start_step",
        "ema_update_every",
        "ema_include_buffers",
        "ema_eval_student",
        "ema_eval_ema",
        "ema_primary_checkpoint",
        "ema_backend",
    }.intersection(parameters)


def test_ema_trainer_checkpoint_contains_ema_state(tmp_path):
    model = TinyBinaryModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = EMATrainer(
        model,
        optimizer,
        DummyCls.MANAGER,
        TrainerDataLoaders(train=[], valid=[]),
        TrainerComponents(evaluator_manager=None),
        _cfg(tmp_path),
        ema_config=EMAConfig(decays=(0.9,)),
    )

    path = trainer.save_checkpoint(tag="unit", metrics={"overall": {"auc": 0.5}})
    assert path is not None
    payload = torch.load(path, map_location="cpu")

    assert "ema_state_dict" in payload
    assert payload["ema_state_dict"]["decays"] == [0.9]
    assert "ema0p9" in payload["ema_state_dict"]["teachers"]


def test_ema_trainer_can_disable_ema_with_config(tmp_path):
    model = TinyBinaryModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = EMATrainer(
        model,
        optimizer,
        DummyCls.MANAGER,
        TrainerDataLoaders(train=[], valid=[]),
        TrainerComponents(evaluator_manager=None),
        _cfg(tmp_path),
        ema_config=EMAConfig(decays=None, eval_ema=False),
    )

    assert trainer.ema is None
    path = trainer.save_checkpoint(tag="noema")
    assert path is not None
    payload = torch.load(path, map_location="cpu")
    assert payload["ema_state_dict"] is None
