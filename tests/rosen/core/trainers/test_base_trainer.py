"""Tests for the ordinary trainer flow."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from rosen.core.train import EarlyStopper, TrainerComponents, TrainerDataLoaders
from rosen.core.trainers import BaseTrainer, BaseTrainerWithLife
from rosen.evaluate.evaluator import PointwiseEvaluator
from rosen.evaluate.manager import EvaluatorManager
from rosen.models.base import BaseModel
from tests.support.model_fixtures import DummyCls


class BaseToyModel(BaseModel):
    def __init__(self):
        super().__init__(DummyCls.MANAGER)
        self.weight = torch.nn.Parameter(torch.ones(()))

    def calculate_loss(self, interaction):
        preds = self.predict(interaction)
        labels = interaction[DummyCls.LABEL].float().view(-1)
        return torch.nn.functional.binary_cross_entropy(preds.clamp(1e-7, 1.0 - 1e-7), labels)

    def predict(self, interaction):
        x = interaction[DummyCls.USER_ID].float().view(-1, 1)
        x = self.obp.cut("score_feature", x)
        self.obp.watch("score_input", x)
        logits = (x.view(-1) - 2.5) * self.weight
        return torch.sigmoid(logits)


class CycleToyModel(BaseToyModel):
    def __init__(self):
        super().__init__()
        self.cycle_calls = []

    def on_train_epoch_start(self, ctx):
        self.cycle_calls.append(("start", ctx.epoch))

    def on_train_epoch_end(self, ctx):
        self.cycle_calls.append(("end", ctx.epoch))

    def on_eval_epoch_end(self, metrics, ctx):
        self.cycle_calls.append(("eval", ctx.epoch, metrics))


class CustomToyModel(BaseToyModel):
    def custom_train_step(self, batch, ctx):
        raw_loss = self.calculate_loss(batch)
        self.lsp.add("task", raw_loss)
        loss = self.lsp.settle(raw_loss)
        loss.backward()
        ctx.optimizer.step()
        return loss


class MultiSettlementToyModel(BaseToyModel):
    def custom_train_step(self, batch, ctx):
        main_raw = self.calculate_loss(batch)
        self.lsp.add("task", main_raw)
        main_loss = self.lsp.settle(main_raw)
        main_loss.backward()
        ctx.optimizer.step()
        ctx.optimizer.zero_grad(set_to_none=True)

        with self.lsp.scope("adversarial"):
            adversarial_raw = self.calculate_loss(batch)
            self.lsp.add("task", adversarial_raw)
            adversarial_loss = self.lsp.settle(adversarial_raw)
        adversarial_loss.backward()
        ctx.optimizer.step()
        return adversarial_loss


class UnsettledCustomToyModel(BaseToyModel):
    def custom_train_step(self, batch, ctx):
        del ctx
        return self.calculate_loss(batch)


class RecordingTimer:
    def __init__(self):
        self.calls = []

    @contextmanager
    def __call__(self, name):
        self.calls.append(name)
        yield


def test_base_trainer_does_not_run_obp_cut_audit():
    interaction = DummyCls.make_interaction(
        batch_size=4,
        **{
            DummyCls.USER_ID: torch.tensor([0, 5, 0, 5]),
            DummyCls.LABEL: torch.tensor([0.0, 1.0, 0.0, 1.0]),
        },
    )
    model = BaseToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    evaluator = EvaluatorManager()
    evaluator.register("overall", PointwiseEvaluator("auc", "logloss"))
    cfg = SimpleNamespace(device="cpu", train_log_interval=100, max_epochs=1, ckpt_dir="")
    trainer = BaseTrainer(
        model,
        optimizer,
        DummyCls.MANAGER,
        TrainerDataLoaders(train=[], valid=[interaction]),
        TrainerComponents(evaluator_manager=evaluator),
        cfg,
    )

    metrics = trainer.evaluate_epoch()

    assert metrics["overall"]["auc"] == 1.0
    assert not hasattr(trainer, "last_cut_audit_rows")


def test_base_trainer_honors_optional_max_train_steps_without_experiment_subclass():
    model = BaseToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = SimpleNamespace(
        device="cpu",
        train_log_interval=100,
        max_epochs=1,
        max_train_steps=2,
        ckpt_dir="",
    )
    batches = [DummyCls.make_interaction(batch_size=4) for _ in range(5)]
    trainer = BaseTrainer(
        model,
        optimizer,
        DummyCls.MANAGER,
        TrainerDataLoaders(train=batches, valid=[]),
        TrainerComponents(),
        cfg,
    )

    trainer.train_epoch()

    assert trainer.global_step == 2


def test_base_trainer_uses_test_loader_and_named_timer_component():
    interaction = DummyCls.make_interaction(
        batch_size=4,
        **{
            DummyCls.USER_ID: torch.tensor([0, 5, 0, 5]),
            DummyCls.LABEL: torch.tensor([0.0, 1.0, 0.0, 1.0]),
        },
    )
    model = BaseToyModel()
    evaluator = EvaluatorManager()
    evaluator.register("overall", PointwiseEvaluator("auc"))
    timer = RecordingTimer()
    trainer = BaseTrainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        DummyCls.MANAGER,
        TrainerDataLoaders(
            train=[interaction],
            valid=[interaction],
            test=[interaction],
        ),
        TrainerComponents(evaluator_manager=evaluator, timer=timer),
        SimpleNamespace(
            device="cpu",
            train_log_interval=100,
            max_epochs=1,
            ckpt_dir="",
        ),
    )

    valid_metrics = trainer.run()
    test_metrics = trainer.evaluate_test()

    assert valid_metrics["overall"]["auc"] == 1.0
    assert test_metrics["overall"]["auc"] == 1.0
    assert timer.calls == ["train_epoch", "valid_epoch", "test_epoch"]


def test_base_trainer_run_does_not_restart_after_global_step_limit():
    model = BaseToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = SimpleNamespace(
        device="cpu",
        train_log_interval=100,
        max_epochs=3,
        max_train_steps=2,
        ckpt_dir="",
    )
    batches = [DummyCls.make_interaction(batch_size=4) for _ in range(5)]
    trainer = BaseTrainer(
        model,
        optimizer,
        DummyCls.MANAGER,
        TrainerDataLoaders(train=batches, valid=None),
        TrainerComponents(),
        cfg,
    )

    trainer.run()

    assert trainer.global_step == 2
    assert trainer.epoch == 1


def test_trainer_components_rejects_implicit_early_stopper_duck_types():
    with pytest.raises(TypeError, match="must be an EarlyStopper or None"):
        TrainerComponents(early_stopper=object())

    components = TrainerComponents(early_stopper=EarlyStopper())
    assert isinstance(components.early_stopper, EarlyStopper)


def test_base_trainer_does_not_interpret_training_cycle_protocol():
    model = CycleToyModel()
    trainer = BaseTrainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        DummyCls.MANAGER,
        TrainerDataLoaders(
            train=[DummyCls.make_interaction(batch_size=4)],
            valid=[],
        ),
        TrainerComponents(),
        SimpleNamespace(device="cpu", train_log_interval=100, max_epochs=1, ckpt_dir=""),
    )

    trainer.train_epoch()

    assert model.cycle_calls == []


def test_base_trainer_with_life_interprets_training_cycle_protocol():
    model = CycleToyModel()
    trainer = BaseTrainerWithLife(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        DummyCls.MANAGER,
        TrainerDataLoaders(
            train=[DummyCls.make_interaction(batch_size=4)],
            valid=[],
        ),
        TrainerComponents(),
        SimpleNamespace(device="cpu", train_log_interval=100, max_epochs=1, ckpt_dir=""),
    )

    trainer.train_epoch()

    assert model.cycle_calls == [("start", 0), ("end", 0)]


def test_base_trainer_rejects_lifecycle_health_component():
    model = BaseToyModel()
    with pytest.raises(ValueError, match="use BaseTrainerWithLife"):
        BaseTrainer(
            model,
            torch.optim.SGD(model.parameters(), lr=0.1),
            DummyCls.MANAGER,
            TrainerDataLoaders(train=[], valid=[]),
            TrainerComponents(health_monitor=object()),
            SimpleNamespace(device="cpu", train_log_interval=100, max_epochs=1, ckpt_dir=""),
        )


def test_base_trainer_accepts_model_owned_custom_loss_settlement():
    model = CustomToyModel()
    trainer = BaseTrainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        DummyCls.MANAGER,
        TrainerDataLoaders(
            train=[DummyCls.make_interaction(batch_size=4)],
            valid=None,
        ),
        TrainerComponents(),
        SimpleNamespace(device="cpu", train_log_interval=100, max_epochs=1, ckpt_dir=""),
    )

    trainer.train_epoch()

    assert model.lsp.summary()["settlements"]["main"]["terms"]["task"]["raw"] > 0.0


def test_base_trainer_rejects_custom_step_that_does_not_settle_loss_port():
    model = UnsettledCustomToyModel()
    trainer = BaseTrainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        DummyCls.MANAGER,
        TrainerDataLoaders(
            train=[DummyCls.make_interaction(batch_size=4)],
            valid=None,
        ),
        TrainerComponents(),
        SimpleNamespace(device="cpu", train_log_interval=100, max_epochs=1, ckpt_dir=""),
    )

    with pytest.raises(RuntimeError, match="must call model.lsp.settle"):
        trainer.train_epoch()


def test_base_trainer_keeps_main_implicit_and_allows_explicit_extra_scopes():
    model = MultiSettlementToyModel()
    trainer = BaseTrainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        DummyCls.MANAGER,
        TrainerDataLoaders(
            train=[DummyCls.make_interaction(batch_size=4)],
            valid=None,
        ),
        TrainerComponents(),
        SimpleNamespace(device="cpu", train_log_interval=100, max_epochs=1, ckpt_dir=""),
    )

    trainer.train_epoch()

    settlements = model.lsp.summary()["settlements"]
    assert tuple(settlements) == ("main", "adversarial")
    assert settlements["main"]["terms"]["task"]["raw"] > 0.0
    assert settlements["adversarial"]["terms"]["task"]["raw"] > 0.0
