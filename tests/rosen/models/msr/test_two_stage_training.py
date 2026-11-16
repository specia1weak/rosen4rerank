from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rosen.core.train import (
    EMAConfig,
    EarlyStopper,
    EvaluationFeedback,
    StageReport,
    StageSnapshot,
    TrainerComponents,
    TrainerDataLoaders,
    TwoStagePhase,
    TwoStageTrainingProtocol,
)
from rosen.core.trainers import (
    BaseTrainer,
    BaseTrainerWithLife,
    EMATrainer,
    TwoStageRunner,
)
from rosen.models.base import BaseModel
from rosen.models.msr.multiemb.model import MultiEmbModel
from rosen.models.msr.ssim.model import SSIMModel
from rosen.utils.observatory import HealthMonitor, HealthMonitorConfig
from tests.support.model_fixtures import DummyCls


def _cfg(tmp_path=None, *, max_epochs=2):
    return SimpleNamespace(
        device="cpu",
        train_log_interval=100,
        max_epochs=max_epochs,
        ckpt_dir=str(tmp_path) if tmp_path is not None else "",
        experiment_name="two_stage",
        dataset_name="dummy",
    )


def _multiemb():
    return MultiEmbModel(
        DummyCls.MANAGER,
        DummyCls.NUM_DOMAINS,
        branch_hidden_dims=(8,),
        mask_hidden_dims=(8,),
        search_epochs=1,
        search_steps_hint=2,
    )


def _ssim():
    return SSIMModel(
        DummyCls.MANAGER,
        DummyCls.NUM_DOMAINS,
        branch_hidden_dims=(8,),
        mask_hidden_dims=(8,),
        search_epochs=1,
        search_steps_hint=2,
    )


class _FeedbackTrainer(BaseTrainerWithLife):
    """Small real Trainer whose evaluator reports student and EMA targets."""

    @torch.no_grad()
    def evaluate_epoch(self):
        student_metrics = {"overall": {"auc": 0.7}}
        primary_metrics = {"overall": {"auc": 0.8}}
        self._notify_evaluation_end(
            primary_metrics,
            feedback=EvaluationFeedback(
                metrics_by_target={
                    "student": student_metrics,
                    "ema0p5": primary_metrics,
                },
                primary_target="ema0p5",
                primary_metrics=primary_metrics,
            ),
        )
        return primary_metrics


def test_two_stage_runner_uses_new_model_and_fresh_training_runtime(tmp_path):
    stage_one_model = _multiemb()
    trainers: list[BaseTrainer] = []

    def trainer_factory(model, stage):
        assert isinstance(stage, TwoStagePhase)
        run_cfg = _cfg(tmp_path / stage.value)
        trainer = _FeedbackTrainer(
            model,
            torch.optim.Adam(model.parameters(), lr=1e-3),
            DummyCls.MANAGER,
            TrainerDataLoaders(
                train=[DummyCls.make_interaction(batch_size=8)],
                valid=[],
            ),
            TrainerComponents(
                early_stopper=EarlyStopper(),
                health_monitor=HealthMonitor(HealthMonitorConfig(snapshot_every=1)),
            ),
            run_cfg,
        )
        trainers.append(trainer)
        return trainer

    assert isinstance(stage_one_model, TwoStageTrainingProtocol)
    result = TwoStageRunner(trainer_factory).run(stage_one_model)

    assert trainers == [result.stage_one_trainer, result.stage_two_trainer]
    assert result.stage_one_trainer is not result.stage_two_trainer
    assert result.stage_one_trainer.epoch == 1
    assert result.stage_two_trainer.epoch == 2
    assert result.stage_one_report.epoch == 1
    assert result.stage_one_report.metrics_for("student") == {"overall": {"auc": 0.7}}
    assert result.stage_one_report.primary_target == "ema0p5"

    stage_two_model = result.stage_two_model
    assert stage_two_model is result.stage_two_trainer.model
    assert stage_two_model is not stage_one_model
    assert stage_one_model.training_phase == "search"
    assert stage_two_model.training_phase == "retrain"
    assert all(parameter.requires_grad for parameter in stage_one_model.domain_hypernet.parameters())
    assert not any(parameter.requires_grad for parameter in stage_two_model.domain_hypernet.parameters())
    assert stage_one_model._search_best_metric == 0.7
    assert stage_two_model._search_best_metric == 0.7

    first_optimizer = result.stage_one_trainer.optimizer
    second_optimizer = result.stage_two_trainer.optimizer
    assert first_optimizer is not second_optimizer
    assert first_optimizer.state
    assert second_optimizer.state
    stage_two_parameter_ids = {id(parameter) for parameter in stage_two_model.parameters()}
    assert {
        id(parameter)
        for group in second_optimizer.param_groups
        for parameter in group["params"]
    }.issubset(stage_two_parameter_ids)
    assert result.stage_one_trainer.early_stopper is not result.stage_two_trainer.early_stopper
    assert result.stage_one_trainer.health_monitor is not result.stage_two_trainer.health_monitor

    stage_one_checkpoints = list((tmp_path / "stage_one").glob("*_best.pt"))
    assert stage_one_checkpoints
    stage_one_checkpoint = torch.load(stage_one_checkpoints[0], map_location="cpu")
    assert stage_one_checkpoint["model_trainer_state_dict"]["search_best_metric"] == 0.7
    assert stage_one_checkpoint["model_trainer_state_dict"]["search_best_state"] is not None

    stage_two_checkpoints = list((tmp_path / "stage_two").glob("*_best.pt"))
    assert stage_two_checkpoints
    stage_two_checkpoint = torch.load(stage_two_checkpoints[0], map_location="cpu")
    assert stage_two_checkpoint["model_trainer_state_dict"]["search_best_metric"] == 0.7
    assert stage_two_checkpoint["model_trainer_state_dict"]["search_best_state"] is None

    stage_one_train_snapshot = result.stage_one_trainer.health_monitor.store.list(stage="train")[0]
    assert "loss.task.own" in stage_one_train_snapshot.signals
    assert "loss.regularization.shared" in stage_one_train_snapshot.signals

    current_state = stage_one_model.state_dict()
    for name, value in result.stage_snapshot.model_state.items():
        assert isinstance(value, torch.Tensor)
        assert value.device.type == "cpu"
        assert value.data_ptr() != current_state[name].data_ptr()


def test_stage_two_ema_is_initialized_from_new_model_without_phase_reseed():
    stage_one_model = _ssim()
    trainers: list[EMATrainer] = []

    def trainer_factory(model, stage):
        assert isinstance(stage, TwoStagePhase)
        trainer = EMATrainer(
            model,
            torch.optim.Adam(model.parameters(), lr=1e-3),
            DummyCls.MANAGER,
            TrainerDataLoaders(train=[], valid=[]),
            TrainerComponents(early_stopper=EarlyStopper()),
            _cfg(max_epochs=1),
            ema_config=EMAConfig(
                decays=(0.5,),
                eval_student=False,
                eval_ema=False,
            ),
        )
        trainers.append(trainer)
        return trainer

    result = TwoStageRunner(trainer_factory).run(stage_one_model)

    stage_one_trainer, stage_two_trainer = trainers
    assert stage_one_trainer.ema is not stage_two_trainer.ema
    stage_one_teacher = stage_one_trainer.ema.teachers[0.5]
    stage_two_teacher = stage_two_trainer.ema.teachers[0.5]
    assert stage_one_teacher is not stage_two_teacher
    for name, stage_two_value in result.stage_two_model.state_dict().items():
        assert torch.equal(stage_two_teacher.state_dict()[name], stage_two_value), name
    assert not any(
        parameter.requires_grad
        for parameter in result.stage_two_model.domain_hypernet.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in result.stage_two_model.mask_heads.parameters()
    )


def test_stage_snapshot_is_a_detached_cpu_clone():
    model = _multiemb()
    report = StageReport(
        epoch=3,
        global_step=12,
        metrics_by_target={"student": {"overall": {"auc": 0.75}}},
        primary_target="student",
        primary_metrics={"overall": {"auc": 0.75}},
    )

    snapshot = StageSnapshot.capture(model, report)
    name, parameter = next(iter(model.named_parameters()))
    captured = snapshot.model_state[name].clone()
    with torch.no_grad():
        parameter.add_(1.0)

    assert snapshot.report is report
    assert snapshot.model_state[name].device.type == "cpu"
    assert torch.equal(snapshot.model_state[name], captured)
    assert not torch.equal(snapshot.model_state[name], parameter.detach().cpu())


class _InvalidTwoStageModel(BaseModel):
    def __init__(self):
        super().__init__(DummyCls.MANAGER)
        self.weight = torch.nn.Parameter(torch.ones(()))

    def calculate_loss(self, interaction):
        del interaction
        return self.weight.square()

    def predict(self, interaction):
        return torch.zeros(len(interaction), device=self.weight.device)

    def should_finish_stage_one(self, report):
        del report
        return True

    def initialize_stage_two(self, snapshot):
        del snapshot
        return self


def test_two_stage_runner_rejects_in_place_stage_transition():
    model = _InvalidTwoStageModel()

    def trainer_factory(current_model, stage):
        assert isinstance(stage, TwoStagePhase)
        return BaseTrainer(
            current_model,
            torch.optim.SGD(current_model.parameters(), lr=0.1),
            DummyCls.MANAGER,
            TrainerDataLoaders(train=[], valid=[]),
            TrainerComponents(),
            _cfg(max_epochs=1),
        )

    with pytest.raises(ValueError, match="must return a new model object"):
        TwoStageRunner(trainer_factory).run(model)


def test_two_stage_runner_rejects_reused_trainer():
    model = _multiemb()
    shared_trainer = BaseTrainer(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        DummyCls.MANAGER,
        TrainerDataLoaders(train=[], valid=[]),
        TrainerComponents(),
        _cfg(max_epochs=1),
    )

    def trainer_factory(current_model, stage):
        assert isinstance(stage, TwoStagePhase)
        shared_trainer.model = current_model
        return shared_trainer

    with pytest.raises(ValueError, match="fresh Trainer"):
        TwoStageRunner(trainer_factory).run(model)


def test_two_stage_runner_rejects_checkpoint_filename_collisions(tmp_path):
    model = _multiemb()

    def trainer_factory(current_model, stage):
        assert isinstance(stage, TwoStagePhase)
        return BaseTrainer(
            current_model,
            torch.optim.Adam(current_model.parameters(), lr=1e-3),
            DummyCls.MANAGER,
            TrainerDataLoaders(train=[], valid=[]),
            TrainerComponents(),
            _cfg(tmp_path, max_epochs=1),
        )

    with pytest.raises(ValueError, match="different checkpoint directories"):
        TwoStageRunner(trainer_factory).run(model)


def test_two_stage_runner_rejects_reused_one_shot_loader():
    model = _multiemb()
    one_shot_train_loader = iter(())

    def trainer_factory(current_model, stage):
        assert isinstance(stage, TwoStagePhase)
        return BaseTrainer(
            current_model,
            torch.optim.Adam(current_model.parameters(), lr=1e-3),
            DummyCls.MANAGER,
            TrainerDataLoaders(train=one_shot_train_loader, valid=[]),
            TrainerComponents(),
            _cfg(max_epochs=1),
        )

    with pytest.raises(ValueError, match="rebuild one-shot train_loader"):
        TwoStageRunner(trainer_factory).run(model)
