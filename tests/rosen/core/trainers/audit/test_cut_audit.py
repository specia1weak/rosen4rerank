"""Tests for cut-audit trainer integration."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from rosen.core.train import TrainerComponents, TrainerDataLoaders
from rosen.core.trainers.audit import CutAuditTrainer
from rosen.evaluate.evaluator import PointwiseEvaluator
from rosen.evaluate.manager import EvaluatorManager
from rosen.models.base import BaseModel
from tests.support.model_fixtures import DummyCls


class CutAuditToyModel(BaseModel):
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


def test_cut_audit_trainer_evaluate_epoch_runs_one_at_a_time_cut_audit():
    interaction = DummyCls.make_interaction(
        batch_size=4,
        **{
            DummyCls.USER_ID: torch.tensor([0, 5, 0, 5]),
            DummyCls.LABEL: torch.tensor([0.0, 1.0, 0.0, 1.0]),
        },
    )
    model = CutAuditToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    evaluator = EvaluatorManager()
    evaluator.register("overall", PointwiseEvaluator("auc", "logloss"))
    cfg = SimpleNamespace(device="cpu", train_log_interval=100, max_epochs=1, ckpt_dir="")
    trainer = CutAuditTrainer(
        model,
        optimizer,
        DummyCls.MANAGER,
        TrainerDataLoaders(train=[], valid=[interaction]),
        TrainerComponents(evaluator_manager=evaluator),
        cfg,
    )

    metrics = trainer.evaluate_epoch()

    assert metrics["overall"]["auc"] == 1.0
    assert len(trainer.last_cut_audit_rows) == 1
    row = trainer.last_cut_audit_rows[0]
    assert row["cut"] == "score_feature"
    assert row["metric_changes"]["overall.auc_drop"] == 0.5
    assert row["metric_changes"]["overall.logloss_delta"] > 0.0
    assert "score_input" in row["watch_stats"]
    assert row["watch_stats"]["score_input"]["l2_mean"] == 0.0
    assert row["watch_changes"]["score_input"]["l2_mean_delta"] < 0.0
