"""Tests for the post-training observation audit workflow."""

from __future__ import annotations

import torch

from rosen.models.base import BaseModel
from tests.support.model_fixtures import DummyCls
from rosen.utils.observatory.audit.observation import ObservationAuditWorkflow


class ObservationAuditToyModel(BaseModel):
    def __init__(self):
        super().__init__(DummyCls.MANAGER)
        self.weight = torch.nn.Parameter(torch.ones(()))

    def _logits(self, interaction):
        x = interaction[DummyCls.USER_ID].float().view(-1, 1) * self.weight
        x = self.obp.cut("score_feature", x)
        x = self.obp.tap_gradient("score_feature", x)
        logits = x.view(-1) - 2.5
        self.obp.observe("logits", logits, watch=True, gradient=True)
        return logits

    def predict(self, interaction):
        return self._logits(interaction)

    def calculate_loss(self, interaction):
        labels = interaction[DummyCls.LABEL].float().view(-1)
        return torch.nn.functional.binary_cross_entropy_with_logits(self._logits(interaction), labels)


def test_observation_audit_workflow_writes_sensitivity_and_gradient_reports(tmp_path):
    train_batch = DummyCls.make_interaction(
        batch_size=8,
        **{
            DummyCls.USER_ID: torch.tensor([0, 4, 1, 5, 0, 6, 1, 7]),
            DummyCls.DOMAIN: torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]),
            DummyCls.LABEL: torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32),
        },
    )
    test_batch = DummyCls.make_interaction(
        batch_size=4,
        **{
            DummyCls.USER_ID: torch.tensor([0, 4, 1, 5]),
            DummyCls.DOMAIN: torch.tensor([0, 0, 1, 1]),
            DummyCls.LABEL: torch.tensor([0, 1, 0, 1], dtype=torch.float32),
        },
    )
    model = ObservationAuditToyModel()
    workflow = ObservationAuditWorkflow(
        model,
        DummyCls.MANAGER,
        result_loader=[test_batch],
        gradient_loader=[train_batch],
        device="cpu",
        seed=2026,
        output_dir=tmp_path,
        run_name="toy",
        domains=[0, 1],
        fields=[],
        audit_batches=1,
        gradient_modes=["natural"],
    )

    outputs = workflow.run()

    assert any(row["field"] == "score_feature" for row in outputs["sensitivity"])
    assert any(row["tap"] == "score_feature" for row in outputs["gradient_batches"])
    assert (tmp_path / "toy.metrics.csv").exists()
    assert (tmp_path / "toy.sensitivity.csv").exists()
    assert (tmp_path / "toy.gradient_batches.csv").exists()
    assert (tmp_path / "toy.gradient_summary.csv").exists()
    assert (tmp_path / "toy.md").exists()
