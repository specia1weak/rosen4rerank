from __future__ import annotations

import pytest
import torch

from rosen.core.train import LossPort


def test_loss_port_collects_named_weighted_terms_without_cross_step_state():
    port = LossPort()
    task = torch.tensor(2.0, requires_grad=True)
    auxiliary = torch.tensor(3.0, requires_grad=True)

    port.start_step(step=1, epoch=0, batch_idx=0, stage="train")
    with port.scope():
        assert port.scope_name == "main"
        port.add("task", task)
        port.add("aux", auxiliary, weight=0.1)
        loss = port.settle()

        assert torch.allclose(loss, torch.tensor(2.3))
        assert port.active
        assert not port.has_terms

    assert not port.active
    assert not port.has_terms
    summary = port.summary()
    assert summary["step"] == 1
    assert summary["stage"] == "train"
    assert summary["settlements"]["main"]["total"] == pytest.approx(2.3)
    assert summary["settlements"]["main"]["terms"] == {
        "task": {"raw": 2.0, "weight": 1.0, "weighted": 2.0},
        "aux": {"raw": 3.0, "weight": 0.1, "weighted": pytest.approx(0.3)},
    }
    loss.backward()
    assert task.grad is not None
    assert auxiliary.grad is not None

    port.start_step(step=2, epoch=0, batch_idx=1, stage="train")
    assert port.summary()["settlements"] == {}
    with port.scope():
        next_loss = port.settle(torch.tensor(4.0))
    assert next_loss.item() == 4.0
    assert port.summary(clear=True)["settlements"]["main"]["total"] == 4.0
    assert port.summary()["settlements"] == {}


def test_loss_port_rejects_duplicate_names_and_out_of_scope_add():
    port = LossPort()
    with pytest.raises(RuntimeError):
        port.add("task", torch.tensor(1.0))

    with port.scope():
        port.add("task", torch.tensor(1.0))
        with pytest.raises(ValueError):
            port.add("task", torch.tensor(2.0))
        port.settle()


def test_loss_port_rejects_scope_exit_with_unsettled_terms():
    port = LossPort()

    with pytest.raises(RuntimeError, match="must call settle"):
        with port.scope("main"):
            port.add("task", torch.tensor(1.0))

    assert not port.active
    assert not port.has_terms
    assert not port.has_settlements


def test_loss_port_rejects_empty_scope_names():
    port = LossPort()

    with pytest.raises(ValueError, match="cannot be empty"):
        with port.scope("  "):
            pass


def test_loss_port_requires_distinct_scopes_for_multiple_settlements():
    port = LossPort()
    port.start_step(step=1, epoch=0, batch_idx=0, stage="train")

    with port.scope("main"):
        port.settle(torch.tensor(1.0))
    with port.scope("main"):
        with pytest.raises(RuntimeError, match="distinct scope names"):
            port.settle(torch.tensor(2.0))
    with port.scope("architecture"):
        port.settle(torch.tensor(3.0))

    assert tuple(port.summary()["settlements"]) == ("main", "architecture")
