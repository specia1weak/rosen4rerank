"""All MSR branches share raw training scores and probability prediction."""
from types import SimpleNamespace

import pytest
import torch
from sklearn.metrics import log_loss

from rosen.core.train import TrainerComponents, TrainerDataLoaders
from rosen.core.trainers import BaseTrainer
from rosen.evaluate import PointwiseEvaluator, EvaluatorManager
from rosen.models.msr import MSRModel
from tests.support.msr_contract_cases import CASES, make_case


@pytest.mark.parametrize('name', CASES)
def test_msr_raw_logits_training_and_probability_prediction(name, monkeypatch):
    model, inter = make_case(name)
    model.eval()
    logits = model.logits(inter)
    probabilities = model.predict(inter)
    assert type(model).predict is MSRModel.predict
    assert logits.shape == probabilities.shape == (6,)
    assert logits.requires_grad and torch.isfinite(logits).all()
    torch.testing.assert_close(probabilities, logits.sigmoid(), rtol=0, atol=0)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    def prediction_is_not_a_training_interface(*args):
        raise AssertionError('calculate_loss must not consume predict probabilities')
    monkeypatch.setattr(model, 'predict', prediction_is_not_a_training_interface)
    loss = model.calculate_loss(inter)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize('name', ['sharedbottom', 'star', 'mmoe', 'crocodile', 'hamur'])
def test_native_msr_tensor_predict_works_with_trainer_and_logloss(name):
    model, inter = make_case(name)
    manager = EvaluatorManager()
    manager.register('overall', PointwiseEvaluator('auc', 'logloss'))
    trainer = BaseTrainer(model, torch.optim.SGD(model.parameters(), lr=0.), model.manager,
                          TrainerDataLoaders([], [inter]), TrainerComponents(evaluator_manager=manager),
                          SimpleNamespace(device='cpu'))
    result = trainer.evaluate_epoch()['overall']
    with torch.no_grad():
        expected = model.logits(inter).sigmoid().numpy()
    assert result['logloss'] == pytest.approx(log_loss(inter[model.LABEL].numpy(), expected), abs=1e-7)
    assert 0 <= result['auc'] <= 1
