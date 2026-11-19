"""Sequence CTR families use the same logits/probability boundary as MSR."""
from types import SimpleNamespace

import pytest
import torch
from sklearn.metrics import log_loss

from rosen.core.train import TrainerComponents, TrainerDataLoaders
from rosen.core.trainers import BaseTrainer
from rosen.evaluate import PointwiseEvaluator, EvaluatorManager
from rosen.models.msr.base import MSRModel
from tests.support.unified_contract_cases import CASES, make_case


@pytest.mark.parametrize('name', CASES)
@pytest.mark.parametrize('training', [False, True])
def test_unified_training_uses_logits_and_prediction_returns_probabilities(name, training, monkeypatch):
    model, inter = make_case(name)
    model.train(training)
    torch.manual_seed(730)
    logits = model.logits(inter)
    torch.manual_seed(730)
    scores = model.predict(inter)
    assert type(model).predict is MSRModel.predict
    assert scores.shape == logits.shape == (6,)
    torch.testing.assert_close(scores, logits.sigmoid(), rtol=0, atol=0)

    def prediction_is_not_training(*args):
        raise AssertionError('calculate_loss must not consume probability predictions')

    monkeypatch.setattr(model, 'predict', prediction_is_not_training)
    loss = model.calculate_loss(inter)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize('name', [
    'HeadedCTRModel', 'HyFormerModel', 'RankMixerModel', 'TokenMixerModel',
    'MixFormerModel', 'CrossEventReaderMixFormerModel', 'DomainViewLogitMixFormerModel',
    'SASRecRankerModel', 'DVCSeqStage2SASRecRankerModel', 'UniRankOneTransModel',
])
def test_unified_tensor_predictions_use_default_trainer_evaluation(name):
    model, inter = make_case(name)
    evaluators = EvaluatorManager()
    evaluators.register('overall', PointwiseEvaluator('auc', 'logloss'))
    trainer = BaseTrainer(model, torch.optim.SGD(model.parameters(), lr=0.), model.manager,
                          TrainerDataLoaders([], [inter]), TrainerComponents(evaluator_manager=evaluators),
                          SimpleNamespace(device='cpu'))
    result = trainer.evaluate_epoch()['overall']
    with torch.no_grad():
        expected = model.logits(inter).sigmoid().numpy()
    assert result['logloss'] == pytest.approx(log_loss(inter[model.LABEL].numpy(), expected), abs=1e-7)
    assert 0 <= result['auc'] <= 1
