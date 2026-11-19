from types import SimpleNamespace

import torch

from rosen.core.train import EarlyStopper, TrainerComponents, TrainerDataLoaders
from rosen.core.trainers import BaseTrainer
from rosen.evaluate import EvaluatorManager, PointwiseEvaluator
from tests.rosen.core.trainers.test_base_trainer import BaseToyModel
from tests.support.model_fixtures import DummyCls


def test_best_checkpoint_tracks_the_actual_save_path_with_irregular_steps(tmp_path, monkeypatch):
    model = BaseToyModel()
    config = SimpleNamespace(device='cpu', ckpt_dir=str(tmp_path), max_epochs=3,
                             experiment_name='checkpoint', dataset_name='synthetic')
    trainer = BaseTrainer(model, torch.optim.SGD(model.parameters(), lr=.01), DummyCls.MANAGER,
                          TrainerDataLoaders([], None), TrainerComponents(early_stopper=EarlyStopper()), config)

    def train():
        trainer.global_step += [2, 1, 7][trainer.epoch]
        trainer.epoch += 1

    monkeypatch.setattr(trainer, 'train_epoch', train)
    monkeypatch.setattr(trainer, 'evaluate_epoch', lambda: {'overall': {'auc': [.5, .9, .7][trainer.epoch-1]}})
    trainer.run()
    assert trainer.global_step == 10 and trainer.early_stopper.best_epoch == 2
    assert trainer.best_checkpoint_path.name == 'epoch_0002_step_00000003_best.pt'
    assert trainer.last_checkpoint_path == trainer.best_checkpoint_path
    saved = trainer.load_checkpoint(trainer.best_checkpoint_path)
    assert saved['global_step'] == 3 and saved['epoch'] == 2


def test_validation_and_test_logs_have_explicit_stage_epoch_and_step(tmp_path):
    model = BaseToyModel()
    data = DummyCls.make_interaction(batch_size=4, **{DummyCls.USER_ID: torch.tensor([0, 5, 0, 5]),
                                                     DummyCls.LABEL: torch.tensor([0., 1., 0., 1.])})
    em = EvaluatorManager(log_path=tmp_path / 'evaluation.log')
    em.register('point', PointwiseEvaluator('auc'))
    trainer = BaseTrainer(model, torch.optim.SGD(model.parameters(), lr=.01), DummyCls.MANAGER,
                          TrainerDataLoaders([], [data], [data]), TrainerComponents(evaluator_manager=em),
                          SimpleNamespace(device='cpu'))
    trainer.epoch, trainer.global_step = 2, 17
    assert trainer.evaluate_epoch() == trainer.evaluate_test()
    log = (tmp_path / 'evaluation.log').read_text()
    assert log.count('Stage: valid') == log.count('Stage: test') == 1
    assert log.count('Epoch: 02') == log.count('Step: 17') == 2
