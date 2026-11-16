"""逐epoch观测不改变训练轨迹，不恢复best，不使用test决定停止。"""
from types import SimpleNamespace

import pytest
import torch
from rosen.core.interaction import Interaction
from rosen.core.trainers import BaseTrainer
from rosen.core.train.context import TrainerComponents, TrainerDataLoaders
from workspace.kr_time_budget.tests.test_v3 import model_batch
from workspace.kr_time_budget.pipeline.exit_model import KRExitModel
from workspace.kr_time_budget.pipeline.exit_training import (
    EXPERIMENTS, joint_parameter_groups, state_hash,
)
from workspace.kr_time_budget.pipeline.run_epochwise_exit import EpochwiseExitTrainer, optimizer_hash


@pytest.mark.parametrize('name', ['mixture_step_pace', 'fusion'])
def test_observing_changed_test_labels_preserves_five_epoch_training(model_batch, tmp_path, name):
    _, raw, manager = model_batch
    n = len(raw['y_continue'])
    values = dict(raw, elapsed_min_raw=raw['elapsed_min'].clone(),
        req_idx_raw=(raw['req_idx']+1).long(), request_id=torch.arange(n),
        session_id=torch.arange(n), user_id_raw=torch.ones(n, dtype=torch.long),
        tab_raw=torch.zeros(n, dtype=torch.long))
    batch = Interaction(values)
    changed_test = Interaction(dict(values, y_continue=1-values['y_continue']))
    cfg = SimpleNamespace(device='cpu', max_epochs=5, ckpt_dir='',
        experiment_name='test', dataset_name='synthetic', train_log_interval=1000)
    trainers = []
    for index in range(3):
        torch.manual_seed(2026)
        model = KRExitModel(manager, **EXPERIMENTS[name])
        args = (model, torch.optim.AdamW(joint_parameter_groups(model), weight_decay=0), manager,
                TrainerDataLoaders([batch], [batch], [batch if index != 2 else changed_test]),
                TrainerComponents(), cfg)
        if index == 0:
            trainer = BaseTrainer(*args)
        else:
            output = tmp_path/f'observed{index}'
            output.mkdir()
            trainer = EpochwiseExitTrainer(*args, output=output)
        trainer.run()
        trainers.append(trainer)
    reference = trainers[0]
    for trainer in trainers[1:]:
        assert trainer.global_step == reference.global_step == 5
        assert trainer.epoch == 5
        assert state_hash(trainer.model) == state_hash(reference.model)
        assert optimizer_hash(trainer.optimizer) == optimizer_hash(reference.optimizer)
        assert [r['epoch'] for r in trainer.records] == [1, 2, 3, 4, 5]
        for row in trainer.records:
            ckpt = torch.load(trainer.output/f'epoch_{row["epoch"]:02d}/checkpoint.pt', weights_only=False)
            assert ckpt['epoch'] == ckpt['global_step'] == row['epoch']
            assert ckpt['state_sha256'] == row['state_sha256']
            assert {int(s['step']) for s in ckpt['optimizer_state_dict']['state'].values()} == {row['epoch']}
    a, b = trainers[1:]
    assert [r['valid'] for r in a.records] == [r['valid'] for r in b.records]
    assert any(x['test']['logloss'] != y['test']['logloss'] for x, y in zip(a.records, b.records))


def test_metric_audit_allows_roundoff_but_rejects_score_and_coverage_errors():
    from workspace.kr_time_budget.pipeline.report_epochwise_exit import metric_difference
    reference = {'auc': .7, 'group': {'auc': .6, 'eligible_rows': 100}, 'undefined': None}
    rounded = {'auc': .7, 'group': {'auc': .6+1e-16, 'eligible_rows': 100}, 'undefined': None}
    assert 0 < metric_difference(rounded, reference) < 1e-12
    for changed in [dict(reference, auc=.70001),
                    dict(reference, group={'auc': .6, 'eligible_rows': 99}),
                    dict(reference, auc=float('nan'))]:
        with pytest.raises(AssertionError):
            metric_difference(changed, reference)
