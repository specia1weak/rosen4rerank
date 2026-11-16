"""UID置0同时覆盖两套Omni，保留raw UID分组和所有其他输入。"""
import pytest
import torch
from rosen.core.interaction import Interaction
from workspace.kr_time_budget.tests.test_v3 import model_batch
from workspace.kr_time_budget.pipeline.exit_model import KRExitModel
from workspace.kr_time_budget.pipeline.exit_training import EXPERIMENTS
from workspace.kr_time_budget.pipeline.run_uid_zero import zero_uid_inputs, uid_tables


@pytest.mark.parametrize('name', list(EXPERIMENTS))
def test_uid_mask_covers_both_towers_and_fusion_without_erasing_metric_identity(model_batch, name):
    _, raw, manager = model_batch
    n = len(raw['y_continue'])
    batch = Interaction(dict(raw, user_id_raw=torch.arange(n), request_id=torch.arange(n),
        req_idx_raw=(raw['req_idx']+1).long(), elapsed_min_raw=raw['elapsed_min'].clone()))
    original_uid = batch['user_id'].clone()
    masked = zero_uid_inputs({'test': [batch]})['test'][0]
    assert torch.equal(batch['user_id'], original_uid) and torch.count_nonzero(original_uid) > 0
    assert torch.count_nonzero(masked['user_id']) == 0
    assert masked['user_id_raw'].unique().numel() == n
    assert all(torch.equal(masked[k], batch[k]) for k in batch if k != 'user_id')
    model = KRExitModel(manager, **EXPERIMENTS[name])
    # 预算、内容与Fusion视图的UID槽位全部是严格零向量，而非共用一个可学习用户。
    for view in [model.budget_view, model.content_view, model.fusion_view]:
        assert torch.count_nonzero(view(masked)[:, view.field_slices['user_id']]) == 0
    before = model.predict(masked)
    changed = Interaction(dict(batch.interaction, user_id=torch.zeros_like(batch['user_id']),
                               user_id_raw=batch['user_id_raw']+1000))
    assert torch.equal(before, model.predict(changed))
    model.calculate_loss(masked).backward()
    for table in uid_tables(model).values():
        assert table.padding_idx == 0
        assert table.weight.grad is None or torch.count_nonzero(table.weight.grad) == 0
    head = model.fusion_net if name.startswith('fusion') else model.content_net
    assert head[-1].weight.grad.abs().sum() > 0
