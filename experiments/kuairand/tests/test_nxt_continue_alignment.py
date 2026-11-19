"""新双塔与历史0.7基线的活跃路径对齐；历史实现只作为测试参考。"""
import json

import polars as pl
import torch

from experiments.kuairand.nxt_continue_v1 import (
    FEEDBACK_FIELDS, TimeBudgetExitModel, load_data, prepare_baseline_data, prepare_data,
)
from experiments.kuairand.tests.test_nxt_continue_data import raw_source
from workspace.kr_time_budget.pipeline.current_feedback import CurrentFeedbackExitModel


def test_frozen_data_order_pace_and_weak_loss_match_history(raw_source, tmp_path):
    source = prepare_data(raw_source, tmp_path / 'raw_processed')
    # 构造冻结v6交付表，故意采用非排序行序，检查新入口不会重新shuffle/编码。
    for split in ('train', 'valid', 'test'):
        frame = pl.read_parquet(source / f'{split}.parquet').reverse().drop('pace', 'elapsed_min_raw')
        frame.write_parquet(source / f'{split}.parquet')
    manifest_path = source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(version=6, feedback_fields=FEEDBACK_FIELDS)
    manifest_path.write_text(json.dumps(manifest))
    target = prepare_baseline_data(source, tmp_path / 'aligned')
    old_frame = pl.read_parquet(source / 'train.parquet')
    new_frame = pl.read_parquet(target / 'train.parquet')
    assert new_frame.select(old_frame.columns).equals(old_frame)
    manager, loaders = load_data(target, batch_size=8)
    batch = next(iter(loaders['train']))
    assert (batch['user_id'] == 0).all()
    assert torch.equal(batch['pace'], torch.log1p(batch['elapsed_min_raw'] / batch['req_idx_raw'].float()))
    kwargs = dict(left_weight=.1, right_weight=.1, max_aux_ratio=.1, requests_per_session=2.)
    torch.manual_seed(2026)
    old = CurrentFeedbackExitModel(manager, mode='ours', mixture=True, step_only=True, add_pace=True, **kwargs)
    torch.manual_seed(2026)
    new = TimeBudgetExitModel(manager, num_components=2, add_pace=True, **kwargs)
    old_named = dict(old.named_parameters())
    mappings = []
    for name, parameter in new.named_parameters():
        target_name = name.replace('budget_encoder.', 'budget_net.', 1) if name.startswith('budget_encoder.') else name
        if name.startswith('step_head.proj.'):
            target_name = name.replace('step_head.proj.', 'budget_net.2.', 1)
        reference = old_named[target_name]
        selected = reference[6:12] if name.startswith('step_head.proj.') else reference
        assert torch.equal(parameter, selected), name
        mappings.append((name, parameter, reference))
    # 独立6通道head与旧12通道head在小batch可采用不同GEMM归约顺序。
    torch.testing.assert_close(new.predict(batch), old.predict(batch), rtol=0, atol=2e-7)
    old_loss, new_loss = old.calculate_loss(batch), new.calculate_loss(batch)
    torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=2e-7)
    old_loss.backward()
    new_loss.backward()
    for name, parameter, reference in mappings:
        grad = reference.grad
        if grad is not None and name.startswith('step_head.proj.'):
            grad = grad[6:12]
        assert (grad is None) == (parameter.grad is None), name
        if grad is not None:
            torch.testing.assert_close(grad, parameter.grad, rtol=1e-5, atol=2e-7)
    assert old.last_loss_terms['auxiliary_main_ratio'] <= .1 + 1e-12
    # 当前反馈要确实生效，预算参数网仍只读背景/先验。
    with torch.no_grad():
        new.feedback_projection.weight.fill_(.1)
    changed = batch[:]
    for field in FEEDBACK_FIELDS:
        changed[field] = torch.zeros_like(batch[field])
    assert not torch.equal(new.predict(batch), new.predict(changed))
    assert torch.equal(new.step_head.raw(new.budget_encoder(new.budget_view(batch))),
                       new.step_head.raw(new.budget_encoder(new.budget_view(changed))))
