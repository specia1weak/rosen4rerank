import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader
from rosen.data.dataset import ParquetStreamDataset
from experiments.movielens.tests.test_training import example_features
from experiments.movielens.pipeline.current_schema import manager_at,FEATURES,EXTRA_COLUMNS,CURRENT_RATING
from experiments.movielens.pipeline.build_current_information import current_information
from experiments.movielens.pipeline.current_model import CurrentML1MModel


def test_current_feedback_boundary_and_explicit_scaling():
    a = current_information(example_features())
    b = current_information(example_features([5,5,1,1,1,1]))
    assert a.head(2).select([c for c in FEATURES if c!='pace']).equals(b.head(2).select([c for c in FEATURES if c!='pace']))
    assert a[CURRENT_RATING][1] == pytest.approx(.2)
    assert b[CURRENT_RATING][1] == pytest.approx(1.)
    # 改未来rating不改变第2步当前评分或前序历史，下一步才受影响。
    c = current_information(example_features([5,1,1,1,1,1]))
    assert a.head(2).select(FEATURES+[CURRENT_RATING]).equals(c.head(2).select(FEATURES+[CURRENT_RATING]))
    assert a['pace'][2] == pytest.approx(torch.log1p(torch.tensor((100/60)/3)).item())


@pytest.mark.parametrize('mixture',[False,True])
def test_real_omni_graph_routes_initialization_and_rating_projection(tmp_path,mixture):
    raw = current_information(example_features())
    mgr = manager_at(tmp_path);mgr.fit(raw.lazy())
    frame = mgr.transform(raw.lazy()).collect()
    batch = next(iter(DataLoader(ParquetStreamDataset(frame.lazy(),mgr,batch_size=32,
        shuffle=False,drop_last=False,extra_col_names=EXTRA_COLUMNS),batch_size=None)))
    batch['user_id'] = torch.zeros_like(batch['user_id'])
    common = None
    for variant in ['none','weak','strong','detach_0.1','detach_1']:
        torch.manual_seed(2026)
        m = CurrentML1MModel(mgr,mixture=mixture,variant=variant,requests_per_session=3.)
        if common is None:common = {k:v.clone() for k,v in m.state_dict().items()}
        else:assert all(torch.equal(v,common[k]) for k,v in m.state_dict().items())
        terms = m.loss_terms(batch)
        budget = list(m.budget_net.parameters())
        grads = torch.autograd.grad(terms['joint_bce'],budget,allow_unused=True)
        assert all(g is None for g in grads) if variant.startswith('detach_') else any(g is not None and g.norm()>0 for g in grads)
        m.calculate_loss(batch).backward()
        assert m.rating_projection.weight.grad.norm() > 0
        assert m.budget_net[-1].weight.grad.norm() > 0
        changed = dict(batch.interaction,current_rating_scaled=1-batch[CURRENT_RATING])
        assert torch.equal(m.budget_raw(batch),m.budget_raw(changed))
        with torch.no_grad():m.rating_projection.weight.fill_(.1)
        assert not torch.equal(m.predict(batch),m.predict(changed))
