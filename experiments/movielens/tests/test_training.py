import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader
from rosen.data.dataset import ParquetStreamDataset
from experiments.movielens.pipeline.prepare_sessions import sessionize
from experiments.movielens.pipeline.prepare_training import make_features
from experiments.movielens.pipeline.schema import manager_at,FEATURES,EXTRAS,BUDGET_FIELDS
from experiments.movielens.pipeline.model import ML1MContinuationModel
from experiments.kuairand.pipeline.budget_stop_gradient import gradient_route_audit


def example_features(ratings=None):
    raw=pl.DataFrame(dict(event_id=list(range(6)),user_id=[1]*6,movie_id=[1,2,1,2,1,2],
        rating=ratings or [5,1,4,2,3,5],timestamp=[100,100,200,2101,2102,2103]))
    events,sessions=sessionize(raw)
    users=pl.DataFrame(dict(user_id=[1],gender=['F'],age=['25'],occupation=['1'],zip3=['123']))
    movies=pl.DataFrame(dict(movie_id=[1,2],genres=['Drama|Comedy','Action'],release_year=[1995,2000]))
    return make_features(events,sessions,users,movies)


def test_current_and_future_ratings_do_not_change_current_inputs():
    a=example_features();b=example_features([5,5,1,1,1,1])
    # 改变当前第2条和未来rating，前两条全部模型输入不变。
    assert a.head(2).select(FEATURES).equals(b.head(2).select(FEATURES))
    assert a['previous_rating_scaled'][1]==1.
    assert a['pace_previous'][0]==a['pace_previous'][1]==0.
    assert a['y_continue'].to_list()==[1,1,0,1,1,0]
    assert a['past_session_steps_log'][0]==0
    assert abs(a['past_session_steps_log'][3]-torch.log1p(torch.tensor(3.)).item())<1e-6
    for g in a.partition_by('session_id'):
        assert all(g[c].n_unique()==1 for c in BUDGET_FIELDS)


@pytest.mark.parametrize('mixture',[False,True])
def test_actual_omni_batch_gradient_and_raw_clock_separation(tmp_path,mixture):
    f=example_features()
    manager=manager_at(tmp_path/'schema');manager.fit(f.lazy())
    frame=manager.transform(f.lazy()).collect()
    batch=next(iter(DataLoader(ParquetStreamDataset(frame.lazy(),manager,batch_size=32,
        shuffle=False,drop_last=False,extra_col_names=EXTRAS),batch_size=None)))
    batch['user_id']=torch.zeros_like(batch['user_id'])
    model=ML1MContinuationModel(manager,mixture=mixture,requests_per_session=3.)
    audit=gradient_route_audit(model,batch)
    assert audit['joint_to_budget']=='all None' and audit['total_budget_equals_scaled_left']
    changed=dict(batch.interaction,elapsed_min_raw=batch['elapsed_min_raw']+10000,rating_audit=batch['rating_audit']*0)
    assert torch.equal(model.predict(batch),model.predict(changed))
    loss=model.calculate_loss(batch);loss.backward()
    assert torch.isfinite(loss)
    assert model.budget_net[-1].weight.grad.abs().sum()>0
    assert model.content_net[-1].weight.grad.abs().sum()>0
    assert model.last_loss_terms['auxiliary_main_ratio']<=.1+1e-12


def test_schema_vocabulary_train_only(tmp_path):
    f=example_features();manager=manager_at(tmp_path/'schema');manager.fit(f.lazy())
    setting=manager.get_setting('movie_id');before=dict(setting.vocab)
    manager.transform(f.with_columns(pl.lit(99999).alias('movie_id')).lazy()).collect()
    assert setting.vocab==before and '99999' not in setting.vocab
