import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from workspace.kr_time_budget.pipeline.build_requests import sessionize, aggregate_requests
from workspace.kr_time_budget.pipeline.features import past_daily
from workspace.kr_time_budget.pipeline.run_exit import grouped_auc
from workspace.kr_time_budget.pipeline.time_budget import mixture_survival


def events(n=14):
    f = pl.DataFrame({'user_id': [1]*n, 'time_ms': list(range(0,n*10000,10000)), 'source_order': [0]*n, 'source_row': list(range(n)), 'play_time_ms':[5000]*n})
    f = sessionize(f).with_columns(
        pl.arange(0,n).alias('video_id'), pl.lit('tag').alias('tag'), pl.lit(1).alias('author_id'),
        pl.lit(10.).alias('dur_s'), pl.lit(0).alias('is_rand'), pl.lit(.5).alias('comp'),
        pl.lit(0).alias('long_view'), pl.lit(0).alias('pos'), pl.lit(0).alias('is_hate'),
        pl.lit(20220409).alias('date'), pl.lit(20220409).alias('session_date'),
        pl.lit(12).alias('session_hour'), pl.lit(1).alias('session_tab'), pl.lit(135.).alias('session_wall_s'),
        pl.lit(2).alias('session_requests'), pl.lit(0).alias('day_prev_sess_cnt'), pl.lit(0.).alias('day_prev_time_s'),
    )
    return f


def test_reverse_chunks_fold_remainder_and_future_labels():
    r = aggregate_requests(events())
    assert r['req_idx'].to_list() == [1,2]
    assert r['slate_items'].to_list() == [list(range(2,8)),list(range(8,14))]
    assert r['history_items'][0].to_list() == [0,1]
    assert r['elapsed_time_s'].to_list() == [20.,80.]
    assert r['elapsed_end_s_DIAGNOSTIC'].to_list() == [75.,135.]
    assert r['y_continue'].to_list() == [1,0]
    assert r['remaining_watch_s'].to_list() == [60.,30.]
    assert r['prev1_play_s'].to_list() == [10.,30.]


def test_current_feedback_never_enters_current_preexposure_features():
    a = aggregate_requests(events())
    e = events().with_columns(pl.when(pl.col('req_idx') == 1).then(90000).otherwise(pl.col('play_time_ms')).alias('play_time_ms'), pl.when(pl.col('req_idx') == 1).then(1.).otherwise(pl.col('comp')).alias('comp'))
    b = aggregate_requests(e)
    for c in ['slate_items','history_items','elapsed_time_s','prev1_play_s','prev1_comp_rate','sess_cum_comp_rate']:
        assert a[c][0] == b[c][0] if a[c].dtype != pl.List(pl.Int64) else a[c][0].to_list() == b[c][0].to_list()
    assert a['curr_watch_s'][0] != b['curr_watch_s'][0]


def test_idle_threshold_uses_end_of_previous_play():
    f = events(3).select('user_id','time_ms','source_order','source_row','play_time_ms').with_columns(pl.Series('time_ms',[0,1805000,3610001]))
    r = sessionize(f)
    assert r['session_id'].to_list() == [1,1,2]


def test_priors_exclude_same_day_and_future():
    obs = pl.DataFrame({'video_id':[1,1,1], 'date':[1,2,3], 'score':[1.,100.,10000.]})
    q = pl.DataFrame({'video_id':[1,1,2], 'date':[2,3,3]})
    r = past_daily(q,obs,['video_id'],['score']).sort(['date','video_id'])
    assert r['sum_score'].to_list() == [1.,101.,0.]
    assert r['prior_n'].to_list() == [1.,2.,0.]


def test_survival_log_ratio_is_valid_even_when_survival_underflows():
    raw = torch.zeros(5, 6, requires_grad=True)
    elapsed = torch.tensor([0.,1.,10.,100.,10000.])
    gamma,s,log_s = mixture_survival(raw,elapsed,torch.ones(5))
    assert torch.isfinite(gamma).all() and ((gamma>=0)&(gamma<=1)).all()
    assert torch.all(s[1:] <= s[:-1])
    zero,_,_ = mixture_survival(raw,elapsed,torch.zeros(5))
    assert torch.allclose(zero,torch.ones_like(zero))
    (-gamma.clamp(min=1e-6).log().mean()).backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_grouped_auc_weighting_ties_and_single_class_exclusions():
    y=np.array([0,1,0,1,1,1,0,1])
    p=np.array([.1,.1,.2,.9,.4,.5,.8,.7])
    g=np.array([0,0,1,1,2,2,3,3])
    result=grouped_auc(y,p,g)
    expected=np.mean([roc_auc_score(y[g==i],p[g==i]) for i in [0,1,3]])
    assert abs(result['auc']-expected)<1e-9
    assert result['eligible_groups']==3
    assert result['eligible_rows']==6


def test_model_product_shared_initialization_masks_and_input_allowlist(tmp_path):
    from workspace.kr_time_budget.pipeline.features import make_manager, NUMS, CATS, B1, B2, B3
    from workspace.kr_time_budget.pipeline.time_budget import ExitModel
    from workspace.kr_time_budget.pipeline.run_exit import state_hash
    data = {c: [0., 1.] for c in NUMS}
    data.update({c: ['a', 'b'] for c in CATS})
    data.update({c: [[1, 2, 3, 4, 5, 6]] * 2 for c in ['slate_items', 'slate_tags', 'history_items', 'history_tags']})
    data['slate_durations'] = [[.1]*6, [.2]*6]
    data['y_continue'] = [0, 1]
    raw = pl.DataFrame(data)
    mgr = make_manager(tmp_path, raw)
    mgr.fit(raw.lazy())
    frame = mgr.transform(raw.lazy()).collect()
    from rosen.data.padding import FeatureContext, TensorFormatter
    from rosen.data.dataset import arrow_table_to_batch_dict
    formatter = TensorFormatter(FeatureContext.from_manager(mgr))
    batch = formatter.format(arrow_table_to_batch_dict(frame.to_arrow())).interaction
    batch.update(elapsed_minutes=torch.tensor([0., 10.]), delta_minutes=torch.tensor([1., 1.]))
    models = []
    for mode in ['content','budget','fusion','ours']:
        torch.manual_seed(2026)
        models.append(ExitModel(mgr, mode=mode))
    assert len({state_hash(m) for m in models}) == 1
    ours = models[-1]
    y, gamma, stay, _, _ = ours.components(batch)
    assert torch.equal(y, gamma * stay)
    assert torch.equal(ours.predict(batch), y)
    ours.stage = 1
    assert torch.equal(ours.predict(batch), gamma)
    ours.calculate_loss(batch).backward()
    assert any(p.grad is not None for p in ours.budget_net.parameters())
    assert all(p.grad is None for p in ours.content_net.parameters())
    # Changing labels or adding diagnostics cannot alter predictions.
    ours.eval()
    before = ours.predict(batch)
    batch['y_continue'] = 1 - batch['y_continue']
    batch['elapsed_end_s_DIAGNOSTIC'] = torch.tensor([1e9, 1e9])
    batch['remaining_watch_s'] = torch.tensor([1e9, 1e9])
    assert torch.equal(before, ours.predict(batch))
    content = models[0]
    old = content.predict(batch)
    changed = dict(batch)
    for c in B1 + B2:
        changed[c] = torch.ones_like(batch[c]) * 100
    assert torch.equal(old, content.predict(changed))
    budget = models[1]
    old = budget.predict(batch)
    changed = dict(batch)
    for c in B3:
        changed[c] = torch.ones_like(batch[c]) * 100
    assert torch.equal(old, budget.predict(changed))


def test_terminal_only_session_auc_is_trivial_for_negative_request_index():
    # A metric-design limitation, independent of any fitted budget mechanism.
    y = np.array([1, 1, 0, 1, 0, 0])
    index = np.array([1, 2, 3, 1, 2, 1])
    sessions = np.array([1, 1, 1, 2, 2, 3])
    result = grouped_auc(y, -index, sessions)
    assert result['auc'] == 1.0
    assert result['eligible_groups'] == 2
    assert result['eligible_rows'] == 5
