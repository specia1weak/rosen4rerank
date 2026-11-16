"""反馈关联、分组相关和User-AUC的独立小样本核验。"""
import polars as pl
import pytest
from scipy.stats import spearmanr
from workspace.kr_time_budget.pipeline.backtest_components import (
    feedback_from_events, grouped_spearman, user_auc,
)


def test_feedback_uses_current_event_membership_and_preserves_item_order():
    events=pl.DataFrame(dict(event_id=list(range(12)),video_id=list(range(100,112)),
        session_id=[7]*12,user_id=[3]*12,req_idx=[1]*6+[2]*6,
        is_click=[0]*6+[1,1,0,0,0,0],long_view=[1,0,1,0,1,0]+[1]*6,
        play_time_ms=[1000]*6+[400000]*6,play_clip=[1000]*6+[300000]*6,
        duration_ms=[60000]*12,time_ms=[i*1000 for i in range(12)],session_start_ms=[0]*12))
    members=pl.DataFrame(dict(event_id=list(range(12)),request_id=[9]*6+[10]*6,
                             event_idx=list(range(12)))).reverse()
    result=feedback_from_events(members.lazy(),events.reverse().lazy()).collect().sort('request_id')
    assert result['curr_longview_cnt'].to_list()==[3,6]
    assert result['curr_click_cnt'].to_list()==[0,2]
    assert result['curr_longview_rate'].to_list()==[.5,1.]
    assert result['observed_items'].to_list()==[list(range(100,106)),list(range(106,112))]
    assert result['curr_watch_min'].to_list()==[.1,40.]
    assert result['curr_watch_clip_min'].to_list()==[.1,30.]
    assert result['recomputed_elapsed_min'].to_list()==[6000/60000,311000/60000]
    duplicate=pl.concat([members,members.head(1)])
    with pytest.raises(pl.exceptions.ComputeError):
        feedback_from_events(duplicate.lazy(),events.lazy()).collect()


def test_grouped_spearman_ties_and_constant_exclusion_match_scipy():
    frame=pl.DataFrame(dict(group=[1,1,1,2,2,2,2,3,3],
        score=[1.,1.,2.,4.,3.,2.,1.,0.,0.],target=[2.,2.,1.,1.,3.,2.,4.,0.,1.]))
    result=grouped_spearman(frame,['score'],['target'],['group'])[('score','target')]
    a=float(spearmanr([1,1,2],[2,2,1]).statistic)
    b=float(spearmanr([4,3,2,1],[1,3,2,4]).statistic)
    assert result['value']==pytest.approx((a*3+b*4)/7)
    assert result['eligible_rows']==7 and result['eligible_groups']==2 and result['total_groups']==3


def test_grouped_spearman_does_not_confuse_pooled_depth_with_within_group_order():
    frame=pl.DataFrame(dict(group=[1]*3+[2]*3,score=[1.,2.,3.,10.,11.,12.],target=[3.,2.,1.,12.,11.,10.]))
    assert spearmanr(frame['score'],frame['target']).statistic>0
    assert grouped_spearman(frame,['score'],['target'],['group'])[('score','target')]['value']==pytest.approx(-1)


def test_external_user_auc_row_weighted_ties_and_single_class_users():
    frame=pl.DataFrame(dict(user_id_raw=[1,1,2,2,2,3,3],score=[1.,1.,1.,3.,2.,0.,1.],
                            label=[0,1,0,1,1,1,1]))
    result=user_auc(frame,['score'],'label')['score']
    assert result['value']==pytest.approx(.8)
    assert result['eligible_users']==2 and result['eligible_rows']==5 and result['total_users']==3


def test_large_constant_groups_are_explicitly_excluded_even_if_numeric_corr_is_finite():
    n=10003
    frame=pl.DataFrame(dict(group=[1]*n,score=[7.]*n,target=[float(i%7) for i in range(n)]))
    result=grouped_spearman(frame,['score'],['target'],['group'])[('score','target')]
    assert result['value'] is None and result['eligible_rows']==0 and result['eligible_groups']==0
