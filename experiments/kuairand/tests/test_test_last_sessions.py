import polars as pl
from experiments.kuairand.prepare_test_last_sessions import classify_test_tails


def test_tail_confirmation_uses_idle_window_not_last_marker_alone():
    sessions=pl.DataFrame(dict(user_id=[1,2,3,4,5,6],session_id=[11,22,33,44,55,66],
        first_session=[1,2,3,44,5,6],last_session=[11,22,33,44,55,66],
        session_date=[20220508]*6,end_date=[20220508]*6,session_n=[12,12,12,12,5,12],tab=[1,1,1,1,1,9]))
    ends=pl.DataFrame(dict(session_id=[11,22,33,44,55,66],last_event_ms=[1_000_000]*6,
        last_play_clip=[0,100_000,0,0,0,0]))
    # 11/33窗口>30分钟；22加上最后视频观看后不够30分钟；其他结构过滤保留。
    actual=classify_test_tails(sessions,ends,cutoff_ms=2_800_001,tabs=[1])
    assert actual['session_id'].to_list()==[11,22,33]
    assert actual['end_observed'].to_list()==[True,False,True]
    exact=classify_test_tails(sessions,ends,cutoff_ms=2_800_000,tabs=[1])
    assert not exact['end_observed'].any()
