import polars as pl
from experiments.movielens.pipeline.prepare_sessions import sessionize, audit


def test_strict_gap_boundary_user_isolation_ties_and_one_rating_steps():
    # 原文件行序与时间序不同；同秒movie_id顺序与原行序相反。
    raw = pl.DataFrame(dict(event_id=[0, 1, 2, 3, 4, 5, 6],
        user_id=[2, 1, 1, 1, 1, 1, 2], movie_id=[9, 8, 7, 6, 5, 4, 3],
        rating=[5, 4, 3, 2, 1, 4, 5], timestamp=[5000, 100, 100, 1900, 3701, 3701, 5000]))
    events, sessions = sessionize(raw)
    assert events['event_id'].to_list() == [1, 2, 3, 4, 5, 0, 6]
    assert events['session_id'].to_list() == [1, 1, 1, 2, 2, 3, 3]
    assert events['step_idx'].to_list() == [1, 2, 3, 1, 2, 1, 2]
    assert events['elapsed_min'].to_list() == [0, 0, 30, 0, 0, 0, 0]
    assert sessions['steps'].to_list() == [3, 2, 2]
    assert sessions['last_observed_session'].to_list() == [False, True, True]
    assert audit(raw, events, sessions)['independent_numpy_session_and_step_exact']


def test_single_rating_session_is_preserved():
    raw = pl.DataFrame(dict(event_id=[0], user_id=[1], movie_id=[1], rating=[5], timestamp=[100]))
    events, sessions = sessionize(raw)
    assert events.height == sessions.height == 1
    assert events['step_idx'][0] == 1 and sessions['span_min'][0] == 0
    assert sessions['first_observed_session'][0] and sessions['last_observed_session'][0]
    assert audit(raw, events, sessions)['original_rows_and_values_preserved']
