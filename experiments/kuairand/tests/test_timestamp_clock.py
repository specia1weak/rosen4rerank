import numpy as np
import polars as pl

from experiments.kuairand.pipeline.timestamp_clock import timestamp_boundaries


def test_clock_ignores_playback_and_includes_head_history_origin():
    # 0号片段只作为历史，但它的首事件仍是本场起点；输入刻意打乱。
    events=pl.DataFrame({'session_id':[1,1,1,1], 'req_idx':[2,0,1,1],
        'time_ms':[240000,0,180000,60000], 'play_time_ms':[10000,300000,120000,0]}).lazy()
    sessions,requests=timestamp_boundaries(events)
    requests=requests.collect().sort('req_idx')
    np.testing.assert_array_equal(requests['elapsed_min'],[0.,3.,4.])
    assert sessions.collect()['session_min'][0]==4.
    changed=events.with_columns(pl.lit(9999999).alias('play_time_ms'))
    _,other=timestamp_boundaries(changed)
    assert requests.equals(other.collect().sort('req_idx'))


def test_timestamp_ties_keep_zero_without_fabricated_duration():
    events=pl.DataFrame({'session_id':[1,1,2], 'req_idx':[1,2,1],
                         'time_ms':[5000,5000,9000]}).lazy()
    sessions,requests=timestamp_boundaries(events)
    assert (requests.collect()['elapsed_min']==0).all()
    assert (sessions.collect()['session_min']==0).all()
