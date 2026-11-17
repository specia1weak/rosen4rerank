"""小型真实Omni批次，供边界与梯度路由测试使用。"""
import polars as pl
import pytest
import torch
from rosen.data.dataset import arrow_table_to_batch_dict
from rosen.data.padding import FeatureContext, TensorFormatter
from experiments.kuairand.pipeline.kr_request_schema import request_manager
from experiments.kuairand.pipeline.exit_model import KRExitModel

@pytest.fixture
def model_batch(tmp_path):
    mgr=request_manager(tmp_path)
    data={s.field_name: ([0.,1.] if not s.requires_embedding_module else ['1','2']) for s in mgr.settings}
    for c in ['hist_tags','slate_tags']:data[c]=['20,43','20,67']
    for c in ['slate_video_types','slate_music_types']:data[c]=['1,1','1,2']
    data['y_continue']=[0,1]
    raw=pl.DataFrame(data)
    # min_freq=10 is deliberate in production; repeat here to exercise real user vocab.
    raw=pl.concat([raw]*6)
    mgr.fit(raw.lazy())
    frame=mgr.transform(raw.lazy()).collect()
    b=TensorFormatter(FeatureContext.from_manager(mgr)).format(arrow_table_to_batch_dict(frame.to_arrow())).interaction
    torch.manual_seed(2026)
    return KRExitModel(mgr),b,mgr
