"""当前刷反馈实验：预测时点为该刷反馈已收齐之后，再预测下一刷。

保持原54字段、当前刷pace、静态预算34字段及独立Omni不变。
额外4列为当前6条曝光的click/long_view/positive/hate次数÷6，Float32[0,1]。
positive沿用like/comment/follow/forward任一为1，不读取下一刷、会话长度或退出标签。
4列只进入内容/Fusion首层，预算塔完全不读。schema及词表沿用冻结版本；
新增稠密列已在上游缩放，作为显式旁路读取，处理方式与pace一致。
使用零初始4→64线性投影加在原首层输出上，等价于首层拼接4列；
原首层GEMM不改，保证开关前向及共有参数初值精确相等。
"""
import json
from pathlib import Path
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from rosen.data.dataset import ParquetStreamDataset
from experiments.kuairand.pipeline.exit_model import KRExitModel
from experiments.kuairand.pipeline.limited_auxiliary_exit import LimitedAuxiliaryExitModel
from experiments.kuairand.pipeline.kr_request_schema import EXTRA_COLUMNS
from experiments.kuairand.pipeline.previous_pace import PREVIOUS_COLUMNS

FEEDBACK_FIELDS=['current_click_rate','current_longview_rate','current_positive_rate','current_hate_rate']


def feedback_rows(events):
    """events含request_id及当前原始反馈；调用方先按已冻结成员关联事件。"""
    positive=pl.any_horizontal(*[pl.col(c)==1 for c in ['is_like','is_comment','is_follow','is_forward']])
    return events.group_by('request_id').agg(pl.len().alias('feedback_event_count'),
        *[(pl.col(c)==1).sum().cast(pl.Float32).truediv(6.).alias(out) for c,out in
          [('is_click',FEEDBACK_FIELDS[0]),('long_view',FEEDBACK_FIELDS[1]),('is_hate',FEEDBACK_FIELDS[3])]],
        positive.sum().cast(pl.Float32).truediv(6.).alias(FEEDBACK_FIELDS[2]))


def add_feedback_projection(model):
    # No extra RNG consumption changes future shared initialization or the training stream.
    with torch.random.fork_rng():
        model.feedback_projection=nn.Linear(len(FEEDBACK_FIELDS),64,bias=False)
    nn.init.zeros_(model.feedback_projection.weight)


def feedback_probability(model,batch,network,view):
    hidden=network[0](model.network_view(batch,view))
    if model.use_current_feedback:
        x=torch.stack([batch[c].float() for c in FEEDBACK_FIELDS],-1)
        hidden=hidden+model.feedback_projection(x)
    return network[1:](hidden).squeeze(-1).sigmoid()


class CurrentFeedbackExitModel(LimitedAuxiliaryExitModel):
    def __init__(self,*args,use_current_feedback=True,**kwargs):
        super().__init__(*args,**kwargs)
        self.use_current_feedback=use_current_feedback
        add_feedback_projection(self)

    def components(self,batch):
        gamma,s,ls=self.budget_components(batch)
        stay=feedback_probability(self,batch,self.content_net,self.content_view)
        return gamma*stay,gamma,stay,s,ls


class CurrentFeedbackFusion(KRExitModel):
    def __init__(self,*args,use_current_feedback=True,**kwargs):
        super().__init__(*args,**kwargs)
        self.use_current_feedback=use_current_feedback
        add_feedback_projection(self)

    def predict(self,batch):
        return feedback_probability(self,batch,self.fusion_net,self.fusion_view)


def load_feedback_batches(data,manager,row_limits=None):
    data=Path(data)
    manifest=json.loads((data/'manifest.json').read_text())
    if manifest['version']!=6 or manifest['feedback_fields']!=FEEDBACK_FIELDS:
        raise ValueError('Requires v6 current-feedback dataset')
    result={}
    for split in ['train','valid','test']:
        frame=pl.scan_parquet(data/f'{split}.parquet').with_columns(pl.col('elapsed_min').alias('elapsed_min_raw'))
        if row_limits and split in row_limits:frame=frame.head(row_limits[split])
        ds=ParquetStreamDataset(frame,manager,batch_size=4096,shuffle=False,drop_last=False,
            extra_col_names=EXTRA_COLUMNS+['elapsed_min_raw']+PREVIOUS_COLUMNS+FEEDBACK_FIELDS)
        result[split]=list(DataLoader(ds,batch_size=None,num_workers=0))
    return result
