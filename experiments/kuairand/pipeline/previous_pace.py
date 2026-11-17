"""pace时间边界：纯时间戳，截止上一刷；首刷为0，不使用头部余数刷。

在物理shuffle之前按(session_id, req_idx_raw-1)精确连接上一刷；
不在batch中shift，不用当前elapsed反推，不读当前视频的播放/互动。
维持模型参数结构，只替换追加给内容塔的一个pace标量。
"""
import json
from pathlib import Path
import polars as pl
import torch
from torch.utils.data import DataLoader
from rosen.data.dataset import ParquetStreamDataset
from experiments.kuairand.pipeline.kr_request_schema import EXTRA_COLUMNS
from experiments.kuairand.pipeline.limited_auxiliary_exit import LimitedAuxiliaryExitModel
from experiments.kuairand.pipeline.timestamp_clock import CLOCK

PREVIOUS_COLUMNS=['prev_elapsed_min_raw','prev_req_idx_raw']
PACE_DEFINITION='log1p(prev_elapsed_min_raw / max(prev_req_idx_raw,1)); first_req=0'


def add_previous_clock(frame):
    """键连接保持原物理行序；缺失前序刷不能偷偷填0，除非当前为首刷。"""
    old_columns=frame.columns
    lookup=frame.select('session_id',
        (pl.col('req_idx_raw')+1).alias('req_idx_raw'),
        pl.col('elapsed_min').alias('prev_elapsed_min_raw'),
        pl.col('req_idx_raw').alias('prev_req_idx_raw'))
    out=frame.join(lookup,on=['session_id','req_idx_raw'],how='left',validate='1:1',maintain_order='left')
    missing=out.filter((pl.col('req_idx_raw')>1)&pl.col('prev_req_idx_raw').is_null()).height
    if missing:
        raise ValueError(f'{missing} requests missing previous screen in the same split/session')
    out=out.with_columns(*[pl.when(pl.col('req_idx_raw')==1).then(pl.lit(0)).otherwise(pl.col(c)).cast(
        pl.Float32 if c=='prev_elapsed_min_raw' else pl.Int64).alias(c) for c in PREVIOUS_COLUMNS])
    assert out.select(old_columns).equals(frame)
    return out


def previous_pace(batch):
    return torch.log1p(batch['prev_elapsed_min_raw'].float().clamp(min=0)/
                       batch['prev_req_idx_raw'].float().clamp(min=1))


class PreviousPaceExitModel(LimitedAuxiliaryExitModel):
    """与旧模型参数逐值同初始；current为严格配对对照，previous为当前方案。"""
    def __init__(self,*args,pace_scope='previous',**kwargs):
        if pace_scope not in {'previous','current'}:
            raise ValueError(pace_scope)
        super().__init__(*args,**kwargs)
        self.pace_scope=pace_scope

    def network_view(self,batch,view):
        if self.pace_scope=='current':
            return super().network_view(batch,view)
        x=view(batch)
        return torch.cat([x,previous_pace(batch).unsqueeze(-1)],dim=-1) if self.add_pace else x


def load_previous_batches(data,manager,row_limits=None):
    data=Path(data);m=json.loads((data/'manifest.json').read_text())
    if m['version']!=5 or m['protocol']['clock']!=CLOCK or m['protocol']['pace']!=PACE_DEFINITION:
        raise ValueError('Requires complete v5 previous-pace data')
    batches={}
    for split in ['train','valid','test']:
        frame=pl.scan_parquet(data/f'{split}.parquet').with_columns(pl.col('elapsed_min').alias('elapsed_min_raw'))
        if row_limits and split in row_limits:frame=frame.head(row_limits[split])
        dataset=ParquetStreamDataset(frame,manager,batch_size=4096,shuffle=False,drop_last=False,
            extra_col_names=EXTRA_COLUMNS+['elapsed_min_raw']+PREVIOUS_COLUMNS)
        batches[split]=list(DataLoader(dataset,batch_size=None,num_workers=0))
    return batches
