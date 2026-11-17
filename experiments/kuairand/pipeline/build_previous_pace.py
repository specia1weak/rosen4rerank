"""v4纯时间戳数据上添加上一刷分钟/屏号；不改原54字段、标签和物理顺序。"""
from contextlib import redirect_stdout,redirect_stderr
import json
import shutil
from pathlib import Path
import polars as pl
from experiments.kuairand.pipeline.exit_training import WORKSPACE
from experiments.kuairand.pipeline.build_timestamp_data import sha
from experiments.kuairand.pipeline.previous_pace import add_previous_clock,PACE_DEFINITION


def run(output):
    source=WORKSPACE/'data/v4_timestamp'
    m=json.loads((source/'manifest.json').read_text())
    paths=[source/'manifest.json',*[source/f'{s}.parquet' for s in ['train','valid','test']],
           Path(__file__),Path(__file__).with_name('previous_pace.py')]
    hashes={str(p):sha(p) for p in paths}
    splits={}
    for split in ['train','valid','test']:
        frame=pl.read_parquet(source/f'{split}.parquet')
        out=add_previous_clock(frame)
        out.write_parquet(output/f'{split}.parquet',row_group_size=65536)
        assert out.select(frame.columns).equals(frame)
        assert (out.filter(pl.col('req_idx_raw')==1)['prev_elapsed_min_raw']==0).all()
        assert (out.filter(pl.col('req_idx_raw')>1)['prev_req_idx_raw']==
                out.filter(pl.col('req_idx_raw')>1)['req_idx_raw']-1).all()
        splits[split]={**m['splits'][split],'rows':out.height,'first_requests':out.filter(pl.col('req_idx_raw')==1).height,
            'original_columns_and_order_exact':True,'sha256':sha(output/f'{split}.parquet')}
        print('SPLIT',split,splits[split],flush=True)
    shutil.copytree(source/'schema',output/'schema')
    manifest={**m,'version':5,'parent_data':str(source),'parent_manifest_sha256':sha(source/'manifest.json'),
        'protocol':{**m['protocol'],'pace':PACE_DEFINITION,'pace_source':'keyed previous request, before batch/shuffle; no current-screen data'},
        'splits':splits,'added_audit_columns':['prev_elapsed_min_raw','prev_req_idx_raw'],
        'source_hashes':hashes}
    assert all(sha(p)==h for p,h in hashes.items())
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (WORKSPACE/'data/current.json').write_text(json.dumps(dict(data='v5_previous_pace',version=5,
        clock=m['protocol']['clock'],pace=PACE_DEFINITION,new_model_results_available=False),indent=2)+'\n')
    print('READY',output,flush=True)


if __name__=='__main__':
    output=WORKSPACE/'data/v5_previous_pace'
    output.mkdir(parents=True,exist_ok=False);(output/'logs').mkdir()
    with (output/'logs/build.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):run(output)
    print('完成：',output)
