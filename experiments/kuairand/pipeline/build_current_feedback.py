"""只按冻结request_events成员聚合当刷反馈；旧数据和schema只读。"""
from contextlib import redirect_stdout,redirect_stderr
import json
from pathlib import Path
import shutil
import polars as pl
from experiments.kuairand.pipeline.exit_training import WORKSPACE
from experiments.kuairand.pipeline.current_feedback import FEEDBACK_FIELDS,feedback_rows
from experiments.kuairand.pipeline.build_timestamp_data import sha


def run(output):
    parent=WORKSPACE/'data/v5_previous_pace'
    raw=WORKSPACE/'data/v3'
    paths=[parent/'manifest.json',raw/'events.parquet',raw/'request_events.parquet',
        *[parent/f'{s}.parquet' for s in ['train','valid','test']],
        Path(__file__).resolve(),Path(__file__).with_name('current_feedback.py').resolve()]
    hashes={str(p):sha(p) for p in paths}
    members=pl.scan_parquet(raw/'request_events.parquet').filter(pl.col('request_id').is_not_null()).select('event_id','request_id')
    fields=['event_id','is_click','long_view','is_like','is_comment','is_follow','is_forward','is_hate']
    events=pl.scan_parquet(raw/'events.parquet').select(fields)
    for field in fields[1:]:
        values=events.select(pl.col(field).drop_nulls().unique()).collect()[field].to_list()
        assert set(values)<=set([0,1]),(field,values)
    payload=members.join(events,on='event_id',how='left',validate='1:1')
    stats=feedback_rows(payload).collect()
    assert stats['feedback_event_count'].min()==stats['feedback_event_count'].max()==6
    assert stats.select(FEEDBACK_FIELDS).null_count().sum_horizontal().item()==0
    assert stats.select(pl.all().exclude('request_id','feedback_event_count').is_between(0,1).all()).row(0)==(True,)*4
    stats.write_parquet(output/'request_feedback.parquet')
    splits={}
    for split in ['train','valid','test']:
        original=pl.read_parquet(parent/f'{split}.parquet')
        joined=original.join(stats.select('request_id',*FEEDBACK_FIELDS),on='request_id',how='left',validate='1:1',maintain_order='left')
        assert joined.select(original.columns).equals(original)
        assert joined.select(FEEDBACK_FIELDS).null_count().sum_horizontal().item()==0
        joined.write_parquet(output/f'{split}.parquet',row_group_size=65536)
        splits[split]=dict(rows=joined.height,sessions=joined['session_id'].n_unique(),
            sha256=sha(output/f'{split}.parquet'),parent_columns_and_physical_order_exact=True)
        print('SPLIT',split,splits[split],flush=True)
    shutil.copytree(parent/'schema',output/'schema')
    protocol=dict(parent=read(parent/'manifest.json')['protocol'],feedback='current six observed events; counts/6; no shift; after feedback available',
        prediction='after current screen feedback is available, before observing next screen',
        pace='current log1p(elapsed_min_raw/req_idx_raw)',excludes='current playback duration, last-event indicators, future/session-length features',
        label='unchanged next reconstructed screen exists',event_membership='frozen reverse-six chunks; not native online request IDs')
    assert stats.height==sum(s['rows'] for s in splits.values())
    assert all(sha(p)==h for p,h in hashes.items())
    manifest=dict(version=6,parent=str(parent),schema='frozen parent 54-field schema; 4 explicitly scaled dense sidecar features',
        feedback_fields=FEEDBACK_FIELDS,splits=splits,protocol=protocol,source_hashes=hashes)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETE',flush=True)


def read(path):return json.loads(path.read_text())


if __name__=='__main__':
    out=WORKSPACE/'data/v6_current_feedback';out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir()
    with (out/'logs/build.log').open('w') as f,redirect_stdout(f),redirect_stderr(f):run(out)
