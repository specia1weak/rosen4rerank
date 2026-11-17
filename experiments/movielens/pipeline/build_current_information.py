"""保留旧ML1M成员/切分/行序，当前累计pace替换上一步pace，追加rating/5。"""
from contextlib import redirect_stdout,redirect_stderr
import json
from pathlib import Path
import polars as pl
from experiments.movielens.pipeline.prepare_sessions import WORKSPACE,sha
from experiments.movielens.pipeline.current_schema import manager_at,FEATURES,CURRENT_RATING


def current_information(frame):
    return frame.drop('pace_previous').with_columns(
        (pl.col('elapsed_min_raw')/pl.col('req_idx_raw')).log1p().alias('pace'),
        (pl.col('rating_audit')/5.).cast(pl.Float32).alias(CURRENT_RATING))


def build(output):
    parent = WORKSPACE/'data/continuation_v1'
    old = json.loads((parent/'manifest.json').read_text())
    sources = {**old['source_hashes'], **{str(p.resolve()):sha(p) for p in [
        Path(__file__),Path(__file__).with_name('current_schema.py'),parent/'manifest.json',
        *[parent/f'{s}{suffix}.parquet' for s in ['train','valid','test'] for suffix in ['', '_raw']]]}}
    assert all(sha(p)==h for p,h in sources.items())
    manager = manager_at(output/'schema')
    manager.fit(current_information(pl.read_parquet(parent/'train_raw.parquet')).lazy())
    splits = {}
    for split in ['train','valid','test']:
        raw = current_information(pl.read_parquet(parent/f'{split}_raw.parquet'))
        assert raw[CURRENT_RATING].is_between(.2,1.).all()
        encoded = manager.transform(raw.lazy()).collect()
        original = pl.read_parquet(parent/f'{split}.parquet')
        order = original.select('request_id')
        encoded = order.join(encoded,on='request_id',how='left',validate='1:1',maintain_order='left')
        keep = [c for c in original.columns if c!='pace_previous']
        assert encoded.select(keep).equals(original.select(keep))
        assert encoded.height == original.height and encoded['pace'].min() >= 0
        encoded.write_parquet(output/f'{split}.parquet',row_group_size=65536)
        splits[split] = {**old['splits'][split], 'sha256':sha(output/f'{split}.parquet'),
            'original_values_labels_order_exact_except_pace':True}
        print('SPLIT',split,splits[split],flush=True)
    manifest = dict(version=2,parent=str(parent),source_hashes=sources,splits=splits,features=FEATURES,
        current_rating_input=True,additional_dense_feature=CURRENT_RATING,
        protocol={**old['protocol'],'target':'after current rating is known predict next rating in same session',
            'pace':'log1p(elapsed_min_raw/req_idx_raw), cumulative current elapsed minutes per rating',
            'rating':'current rating/5 as content/Fusion dense sidecar; budget never reads it',
            'unchanged':'session membership, splits, original columns except pace, physical order'})
    assert all(sha(p)==h for p,h in sources.items())
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETE',flush=True)


if __name__ == '__main__':
    output = WORKSPACE/'data/current_information_v2'
    output.mkdir(parents=True,exist_ok=False);(output/'logs').mkdir()
    print('日志：',output/'logs/build.log',flush=True)
    with (output/'logs/build.log').open('w') as f,redirect_stdout(f),redirect_stderr(f):build(output)
