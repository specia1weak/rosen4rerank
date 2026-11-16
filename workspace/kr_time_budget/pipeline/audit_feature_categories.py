"""Audit the 30+6+18 contract against existing KR v3 data, without rebuilding it."""
import hashlib
import json
from pathlib import Path
import polars as pl
from torch.utils.data import DataLoader
from rosen.data.dataset import ParquetStreamDataset
from workspace.kr_time_budget.pipeline.kr_request_schema import (
    request_manager, FEATURES, EXTRA_COLUMNS, FEATURE_CATEGORIES, USER_PROFILE_FIELDS,
    SESSION_CONTEXT_FIELDS, BUDGET_PRIOR_FIELDS, SESSION_CLOCK_FIELDS,
    CONTENT_HISTORY_FIELDS, CONTENT_SLATE_FIELDS, BUDGET_PARAMETER_FIELDS,
    BUDGET_SCOPE_FIELDS, CONTENT_TOWER_FIELDS, CONTENT_MASK_FIELDS, LOG1P_FIELDS,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = ROOT/'data/v3'
    manifest_path = data/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['features'] == FEATURES
    manager = request_manager(data/'schema', load=True)
    assert {s.field_name for s in manager.settings} == set(FEATURES)
    assert all(manager.get_setting(c).min_val == 0 and manager.get_setting(c).max_val == 1
               for c in ['elapsed_min', 'req_idx'])
    output = {
        'contract': 'shared30_budget6_content18', 'data': 'workspace/kr_time_budget/data/v3',
        'dataset_rebuild_required': False, 'dataset_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'categories': FEATURE_CATEGORIES,
        'subgroups': {'shared_user': USER_PROFILE_FIELDS, 'shared_context': SESSION_CONTEXT_FIELDS,
                     'budget_priors': BUDGET_PRIOR_FIELDS, 'clocks': SESSION_CLOCK_FIELDS,
                     'content_history': CONTENT_HISTORY_FIELDS, 'content_slate': CONTENT_SLATE_FIELDS},
        'target_routes': {'budget_parameters': BUDGET_PARAMETER_FIELDS, 'budget_scope': BUDGET_SCOPE_FIELDS,
                          'content': CONTENT_TOWER_FIELDS, 'content_mask': sorted(CONTENT_MASK_FIELDS), 'fusion': FEATURES},
        'existing_v5_budget_formula': {'elapsed_min': 'consumed, minutes, delta=1 minute',
                                     'req_idx': 'not consumed; dual-clock formula pending'},
        'model_status': 'classification complete; historical v5 routing unchanged; no new full training',
        'log1p_fields': LOG1P_FIELDS, 'splits': {},
    }
    raw = pl.scan_parquet(data/'features_raw.parquet')
    stable = raw.group_by('session_id').agg(*[pl.col(c).n_unique().alias(c) for c in BUDGET_PARAMETER_FIELDS])
    maxima = stable.select(*[pl.col(c).max().alias(c) for c in BUDGET_PARAMETER_FIELDS]).collect().row(0, named=True)
    assert all(value == 1 for value in maxima.values()), maxima
    output['budget_parameter_fields_constant_within_session'] = maxima
    for split in ['train', 'valid', 'test']:
        scan = pl.scan_parquet(data/f'{split}.parquet')
        stored = set(scan.collect_schema().names())
        assert set(FEATURES) <= stored
        ids = scan.select('request_id').collect()['request_id']
        expected = manifest['splits'][split]
        order_hash = hashlib.sha256(ids.to_numpy().tobytes()).hexdigest()
        assert len(ids) == expected['rows'] and order_hash == expected['request_order_sha256']
        nulls = scan.select(*[pl.col(c).null_count().alias(c) for c in FEATURES]).collect().row(0, named=True)
        assert not any(nulls.values()), nulls
        # Separate metadata categories from actual runtime decoding: one native batch per split.
        batch = next(iter(DataLoader(ParquetStreamDataset(data/f'{split}.parquet', manager,
                     batch_size=8, shuffle=False, drop_last=False, extra_col_names=EXTRA_COLUMNS), batch_size=None)))
        assert all(c in batch.interaction for c in FEATURES)
        output['splits'][split] = {'rows': len(ids), 'request_order_sha256': order_hash,
                                  'feature_count': len(FEATURES), 'null_features': 0,
                                  'native_batch_rows': len(batch['request_id'])}
        print('VERIFIED', split, len(ids), 'rows; 54 fields; original order preserved', flush=True)
    target = ROOT/'feature_categories.json'
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False)+'\n')
    print('COMPLETE', target, 'v3 reused; req_idx formula explicitly pending', flush=True)


if __name__ == '__main__':
    main()
