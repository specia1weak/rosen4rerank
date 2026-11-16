"""Read-only v5 review: real-data scales/history and three in-memory gradient steps.

No saved model, schema or dataset is modified. Output is printed as JSON.
"""
import json
from pathlib import Path
import polars as pl
import torch
from torch.utils.data import DataLoader
from rosen.data.dataset import ParquetStreamDataset
from rosen.core.train.two_stage import StageSnapshot, StageReport
from rosen.utils.optimize import split_model_params_by_decay
from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager, EXTRA_COLUMNS, HISTORY_NUMERIC_FIELDS
from workspace.kr_time_budget.pipeline.time_budget_v5 import ExitModelV5

ROOT = Path('workspace/kr_time_budget')


def main():
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    data = ROOT/'data/v3'
    manager = request_manager(data/'schema', load=True)
    warm = ExitModelV5(manager)
    checkpoint = torch.load(ROOT/'runs/exit_v5/ours/stage_one/best.pt', map_location='cpu', weights_only=False)
    warm.load_state_dict(checkpoint['state_dict'])
    snapshot = StageSnapshot.capture(warm, StageReport(1, 421, {}, 'student', {}))
    model = warm.initialize_stage_two(snapshot)
    loader = DataLoader(ParquetStreamDataset(data/'train.parquet', manager, batch_size=4096,
                        shuffle=False, drop_last=False, extra_col_names=EXTRA_COLUMNS), batch_size=None)
    batches = []
    for batch in loader:
        batches.append(batch.interaction)
        if len(batches) == 3:
            break
    result = {'scope': 'KR data/v3 train first 3 physical batches; full-train history/scale scans; no full retraining',
              'warmup_checkpoint': 'runs/exit_v5/ours/stage_one/best.pt', 'gradient_steps': []}
    optimizer = torch.optim.AdamW(split_model_params_by_decay(model, weight_decay=1e-6), lr=.001, weight_decay=0)
    for step, batch in enumerate(batches, 1):
        optimizer.zero_grad(set_to_none=True)
        loss = model.calculate_loss(batch)
        loss.backward()
        watched = {'head': model.content_net[-1].weight, 'hidden0': model.content_net[0].weight,
                   'hidden2': model.content_net[2].weight,
                   'slate_tag_embedding': model.omni_embedding.emb_modules['slate_tags'].embedding.weight}
        norms = {name: parameter.grad.norm().item() for name, parameter in watched.items()}
        if step == 1:
            assert norms['head'] > 0 and norms['hidden0'] == norms['hidden2'] == norms['slate_tag_embedding'] == 0
        else:
            assert all(value > 0 for value in norms.values())
        assert all(p.grad is None for module in model.budget_modules for p in module.parameters())
        result['gradient_steps'].append({'step': step, 'loss': loss.item(), 'gradient_norms': norms})
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    # First-screen histories include req0 head remainders. Zero is not universally first-screen.
    raw = pl.scan_parquet(data/'requests_base.parquet').filter(pl.col('split') == 'train')
    first = raw.filter(pl.col('req_idx') == 1).select(
        'request_id', 'session_n', *HISTORY_NUMERIC_FIELDS, 'hist_tags').collect()
    nonzero = pl.any_horizontal(*[pl.col(c) != 0 for c in HISTORY_NUMERIC_FIELDS])
    result['first_request_history'] = {
        'first_request_rows': first.height,
        'with_head_remainder': first.filter(pl.col('session_n') % 6 != 0).height,
        'with_nonzero_feedback': first.filter(nonzero).height,
        'with_nonempty_tag_string': first.filter(pl.col('hist_tags').str.replace_all(',', '').str.len_chars() > 0).height,
    }
    hist = pl.scan_parquet(data/'train.parquet').select('req_idx_raw', 'hist_tags').with_columns(
        pl.when(pl.col('req_idx_raw') <= 2).then(pl.col('req_idx_raw').cast(pl.String)).otherwise(pl.lit('3+')).alias('position'),
        pl.col('hist_tags').list.len().alias('length'))
    result['history_lengths_full_train'] = hist.group_by('position').agg(
        pl.len().alias('rows'), pl.col('length').mean().alias('mean_tokens'),
        pl.col('length').min().alias('min_tokens'), pl.col('length').max().alias('max_tokens'),
        (pl.col('length') == 0).mean().alias('empty_fraction'),
        (pl.col('length') == 15).mean().alias('full_15_fraction')).sort('position').collect().to_dicts()
    result['dense_full_train'] = {}
    for name in ['elapsed_min', 'req_idx', 'prev1_click_cnt', 'slate_unique_tag_cnt', 'slate_dur_mean_min', 'sess_cum_hate_cnt']:
        result['dense_full_train'][name] = pl.scan_parquet(data/'train.parquet').select(
            pl.col(name).min().alias('min'), pl.col(name).quantile(.5).alias('p50'),
            pl.col(name).quantile(.99).alias('p99'), pl.col(name).max().alias('max')).collect().row(0, named=True)

    # No-history is padding only: no single placeholder embedding is introduced.
    fresh = warm.initialize_stage_two(snapshot)
    batch = batches[0]
    empty = dict(batch, hist_tags=torch.zeros_like(batch['hist_tags']))
    hist_setting = manager.get_setting('hist_tags')
    hist_vector = hist_setting.compute_tensor(empty, fresh.omni_embedding.emb_modules)
    assert torch.count_nonzero(hist_vector) == 0
    result['empty_history_embedding_max_abs'] = hist_vector.abs().max().item()
    result['history_agg'] = hist_setting.agg
    table = fresh.omni_embedding.emb_modules['hist_tags']
    result['embedding_init_std'] = table.init_std
    result['fresh_slate_tag_embedding_std'] = fresh.omni_embedding.emb_modules['slate_tags'].embedding.weight[1:].std().item()
    # Record actual network-input scales; clocks are zero in v5 Ours content input.
    fields = fresh.omni_embedding(batch, split_by='name')
    result['first_batch_field_rms'] = {name: fields[name].square().mean().sqrt().item()
        for name in ['elapsed_min', 'req_idx', 'slate_tags', 'hist_tags', 'prev1_click_cnt', 'slate_unique_tag_cnt']}
    result['clock_content_slices_are_zero'] = True
    offset = 0
    content = fresh.content_x(batch)
    for name, value in fields.items():
        if name in {'elapsed_min', 'req_idx'}:
            assert torch.count_nonzero(content[:, offset:offset+value.shape[-1]]) == 0
        offset += value.shape[-1]

    # Explicitly demonstrate that EXTRA_COLUMNS names do not preserve an independent raw copy.
    raw_ids = [int(i) for i in batch['request_id'][:8]]
    rows = pl.scan_parquet(data/'train.parquet').filter(pl.col('request_id').is_in(raw_ids)).collect()
    transformed = rows.lazy().with_columns(pl.col('elapsed_min').log1p())
    demo = next(iter(DataLoader(ParquetStreamDataset(transformed, manager, batch_size=8, shuffle=False,
        drop_last=False, extra_col_names=EXTRA_COLUMNS), batch_size=None)))
    result['same_name_extra_demo'] = {'raw_elapsed': rows['elapsed_min'][0],
        'batch_elapsed_after_log1p': demo['elapsed_min'][0].item(), 'independent_elapsed_raw_key_exists': 'elapsed_min_raw' in demo.interaction}
    result['conclusion'] = 'Zero head blocks upstream gradients on step1 only; hidden and embedding gradients are nonzero on steps2 and3.'
    print('AUDIT_JSON_BEGIN')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
