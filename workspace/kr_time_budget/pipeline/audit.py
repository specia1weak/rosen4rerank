"""Audit offline exit artifacts; no training or datasource mutation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from workspace.kr_time_budget.pipeline.run_exit import grouped_auc, metrics


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('workspace/kr_time_budget/data/v1'))
    ap.add_argument('--runs', type=Path, default=Path('workspace/kr_time_budget/runs/exit_v1'))
    args = ap.parse_args()
    req = pl.read_parquet(args.data / 'requests.parquet')
    checks = {'request_rows': req.height, 'request_id_unique': req['request_id'].n_unique() == req.height, 'raw_nulls': int(req.null_count().sum_horizontal().item()), 'six_items_only': req['n_items'].unique().to_list(), 'one_fold_per_user': req.group_by('user_id').agg(pl.col('fold').n_unique().alias('n')).filter(pl.col('n') != 1).height == 0, 'one_terminal_label_per_session': req.group_by('session_id').agg((pl.col('y_continue') == 0).sum().alias('n')).filter(pl.col('n') != 1).height == 0}
    timed = req.sort(['session_id', 'req_idx']).with_columns(pl.col('elapsed_end_s_DIAGNOSTIC').shift(1).over('session_id').alias('previous_screen_end_s'))
    overlap = timed.filter(pl.col('previous_screen_end_s') > pl.col('elapsed_time_s') + 1e-6)
    overlap_s = overlap['previous_screen_end_s'] - overlap['elapsed_time_s']
    timing_audit = {'assumption': 'time_ms interpreted as start, as proposed by Argon; official source only says interaction timestamp', 'adjacent_complete_requests': timed['previous_screen_end_s'].drop_nulls().len(), 'overlapping_requests': overlap.height, 'fraction_of_all_requests': overlap.height / req.height, 'overlap_median_s': overlap_s.median(), 'overlap_max_s': overlap_s.max(), 'causal_status': 'INVALID_FOR_CAUSAL_COMPARISON', 'reason': 'Timestamp semantics and feedback availability are unverified; proxy slates are formed using future session length. These artifacts are only offline predictive screening.'}
    schema_checks = []
    manifests = []
    for fold in range(5):
        data = args.data / f'fold_{fold}'
        if not (data / 'manifest.json').exists():
            continue
        train = pl.scan_parquet(data / 'train.parquet')
        test = pl.scan_parquet(data / 'test.parquet')
        tu = train.select('user_id').unique().collect()['user_id']
        vu = test.select('user_id').unique().collect()['user_id']
        assert not set(tu) & set(vu)
        # Persist file identity and exactly the physical row/batch order consumed.
        schema_checks.append({'fold': fold, 'train_sha256': file_hash(data / 'train.parquet'), 'test_sha256': file_hash(data / 'test.parquet'), 'train_order_sha256': hashlib.sha256(train.select('request_id').collect()['request_id'].to_numpy().tobytes()).hexdigest(), 'schema_sha256': file_hash(data / 'schema/feature_meta.json'), 'schema_artifact_hashes': {str(p.relative_to(data / 'schema')): file_hash(p) for p in sorted((data / 'schema').rglob('*')) if p.is_file()}, 'train_test_users_disjoint': True, 'transformed_train_nulls': train.select(pl.all().null_count()).collect().sum_horizontal().item(), 'transformed_test_nulls': test.select(pl.all().null_count()).collect().sum_horizontal().item()})
        fold_manifests = []
        for mode in ['content', 'budget', 'fusion', 'ours']:
            p = args.runs / f'fold_{fold}' / mode / 'manifest.json'
            if p.exists():
                m = json.loads(p.read_text())
                fold_manifests.append(m)
        hashes = {m['initial_state_sha256'] for m in fold_manifests}
        assert len(hashes) <= 1
        manifests.append({'fold': fold, 'common_initialization_exact': len(hashes) == 1 and len(fold_manifests) == 4, 'initial_hash': list(hashes), 'model_manifests': fold_manifests})
    summary = {}
    for mode in ['content', 'budget', 'fusion', 'ours']:
        paths = sorted(args.runs.glob(f'fold_*/{mode}/predictions.parquet'))
        if not paths:
            continue
        p = pl.read_parquet(paths)
        assert p['request_id'].n_unique() == p.height
        fold_metrics = [json.loads(f.with_name('metrics.json').read_text()) for f in paths]
        m = metrics(p)
        m['complete_folds'] = len(paths)
        m['full_train_full_test'] = all(f['full_train'] and f['full_test'] for f in fold_metrics)
        m['all_requests_predicted_once'] = p.height == req.height
        m['per_fold'] = fold_metrics
        m['fold_auc_mean'] = float(np.mean([f['overall.auc'] for f in fold_metrics]))
        m['fold_auc_std'] = float(np.std([f['overall.auc'] for f in fold_metrics]))
        summary[mode] = m
    # A calibrated constant and two clock-only ranks for context. No test fitting.
    constant_parts = []
    for fold in range(5):
        part = req.filter(pl.col('fold') == fold).select('request_id', 'user_id', 'session_id', 'y_continue')
        prior = req.filter(pl.col('fold') != fold)['y_continue'].mean()
        constant_parts.append(part.with_columns(pl.lit(prior).alias('prediction')))
    constant = metrics(pl.concat(constant_parts))
    clock = {}
    for c in ['elapsed_time_s', 'req_idx']:
        for sign in [1, -1]:
            p = req.with_columns((sign * pl.col(c)).alias('p'))
            clock[f'{sign:+d}*{c}'] = {'overall.auc': roc_auc_score(p['y_continue'].to_numpy(), p['p'].to_numpy()), 'user_gauc': grouped_auc(p['y_continue'].to_numpy(), p['p'].to_numpy(), p['user_id'].to_numpy()), 'session_gauc': grouped_auc(p['y_continue'].to_numpy(), p['p'].to_numpy(), p['session_id'].to_numpy())}
    survival_audit = None
    ours = sorted(args.runs.glob('fold_*/ours/predictions.parquet'))
    if ours:
        p = pl.read_parquet(ours).join(req.select('request_id','req_idx'), on='request_id', validate='1:1').sort(['session_id','req_idx'])
        p = p.with_columns(pl.col('log_survival').diff().over('session_id').alias('log_s_increase'))
        # These are heuristic scores, never logging propensities / identified IPCW.
        p = p.with_columns((-pl.col('log_survival')).clip(0, np.log(10)).exp().alias('inverse_survival_capped10_heuristic'))
        w = p['inverse_survival_capped10_heuristic'].to_numpy().astype(np.float64)
        survival_audit = {'rows': p.height, 'product_max_abs_error': float((p['prediction'] - p['gamma'] * p['content_stay']).abs().max()), 'survival_underflow_zero_fraction': float((p['survival'] == 0).mean()), 'survival_increases_within_session_fraction': float((p['log_s_increase'].drop_nulls() > 1e-6).mean()), 'heuristic_kappa_cap': 10, 'heuristic_kappa_mean': float(w.mean()), 'heuristic_kappa_cap_fraction': float((w >= 10-1e-6).mean()), 'heuristic_kappa_effective_sample_size': float(w.sum()**2 / (w*w).sum()), 'status': 'NOT_IDENTIFIED_AS_IPCW_OR_LOGGING_PROPENSITY; parameters depend on changing history/slate, so S(e_t|X_t) is not one coherent session survival curve; no NCIS policy claim'}
        p.select('request_id','fold','gamma','content_stay','log_survival','inverse_survival_capped10_heuristic').write_parquet(args.runs / 'survival_diagnostics.parquet')
    sources = sorted(Path('workspace/kr_time_budget/pipeline').glob('*.py')) + [Path('workspace/kr_time_budget/pipeline/time_budget.py')]
    audit = {'checks': checks, 'timing_audit': timing_audit, 'fold_data': schema_checks, 'paired_initializations': manifests, 'summary': summary, 'train_prevalence_constant': constant, 'clock_rank_diagnostics': clock, 'survival_audit': survival_audit, 'source_files': {str(p): file_hash(p) for p in sources}, 'comparison_level': 'NON_CAUSAL_ARCHITECTURE_SCREENING; same initial tensors/data order/batches/8epochs; different masks, head/objective and two-stage LR', 'future_work': {'LTV': 'labels ready; deferred per latest Argon note limiting current stage to 6.1; no CAVE uAUC/bAUC definitions supplied', 'Generator_NCIS': 'not run: no logged slate action propensities, real request boundaries, generator/target-policy specification; survival scores are not behavior-policy probabilities'}}
    (args.runs / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps({'checks': checks, 'models': {k:{key: m[key] for key in ['overall.auc','user_gauc','session_gauc','logloss','complete_folds','full_train_full_test']} for k,m in summary.items()}, 'survival_audit': survival_audit}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
